"""The decode FFN half: the attention output's mix, mHC's FFN front, the norm and the MoE (or dense) FFN in one NKI call
(kernels/ffn_half.py, ffn_half_pb.py, ffn_half_shared.py, dense_half.py, moe_half.py).

On the host:
- half_body_pb's Sinkhorn hooks sit at their points (H0 between the router's logits matmuls and sigmoid_exp, H1 between
  the shared gate / up matmuls and gub's copy, H2 between the a^T matmul and at's copy, and L(it) between routed group
  it's down matmuls and the y product, only for it <= n), and the schedule (pieces / step B1 4 / 10, B2 5 / 8, B4 7 / 6,
  B6 8 / 5, B8 10 / 4 at K 8, Q 4) covers the 39 half-rounds once, in order, on hooks both programs reach;
- ffn_half_shared's body (the fused DSA layer's FFN half) is the served kernel's, AST for AST, and its helpers are the
  served module's objects; dense_half's front is the served kernel's and its SwiGLU lines the shared expert's;
- the packed layouts (moe_half's, dense_half's) against the loaded weights, unit by unit;
- the dense half's wiring: the predicate, the dispatch, and its host branch as the served dense FFN block bit for bit;
- run() launches ffn_half_pb's kernel with its REV.
In the simulator (the Neuron venv): the whole FFN half against its emulation (the mix, mHC, the norm, and moe_half's
emulation of the MoE on the kernel's own normed input), with positive controls. Each source pin fails on a planted
source."""

import ast
import importlib.util
import pathlib
import sys
import types
import zlib

import pytest
import torch

from trnrt.kernels import dense_half as DH
from trnrt.kernels import ffn_half as FH
from trnrt.kernels import ffn_half_pb as FB
from trnrt.kernels import ffn_half_shared as FS
from trnrt.kernels import moe_dedupe as mdd
from trnrt.kernels import moe_half as mh
from trnrt.kernels import mhc_split as MS
from trnrt.kernels import rowiso as RI
from trnrt.kernels import shell
from trnrt.kernels import shell as SH
from trnrt.models import hybrid as HY
from trnrt.models.decoder import rms_norm

NKI = importlib.util.find_spec("nki") is not None
needs_nki = pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
KDIR = pathlib.Path(FH.__file__).parent
PB = (KDIR / "ffn_half_pb.py").read_text()
SHARED = (KDIR / "ffn_half_shared.py").read_text()
DENSE = (KDIR / "dense_half.py").read_text()
P = 128


def _fn(src: str, name: str) -> ast.FunctionDef:
    for f in ast.walk(ast.parse(src)):
        if isinstance(f, ast.FunctionDef) and f.name == name:
            return f
    raise AssertionError(f"no {name}")


def _body(f: ast.FunctionDef) -> list:
    doc = isinstance(f.body[0], ast.Expr) and isinstance(f.body[0].value, ast.Constant)
    return f.body[1:] if doc else f.body


# -- ffn_half_pb: the Sinkhorn's hooks and their schedule ----------------------------------------------------------

def _is_piece(s) -> bool:
    return (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Name)
            and s.value.func.id == "_piece")

def _linear(stmts):
    """Every statement in source order, compound ones before their children."""
    for s in stmts:
        yield s
        for field in ("body", "orelse"):
            if hasattr(s, field) and isinstance(getattr(s, field), list):
                yield from _linear(getattr(s, field))

def _calls(s) -> list[str]:
    """The nisa.* / shell.* calls a simple statement makes (not a compound one's children)."""
    if isinstance(s, (ast.For, ast.If, ast.While)):
        return []
    return [ast.unparse(n) for n in ast.walk(s) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id in ("nisa", "shell")]

def _hook_sites(pb: str) -> list[tuple[str, str, str]]:
    """(hook index expression, the last instruction before it, the first after it) per _piece call, in source order."""
    seq = list(_linear(_body(_fn(pb, "half_body_pb"))))
    out = []
    for i, s in enumerate(seq):
        if _is_piece(s):
            before = next(c for t in reversed(seq[:i]) for c in reversed(_calls(t)))
            after = next(c for t in seq[i + 1:] for c in _calls(t))
            out.append((ast.unparse(s.value.args[6]), before, after))
    return out

WANT = [("0", "nisa.nc_matmul(dst=pl", "shell.sigmoid_exp(dst=sc, src=pl"),
        ("1", "nisa.nc_matmul(dst=gps", "nisa.tensor_copy(dst=gub, src=gps"),
        ("2", "nisa.nc_matmul(dst=aps", "nisa.tensor_copy(dst=at, src=aps"),
        ("2 + it", "nisa.nc_matmul(dst=pd[:, s, c, :]", "nisa.tensor_tensor(dst=y.flatten_dims(2, 3)")]

def _sites_ok(pb: str) -> bool:
    got = _hook_sites(pb)
    return len(got) == len(WANT) and all(p == wp and b.startswith(wb) and a.startswith(wa)
                                         for (p, b, a), (wp, wb, wa) in zip(got, WANT))

def _covers(n: int, step: int, n_half: int = 39) -> bool:
    """Hooks 0 .. 2 + n's pieces [p step, min((p + 1) step, n_half)) cover 0 .. n_half - 1 once, in order."""
    got = [r for p in range(3 + n) for r in range(p * step, min((p + 1) * step, n_half))]
    return got == list(range(n_half))

def test_hooks_sit_at_the_preregistered_points():
    assert _sites_ok(PB), _hook_sites(PB)

def test_the_loop_hook_is_guarded_by_n_inside_stage_c():
    f = _fn(PB, "half_body_pb")
    stage = next(s for s in _linear(_body(f)) if isinstance(s, ast.If) and ast.unparse(s.test) == "1 <= it <= NL")
    guard = [s for s in stage.body if isinstance(s, ast.If) and ast.unparse(s.test) == "it <= n_hk"]
    assert len(guard) == 1 and all(_is_piece(b) for b in guard[0].body)

@pytest.mark.parametrize("T,pieces,step", [(1, 4, 10), (2, 5, 8), (4, 7, 6), (6, 8, 5), (8, 10, 4)])
def test_schedule_is_the_preregistered_table(T, pieces, step):
    K, Q = 8, 4  # num_experts_per_tok, moe_dedupe.group_size(1)
    NQ = -(-T * K // Q)
    n, st = FB.pb_hooks(NQ, 39)
    assert st == step and sum(1 for p in range(3 + n) if p * st < 39) == pieces
    assert _covers(n, st)
    assert n <= NQ - (NQ + 1) // 2  # program 1's group count, the smaller: both programs reach every Lk

@pytest.mark.parametrize("plant", ["hook-moved", "hook-dropped", "floor-step"])
def test_a_planted_change_breaks_a_hook_pin(plant):
    if plant == "hook-moved":  # H1 after gub's copy instead of before it
        h1 = "        _piece(cp, comb_out, hc, iters, eps, skf, 1, step)  # H1\n"
        cp_ = "        nisa.tensor_copy(dst=gub, src=gps, engine=nisa.vector_engine)\n"
        pb = PB.replace(h1, "").replace(cp_, cp_ + h1)
        assert pb != PB and not _sites_ok(pb)
    elif plant == "hook-dropped":  # H2 gone: the rounds would sit elsewhere
        pb = PB.replace("        _piece(cp, comb_out, hc, iters, eps, skf, 2, step)  # H2\n", "")
        assert pb != PB and not _sites_ok(pb)
    else:  # step = floor(39 / hooks): the last half-rounds never issued
        assert not all(_covers(n, 39 // (3 + n)) for n in (1, 2, 3, 4, 8))


def test_rev_is_one_crc_of_the_source_and_its_helpers():
    own = SH.kernel_rev(FB.__file__, "# -- the kernel source", "    ffn_half_kernel = None")
    assert FB.REV == zlib.crc32(repr((own, SH.REV, MS.REV, mh.REV, RI.REV, FB.P)).encode()) ^ SH.MIX_REV
    assert FH.REV == FB.REV
    # an edit of shell moves SH.REV and mh.REV (which folds it) by the same bits: the one CRC moves, an XOR would not
    d = 0x5A5A
    assert zlib.crc32(repr((own, SH.REV ^ d, MS.REV, mh.REV ^ d, RI.REV, FB.P)).encode()) ^ SH.MIX_REV != FB.REV
    assert zlib.crc32(repr((own, SH.REV, MS.REV, mh.REV, RI.REV ^ 1, FB.P)).encode()) ^ SH.MIX_REV != FB.REV


def test_run_launches_the_pb_kernel(monkeypatch):
    """run() with a stand-in wrap_nki: ffn_half_pb's kernel, with its REV and the served row isolation."""
    T, hc, H, E, npg = 1, 4, 256, 8, 2
    seen = []

    class Wrapped:  # wrap_nki(k)[grid](**args)
        def __init__(self, k):
            self.k = k

        def __getitem__(self, grid):
            def call(**args):
                seen.append((self.k, args["rev"], args["fin"], grid))
                return (torch.zeros(T, hc * H, dtype=torch.bfloat16), torch.zeros(T, hc), torch.zeros(T, hc * hc),
                        torch.zeros(npg, 128, T, H // 128), torch.zeros(npg, T, H))
            return call

    hop = types.ModuleType("libtorch_neuronx_lite.nki.nki_hop")
    hop.wrap_nki = Wrapped
    for name in ("libtorch_neuronx_lite", "libtorch_neuronx_lite.nki"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "libtorch_neuronx_lite.nki.nki_hop", hop)
    kpb = object()
    monkeypatch.setattr(FB, "ffn_half_kernel", kpb)
    S = torch.zeros(T, hc, H, dtype=torch.bfloat16)
    pend = (torch.zeros(T, hc), torch.zeros(T, hc, hc), torch.zeros(npg, T, H, dtype=torch.bfloat16))
    blobs = {torch.bfloat16: torch.zeros(1), torch.float32: torch.zeros(1), "fn": torch.zeros(1),
             "base": torch.zeros(1), "scale": torch.zeros(1), "post_norm": torch.zeros(1)}
    FH.run(S, pend, blobs, torch.zeros(E, 1), Is=128, K=8, scale=2.5, group=4, ring=3, limit=10.0, rsp=0, hc=hc,
           iters=20, hc_eps=1e-6, rms_eps=1e-5, grid=npg)
    assert seen == [(kpb, FB.REV, RI.FIN, npg)], seen


# -- ffn_half_shared: the served kernel's body for the fused DSA layer ----------------------------------------------

def _sbody(f: ast.FunctionDef) -> list:
    return [ast.unparse(s) for s in _body(f)]


def _args(f: ast.FunctionDef) -> list:
    return [(a.arg, ast.unparse(a.annotation) if a.annotation else None) for a in f.args.args]


def _pinned(served: str, shared: str) -> bool:
    k, b = _fn(served, "ffn_half_kernel"), _fn(shared, "ffn_half_body")
    return _sbody(k) == _sbody(b) and [a for a in _args(k) if a[0] != "rev"] == _args(b)


def _imports(src: str) -> dict:
    """{alias: module} of the module-level `from . import x as y` lines."""
    out = {}
    for s in ast.parse(src).body:
        if isinstance(s, ast.ImportFrom) and s.level == 1 and s.module is None:
            out.update({a.asname or a.name: a.name for a in s.names})
    return out


def test_body_and_arguments_are_the_served_kernels():
    assert _pinned(PB, SHARED)
    assert len(_sbody(_fn(SHARED, "ffn_half_body"))) >= 15  # the whole body (18 statements), not an empty match


def test_helpers_are_the_served_objects_not_copies():
    tree = ast.parse(SHARED)
    defs = {f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)}
    assert defs == {"ffn_half_body"}  # nothing else defined here: every callee is imported
    assert "_norm_xc, half_body_pb = FB._norm_xc, FB.half_body_pb" in SHARED
    served, shared = _imports(PB), _imports(SHARED)
    for alias in ("SH", "MS"):  # the names the body calls through
        assert shared[alias] == served[alias]
    assert _imports(SHARED)["FB"] == "ffn_half_pb"
    assert FS.P == 128 and "P = 128" in PB


def test_rev_covers_the_body_and_folds_the_served_rev():
    body = SH.kernel_rev(FS.__file__, "# -- the body source", "    ffn_half_body = None")
    assert FS.REV == zlib.crc32(repr((body, FB.REV)).encode())
    assert FS.REV != zlib.crc32(repr((body, FB.REV ^ 1)).encode())  # a served change (source or knob) re-keys
    a = SHARED.index("# -- the body source")
    assert SHARED.index("def ffn_half_body", a) < SHARED.index("    ffn_half_body = None", a)  # the body is covered


@pytest.mark.parametrize("plant", ["served-drift", "shared-drift", "statement-dropped", "argument-renamed",
                                   "annotation-changed", "rev-kept"])
def test_a_planted_change_breaks_the_pin(plant):
    if plant == "served-drift":  # the served kernel changes; the copy does not follow
        old = "s_pv = nl.ndarray((T, F), dtype=bf16, buffer=nl.private_hbm)"
        assert PB.count(old) == 1
        assert not _pinned(PB.replace(old, old.replace("private_hbm", "shared_hbm")), SHARED)
    elif plant == "shared-drift":
        old = "_, post_out, comb_out, cp, xb = MS.front_body(s_pv, fnb, base, scale, hc, iters, eps, rms_eps, sig, skf)"
        assert SHARED.count(old) == 1
        assert not _pinned(PB, SHARED.replace(old, old.replace("sig, skf", "skf, sig")))
    elif plant == "statement-dropped":
        old = "        VE = nisa.vector_engine\n"
        assert SHARED.count(old) == 1
        assert not _pinned(PB, SHARED.replace(old, ""))
    elif plant == "argument-renamed":
        assert not _pinned(PB, SHARED.replace("o_rb: int, dbg: int", "o_rb: int, dbg2: int", 1))
    elif plant == "annotation-changed":
        assert not _pinned(PB, SHARED.replace("rms_eps: float, sig: int", "rms_eps: float, sig: float", 1))
    else:  # the copy taking `rev` too would not be the kernel's arguments less rev
        assert not _pinned(PB, SHARED.replace("dbg: int, fin: int):", "dbg: int, fin: int, rev: int):", 1))


# -- dense_half: the dense layers' FFN half ------------------------------------------------------------------------

FRONT_FIRST = "sh = nl.ndarray((P, hc, T, W), dtype=bf16, buffer=nl.sbuf)"
FRONT_LAST = "xn, xc = _norm_xc(xb, pnorm, rms_eps, fin, T, H)"


def _front(src: str, name: str) -> list:
    """The kernel's statements from the mix's SBUF tile through _norm_xc, unparsed."""
    st = [ast.unparse(s) for s in _fn(src, name).body]
    a = st.index(FRONT_FIRST)
    return st[a:st.index(FRONT_LAST, a) + 1]


def test_front_is_the_served_kernels():
    served, dense = _front(PB, "ffn_half_kernel"), _front(DENSE, "dense_half_kernel")
    assert served == dense
    assert len(dense) >= 6  # the mix, s_pv, its copy loop, front_body, the norm (not an empty match)


@pytest.mark.parametrize("side", ["served", "dense"])
def test_front_pin_catches_a_plant(side):
    old = "SH.mix_in_h(sh, S, post_p, comb_p, Yp, S_out, hc, T, H, npg)"
    new = "SH.mix_in_h(sh, S, post_p, comb_p, Yp, S_out, hc, T, H, 1)"
    served, dense = PB, DENSE
    if side == "served":
        assert served.count(old) >= 1
        served = served.replace(old, new)
    else:
        assert dense.count(old) == 1
        dense = dense.replace(old, new)
    assert _front(served, "ffn_half_kernel") != _front(dense, "dense_half_kernel")


def test_helpers_are_the_served_objects():
    if DH.dense_body is None:
        pytest.skip("no nki in this venv: the kernel's helpers are not defined")
    assert DH._norm_xc is FB._norm_xc and DH._piece is FB._piece
    assert DH.SH is FB.SH and DH.MS is FB.MS


def test_body_swiglu_lines_are_half_body_pbs():
    """The SwiGLU's elementwise statements, word for word the shared expert's (the tiled matmuls are the named
    divergence)."""
    shared = [ast.unparse(s) for s in _fn(PB, "half_body_pb").body]
    dense = [ast.unparse(s) for s in _fn(DENSE, "dense_body").body]
    lines = ("gub = nl.ndarray((T, J), dtype=bf16, buffer=nl.sbuf)",
             "gcl = nl.ndarray((T, ih), dtype=bf16, buffer=nl.sbuf)",
             "slf = nl.ndarray((T, ih), dtype=f32, buffer=nl.sbuf)",
             "nisa.activation(dst=slf, op=nl.silu, data=gcl)",
             "ab = nl.ndarray((T, ih), dtype=bf16, buffer=nl.sbuf)")
    for ln in lines:
        assert ln in shared and ln in dense, ln
    sh = [s for s in shared if s.startswith(("nisa.tensor_scalar(dst=gcl", "nisa.tensor_scalar(dst=ucl",
                                              "nisa.tensor_tensor(dst=ab", "nisa.tensor_copy(dst=slb"))]
    de = [s.replace("engine=VE", "engine=nisa.vector_engine") for s in dense
          if s.startswith(("nisa.tensor_scalar(dst=gcl", "nisa.tensor_scalar(dst=ucl", "nisa.tensor_tensor(dst=ab",
                           "nisa.tensor_copy(dst=slb"))]
    assert len(sh) == 4 and sh == de


def _layer_weights(Ir=384, H=256, seed=3):
    from trnrt.models.quant import quantize_fp8_rows

    g = torch.Generator().manual_seed(seed)
    gu8, gus = quantize_fp8_rows(torch.randn(2 * Ir, H, generator=g) * H ** -0.5)
    dn8, dns = quantize_fp8_rows(torch.randn(H, Ir, generator=g) * Ir ** -0.5)
    return gu8, gus, dn8, dns


@pytest.mark.parametrize("npg", [1, 2])
def test_pack_layout(npg):
    from trnrt.models.quant import dequant

    Ir, H, P = 384, 256, 128
    gu8, gus, dn8, dns = _layer_weights(Ir, H)
    blob = DH.pack(gu8, gus, dn8, dns, torch.bfloat16, npg)
    off = DH.offsets(H, Ir, npg)
    ih, C = Ir // npg, H // P
    w = dequant(gu8, gus, torch.bfloat16).to(torch.bfloat16)
    d = dequant(dn8, dns, torch.bfloat16).to(torch.bfloat16)
    gu = blob[off["o_gu"]:off["o_gu"] + npg * P * C * 2 * ih].view(npg, P, C, 2 * ih)
    dn = blob[off["o_dn"]:off["o_dn"] + npg * ih * H].view(npg, ih, H)
    for p in range(npg):
        for (i, c, j) in ((0, 0, 0), (5, 1, ih - 1), (P - 1, C - 1, ih), (17, 0, 2 * ih - 1)):
            row = p * ih + j if j < ih else Ir + p * ih + (j - ih)
            assert torch.equal(gu[p, i, c, j], w[row, c * P + i]), (p, i, c, j)
        assert torch.equal(dn[p], d[:, p * ih:(p + 1) * ih].t())
    # a planted wrong layout (k read as h = i C + c instead of c 128 + i) is caught by the same reads
    assert any(not torch.equal(gu[0, i, c, 0], w[0, i * C + c]) for (i, c) in ((5, 1), (17, 0), (100, 1)))


def test_emulate_is_hybrids_swiglu():
    """emulate's y against hybrid._swiglu on model._w's dequant, with the same x_n: bit for bit (both the served host
    arithmetic; a divergence here means emulate no longer replays today's form)."""
    from types import SimpleNamespace

    from trnrt.models import hybrid
    from trnrt.models.decoder import rms_norm
    from trnrt.models.quant import dequant

    Ir, H, hc, T = 384, 256, 4, 4
    gu8, gus, dn8, dns = _layer_weights(Ir, H)
    g = torch.Generator().manual_seed(7)
    O = 2 * hc + hc * hc
    S = torch.randn(T, hc, H, generator=g).bfloat16()
    pend = (torch.rand(T, hc, generator=g) * 2, torch.full((T, hc, hc), 1 / hc), (0.5 * torch.randn(2, T, H)).bfloat16())
    fn = (0.05 * torch.randn(O, hc * H, generator=g)).bfloat16()
    base, scale = 0.1 * torch.randn(O, generator=g), 1 + 0.1 * torch.randn(3, generator=g)
    pn = (1 + 0.1 * torch.randn(H, generator=g)).bfloat16()
    Sn, post, comb, xn, y = DH.emulate(S, pend, fn, base, scale, pn, gu8, gus, dn8, dns, 20, 1e-6, 1e-5, 10.0)
    layer = SimpleNamespace(gate_up=gu8, gate_up_scale=gus, down=dn8, down_scale=dns)
    model = SimpleNamespace(_w=lambda lyr, n: dequant(getattr(lyr, n), getattr(lyr, n + "_scale"), torch.bfloat16))
    want = hybrid._swiglu(model, layer, xn, "gate_up", "down", 10.0)
    assert torch.equal(y, want)
    assert torch.isfinite(y.float()).all() and y.abs().max() > 0
    # and x_n is decoder.rms_norm's of the front's x (the served FFN block's input)
    from trnrt.kernels import mhc_mix as MM
    from trnrt.kernels import mhc_split as MS

    Sn2 = MM.emulate(*pend[:2], pend[2].float().sum(0).bfloat16(), S).view(T, hc, H)
    x2, _, _ = MS.emulate(Sn2, fn, base, scale, 20, 1e-6, 1e-5)
    assert torch.equal(xn, rms_norm(x2, pn, 1e-5))


def test_rev_covers_the_served_front():
    """REV folds ffn_half_pb.REV (the front's objects and every served knob they read) and this kernel's source."""
    assert isinstance(DH.REV, int) and DH.REV != FB.REV
    src = SH.kernel_rev(DH.__file__, "# -- the kernel source", "    dense_body = dense_half_kernel = None")
    assert src != 0


# -- moe_half: the MoE half's packed layout ------------------------------------------------------------------------

def test_packed_blobs_at_the_served_shapes():
    """E 288, H 4096, Is 64 per rank, two programs: one bf16 blob (the router's [k, e] copy in two blocks of 144
    experts, each program's 32 + 32
    gate / up rows and 32 down columns) and one fp32 blob (the bias): 3.93 MB per layer per rank."""
    lay = shell.packed_layout(mh.shapes(288, 4096, 64, 2))
    assert set(lay) == {torch.bfloat16, torch.float32}
    nb, idx = lay[torch.bfloat16]
    assert idx["rt"] == (0, (2, P, 32, 144)) and idx["gu"][1] == (2, P, 32, 64) and idx["dn"][1] == (2, 32, 4096)
    assert nb == 128 * 32 * 288 + 2 * 128 * 32 * 64 + 2 * 32 * 4096  # each a multiple of ALIGN already
    assert lay[torch.float32][0] == 512 and lay[torch.float32][1]["rb"] == (0, (1, 288))
    assert abs(nb * 2 + 512 * 4 - 3.93e6) < 0.01e6


def fp8_layer(E=8, H=256, Is=8, seed=0):
    """A stand-in MoE layer's router, bias and FP8 shared expert with per-row 128-column block scales, as loaded."""
    from torch import nn

    g = torch.Generator().manual_seed(seed)
    layer = nn.Module()
    layer.router = nn.Parameter((torch.randn(E, H, generator=g) * 0.05).bfloat16(), requires_grad=False)
    layer.router_bias = nn.Parameter(torch.randn(E, generator=g) * 0.05, requires_grad=False)
    layer.shared_gate_up = nn.Parameter((torch.randn(2 * Is, H, generator=g) * 16).to(torch.float8_e4m3fn),
                                        requires_grad=False)
    layer.shared_gate_up_scale = nn.Parameter(torch.rand(2 * Is, H // P, generator=g) * 0.01 + 1e-3,
                                              requires_grad=False)
    layer.shared_down = nn.Parameter((torch.randn(H, Is, generator=g) * 16).to(torch.float8_e4m3fn),
                                     requires_grad=False)
    layer.shared_down_scale = nn.Parameter(torch.rand(H, 1, generator=g) * 0.01 + 1e-3, requires_grad=False)
    return layer


def test_fill_lays_out_the_loaded_weights(monkeypatch):
    """Every unit of the packed copies against quant.dequant of the loaded weights (model._w's read): the router as
    [i, c, e] = W_r[e, c 128 + i]; program p's gate_up [i, c, j] = Wgu[row_p(j), c 128 + i] with its gate rows, then
    its up rows; its down [j, h] = Wd[h, p Is / 2 + j]; the bias. A copy without the [i, c] permute fails (control).
    The router's copy in two blocks of E / 2 experts: block p, [i, c, e] = W_r[p E / 2 + e, c 128 + i]."""
    from trnrt.models.quant import dequant

    monkeypatch.setattr(shell, "programs", lambda: 2)
    E, H, Is = 8, 256, 8
    layer = fp8_layer(E, H, Is)
    shell.register_packed(layer, mh.PREFIX, mh.shapes(E, H, Is, 2))
    mh.fill(layer, torch.bfloat16)
    v = shell.packed_views(layer, mh.PREFIX, mh.shapes(E, H, Is, 2))
    w = dequant(layer.shared_gate_up.data, layer.shared_gate_up_scale.data, torch.bfloat16)
    d = dequant(layer.shared_down.data, layer.shared_down_scale.data, torch.bfloat16)
    ih = Is // 2
    for p, i, c, e in [(0, 0, 0, 0), (0, 5, 1, 3), (1, 127, 1, 3), (1, 64, 0, 2)]:
        assert v["rt"][p, i, c, e] == layer.router[p * (E // 2) + e, c * P + i]
    for p in range(2):
        rows = list(range(p * ih, (p + 1) * ih)) + list(range(Is + p * ih, Is + (p + 1) * ih))
        for i, c, j in [(0, 0, 0), (9, 1, 2), (127, 1, 2 * ih - 1), (64, 0, ih)]:
            assert v["gu"][p, i, c, j] == w[rows[j], c * P + i]
        for j, h in [(0, 0), (ih - 1, H - 1), (2, 77)]:
            assert v["dn"][p, j, h] == d[h, p * ih + j]
    assert torch.equal(v["rb"][0], layer.router_bias.float())
    flat = torch.cat([w[0:ih], w[Is:Is + ih]]).reshape(P, H // P, 2 * ih)  # the same bytes, no [i, c] permute
    assert not torch.equal(v["gu"][0], flat)


# -- the dense half's wiring (models/hybrid.py ffn_half_takes, kernels/ffn_half.py decode, dense_half.takes / decode) --

HC, HD, TD = 4, 64, 3


class _Lin:
    mla = None


def _pred(monkeypatch, moe=False, kda=True, lim=10.0, rows=3, dec=True, sp=False, dev="cpu", blob=True, host=True):
    monkeypatch.setattr(HY, "LinearSpec", _Lin)
    monkeypatch.setattr(HY, "SHELL_HOST", host)
    ns = types.SimpleNamespace
    model = ns(cfg=ns(hybrid=ns(family="glm5_next", swiglu_limit=lim)))
    layer = ns(moe=moe, spec=_Lin() if kda else ns(mla=object()))
    if blob:
        layer.dh_blob_bf16 = torch.zeros(1)
        if moe:
            layer.mh_blob_bf16 = torch.zeros(1)
    return HY.ffn_half_takes(model, layer, torch.zeros(rows, 8, device=dev), dec, sp)


def test_the_predicate_takes_a_dense_kda_decode_layer_on_its_own_conditions(monkeypatch):
    """A dense KDA layer's decode call: replicated streams, rows under MOE_HALF_BELOW and not in SKIP_T, the blob on
    the device, SHELL_HOST on the host; an MoE layer by its own blob; nothing else."""
    assert _pred(monkeypatch) is True
    assert _pred(monkeypatch, host=False) is False
    assert _pred(monkeypatch, dev="meta") is True and _pred(monkeypatch, dev="meta", blob=False) is False
    assert _pred(monkeypatch, moe=True, dev="meta") is True and _pred(monkeypatch, moe=True, dev="meta", blob=False) is False
    assert _pred(monkeypatch, kda=False) is False and _pred(monkeypatch, lim=None) is False
    assert _pred(monkeypatch, rows=HY.MOE_HALF_BELOW) is False and _pred(monkeypatch, rows=HY.MOE_HALF_BELOW - 1) is True
    assert _pred(monkeypatch, dec=False) is False and _pred(monkeypatch, sp=True) is False


def test_skip_t_keeps_the_xla_dense_ffn_at_two_rows(monkeypatch):
    """At 2 rows the XLA dense FFN is faster than the dense half: those calls keep it; 1, 3, 4, 6 and 8 rows take the
    dense half; an MoE layer is not affected."""
    assert DH.SKIP_T == frozenset((2,))
    assert [_pred(monkeypatch, rows=r, dev="meta") for r in (1, 2, 3, 4, 6, 8)] == [True, False, True, True, True, True]
    assert _pred(monkeypatch, rows=2, moe=True, dev="meta") is True


def test_decode_dispatches_by_layer_kind(monkeypatch):
    monkeypatch.setattr(FH, "_decode_moe", lambda *a: "moe")
    monkeypatch.setattr(DH, "decode", lambda *a: "dense")
    L = types.SimpleNamespace
    assert FH.decode(None, L(moe=False), None, None) == "dense" and FH.decode(None, L(moe=True), None, None) == "moe"


def _takes_layer(**kw):
    a = dict(moe=False, gate_up_scale=torch.ones(1), down_scale=torch.ones(1))
    a.update(kw)
    return types.SimpleNamespace(**a)


@pytest.mark.parametrize("layer,fam,lim,want", [
    ({}, "glm5_next", 10.0, True), ({"moe": True}, "glm5_next", 10.0, False),
    ({"gate_up_scale": None}, "glm5_next", 10.0, False), ({"down_scale": None}, "glm5_next", 10.0, False),
    ({}, "other", 10.0, False), ({}, "glm5_next", None, False)])
def test_dense_half_takes(layer, fam, lim, want):
    cfg = types.SimpleNamespace(hybrid=types.SimpleNamespace(family=fam, swiglu_limit=lim))
    assert DH.takes(_takes_layer(**layer), cfg) is want
    assert DH.takes(_takes_layer(), types.SimpleNamespace(hybrid=None)) is False


def _dense_model_layer(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    hy = types.SimpleNamespace(hc=HC, hc_eps=1e-6, sinkhorn_iters=20, family="glm5_next", swiglu_limit=10.0)
    model = types.SimpleNamespace(cfg=types.SimpleNamespace(hybrid=hy, rms_norm_eps=1e-5, hidden_size=HD))
    layer = types.SimpleNamespace(moe=False, post_norm=(1 + 0.1 * torch.randn(HD, generator=g)).bfloat16())
    for b in ("attn", "ffn"):
        setattr(layer, f"hc_{b}_fn", torch.randn(HC * (2 + HC), HC * HD, generator=g) * 0.05)
        setattr(layer, f"hc_{b}_base", torch.randn(HC * (2 + HC), generator=g) * 0.1)
        setattr(layer, f"hc_{b}_scale", torch.ones(3))
    S = (torch.randn(TD, HC, HD, generator=g) * 0.5).bfloat16()
    post_a = torch.rand(TD, HC, generator=g) * 2
    comb_a = torch.rand(TD, HC, HC, generator=g)
    Y_a = (torch.randn(2, TD, HD, generator=g) * 0.3).bfloat16()  # the KDA shell call's two programs' slots
    return model, layer, S, (post_a, comb_a, Y_a)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_dense_host_branch_is_the_served_dense_ffn_block(monkeypatch, seed):
    """dense_half.decode's host branch (SHELL_HOST) is the served dense FFN block bit for bit: the KDA shell's slots
    summed and mixed (mix_out), _mhc, rms_norm, the FFN, and the block's mix."""
    model, layer, S, (post_a, comb_a, Y_a) = _dense_model_layer(seed)

    def ffn(model, layer, sp):
        return lambda x: (x.float() * 1.5 - 0.25).to(x.dtype)
    monkeypatch.setattr(HY, "_ffn", ffn)
    monkeypatch.setattr(HY, "SHELL_HOST", True)
    st = HY.mix_out(post_a, comb_a, Y_a.float().sum(0).to(S.dtype), S, True)
    want = HY._block(model, layer, "ffn", st, layer.post_norm, ffn(model, layer, False), True)
    Sn, post, comb, y = DH.decode(model, layer, S, (post_a, comb_a, Y_a))
    assert torch.equal(Sn.reshape(TD, HC * HD), st)
    assert torch.equal(HY.mix_out(post, comb, y, Sn, True), want)  # layer()'s FFN-half tail


# -- the whole FFN half in the simulator, against its emulation ------------------------------------------------------

E, H, IS, K = 24, 1024, 16, 8


class _Model:
    """The model surface the FFN half's host forms read: decoder's router and expert reads, the model's dequant."""

    from trnrt.models.decoder import DecoderForCausalLM as _D

    _route, _experts, _w = _D._route, _D._experts, _D._w
    MOE_GATHER_MAX_PAIRS = _D.MOE_GATHER_MAX_PAIRS

    def __init__(self, scale: float = 2.5, limit: float = 10.0):
        hy = types.SimpleNamespace(hc=HC, hc_eps=1e-6, sinkhorn_iters=20, family="glm5_next", swiglu_limit=limit)
        self.cfg = types.SimpleNamespace(hybrid=hy, rms_norm_eps=1e-5, hidden_size=H, num_experts=E,
                                         num_experts_per_tok=K, norm_topk_prob=True, routed_scaling_factor=scale)
        self.dtype, self.moe_inter = torch.bfloat16, 64

    def _out_reduce(self, y):
        return y


def _ffn_case(T: int, npg: int, seed: int = 5):
    """A stand-in MoE layer (FP8 shared expert with block scales, the routed experts in the tile blob, the FFN block's
    hyper-connection) and the call's inputs: the streams and the attention output's pending mix."""
    from tests.test_moe_dedupe import experts128
    from trnrt.kernels import mhc_fnj as MF

    g = torch.Generator().manual_seed(seed)
    layer = fp8_layer(E, H, IS, seed)
    layer.router.data = (torch.randn(E, H, generator=g) * H ** -0.5).bfloat16()
    layer.w_blob = mdd.pack(*experts128(seed=seed, e=E, h=H))
    layer.moe, layer.moe_blob, layer.moe_tiles, layer.down_t = True, True, True, True
    O = 2 * HC + HC * HC
    layer.hc_ffn_fn = (torch.randn(O, HC * H, generator=g) * 0.02).bfloat16()
    layer.hc_ffn_base = torch.randn(O, generator=g) * 0.5
    layer.hc_ffn_scale = torch.rand(3, generator=g) + 0.5
    layer.post_norm = (1 + 0.1 * torch.randn(H, generator=g)).bfloat16()
    blobs = FH.pack(layer.router, layer.router_bias, layer.shared_gate_up, layer.shared_gate_up_scale,
                    layer.shared_down, layer.shared_down_scale, layer.hc_ffn_fn, layer.post_norm, layer.hc_ffn_base,
                    layer.hc_ffn_scale, npg=npg)
    blobs["fn"] = MF.fn_j(layer.hc_ffn_fn)  # the kernel's front reads fn j-major (mhc_fnj)
    S = (torch.randn(T, HC, H, generator=g) * 0.5).bfloat16()
    pend = (torch.rand(T, HC, generator=g) * 2, torch.softmax(torch.randn(T, HC, HC, generator=g), -1),
            (torch.randn(npg, T, H, generator=g) * 0.3).bfloat16())
    return layer, blobs, S, pend


def _ffn_sim(layer, blobs, S, pend, npg: int, monkeypatch):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setattr(shell, "programs", lambda: npg)
    return FH.run(S, pend, blobs, layer.w_blob, Is=IS, K=K, scale=2.5, group=mdd.group_size(1), ring=FH.RING,
                  limit=10.0, rsp=0, hc=HC, iters=20, hc_eps=1e-6, rms_eps=1e-5, simulate=True, grid=npg, dbg=1)


def _ulps(a, b):
    return (a.contiguous().view(torch.int16).int() - b.contiguous().view(torch.int16).int()).abs()


@needs_nki
@pytest.mark.parametrize("T,npg", [(1, 2), (4, 2), (8, 2), (4, 1)])
def test_the_ffn_half_in_the_simulator_is_its_emulation(T, npg, monkeypatch):
    """The kernel against its host form, step by step:
    - the mixed streams bit for bit mhc_mix's emulation, post and comb within fp32 summation order of _mhc's;
    - the normed input within two bf16 ulps of rms_norm's, most elements equal (the served norm drops one bf16 rounding
      of x r and the Newton step of the scalar engine's rsqrt);
    - on the kernel's own normed input (x pinned), the kernel's experts and weights are decoder._route's, and its
      output (routed partials permuted and added, shared partials added, in bf16) is moe_half's emulation (decoder's
      router, the pairs kernel's arithmetic, the clamped shared expert) within the bf16 output's rounding.
    The positive controls: the emulation with a wrong routed scale or clamp misses that bound."""
    layer, blobs, S, pend = _ffn_case(T, npg)
    model = _Model()
    Sn, post, comb, rp, sp, xn, ids, wts = _ffn_sim(layer, blobs, S, pend, npg, monkeypatch)
    with torch.no_grad():
        hSn, hpost, hcomb, _ = FH._host(model, layer, S, pend)
    assert torch.equal(Sn, hSn), (Sn.float() - hSn.float()).abs().max()
    assert torch.allclose(post, hpost, rtol=1e-4, atol=1e-6) and torch.allclose(comb, hcomb, rtol=1e-4, atol=1e-6)
    x, _, _ = HY._mhc(model, layer, "ffn", hSn, True)
    u = _ulps(xn, rms_norm(x, layer.post_norm, 1e-5))
    assert int(u.max()) <= 2 and (u == 0).float().mean() > 0.9, (int(u.max()), (u == 0).float().mean())
    with torch.no_grad():
        topv, topi = model._route(layer, xn)
        want = mh.emulate(model, layer, xn)
        got = (SH.add_partials(rp, torch.bfloat16) + SH.add_partials(sp, torch.bfloat16)).float()
    assert torch.equal(ids.long().sort(-1).values, topi.sort(-1).values)
    order = lambda i, v: torch.gather(v.float(), 1, i.long().argsort(-1))  # noqa: E731
    assert torch.allclose(order(ids, wts), order(topi, topv), rtol=1e-2, atol=1e-3)
    bound = 0.01 * want.float().abs().max()
    assert (got - want.float()).abs().max() <= bound, (got - want.float()).abs().max()
    for wrong in (_Model(scale=2.0), _Model(limit=5.0)):
        with torch.no_grad():
            off = mh.emulate(wrong, layer, xn).float()
        assert (got - off).abs().max() > bound
