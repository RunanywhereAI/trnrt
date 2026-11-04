"""The hyper-connection kernels (kernels/mhc_pre.py, mhc_mix.py, mhc_split.py, mhc_fnj.py).

On the host: each emulation is the model's own form bit for bit (models/hybrid.py _mhc, mix_out), the model takes a
kernel in decode calls only and hands it the layer's own weights and constants, the split kernel's instructions are
mhc_pre's, the FFN side's j-major front (mhc_fnj) is mhc_split's front but its named lines, and the on-chip pre
broadcast is the KDA shell's. Each source pin is shown to fail on a planted source.

In the simulator (the Neuron venv): every kernel against its emulation, the served forms against the forms they
replaced, bit for bit where the arithmetic is the same, and the parser frontend takes them for trn2.
"""

from __future__ import annotations

import ast
import collections
import importlib.util
import pathlib
import types

import numpy as np
import pytest
import torch

from trnrt.kernels import mhc_fnj as MF
from trnrt.kernels import mhc_mix as mm
from trnrt.kernels import mhc_pre as mp
from trnrt.kernels import mhc_split as MS
from trnrt.models import hybrid

NKI = importlib.util.find_spec("nki") is not None
needs_nki = pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
HC, EPS, ITERS, RMS_EPS = 4, 1e-6, 20, 1e-5
KDIR = pathlib.Path(mp.__file__).parent
PSRC = (KDIR / "mhc_pre.py").read_text()
SSRC = (KDIR / "mhc_split.py").read_text()
FSRC = (KDIR / "mhc_fnj.py").read_text()
HSRC = (KDIR / "shell.py").read_text()
P = MS.P


def _bf(t):
    import ml_dtypes

    return np.ascontiguousarray(t.float().numpy()).astype(ml_dtypes.bfloat16)


# -- mhc_mix: the output streams ----------------------------------------------------------------------------------


def mix_case(T: int, H: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    post = 2 * torch.sigmoid(torch.randn(T, HC, generator=g))
    comb = torch.softmax(torch.randn(T, HC, HC, generator=g) * 3, dim=-1)
    y = (torch.randn(T, H, generator=g) * 0.5).bfloat16()
    S = (torch.randn(T, HC, H, generator=g) * 0.5).bfloat16()
    return post, comb, y, S


def test_mix_emulation_is_mix_out():
    post, comb, y, S = mix_case(5, 256)
    assert torch.equal(mm.emulate(post, comb, y, S), hybrid.mix_out(post, comb, y, S))


def test_mix_out_takes_the_kernel_in_decode_calls_only(monkeypatch):
    post, comb, y, S = mix_case(3, 256)
    calls = []
    monkeypatch.setattr(mm, "takes", lambda device, H, T: True)
    monkeypatch.setattr(mm, "mix", lambda *a: calls.append(1) or mm.emulate(*a))
    want = hybrid.mix_out(post, comb, y, S)
    assert not calls
    assert torch.equal(hybrid.mix_out(post, comb, y, S, decode=True), want) and calls == [1]


def test_mix_takes_on_a_device_within_its_shapes():
    dev = torch.device("meta")
    assert mm.takes(dev, 4096, 8)
    assert not mm.takes(dev, 4096 + 64, 8) and not mm.takes(dev, 4096, mm.P + 1)
    assert not mm.takes(torch.device("cpu"), 4096, 8)


@needs_nki
@pytest.mark.parametrize("T,H", [(1, 256), (4, 256), (4, 4096), (8, 4096)])
def test_mix_simulator_equals_emulation(T, H, monkeypatch):
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    post, comb, y, S = mix_case(T, H, seed=T + H)
    got = np.asarray(
        nki.simulate(mm.mhc_mix_kernel)(
            post=post.numpy().astype(np.float32),
            comb=comb.reshape(T, HC * HC).numpy().astype(np.float32),
            y=_bf(y),
            S=_bf(S.reshape(T, HC * H)),
            hc=HC,
            rev=mm.REV,
        )
    )
    got = torch.from_numpy(got.astype(np.float32))
    want = mm.emulate(post, comb, y, S).float()
    assert torch.equal(got, want), (got - want).abs().max()


@needs_nki
@pytest.mark.parametrize("T", [1, 8])
def test_mix_parser_takes_the_kernel(T):
    import ml_dtypes
    from nki.compiler.frontend import FrontendKind, source_to_mlir

    H, z = 4096, np.zeros
    assert str(
        source_to_mlir(
            mm.mhc_mix_kernel,
            post=z((T, HC), np.float32),
            comb=z((T, HC * HC), np.float32),
            y=z((T, H), ml_dtypes.bfloat16),
            S=z((T, HC * H), ml_dtypes.bfloat16),
            hc=HC,
            rev=mm.REV,
            target="trn2",
            frontend=FrontendKind.PARSER,
        )
    )


# -- mhc_pre: the hyper-connection before a block ------------------------------------------------------------------


def stand_in(T: int = 5, H: int = 256, seed: int = 1):
    """A layer under GLM-5.3-Flash's released comb prior (0 on the diagonal, -8 off it), fn in the model dtype (bf16,
    as served: hybrid.init_layer), the streams bf16."""
    g = torch.Generator().manual_seed(seed)
    hy = types.SimpleNamespace(hc=HC, hc_eps=EPS, sinkhorn_iters=ITERS, family="glm5_next")
    model = types.SimpleNamespace(cfg=types.SimpleNamespace(hybrid=hy, rms_norm_eps=RMS_EPS, hidden_size=H))
    base = torch.randn(HC * (2 + HC), generator=g) * 0.1
    base[2 * HC :] = (torch.full((HC, HC), -8.0) + 8.0 * torch.eye(HC)).reshape(-1)
    fn = (torch.randn(HC * (2 + HC), HC * H, generator=g) * 3.0 / (HC * H) ** 0.5).bfloat16()
    scale = torch.tensor([1.0, 0.5, 2.0])
    layer = types.SimpleNamespace(hc_attn_fn=fn, hc_attn_base=base, hc_attn_scale=scale)
    S = (torch.randn(T, HC, H, generator=g) * 0.5).bfloat16()
    return model, layer, S


def pre_args(layer, S):
    return S, layer.hc_attn_fn, layer.hc_attn_base, layer.hc_attn_scale, ITERS, EPS, RMS_EPS


def test_pre_emulation_is_mhc():
    model, layer, S = stand_in()
    for got, want in zip(mp.emulate(*pre_args(layer, S)), hybrid._mhc(model, layer, "attn", S)):
        assert got.dtype == want.dtype and torch.equal(got, want)


def test_mhc_takes_the_kernel_in_decode_calls_only(monkeypatch):
    """With the kernel available, _mhc calls it only in a decode call, with the layer's fn, base and scale and the
    model's round count, eps and rms_eps (the emulation's result through the branch equals _mhc's own, and a wrong
    constant would move it: the second half)."""
    model, layer, S = stand_in()
    calls = []
    monkeypatch.setattr(mp, "takes", lambda device, T, H, hc, dt: True)
    monkeypatch.setattr(mp, "pre", lambda *a: calls.append(a) or mp.emulate(*a))
    want = hybrid._mhc(model, layer, "attn", S)
    assert not calls
    got = hybrid._mhc(model, layer, "attn", S, decode=True)
    assert len(calls) == 1 and all(torch.equal(g, w) for g, w in zip(got, want))
    a = list(pre_args(layer, S))
    for i, wrong in ((4, ITERS - 1), (5, RMS_EPS), (6, EPS)):  # the positive control: each constant, if wrong
        b = list(a)
        b[i] = wrong
        assert any(not torch.equal(g, w) for g, w in zip(mp.emulate(*b), want)), i


def test_pre_takes_on_a_device_within_its_shapes():
    dev = torch.device("meta")
    assert mp.takes(dev, 8, 4096, HC, torch.bfloat16)
    assert not mp.takes(torch.device("cpu"), 4, 4096, HC, torch.bfloat16)
    assert not mp.takes(dev, mp.MAX_T + 1, 4096, HC, torch.bfloat16)
    assert not mp.takes(dev, 8, 4096 + 64, HC, torch.bfloat16)
    assert not mp.takes(dev, 8, 4096, HC, torch.float32)


def _sim_pre(T: int, H: int, seed: int, sig: int = mp.SIGMOID, skf: int = mp.SK):
    import nki

    model, layer, S = stand_in(T, H, seed)
    x, post, comb = nki.simulate(mp.mhc_pre_kernel)(
        S=_bf(S.reshape(T, HC * H)),
        fn=_bf(layer.hc_attn_fn),
        base=layer.hc_attn_base.numpy().astype(np.float32),
        scale=layer.hc_attn_scale.numpy().astype(np.float32),
        hc=HC,
        iters=ITERS,
        eps=EPS,
        rms_eps=RMS_EPS,
        sig=sig,
        rev=mp.REV,
        skf=skf,
    )
    got = (
        torch.from_numpy(np.asarray(x).astype(np.float32)),
        torch.from_numpy(np.asarray(post)),
        torch.from_numpy(np.asarray(comb)).view(T, HC, HC),
    )
    return got, mp.emulate(*pre_args(layer, S))


@needs_nki
@pytest.mark.parametrize("T,H", [(1, 256), (4, 256), (4, 4096), (8, 4096)])
def test_pre_simulator_matches_emulation(T, H, monkeypatch):
    """x: the kernel differs from the host's in its last fp32 bits (summation order, the scalar engine's functions),
    so an x differs where the hc-term sum cancels (|x| far below the terms) or sits at a bf16 rounding boundary: within
    one bf16 ulp of x plus 1e-6 of the terms' magnitude. comb: well inside the gap one Sinkhorn round makes."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    model, layer, S = stand_in(T, H, T + H)
    (x, post, comb), (wx, wpost, wcomb) = _sim_pre(T, H, seed=T + H)
    ulp = (x.bfloat16().view(torch.int16).int() - wx.view(torch.int16).int()).abs()
    terms = S.float().abs().sum(1)
    bf_ulp = (wx.float().abs().log2().floor() - 7).exp2()
    assert ((x - wx.float()).abs() <= bf_ulp + 1e-6 * terms).all() and (ulp == 0).float().mean() > 0.99
    assert (post - wpost).abs().max() <= 1e-6
    err = (comb - wcomb).abs().max().item()
    a = list(pre_args(layer, S))
    gap = min((mp.emulate(*(a[:4] + [n] + a[5:]))[2] - wcomb).abs().max().item() for n in (ITERS - 1, ITERS + 1))
    assert err <= 1e-5 and err < gap / 100, (err, gap)


@needs_nki
@pytest.mark.parametrize("T", [1, 4])
def test_sinkhorn_eps_in_the_reduce(T, monkeypatch):
    """The served Sinkhorn (skf 1: the + eps inside the reduce, by an eps row and column, 3 instructions a half-round
    instead of 4) is the plain form (skf 0) bit for bit in the simulator, whose reduce adds in order. x and post are
    computed before the Sinkhorn and never move. The positive control: the two forms must trace differently."""
    import ml_dtypes
    from nki.compiler.frontend import FrontendKind, source_to_mlir

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    (x0, p0, c0), _ = _sim_pre(T, 256, seed=T + 7, skf=0)
    (x1, p1, c1), _ = _sim_pre(T, 256, seed=T + 7, skf=1)
    assert torch.equal(c1, c0) and torch.equal(x1, x0) and torch.equal(p1, p0)
    O, z = HC * (2 + HC), np.zeros

    def adds(skf, frontend):
        m = str(
            source_to_mlir(
                mp.mhc_pre_kernel,
                S=z((T, HC * 4096), ml_dtypes.bfloat16),
                fn=z((O, HC * 4096), ml_dtypes.bfloat16),
                base=z((O,), np.float32),
                scale=z((3,), np.float32),
                hc=HC,
                iters=ITERS,
                eps=EPS,
                rms_eps=RMS_EPS,
                sig=mp.SIGMOID,
                rev=mp.REV,
                skf=skf,
                target="trn2",
                frontend=frontend,
            )
        )
        return m.count("nisa.tensor_scalar_arith")

    for fe in (FrontendKind.PARSER, FrontendKind.TRACER):
        assert adds(0, fe) - adds(1, fe) == 2 * ITERS - 1, fe


@needs_nki
@pytest.mark.parametrize("T", [1, 8])
def test_pre_parser_takes_the_kernel(T):
    import ml_dtypes
    from nki.compiler.frontend import FrontendKind, source_to_mlir

    H, z, O = 4096, np.zeros, HC * (2 + HC)
    assert str(
        source_to_mlir(
            mp.mhc_pre_kernel,
            S=z((T, HC * H), ml_dtypes.bfloat16),
            fn=z((O, HC * H), ml_dtypes.bfloat16),
            base=z((O,), np.float32),
            scale=z((3,), np.float32),
            hc=HC,
            iters=ITERS,
            eps=EPS,
            rms_eps=RMS_EPS,
            sig=mp.SIGMOID,
            rev=mp.REV,
            skf=mp.SK,
            target="trn2",
            frontend=FrontendKind.PARSER,
        )
    )


# -- the pre broadcast on chip (pre_body, front_body) --------------------------------------------------------------

SHELL_BC = [  # shell.mhc_pre_h's broadcast, as it reads there
    "nisa.affine_select(dst=D, pattern=[[1, T], [0, hc]], channel_multiplier=-1, offset=0, on_true_tile=pre.ap(pattern="
    "[[hc, T], [0, T], [1, hc]], offset=0), on_false_value=0.0, cmp_op=nl.equal)",
    "nisa.nc_matmul(dst=pbp, stationary=ones_t, moving=D, accumulate=False)",
    "nisa.tensor_copy(dst=pb, src=pbp, engine=nisa.vector_engine)",
]


def _fn(src: str, name: str) -> ast.FunctionDef:
    for f in ast.walk(ast.parse(src)):
        if isinstance(f, ast.FunctionDef) and f.name == name:
            return f
    raise AssertionError(f"no {name}")


def _ifs(f: ast.FunctionDef, test: str) -> list[ast.If]:
    return [s for s in ast.walk(f) if isinstance(s, ast.If) and ast.unparse(s.test) == test]


def pbc_errors(src: str, fname: str, flag: str) -> list[str]:
    f = _fn(src, fname)
    on, off = _ifs(f, flag), _ifs(f, f"not {flag}")
    bad = []
    if (
        len(on) != 1
        or not on[0].orelse
        or [ast.unparse(s) for s in on[0].orelse] != ["nisa.dma_copy(dst=pscr, src=pre)"]
    ):
        return [f"{fname}: no single `if {flag}:` with the pscr write as its else"]
    calls = [ast.unparse(s.value) for s in on[0].body if isinstance(s, ast.Expr)]
    if [c for c in calls if not c.startswith("nisa.memset")] != SHELL_BC:
        bad.append(f"{fname}: the broadcast is not shell's: {calls}")
    offs = [[ast.unparse(s) for s in i.body] for i in off]
    if offs != [
        ["pscr = nl.ndarray((T, hc), dtype=f32, buffer=nl.private_hbm)"],
        [
            "pb = nl.ndarray((P, T, hc), dtype=f32, buffer=nl.sbuf)",
            "nisa.dma_copy(dst=pb, src=pscr.ap(pattern=[[0, P], [hc, T], [1, hc]], offset=0))",
        ],
    ]:
        bad.append(f"{fname}: the round trip's pscr statements moved: {offs}")
    return bad


def same_pbc_block(psrc: str, ssrc: str) -> bool:
    a = _ifs(_fn(psrc, "pre_body"), "PBC")
    b = _ifs(_fn(ssrc, "front_body"), "MP.PBC")
    return bool(a and b) and [ast.unparse(s) for s in a[0].body] == [ast.unparse(s) for s in b[0].body]


def test_pre_body_and_front_body_hold_shells_broadcast():
    assert pbc_errors(PSRC, "pre_body", "PBC") == []
    assert pbc_errors(SSRC, "front_body", "MP.PBC") == []
    assert same_pbc_block(PSRC, SSRC)
    assert all(c in ast.unparse(_fn(HSRC, "mhc_pre_h")) for c in SHELL_BC)


def test_the_broadcast_is_served_and_folded_into_both_revs():
    assert mp.PBC is True and mp.PBC_REV != 0
    assert "^ PBC_REV" in PSRC.split("\nREV = ", 1)[1].split("\n", 1)[0]
    assert "^ MP.PBC_REV" in SSRC.split("\nREV = ", 1)[1].split("\n", 1)[0]


def test_the_broadcast_pins_fail_on_planted_sources():
    assert pbc_errors(
        PSRC.replace("channel_multiplier=-1, offset=0,\n", "channel_multiplier=1, offset=0,\n", 1), "pre_body", "PBC"
    )
    assert pbc_errors(PSRC.replace("moving=D, accumulate=False)", "moving=D, accumulate=True)", 1), "pre_body", "PBC")
    assert not same_pbc_block(PSRC, SSRC.replace("on_false_value=0.0,", "on_false_value=1.0,", 1))


@needs_nki
@pytest.mark.parametrize("kernel", ["pre", "split"])
@pytest.mark.parametrize("T,nan", [(1, False), (4, False), (4, True), (8, True)])
def test_the_broadcast_is_bit_equal_to_the_round_trip_in_the_simulator(kernel, T, nan, monkeypatch):
    """x, post and comb with the on-chip broadcast against the HBM round trip it replaced, bit for bit (NaN bits
    included) at H 256. A NaN element stands for a stream a warmup's zero inputs make NaN."""
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

    def run():
        _, layer, S = stand_in(T, 256, T + 3)
        if nan:
            S = S.clone()
            S[min(1, T - 1), 1, 5] = float("nan")
        kw = dict(
            S=_bf(S.reshape(T, HC * 256)),
            fn=_bf(layer.hc_attn_fn),
            base=layer.hc_attn_base.numpy().astype(np.float32),
            scale=layer.hc_attn_scale.numpy().astype(np.float32),
            hc=HC,
            iters=ITERS,
            eps=EPS,
            rms_eps=RMS_EPS,
            sig=mp.SIGMOID,
            skf=mp.SK,
        )
        if kernel == "pre":
            out = nki.simulate(mp.mhc_pre_kernel)(rev=mp.REV, **kw)
        else:
            kw["fn"] = _bf(MF.fn_j(layer.hc_attn_fn))  # the split kernel's front is the j-major one (mhc_fnj)
            out = nki.simulate(MS.mhc_split_kernel)(rev=MS.REV, chunks=3, **kw)
        return [np.asarray(o).view(np.uint16 if np.asarray(o).dtype.itemsize == 2 else np.uint32).copy() for o in out]

    monkeypatch.setattr(mp, "PBC", False)
    off = run()
    monkeypatch.setattr(mp, "PBC", True)
    on = run()
    assert len(on) == len(off) == 3
    for a, b in zip(off, on):
        assert a.shape == b.shape and (a == b).all()


# -- mhc_split: the front and the Sinkhorn pieces ------------------------------------------------------------------


def _nisa_calls(src: str, names: set[str]) -> dict[str, list[str]]:
    """The nisa.* calls of each named function, in source order, as normalised source text."""
    out = {}
    for f in ast.walk(ast.parse(src)):
        if isinstance(f, ast.FunctionDef) and f.name in names:
            calls = [
                n
                for n in ast.walk(f)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "nisa"
            ]
            calls.sort(key=lambda n: (n.lineno, n.col_offset))
            out[f.name] = [ast.unparse(n) for n in calls]
    return out


def test_split_instructions_are_pre_bodys():
    """front_body + sinkhorn_body hold exactly pre_body's nisa calls (the same text, the same counts), sinkhorn_body's
    in pre_body's order; a call missing from front_body breaks the count equality (the pin is not vacuous)."""
    pre = _nisa_calls(PSRC, {"pre_body"})["pre_body"]
    split = _nisa_calls(SSRC, {"front_body", "sinkhorn_body"})
    assert collections.Counter(split["front_body"] + split["sinkhorn_body"]) == collections.Counter(pre)
    it = iter(pre)
    assert all(any(c == p for p in it) for c in split["sinkhorn_body"])
    assert collections.Counter(split["front_body"][1:] + split["sinkhorn_body"]) != collections.Counter(pre)


@pytest.mark.parametrize("chunks", [1, 2, 3, 5, 13, 39])
def test_split_chunks_partition_the_half_rounds(chunks):
    b = MS.chunk_bounds(20, chunks)
    assert b[0][0] == 0 and b[-1][1] == 39 and len(b) <= chunks
    assert all(hi > lo for lo, hi in b) and all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))


@pytest.mark.parametrize("chunks", [0, 40])
def test_split_chunks_out_of_range_refused(chunks):
    with pytest.raises(ValueError):
        MS.chunk_bounds(20, chunks)


def test_split_emulation_is_mhc_pres():
    g = torch.Generator().manual_seed(0)
    T, H = 3, 256
    S = torch.randn(T, HC, H, generator=g).bfloat16()
    fn = (0.05 * torch.randn(2 * HC + HC * HC, HC * H, generator=g)).bfloat16()
    base, scale = 0.1 * torch.randn(2 * HC + HC * HC, generator=g), 1 + 0.1 * torch.randn(3, generator=g)
    a = MS.emulate(S, fn, base, scale, 20, 1e-6, 1e-6)
    b = mp.emulate(S, fn, base, scale, 20, 1e-6, 1e-6)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert isinstance(MS.REV, int) and MS.REV != mp.REV


# -- mhc_fnj: the FFN side's j-major front -------------------------------------------------------------------------

# front_body's statements that change (None: dropped), in front_body's order; everything else is the same text
NAMED = [
    ("O = fn.shape[0]", "O = fn.shape[2]"),
    (
        "wk = nl.ndarray((P, O, NJ), dtype=bf16, buffer=nl.sbuf)",
        "wkj = nl.ndarray((P, NJ, O), dtype=bf16, buffer=nl.sbuf)",
    ),
    (
        "nisa.dma_copy(dst=wk, src=fn.ap(pattern=[[NJ, P], [F, O], [1, NJ]], offset=0))",
        "nisa.dma_copy(dst=wkj, src=fn)",
    ),
    ("wkj = nl.ndarray((P, NJ, O), dtype=bf16, buffer=nl.sbuf)", None),
    (
        "nisa.tensor_copy(dst=wkj, src=wk.ap(pattern=[[O * NJ, P], [1, NJ], [NJ, O]], offset=0), engine=nisa.vector_engine)",
        None,
    ),
]


def _stmts(f: ast.FunctionDef) -> list:
    doc = isinstance(f.body[0], ast.Expr) and isinstance(f.body[0].value, ast.Constant)
    return [ast.unparse(s) for s in (f.body[1:] if doc else f.body)]


def _fnj_pin_errors(split: str, fnj: str) -> list:
    want, bad = _stmts(_fn(split, "front_body")), []
    for old, new in reversed(NAMED):  # the later lines first: the replacement adds a wkj allocation
        i = [k for k, s in enumerate(want) if s == old]
        if len(i) != 1:
            bad.append(f"front_body: {old!r} {len(i)} times")
            continue
        want[i[0] : i[0] + 1] = [] if new is None else [new]
    got = _stmts(_fn(fnj, "front_body_fnj"))
    if got != want:
        bad.append(f"front_body_fnj differs from front_body beyond the named lines ({len(got)} vs {len(want)})")
    return bad


def _ap(pattern, offset, env, counts):
    pat, base = eval(pattern, {}, env), eval(offset, {}, env)
    assert [c for _, c in pat] == counts
    idx = torch.tensor(base)
    for k, (stride, n) in enumerate(pat):
        shape = [1] * len(pat)
        shape[k] = n
        idx = idx + torch.arange(n).view(shape) * stride
    return idx


def _fnj_layout_errors(split: str) -> list:
    """fn_j(fn) against front_body's two steps: wk = fn's k-layout read, wkj = wk's j-major read."""
    calls = {
        ast.unparse(c.func) + ":" + ast.unparse(next(k.value for k in c.keywords if k.arg == "dst")): c
        for c in ast.walk(_fn(split, "front_body"))
        if isinstance(c, ast.Call)
        and ast.unparse(c.func) in ("nisa.dma_copy", "nisa.tensor_copy")
        and any(k.arg == "dst" for k in c.keywords)
    }
    src_ap = lambda c: next(k.value for k in c.keywords if k.arg == "src")  # noqa: E731
    ld, cp = src_ap(calls["nisa.dma_copy:wk"]), src_ap(calls["nisa.tensor_copy:wkj"])
    kw = lambda a: {k.arg: ast.unparse(k.value) for k in a.keywords}  # noqa: E731
    O, NJ = 24, 128
    env = dict(P=P, O=O, NJ=NJ, F=P * NJ)
    fn = torch.arange(O * P * NJ, dtype=torch.int64).view(O, P * NJ)
    wk = fn.reshape(-1)[_ap(kw(ld)["pattern"], kw(ld)["offset"], env, [P, O, NJ]).reshape(-1)].view(P, O, NJ)
    wkj = wk.reshape(-1)[_ap(kw(cp)["pattern"], kw(cp)["offset"], env, [P, NJ, O]).reshape(-1)].view(P, NJ, O)
    return [] if torch.equal(MF.fn_j(fn), wkj) else ["fn_j is not front_body's j-major operand"]


def test_front_body_fnj_is_front_body_but_the_named_lines():
    assert _fnj_pin_errors(SSRC, FSRC) == []
    assert len(_stmts(_fn(FSRC, "front_body_fnj"))) >= 60  # the whole body, not an empty match


def test_fn_j_is_the_operand_front_body_builds():
    assert _fnj_layout_errors(SSRC) == []
    t = MF.fn_j(torch.randn(24, 4 * 4096).bfloat16())
    assert tuple(t.shape) == (P, 128, 24) and t.is_contiguous()


def test_the_ffn_side_takes_the_j_major_fn_and_folds_its_rev():
    layer = types.SimpleNamespace(hc_ffn_fn=1, hc_ffn_fnj=2)
    assert MS.FNJ is True and MS.fn_arg(layer) == 2
    assert MS.REV == MS.REV_K ^ MF.REV  # the k-layout front's REV (the attention side's) and the FFN side's apart
    a = SSRC.index("try:  # the Neuron venv")
    b = SSRC.index("    mhc_split_kernel = None", a)
    assert "FNJ" not in SSRC[a:b]  # the rebinding lives outside the hashed span


@pytest.mark.parametrize("plant", ["fnj-drift", "split-drift", "layout"])
def test_a_planted_change_breaks_a_fnj_pin(plant):
    if plant == "fnj-drift":  # the copy's body drifts beyond the named lines
        old = "        nisa.reciprocal(dst=s, data=s)\n"
        assert FSRC.count(old) == 1
        assert _fnj_pin_errors(SSRC, FSRC.replace(old, ""))
    elif plant == "split-drift":  # the k-layout body changes; the copy does not follow
        old = (
            "        nisa.tensor_scalar(dst=cx, data=x, op0=nl.multiply, operand0=65537.0, engine=nisa.vector_engine)\n"
        )
        assert SSRC.count(old) == 1
        assert _fnj_pin_errors(SSRC.replace(old, old.replace("65537.0", "65536.0")), FSRC)
    else:  # front_body's re-layout read a different way: fn_j no longer its operand
        old = "src=wk.ap(pattern=[[O * NJ, P], [1, NJ], [NJ, O]], offset=0)"
        assert SSRC.count(old) == 1
        assert _fnj_layout_errors(SSRC.replace(old, "src=wk.ap(pattern=[[O * NJ, P], [O, NJ], [1, O]], offset=0)"))


def test_fnj_copy_is_decode_only():
    """hc_ffn_fnj is listed in layer._decode_only where init_layer makes it: a plain layer parameter would shift every
    prefill graph's lifted tensor indices and re-key prefill graphs that never read it."""
    src = (KDIR.parent / "models" / "hybrid.py").read_text()
    body = [ast.unparse(s) for s in _fn(src, "init_layer").body]
    i = body.index("layer.hc_ffn_fnj = p(128, hc * H // 128, n)")
    assert body[i + 1] == "layer._decode_only = tuple(getattr(layer, '_decode_only', ())) + ('hc_ffn_fnj',)"


SERVED_T = (1, 2, 4, 6, 8)
H_GLM = 4096
O_GLM = 2 * HC + HC * HC
NJ_GLM = HC * H_GLM // P


def _fnj_inputs(T: int):
    """The released comb prior (base 0 / -8) at GLM-5.3-Flash's shapes (hc 4, H 4096: O 24, NJ 128)."""
    import ml_dtypes

    rng = np.random.default_rng(4100 + T)
    S = rng.standard_normal((T, HC * H_GLM)).astype(np.float32)
    fn = (0.05 * rng.standard_normal((O_GLM, HC * H_GLM))).astype(np.float32)
    base = (0.1 * rng.standard_normal(O_GLM)).astype(np.float32)
    comb = np.full((HC, HC), -8.0, np.float32)
    np.fill_diagonal(comb, 0.0)
    base[2 * HC :] = comb.reshape(-1)
    scale = (1.0 + 0.1 * rng.standard_normal(3)).astype(np.float32)
    bf = ml_dtypes.bfloat16
    return S.astype(bf), fn.astype(bf), base, scale


if NKI:
    import nki as _nki

    @_nki.jit
    def _k_front_k(S, fn, base, scale, hc: int, iters: int, eps: float, rms_eps: float, sig: int, skf: int):
        x_out, post_out, comb_out, cp, _ = MS.front_body_k(S, fn, base, scale, hc, iters, eps, rms_eps, sig, skf)
        MS.sinkhorn_body(cp, comb_out, hc, iters, eps, skf, 0, 2 * iters - 1)
        return x_out, post_out, comb_out

    @_nki.jit
    def _k_front_fnj(S, fn, base, scale, hc: int, iters: int, eps: float, rms_eps: float, sig: int, skf: int):
        x_out, post_out, comb_out, cp, _ = MF.front_body_fnj(S, fn, base, scale, hc, iters, eps, rms_eps, sig, skf)
        MS.sinkhorn_body(cp, comb_out, hc, iters, eps, skf, 0, 2 * iters - 1)
        return x_out, post_out, comb_out


def _sim_front(monkeypatch, kernel, S, fn, base, scale):
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    minor = int(str(getattr(nki, "__version__", "0.6")).split(".")[1])
    if minor >= 7:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
        k = kernel
    else:
        k = kernel[2]
    outs = nki.simulate(k)(
        S=S, fn=fn, base=base, scale=scale, hc=HC, iters=ITERS, eps=1e-6, rms_eps=1e-6, sig=mp.SIGMOID, skf=mp.SK
    )
    return [np.asarray(o) for o in outs]


def _same(a, b) -> list[bool]:
    return [x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes() for x, y in zip(a, b)]


@needs_nki
@pytest.mark.parametrize("T", SERVED_T)
def test_fnj_bit_equal_to_the_k_layout_front_at_served_T(monkeypatch, T):
    """At every served decode T at GLM-5.3-Flash's shapes, front_body_fnj on fn_j(fn) gives the k-layout front's x /
    post / comb on fn bit for bit, each followed by the whole Sinkhorn."""
    S, fn, base, scale = _fnj_inputs(T)
    ref = _sim_front(monkeypatch, _k_front_k, S, fn, base, scale)
    got = _sim_front(monkeypatch, _k_front_fnj, S, _bf(MF.fn_j(torch.from_numpy(fn.astype(np.float32)))), base, scale)
    assert len(ref) == len(got) == 3
    assert _same(ref, got) == [True, True, True], f"T {T}: x / post / comb bit-equal {_same(ref, got)}"
    assert np.isfinite(np.asarray(ref[1], np.float32)).all() and float(np.abs(np.asarray(ref[0], np.float32)).max()) > 0


WRONG = {  # [P, NJ, O] tensors holding fn in a wrong order
    "o_major": lambda f: f.reshape(O_GLM, P, NJ_GLM).permute(1, 0, 2).contiguous().reshape(P, NJ_GLM, O_GLM),
    "pj_swapped": lambda f: f.reshape(O_GLM, NJ_GLM, P).permute(2, 1, 0),  # k = j 128 + p
}


@needs_nki
@pytest.mark.parametrize("name", sorted(WRONG))
def test_fnj_wrong_layout_fails(monkeypatch, name):
    """The mutation: the bitwise check above is not vacuous at these shapes."""
    S, fn, base, scale = _fnj_inputs(4)
    ref = _sim_front(monkeypatch, _k_front_k, S, fn, base, scale)
    got = _sim_front(
        monkeypatch, _k_front_fnj, S, _bf(WRONG[name](torch.from_numpy(fn.astype(np.float32)))), base, scale
    )
    assert not _same(ref, got)[1], f"{name}: post passed the bitwise check"
