"""The context-parallel DSA layers' slot attention and long-context selection kernels.

- kernels/dsa_slots.py, dsa_slots_n.py: the slot attention (dsa_decode's arithmetic plus the log-sum-exp) over a run-time
  row count; dsa_slots_n_s.py: its rows split over the two physical cores at LNC 2 (the prefill's call and the decode
  fallback of the DSA shell); dsa_slots_n_s2.py: the same split kernel reading its rows and bias in their natural layout
  (dsa_slots_c's decode fallback); dsa_slots_c.py: each decode row over its compacted live slots.
- kernels/dsa_long_select.py, dsa_long_pipe.py, dsa_long_pipe_s.py: the prefill chunk's exact local selection (the
  kernel's algorithm step for step against models/dsa_long.select_reference), its pipelined form and that form's query
  tiles split over the two physical cores.

On the host: the emulations, the dispatch at the served LNC split, the rows per iteration, the bf16-bias REV routing, the
natural-layout kernel's pin against the served sources (comments stripped; planted drifts break it). In the NKI
simulator (Neuron venv): every split kernel against its whole kernel bit for bit, the natural-layout kernel against the
split one, the bf16 bias against the fp32 one (with a positive control on the matmul operands), dsa_slots_c against its
emulation.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import re
import zlib

import pytest
import torch

from trnrt.kernels import dsa_long_pipe as dlp
from trnrt.kernels import dsa_long_select as dl
from trnrt.kernels import dsa_slots as ds
from trnrt.kernels import dsa_slots_c
from trnrt.kernels import dsa_slots_n as dsn
from trnrt.kernels import dsa_slots_n_s as dns
from trnrt.kernels import dsa_slots_n_s2 as dn2
from trnrt.models import dsa_long

NEG = dsa_long.NEG_INF
KDIR = pathlib.Path(dsn.__file__).parent
needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")


def kernel_args(q_lat, kc2, rows, bias, rpi):
    """(the slot kernels' tensor arguments, the padded row count), laid out as dsa_slots_n.attend lays them out (the
    served, permuted layout)."""
    N, H, R = q_lat.shape
    NS = rows.shape[1]
    NCH = NS // 128
    Np = -(-N // rpi) * rpi
    q = q_lat.to(torch.bfloat16)
    rw, bs = rows, bias.float()
    if Np != N:
        q = torch.cat([q, q.new_zeros(Np - N, H, R)])
        rw = torch.cat([rw, rw.new_zeros(Np - N, NS)])
        bs = torch.cat([bs, bs.new_full((Np - N, NS, ds.KP), ds.NEG_INF)])
    kcp = kc2.reshape(kc2.shape[0] // ds.KP, ds.KP * R)
    NI = Np // rpi
    rows_t = rw.to(torch.int32).reshape(NI, rpi, NCH, 128).permute(0, 1, 3, 2).contiguous()
    bias_t = bs.reshape(NI, rpi, NCH, 128, ds.KP).permute(0, 1, 2, 4, 3).reshape(NI, rpi, NCH * ds.KP * 128).contiguous()
    eye = torch.eye(128).to(torch.bfloat16)
    return dict(q_lat=q.reshape(NI, rpi, H, R).contiguous(), kc=kcp, rows_t=rows_t, bias=bias_t, identb=eye), Np


def _inputs(N, H, R, NS, fp8, seed):
    g = torch.Generator().manual_seed(seed)
    rows_cache = 4 * 256
    q = torch.randn(N, H, R, generator=g)
    kc = torch.randn(rows_cache, R, generator=g) * 0.5
    kc = kc.to(torch.float8_e4m3fn) if fp8 else kc.bfloat16()
    rows = torch.randint(0, rows_cache // ds.KP, (N, NS), generator=g)
    bias = torch.where(torch.rand(N, NS, ds.KP, generator=g) < 0.2, ds.NEG_INF, 0.0)
    return q, kc, rows, bias


def _code(text: str) -> str:
    """A span's code: docstrings, comments and blank lines dropped, trailing whitespace stripped."""
    out = []
    for ln in re.sub(r'"""[\s\S]*?"""', "", text).splitlines():
        s = re.sub(r"\s+#.*$", "", ln).rstrip()
        if s.strip() and not s.lstrip().startswith("#"):
            out.append(s)
    return "\n".join(out)


# -- the slot attention: emulation and dispatch ------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_slots_emulation(dtype):
    """dsa_slots.emulate: dsa_decode.emulate's output, and the log-sum-exp of the scaled, biased scores per (row, head);
    a row whose every slot is NEG_INF stays finite (uniform weights, lse ~NEG_INF)."""
    from trnrt.kernels import dsa_decode

    g = torch.Generator().manual_seed(6)
    N, H, R, NS = 5, 4, 64, 6
    kc = torch.randn(64 * 4, 1, R, generator=g).to(dtype)
    q = torch.randn(N, H, R, generator=g) * 0.1
    rows = torch.randint(0, 64, (N, NS), generator=g)
    bias = torch.zeros(N, NS, 4)
    bias[:, 4:] = NEG
    bias[1, 3, 2:] = NEG
    bias[3] = NEG
    o, lse = ds.emulate(q, kc, rows, bias, 0.125, lse=True)
    assert torch.equal(o, dsa_decode.emulate(q, kc.reshape(-1, R), rows, bias, 0.125))
    assert torch.isfinite(o).all() and torch.isfinite(lse).all()
    tok = (rows.unsqueeze(-1) * 4 + torch.arange(4)).reshape(N, -1)
    K = kc.reshape(-1, R)[tok].float()
    if dtype != torch.float32:
        q = q.to(torch.bfloat16).float()
    s = torch.einsum("bhr,btr->bht", q, K) * 0.125 + bias.reshape(N, 1, -1)
    torch.testing.assert_close(lse, torch.logsumexp(s, -1), rtol=1e-6, atol=1e-5)
    assert (lse[3] < -1e29).all()
    tol = 1e-5 if dtype == torch.float32 else 5e-3  # (a bf16 cache rounds p to bf16, as the kernel does)
    torch.testing.assert_close(o[3], K[3].mean(0).expand(H, R), rtol=tol, atol=tol)  # uniform over the row's tokens


def test_dispatch_at_the_served_split(monkeypatch):
    """At grid 2 (LNC 2, where the split names dsa_slots_n): dsa_slots_c's decode fallback ("c") takes the
    natural-layout kernel; the DSA shell's fallback ("shell") and the context-parallel prefill's call (False) take the
    split kernel; at grid 1 every call takes dsa_slots_n's own kernel."""
    from trnrt import platform

    monkeypatch.setattr(platform, "nki_grid", lambda: 2)
    assert dn2.NAT and dn2.SITES == ("c",) and ds.QKW
    split = (dns.dsa_slots_n_s_kernel, dns.REV)
    assert dsn._nat("c") and (dsn._kernel("c"), dsn._rev("c")) == (dn2.dsa_slots_n_s_kernel, dn2.REV)
    for site in ("shell", False):
        assert not dsn._nat(site) and (dsn._kernel(site), dsn._rev(site)) == split
    assert dsn._rev(False, True) == dns.REV_BB
    monkeypatch.setattr(platform, "nki_grid", lambda: 1)  # no split: this module's kernel and REV
    assert not dsn._nat("c") and (dsn._kernel("c"), dsn._rev("c")) == (dsn.dsa_slots_n_kernel, dsn.REV)
    assert dsn._rev(False, True) == dsn.REV


def test_rows_per_iteration():
    """Decode calls (at most PREFILL_ROWS rows) iterate over min(RPI, N) rows; a prefill chunk's slot classes (more rows)
    over RPI_PREFILL."""
    assert (ds.RPI, ds.RPI_PREFILL, ds.PREFILL_ROWS, ds.FIT) == (4, 16, 8, True)
    assert {N: dsn._rpi(N) for N in (1, 2, 4, 6, 8)} == {1: 1, 2: 2, 4: 4, 6: 4, 8: 4}
    assert all(dsn._rpi(N) == 16 for N in (9, 128, 1024, 4096))


def test_bias_bf16_lines_count_only_in_rev_bb():
    """The bf16-bias branch's lines (BB_TAG) count only in REV_BB, so REV, the rev of every fp32-bias call (the decode
    fallbacks), is the source without them."""
    src = open(dns.__file__).read()
    assert sum(ln.endswith(dns.BB_TAG) for ln in src.splitlines()) == 4
    assert dns.REV == dns._kernel_rev(False) != dns.REV_BB == dns._kernel_rev(True)


# -- dsa_slots_n_s2: the natural-layout kernel, pinned against the served sources ----------------------------------

SLOTS = (KDIR / "dsa_slots.py").read_text()
SERVED = (KDIR / "dsa_slots_n_s.py").read_text()
MINE = (KDIR / "dsa_slots_n_s2.py").read_text()

# the three reads _row_nat rewrites: (served statement, natural-layout statement)
READS = [
    ('src=X["rows_t"].ap(pattern=[[NCH, 128], [1, NCH]], offset=k * 128 * NCH,',
     'src=X["rows_t"].ap(pattern=[[1, 128], [128, NCH]], offset=k * 128 * NCH,'),
    ('nisa.dma_copy(dst=br, src=X["bias"].ap(pattern=[[T, 1], [1, T]], offset=k * T, scalar_offset=it,\n'
     '                                                   indirect_dim=0))',
     'nisa.dma_copy(dst=br.ap(pattern=[[T, 1], [KP * 128, NCH], [128, KP], [1, 128]], offset=0),\n'
     '                          src=X["bias"].ap(pattern=[[T, 1], [128 * KP, NCH], [1, KP], [KP, 128]], offset=k * T,\n'
     '                                           scalar_offset=it, indirect_dim=0))'),
    ('nisa.dma_copy(dst=S, src=X["bias"].ap(pattern=[[0, H], [1, T]], offset=k * T, scalar_offset=it,\n'
     '                                                  indirect_dim=0))',
     'nisa.dma_copy(dst=S.ap(pattern=[[T, H], [KP * 128, NCH], [128, KP], [1, 128]], offset=0),\n'
     '                          src=X["bias"].ap(pattern=[[0, H], [128 * KP, NCH], [1, KP], [KP, 128]], offset=k * T,\n'
     '                                           scalar_offset=it, indirect_dim=0))'),
]


def _region(src: str) -> str:
    a = src.index("if dsn.dsa_slots_n_kernel is not None:")
    return src[a:src.index("    dsa_slots_n_s_kernel = None", a)]


def _row_nat(slots: str) -> str:
    """dsa_slots._row renamed _row_nat with its three layout reads rewritten (code only), or "" if a read is missing."""
    a = slots.index("    def _row(X, it, k):")
    ends = [x for x in (slots.find("\n    def ", a + 10), slots.find("\n    @nki.jit", a + 10)) if x > 0]
    r = _code(slots[a:min(ends) + 1]).replace("    def _row(X, it, k):", "    def _row_nat(X, it, k):", 1)
    for old, new in READS:
        if r.count(old) != 1:
            return ""
        r = r.replace(old, new)
    return r


def _expected(slots: str, served: str) -> str:
    """The served split kernel's code without its bf16-bias branch (the BB_TAG lines: a prefill call's form; the decode
    fallback's bias is fp32), with _row_nat in _row's place."""
    k = "".join(ln for ln in _region(served).splitlines(True) if not ln.rstrip("\n").endswith(dns.BB_TAG))
    k = _code(k)
    k = k.replace("from .dsa_slots import KTR, _pair, _psum, _row, _sb",
                  "from .dsa_slots import KTR, _pair, _psum, _sb", 1)
    k = k.replace("    KP = ds.KP\n", "    KP = ds.KP\n    VE = nisa.vector_engine\n" + _row_nat(slots) + "\n", 1)
    k = k.replace("NCH = rows_t.shape[3]", "NCH = rows_t.shape[2]", 1)
    return k.replace("_row(X, it, k)", "_row_nat(X, it, k)", 1)


def _pinned(slots: str, served: str, mine: str) -> bool:
    return bool(_row_nat(slots)) and _code(_region(mine)) == _expected(slots, served)


def test_kernel_is_the_served_kernel_with_the_natural_reads():
    """dsa_slots_n_s2's kernel source is dsa_slots_n_s's (less its bf16-bias branch) with dsa_slots._row copied as
    _row_nat, whose three layout reads (rows, the qkw bias, the broadcast bias) are rewritten and nothing else, and NCH
    read from the natural rows' shape."""
    assert _pinned(SLOTS, SERVED, MINE)


@pytest.mark.parametrize("plant", ["rows-left-served", "bias-strides-swapped", "row-drift", "kernel-drift"])
def test_a_planted_change_breaks_the_pin(plant):
    slots, served, mine = SLOTS, SERVED, MINE
    if plant == "rows-left-served":
        mine = MINE.replace("pattern=[[1, 128], [128, NCH]], offset=k * 128 * NCH", "pattern=[[NCH, 128], [1, NCH]], "
                            "offset=k * 128 * NCH", 1)
    elif plant == "bias-strides-swapped":  # kp and p read in each other's order
        mine = MINE.replace("[[T, 1], [128 * KP, NCH], [1, KP], [KP, 128]]", "[[T, 1], [128 * KP, NCH], [KP, KP], "
                            "[1, 128]]", 1)
    elif plant == "row-drift":  # dsa_slots._row moves, the copy does not
        slots = SLOTS.replace("x = k % 2", "x = (k + 1) % 2", 1)
    else:
        served = SERVED.replace("split = npg == 2 and rpi >= 2", "split = npg == 2 and rpi >= 3", 1)
    assert (slots, served, mine) != (SLOTS, SERVED, MINE)
    assert not _pinned(slots, served, mine)


def test_s2_rev_is_its_own():
    assert dn2.REV == zlib.crc32(_region(MINE).encode()) ^ dns.REV
    assert dn2.REV not in (dns.REV, dsn.REV)


def test_attend_lays_out_naturally_only_for_the_natural_kernel():
    """dsa_slots_n.attend's layouts under _nat are the two plain reshapes; the served permutes otherwise."""
    f = next(n for n in ast.walk(ast.parse((KDIR / "dsa_slots_n.py").read_text()))
             if isinstance(n, ast.FunctionDef) and n.name == "attend")
    br = [n for n in ast.walk(f) if isinstance(n, ast.If) and ast.unparse(n.test) == "_nat(fallback)"]
    assert len(br) == 1
    assert [ast.unparse(s) for s in br[0].body] == [
        "rows_t = rw.to(torch.int32).reshape(NI, rpi, NCH, 128).contiguous()",
        "bias_t = bs.reshape(NI, rpi, NCH * 128 * KP).contiguous()"]
    assert [ast.unparse(s) for s in br[0].orelse] == [
        "rows_t = rw.to(torch.int32).reshape(NI, rpi, NCH, 128).permute(0, 1, 3, 2).contiguous()",
        "bias_t = bs.reshape(NI, rpi, NCH, 128, KP).permute(0, 1, 2, 4, 3).reshape(NI, rpi, NCH * KP * 128)"
        ".contiguous()"]


def test_only_dsa_slots_c_passes_a_fallback_site():
    """dsa_slots_c's fallback is the "c" site; the model's calls (the context-parallel prefill's) pass none."""
    def calls(path):
        return [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == "dsa_slots_n.attend"]

    def fb(c):
        return next((ast.unparse(k.value) for k in c.keywords if k.arg == "fallback"), None)

    sc = calls(KDIR / "dsa_slots_c.py")
    assert len(sc) == 1 and fb(sc[0]) == "'c'"
    mla = calls(KDIR.parent / "models" / "mla.py")
    assert mla and not any(fb(c) for c in mla)


def _natural(args: dict, NI: int, rpi: int) -> dict:
    """The served kernel arguments with rows_t and bias back in attend's natural layout (its _nat branch)."""
    rt, bt = args["rows_t"], args["bias"]
    NCH = rt.shape[3]
    return dict(args, rows_t=rt.permute(0, 1, 3, 2).contiguous(),
                bias=bt.reshape(NI, rpi, NCH, ds.KP, 128).permute(0, 1, 2, 4, 3).reshape(NI, rpi, -1).contiguous())


def test_the_natural_arguments_differ_from_the_served_ones():
    """The positive control for the natural-layout simulator test: the two layouts really differ on these inputs."""
    q, kc, rows, bias = _inputs(8, 8, 512, 256, 1, seed=7)
    args, Np = kernel_args(q, kc, rows, bias, ds.RPI)
    nat = _natural(args, Np // ds.RPI, ds.RPI)
    assert not torch.equal(nat["rows_t"].flatten(), args["rows_t"].flatten())
    assert not torch.equal(nat["bias"].flatten(), args["bias"].flatten())


# -- the slot kernels in the NKI simulator ---------------------------------------------------------------------------


@needs_nki
@pytest.mark.parametrize("fp8,lse,N,n", [(1, 1, 8, 8), (0, 1, 7, 3), (1, 0, 12, 6)])  # RPI 4: 2/2, 1/2, 2/3 iterations
def test_split_rows_are_the_whole_kernel(monkeypatch, fp8, lse, N, n):
    """dsa_slots_n_s at grid 2 gives dsa_slots_n's bits, o and lse, over the rows the run-time count covers, fp8 and bf16
    latents, a count below the buffer's rows and an odd row count (padding) included."""
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    H, R, NS = 8, 512, 128
    q, kc, rows, bias = _inputs(N, H, R, NS, fp8, seed=N + n)
    rpi = ds.RPI
    args, Np = kernel_args(q, kc, rows, bias, rpi)
    it = torch.tensor([[-(-n // rpi)]], dtype=torch.int32)  # attend's ceil(n / rpi)
    kw = dict(args, n_iter=it, scale=0.05, fp8=fp8, want_lse=lse, qkw=ds.QKW)
    one = nki.simulate(dsn.dsa_slots_n_kernel)(**kw, rev=dsn.REV)
    two = nki.simulate(dns.dsa_slots_n_s_kernel[2])(**kw, rev=dns.REV)
    one, two = (one, two) if lse else ((one,), (two,))
    rows_run = int(it) * rpi  # the rows the run-time count covers (the rest are never written)
    for a, b in zip(one, two):
        a = torch.as_tensor(a).reshape(Np, H, -1)[:rows_run]
        b = torch.as_tensor(b).reshape(Np, H, -1)[:rows_run]
        assert torch.equal(a, b), (fp8, lse, N, n, (a != b).sum().item())
    ref = dsn.emulate(q, kc.float() if not fp8 else kc, rows, bias, 0.05, torch.tensor(n), bool(lse))
    ref = ref[0] if lse else ref
    got = torch.as_tensor(two[0]).reshape(Np, H, R)[:n]
    assert torch.allclose(got.float(), ref[:n].float(), rtol=5e-2, atol=5e-2)  # and it is dsa_slots_n's arithmetic


@needs_nki
@pytest.mark.parametrize("fp8,N,n", [(1, 20, 20), (0, 18, 13)])  # RPI 16: one full / one partial + padding
def test_bias_bf16_is_the_fp32_bias(monkeypatch, fp8, N, n):
    """The split kernel at grid 2 with the bias in bf16 (a prefill call's, at RPI_PREFILL 16) against the same call in
    fp32: o bit for bit on every row the run-time count covers, fully masked and padded rows included; lse on every row
    with a live slot. Positive control: the bf16 form's K = 1 bias matmuls took bf16 operands, the fp32 form's fp32."""
    import nki
    import nki.isa as nisa

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    H, R, NS, rpi = 8, 512, 256, 16
    q, kc, rows, bias = _inputs(N, H, R, NS, fp8, seed=N + n)
    bias[3] = ds.NEG_INF  # a fully masked row
    args, Np = kernel_args(q, kc, rows, bias, rpi)
    it = torch.tensor([[-(-n // rpi)]], dtype=torch.int32)
    kw = dict(args, n_iter=it, scale=0.05, fp8=fp8, want_lse=1, qkw=1)
    real, seen = nisa.nc_matmul, []

    def rec(*a, **k):
        st, mv = k.get("stationary"), k.get("moving")
        if st is not None and st.shape[0] == 1:  # the K = 1 bias matmul
            seen.append((str(st.dtype), str(mv.dtype)))
        return real(*a, **k)

    monkeypatch.setattr(nisa, "nc_matmul", rec)
    a = nki.simulate(dns.dsa_slots_n_s_kernel[2])(**kw, rev=dns.REV)
    k32, seen[:] = list(seen), []
    b = nki.simulate(dns.dsa_slots_n_s_kernel[2])(**dict(kw, bias=args["bias"].to(torch.bfloat16)), rev=dns.REV_BB)
    k16 = list(seen)
    assert k32 and all("float32" in s and "float32" in m for s, m in k32), k32[:2]
    assert k16 and all("bfloat16" in s and "bfloat16" in m for s, m in k16), k16[:2]
    rows_run = int(it) * rpi
    oa, ob = (torch.as_tensor(x[0]).reshape(Np, H, R)[:rows_run] for x in (a, b))
    la, lb = (torch.as_tensor(x[1]).reshape(Np, H)[:rows_run] for x in (a, b))
    assert torch.equal(oa.view(torch.int32), ob.view(torch.int32)), (oa != ob).sum().item()
    full = torch.cat([bias, bias.new_full((Np - N, NS, ds.KP), ds.NEG_INF)])[:rows_run]
    live = (full > ds.NEG_INF / 2).flatten(1).any(1)
    assert not live[3] and live.sum() >= rows_run - (Np - N) - 1
    assert torch.equal(la[live].view(torch.int32), lb[live].view(torch.int32)), (la[live] != lb[live]).sum().item()


@needs_nki
@pytest.mark.parametrize("fp8,lse,N,n", [(1, 1, 8, 8), (0, 1, 7, 3), (1, 0, 12, 6), (1, 1, 8, 5)])
def test_natural_layout_kernel_is_the_split_kernel_bit_for_bit(monkeypatch, fp8, lse, N, n):
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    H, R, NS = 8, 512, 256  # NCH 2: the slot-major / kp-minor bias order is exercised across chunks
    q, kc, rows, bias = _inputs(N, H, R, NS, fp8, seed=53 + N + n)
    rpi = ds.RPI
    args, Np = kernel_args(q, kc, rows, bias, rpi)
    it = torch.tensor([[-(-n // rpi)]], dtype=torch.int32)
    kw = dict(args, n_iter=it, scale=0.05, fp8=fp8, want_lse=lse, qkw=ds.QKW)
    served = nki.simulate(dns.dsa_slots_n_s_kernel[2])(**kw, rev=dns.REV)
    nat = nki.simulate(dn2.dsa_slots_n_s_kernel[2])(**_natural(kw, Np // rpi, rpi), rev=dn2.REV)
    served, nat = (served, nat) if lse else ((served,), (nat,))
    rows_run = int(it) * rpi
    for a, b in zip(served, nat):
        a = torch.as_tensor(a).reshape(Np, H, -1)[:rows_run]
        b = torch.as_tensor(b).reshape(Np, H, -1)[:rows_run]
        assert torch.equal(a.view(torch.int32), b.view(torch.int32)), (fp8, lse, N, n, (a != b).sum().item())


# -- dsa_slots_c: each decode row over its compacted live slots -----------------------------------------------------


def _slots_c_case(N=6, H=8, R=128, NS=256, pools=512, seed=1, dtype=torch.float32):
    """(q, kc, rows, bias): the context-parallel decode shape in small, each row a random set of live slots (a different
    count per row), the tail pool's visible prefix on every other row, one row fully masked."""
    g = torch.Generator().manual_seed(seed)
    kc = torch.randn(pools * 4, R, generator=g).to(dtype)
    q = (torch.randn(N, H, R, generator=g) * 0.1).to(dtype)
    rows = torch.randint(0, pools, (N, NS), generator=g)
    bias = torch.full((N, NS, 4), NEG)
    for b in range(N):
        sel = torch.randperm(NS - 56, generator=g)[: 10 + 9 * b]
        bias[b, sel] = 0.0
        if b % 2 == 0:
            bias[b, NS - 56, : 1 + b % 4] = 0.0
    bias[N // 2] = NEG
    return q, kc, rows, bias


def test_slots_c_plan():
    """dsa_slots_c.plan: every live slot of a row exactly once in its compacted list (pool row and live tokens), the
    rest row 0 with nv 0, partition-major order; `fits` false past C live slots."""
    q, kc, rows, bias = _slots_c_case(N=7, NS=640)
    bias[6, :300] = 0.0  # over C
    nv = dsa_slots_c.live_tokens(bias)
    crow, cnv, fits = dsa_slots_c.plan(rows, nv)
    for b in range(6):
        live = (nv[b] > 0).nonzero().view(-1)
        n = live.numel()
        assert bool(fits[b]) and n <= dsa_slots_c.C
        p, ch = live % 128, live // 128
        order = live[torch.argsort(p * 8 + ch)]  # partition-major
        assert torch.equal(crow[b, :n], rows[b, order]) and torch.equal(cnv[b, :n], nv[b, order])
        assert (crow[b, n:] == 0).all() and (cnv[b, n:] == 0).all()
    assert not bool(fits[6])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_slots_c_emulation(dtype):
    """dsa_slots_c.emulate: each row over its compacted live slots equals dsa_slots.emulate over all slots (to fp32
    summation order) on every row with a visible token; a call with a row over C live slots is dsa_slots.emulate
    exactly (attend's dsa_slots_n call)."""
    q, kc, rows, bias = _slots_c_case(NS=640, dtype=dtype)
    o, lse = dsa_slots_c.emulate(q, kc, rows, bias, 0.125, lse=True)
    wo, wl = ds.emulate(q, kc, rows, bias, 0.125, lse=True)
    real = (bias > NEG / 2).any(-1).any(-1)
    assert not real.all() and real.sum() == q.shape[0] - 1
    tol = 1e-5 if dtype == torch.float32 else 5e-3
    torch.testing.assert_close(o[real], wo[real], rtol=tol, atol=tol)
    torch.testing.assert_close(lse[real], wl[real], rtol=1e-6, atol=1e-5)
    assert (lse[~real] < -1e29).all() and torch.isfinite(o).all()
    over = bias.clone()
    over[0, :200] = 0.0
    o2, l2 = dsa_slots_c.emulate(q, kc, rows, over, 0.125, lse=True)
    w2, wl2 = ds.emulate(q, kc, rows, over, 0.125, lse=True)
    assert torch.equal(o2, w2) and torch.equal(l2, wl2)


@needs_nki
@pytest.mark.parametrize("over", [False, True])
def test_slots_c_simulator(over, monkeypatch):
    """dsa_slots_c's kernel under nki.simulate: every row equals dsa_slots_c.emulate (rows padded to RPI); a call with a
    row over C live slots gives the kernel no iteration (attend's dsa_slots_n call takes it)."""
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    q, kc, rows, bias = _slots_c_case(N=6, R=512, NS=256, dtype=torch.bfloat16)  # (the kernel's PSUM holds R = 512)
    if over:
        bias[0, :150] = 0.0
    kw, N, Np, ov = dsa_slots_c.kernel_args(q, kc, rows, bias, 0.125, lse=True)
    assert float(ov) == float(over) and int(kw["n_c"]) == (0 if over else Np // min(dsa_slots_c.RPI, N))
    if over:
        return
    o, ls = nki.simulate(dsa_slots_c.dsa_slots_c_kernel)(**kw)
    H, R = q.shape[1], q.shape[2]
    o = torch.as_tensor(o).float().reshape(Np, H, R)[:N]
    ls = torch.as_tensor(ls).float().reshape(Np, H)[:N]
    wo, wl = dsa_slots_c.emulate(q.float(), kc, rows, bias, 0.125, lse=True)
    real = (bias > NEG / 2).any(-1).any(-1)
    err = ((o - wo)[real].abs().max() / wo[real].abs().max()).item()
    assert err < 1e-2, err
    assert ((ls - wl)[real].abs().max()).item() < 1e-2
    assert (ls[~real] < -1e29).all()


# -- the long-context selection -----------------------------------------------------------------------------------------


def _subs(P: int, keep: int):
    """Every sub-block size the kernel takes at P pools (0: one level)."""
    return [sub for sub in (0, 2, 4, 8, 16, 32) if dl.supported(32, 128, P, keep, sub)]


def _wide_cases():
    """(name, scores, keep) at the kernel's keep sizes: tests/test_dsa_long.py's adversarial kinds where keep divides by
    8, plus keep 512 over thousands of pools with plateaus at the threshold across sub-block boundaries."""
    from tests.test_dsa_long import _cases

    out = [(n, s, k) for n, s, k in _cases() if k % 8 == 0]
    g = torch.Generator().manual_seed(2)
    for P in (2112, 8448):
        N = 6
        npool = torch.tensor([0, 511, 512, 513, P // 2, P])
        cand = torch.arange(P).view(1, P) < npool.view(N, 1)
        for kind in ("randn", "int", "plateau", "zeros"):
            if kind == "randn":
                s = torch.randn(N, P, generator=g)
            elif kind == "int":
                s = torch.randint(0, 3, (N, P), generator=g).float()
            elif kind == "plateau":
                s = torch.full((N, P), 1.0)
                s[:, ::7] = 2.0
                s[:, -5:] = 9.0
            else:
                s = torch.zeros(N, P)
            out.append((f"{kind} P{P} k512", torch.where(cand, s, NEG), 512))
    return out


def test_selection_algorithm_equals_reference():
    """emulate_algorithm (the kernel's steps: sub-block maxima, extraction rounds, ascending sub-block order, candidates,
    the final sort) == select_reference, pools and their scores, for every sub-block size the kernel takes, on
    adversarial ties."""
    n = 0
    for name, sc, keep in _wide_cases():
        want_p, want_c = dsa_long.select_reference(sc, keep)
        k = torch.arange(keep).view(1, keep)
        want_v = torch.where(k < want_c.view(-1, 1), sc.gather(1, want_p), NEG)
        for sub in _subs(sc.shape[1], keep):
            got_p, got_v = dl.emulate_algorithm(sc, keep, sub)
            assert torch.equal(got_p, want_p), f"{name} sub {sub}"
            assert torch.equal(got_v, want_v), f"{name} sub {sub}"
            n += 1
    assert n > 50


def test_pick_sub():
    """The two levels are used only where they extract over fewer values than the whole row."""
    assert dl.pick_sub(2112, 512) in (0, 2)
    for P in (32768, 131072, 262144):
        sub = dl.pick_sub(P, 512)
        Pp = -(-P // dl.CH) * dl.CH
        assert sub and Pp // sub + 512 * sub < Pp and dl.supported(32, 128, P, 512, sub)
    with pytest.raises(NotImplementedError):
        dl.pick_sub(16384 * 64, 512)


def test_emulate_is_reference_on_its_scores():
    """emulate(): the scores of emulate_scores, then the exact selection, with the selected pools' scores and NEG_INF
    past the count."""
    g = torch.Generator().manual_seed(4)
    N, Hi, D, P, keep = 12, 32, 128, 3000, 64
    q = torch.randn(N, Hi, D, generator=g).to(torch.bfloat16)
    w = torch.randn(N, Hi, generator=g) * Hi ** -0.5
    pk = torch.randn(P, D, generator=g).to(torch.bfloat16)
    npool = torch.tensor([0, 1, 63, 64, 65, 1500, 2999, 3000, 3000, 700, 9, 2048])
    pools, cnt, vals = dl.emulate(q, w, pk, npool, keep, 128 ** -0.5)
    cand = torch.where(torch.arange(P).view(1, P) < npool.view(N, 1), 0.0, NEG)
    sc = dl.emulate_scores(q, w, pk, cand, 128 ** -0.5)
    rp, rc = dsa_long.select_reference(sc, keep)
    assert torch.equal(pools, rp) and torch.equal(cnt, rc) and torch.equal(cnt, npool.clamp(max=keep))
    k = torch.arange(keep).view(1, keep)
    assert torch.equal(vals, torch.where(k < cnt.view(N, 1), sc.gather(1, pools), NEG))
    ref = dsa_long.scores(q, w, pk, npool, 128 ** -0.5)
    vis = cand == 0
    assert ((ref - sc).abs()[vis] <= 1e-5 * ref[vis].abs().clamp(min=1)).all()  # the model's scores, to fp32 order


def test_kernel_input_layout():
    """kernel_inputs: tile t's queries transposed head-major ([t, d, h, q] = qI[128 t + q, h, d]), weights and candidate
    counts per tile, padded query rows with no candidate, pools padded to whole chunks of 512."""
    g = torch.Generator().manual_seed(5)
    N, Hi, D, P = 200, 32, 128, 1000
    q = torch.randn(N, Hi, D, generator=g)
    w = torch.randn(N, Hi, generator=g)
    pk = torch.randn(P, D, generator=g)
    npool = torch.randint(0, P + 1, (N,), generator=g)
    kin = dl.kernel_inputs(q, w, pk, npool)
    assert kin["qT"].shape == (2, D, Hi, 128) and kin["pk"].shape == (1024, D)
    t, qq, h, d = 1, 50, 7, 33
    assert kin["qT"][t, d, h, qq] == q[128 * t + qq, h, d].to(torch.bfloat16)
    assert kin["w"][t, qq, h] == w[128 * t + qq, h]
    assert kin["npool"][t, qq, 0] == float(npool[128 * t + qq])
    assert (kin["npool"].reshape(-1)[N:] == 0).all() and (kin["pk"][P:] == 0).all()


def test_value_order():
    """The value-ordered local lists (the slot classes' form): the same pools, scores and count as the ascending form,
    ordered by score descending and equal scores by pool ascending, 0 / NEG_INF past the count (integer scores for ties;
    a context shorter than keep)."""
    g = torch.Generator().manual_seed(9)
    N, Hi, D, P, keep = 6, 4, 128, 700, 64
    qI = torch.randint(-2, 3, (N, Hi, D), generator=g).to(torch.bfloat16)
    w = torch.randint(0, 3, (N, Hi), generator=g).float()
    pk = torch.randint(-1, 2, (P, D), generator=g).to(torch.bfloat16)
    npool = torch.tensor([0, 5, keep, keep + 3, 400, P])
    a_p, a_c, a_v = dl.emulate(qI, w, pk, npool, keep, 0.25)
    b_p, b_c, b_v = dl.emulate(qI, w, pk, npool, keep, 0.25, vorder=True)
    assert torch.equal(a_c, b_c)
    assert all(torch.equal(x, y) for x, y in zip(dl.value_order(a_p, a_c, a_v), (b_p, b_c, b_v)))
    for n in range(N):
        c = int(a_c[n])
        pairs = list(zip(b_v[n, :c].tolist(), b_p[n, :c].tolist()))
        assert pairs == sorted(pairs, key=lambda x: (-x[0], x[1]))
        assert (b_p[n, c:] == 0).all() and (b_v[n, c:] < -1e29).all()
        assert dict(zip(b_p[n, :c].tolist(), b_v[n, :c].tolist())) == dict(zip(a_p[n, :c].tolist(),
                                                                                a_v[n, :c].tolist()))


def test_long_pipe_dispatch(monkeypatch):
    """At grid 2 (the served LNC split names dsa_long_pipe) the split kernel and its REV; at grid 1 the whole kernel."""
    from trnrt import platform
    from trnrt.kernels import dsa_long_pipe_s

    monkeypatch.setattr(platform, "nki_grid", lambda: 2)
    assert (dlp._kernel(), dlp._rev()) == (dsa_long_pipe_s.dsa_long_pipe_s_kernel, dsa_long_pipe_s.REV)
    assert dsa_long_pipe_s.REV not in (dlp.REV, dlp.REV_LNC2)
    monkeypatch.setattr(platform, "nki_grid", lambda: 1)
    assert (dlp._kernel(), dlp._rev()) == (dlp.dsa_long_pipe_kernel, dlp.REV)


def _pipe_inputs(NT, Hi, P, keep, seed):
    g = torch.Generator().manual_seed(seed)
    N = NT * 128
    qI = torch.randn(N, Hi, 128, generator=g)
    w = torch.rand(N, Hi, generator=g)
    pk = torch.randn(P, 128, generator=g)
    npool = torch.randint(keep, P + 1, (N,), generator=g)
    npool[::7] = torch.randint(0, keep, (len(npool[::7]),), generator=g)  # fewer visible pools than keep
    return qI, w, pk, npool


@needs_nki
@pytest.mark.parametrize("NT,sub,ktall,vorder", [(2, 8, 1, 0), (3, 8, 0, 1), (4, 0, 1, 0)])
def test_split_tiles_are_the_whole_kernel(monkeypatch, NT, sub, ktall, vorder):
    """dsa_long_pipe_s at grid 2 gives dsa_long_pipe's grid-1 bits, pools and scores, over two-level and one-level
    selection, an odd tile count, value order, both K^T forms and rows with fewer visible pools than keep."""
    import nki

    from trnrt.kernels import dsa_long_pipe_s

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    Hi, P, keep = 4, 2048, 64
    assert dl.supported(Hi, 128, P, keep, sub)
    qI, w, pk, npool = _pipe_inputs(NT, Hi, P, keep, seed=NT)
    kw = dict(dl.kernel_inputs(qI, w, pk, npool), keep=keep, scale=0.088, sub=sub or dl.CH,
              lsub=(sub or dl.CH).bit_length() - 1, one=int(sub == 0), per_chunk=dlp.PER_CHUNK, ktall=ktall,
              vorder=vorder)
    one = nki.simulate(dlp.dsa_long_pipe_kernel)(**kw, rev=dlp.REV)
    two = nki.simulate(dsa_long_pipe_s.dsa_long_pipe_s_kernel[2])(**kw, rev=dsa_long_pipe_s.REV)
    for name, a, b in zip(("pools", "scores"), one, two):
        a, b = torch.as_tensor(a), torch.as_tensor(b)
        assert torch.equal(a, b), (NT, sub, name, (a != b).sum().item())
    pools = torch.as_tensor(two[0]).reshape(NT * 128, keep)
    cnt = npool.clamp(max=keep)
    for r in range(0, NT * 128, 37):  # the tiles hold a selection (sanity, not the claim)
        got = pools[r, :int(cnt[r])]
        assert ((got >= 0) & (got < int(npool[r]))).all(), r
