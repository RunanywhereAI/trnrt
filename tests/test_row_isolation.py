"""Row isolation under a non-finite row (kernels/rowiso.py, fin=2, the served form). kernels/kda_shell.py puts the rows
on the contraction of PE transposes (identity matmuls) for its conv path, so without isolation one row's NaN reached
every row in the simulator (NaN * 0 = NaN). With `fin` each row's NaN / inf is zeroed first, by bit operations only.
Under the NKI simulator (the Neuron venv):
- a NaN or an inf in one row's input (and in one row's conv history): every other row of every output bit for bit the
  clean run's;
- the positive control: the kernel without isolation (fin=0) on the same input breaks that;
- finite inputs: every fin bit for bit fin=0 on every output (the isolation moves no finite bit).
On the host: the helpers' arithmetic over every bf16 bit pattern."""

import importlib.util

import numpy as np
import pytest
import torch

from tests import test_kda_shell as TKS
from trnrt.kernels import kda_shell as KS
from trnrt.kernels import rowiso as RI
from trnrt.kernels import shell

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (simulator)")
T = 4
BAD = 1  # the row made non-finite (position > 0, so it reads its own state rows)


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype]) if t.is_floating_point() else t


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


# -- the KDA shell ------------------------------------------------------------------------------------------------------


def _kda_case(seed: int = 41):
    c = TKS.case(T, seed, 2)
    L = c["L"]
    t = KS.arrange(
        c["wcat"],
        L.conv_w,
        L.f_b,
        L.g_b,
        L.dt_bias,
        L.out,
        c["fn"],
        L.in_norm,
        L.A_log,
        L.o_norm,
        c["base"],
        c["scale"],
        TKS.NV,
        2,
        TKS.HC,
    )
    c["blobs"] = KS.pack(t, TKS.H, TKS.NV, TKS.K, 2, TKS.HC)
    return c


def _kda(c, S, conv, fin: int, cbat: int):
    L = c["L"]
    _, post, comb, Pp, cpo, rpo = KS.run(
        S,
        c["blobs"],
        conv.clone(),
        L.rec_state.clone(),
        c["slots"],
        H=TKS.H,
        nv=TKS.NV,
        hc=TKS.HC,
        iters=TKS.ITERS,
        hc_eps=TKS.HC_EPS,
        rms_eps=TKS.EPS,
        lb=TKS.LB,
        simulate=True,
        grid=2,
        recb=cbat,
        cbat=cbat,
        fin=fin,
    )
    wr = c["slots"][1].long()
    return {"post": post, "comb": comb, "P": Pp.transpose(0, 1), "conv": cpo[wr], "rec": rpo[wr]}  # rows first


def _others_same(clean: dict, got: dict) -> list:
    """The outputs whose rows other than BAD differ from the clean run's (bit for bit)."""
    keep = [r for r in range(T) if r != BAD]
    return [k for k in clean if not _same(clean[k][keep], got[k][keep])]


def _kda_bad(c, what: str):
    S, conv = c["S"].clone(), c["L"].conv_state.clone()
    if what == "S_nan":
        S[BAD, 0, 5] = float("nan")
    elif what == "S_inf":
        S[BAD, 2, 9] = float("inf")
    else:  # "conv_nan": the bad row's conv history (its read row)
        conv[int(c["slots"][0, BAD]), 1, 7] = float("nan")
    return S, conv


@needs_nki
@pytest.mark.parametrize("fin", [1, 2])
@pytest.mark.parametrize("cbat", [0, 1])
@pytest.mark.parametrize("what", ["S_nan", "S_inf", "conv_nan"])
def test_kda_shell_keeps_a_bad_row_in_its_row(what, cbat, fin, monkeypatch):
    """fin 2 (rowiso) zeroes xn, upstream of pr, and this program's conv history columns: every injection here (S, and
    the conv history the transposes read) is at or upstream of a fin 2 site."""
    if what == "conv_nan" and cbat == 0:
        pytest.skip("cbat 0 reads each row's conv history alone (no transpose across rows)")
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    c = _kda_case()
    clean = _kda(c, c["S"], c["L"].conv_state, fin, cbat)
    S, conv = _kda_bad(c, what)
    got = _kda(c, S, conv, fin, cbat)
    assert _others_same(clean, got) == []
    assert all(torch.isfinite(got[k]).all() for k in ("P", "conv", "rec"))  # zeroed, not poisoned, in the bad row too


@needs_nki
@pytest.mark.parametrize(("what", "cbat"), [("conv_nan", 1)])
def test_kda_shell_served_form_spreads(what, cbat):
    """The positive control: fin=0 (no isolation) lets the conv path's transposes carry the bad row into the others. The
    projection is weights-stationary, so only the conv path's transposes contract over the rows: the control is a bad
    conv history row; an S row stays in its row (the next test)."""
    c = _kda_case()
    clean = _kda(c, c["S"], c["L"].conv_state, 0, cbat)
    S, conv = _kda_bad(c, what)
    assert _others_same(clean, _kda(c, S, conv, 0, cbat)) != []


@needs_nki
@pytest.mark.parametrize(("what", "cbat"), [("S_nan", 1), ("S_inf", 0)])
def test_kda_shell_served_form_keeps_a_bad_S_row(what, cbat):
    """The weights-stationary projection puts the features on the partitions straight from the PE, so it has no PE
    transposes over the rows for a bad S row's NaN or inf to cross, even without isolation (fin 0). Not a relaxed test:
    test_kda_shell_served_form_spreads keeps a case that spreads, so a detector that never fires cannot pass both. The
    bad row's own P must be non-finite (the injection got in)."""
    c = _kda_case()
    clean = _kda(c, c["S"], c["L"].conv_state, 0, cbat)
    S, conv = _kda_bad(c, what)
    got = _kda(c, S, conv, 0, cbat)
    assert _others_same(clean, got) == []
    assert not torch.isfinite(got["P"][BAD]).all()


@needs_nki
@pytest.mark.parametrize("fin", [1, 2])
@pytest.mark.parametrize("cbat", [0, 1])
def test_kda_shell_fin_moves_no_finite_bit(cbat, fin, monkeypatch):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    c = _kda_case(seed=43)
    on, off = _kda(c, c["S"], c["L"].conv_state, fin, cbat), _kda(c, c["S"], c["L"].conv_state, 0, cbat)
    assert [k for k in on if not _same(on[k], off[k])] == []


def _fz_reference(bits: torch.Tensor) -> torch.Tensor:
    """finite_or_zero's arithmetic on bf16 bit patterns (int64 0 .. 65535), in fp32 as the engine computes it."""
    b = bits.to(torch.float32)
    f = b - 32768.0 * (b > 32767.5).float()
    keep = 1.0 - (f > 32639.5).float()
    return (b * keep).to(torch.int64)


def test_the_arithmetic_finds_exactly_the_non_finite_bf16():
    """On the host over every bf16 bit pattern: a value is zeroed iff its exponent bits are all set (NaN, +-inf); every
    other pattern, -0 and the subnormals included, keeps its bits; every intermediate is an integer below 2^24 (exact
    in fp32)."""
    bits = torch.arange(1 << 16, dtype=torch.int64)
    out = _fz_reference(bits)
    nonfinite = (bits & 0x7F80) == 0x7F80
    assert int(nonfinite.sum()) == 2 * 128  # +-inf and the NaNs: 2^7 mantissas per sign
    assert torch.equal(out[nonfinite], torch.zeros_like(out[nonfinite]))
    assert torch.equal(out[~nonfinite], bits[~nonfinite])


if importlib.util.find_spec("nki") is not None:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def _fz_kernel(x):
        """shell.finite_or_zero on a bf16 [p, n] input, returned."""
        p, n = x.shape
        t = nl.ndarray((p, n), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x)
        f = nl.ndarray((p, n), dtype=nl.float32, buffer=nl.sbuf)
        d = nl.ndarray((p, n), dtype=nl.uint16, buffer=nl.sbuf)
        shell.finite_or_zero(t.view(nl.uint16), d, f)
        out = nl.ndarray((p, n), dtype=nl.uint16, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=d)
        return out


def _fz_sim(bits: np.ndarray, monkeypatch) -> np.ndarray:
    import ml_dtypes

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    out = nki.simulate(_fz_kernel)(x=bits.astype(np.uint16).view(ml_dtypes.bfloat16))
    return np.asarray(out).astype(np.int64)


@needs_nki
def test_finite_or_zero_every_bf16_pattern_in_the_simulator(monkeypatch):
    """The kernel helper over all 65,536 bf16 patterns ([128, 512]): bit for bit the reference above."""
    bits = np.arange(1 << 16, dtype=np.int64).reshape(128, 512)
    got = _fz_sim(bits, monkeypatch)
    want = _fz_reference(torch.from_numpy(bits)).numpy()
    assert np.array_equal(got, want), int((got != want).sum())


@needs_nki
def test_finite_or_zero_mixed_neighbours_in_the_simulator(monkeypatch):
    """A non-finite value beside a finite one, both orders, in one row: the finite one keeps its bits exactly (each bf16
    is its own lane: no packing, so neighbours cannot touch), the non-finite one becomes +0."""
    nan, qnan, inf, ninf, nzero, sub, big, one = 0x7FC0, 0x7FA5, 0x7F80, 0xFF80, 0x8000, 0x0001, 0x7F7F, 0x3F80
    row = [nan, one, one, nan, inf, nzero, nzero, ninf, qnan, sub, sub, qnan, 0xFFC1, big, big, 0xFFFF]
    bits = np.array([row] * 4, dtype=np.int64)
    got = _fz_sim(bits, monkeypatch)
    nonfinite = (bits & 0x7F80) == 0x7F80
    assert np.array_equal(got[nonfinite], np.zeros(int(nonfinite.sum()), dtype=np.int64))
    assert np.array_equal(got[~nonfinite], bits[~nonfinite])


# -- fin 2 (kernels/rowiso.py) -------------------------------------------------------------------------------------


def _fz3_reference(bits: torch.Tensor) -> torch.Tensor:
    """finite_or_zero3's arithmetic on bf16 bit patterns, in fp32 as the engine computes it."""
    b = bits.to(torch.float32)
    m = (b > 32767.5).float() * -32768.0 + b
    return (b * (m < 32639.5).float()).to(torch.int64)


def test_finite_or_zero3_is_finite_or_zero_on_every_pattern():
    """The three-instruction form computes fin=1's five-instruction result on every bf16 bit pattern (so fin 2 moves no
    finite bit and zeroes exactly the NaN / +-inf), every intermediate an integer below 2^24."""
    bits = torch.arange(1 << 16, dtype=torch.int64)
    assert torch.equal(_fz3_reference(bits), _fz_reference(bits))


if importlib.util.find_spec("nki") is not None:

    @nki.jit
    def _fz3_kernel(x):
        """rowiso.finite_or_zero3 on a bf16 [p, n] input, returned."""
        p, n = x.shape
        t = nl.ndarray((p, n), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x)
        f = nl.ndarray((p, n), dtype=nl.float32, buffer=nl.sbuf)
        d = nl.ndarray((p, n), dtype=nl.uint16, buffer=nl.sbuf)
        RI.finite_or_zero3(t.view(nl.uint16), d, f)
        out = nl.ndarray((p, n), dtype=nl.uint16, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=d)
        return out


def _bf16_bits(bits: np.ndarray):
    import ml_dtypes

    return bits.astype(np.uint16).view(ml_dtypes.bfloat16)


@needs_nki
def test_finite_or_zero3_every_bf16_pattern_in_the_simulator(monkeypatch):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    bits = np.arange(1 << 16, dtype=np.int64).reshape(128, 512)
    got = np.asarray(nki.simulate(_fz3_kernel)(x=_bf16_bits(bits))).astype(np.int64)
    want = _fz_reference(torch.from_numpy(bits)).numpy()
    assert np.array_equal(got, want), int((got != want).sum())


@needs_nki
def test_kda_shell_fin2_holds_a_projection_overflow(monkeypatch):
    """fin 2's narrower contract at kda_shell. fin 2 zeroes xn, so a FINITE xn row whose projection overflows to inf is
    not zeroed. The weights-stationary projection has no PE transposes over the rows, so the inf stays in its row. That the overflow happened in the kernel: BAD's row differs between fin 1 (which zeroes the inf)
    and fin 2 (which keeps it), as on finite inputs every fin is bit for bit the others (test_kda_shell_fin_moves_no_
    finite_bit); BAD's own outputs come out finite (the gate saturates the inf). fin 1 holds as before. Built by aiming
    f_a's first row at row BAD's xn: s sign(xn_BAD), s making BAD's dot product 1.5 fp32_max while every other row's
    stays below fp32_max / 8 (asserted on the host first).
    Why it cannot happen served [inference, not measured on the released weights]: |pr_c| = |sum_h xn_h W_hc| <=
    ||xn||_2 ||W_c||_2 <= (max|in_norm| sqrt(H)) (sqrt(H) max|W|) = H max|in_norm| max|W|, as x / rms(x) has norm
    sqrt(H); an overflow needs max|in_norm| max|W| >= fp32_max / H, ~8.3e34 at H 4096, where trained norm weights and
    projections are O(1)."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    c = _kda_case(seed=47)
    L = c["L"]
    x = TKS._kernel_x(c, 2, False)[0].float()  # the kernel's own x (mhc_pre's), [T, H]
    xn = L.in_norm.float() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + TKS.EPS)).to(torch.bfloat16).float()
    sgn = torch.sign(xn[BAD])
    fmax = torch.finfo(torch.float32).max
    s = 1.5 * fmax / float((xn[BAD] * sgn).sum())
    dots = (xn * sgn).sum(-1) * s
    keep = [r for r in range(T) if r != BAD]
    assert float(dots[BAD]) > fmax and float(dots[keep].abs().max()) < fmax / 8, dots  # the construction holds
    w = c["wcat"].clone()
    w[3 * TKS.NV * TKS.D] = (sgn * s).to(w.dtype)  # f_a's first row (wcat = in_qkv, f_a, g_a, in_b)
    t = KS.arrange(
        w,
        L.conv_w,
        L.f_b,
        L.g_b,
        L.dt_bias,
        L.out,
        c["fn"],
        L.in_norm,
        L.A_log,
        L.o_norm,
        c["base"],
        c["scale"],
        TKS.NV,
        2,
        TKS.HC,
    )
    c2 = dict(c, wcat=w, blobs=KS.pack(t, TKS.H, TKS.NV, TKS.K, 2, TKS.HC))
    S_ok = c["S"].clone()
    S_ok[BAD] = c["S"][(BAD + 1) % T]  # a finite row the weights are not aimed at
    ref = _kda(c2, S_ok, L.conv_state, 1, 1)
    f1 = _kda(c2, c["S"], L.conv_state, 1, 1)
    assert _others_same(ref, f1) == []  # fin 1 zeroes the inf: it holds
    got = _kda(c2, c["S"], L.conv_state, 2, 1)
    assert _others_same(ref, got) == []  # fin 2 keeps it in its row
    assert any(not _same(f1[k][BAD], got[k][BAD]) for k in got)  # the overflow did happen (fin 1 zeroed something)


def _count_isa(monkeypatch, run, names=("tensor_scalar", "scalar_tensor_tensor", "tensor_tensor", "nc_matmul")):
    """How many times the trace calls each nisa op (the forms must trace differently, or a bit-equality compares a
    form with itself)."""
    counts = dict.fromkeys(names, 0)
    real = {n: getattr(nisa, n) for n in names}
    for n in names:

        def spy(*a, _n=n, **kw):
            counts[_n] += 1
            return real[_n](*a, **kw)

        monkeypatch.setattr(nisa, n, spy)
    try:
        run()
    finally:
        for n in names:
            monkeypatch.setattr(nisa, n, real[n])
    return counts


def _per_trace(d0: dict, d1: dict, want: dict) -> bool:
    """d1 - d0 is `want` once per traced program (the simulator traces a grid's programs one or npg times)."""
    d = {k: d1[k] - d0[k] for k in want}
    return any(d == {k: v * n for k, v in want.items()} for n in (1, 2))


@needs_nki
@needs_nki
def test_fin2_traces_differently_from_fin1(monkeypatch):
    """The positive control: per program, kda_shell fin 2 traces 7 tensor_scalar and 3 tensor_tensor fewer and 1
    scalar_tensor_tensor more than fin 1: fin 1's three finite_or_zero sites (5 instructions each: ft and beta, and the
    conv history) against fin 2's two finite_or_zero3 sites (3 each)."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    c = _kda_case()
    k = {f: _count_isa(monkeypatch, lambda f=f: _kda(c, c["S"], c["L"].conv_state, f, 1)) for f in (1, 2)}
    assert _per_trace(
        k[1], k[2], {"tensor_scalar": -7, "scalar_tensor_tensor": 1, "tensor_tensor": -3, "nc_matmul": 0}
    ), k


def test_the_served_isolation_is_fin_2():
    assert RI.FIN == 2
