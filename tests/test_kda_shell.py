"""kernels/kda_shell.py, the KDA attention block of a decode step as one NKI call (the previous block's mix in, mhc_pre,
the KDA half, per-program partials of the output projection), and its wiring into the model.

Host:
- kda_half.emulate (the half the shell's emulation runs) is linear_attn's served decode form on a stand-in layer with
  GLM-5.3-Flash's per-rank shapes (2 heads of 128, conv 768 x 4, hidden 4096), from the same state rows; a wrong gate
  bound is caught;
- the packed layout holds every arranged tensor; the per-program partials add up to the half; the mix-in is mhc_mix's;
- init_layer gives a KDA layer its two packed decode-only blobs, fill() packs arrange() of the loaded weights into them,
  and the call's composition (the shell's S, post, comb and partials, then mix_out) is the served attention block;
- decode() passes the layer's blobs, dims and constants through, a sequence start reading the zero row.
Simulator (Neuron venv): the kernel against the emulation at grids 1 and 2 with and without the mix-in, x pinned to
the kernel's own mhc_pre, twelve steps of state, the released comb prior, and each pair of served forms (the
row-batched recurrence and the pair pipeline, the batched xn and the per-row xn, the batched conv and the row-at-a-time
conv) bit for bit, with positive controls that must fail."""

from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trnrt.kernels import kda_half as KH
from trnrt.kernels import kda_shell as KS
from trnrt.kernels import mhc_mix as MM
from trnrt.kernels import shell as SH
from trnrt.models import hybrid as HY
from trnrt.models import linear_attn as LA
from trnrt.models.decoder import rms_norm

H, NV, D, K, LB, EPS = 4096, 2, 128, 4, -5.0, 1e-5
CD = 3 * NV * D
HC, ITERS, HC_EPS = 4, 20, 1e-6
O = 2 * HC + HC * HC
NKI = importlib.util.find_spec("nki") is not None
bf = torch.bfloat16


def stand_in(seed: int, R: int = 9):
    """A KDA layer at GLM-5.3-Flash's per-rank shapes (tp 32: 2 heads of 128), with state pools of R rows."""
    g = torch.Generator().manual_seed(seed)

    def w(*shape, s=0.02):
        return (torch.randn(*shape, generator=g) * s).to(bf)

    spec = SimpleNamespace(kind="kda", conv_kernel=K, head_k_dim=D, head_v_dim=D, gate_act="sigmoid", lower_bound=LB)
    L = SimpleNamespace(spec=spec, nk=NV, nv=NV, conv_w=w(CD, K, s=0.3), f_a=w(D, H), f_b=w(NV * D, D, s=0.1),
                        dt_bias=torch.randn(NV * D, generator=g) * 0.5, A_log=torch.randn(NV, generator=g) * 0.3,
                        g_a=w(D, H), g_b=w(NV * D, D, s=0.1), o_norm=1 + torch.randn(D, generator=g) * 0.1,
                        in_qkv=w(CD, H), in_b=w(NV, H), out=w(H, NV * D),
                        in_norm=(1 + torch.randn(H, generator=g) * 0.1).to(bf),
                        conv_state=torch.randn(R, K - 1, CD, generator=g).to(bf),
                        rec_state=torch.randn(R, NV, D, D, generator=g) * 0.1)
    m = SimpleNamespace(cfg=SimpleNamespace(rms_norm_eps=EPS, hybrid=None), dtype=bf, page_size=256,
                        _w=lambda layer, name: getattr(layer, name))
    return m, L, g


@pytest.mark.parametrize("B", [1, 2, 4])
def test_the_half_emulation_is_the_served_decode_form(B):
    """Two sides from different sources: kda_half.emulate (one fused projection, written for the kernel) against
    linear_attn._mix_rows' decode form; one row starts its sequence (position 0: zero conv window and state)."""
    m, L, g = stand_in(B)
    R = L.rec_state.shape[0]
    x = (torch.randn(B, H, generator=g) * 0.5).to(bf)
    positions = torch.arange(B) + 7
    positions[0] = 0
    slot = torch.randperm(R - 1, generator=g)[:B] + 1
    conv0, rec0 = L.conv_state.clone(), L.rec_state.clone()
    with torch.no_grad():
        want = LA._mix_rows(m, L, rms_norm(x, L.in_norm, EPS), positions, torch.zeros(B, dtype=torch.long), slot)
    want_conv, want_rec = L.conv_state.clone(), L.rec_state.clone()
    slots = torch.stack([torch.where(positions > 0, slot, torch.full_like(slot, R)), slot]).to(torch.int32)
    wcat = torch.cat([L.in_qkv, L.f_a, L.g_a, L.in_b])
    with torch.no_grad():
        y, conv, rec = KH.emulate(x, L.in_norm, wcat, L.conv_w, conv0, L.f_b, L.g_b, L.dt_bias, L.A_log, L.o_norm,
                                  L.out, rec0, slots, LB, EPS)
    # the fused projection may sum its columns in another blocking than four separate linears: allow bf16 ulps
    assert torch.allclose(y.float(), want.float(), rtol=2e-2, atol=2e-3), (y.float() - want.float()).abs().max()
    # row 0 is the scratch row, which _write_rows' one-row form also writes
    assert torch.allclose(conv[1:].float(), want_conv[1:].float(), rtol=1e-2, atol=1e-3)
    assert torch.allclose(rec[1:], want_rec[1:], rtol=1e-3, atol=1e-4), (rec[1:] - want_rec[1:]).abs().max()
    assert not torch.equal(want_conv[slot], conv0[slot])


def test_a_wrong_gate_is_caught():
    """Positive control: the emulation with the gate's lower bound off by one is NOT the served form (else the test
    above could pass a broken emulation)."""
    m, L, g = stand_in(11)
    B = 2
    x = (torch.randn(B, H, generator=g) * 0.5).to(bf)
    positions = torch.tensor([3, 4])
    slot = torch.tensor([1, 2])
    conv0, rec0 = L.conv_state.clone(), L.rec_state.clone()
    with torch.no_grad():
        want = LA._mix_rows(m, L, rms_norm(x, L.in_norm, EPS), positions, torch.zeros(B, dtype=torch.long), slot)
        y, _, _ = KH.emulate(x, L.in_norm, torch.cat([L.in_qkv, L.f_a, L.g_a, L.in_b]), L.conv_w, conv0, L.f_b,
                             L.g_b, L.dt_bias, L.A_log, L.o_norm, L.out, rec0,
                             torch.stack([slot, slot]).to(torch.int32), LB - 1.0, EPS)
    assert not torch.allclose(y.float(), want.float(), rtol=2e-2, atol=2e-3)


def case(T: int, seed: int, npg: int):
    """The stand-in layer plus a hyper-connection, T rows."""
    m, L, g = stand_in(seed, R=2 * T + 1)
    R = L.rec_state.shape[0]
    fn = (torch.randn(O, HC * H, generator=g) * 0.02).to(bf)
    base = torch.randn(O, generator=g) * 0.5
    scale = torch.rand(3, generator=g) + 0.5
    S = (torch.randn(T, HC, H, generator=g) * 0.5).to(bf)
    positions = torch.arange(T) + 7
    positions[0] = 0
    slot = torch.randperm(R - 1, generator=g)[:T] + 1
    slots = SH.slots(slot, positions, R)
    wcat = torch.cat([L.in_qkv, L.f_a, L.g_a, L.in_b])
    post_p = torch.rand(T, HC, generator=g) * 2
    comb_p = torch.softmax(torch.randn(T, HC, HC, generator=g), -1)
    Yp = (torch.randn(npg, T, H, generator=g) * 0.3).to(bf)
    return dict(L=L, fn=fn, base=base, scale=scale, S=S, slots=slots, wcat=wcat, pend=(post_p, comb_p, Yp))


def emulated(c, npg: int, mix: bool, lb: float = LB):
    L = c["L"]
    with torch.no_grad():
        return KS.emulate(c["S"], c["fn"], c["base"], c["scale"], ITERS, HC_EPS, EPS, L.in_norm, c["wcat"], L.conv_w,
                          L.conv_state.clone(), L.f_b, L.g_b, L.dt_bias, L.A_log, L.o_norm, L.out, L.rec_state.clone(),
                          c["slots"], lb, npg, pend=c["pend"] if mix else None)


def test_arrange_fn_is_h_layout_per_stream():
    fn = torch.randn(O, HC * H)
    f = SH.arrange_fn(fn, HC)
    W = H // 128
    assert f.shape == (128, HC, W, O)
    for p, n, j, o in ((0, 0, 0, 0), (5, 2, 31, 17), (127, 3, 7, 23)):
        assert f[p, n, j, o] == fn[o, n * H + p * W + j]


@pytest.mark.parametrize("npg", [1, 2])
def test_pack_holds_every_arranged_tensor(npg):
    c = case(2, 3, npg)
    L = c["L"]
    t = KS.arrange(c["wcat"], L.conv_w, L.f_b, L.g_b, L.dt_bias, L.out, c["fn"], L.in_norm, L.A_log, L.o_norm,
                   c["base"], c["scale"], NV, npg, HC)
    blobs = KS.pack(t, H, NV, K, npg, HC)
    offs = KS.offsets(H, NV, K, npg, HC)
    for name, (shape, dt) in KS.shapes(H, NV, K, npg, HC).items():
        n = 1
        for d in shape:
            n *= d
        o = offs["o_" + name]
        assert torch.equal(blobs[dt][o:o + n].view(shape), t[name].to(dt).reshape(shape)), name


def test_partials_add_up_to_the_half():
    """npg 2: the programs' partials (each its own head's columns, rounded to bf16) add to kda_half's half within bf16
    roundings; npg 1's single partial is it exactly."""
    c = case(2, 5, 2)
    L = c["L"]
    from trnrt.kernels import mhc_pre as MP

    x, _, _ = MP.emulate(c["S"], c["fn"], c["base"], c["scale"], ITERS, HC_EPS, EPS)
    with torch.no_grad():
        y, conv, rec = KH.emulate(x, L.in_norm, c["wcat"], L.conv_w, L.conv_state.clone(), L.f_b, L.g_b, L.dt_bias,
                                  L.A_log, L.o_norm, L.out, L.rec_state.clone(), c["slots"], LB, EPS)
    _, _, _, P2, conv2, rec2 = emulated(c, 2, False)
    _, _, _, P1, _, _ = emulated(c, 1, False)
    assert P2.shape == (2, 2, H) and torch.equal(P1[0], y)
    assert torch.allclose(P2.float().sum(0), y.float(), rtol=2e-2, atol=4e-3)
    assert torch.equal(conv2, conv) and torch.equal(rec2, rec)


def test_the_mix_in_is_mhc_mix_of_the_summed_partials():
    c = case(2, 7, 2)
    post_p, comb_p, Yp = c["pend"]
    Sn, _, _, _, _, _ = emulated(c, 2, True)
    want = MM.emulate(post_p, comb_p, Yp.float().sum(0).to(bf), c["S"]).view(2, HC, H)
    assert torch.equal(Sn, want)


def _simulated(T: int, grid: int, mix: bool, lb: float = LB, seed: int | None = None, comb_scale: float = 1.0,
               c=None, iters: int = ITERS):
    c = case(T, 30 + 4 * T + grid if seed is None else seed, grid) if c is None else c
    L = c["L"]
    want = emulated(c, grid, mix)
    t = KS.arrange(c["wcat"], L.conv_w, L.f_b, L.g_b, L.dt_bias, L.out, c["fn"], L.in_norm, L.A_log, L.o_norm,
                   c["base"], c["scale"], NV, grid, HC)
    blobs = KS.pack(t, H, NV, K, grid, HC)
    pend = None
    if mix:
        post_p, comb_p, Yp = c["pend"]
        pend = (post_p, comb_p * comb_scale, Yp)
    got = KS.run(c["S"], blobs, L.conv_state.clone(), L.rec_state.clone(), c["slots"], H=H, nv=NV, hc=HC,
                 iters=iters, hc_eps=HC_EPS, rms_eps=EPS, lb=lb, pend=pend, simulate=True, grid=grid)
    return want, got


def _close(want, got, mix: bool):
    Sn, post, comb, Pp, conv, rec = want
    gSn, gpost, gcomb, gPp, gconv, grec = got
    if mix:
        assert torch.equal(gSn, Sn), (gSn.float() - Sn.float()).abs().max()
    assert torch.allclose(gpost, post, rtol=1e-4, atol=1e-6), (gpost - post).abs().max()
    assert torch.allclose(gcomb, comb, rtol=1e-4, atol=1e-6), (gcomb - comb).abs().max()
    d = (gPp.float() - Pp.float()).abs()
    assert torch.allclose(gPp.float(), Pp.float(), rtol=2e-2, atol=2e-3), d.max()
    assert d.mean() < 1e-2 * Pp.float().abs().mean(), (d.mean(), Pp.float().abs().mean())
    assert torch.allclose(gconv.float(), conv.float(), rtol=1e-2, atol=1e-3)
    # x is the kernel's own mhc_pre here, so a 1-ulp bf16 flip in it moves a row's whole state update: measured, one
    # to three written rows of T = 8 off by up to 2e-3, every other row ~1e-8, the summed relative error
    # <= 2e-5. So a summed relative bound (the control below breaks it by far) and a max bound for a gross error.
    dr = (grec - rec).abs()
    assert dr.sum() <= 1e-4 * rec.abs().sum(), (dr.sum() / rec.abs().sum(), dr.max())
    assert dr.max() <= 5e-3, dr.max()


def _rec_rel(want, got):
    return ((got[5] - want[5]).abs().sum() / want[5].abs().sum()).item()


@pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("T,grid,mix", [(1, 1, False), (2, 1, True), (1, 2, False), (1, 2, True), (5, 2, True),
                                        (8, 2, False), (8, 1, False), (8, 2, True)])
def test_the_kernel_in_the_simulator_is_the_emulation(T, grid, mix, monkeypatch):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    want, got = _simulated(T, grid, mix)
    _close(want, got, mix)


@pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
def test_the_simulator_comparison_can_fail(monkeypatch):
    """Positive controls: a wrong gate bound must move the partials and states; a wrong comb the mixed streams."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    want, got = _simulated(2, 2, True, lb=LB - 1.0, seed=77)
    assert not torch.allclose(got[3].float(), want[3].float(), rtol=2e-2, atol=2e-3)
    assert _rec_rel(want, got) > 1e-3, _rec_rel(want, got)
    want, got = _simulated(2, 2, True, seed=78, comb_scale=1.5)
    assert not torch.equal(got[0], want[0])


# -- x pinned: the emulation's KDA half on the kernel's own mhc_pre output ----------------------------------------

if NKI:
    import nki
    import nki.language as nl

    @nki.jit
    def _pre_only(S, post_p, comb_p, Yp, bb, bf, H: int, hc: int, iters: int, hc_eps: float, rms_eps: float,
                  mix_in: int, o_fnh: int, o_base: int, o_scale: int):
        """The shell's prologue alone (the same shell helpers, the same arguments): x [T, H] bf16, post, comb."""
        T = S.shape[0]
        W, P_ = H // 128, 128
        O_ = 2 * hc + hc * hc
        x_out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        post_out = nl.ndarray((T, hc), dtype=nl.float32, buffer=nl.shared_hbm)
        comb_out = nl.ndarray((T, hc * hc), dtype=nl.float32, buffer=nl.shared_hbm)
        S_out = nl.ndarray((T, hc * H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        fw = nl.ndarray((P_, hc, W, O_), dtype=nl.bfloat16, buffer=nl.sbuf)
        SH.dma_static(dst=fw, src=bb.ap(pattern=[[hc * W * O_, P_], [W * O_, hc], [O_, W], [1, O_]], offset=o_fnh))
        sh = nl.ndarray((P_, hc, T, W), dtype=nl.bfloat16, buffer=nl.sbuf)
        if mix_in == 0:
            SH.streams_h(S, sh, hc, T, H)
        else:
            SH.mix_in_h(sh, S, post_p, comb_p, Yp, S_out, hc, T, H, Yp.shape[0])
        xh = SH.mhc_pre_h(sh, fw, bf, o_base, o_scale, post_out, comb_out, hc, T, H, iters, hc_eps, rms_eps)
        SH.dma_static(dst=x_out.ap(pattern=[[W, P_], [H, T], [1, W]], offset=0), src=xh)
        return x_out, post_out, comb_out


def _kernel_x(c, grid: int, mix: bool, S=None):
    """The kernel's own x (and post, comb) for case c: _pre_only in the simulator at grid 1 on the shell's inputs."""
    import ml_dtypes
    import numpy as np

    L = c["L"]
    t = KS.arrange(c["wcat"], L.conv_w, L.f_b, L.g_b, L.dt_bias, L.out, c["fn"], L.in_norm, L.A_log, L.o_norm,
                   c["base"], c["scale"], NV, grid, HC)
    blobs = KS.pack(t, H, NV, K, grid, HC)
    offs = KS.offsets(H, NV, K, grid, HC)
    S = c["S"] if S is None else S
    T = S.shape[0]
    post_p, comb_p, Yp = c["pend"]

    def b16(x):
        return x.float().numpy().astype(ml_dtypes.bfloat16)

    x, post, comb = nki.simulate(_pre_only)(
        S=b16(S.reshape(T, HC * H)), post_p=post_p.float().numpy(), comb_p=comb_p.float().reshape(T, HC * HC).numpy(),
        Yp=b16(Yp), bb=b16(blobs[torch.bfloat16]), bf=blobs[torch.float32].numpy(), H=H, hc=HC, iters=ITERS,
        hc_eps=HC_EPS, rms_eps=EPS, mix_in=int(mix), o_fnh=offs["o_fnh"], o_base=offs["o_base"],
        o_scale=offs["o_scale"])
    f = lambda a: torch.from_numpy(np.asarray(a).astype(np.float32))  # noqa: E731
    return f(x).to(bf), f(post), f(comb).view(T, HC, HC)


def _pinned(c, x, grid: int, conv, rec, qkv=None):
    L = c["L"]
    with torch.no_grad():
        return KS.emulate_half(x, L.in_norm, c["wcat"], L.conv_w, conv, L.f_b, L.g_b, L.dt_bias, L.A_log, L.o_norm,
                               L.out, rec, c["slots"], LB, EPS, grid, qkv=qkv)


@pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("T,grid,mix", [(8, 2, False), (8, 1, False), (8, 2, True), (5, 2, True)])
def test_with_x_pinned_the_state_is_tight(T, grid, mix, monkeypatch):
    """The emulation's KDA half on the kernel's own x: the partials and the states then meet the half's tight bounds
    (rec rtol 1e-3 / atol 1e-4, as the emulation check above with x shared). That the prologue here is the shell's: its
    post and comb equal the shell's bit for bit."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    seed = 30 + 4 * T + grid
    want, got = _simulated(T, grid, mix, seed=seed)
    c = case(T, seed, grid)
    x, post, comb = _kernel_x(c, grid, mix)
    assert torch.equal(post, got[1]) and torch.equal(comb, got[2])
    Pp, conv, rec = _pinned(c, x, grid, c["L"].conv_state.clone(), c["L"].rec_state.clone())
    assert torch.allclose(got[5], rec, rtol=1e-3, atol=1e-4), (got[5] - rec).abs().max()
    assert torch.allclose(got[4].float(), conv.float(), rtol=1e-2, atol=1e-3)
    assert torch.allclose(got[3].float(), Pp.float(), rtol=2e-2, atol=2e-3), (got[3].float() - Pp.float()).abs().max()


def _bf16_ulps(a, b):
    ai = a.contiguous().view(torch.int16).to(torch.int32)
    bi = b.contiguous().view(torch.int16).to(torch.int32)
    return (ai - bi).abs()


def _seam(grid: int, steps: int = 12, skip_write_at: int | None = None, pin_qkv: bool = False):
    """Consecutive decode steps (new streams each step, the same 4 sequences, positions advancing), the kernel on its
    own pools and the emulation on its own, x pinned to the kernel's each step; with pin_qkv also the projection's q, k,
    v (the kernel's newest conv entry). Per step: the pin, the conv states' largest difference in bf16 ulps and the
    count of differing elements, and the state error per (sequence, head). skip_write_at: the step at which the
    kernel's write of sequence 2's states goes to row R (skipped): the control the checks must catch."""
    T = 4
    c = case(T, 91, grid)
    L = c["L"]
    R = L.rec_state.shape[0]
    slot = torch.tensor([1, 3, 5, 7])
    t = KS.arrange(c["wcat"], L.conv_w, L.f_b, L.g_b, L.dt_bias, L.out, c["fn"], L.in_norm, L.A_log, L.o_norm,
                   c["base"], c["scale"], NV, grid, HC)
    blobs = KS.pack(t, H, NV, K, grid, HC)
    g = torch.Generator().manual_seed(92)
    kconv, krec = L.conv_state.clone(), L.rec_state.clone()
    pconv, prec = kconv.clone(), krec.clone()
    rows = slot.tolist()
    out = []
    for s in range(steps):
        S = (torch.randn(T, HC, H, generator=g) * 0.5).to(bf)
        c["S"] = S
        c["slots"] = SH.slots(slot, torch.tensor([s, s + 3, s + 1, s + 9]), R)
        kslots = c["slots"].clone()
        if s == skip_write_at:
            kslots[1, 2] = R
        _, post, _, _, kconv, krec = KS.run(S, blobs, kconv, krec, kslots, H=H, nv=NV, hc=HC, iters=ITERS,
                                            hc_eps=HC_EPS, rms_eps=EPS, lb=LB, simulate=True, grid=grid)
        x, xpost, _ = _kernel_x(c, grid, False, S=S)
        qkv = kconv[slot, K - 2] if pin_qkv else None  # each row's newest conv entry: this step's projected q, k, v
        _, pconv, prec = _pinned(c, x, grid, pconv, prec, qkv=qkv)
        u = _bf16_ulps(kconv[rows], pconv[rows])
        rec = (krec[rows] - prec[rows]).abs().flatten(2).amax(-1)  # [seq, head]
        out.append(dict(pin=torch.equal(xpost, post), ulps=int(u.max()), ndiff=int((u > 0).sum()), rec=rec))
    return out


def _seam_x_ok(steps) -> list:
    """Chain 1 (x pinned): what failed, empty when clean. The pin held; the conv states differ by at most 1 bf16 ulp
    per element in at most 8 elements a step (the fused projection sums its k-tiles in another order than F.linear:
    its bf16 rounding flips now and then, and the recurrent state keeps such a difference); no compounding: every
    state error <= 5e-4 (one flip's measured effect <= 2.9e-4)."""
    bad = []
    for i, st in enumerate(steps):
        if not st["pin"]:
            bad.append(f"step {i}: the x pin broke")
        if st["ulps"] > 1 or st["ndiff"] > 8:
            bad.append(f"step {i}: conv differs by {st['ulps']} ulp in {st['ndiff']} elements")
        if float(st["rec"].max()) > 5e-4:
            bad.append(f"step {i}: state error {float(st['rec'].max()):.1e} above 5e-4")
    return bad


def _seam_qkv_ok(steps) -> list:
    """Chain 2 (x and the projection's q, k, v pinned): the conv states equal bit for bit, the state errors at the
    one-step level (<= 1e-6) at every step, with no growth (the last three steps no more than 4x the first three)."""
    bad = []
    for i, st in enumerate(steps):
        if not st["pin"]:
            bad.append(f"step {i}: the x pin broke")
        if st["ndiff"]:
            bad.append(f"step {i}: conv differs in {st['ndiff']} elements with q, k, v pinned")
        if float(st["rec"].max()) > 1e-6:
            bad.append(f"step {i}: state error {float(st['rec'].max()):.1e} above 1e-6")
    first = max(float(x["rec"].max()) for x in steps[:3])
    last = max(float(x["rec"].max()) for x in steps[-3:])
    if last > max(4 * first, 1e-7):
        bad.append(f"growth: {first:.1e} over the first three steps, {last:.1e} over the last three")
    return bad


@pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("grid", [2])
def test_twelve_steps_carry_the_state_without_growth(grid, monkeypatch):
    """The recurrent seam over 12 steps. Chain 1, x pinned: _seam_x_ok. Chain 2, x and q, k, v pinned (the projection's
    rounding taken out): _seam_qkv_ok, the seam at the tight level. Control: the kernel skipping one step's state write
    for one sequence must fail chain 2's checks."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    one = _seam(grid)
    two = _seam(grid, pin_qkv=True)
    print("chain 1 (x pinned) state error by step", [f"{float(x['rec'].max()):.1e}" for x in one])
    print("chain 2 (x, qkv pinned) state error by step", [f"{float(x['rec'].max()):.1e}" for x in two])
    assert not _seam_x_ok(one), _seam_x_ok(one)
    assert not _seam_qkv_ok(two), _seam_qkv_ok(two)
    control = _seam(grid, steps=8, skip_write_at=5, pin_qkv=True)
    assert _seam_qkv_ok(control), "a skipped state write passed the seam checks"


# -- the released prior: the checkpoint's own hc_attn_* ------------------------------------------------------------

CKPT = os.environ.get("TRNRT_TEST_GLM_CKPT", "")  # a local GLM-5.3-Flash checkpoint directory
HAVE_CKPT = bool(CKPT) and os.path.exists(os.path.join(CKPT, "model.safetensors.index.json"))


def _released_hc(layer: int = 0) -> dict:
    """One layer's hc_attn_fn / _base / _scale from the local checkpoint (one tensor each, safetensors)."""
    import json

    from safetensors import safe_open

    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for n in ("fn", "base", "scale"):
        k = f"model.language_model.layers.{layer}.hc_attn_{n}"
        with safe_open(os.path.join(CKPT, idx[k]), "pt") as f:
            out[n] = f.get_tensor(k)
    return out


@pytest.mark.skipif(not NKI or not HAVE_CKPT, reason="needs the NKI package and the GLM-5.3-Flash checkpoint (the box)")
def test_the_prologue_under_the_released_prior(monkeypatch):
    """mix_in_h and mhc_pre_h under the checkpoint's own hc_attn_* of layer 0. Its comb base is a near-identity prior
    (0 on the diagonal, -8 off), under which 20 Sinkhorn rounds are a truncation, not a fixed point (with a
    flat base they are exact, and a wrong round count passes unseen). S_n bit-exact, post and comb within 1e-4 of the
    emulation; and the kernel at 21 rounds must miss comb at that bound: the check sees the truncation."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    hc = _released_hc(0)
    base = hc["base"].float()
    cb = base[2 * HC:].view(HC, HC)
    print("released comb base: diagonal", cb.diag().tolist(), "off-diagonal range",
          (cb - torch.diag(cb.diag())).min().item(), (cb + torch.eye(HC) * 1e9).min().item())
    c = case(2, 101, 2)
    c["fn"], c["base"], c["scale"] = hc["fn"].to(bf), base, hc["scale"].float()
    want, got = _simulated(2, 2, True, c=c)
    _close(want, got, True)
    _, got21 = _simulated(2, 2, True, c=c, iters=ITERS + 1)
    assert not torch.allclose(got21[2], want[2], rtol=1e-4, atol=1e-6), (got21[2] - want[2]).abs().max()


@pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("T,grid,mix", [(1, 2, False), (5, 2, True), (8, 2, False), (8, 1, True)])
def test_the_row_batched_recurrence_is_the_pair_pipeline_bit_for_bit(T, grid, mix, monkeypatch):
    """recb 1 (rows batched, stage-major: served from BATCH_MIN_T rows) against recb 0 (the pair pipeline, served below)
    on the same inputs: the
    same fp32 operations per element in another order of instructions, so every output equal bit for bit. The
    emulation comparison has rounding slack a wrong pair index could hide in; this has none."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    c = case(T, 60 + T + grid, grid)
    L = c["L"]
    t = KS.arrange(c["wcat"], L.conv_w, L.f_b, L.g_b, L.dt_bias, L.out, c["fn"], L.in_norm, L.A_log, L.o_norm,
                   c["base"], c["scale"], NV, grid, HC)
    blobs = KS.pack(t, H, NV, K, grid, HC)
    outs = {}
    for recb in (0, 1):
        outs[recb] = KS.run(c["S"], blobs, L.conv_state.clone(), L.rec_state.clone(), c["slots"], H=H, nv=NV, hc=HC,
                            iters=ITERS, hc_eps=HC_EPS, rms_eps=EPS, lb=LB, pend=c["pend"] if mix else None,
                            simulate=True, grid=grid, recb=recb)
    for name, a, b in zip(("S", "post", "comb", "P", "conv", "rec"), outs[0], outs[1]):
        if a is None:
            continue
        assert torch.equal(a, b), (name, (a.float() - b.float()).abs().max())


@pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("T,grid,mix", [(1, 2, False), (5, 2, True), (8, 2, False), (8, 1, True), (16, 2, False)])
def test_the_batched_xn_is_the_per_row_xn_bit_for_bit(T, grid, mix, monkeypatch):
    """The batched xn (two vector ops over [128, W, B], rx broadcast along W and in_norm along B; served at every T but
    XNB_SKIP_T) against the per-row loop (two ops a row; served at XNB_SKIP_T, where it is faster) on the same inputs:
    the same fp32 multiplies and bf16 roundings, so every output equal bit for bit."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    c = case(T, 70 + T + grid, grid)
    L = c["L"]
    t = KS.arrange(c["wcat"], L.conv_w, L.f_b, L.g_b, L.dt_bias, L.out, c["fn"], L.in_norm, L.A_log, L.o_norm,
                   c["base"], c["scale"], NV, grid, HC)
    blobs = KS.pack(t, H, NV, K, grid, HC)
    outs = {}
    for xnb in (False, True):
        monkeypatch.setattr(KS, "XNB", xnb)
        outs[xnb] = KS.run(c["S"], blobs, L.conv_state.clone(), L.rec_state.clone(), c["slots"], H=H, nv=NV, hc=HC,
                           iters=ITERS, hc_eps=HC_EPS, rms_eps=EPS, lb=LB, pend=c["pend"] if mix else None,
                           simulate=True, grid=grid)
    for name, a, b in zip(("S", "post", "comb", "P", "conv", "rec"), outs[False], outs[True]):
        if a is None:
            continue
        assert torch.equal(a, b), (name, (a.float() - b.float()).abs().max())
    assert outs[True][3].float().abs().sum() > 0  # (not vacuous: P carries the projection of xn)


@pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("T,grid,mix,pad", [(1, 2, False, False), (4, 2, True, False), (6, 2, False, True),
                                            (8, 2, False, False), (8, 1, True, True)])
def test_the_batched_conv_is_the_row_at_a_time_conv_bit_for_bit(T, grid, mix, pad, monkeypatch):
    """cbat 1 (every row's conv state read by one indirect DMA with a vector offset, written back by another, 3 hp
    transposes in all) against cbat 0 (a DMA and 3 hp transposes per row): the same data moved and transposed, so every
    output equal bit for bit, the conv pool included. pad: the last row reads slot -1 (a padded row): both forms skip
    it (oob_mode skip) and see zeros."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    c = case(T, 90 + T + grid, grid)
    L = c["L"]
    slots = c["slots"].clone()
    if pad:
        slots[0, -1] = -1
    t = KS.arrange(c["wcat"], L.conv_w, L.f_b, L.g_b, L.dt_bias, L.out, c["fn"], L.in_norm, L.A_log, L.o_norm,
                   c["base"], c["scale"], NV, grid, HC)
    blobs = KS.pack(t, H, NV, K, grid, HC)
    outs = {}
    for cbat in (0, 1):
        outs[cbat] = KS.run(c["S"], blobs, L.conv_state.clone(), L.rec_state.clone(), slots, H=H, nv=NV, hc=HC,
                            iters=ITERS, hc_eps=HC_EPS, rms_eps=EPS, lb=LB, pend=c["pend"] if mix else None,
                            simulate=True, grid=grid, cbat=cbat)
    for name, a, b in zip(("S", "post", "comb", "P", "conv", "rec"), outs[0], outs[1]):
        if a is None:
            continue
        assert torch.equal(a, b), (name, (a.float() - b.float()).abs().max())


@pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
def test_the_batched_conv_traces_fewer_transposes(monkeypatch):
    """The positive control: at T = 6 the row-at-a-time conv traces 2 x 6 x 3 hp transposes, the batched one 2 x 3 hp;
    equal traces would make the bit-equality above compare a form with itself."""
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    T, grid = 6, 2
    c = case(T, 7, grid)
    L = c["L"]
    t = KS.arrange(c["wcat"], L.conv_w, L.f_b, L.g_b, L.dt_bias, L.out, c["fn"], L.in_norm, L.A_log, L.o_norm,
                   c["base"], c["scale"], NV, grid, HC)
    blobs = KS.pack(t, H, NV, K, grid, HC)
    counts = {}
    for cbat in (0, 1):
        calls = []
        real = nki.isa.nc_matmul

        def spy(*a, **kw):
            calls.append(1)
            return real(*a, **kw)
        monkeypatch.setattr(nki.isa, "nc_matmul", spy)
        KS.run(c["S"], blobs, L.conv_state.clone(), L.rec_state.clone(), c["slots"], H=H, nv=NV, hc=HC, iters=ITERS,
               hc_eps=HC_EPS, rms_eps=EPS, lb=LB, simulate=True, grid=grid, cbat=cbat)
        monkeypatch.setattr(nki.isa, "nc_matmul", real)
        counts[cbat] = len(calls)
    hp = NV // grid
    print(f"\n  T={T}: nc_matmul calls traced row-at-a-time {counts[0]}, batched {counts[1]} (hp {hp})")
    assert counts[0] - counts[1] == 2 * (T - 1) * 3 * hp * grid or counts[0] - counts[1] == 2 * (T - 1) * 3 * hp, counts


def test_the_batched_forms_apply_from_batch_min_t():
    """The row-batched recurrence and the batched conv from BATCH_MIN_T rows on, the per-row forms (0, 0) below (one
    and two rows are faster without the batching); a value given to run() wins whatever T."""
    assert (KS.RECB, KS.CBAT, KS.BATCH_MIN_T) == (1, 1, 3)
    assert [KS.forms(t) for t in (1, 2, 3, 4, 6, 8)] == [(0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1)]
    assert KS.forms(2, recb=1) == (1, 0) and KS.forms(8, cbat=0) == (1, 0)


def test_the_per_row_xn_is_kept_at_the_skip_row_count():
    """The batched xn costs more than the per-row form at XNB_SKIP_T rows: the kernel's guards keep the per-row loop
    there, in the hashed source (so a NEFF of one form is never served for the other)."""
    assert KS.XNB is True and KS.XNB_SKIP_T == 6
    src = open(KS.__file__).read()
    span = src[src.index("try:  # the Neuron venv"):src.index("    kda_shell_kernel = None")]
    assert span.count("if XNB and B != XNB_SKIP_T:") == 1
    assert span.count("for b in range(0 if XNB and B != XNB_SKIP_T else B):") == 1



# -- the wiring (models/hybrid.py init_layer / load_layer, kernels/kda_shell.py register / fill / decode) -------------

def _cfg():
    hy = HY.HybridSpec("glm5_next", hc=HC, sinkhorn_iters=ITERS, hc_eps=HC_EPS, swiglu_limit=7.0)
    return SimpleNamespace(hybrid=hy, hidden_size=H, rms_norm_eps=EPS)


def _layer(monkeypatch, npg: int = 2, seed: int = 1):
    """The stand-in KDA layer as an nn.Module, then hybrid.init_layer at npg programs."""
    monkeypatch.setattr(HY, "LinearSpec", SimpleNamespace)  # the stand-in spec counts as a linear-attention spec
    monkeypatch.setattr(KH, "half_programs", lambda: npg)
    _, L, g = stand_in(seed, R=5)
    layer = nn.Module()
    for k, v in vars(L).items():
        setattr(layer, k, v)
    layer.spec.gate_rank = 128
    layer.pack_moe = False

    def p(*shape):
        return nn.Parameter(torch.empty(*shape, dtype=bf), requires_grad=False)

    HY.init_layer(layer, _cfg(), layer.spec, 0, 32, p)
    layer.hc_attn_fn.data.copy_((torch.randn(2 * HC + HC * HC, HC * H, generator=g) * 0.02).to(bf))
    layer.hc_attn_base.data.copy_(torch.randn(2 * HC + HC * HC, generator=g) * 0.5)
    layer.hc_attn_scale.data.copy_(torch.rand(3, generator=g) + 0.5)
    return layer, g


@pytest.mark.parametrize("npg", [1, 2])
def test_a_kda_layer_gets_two_packed_decode_only_blobs(monkeypatch, npg):
    layer, _ = _layer(monkeypatch, npg)
    assert layer.ks_dims == (H, NV, K, npg, HC)
    assert tuple(n for n in layer._decode_only if n.startswith("ks_")) == ("ks_blob_bf16", "ks_blob_f32")
    lay = SH.packed_layout(KS.shapes(H, NV, K, npg, HC))
    assert layer.ks_blob_bf16.numel() == lay[torch.bfloat16][0] and layer.ks_blob_f32.numel() == lay[torch.float32][0]


@pytest.mark.parametrize("npg", [1, 2])
def test_fill_packs_arrange_of_the_loaded_weights(monkeypatch, npg):
    layer, _ = _layer(monkeypatch, npg)
    KS.fill(layer)
    wcat = torch.cat([layer.in_qkv, layer.f_a, layer.g_a, layer.in_b])
    t = KS.arrange(wcat, layer.conv_w, layer.f_b, layer.g_b, layer.dt_bias, layer.out, layer.hc_attn_fn.data,
                   layer.in_norm, layer.A_log, layer.o_norm, layer.hc_attn_base.data, layer.hc_attn_scale.data, NV, npg,
                   HC)
    for name, v in SH.packed_views(layer, "ks_", KS.shapes(*layer.ks_dims)).items():
        assert torch.equal(v, t[name].to(v.dtype).reshape(v.shape)), name


@pytest.mark.parametrize("npg", [1, 2])
def test_the_shell_composition_is_the_served_attention_block(monkeypatch, npg):
    """hybrid._block's attention block (on the host: _mhc, rms_norm, the KDA half as kda_half's emulation, mix_out)
    against the shell's (S, post, comb, partials) mixed back by mix_out, as hybrid.layer does."""
    layer, g = _layer(monkeypatch, seed=3)
    T, R = 3, layer.rec_state.shape[0]
    model = SimpleNamespace(cfg=_cfg())
    streams = (torch.randn(T, HC * H, generator=g) * 0.5).to(bf)
    positions = torch.tensor([0, 9, 4])
    slots = SH.slots(torch.tensor([1, 3, 2]), positions, R)
    wcat = torch.cat([layer.in_qkv, layer.f_a, layer.g_a, layer.in_b])

    def half(xn):
        return KH.emulate(xn, layer.in_norm, wcat, layer.conv_w, layer.conv_state.clone(), layer.f_b, layer.g_b,
                          layer.dt_bias, layer.A_log, layer.o_norm, layer.out, layer.rec_state.clone(), slots, LB, EPS,
                          norm=False)[0]

    with torch.no_grad():
        served = HY._block(model, layer, "attn", streams, layer.in_norm, half, False)
        S, post, comb, Pp, _, _ = KS.emulate(streams.view(T, HC, H), layer.hc_attn_fn, layer.hc_attn_base,
                                             layer.hc_attn_scale, ITERS, HC_EPS, EPS, layer.in_norm, wcat, layer.conv_w,
                                             layer.conv_state.clone(), layer.f_b, layer.g_b, layer.dt_bias, layer.A_log,
                                             layer.o_norm, layer.out, layer.rec_state.clone(), slots, LB, npg)
        shelled = HY.mix_out(post, comb, Pp.float().sum(0).to(bf), S, True)
    if npg == 1:
        assert torch.equal(shelled, served)
    else:  # each program's partial rounded to bf16 before the sum: one more rounding than the served half
        d = (shelled.float() - served.float()).abs()
        assert torch.allclose(shelled.float(), served.float(), rtol=2e-2, atol=4e-3), d.max()
        assert not torch.equal(shelled, served)  # the extra rounding is really there (or this test compares nothing)


def test_decode_passes_the_layer_through(monkeypatch):
    """decode(): the slots from state_slot and positions (position 0 reads the zero row, the pools' last), the layer's
    blobs and dims, the model's hyper-connection constants, the grid the blobs were packed for, and the partials
    all-reduced as one call."""
    layer, _ = _layer(monkeypatch, 2)
    seen = {}

    def fake_run(S, blobs, conv, rec, slots, **kw):
        seen.update(kw, slots=slots, blobs=blobs)
        T = S.shape[0]
        return None, torch.zeros(T, HC), torch.zeros(T, HC, HC), torch.ones(2, T, H, dtype=bf), conv, rec

    monkeypatch.setattr(KS, "run", fake_run)
    reduced = []
    model = SimpleNamespace(cfg=_cfg(), _attn_all_reduce=lambda x: reduced.append(x) or x * 2)
    S = torch.zeros(2, HC, H, dtype=bf)
    out, (post, comb, Y) = KS.decode(model, layer, S, torch.tensor([0, 5]), torch.tensor([2, 4]))
    R = layer.rec_state.shape[0]
    assert out is S and seen["slots"].tolist() == [[R - 1, 4], [2, 4]]
    assert seen["grid"] == 2 and seen["iters"] == ITERS and seen["hc_eps"] == HC_EPS and seen["lb"] == LB
    assert seen["blobs"][torch.bfloat16] is layer.ks_blob_bf16 and seen["blobs"][torch.float32] is layer.ks_blob_f32
    assert len(reduced) == 1 and reduced[0].shape == (2, 2, H) and torch.equal(Y, reduced[0] * 2)


def test_the_shell_takes_decode_steps_of_up_to_eight_rows():
    """The served decode batch cap is 8: every compiled decode bucket (1, 2, 4, 6, 8) takes the shell."""
    assert all(HY.kda_shell_rows(r) for r in (1, 2, 4, 6, 8))
    assert not HY.kda_shell_rows(9)
