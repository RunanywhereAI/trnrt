"""kernels/mhc_seam.py, one mHC seam (the previous block's output mix and this layer's attention pre) as one kernel, and
its wiring (models/hybrid.py MHC_SEAM, EngineConfig.mhc_seam).

Host: fn_w's layout; emulate() is the separate pair's host forms (mix_out's arithmetic, then mhc_pre.emulate); seam() on
host tensors is emulate(); with MHC_SEAM a layer gets the decode-only fn_w copy at init and load, and without it none;
_seam_takes' conditions (an FFN-half decode call, not the fused DSA layer's rows, an even T); defers() leaves the FFN mix
to a next layer that takes the seam; layer() runs the seam in place of mix_out + the attention pre (the route:
tests/test_kda_routes.py).
Simulator (Neuron venv): the kernel against emulate() at grid 1 and split over the LNC pair, S' bit for bit and the
pre within fp32 summation-order error, and each planted error caught."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trnrt.kernels import mhc_pre, mhc_seam
from trnrt.models import hybrid as HY

NKI = importlib.util.find_spec("nki") is not None
needs_nki = pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
ITERS, EPS, RMS_EPS = 20, 1e-6, 1e-5


def _inputs(T=6, hc=4, H=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(T, hc * H, generator=g).to(torch.bfloat16)
    y = (0.5 * torch.randn(T, H, generator=g)).to(torch.bfloat16)
    post = torch.rand(T, hc, generator=g) * 1.9 + 0.05
    comb = (torch.eye(hc) * 0.9 + 0.1 * torch.rand(T, hc, hc, generator=g)).reshape(T, hc * hc)
    n_mix = 2 * hc + hc * hc
    fn = (0.02 * torch.randn(n_mix, hc * H, generator=g)).to(torch.bfloat16)
    base = torch.randn(n_mix, generator=g) * 0.1
    base[2 * hc :] = (torch.full((hc, hc), -8.0) + 8.0 * torch.eye(hc)).reshape(-1)  # the released comb prior's form
    scale = torch.tensor([1.0, 0.5, 2.0])
    return S, y, post, comb, fn, base, scale


def test_fn_w_layout():
    hc, H = 4, 4096
    fn = torch.randn(24, hc * H).to(torch.bfloat16)
    w = mhc_seam.fn_w(fn, hc)
    assert tuple(w.shape) == (128, hc * H // 128, 24)
    for p, n, j, o in ((0, 0, 0, 0), (5, 2, 7, 11), (127, 3, 31, 23)):
        assert float(w[p, n * 32 + j, o]) == float(fn[o, n * H + p * 32 + j])


def test_emulate_is_the_separate_pair_on_normals():
    """S' = the mix with one cast (equal to mix_out's barrier form on normals); (x, post, comb) = mhc_pre.emulate on S'."""
    S, y, post, comb, fn, base, scale = _inputs()
    T, hc, H = S.shape[0], 4, 256
    S2, x, p2, c2 = mhc_seam.emulate(S, y, post, comb, fn, base, scale, ITERS, EPS, RMS_EPS)
    want = HY.mix_out(post, comb.view(T, hc, hc), y, S.view(T, hc, H))
    assert torch.equal(S2, want)
    xe, pe, ce = mhc_pre.emulate(S2.view(T, hc, H), fn, base, scale, ITERS, EPS, RMS_EPS)
    assert torch.equal(x, xe) and torch.equal(p2, pe) and torch.equal(c2, ce.reshape(T, hc * hc))


def test_host_seam_is_emulate():
    S, y, post, comb, fn, base, scale = _inputs()
    T, hc, H = S.shape[0], 4, 256
    got = mhc_seam.seam(
        post, comb.view(T, hc, hc), y, S.view(T, hc, H), mhc_seam.fn_w(fn, hc), fn, base, scale, ITERS, EPS, RMS_EPS
    )
    want = mhc_seam.emulate(S, y, post, comb, fn, base, scale, ITERS, EPS, RMS_EPS)
    assert all(torch.equal(a, b) for a, b in zip(got[:3], want[:3], strict=True))
    assert torch.equal(got[3], want[3].view(T, hc, hc))


def test_takes_an_even_row_count_on_the_device():
    dev = torch.device("meta")
    if mhc_seam.mhc_seam_kernel is None:  # no nki here: the kernel is the first condition
        assert not mhc_seam.takes(dev, 8, 4096, 4)
        return
    assert [mhc_seam.takes(dev, T, 4096, 4) for T in (1, 2, 3, 8, 16, 24, 128, 130)] == [
        False,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    assert not mhc_seam.takes(torch.device("cpu"), 8, 4096, 4) and not mhc_seam.takes(dev, 8, 4096 + 64, 4)


# -- the wiring ---------------------------------------------------------------------------------------------------------


def _cfg(hc=4, H=256):
    hy = HY.HybridSpec("glm5_next", hc=hc, sinkhorn_iters=ITERS, hc_eps=EPS, swiglu_limit=7.0)
    return SimpleNamespace(hybrid=hy, hidden_size=H, rms_norm_eps=RMS_EPS)


def _layer(monkeypatch, on: bool, hc=4, H=256):
    """A non-KDA layer (no shell blobs) after hybrid.init_layer, MHC_SEAM on or off."""
    monkeypatch.setattr(HY, "MHC_SEAM", on)
    layer = nn.Module()

    def p(*shape):
        return nn.Parameter(torch.empty(*shape, dtype=torch.bfloat16), requires_grad=False)

    HY.init_layer(layer, _cfg(hc, H), SimpleNamespace(), 0, 32, p)
    return layer


def test_the_fn_w_copy_is_decode_only_and_only_with_the_seam(monkeypatch):
    off = _layer(monkeypatch, False)
    assert not hasattr(off, "hc_attn_fnw") and "hc_attn_fnw" not in off._decode_only
    on = _layer(monkeypatch, True)
    assert tuple(on.hc_attn_fnw.shape) == (128, 4 * 256 // 128, 24) and "hc_attn_fnw" in on._decode_only
    names = [n for n, _ in on.named_parameters()]
    assert names.index("hc_attn_fnw") == names.index("hc_ffn_fnj") + 1  # after the FFN half's copy

    _, _, _, _, fn, base, scale = _inputs()
    fnf = torch.randn(24, 4 * 256).to(torch.bfloat16)
    ck = {
        "L.hc_attn_fn": fn, "L.hc_attn_base": base, "L.hc_attn_scale": scale,
        "L.hc_ffn_fn": fnf, "L.hc_ffn_base": base, "L.hc_ffn_scale": scale,
    }  # fmt: skip
    HY.load_layer(on, SimpleNamespace(get=lambda k: ck[k]), "L.", _cfg(), 0, 1, torch.bfloat16)
    assert torch.equal(on.hc_attn_fnw.data, mhc_seam.fn_w(fn, 4))


def test_seam_takes_an_ffn_half_decode_layer_outside_the_fused_dsa_rows(monkeypatch):
    monkeypatch.setattr(HY, "MHC_SEAM", True)
    monkeypatch.setattr(mhc_seam, "takes", lambda device, T, H, hc: T % 2 == 0)
    monkeypatch.setattr(HY, "_mk4_rows_on", lambda rows: rows <= 24)
    model = SimpleNamespace(cfg=_cfg())
    kda = SimpleNamespace(hc_attn_fnw=object(), spec=SimpleNamespace())
    dsa = SimpleNamespace(hc_attn_fnw=object(), spec=SimpleNamespace(mla=object()))

    def takes(layer, T=8, fh=True, dec=True, sp=False):
        return HY._seam_takes(model, layer, torch.zeros(T, 4), dec, sp, fh)

    assert takes(kda) and takes(kda, T=24)
    assert not takes(kda, fh=False) and not takes(kda, dec=False) and not takes(kda, sp=True)
    assert not takes(kda, T=7)  # the kernel's shapes
    assert not takes(SimpleNamespace(spec=SimpleNamespace()))  # no fn_w copy
    assert not takes(dsa) and takes(dsa, T=26)  # the fused DSA layer's rows keep mhc_front
    monkeypatch.setattr(HY, "MHC_SEAM", False)
    assert not takes(kda)


def test_defers_to_a_next_layer_that_takes_the_seam(monkeypatch):
    seen = []
    monkeypatch.setattr(HY, "kda_shell_takes", lambda *a: False)
    monkeypatch.setattr(HY, "ffn_half_takes", lambda model, layer, h, dec, sp: layer.fh)
    monkeypatch.setattr(HY, "_seam_takes", lambda model, layer, h, dec, sp, fh: seen.append(fh) or fh)
    model = SimpleNamespace(tp_size=1, _sp_on=lambda: False)
    h, pos, table = torch.zeros(8, 4), torch.zeros(8), torch.zeros(8, 2)
    monkeypatch.setattr(HY, "MHC_SEAM", False)
    assert not HY.defers(model, None, SimpleNamespace(fh=True), h, pos, table, None, None) and seen == []
    monkeypatch.setattr(HY, "MHC_SEAM", True)
    assert HY.defers(model, None, SimpleNamespace(fh=True), h, pos, table, None, None)
    assert not HY.defers(model, None, SimpleNamespace(fh=False), h, pos, table, None, None)
    assert seen == [True, False]


def test_layer_runs_the_seam_in_place_of_the_mix_and_the_pre(monkeypatch):
    """layer() with a pend and a seam-taking FFN-half layer: one seam call, no mix_out of the pend, and _attn_unmixed
    gets the seam's (x, post, comb) as its pre; without the seam, mix_out and _mhc as before."""
    calls = []
    T, hc, H = 4, 4, 256
    model = SimpleNamespace(cfg=_cfg(hc, H), tp_size=1, _sp_on=lambda: False)
    layer = SimpleNamespace(spec=SimpleNamespace(), in_norm=torch.ones(H))
    h = torch.zeros(T, hc * H, dtype=torch.bfloat16)
    pend = ("post", "comb", "y")
    monkeypatch.setattr(HY, "ffn_half_takes", lambda *a: True)
    monkeypatch.setattr(HY, "kda_shell_takes", lambda *a: False)
    monkeypatch.setattr(HY, "_mk4_rows_on", lambda rows: False)
    monkeypatch.setattr(HY, "_seam_takes", lambda *a: True)
    monkeypatch.setattr(HY, "_seam", lambda m, lay, p, s: calls.append(("seam", p)) or (s + 1, ("x", "p2", "c2")))
    monkeypatch.setattr(HY, "mix_out", lambda *a: calls.append(("mix_out",)) or a[3].reshape(T, hc * H))
    monkeypatch.setattr(
        HY, "_attn_unmixed", lambda model, layer, streams, fn, dec, pre=None: calls.append(("attn", pre)) or (0, 0)
    )
    from trnrt.kernels import ffn_half

    monkeypatch.setattr(ffn_half, "decode", lambda *a, **k: (torch.zeros(T, hc, H), None, None, None))
    table = torch.zeros(T, 2)
    for seam_on in (True, False):
        calls.clear()
        monkeypatch.setattr(HY, "MHC_SEAM", seam_on)
        HY.layer(model, layer, h, torch.zeros(T), None, table, None, None, pend=pend, defer=True)
        if seam_on:
            assert calls == [("seam", pend), ("attn", ("x", "p2", "c2"))]
        else:
            assert calls == [("mix_out",), ("attn", None)]


# -- the kernel in the simulator -------------------------------------------------------------------------------------


def _sim(grid: int, T: int, plant: int = 0, H: int = 4096, seed: int = 3):
    import ml_dtypes
    import nki
    import numpy as np

    hc = 4
    S, y, post, comb, fn, base, scale = _inputs(T=T, hc=hc, H=H, seed=seed)

    def bf(t):
        return np.ascontiguousarray(t.float().numpy()).astype(ml_dtypes.bfloat16)

    k = mhc_seam.mhc_seam_kernel
    outs = nki.simulate(k[grid] if grid > 1 else k)(
        S=bf(S), y=bf(y), post=post.numpy(), comb=comb.numpy(), fnw=bf(mhc_seam.fn_w(fn, hc)), base=base.numpy(),
        scale=scale.numpy(), hc=hc, iters=ITERS, eps=EPS, rms_eps=RMS_EPS, plant=plant, rev=mhc_seam.REV,
    )  # fmt: skip
    got = [torch.from_numpy(np.asarray(o).astype(np.float32)) for o in outs]
    want = [t.float() for t in mhc_seam.emulate(S, y, post, comb, fn, base, scale, ITERS, EPS, RMS_EPS)]
    return got, want


def _close(got, want) -> bool:
    S2, x, p2, c2 = got
    return (
        torch.equal(S2, want[0])
        and torch.allclose(x, want[1], rtol=1.6e-2, atol=1e-2)
        and torch.allclose(p2, want[2], rtol=1e-4, atol=1e-5)
        and torch.allclose(c2, want[3], rtol=1e-3, atol=1e-5)
    )


@needs_nki
@pytest.mark.parametrize(("grid", "T"), [(1, 4), (2, 2), (2, 8), (2, 24)])
def test_the_kernel_in_the_simulator_is_the_emulation(grid, T, monkeypatch):
    """S' bit for bit (mix_wide's ops); the pre within the summation-order error of the h-layout projection."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    got, want = _sim(grid, T)
    assert _close(got, want), [float((a - b).abs().max()) for a, b in zip(got, want, strict=True)]


@needs_nki
@pytest.mark.parametrize("plant", [1, 2, 3, 4])
def test_each_planted_error_is_caught(plant, monkeypatch):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    got, want = _sim(2, 8, plant=plant)
    assert not _close(got, want)
