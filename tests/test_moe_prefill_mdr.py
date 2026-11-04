"""MDR, the fused stage-C drain (kernels/moe_prefill_mdr.py): split3 is exact in every summation order, the fused table
and its arithmetic equal the kernel's two-step drain bit for bit, the selector, the input guards, the load builds the
fused table, and under the NKI simulator (the Neuron venv) the fused kernel's output is moe_prefill's bit for bit at grid
1 and 2, while its positive control (pc 3) moves it on GLM's scales."""

import importlib.util
import os
import types

import pytest
import torch

from tests.test_moe_prefill import checkpoint_experts, routing
from trnrt.kernels import moe_dedupe as mdd
from trnrt.kernels import moe_prefill as mp
from trnrt.kernels import moe_prefill_mdr as md

NKI = importlib.util.find_spec("nki") is not None


def _rne_split3(t):  # round-to-nearest pieces: the split test_split3_is_exact must be able to reject
    hi = t.bfloat16().float()
    mid = (t - hi).bfloat16().float()
    return hi, mid, (t - hi - mid).bfloat16().float()


def _orders(hi, mid, lo):
    return [(hi + mid) + lo, (hi + lo) + mid, (mid + lo) + hi]


def test_own_rev():
    assert md.REV != mp.REV


def test_split3_is_exact():
    """split3's three bf16 pieces sum to t exactly in fp32 in every order, over random fp32 values across 40 binades of
    both signs and over GLM-5.3-Flash's loaded down scales; round-to-nearest pieces fail some order on the same values."""
    g = torch.Generator().manual_seed(0)
    t = torch.rand(400000, generator=g) * torch.exp2(torch.randint(-30, 10, (400000,), generator=g).float())
    t = torch.cat([t, -t[:1000], torch.tensor([1.0, 2.0**-20, 3.0, 0.0])])
    blob = mdd.pack(*checkpoint_experts()[0])
    t = torch.cat([t, mdd.unpack_down(blob, 1024)[1].float().flatten()])
    p = md.split3(t)
    assert p.dtype == torch.bfloat16 and tuple(p.shape) == (t.numel(), 3)
    hi, mid, lo = p.float().unbind(-1)
    for s in _orders(hi, mid, lo):
        assert torch.equal(s, t)
    assert not all(torch.equal(s, t) for s in _orders(*_rne_split3(t)))


def test_fused_drain_table_and_arithmetic():
    """fused_drain_table holds T = s f as three pieces (piece p of column h at p H + h) that sum to it exactly;
    bf16(fp32(y T)) equals the kernel's two-step drain bf16(fp32(y s)) f bit for bit (f a power of two); the control
    bf16(bf16(y) T) does not on GLM's scales."""
    blob = mdd.pack(*checkpoint_experts()[0])
    dsc, dfr = mp.down_factors(blob, 1024)
    d3 = md.fused_drain_table(dsc, dfr, 1024)
    assert d3.dtype == torch.bfloat16 and tuple(d3.shape) == (6, 3 * 1024)
    pc = d3.view(6, 3, 1024).float()
    T = (pc[:, 0] + pc[:, 1]) + pc[:, 2]
    assert torch.equal(T, mdd.unpack_down(blob, 1024)[1][:, 0, :].float())
    k = torch.arange(8)
    s = dsc[:, 2 * (k % 4) + k // 4].repeat_interleave(128, 1)
    y = torch.randn(6, 512, 1024, generator=torch.Generator().manual_seed(2)) * 40
    two = ((y * s[:, None]).bfloat16().float() * dfr.float()[:, None]).bfloat16()
    one = (y * T[:, None]).bfloat16()
    assert torch.equal(two, one)
    assert not torch.equal((y.bfloat16().float() * T[:, None]).bfloat16(), one)


@pytest.mark.parametrize("B", [64, 128])
def test_sel3_matrix(B):
    m = md.sel3_matrix(B)
    G = 128 // B
    assert m.dtype == torch.bfloat16 and tuple(m.shape) == (3 * G, 128)
    r, lane = torch.arange(3 * G).view(-1, 1), torch.arange(128).view(1, -1)
    assert torch.equal(m, (lane // B == r // 3).to(torch.bfloat16))


def test_inputs_guard_the_table():
    """The fused drain refuses the plain path (down None) and the plain column table, and with the fused table passes
    pc 2 (3 the control), sel3 in dsc's place, T as dfr and its own REV; the rest is moe_prefill.kernel_inputs'."""
    blob = mdd.pack(*checkpoint_experts()[0])
    down = mp.down_factors(blob, 1024)
    fused = (down[0], md.fused_drain_table(*down, 1024))
    x, topv, topi = routing(256, skew=True)
    with pytest.raises(ValueError, match="the fused drain needs the per-column path"):
        md.kernel_inputs(x, topv, topi, blob, 1, 0.05, B=64, down=None)
    with pytest.raises(ValueError, match="build it with fused_drain_table at load"):
        md.kernel_inputs(x, topv, topi, blob, 1, 0.05, B=64, down=down)
    a = md.kernel_inputs(x, topv, topi, blob, 1, 0.05, B=64, down=fused)
    b = mp.kernel_inputs(x, topv, topi, blob, 1, 0.05, B=64, down=down)
    assert a["pc"] == 2 and b["pc"] == 1 and a["rev"] == md.REV and b["rev"] == mp.REV
    assert torch.equal(a["dsc"], md.sel3_matrix(64)) and a["dfr"] is fused[1]
    assert set(a) == set(b)
    for k in set(a) - {"pc", "dsc", "dfr", "rev"}:
        va, vb = a[k], b[k]
        assert torch.equal(va, vb) if isinstance(va, torch.Tensor) else va == vb, k
    assert md.kernel_inputs(x, topv, topi, blob, 1, 0.05, B=64, down=fused, ctl=1)["pc"] == 3


def test_pack_experts_builds_the_fused_table():
    """DecoderLayer.pack_experts stores fused_drain_table's [E, 3 H] in place of the column factors (moe_prefill_down
    then hands it to the prefill kernel), the chunk scales as down_factors gives them."""
    from trnrt.models import decoder

    w = checkpoint_experts()[0]
    layer = types.SimpleNamespace(pack_tiles=True, w_gu=w[0], w_gu_scale=w[1], w_down=w[2], w_down_scale=w[3])
    decoder.DecoderLayer.pack_experts(layer)
    dsc, tab = decoder.moe_prefill_down(layer)
    down = mp.down_factors(mdd.pack(*w), 1024)
    assert torch.equal(dsc, down[0]) and torch.equal(tab, md.fused_drain_table(*down, 1024))


def _sim(kernel, a, grid: int):
    import nki

    k = kernel[grid] if grid > 1 else kernel
    return torch.as_tensor(nki.simulate(k)(**a))


@pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("grid", [1, 2])
@pytest.mark.parametrize("B", [128, 64])
def test_nki_simulator_fused_drain(B, grid, monkeypatch):
    """The fused drain leaves moe_prefill's output as it was, bit for bit, at grid 1 and at grid 2 (the served LNC
    split); on GLM's scales its positive control (pc 3) changes it, so the comparison can fail."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    blob = mdd.pack(*checkpoint_experts()[0])
    down = mp.down_factors(blob, 1024)
    fused = (down[0], md.fused_drain_table(*down, 1024))
    x, topv, topi = routing(256, skew=True)
    plain = _sim(mp.kernel(), mp.kernel_inputs(x, topv, topi, blob, 1, 0.05, B=B, down=down), grid)
    f = _sim(md.kernel(), md.kernel_inputs(x, topv, topi, blob, 1, 0.05, B=B, down=fused), grid)
    c = _sim(md.kernel(), md.kernel_inputs(x, topv, topi, blob, 1, 0.05, B=B, down=fused, ctl=1), grid)
    assert torch.equal(plain.view(torch.int16), f.view(torch.int16))
    assert not torch.equal(plain.view(torch.int16), c.view(torch.int16))


@pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("pc", [2, 3])
def test_trn2_parser_takes_the_kernel(pc, monkeypatch):
    """The trn2 parser frontend (the compile's first stage, stricter than nki.simulate) takes the fused kernel at the
    served shapes' form."""
    import ml_dtypes
    from nki.compiler.frontend import FrontendKind, source_to_mlir

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    blob = mdd.pack(*checkpoint_experts()[0])
    down = mp.down_factors(blob, 1024)
    x, topv, topi = routing(256, skew=True)
    a = md.kernel_inputs(
        x, topv, topi, blob, 1, 0.05, B=64, down=(down[0], md.fused_drain_table(*down, 1024)), ctl=pc - 2
    )

    def np_(v):
        if not isinstance(v, torch.Tensor):
            return v
        return v.float().numpy().astype(ml_dtypes.bfloat16) if v.dtype == torch.bfloat16 else v.numpy()

    kw = {k: np_(v) for k, v in a.items()}
    assert str(source_to_mlir(md.kernel(), **kw, target="trn2", frontend=FrontendKind.PARSER))


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
