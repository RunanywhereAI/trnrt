"""TSB, the prefill MoE's token scatter built in SBUF (kernels/moe_prefill_tsb.py), the served prefill MoE kernel: its own
REV beside moe_prefill's and MDR's, MDR's inputs, the dispatch takes it from MOE_PREFILL_MIN_TOKENS rows, the trn2 parser
takes the kernel, and under the NKI simulator its output is MDR's bit for bit (ts is the scatter's, exactly), on uniform
and skewed routing, at both block sizes and both grids."""

import importlib.util
import types

import pytest
import torch

from tests.test_moe_prefill import checkpoint_experts, routing
from trnrt.kernels import moe_dedupe as mdd
from trnrt.kernels import moe_prefill as mp
from trnrt.kernels import moe_prefill_mdr as md
from trnrt.kernels import moe_prefill_tsb as tb

NKI = importlib.util.find_spec("nki") is not None


def test_own_rev():
    assert len({tb.REV, md.REV, mp.REV}) == 3


def _args(mod, T=256, B=64, skew=True):
    blob = mdd.pack(*checkpoint_experts()[0])
    down = mp.down_factors(blob, 1024)
    x, topv, topi = routing(T, skew=skew)
    return mod.kernel_inputs(x, topv, topi, blob, 1, 0.05, B=B, down=(down[0], md.fused_drain_table(*down, 1024)))


def test_inputs_are_mdrs_with_its_own_rev():
    a, b = _args(tb), _args(md)
    assert a["rev"] == tb.REV and b["rev"] == md.REV and set(a) == set(b)
    for k in set(a) - {"rev"}:
        va, vb = a[k], b[k]
        assert (torch.equal(va, vb) if isinstance(va, torch.Tensor) else va == vb), k


def test_the_dispatch_takes_tsb_from_min_tokens(monkeypatch):
    """hybrid._moe_clamped calls moe_prefill_tsb with the layer's fused table from MOE_PREFILL_MIN_TOKENS rows on and
    moe_dedupe below (stand-in kernels: the routing only)."""
    from trnrt.kernels import moe_dedupe
    from trnrt.models import decoder, hybrid

    calls = []
    monkeypatch.setattr(decoder, "MOE_PREFILL_MIN_TOKENS", 4)
    monkeypatch.setattr(mp, "moe_prefill", lambda x, *a, **k: calls.append("plain") or x)
    monkeypatch.setattr(md, "moe_prefill_mdr", lambda x, *a, **k: calls.append("mdr") or x)
    monkeypatch.setattr(tb, "moe_prefill_tsb", lambda x, *a, **k: calls.append(("tsb", k.get("down"))) or x)
    monkeypatch.setattr(moe_dedupe, "moe_dedupe", lambda x, *a, **k: calls.append("dedupe") or x)
    model = types.SimpleNamespace(cfg=types.SimpleNamespace(num_experts_per_tok=2), moe_inter=4,
                                  _route=lambda layer, x: (torch.ones(x.shape[0], 2),
                                                           torch.zeros(x.shape[0], 2, dtype=torch.long)))
    layer = types.SimpleNamespace(moe_blob=True, moe_tiles=True, w_blob=None, moe_prefill_dq=False,
                                  moe_prefill_dsc="s", moe_prefill_dfr="t")
    monkeypatch.setattr(torch.Tensor, "device", property(lambda self: torch.device("meta")), raising=False)
    hybrid._moe_clamped(model, layer, torch.zeros(4, 8), 10.0)
    hybrid._moe_clamped(model, layer, torch.zeros(3, 8), 10.0)
    assert calls == [("tsb", ("s", "t")), "dedupe"]


def _sim(kernel, a, grid: int):
    import nki

    k = kernel[grid] if grid > 1 else kernel
    return torch.as_tensor(nki.simulate(k)(**a))


@pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("skew", [True, False])
@pytest.mark.parametrize("grid", [1, 2])
@pytest.mark.parametrize("B", [128, 64])
def test_nki_simulator_tsb_is_mdr_bit_for_bit(B, grid, skew, monkeypatch):
    """TSB's output against MDR's on GLM's loaded scales (the per-column path with the fused drain): equal bit for bit,
    since ts is the scatter's exactly and the rest of the kernel is MDR's."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    o_md = _sim(md.kernel(), _args(md, B=B, skew=skew), grid)
    o_tb = _sim(tb.kernel(), _args(tb, B=B, skew=skew), grid)
    assert o_md.float().abs().max() > 0
    assert torch.equal(o_tb.view(torch.int16), o_md.view(torch.int16)), int((o_tb != o_md).sum())


@pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
def test_trn2_parser_takes_the_kernel(monkeypatch):
    import ml_dtypes
    from nki.compiler.frontend import FrontendKind, source_to_mlir

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    a = _args(tb)

    def np_(v):
        if not isinstance(v, torch.Tensor):
            return v
        return v.float().numpy().astype(ml_dtypes.bfloat16) if v.dtype == torch.bfloat16 else v.numpy()
    assert str(source_to_mlir(tb.kernel(), **{k: np_(v) for k, v in a.items()}, target="trn2", frontend=FrontendKind.PARSER))
