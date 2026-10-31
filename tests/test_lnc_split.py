"""The trn2 work split of the MoE and delta-rule NKI kernels in the NKI simulator: each kernel launched with grid 2, as
on trn2 at LNC=2 (trnrt/platform.py nki_grid), where the two programs are the two physical cores of the logical core and
each runs half of the work (nki.simulate(kernel[2]) runs both programs, their core barrier and sendrecv). Grid 2 must
give what grid 1 gives: the same bits for the kernels whose programs write disjoint outputs (moe_prefill, delta_rule),
and for the decode MoE kernels (each program sums its blocks, then the two fp32 partials are added) the same within
the emulation's tolerance and, with the same inputs, the same bits on every run."""

from __future__ import annotations

import importlib.util

import pytest
import torch

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")


def test_these_kernels_split_their_work():
    from trnrt import platform

    assert all(platform.lnc_split(k) for k in ("delta_rule", "moe_dedupe", "moe_prefill"))


def _sim(kernel, grid, **args):
    import nki

    return nki.simulate(kernel[grid] if grid > 1 else kernel)(**args)


@needs_nki
def test_moe_prefill_halves_are_the_whole_kernel(monkeypatch):
    """Lane tiles and combine token tiles split between the programs, Y in shared HBM behind a core
    barrier: the output equals grid 1's bit for bit (GLM-5.3-Flash's loaded expert layout, both block
    sizes)."""
    from trnrt.kernels import moe_dedupe as mdd
    from trnrt.kernels import moe_prefill as mp
    from tests.test_moe_prefill import checkpoint_experts, routing

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    w_gu, s_gu, w_down, s_down = checkpoint_experts()[0]
    blob = mdd.pack(w_gu, s_gu, w_down, s_down)
    down = mp.down_factors(blob, 1024)
    x, topv, topi = routing(256, skew=True)
    # skp 4: the tiles past the first half in device-loop segments, each program its half of every segment
    for B, skp in ((64, 0), (128, 0), (64, 4)):
        args = mp.kernel_inputs(x, topv, topi, blob, 1, 10.0, B=B, dq=False, down=down, skp=skp)
        one = torch.as_tensor(_sim(mp.kernel(), 1, **args))
        two = torch.as_tensor(_sim(mp.kernel(), 2, **args))
        assert torch.equal(one, two), (B, skp, (one != two).sum().item())
        want = mp.emulate(x, topv, topi, blob, 1, 10.0).float()
        assert (two.float() - want).abs().max() <= 0.01 * want.abs().max()


@needs_nki
def test_moe_dedupe_halves_sum_to_the_whole_kernel(monkeypatch):
    """Blocks of lanes split between the programs and the fp32 partials exchanged by halves of H: within
    the emulation's tolerance of grid 1, the same bits on a second run, and grid 1 itself unchanged."""
    from trnrt.kernels import moe_dedupe as mdd
    from tests.test_moe_dedupe import experts128

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    Hd = 1024
    ws = experts128(e=128, h=Hd)
    blob = mdd.pack(*ws)
    g = torch.Generator().manual_seed(4)
    for T in (16, 32):
        x = torch.randn(T, Hd, generator=g).bfloat16()
        topi = torch.stack([torch.randperm(128, generator=g)[:8] for _ in range(T)])
        topv = (torch.rand(T, 8, generator=g) + 0.1).bfloat16()
        args = mdd.kernel_inputs(x, topv, topi, blob, lanes=2, act=1, limit=10.0)
        S, L, BL = args["slots"], args["lanes"], args["block"]
        assert S * L // BL >= 2, "the case must have two blocks to split"
        one = torch.as_tensor(_sim(mdd.kernel(), 1, **args)).float()
        two = torch.as_tensor(_sim(mdd.kernel(), 2, **args)).float()
        again = torch.as_tensor(_sim(mdd.kernel(), 2, **args)).float()
        want = mdd.emulate(x, topv, topi, blob, 1, 10.0).float()
        assert torch.equal(two, again)
        assert (one - want).abs().max() <= 0.01 * want.abs().max()
        assert (two - want).abs().max() <= 0.01 * want.abs().max()
        # bf16 outputs of fp32 sums taken in another order: at most one bf16 step apart, plus an fp32-sized
        # absolute slack for outputs that are near zero by cancellation
        tol = one.abs() * 2.0 ** -7 + one.abs().max() * 2.0 ** -20
        assert ((two - one).abs() <= tol).all(), (two - one).abs().max()


@needs_nki
def test_moe_pairs_halves_sum_to_the_whole_kernel(monkeypatch):
    """The decode pairs kernel (calls under PAIRS_BELOW tokens) with its pair groups
    split between the programs and the fp32 partials exchanged by halves of the columns: within the emulation's
    tolerance of grid 1, one bf16 step of grid 1, the same bits on a second run; spl 0 at grid 2 is grid 1 exactly."""
    from trnrt.kernels import moe_dedupe as mdd
    from tests.test_moe_dedupe import experts128

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    Hd = 1024
    ws = experts128(e=128, h=Hd)
    blob = mdd.pack(*ws)
    g = torch.Generator().manual_seed(5)
    for T in (1, 2, 8):
        x = torch.randn(T, Hd, generator=g).bfloat16()
        topi = torch.stack([torch.randperm(128, generator=g)[:8] for _ in range(T)])
        topv = (torch.rand(T, 8, generator=g) + 0.1).bfloat16()
        args = mdd.pairs_inputs(x, topv, topi, blob, act=1, limit=10.0)
        assert args["spl"] == 1
        run = lambda grid, **kw: mdd.from_pairs(torch.as_tensor(_sim(mdd.pairs_kernel(), grid, **dict(args, **kw))),
                                                torch.float32)  # noqa: E731
        one, two, again = run(1), run(2), run(2)
        want = mdd.emulate_pairs(x, topv, topi, blob, 1, 10.0).float()
        assert torch.equal(two, again), T
        assert (one - want).abs().max() <= 0.01 * want.abs().max(), T
        assert (two - want).abs().max() <= 0.01 * want.abs().max(), T
        tol = one.abs() * 2.0 ** -7 + one.abs().max() * 2.0 ** -20  # as the dedupe kernel's split, above
        assert ((two - one).abs() <= tol).all(), (T, (two - one).abs().max())
        assert torch.equal(run(2, spl=0), one), T  # unsplit at grid 2: the old instruction stream


@needs_nki
def test_delta_rule_heads_split_is_the_whole_kernel(monkeypatch):
    """Each program runs its own heads: o and the final states equal grid 1's bit for bit (two and four heads a
    rank, at the model's unit group and the default)."""
    from trnrt.kernels import delta_rule as dr
    from tests.test_kda import inputs

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    for H, ug in ((2, 6), (4, 6), (4, None)):
        q, k, v, gate, beta, S0 = inputs(256, H, seed=3, correlated=True)
        args = dr.kernel_inputs(q, k, v, gate, beta, S0, device=torch.device("cpu"), ug=ug)
        o1, S1 = (torch.as_tensor(t) for t in _sim(dr.kernel(), 1, **args))
        o2, S2 = (torch.as_tensor(t) for t in _sim(dr.kernel(), 2, **args))
        assert torch.equal(o1, o2) and torch.equal(S1, S2), (H, ug)
