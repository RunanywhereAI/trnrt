"""The fused DSA layer's send rows (kernels/dsa_sat.py _snd_write_hbm): one contiguous run of rows written HBM -> HBM in
two DMAs with 3-D access patterns. In the NKI simulator: every row of the run gets o[0, t, h, :] in columns 0 .. R - 1 of
row h T + t and lse[0, t, h, 0] in column R, nothing else is written, and a planted wider run differs."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from trnrt.kernels import dsa_sat as SAT

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
POISON = -7777.0

if SAT.nki is not None:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def _snd_kernel(o, lse, poison, r0: int, r1: int):
        """SND := poison, then rows r0 .. r1 - 1 by _snd_write_hbm; SND returned."""
        _, T, H, R = o.shape
        SND = nl.ndarray((H * T, R + 16), dtype=nl.float32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=SND, src=poison)
        rows = []
        for t in range(r0, r1):
            rows.append(t)  # noqa: PERF402 (kernel code: the NKI tracer refuses list builtins)
        SAT._snd_write_hbm(o, lse, SND, T, H, R, rows)
        return SND


def _case(T: int, H: int, R: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(1, T, H, R, generator=g)
    lse = torch.randn(1, T, H, 1, generator=g) * 4
    return o, lse, torch.full((H * T, R + 16), POISON)


def _ref(o, lse, p, r0: int, r1: int):
    """The send buffer the rows r0 .. r1 - 1 leave: row h T + t holds o[0, t, h] then lse[0, t, h]."""
    _, T, H, R = o.shape
    out = p.clone().view(H, T, R + 16)
    out[:, r0:r1, :R] = o[0, r0:r1].permute(1, 0, 2)
    out[:, r0:r1, R] = lse[0, r0:r1, :, 0].t()
    return out.view(H * T, R + 16)


def _sim(o, lse, p, r0, r1):
    return torch.as_tensor(nki.simulate(_snd_kernel)(o.numpy(), lse.numpy(), p.numpy(), r0=r0, r1=r1))


def test_reference_writes_the_run_only():
    o, lse, p = _case(4, 2, 8, seed=1)
    ref = _ref(o, lse, p, 1, 3)
    assert int((ref != POISON).sum()) == 2 * 2 * 9
    assert torch.equal(ref[1 * 4 + 2, :8], o[0, 2, 1]) and ref[0 * 4 + 1, 8] == lse[0, 1, 0, 0]


@needs_nki
@pytest.mark.parametrize(("r0", "r1"), [(0, 24), (0, 12), (12, 24), (5, 6), (0, 0)])
def test_the_run_is_written_in_place(monkeypatch, r0, r1):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    o, lse, p = _case(24, 16, 512, seed=900 + r0 * 31 + r1)
    got = _sim(o, lse, p, r0, r1)
    assert torch.equal(got.view(torch.int32), _ref(o, lse, p, r0, r1).view(torch.int32))


@needs_nki
def test_planted_wider_run_differs(monkeypatch):
    """The positive control: the writer over one extra row is not the shorter run's send buffer."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    o, lse, p = _case(24, 16, 512, seed=77)
    assert not torch.equal(_sim(o, lse, p, 0, 13).view(torch.int32), _ref(o, lse, p, 0, 12).view(torch.int32))
