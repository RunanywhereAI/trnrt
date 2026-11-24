"""The context-parallel prefill's failed-row merge under a run-time trip (models/mla.py CP_FAIL_TRIP,
models/dsa_long.py cp_merge's trip, kernels/dsa_topk_t.py).

On the host: the trip merge gives cp_merge_rows' mask bit for bit at trip 1 (ties, empty lists, fewer valid candidates
than keep) and no selection at trip 0; a merge keeping one fewer differs (the equality is not vacuous); the trip needs
the two-selection form. In the NKI simulator: dsa_topk_t at trip 1 is dsa_topk's selection at the served failed-row
shape (128 x 4096) and a small one, and at trip 0 its body does not run."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from trnrt.kernels import dsa_topk as dk
from trnrt.kernels import dsa_topk_t as dkt
from trnrt.models import dsa_long, mla

A, K, KEEP = 8, 512, 512


def case(N: int, kind: str, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    vals = torch.randn(N, A, K, generator=g)
    if kind == "ties":
        vals = torch.randint(0, 6, (N, A, K), generator=g).float()
    if kind == "zeros":
        vals = torch.zeros(N, A, K)
    if kind == "empty":
        cnt = torch.randint(0, K, (N, A, 1), generator=g)
        vals = torch.where(torch.arange(K).view(1, 1, K) < cnt, vals, torch.full_like(vals, dsa_long.NEG_INF))
    if kind == "few":  # fewer valid candidates than keep in every row
        cnt = torch.randint(0, KEEP // A, (N, A, 1), generator=g)
        vals = torch.where(torch.arange(K).view(1, 1, K) < cnt, vals, torch.full_like(vals, dsa_long.NEG_INF))
    lp = torch.stack([torch.randperm(4096, generator=g)[:K].sort().values for _ in range(N * A)]).view(N, A, K)
    cpool = (lp * A + torch.arange(A).view(1, A, 1)).float()
    return vals, cpool


def _trip(v: int) -> torch.Tensor:
    return torch.tensor([v], dtype=torch.int32)


@pytest.mark.parametrize("N", [1, 8, 128])
@pytest.mark.parametrize("kind", ["randn", "ties", "zeros", "empty", "few"])
@pytest.mark.parametrize("trip", [0, 1])
def test_trip_merge_is_the_tie_search_mask(N, kind, trip):
    vals, cpool = case(N, kind, seed=N + 7)
    want = dsa_long.cp_merge_rows(vals, cpool, KEEP)
    got = dsa_long.cp_merge(vals, cpool, KEEP, None, 4096 * A, two_k=True, trip=_trip(trip))
    if not trip:  # no row failed: no selection, deterministically (the caller's live mask is all False there too)
        want = torch.zeros_like(want)
    assert got.dtype == want.dtype and torch.equal(got, want), (got != want).sum()


def test_trip_merge_positive_control():
    """A merge that keeps one fewer must differ."""
    vals, cpool = case(8, "ties", seed=3)
    want = dsa_long.cp_merge_rows(vals, cpool, KEEP)
    wrong = dsa_long.cp_merge(vals, cpool, KEEP - 1, None, 4096 * A, two_k=True, trip=_trip(1))
    assert not torch.equal(wrong, want)


def test_trip_needs_the_two_selection_form():
    vals, cpool = case(8, "randn")
    with pytest.raises(NotImplementedError):
        dsa_long.cp_merge(vals, cpool, KEEP, None, 4096 * A, two_k=False, trip=_trip(1))


def test_route_defaults_off():
    assert mla.CP_FAIL_TRIP is False


@pytest.mark.parametrize(("name", "value"), [("BIG", float(2**21)), ("VISIBLE", -4e29), ("NEG_INF", -2e30)])
def test_rev_moves_with_the_constants_select_compiles_in(name, value, monkeypatch):
    """_select reads dsa_topk's BIG / VISIBLE / NEG_INF, which lie above dsa_topk's REV span: each moves dsa_topk_t's
    rev, so a changed constant can't replay a NEFF built with the old one."""
    assert dkt._kernel_rev() == dkt.REV
    monkeypatch.setattr(dk, name, value)
    assert dkt._kernel_rev() != dkt.REV


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize(("grid", "rowsplit"), [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("kind", ["randn", "ties", "few"])
def test_nki_simulator_trip_kernel(grid, rowsplit, kind, monkeypatch):
    import nki

    from trnrt import platform

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setattr(platform, "nki_grid", lambda: grid)
    monkeypatch.setattr(platform, "lnc_split", lambda kernel: True)
    monkeypatch.setattr(dk, "ROWSPLIT", rowsplit)
    monkeypatch.setattr(dk, "ROWSPLIT_BIG", rowsplit)
    k = dkt.dsa_topk_t_kernel[grid] if grid > 1 else dkt.dsa_topk_t_kernel
    for N in (128, 8):
        vals, _ = case(N, kind, seed=N + grid)
        s = vals.reshape(N, A * K)
        want = dk.emulate(s, KEEP, True, 1, False)
        args = dk.kernel_inputs(s, KEEP, True)
        args["rev"] = dkt.REV
        out1 = torch.as_tensor(nki.simulate(k)(trip=np.ones((1, 1), np.int32), **args)).reshape(N, A * K)
        assert torch.equal(out1, want), (N, kind)
        out0 = torch.as_tensor(nki.simulate(k)(trip=np.zeros((1, 1), np.int32), **args)).reshape(N, A * K)
        assert not torch.equal(out0, want), (N, kind)
