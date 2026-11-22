"""The context-parallel prefill's failed-row attention with both slot classes counting live entries only (models/mla.py
CP_FAIL_ATTN_TRIP, _cp_attend_classes rows_live).
- Every live entry's o / lse is the unmasked call's bit for bit, with live rows in both classes (small and full buffers);
- forced failures: attention_cp_rows' combine (where(use, o_f[slot], o_r)) gives the same o / lse bit for bit;
- the trip: the two class calls' row counts add up to the live count, so with no failed row neither runs a row;
- positive control: masking out one live entry changes the combine (the equality is not vacuous);
- the route is off by default."""

from __future__ import annotations

import pytest
import torch

from trnrt.kernels import dsa_slots_n
from trnrt.models import dsa_long, mla

H, R, KP, KEEP, SMALL, FULL, NPOOL = 2, 64, 4, 200, 128, 256, 3000


def entries(T: int, nf: int, seed: int):
    """_cp_attend_classes' inputs for T compacted failed-row entries of which the first nf are live (as attention_cp_rows'
    compact gives them): live rows select entries, about half of them past the small buffer's S1 (the full class); the other
    entries select nothing and own no tail (chosen_f & live, tail_own[fi] & live)."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(T, H, R, generator=g)
    kc = torch.randn(NPOOL * KP, R, generator=g)
    rows_sel = torch.randint(1, NPOOL, (T, KEEP), generator=g)
    rows_tail = torch.randint(1, NPOOL, (T, 1), generator=g)
    live = torch.arange(T) < nf
    mine = torch.zeros(T, KEEP, dtype=torch.bool)
    for r in range(nf):
        last = (
            int(torch.randint(1, SMALL - 1, (1,), generator=g))
            if r % 2 == 0
            else int(torch.randint(SMALL, KEEP, (1,), generator=g))
        )
        mine[r, :last] = torch.rand(last, generator=g) > 0.3
        mine[r, last - 1] = True
    tail_own = (torch.rand(T, generator=g) > 0.5) & live
    npool = torch.randint(0, 50, (T,), generator=g)
    positions = npool * KP + torch.randint(0, 8, (T,), generator=g)
    return q, kc, rows_sel, rows_tail, mine, tail_own, npool, positions, live


def attend(args, rows_live):
    q, kc, rows_sel, rows_tail, mine, tail_own, npool, positions, _ = args
    return mla._cp_attend_classes(
        q,
        kc,
        rows_sel,
        rows_tail,
        mine,
        tail_own,
        npool,
        positions,
        0.125,
        KP,
        SMALL,
        FULL,
        rows_live=rows_live,
    )


@pytest.mark.parametrize(("T", "nf"), [(16, 0), (16, 1), (16, 5), (16, 16), (128, 0), (128, 37)])
def test_live_entries_are_bit_for_bit(T, nf):
    args = entries(T, nf, seed=T * 100 + nf)
    oa, la = attend(args, None)
    ob, lb = attend(args, args[-1])
    assert torch.equal(oa[:nf], ob[:nf]) and torch.equal(la[:nf], lb[:nf])
    fits_full = sum(1 for r in range(nf) if r % 2 == 1)
    assert nf == 0 or fits_full > 0 or nf == 1  # live rows in both classes whenever nf >= 2


def combine(o_r, lse_r, o_f, lse_f, fail, nF):
    """attention_cp_rows' where-form combine of the failed rows' attention into the main rows' (the served form)."""
    n = fail.shape[0]
    slot = (torch.cumsum(fail.to(torch.float32), 0) - 1.0).clamp(min=0.0).to(torch.int64)
    use = fail & (slot < nF)
    slot = slot.clamp(max=nF - 1)
    return torch.where(use.view(n, 1, 1), o_f[slot], o_r), torch.where(use.view(n, 1), lse_f[slot], lse_r)


@pytest.mark.parametrize("nfail", [0, 1, 7, 16, 20])
def test_forced_failures_combine_bit_for_bit(nfail):
    """Main rows N, nF entries; nfail rows fail (more than nF: the rows past nF keep o_r in both forms)."""
    N, nF = 64, 16
    g = torch.Generator().manual_seed(nfail + 11)
    fail = torch.zeros(N, dtype=torch.bool)
    fail[torch.randperm(N, generator=g)[:nfail]] = True
    _fi, nf = dsa_long.compact(fail.to(torch.float32).view(1, N), nF)
    nf = int(nf.reshape(()))
    args = entries(nF, nf, seed=nfail + 3)
    o_r = torch.randn(N, H, R, generator=g)
    lse_r = torch.randn(N, H, generator=g)
    oa, la = attend(args, None)
    ob, lb = attend(args, args[-1])
    ca, cb = (
        combine(o_r, lse_r, oa, la, fail, nF),
        combine(o_r, lse_r, ob, lb, fail, nF),
    )
    assert torch.equal(ca[0], cb[0]) and torch.equal(ca[1], cb[1])
    if nfail:  # the combine does take failed rows' attention (else the test would be vacuous)
        assert not torch.equal(ca[0], o_r)


def test_trip_class_counts(monkeypatch):
    """Each class call's n is its class's live count, small first: entries() puts live row r in the small class when r is
    even (its last selected entry before SMALL - 1) and in the full one when odd, so nf live rows are (nf + 1) // 2 small
    and nf // 2 full; 0 and 0 with no failed row (the kernel's row loop runs no iteration)."""
    seen = []
    real = dsa_slots_n.attend

    def spy(q, kc, rows, bias, scale, n, lse=False, fallback=False):
        seen.append(int(n.reshape(())))
        return real(q, kc, rows, bias, scale, n, lse=lse, fallback=fallback)

    monkeypatch.setattr(dsa_slots_n, "attend", spy)
    for nf in (0, 1, 5, 16):
        seen.clear()
        args = entries(16, nf, seed=nf)
        attend(args, args[-1])
        assert seen == [(nf + 1) // 2, nf // 2], (nf, seen)
    seen.clear()
    attend(entries(16, 0, seed=0), None)
    assert seen == [16, 0]  # unmasked: all 16 (selection-less) entries take the small class, as without the route


def test_positive_control_dropped_live_entry():
    """Masking out one live entry leaves its failed row with the unwritten (here zero) o_f: the combine must differ."""
    N, nF = 64, 16
    fail = torch.zeros(N, dtype=torch.bool)
    fail[[3, 9, 40]] = True
    _fi, nf = dsa_long.compact(fail.to(torch.float32).view(1, N), nF)
    nf = int(nf.reshape(()))
    args = entries(nF, nf, seed=5)
    g = torch.Generator().manual_seed(1)
    o_r, lse_r = torch.randn(N, H, R, generator=g), torch.randn(N, H, generator=g)
    good = attend(args, args[-1])
    wrong = attend(args, args[-1] & (torch.arange(nF) < nf - 1))
    assert not torch.equal(
        combine(o_r, lse_r, *good, fail, nF)[0],
        combine(o_r, lse_r, *wrong, fail, nF)[0],
    )


def test_route_default_off():
    assert mla.CP_FAIL_ATTN_TRIP is False
