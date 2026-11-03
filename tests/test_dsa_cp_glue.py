"""kernels/dsa_cp_glue.py, attention_cp_rows' local-K selection glue as one NKI kernel. On the host: emulate() equals
models/mla.py's own chain bit for bit (dsa_long.cp_merge, _local_k_fail, _cp_chosen, _pool_row_flat and
_cp_attend_classes(single=True), whose dsa_slots_n.attend call is replaced by a stand-in that records the rows and bias it
is given). In the NKI simulator (Neuron venv only): the kernel at grid 2 equals the same chain. Rows are hostile on
purpose: ties at the threshold beyond room (the tie search decides), every candidate tied, fewer than keep valid, an
all-NaN row, NaN entries below the selection, certificate failures (a rank's (K + 1)-th, a short merge), padding rows,
the tail owned and not, local pools past the table's clamp bound; ranks 0 and A - 1."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from trnrt.kernels import dsa_cp_glue as G
from trnrt.kernels import dsa_slots_n, dsa_topk
from trnrt.models import dsa_long, mla

A, K, KEEP, KP, PS, C = 8, 120, 512, 4, 256, 128
PPL = PS // (KP * A)
NEG = dsa_long.NEG_INF
KINDS = ["rand", "ties", "alltied", "fewvalid", "nan", "nantail", "certnext", "certshort", "pad", "rand"]


def _inputs(n: int, seed: int, rank: int):
    g = torch.Generator().manual_seed(seed)
    Pt = 64
    Pl = Pt * PS // (KP * A)
    vals = torch.full((A, n, K + 1), NEG)
    lpa = torch.zeros((A, n, K + 1), dtype=torch.int64)
    kinds = [KINDS[r % len(KINDS)] for r in range(n)]
    for r, kind in enumerate(kinds):
        for a in range(A):
            cnt = int(torch.randint(5, 40, (1,), generator=g)) if kind == "fewvalid" else 0 if kind == "pad" else K + 1
            v = torch.sort(torch.randn(K + 1, generator=g) * 3, descending=True).values
            if kind == "ties":
                v[40:110] = 0.25  # 70 a rank x 8 ranks tied, around the keep-th value
                v = torch.sort(v, descending=True).values
            if kind == "alltied":
                v[:] = 1.0
            if kind == "certnext":
                v[K] = v[0] + 1.0  # the (K + 1)-th above everything: the certificate fails
            vv = torch.full((K + 1,), NEG)
            vv[:cnt] = v[:cnt]
            if kind == "nan":
                vv[:] = float("nan")
            if kind == "nantail" and cnt > 100:
                vv[100:cnt] = float("nan")
            vals[a, r] = vv
            lpa[a, r] = torch.randperm(Pl + 64, generator=g)[:K + 1]  # local pools (some past the table's bound)
    for r, kind in enumerate(kinds):  # few valid on every rank, a visible (K + 1)-th on one
        if kind == "certshort":
            vals[:, r, 10:] = NEG
            vals[3, r, K] = -2.0
    cpool = (lpa[:, :, :K] * A + torch.arange(A).view(A, 1, 1)).float()
    table = torch.randperm(4 * Pt, generator=g)[:Pt].long() + 1
    npool = torch.randint(0, Pl * A, (n,), generator=g)
    pos = (npool * KP + torch.randint(0, KP, (n,), generator=g) - torch.randint(0, 2, (n,), generator=g)).clamp(min=0)
    for r, kind in enumerate(kinds):
        if kind == "pad":
            npool[r], pos[r] = 0, 0
    return vals, cpool, lpa[rank, :, :K], table, npool, pos, Pl


def _reference(vals, cpool, lp, table, npool, pos, Pl: int, rank: int, monkeypatch):
    """attention_cp_rows' local-K chain as models/mla.py writes it, on the host."""
    n = vals.shape[1]
    every_val, every_c = vals.permute(1, 0, 2), cpool.permute(1, 0, 2)
    vk = every_val[:, :, :K]
    sel = dsa_long.cp_merge(vk, every_c, KEEP, None, Pl * A)
    fail = mla._local_k_fail(vk, every_val[:, :, K], sel, KEEP)
    rk = torch.tensor([rank])
    chosen = mla._cp_chosen(sel, rk, A, n, K)

    def pool_row(mm):
        pg = torch.floor(mm.to(torch.float32) * (1.0 / PPL)).to(torch.int64)
        return mla._pool_row_flat(table, pg, mm, PPL)

    tail_m = torch.floor(npool.to(torch.float32) * (1.0 / A)).to(torch.int64)
    tail_own = (npool - tail_m * A) == rk.view(())
    rows_tail = pool_row(tail_m.clamp(max=Pl - 1).view(n, 1))
    got = {}

    def stand_in(q_all, kc, rows, bias, scale, nn, lse=False):
        got["rows"], got["bias"] = rows, bias
        return torch.zeros(n, 1, 1), torch.zeros(n, 1)

    monkeypatch.setattr(dsa_slots_n, "attend", stand_in)
    mla._cp_attend_classes(torch.zeros(n, 1, 1), None, pool_row(lp), rows_tail, chosen, tail_own, npool, pos, 1.0, KP,
                           C, 1024, single=True)
    monkeypatch.undo()
    return got["rows"].to(torch.int32), got["bias"].float(), fail, rows_tail.view(n), tail_own


def _glue_inputs(vals, cpool, lp, table, npool, pos, rows_tail, tail_own, rank: int):
    n = vals.shape[1]
    s = vals[:, :, :K].permute(1, 0, 2).reshape(n, A * K).float()
    sel0 = dsa_topk.select(s, KEEP, vis_only=True)  # the call cp_merge_rows makes
    c = cpool.permute(1, 0, 2).reshape(n, A * K).float()
    nx = vals[:, :, K].t().float()
    npos = torch.stack([npool.float(), pos.float()], dim=1)
    return s, c, nx, sel0, npos


def _bits(t):
    return t.contiguous().view(torch.int32)


@pytest.mark.parametrize("rank", [0, A - 1])
def test_emulate_is_mla_chain_bit_for_bit(rank, monkeypatch):
    n = 256
    vals, cpool, lp, table, npool, pos, Pl = _inputs(n, rank, rank)
    ref_rows, ref_bias, ref_fail, rows_tail, tail_own = _reference(vals, cpool, lp, table, npool, pos, Pl, rank,
                                                                   monkeypatch)
    s, c, nx, sel0, npos = _glue_inputs(vals, cpool, lp, table, npool, pos, rows_tail, tail_own, rank)
    rows, bias, fail = G.emulate(s, c, nx, sel0, lp, table, rows_tail, tail_own, npos, rank, KEEP, PPL, KP, C,
                                 dsa_long.TIE_BITS)
    assert torch.equal(rows, ref_rows)
    assert torch.equal(_bits(bias), _bits(ref_bias))  # +0 / -0 included
    assert torch.equal(fail, ref_fail)
    # controls: the tie search decides some rows (tied at t beyond room), and some rows fail the certificate
    t = torch.where(sel0 == 0, s, torch.full_like(s, G.BIG)).amin(-1, keepdim=True)
    tied = ((s == t) & (s > dsa_long.VISIBLE)).float().sum(-1)
    assert bool((tied > KEEP - (s > t).float().sum(-1)).any())
    assert bool(ref_fail.any()) and not bool(ref_fail.all())


def test_takes_needs_whole_tiles_and_a_power_of_two_ppl():
    assert G.takes(256, A, K, C, PPL, 2) and G.takes(4096, A, K, C, PPL, 2)
    assert not G.takes(128, A, K, C, PPL, 2)  # one tile: not one for each of two programs
    assert not G.takes(256, A, K, K, PPL, 2)  # the buffer must hold the K entries and the tail
    assert not G.takes(256, A, K, C, 6, 2)


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("rank", [0, A - 1])
def test_nki_simulator_grid2_is_mla_chain(rank, monkeypatch):
    import nki
    import numpy as np

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    n = 256
    vals, cpool, lp, table, npool, pos, Pl = _inputs(n, rank, rank)
    ref_rows, ref_bias, ref_fail, rows_tail, tail_own = _reference(vals, cpool, lp, table, npool, pos, Pl, rank,
                                                                   monkeypatch)
    s, c, nx, sel0, npos = _glue_inputs(vals, cpool, lp, table, npool, pos, rows_tail, tail_own, rank)
    oh = (torch.arange(A) == rank).float()
    a = G.kernel_args(s, c, nx, sel0, lp, table, rows_tail, tail_own, npos, oh, KEEP, PPL, KP, C, dsa_long.TIE_BITS)
    npa = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in a.items()}
    rows, bias, fail = nki.simulate(G.dsa_cp_glue_kernel[2])(**npa)
    assert torch.equal(torch.from_numpy(np.asarray(rows)).to(torch.int32), ref_rows)
    kb = torch.from_numpy(np.asarray(bias, dtype=np.float32)).view(n, C, KP)
    assert torch.equal(_bits(kb), _bits(ref_bias))
    assert torch.equal(torch.from_numpy(np.asarray(fail, dtype=np.float32)).view(n) > 0.5, ref_fail)
