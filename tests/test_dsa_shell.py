"""kernels/dsa_shell.py, the DSA decode shell's selection (sel2) and attend calls, whose bodies the fused DSA layer
(kernels/dsa_sa.py) runs verbatim.

On the host: emulate() is the context-parallel decode branch's selection (the page-key scores, the device selection,
lval, cpool, pool rows): its rows address the pools cp_pool_rows gives, its scores are the XLA form's up to fp32 order,
its list is a sort's top keep in pool order, and a wrong page width is caught; emulate_merge's slots are the global top
keep and the tail pool.

In the NKI simulator (Neuron venv): sel2's lval / cpool / rows are the served path's bits (the dsa_index kernel's
scores, then the exact selection and compaction) at grids 1 and 2, several rows and contexts, ties at the threshold and
the served shape; its stacked compaction is the per-row one bit for bit (with a dispatch control); attend in its served
form (the merge's rows stacked, the slot table in SBUF, the tail block stacked, the over fallback folded in) gives the
served chain's slot rows, bias, over, o and lse bit for bit (the merge's emulation, then dsa_slots_c's kernel, or
dsa_slots_n_s's over every row when a row is over).
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from trnrt.kernels import dsa_shell as dsh
from trnrt.models import dsa_long

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")

D, PPP = dsh.D, dsh.PPP
# attend's served form: the merge's rows interleaved and stacked, the selection in SBUF, the slot table in SBUF, the
# tail block for every row at once, the over fallback folded in
SERVED_ATTEND = dict(il=1, sb=1, st=1, ss=1, tl=1, fo=1)


def case(B: int, NPg: int, Hi: int, nloc, seed: int = 0, repeat: bool = False, pages: int | None = None):
    """Inputs of one call: q [B, Hi, D] bf16, w [B, Hi] fp32, pkc [Npages, PPP D] bf16, table [B, NPg], nloc [B]. With
    repeat, every row's table repeats a few pages, so pools of different index score exactly the same (ties)."""
    g = torch.Generator().manual_seed(seed)
    npages = pages or 3 * NPg
    pkc = (torch.randn(npages, PPP * D, generator=g) * 0.5).bfloat16()
    table = torch.stack([torch.randperm(npages, generator=g)[:NPg] for _ in range(B)])
    if repeat:
        table[:, 1::7] = table[:, :1]  # every 7th page is the row's first: its 8 pools' scores recur
    q = torch.randn(B, Hi, D, generator=g).bfloat16()
    w = torch.randn(B, Hi, generator=g) * (D**-0.5)
    return q, w, pkc, table, torch.tensor(nloc, dtype=torch.int64)


def test_rows_address_the_cp_pool_rows():
    """emulate's rows are cp_pool_rows' pool rows of the selected local pools, an independent derivation of the page
    layout (page_size 256, kpool 4, A 8: 8 local pools per page)."""
    q, w, pkc, table, nloc = case(2, 256, 4, [2048, 1500])
    lval, cpool, rows, sc = dsh.emulate(q, w, pkc, table, nloc, 128, 8, 3)
    prow = dsa_long.cp_pool_rows(table, 256, 4, 8)  # [B, Pl]
    lp = ((cpool - 3) / 8).long()
    assert torch.equal(rows.long(), prow.gather(1, lp))
    assert torch.equal(cpool, (lp * 8 + 3).float())


def test_scores_are_the_xla_forms():
    q, w, pkc, table, nloc = case(2, 256, 4, [2048, 700], seed=1)
    _, _, _, sc = dsh.emulate(q, w, pkc, table, nloc, 128, 8, 0)
    prow = dsa_long.cp_pool_rows(table, 256, 4, 8)
    pk = pkc.view(-1, D)[prow]  # [B, Pl, D]
    ref = dsa_long.scores(q, w * D**0.5, pk, nloc, D**-0.5)
    live = torch.arange(sc.shape[1]).view(1, -1) < nloc.view(-1, 1)
    torch.testing.assert_close(sc[live], ref[live], rtol=1e-5, atol=1e-5)
    assert torch.all(sc[~live] == dsh.NEG_INF) and torch.all(ref[~live] == dsh.NEG_INF)


@pytest.mark.parametrize("nl0", [2048, 1500, 100, 0])
def test_list_is_the_top_keep_in_pool_order(nl0):
    """lval / cpool against a sort: the keep largest visible scores (lowest index first among equals), ascending by
    pool; NEG_INF and pool 0 past the count."""
    keep = 128
    q, w, pkc, table, nloc = case(1, 256, 4, [nl0], seed=2, repeat=True)
    lval, cpool, rows, sc = dsh.emulate(q, w, pkc, table, nloc, keep, 8, 5)
    s = sc[0]
    vis = torch.nonzero(s > dsa_long.VISIBLE).view(-1)
    order = sorted(vis.tolist(), key=lambda m: (-s[m].item(), m))[:keep]
    want = sorted(order)
    n = len(want)
    lp = ((cpool[0] - 5) / 8).long()
    assert lp[:n].tolist() == want and torch.all(lp[n:] == 0)
    assert torch.equal(lval[0, :n], s[want]) and torch.all(lval[0, n:] == dsh.NEG_INF)


def test_a_wrong_page_width_is_caught(monkeypatch):
    """The positive control: emulate with 4 pools per page (another CP degree's layout) addresses other pools."""
    q, w, pkc, table, nloc = case(1, 256, 4, [2048], seed=3)
    _, _, rows, _ = dsh.emulate(q, w, pkc, table, nloc, 128, 8, 0)
    monkeypatch.setattr(dsh, "PPP", 4)
    _, _, rows4, _ = dsh.emulate(q, w, pkc.view(-1, 4 * D), table, nloc, 128, 8, 0)
    assert not torch.equal(rows, rows4)


# -- the merge and the slot table (emulation) -------------------------------------------------------------------------


def merge_case(T: int, A: int, keep: int, NPg: int, npool, seed: int = 0, levels: int | None = None):
    """Every rank's local list as the context-parallel decode branch makes it (dsa_long.select_device over the rank's
    local pools m, context pool m A + a, of one context score row; lval / cpool past the count NEG_INF / a), gathered in
    rank order: (ev, ec [A, T, keep], the context scores [T, P], rows_sel [T, keep], table [T, NPg], positions [T]).
    levels: scores quantized to that many values (ties across ranks at the threshold)."""
    g = torch.Generator().manual_seed(seed)
    Pl = NPg * PPP
    P = Pl * A
    sc = torch.rand(T, P, generator=g)
    if levels:
        sc = torch.floor(sc * levels) / levels
    npool = torch.tensor(npool, dtype=torch.int64)
    sc = torch.where(torch.arange(P).view(1, P) < npool.view(T, 1), sc, torch.full_like(sc, dsh.NEG_INF))
    ev, ec = [], []
    for a in range(A):
        loc = sc[:, a::A]  # local pool m = context pool m A + a
        lp, lc = dsa_long.select_device(loc, keep)
        k = torch.arange(keep).view(1, keep)
        ev.append(torch.where(k < lc.view(T, 1), loc.gather(1, lp), torch.full_like(loc[:, :keep], dsh.NEG_INF)))
        ec.append((lp * A + a).float())
    rows_sel = torch.randint(0, 1 << 20, (T, keep), generator=g, dtype=torch.int32)
    table = torch.stack([torch.randperm(4 * NPg, generator=g)[:NPg] for _ in range(T)]).to(torch.int32)
    pos = npool * 4 + torch.randint(0, 4, (T,), generator=g)  # npool complete pools, 0-3 tokens of the next
    return torch.stack(ev), torch.stack(ec), sc, rows_sel, table, pos


@pytest.mark.parametrize("levels", [None, 16])
def test_merge_mine_is_the_global_top_keep(levels):
    """emulate_merge's selected slots of rank a are its list entries whose context pool is in the global top keep of
    the context scores (the keep largest, the lowest context pool first among equals): the merge of the local lists
    against the selection over the whole context, an independent derivation."""
    T, A, keep, NPg = 2, 8, 128, 32
    ev, ec, sc, rows_sel, table, pos = merge_case(T, A, keep, NPg, [2000, 600], seed=3, levels=levels)
    for a in (0, 5):
        srows, sbias = dsh.emulate_merge(ev, ec, rows_sel, table, pos, keep, A, a)
        for b in range(T):
            s = sc[b]
            vis = torch.nonzero(s > dsa_long.VISIBLE).view(-1).tolist()
            top = set(sorted(vis, key=lambda m: (-s[m].item(), m))[:keep])
            got = {int(ec[a, b, k]) for k in range(keep) if sbias[b, k, 0] == 0}
            want = {m for m in top if m % A == a}
            assert got == want, (a, b, levels, len(got), len(want))
            assert torch.equal(srows[b, :keep], rows_sel[b])


def test_merge_tail_slot():
    """The tail slot: the incomplete pool's row (pool_row of npool // A), visible on its owner (npool % A) at the tokens
    npool 4 + t <= pos and masked on every other rank; the slots past it are row 0, masked."""
    T, A, keep, NPg = 4, 8, 128, 32
    ev, ec, sc, rows_sel, table, pos = merge_case(T, A, keep, NPg, [2000, 2001, 600, 7], seed=4)
    npool = dsa_long.npools(pos, 4)
    for a in range(A):
        srows, sbias = dsh.emulate_merge(ev, ec, rows_sel, table, pos, keep, A, a)
        for b in range(T):
            n = int(npool[b])
            own = n % A == a
            vis = [own and n * 4 + t <= int(pos[b]) for t in range(4)]
            assert [bool(x == 0) for x in sbias[b, keep]] == vis
            m = n // A
            assert int(srows[b, keep]) == int(table[b, m // PPP]) * PPP + m % PPP
            assert torch.all(srows[b, keep + 1 :] == 0) and torch.all(sbias[b, keep + 1 :] == dsh.NEG_INF)


# -- sel2 in the NKI simulator ------------------------------------------------------------------------------------------


def _served(q, w, pkc, table, nloc, keep, A, rank):
    """The served selection's tensors from the dsa_index kernel's scores (simulated, grid 1), then the exact selection
    and compaction (dsa_long.select_device on the host: the set and order dsa_topk + compact give)."""
    import ml_dtypes
    import nki
    import numpy as np

    from trnrt.kernels import dsa_index

    B, NPg = table.shape
    P = NPg * PPP
    Hi = q.shape[1]
    cand = torch.where(torch.arange(P).view(1, P) < nloc.view(-1, 1), 0.0, dsh.NEG_INF)
    qT = q.reshape(B * Hi, D).t().contiguous()
    wb = w.float().repeat(1, 4).unsqueeze(1).expand(B, 128, 4 * Hi).contiguous()
    bf = lambda t: t.float().numpy().astype(ml_dtypes.bfloat16)  # noqa: E731
    sc = nki.simulate(dsa_index.dsa_index_kernel)(
        qT=bf(qT),
        wb=wb.numpy(),
        pkc=bf(pkc),
        ppg=dsa_index.page_groups(table).numpy(),
        cand=cand.numpy(),
        identb=bf(torch.eye(128)),
        rev=dsa_index.REV,
        dge=0,
        spl=1,
        nr=dsa_index.NR,
        qr=dsa_index.QR,
        dbg=0,
    )
    sc = torch.from_numpy(np.asarray(sc, dtype=np.float32))
    lp, lc = dsa_long.select_device(sc, keep)
    k = torch.arange(keep).view(1, keep)
    lval = torch.where(k < lc.view(B, 1), sc.gather(1, lp), torch.full_like(sc[:, :keep], dsh.NEG_INF))
    cpool = (lp * A + rank).float()
    pg = lp // PPP
    rows = (table.long().gather(1, pg) * PPP + lp % PPP).to(torch.int32)
    return lval, cpool, rows, sc


def _bits(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32)) if a.dtype == torch.float32 else torch.equal(a, b)


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
@pytest.mark.parametrize(
    "B,NPg,nloc,keep,repeat",
    [
        (1, 256, [2048 - 5], 128, False),  # one chunk of 16 columns
        (2, 1024, [8191, 90], 128, True),  # a row under keep, ties
        (5, 256, [1024, 1025, 0, 513, 2047], 128, True),  # an empty row, rows across the programs
        (8, 1024, [8000, 300, 1500, 2000, 64, 1024, 777, 8191], 256, False),  # B 8: four rows per program
    ],
)
def test_sel2_is_the_served_bits(monkeypatch, grid, B, NPg, nloc, keep, repeat):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    q, w, pkc, table, nl_ = case(B, NPg, 8, nloc, seed=70 + B + NPg, repeat=repeat)
    want = _served(q, w, pkc, table, nl_, keep, 8, 6)
    got = dsh.select2(q, w, pkc, table, nl_, keep, 8, torch.tensor([6]), simulate=True, grid=grid, cs=1)
    for name, a, b in zip(("lval", "cpool", "rows"), want, got):
        assert _bits(a, b), (name, grid, B, NPg, nloc, (a != b).sum().item())


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
def test_sel2_ties_and_served_shape(monkeypatch, grid):
    """Forced ties (more tied at the threshold than room: asserted), then the served shape: 4096 pages, Hi 32, keep 512
    (a 1M context and a 640K one, 32768 and 20000 local pools of a rank at CP 8)."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    q, w, pkc, table, nl_ = case(1, 256, 8, [2000], seed=81)
    table = table[:, :2].repeat(1, 128)
    want = _served(q, w, pkc, table, nl_, 128, 8, 0)
    t = want[0][0].min()
    assert int((want[3][0] == t).sum()) > int((want[0][0] == t).sum())
    got = dsh.select2(q, w, pkc, table, nl_, 128, 8, torch.tensor([0]), simulate=True, grid=grid, cs=1)
    for a, b in zip(want, got):
        assert _bits(a, b)
    q, w, pkc, table, nl_ = case(2, 4096, 32, [32768, 20000], seed=82, pages=4096 * 2 + 64)
    want = _served(q, w, pkc, table, nl_, 512, 8, 5)
    got = dsh.select2(q, w, pkc, table, nl_, 512, 8, torch.tensor([5]), simulate=True, grid=grid, cs=1)
    for name, a, b in zip(("lval", "cpool", "rows"), want, got):
        assert _bits(a, b), (name, (a != b).sum().item())


@needs_nki
def test_another_ranks_list_differs(monkeypatch):
    """The positive control on cpool: rank 2's list is not rank 3's (the rank reaches the kernel's arithmetic)."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    q, w, pkc, table, nl_ = case(1, 256, 8, [2048], seed=9)
    a = dsh.select2(q, w, pkc, table, nl_, 128, 8, torch.tensor([3]), simulate=True, grid=2, cs=1)[1]
    b = dsh.select2(q, w, pkc, table, nl_, 128, 8, torch.tensor([2]), simulate=True, grid=2, cs=1)[1]
    assert not torch.equal(a, b)


def _c2s_case(B: int, NPg: int, keep: int, ties: bool, seed: int):
    """select2's inputs for the stacked-compaction tests: rows of every kind in turn (nearly full, part full, fewer
    visible pools than keep, empty, exactly keep); with ties, every row's table two pages repeated (16 distinct scores:
    the keep-th score tied far past room)."""
    P = NPg * PPP
    nloc = []
    for b in range(B):
        nloc.append((P - 5, P // 3 + 7 * b, keep - 9, 0, keep)[b % 5])
    q, w, pkc, table, nl_ = case(B, NPg, 8, nloc, seed=seed)
    if ties:
        table = table[:, :2].repeat(1, NPg // 2)
    return q, w, pkc, table, nl_


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
@pytest.mark.parametrize("B", [1, 2, 3, 4, 5, 6, 8])
def test_compact2_stacked_bitwise(monkeypatch, grid, B):
    """sel2 with its compaction's rows stacked (cs 1, served: _compact2s) is sel2 with them per row (cs 0: _compact2)
    bit for bit: lval, cpool and rows (fp32 compared as int32 bits, so a zero's sign counts), at 1 to 8 rows (groups of
    up to 4 rows a program, short last groups, a program with no row at B 1 grid 2), on both grids and in the
    compaction's three shape regimes: NT 16 / keep 128 (one k tile a row) with ties at the threshold; NT 128 / keep 384
    (PSUM chunks that straddle rows); NT 256 / keep 512 (the served shape) at B 1 and 5."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    shapes = [(256, 128, True), (2048, 384, False)]
    if B in (1, 5):
        shapes.append((4096, 512, False))
    for NPg, keep, ties in shapes:
        q, w, pkc, table, nl_ = _c2s_case(B, NPg, keep, ties, seed=900 + 7 * B + NPg)
        rank = torch.tensor([3])
        a = dsh.select2(q, w, pkc, table, nl_, keep, 8, rank, simulate=True, grid=grid, cs=0)
        b = dsh.select2(q, w, pkc, table, nl_, keep, 8, rank, simulate=True, grid=grid, cs=1)
        assert bool((a[0] > dsh.NEG_INF).any()), ("nothing selected", B, grid, NPg)  # the case selects something
        for name, x, y in zip(("lval", "cpool", "rows"), a, b):
            assert _bits(x, y), (name, B, grid, NPg, keep, int((x != y).sum()))


def _no_compaction(X, ii, bb):
    """A _compact2s that writes nothing (the dispatch control below)."""
    return


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
def test_compact2_stacked_dispatch_control(monkeypatch, grid):
    """The positive control for test_compact2_stacked_bitwise, which would pass if cs never reached the kernel's
    dispatch: with _compact2s replaced by a function that writes nothing, cs 1's cpool (lp A + rank: >= 3 everywhere at
    rank 3) is not cs 0's, while cs 0 is unchanged by the swap. (pf 3: a static argument no other call here passes, so
    no trace of the real _compact2s can be reused.)"""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    q, w, pkc, table, nl_ = _c2s_case(8, 256, 128, False, seed=977)  # B 8: >= 4 rows a program at grid 2 too
    rank = torch.tensor([3])
    a = dsh.select2(q, w, pkc, table, nl_, 128, 8, rank, simulate=True, grid=grid, pf=3, cs=0)
    monkeypatch.setattr(dsh, "_compact2s", _no_compaction)
    a2 = dsh.select2(q, w, pkc, table, nl_, 128, 8, rank, simulate=True, grid=grid, pf=3, cs=0)
    b = dsh.select2(q, w, pkc, table, nl_, 128, 8, rank, simulate=True, grid=grid, pf=3, cs=1)
    for x, y in zip(a, a2):
        assert _bits(x, y)
    assert bool((a[1] >= 3).all())
    assert not _bits(a[1], b[1]), "cs 1 did not reach _compact2s"


# -- attend in the NKI simulator ----------------------------------------------------------------------------------------


def attend_case(T: int, keep: int, NPg: int, npool, seed: int = 0, hot_rank: int | None = None, levels=None):
    """attend's inputs: merge_case's lists (slot rows below the cache's pool rows), every head's latent query [T, 16,
    512] bf16 and an fp8 latent cache of 4096 pool rows. hot_rank: every top score on that rank's pools (its slots then
    exceed 128: the over path)."""
    A = 8
    g = torch.Generator().manual_seed(seed + 100)
    ev, ec, sc, rows_sel, table, pos = merge_case(T, A, keep, NPg, npool, seed=seed, levels=levels)
    if hot_rank is not None:  # rank hot_rank's entries outrank every other rank's
        ev[hot_rank] = torch.where(ev[hot_rank] > dsa_long.VISIBLE, ev[hot_rank] + 10.0, ev[hot_rank])
    rows_sel = torch.randint(0, 4096, (T, keep), generator=g, dtype=torch.int32)
    q = (torch.randn(T, 16, 512, generator=g) * 0.05).bfloat16()
    kc = (torch.randn(4096 * 4, 1, 512, generator=g) * 2).clamp(-200, 200).to(torch.float8_e4m3fn)
    return ev, ec, rows_sel, table, pos, q, kc


def _np(v):
    import ml_dtypes

    if v.dtype == torch.bfloat16:
        return v.float().numpy().astype(ml_dtypes.bfloat16)
    if v.dtype == torch.float8_e4m3fn:
        return v.view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
    return v.numpy()


def _served_attend(ev, ec, rows_sel, table, pos, q, kc, keep, rank, ns):
    """The served chain: emulate_merge's slot rows and bias, then dsa_slots_c's kernel over them in the simulator
    (kernel_args, as dsa_slots_c.attend calls it). (o, lse, srows, sbias, over)."""
    import nki
    import numpy as np

    from trnrt.kernels import dsa_slots_c

    srows, sbias = dsh.emulate_merge(ev, ec, rows_sel, table, pos, keep, 8, rank, 4, ns)
    kw, N, Np, over = dsa_slots_c.kernel_args(q, kc.reshape(kc.shape[0], -1), srows.long(), sbias, 0.05, lse=True)
    conv = {k: (_np(v) if isinstance(v, torch.Tensor) else v) for k, v in kw.items()}
    conv["identb"] = _np(torch.eye(128).to(torch.bfloat16))
    o, ls = nki.simulate(dsa_slots_c.dsa_slots_c_kernel)(**conv)
    H, R = q.shape[1], q.shape[2]
    o = torch.from_numpy(np.asarray(o, dtype=np.float32)).reshape(Np, H, R)[:N]
    ls = torch.from_numpy(np.asarray(ls, dtype=np.float32)).reshape(Np, H)[:N]
    return o, ls, srows, sbias, over.view(1)


def _fallback_sim(q, kc, srows, sbias, grid):
    """The caller's form of an over call without the fold: dsa_slots_n's split kernel (dsa_slots_n_s) over every row in
    the simulator, on its served, permuted arguments. (o [T, H, R], lse [T, H])."""
    import nki
    import numpy as np

    from tests.test_dsa_slots import kernel_args
    from trnrt.kernels import dsa_slots as ds
    from trnrt.kernels import dsa_slots_n_s as dns

    T, H, R = q.shape
    rpi = min(ds.RPI, T) if ds.FIT else ds.RPI
    args, Np = kernel_args(q, kc.reshape(kc.shape[0], -1), srows.long(), sbias, rpi)
    conv = {k: _np(v) for k, v in args.items()}
    it = np.array([[-(-T // rpi)]], dtype=np.int32)
    k = dns.dsa_slots_n_s_kernel
    o, ls = nki.simulate(k[grid] if grid > 1 else k)(
        **conv, n_iter=it, scale=0.05, fp8=1, want_lse=1, rev=dns.REV, qkw=ds.QKW
    )
    o = torch.from_numpy(np.asarray(o, dtype=np.float32)).reshape(Np, H, R)[:T]
    ls = torch.from_numpy(np.asarray(ls, dtype=np.float32)).reshape(Np, H)[:T]
    return o, ls


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
@pytest.mark.parametrize(
    "T,npool,levels,hot",
    [
        (1, [2000], 16, None),
        (2, [2047, 600], 16, None),
        (3, [2000, 900, 1500], None, None),
        (4, [2000, 7, 1500, 2047], 16, None),
        (5, [2000, 2001, 600, 7, 1500], 16, None),
        (2, [1805, 1797], None, 5),
    ],
)
def test_attend_served_form_is_the_served_chain(monkeypatch, grid, T, npool, levels, hot):
    """attend in its served form against the served chain: the slot rows, bias and over, bit for bit; o and lse bit for
    bit where no row is over (the compacted slots), and the fallback's where one is (the fold). Ties across ranks at the
    threshold (levels 16: more tied than room, asserted), rows of 1-5 per call (odd splits across the programs), a row
    with 7 live pools, the hot-rank over path; two ranks each."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    keep, NPg, ns = 128, 32, 256
    ev, ec, rows_sel, table, pos, q, kc = attend_case(T, keep, NPg, npool, seed=T + 210, hot_rank=hot, levels=levels)
    ran = False
    for rank in (0, 5):
        want = _served_attend(ev, ec, rows_sel, table, pos, q, kc, keep, rank, ns)
        got = dsh.attend(
            ev,
            ec,
            rows_sel,
            table,
            pos,
            q,
            kc,
            keep,
            8,
            torch.tensor([rank]),
            0.05,
            ns=ns,
            simulate=True,
            grid=grid,
            **SERVED_ATTEND,
        )
        assert torch.equal(want[2], got[2]) and _bits(want[3], got[3]), ("slots", T, rank)
        assert float(got[4]) == float(want[4]), ("over", T, rank)
        if float(want[4]) == 0:
            assert _bits(want[0], got[0]) and _bits(want[1], got[1]), ("o / lse", T, rank)
        else:
            ran = True
            fb = _fallback_sim(q, kc, want[2], want[3], grid)
            assert _bits(fb[0], got[0]) and _bits(fb[1], got[1]), ("fold", T, rank)
    if hot is not None:
        assert ran, "no rank took the over path"
    if levels:  # the tie search decided: some row's candidates tied at its threshold exceed its room
        over_room = []
        for b in range(T):
            c = ev[:, b, :].reshape(-1)
            t = c.topk(keep).values[-1]
            over_room.append(t > dsa_long.VISIBLE and int((c == t).sum()) > keep - int((c > t).sum()))
        assert any(over_room), "no row needs the tie search"


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
@pytest.mark.parametrize(
    "T,npool,hot,hot_rows,seed",
    [(1, [1805], 5, None, 306), (2, [1805, 1797], 5, None, 7), (4, [1805, 2000, 1797, 600], 5, (2, 4), 309)],
)
def test_attend_fold_control(monkeypatch, grid, T, npool, hot, hot_rows, seed):
    """The positive control for the fold: without it (fo 0) an over row's o is the compaction's, which differs from the
    fallback's, so the fold's equality to the fallback above means its loop ran. hot_rows (2, 4) at T 4: only rows 2-3
    are over, the second program's at grid 2, so the first program's rows 0-1 must take the fallback too (the flags'
    exchange)."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    keep, NPg, ns = 128, 32, 256
    ev, ec, rows_sel, table, pos, q, kc = attend_case(
        T, keep, NPg, npool, seed=seed, hot_rank=hot if hot_rows is None else None
    )
    if hot_rows is not None:  # rank hot's entries outrank every other rank's, in rows hot_rows[0] .. [1] - 1 only
        a, b = hot_rows
        ev[hot, a:b] = torch.where(ev[hot, a:b] > dsa_long.VISIBLE, ev[hot, a:b] + 10.0, ev[hot, a:b])
    kw = dict(SERVED_ATTEND, fo=0)
    ran = False
    for rank in (0, 5):
        base = dsh.attend(
            ev,
            ec,
            rows_sel,
            table,
            pos,
            q,
            kc,
            keep,
            8,
            torch.tensor([rank]),
            0.05,
            ns=ns,
            simulate=True,
            grid=grid,
            **kw,
        )
        got = dsh.attend(
            ev,
            ec,
            rows_sel,
            table,
            pos,
            q,
            kc,
            keep,
            8,
            torch.tensor([rank]),
            0.05,
            ns=ns,
            simulate=True,
            grid=grid,
            **SERVED_ATTEND,
        )
        assert torch.equal(base[2], got[2]) and _bits(base[3], got[3]) and float(base[4]) == float(got[4])
        if float(base[4]):
            ran = True
            want = _fallback_sim(q, kc, base[2], base[3], grid)
            assert not _bits(base[0], want[0]), ("control: fo 0's o is the fallback's", T, rank)
            if hot_rows is not None and grid == 2:  # rows 0-1 are not over: their compaction ran, and differs
                assert not _bits(base[0][: hot_rows[0]], want[0][: hot_rows[0]]), ("control: exchange", rank)
            assert _bits(want[0], got[0]) and _bits(want[1], got[1]), ("o / lse", T, rank)
    assert ran, "no rank took the over path"


@needs_nki
@pytest.mark.parametrize(
    "NPg,T,npool",
    [
        (1024, 6, [60000, 9000, 33000, 1205, 64001, 48000]),
        (4096, 8, [250000, 9000, 131000, 1205, 64001, 48000, 262143, 77777]),
    ],
)
def test_attend_at_the_served_decode_shapes(monkeypatch, NPg, T, npool):
    """attend in its served form at the served decode shapes: the 1024- and 4096-page buckets, B6 and B8, keep 512, 640
    slots, grid 2: the served chain's slot rows, bias, o and lse, bit for bit, on two ranks. (Pages repeat in the table
    so the latent cache stays small; the merge and the slots read it the same.)"""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    keep, ns = 512, 640
    ev, ec, rows_sel, table, pos, q, kc = attend_case(T, keep, NPg, npool, seed=NPg + T)
    table = table % 512  # pool rows < 4096, inside attend_case's latent cache
    for rank in (0, 3):
        want = _served_attend(ev, ec, rows_sel, table, pos, q, kc, keep, rank, ns)
        got = dsh.attend(
            ev,
            ec,
            rows_sel,
            table,
            pos,
            q,
            kc,
            keep,
            8,
            torch.tensor([rank]),
            0.05,
            ns=ns,
            simulate=True,
            grid=2,
            **SERVED_ATTEND,
        )
        assert torch.equal(want[2], got[2]) and _bits(want[3], got[3]), ("slots", NPg, T, rank)
        assert float(got[4]) == float(want[4]), ("over", NPg, T, rank)
        if float(want[4]) == 0:
            assert _bits(want[0], got[0]) and _bits(want[1], got[1]), ("o / lse", NPg, T, rank)


@needs_nki
def test_attend_short_contexts_b8(monkeypatch):
    """attend in its served form at 128-384-token contexts at B8 on two programs (every visible pool selected, fewer
    candidates than keep), keep 512, 640 slots, every row and three ranks, against the served chain bit for bit. (A
    regime where one batch row once differed on the device from its decode step.)"""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    keep, ns, NPg = 512, 640, 1024
    pos = [128, 192, 256, 384, 128, 192, 256, 384]
    npool = [(p + 1) // 4 for p in pos]
    ev, ec, rows_sel, table, _, q, kc = attend_case(8, keep, NPg, npool, seed=91)
    table = table % 512
    pos = torch.tensor(pos)
    for rank in (0, 3, 7):
        want = _served_attend(ev, ec, rows_sel, table, pos, q, kc, keep, rank, ns)
        got = dsh.attend(
            ev,
            ec,
            rows_sel,
            table,
            pos,
            q,
            kc,
            keep,
            8,
            torch.tensor([rank]),
            0.05,
            ns=ns,
            simulate=True,
            grid=2,
            **SERVED_ATTEND,
        )
        assert float(want[4]) == 0 and float(got[4]) == 0, ("over", rank)
        for b in range(8):
            assert torch.equal(want[2][b], got[2][b]) and _bits(want[3][b], got[3][b]), ("slots", rank, b)
            assert _bits(want[0][b], got[0][b]) and _bits(want[1][b], got[1][b]), ("o / lse", rank, b)
