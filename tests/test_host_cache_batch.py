"""Batched host-cache moves (EngineConfig.host_cache_batch: engine/host_cache.py, ModelRunner.kv_load_many /
kv_save_many): a restore's and an eviction's pages in one broadcast each, every cache copied once per run of consecutive
device pages. Against the per-page path, bit for bit: the device caches a batched restore writes (contiguous, fragmented
and single pages, full pages and context parallelism's lps-row pages), the host copies a batched save keeps (each its own
storage), and a HostTier history (evictions that drop and reuse slots inside one offload, write-through, restores) ending
in the same caches, host copies and counters with fewer messages. Plus the run cutter, the one broadcast per batch, the
ascending page pool and the host cache's checkpoint share."""

from __future__ import annotations

import inspect
import itertools
import types

import pytest
import torch

from trnrt.engine import engine as E
from trnrt.engine import model_runner as mr
from trnrt.engine.host_cache import HostTier, page_run_count
from trnrt.engine.kv_pool import PagePool

NP = 24  # device pages

BOUND = (
    "_paged_caches",
    "_settle",
    "_kv_save",
    "_kv_load",
    "_kv_drop",
    "_kv_save_many",
    "_kv_load_many",
    "kv_save",
    "kv_load",
    "kv_drop",
    "kv_save_many",
    "kv_load_many",
    "state_save",
    "state_drop",
)


def _runner(lps: int, seed: int | None = None, sent=None):
    """A runner with four paged caches of different shapes and dtypes (k bf16, v fp8 e4m3, aux fp32, a token-slot state
    bf16), random when seeded, else zero; tp_send records the broadcasts. State checkpoints are not moved here."""
    g = torch.Generator().manual_seed(seed if seed is not None else 0)

    def t(shape, dt):
        x = torch.randn(NP * lps, *shape, generator=g) if seed is not None else torch.zeros(NP * lps, *shape)
        return x.to(dt)

    r = types.SimpleNamespace(
        lps=lps,
        k_caches=[t((1, 64), torch.bfloat16)],
        v_caches=[t((1, 32), torch.float8_e4m3fn)],
        aux_caches=[t((4,), torch.float32)],
        s_caches=[t((128,), torch.bfloat16)],
        _host_kv={},
        _host_ckpt={},
        _last_out=None,
        device=torch.device("cpu"),
        tp_send=sent.append if sent is not None else None,
    )
    for name in BOUND:
        setattr(r, name, types.MethodType(getattr(mr.ModelRunner, name), r))
    r._state_save = lambda slot, row: None
    r._state_drop = lambda slot: None
    return r


def _bits(r) -> list[torch.Tensor]:
    return [
        c.view(torch.uint8)
        if c.dtype.itemsize == 1
        else c.view(torch.int16)
        if c.dtype.itemsize == 2
        else c.view(torch.int32)
        for c in r._paged_caches()
    ]


def _same(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(_bits(a), _bits(b)))


CASES = {
    "contiguous": [5, 6, 7, 8],
    "fragmented": [9, 2, 14, 3, 4, 11],
    "single": [7],
    "mixed": [1, 2, 3, 10, 12, 13, 20],
}


@pytest.mark.parametrize("lps", [8, 4])  # full pages, and context parallelism's lps rows a page (page size 8, cp 2)
@pytest.mark.parametrize("case", sorted(CASES))
def test_a_batched_restore_writes_what_the_per_page_restores_write(lps, case):
    src = _runner(lps, seed=1)
    pages_src = [p for p in range(NP) if p % 3 != 0][: len(CASES[case])]
    for slot, p in enumerate(pages_src):
        src._kv_save(slot, p)
    one, many = _runner(lps), _runner(lps)
    dest = CASES[case]
    for slot, p in enumerate(dest):
        one._host_kv[slot] = src._host_kv[slot]
        many._host_kv[slot] = src._host_kv[slot]
        one._kv_load(slot, p)
    many._kv_load_many(list(range(len(dest))), dest)
    assert _same(one, many)
    assert any(c.abs().sum() > 0 for c in many._paged_caches() if c.dtype != torch.float8_e4m3fn)  # (not vacuous)


@pytest.mark.parametrize("lps", [8, 4])
@pytest.mark.parametrize("case", sorted(CASES))
def test_a_batched_save_keeps_what_the_per_page_saves_keep(lps, case):
    src = _runner(lps, seed=2)
    pages = CASES[case]
    one = {}
    for slot, p in enumerate(pages):
        src._kv_save(slot, p)
        one[slot] = src._host_kv.pop(slot)
    src._kv_save_many(list(range(len(pages))), pages)
    many = src._host_kv
    assert sorted(many) == sorted(one)
    for slot in one:
        for a, b in zip(one[slot], many[slot]):
            assert a.dtype == b.dtype and a.shape == b.shape
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            assert b.untyped_storage().nbytes() == b.numel() * b.element_size()  # its own copy, not a view of the run


def test_the_run_cutter_groups_consecutive_pages_in_page_order():
    pages = [9, 2, 14, 3, 4, 11, 10]
    runs = mr.page_runs(pages)
    assert [(lo, n) for lo, n, _ in runs] == [(2, 3), (9, 3), (14, 1)]
    assert [[pages[j] for j in idx] for _, _, idx in runs] == [[2, 3, 4], [9, 10, 11], [14]]
    assert page_run_count(pages) == 3 == len(runs)
    with pytest.raises(ValueError, match="names a page twice"):
        mr.page_runs([3, 4, 3])


def test_one_broadcast_carries_the_batch_and_the_workers_dispatch_it():
    sent = []
    r = _runner(8, seed=3, sent=sent)
    r.kv_save_many([0, 1, 2], [4, 5, 9])
    r.kv_load_many([0, 1, 2], [10, 11, 12])
    assert sent == [("kv_save_many", None, [[0, 1, 2], [4, 5, 9]]), ("kv_load_many", None, [[0, 1, 2], [10, 11, 12]])]
    src = inspect.getsource(mr.ModelRunner.serve)
    assert '"kv_save_many": self._kv_save_many' in src and '"kv_load_many": self._kv_load_many' in src


def _tier_run(batch: bool):
    """One HostTier history: offloads that fill the host cache and drop and reuse slots inside one offload,
    write-through, and restores into fresh pages. Returns (runner, tier, messages)."""
    sent = []
    r = _runner(8, seed=5, sent=sent)
    tier = HostTier(r, page_size=8, capacity_pages=5, capacity_ckpts=1, batch=batch)
    ids = list(range(100, 100 + 8 * 12))
    tier.offload(ids, 0, [1, 2, 3])  # three saves
    tier.offload(ids, 0, [1, 2, 3, 4, 5, 6])  # write-through for 0..2; 3..5 fill it, then drop the oldest
    other = list(range(500, 500 + 8 * 4))
    tier.offload(other, 0, [8, 9, 10, 11])  # drops and reuses inside one offload
    slots = tier.lookup(other, 0, 4)
    assert len(slots) == 4  # every page of the last offload is held
    tier.load(slots, [15, 16, 18, 19])  # a restore into fresh, fragmented pages: one page per slot
    slots = tier.lookup(ids, 3, 6)
    tier.load(slots, [20, 21, 22][: len(slots)])
    return r, tier, sent


def test_a_host_cache_history_ends_the_same_batched_or_per_page():
    a, ta, sent_a = _tier_run(False)
    b, tb, sent_b = _tier_run(True)
    assert _same(a, b)
    assert sorted(a._host_kv) == sorted(b._host_kv)
    for s in a._host_kv:
        assert all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a._host_kv[s], b._host_kv[s]))
    assert (
        (ta.saved, ta.restored, ta.dropped) == (tb.saved, tb.restored, tb.dropped)
        and ta.restored > 0
        and ta.dropped > 0
    )
    assert list(ta.entries.items()) == list(tb.entries.items())
    assert len(sent_b) < len(sent_a)  # the point: fewer messages
    assert {m[0] for m in sent_b} <= {"kv_save_many", "kv_load_many", "kv_drop"}
    assert tb.restore_runs < ta.restore_runs and ta.restores == tb.restores >= 1


@pytest.mark.parametrize("batch", [False, True])
def test_a_host_cache_smaller_than_one_offload_keeps_the_last_pages(batch):
    """Two host slots, one offload of four pages in descending page order: the per-page order saves 11 and 10, drops slot 0
    for 9 and slot 1 for 8. Batched, the pending saves go out before each drop (HostTier._slot flushes them); sent after
    the drops in page order instead, slot 0 would end holding page 11, not 9."""
    r = _runner(8, seed=6, sent=[])
    tier = HostTier(r, page_size=8, capacity_pages=2, capacity_ckpts=1, batch=batch)
    tier.offload(list(range(300, 300 + 8 * 4)), 0, [11, 10, 9, 8])
    want = {s: [c[p * 8 : (p + 1) * 8] for c in r._paged_caches()] for s, p in ((0, 9), (1, 8))}
    for s, ts in want.items():
        assert all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(r._host_kv[s], ts)), s
    assert tier.saved == 4 and tier.dropped == 2


def _failing_once(r):
    """r.kv_save_many raises on its next call, then is r's own again."""
    real = r.kv_save_many

    def fail(*a, **k):
        r.kv_save_many = real
        raise RuntimeError("planted: the batch's device-to-host copy failed")

    r.kv_save_many = fail


def test_a_batched_save_that_fails_publishes_none_of_its_pages_and_frees_their_slots():
    r = _runner(8, seed=7)
    tier = HostTier(r, page_size=8, capacity_pages=4, capacity_ckpts=1, batch=True)
    ids = list(range(300, 300 + 8 * 3))
    _failing_once(r)
    with pytest.raises(RuntimeError, match="planted"):
        tier.offload(ids, 0, [5, 6, 7])
    assert not tier.entries and not tier.slot_hash and sorted(tier.free_slots) == [0, 1, 2, 3] and tier.saved == 0
    assert tier.lookup(ids, 0, 3) == []  # nothing a restore could take: no slot without its page
    tier.offload(ids, 0, [5, 6, 7])  # the next eviction of the same pages keeps them
    slots = tier.lookup(ids, 0, 3)
    assert len(slots) == 3
    want = [c[5 * 8 : 8 * 8].clone() for c in r._paged_caches()]
    tier.load(slots, [10, 11, 12])  # every slot it found holds its page
    for c, w in zip(r._paged_caches(), want):
        assert torch.equal(c[10 * 8 : 13 * 8].view(torch.uint8), w.view(torch.uint8))


def test_a_failed_save_flushed_by_an_eviction_keeps_the_entry_it_would_have_evicted():
    """Two slots full with path X; path Y's second page evicts X's second (slot 1), and the flush of Y's first page, which
    goes before that drop, fails: X's page stays mapped with its payload, Y's page is not mapped, slot 0 is free."""
    r = _runner(8, seed=8)
    tier = HostTier(r, page_size=8, capacity_pages=2, capacity_ckpts=1, batch=True)
    x, y = list(range(100, 116)), list(range(900, 916))
    tier.offload(x, 0, [3, 4])
    x1 = tier.hashes(x, 2)[1]
    kept = [t.clone() for t in r._host_kv[1]]
    _failing_once(r)
    with pytest.raises(RuntimeError, match="planted"):
        tier.offload(y, 0, [8, 9])  # Y's first page takes slot 0 (X's first dropped), then the flush fails
    assert list(tier.entries.items()) == [(x1, 1)] and tier.free_slots == [0] and tier.saved == 2
    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(r._host_kv[1], kept))
    assert tier.lookup(y, 0, 2) == []


def test_an_ascending_pool_hands_out_runs_and_takes_runs_back():
    p = PagePool(32, ascending=True)
    a = p.alloc(6)
    assert a == sorted(a) and [y - x for x, y in itertools.pairwise(a)] == [1] * 5
    p.alloc(4)
    p.free(a[1:5])  # an evicted leaf's run
    c = p.alloc(4)
    assert c == a[1:5] and len(mr.page_runs(c)) == 1
    q = PagePool(32)  # the default order, last in first out: a freed run comes back reversed
    d = q.alloc(6)
    q.free(d[1:5])
    assert q.alloc(4) == list(reversed(d[1:5])) and q.alloc(0) == []


def test_an_ascending_pool_hands_out_the_lowest_free_page_one_page_at_a_time():
    """A decode takes its pages one at a time: each is the lowest free id too, not the last one freed."""
    p = PagePool(32, ascending=True)
    assert p.alloc(8) == list(range(1, 9))
    p.free([3, 4, 5])  # an evicted run, put back on top of the free list
    assert [p.alloc(1)[0] for _ in range(3)] == [3, 4, 5]
    p.free([7, 2])
    assert p.alloc(1) == [2] and p.alloc(2) == [7, 9]


def test_the_engine_builds_its_pool_in_the_configured_order():
    src = inspect.getsource(E.LLMEngine._finish_init)  # (__init__ runs it past the shards)
    assert "PagePool(num_pages, ascending=cfg.pool_ascending)" in src
    assert "batch=cfg.host_cache_batch" in src and "cfg.host_cache_ckpt_frac" in src


@pytest.mark.parametrize(
    ("frac", "want"),
    [(0.5, (5, 7)), (0.6, (6, 5)), (0.1, (1, 12))],
)
def test_the_checkpoints_take_their_share_of_the_host_cache(frac, want):
    """10-byte checkpoint rows and 7-byte pages in 100 bytes: frac of the budget in whole rows (at least one), the rest
    in whole pages. 0.5 is the even split the host cache always had."""
    assert E.host_tier_split(100, 10, 7, frac) == want
    with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
        E.host_tier_split(100, 10, 7, 1.0)
