"""The save-once host cache (EngineConfig.host_cache_save_once: engine/host_arena.py, ModelRunner.*_so, HostTier's busy
skip and tags) against the write-through one, bit for bit.

The world is four stand-in ranks, attention TP 4 and CP 2: two row groups of two ranks, R = 2. Ranks 0 and 2 hold CP
index 0 and ranks 1 and 3 hold CP index 1; the ranks of one CP index hold identical bytes in every page, the two CP
indexes differ. Rank 0 drives (the real ModelRunner.kv_save_many / kv_load_many / kv_arena_init), the other three run the
real worker dispatch (`getattr(runner, "_" + name)(*args)`), each in its own thread behind a queue, as the calls'
shared-memory channel delivers them: so a rank can lag, and the generation fence and the completion board work for real,
over real shared-memory segments (cross-process mechanics: tests/test_host_arena.py).

  1. a restore under save-once ends in the same device caches on every rank as the write-through host cache, and those
     are the bytes each rank saved; the CP indexes differ (not vacuous);
  2. plants must be caught: a restore of the wrong slot, a restore routed to the other CP index's arena, and a slot that
     holds another page than the host cache looked up (its tag);
  3. a rank that lags a load keeps its slot busy, so the host cache does not evict it; with the busy rule off the reuse is
     a loud error on the lagging rank, not a silent wrong restore;
  4. a HostTier history (offloads that evict and reuse slots inside one offload, write-through, restores) ends the same;
  5. capacity: the same bytes per rank hold R times the pages; the per-page calls are refused, drops send nothing."""

from __future__ import annotations

import itertools
import os
import queue
import threading
import time
import types

import pytest
import torch

from trnrt.engine import host_arena as HA
from trnrt.engine import model_runner as mr
from trnrt.engine.host_cache import HostTier

NP = 24  # device pages per rank
WORLD, A, CP = 4, 4, 2
R = A // CP

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
    "so_possible",
    "kv_arena_init",
    "_kv_arena_init",
    "kv_arena_close",
    "so_active",
    "so_stats",
    "so_busy",
    "_kv_save_many_so",
    "_kv_load_many_so",
)
_tags = itertools.count()
# the stand-in caches' row shapes and dtypes: k bf16, v fp8, aux fp32, a token-slot state bf16
SHAPES = [((1, 64), torch.bfloat16), ((1, 32), torch.float8_e4m3fn), ((4,), torch.float32), ((128,), torch.bfloat16)]


def _rank(r: int, lps: int, seed: int):
    c = (r % A) % CP
    g = torch.Generator().manual_seed(seed * 100 + c)  # one stream a CP index: its ranks hold identical bytes

    def t(shape, dt):
        return torch.randn(NP * lps, *shape, generator=g).to(dt)

    (k, kd), (v, vd), (a, ad), (st, sd) = SHAPES
    ns = types.SimpleNamespace(
        lps=lps,
        k_caches=[t(k, kd)],
        v_caches=[t(v, vd)],
        aux_caches=[t(a, ad)],
        s_caches=[t(st, sd)],
        _host_kv={},
        _last_out=None,
        device=torch.device("cpu"),
        tp_send=None,
        model=types.SimpleNamespace(tp_rank=r, tp_size=WORLD, attn_tp=A, cp=CP, cp_rows=R),
    )
    for name in BOUND:
        setattr(ns, name, types.MethodType(getattr(mr.ModelRunner, name), ns))
    return ns


def _bits(c: torch.Tensor) -> torch.Tensor:
    return (
        c.view(torch.uint8)
        if c.dtype.itemsize == 1
        else c.view(torch.int16)
        if c.dtype.itemsize == 2
        else c.view(torch.int32)
    )


class World:
    """Four stand-in ranks; rank 0 is driven by the test thread, ranks 1-3 by worker threads running the worker dispatch."""

    def __init__(self, lps: int, seed: int, save_once: bool, cap: int = 0):
        self.lps = lps
        self.ranks = [_rank(r, lps, seed) for r in range(WORLD)]
        self.drv = self.ranks[0]
        self.queues: list[queue.Queue] = [queue.Queue() for _ in range(WORLD)]
        self.gates = [threading.Event() for _ in range(WORLD)]
        for g in self.gates:
            g.set()
        self.errors: list[tuple[int, BaseException]] = []
        self.sent: list = []
        self.threads = [threading.Thread(target=self._loop, args=(r,), daemon=True) for r in range(1, WORLD)]
        for t in self.threads:
            t.start()
        self.drv.tp_send = self._bcast
        self.tag = None
        if save_once:
            self.tag = f"t{os.getpid() % 100000}x{next(_tags)}"
            HA.cleanup_stale(self.tag, [HA.arena_name(self.tag, r) for r in range(WORLD)] + [HA.board_name(self.tag)])
            self.drv.kv_arena_init(self.tag, cap)
            self.sync()

    def _bcast(self, msg):
        self.sent.append(msg)
        for r in range(1, WORLD):
            self.queues[r].put(msg)

    def _loop(self, r: int):
        q = self.queues[r]
        while True:
            msg = q.get()
            try:
                if msg is None:
                    return
                self.gates[r].wait()
                name, _key, args = msg[:3]
                getattr(self.ranks[r], "_" + name)(*args)
            except BaseException as e:  # noqa: BLE001  recorded, the test decides
                self.errors.append((r, e))
            finally:
                q.task_done()

    def sync(self, skip=(), timeout: float = 60.0):
        end = time.time() + timeout
        for r in range(1, WORLD):
            while r not in skip and self.queues[r].unfinished_tasks:
                assert time.time() < end, f"rank {r} never finished its queue"
                time.sleep(0.002)

    def snapshot(self):
        return [[c.clone() for c in rk._paged_caches()] for rk in self.ranks]

    def close(self):
        for g in self.gates:
            g.set()
        for r in range(1, WORLD):
            self.queues[r].put(None)
        for t in self.threads:
            t.join(10)
        for rk in self.ranks:
            st = getattr(rk, "_so", None)
            if st is not None:
                st.close()


@pytest.fixture
def world():
    made = []

    def make(lps=8, seed=1, save_once=True, cap=16):
        w = World(lps, seed, save_once, cap)
        made.append(w)
        return w

    yield make
    for w in made:
        w.close()
    for w in made:
        if w.tag is not None:  # nothing of ours is left behind
            for n in [HA.arena_name(w.tag, r) for r in range(WORLD)] + [HA.board_name(w.tag)]:
                with pytest.raises(TimeoutError):
                    HA._Seg.attach(n, timeout=0.05)


def _rows(w: World, pages):
    """Per rank, per cache: the rows of `pages` in the given order."""
    lps = w.lps
    return [
        [torch.cat([c[p * lps : (p + 1) * lps] for p in pages]).clone() for c in rk._paged_caches()] for rk in w.ranks
    ]


def _zero(w: World, pages):
    for rk in w.ranks:
        for c in rk._paged_caches():
            for p in pages:
                c[p * w.lps : (p + 1) * w.lps].zero_()


def history(w: World, src, dst, slots):
    """Save `src` pages to `slots`, free the source pages (zero), restore the slots into `dst`; return the bytes each
    rank saved."""
    saved = _rows(w, src)
    w.drv.kv_save_many(list(slots), list(src))
    w.sync()
    _zero(w, src)
    w.drv.kv_load_many(list(slots), list(dst))
    w.sync()
    return saved


# (source pages, destination pages, host slots)
CASES = {
    "contiguous": ([2, 3, 4, 5], [10, 11, 12, 13], [0, 1, 2, 3]),
    "fragmented": ([1, 7, 3, 9, 5, 11], [14, 2, 20, 4, 17, 8], [0, 1, 2, 3, 4, 5]),
    "single": ([6], [18], [1]),
    "mixed": ([1, 2, 3, 8, 9, 12, 13], [15, 16, 17, 19, 20, 22, 4], [3, 0, 5, 2, 4, 1, 6]),
    "owner0-only": ([2, 3, 4, 5], [10, 11, 12, 13], [0, 2, 4, 6]),
    "owner1-only": ([2, 3, 4, 5], [10, 11, 12, 13], [1, 3, 5, 7]),
    # slots past one arena's capacity (16 over two row groups = 8 each): a wrong arena index collides here only
    "high-slots": ([2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17], [15, 8, 0, 9, 14, 1, 10, 7]),
}


@pytest.mark.parametrize("lps", [8, 4])  # full pages, and context parallelism's lps rows a page
@pytest.mark.parametrize("case", sorted(CASES))
def test_a_restore_under_save_once_ends_in_the_caches_write_through_leaves(world, lps, case):
    src, dst, slots = CASES[case]
    wt, so = world(lps, 1, save_once=False), world(lps, 1, save_once=True, cap=16)
    history(wt, src, dst, slots)
    saved_so = history(so, src, dst, slots)
    assert not wt.errors and not so.errors, (wt.errors, so.errors)
    got_wt, got_so = wt.snapshot(), so.snapshot()
    for r in range(WORLD):
        for i, (a, b) in enumerate(zip(got_wt[r], got_so[r])):
            assert torch.equal(_bits(a), _bits(b)), (
                f"rank {r} cache {i}: save-once restored other bytes than write-through"
            )
    back = _rows(so, dst)  # in dst order; saved in src order, and the batch pairs src[j] with dst[j]
    for r in range(WORLD):
        for a, b in zip(saved_so[r], back[r]):
            assert torch.equal(_bits(a), _bits(b)), f"rank {r}: the restored rows are not the rows it saved"
    assert any(_bits(c).any() for c in back[0])  # not vacuous: bytes moved
    assert any(not torch.equal(_bits(a), _bits(b)) for a, b in zip(back[0], back[1]))  # the CP indexes differ
    assert all(torch.equal(_bits(a), _bits(b)) for a, b in zip(back[0], back[2]))  # the replicas of one CP index agree
    assert so.drv.so_stats()["load_pages"] == len(slots) and so.drv.so_stats()["load_peer_pages"] == sum(
        s % R != 0 for s in slots
    )


def test_a_restore_of_the_wrong_slot_ends_different(world):
    src, dst, slots = CASES["mixed"]
    wt, bad = world(8, 3, save_once=False), world(8, 3, save_once=True)
    history(wt, src, dst, slots)
    saved = _rows(bad, src)
    bad.drv.kv_save_many(list(slots), list(src))
    bad.sync()
    _zero(bad, src)
    swapped = list(slots)
    swapped[0], swapped[1] = swapped[1], swapped[0]  # the slot of page 0 and of page 1 swapped
    bad.drv.kv_load_many(swapped, list(dst))
    bad.sync()
    assert not bad.errors
    for r in range(WORLD):
        assert any(not torch.equal(_bits(a), _bits(b)) for a, b in zip(wt.snapshot()[r], bad.snapshot()[r])), (
            f"rank {r}: the planted swap went unseen"
        )
    assert any(not torch.equal(_bits(a), _bits(b)) for a, b in zip(saved[0], _rows(bad, dst)[0]))


def test_a_restore_read_from_the_other_cp_indexs_arena_ends_different(world, monkeypatch):
    src, dst, slots = CASES["fragmented"]
    wt = world(8, 4, save_once=False)
    history(wt, src, dst, slots)
    # every rank believes the peers of CP index 0: a rank of CP index 1 reads CP index 0's bytes for the row group it does
    # not own
    monkeypatch.setattr(HA, "peers", lambda rank, world, attn_tp, cp: [g * cp for g in range(attn_tp // cp)])
    bad = world(8, 4, save_once=True)
    history(bad, src, dst, slots)
    got, want = bad.snapshot(), wt.snapshot()
    assert any(not torch.equal(_bits(a), _bits(b)) for r in (1, 3) for a, b in zip(got[r], want[r])), (
        "the planted CP-index misrouting went unseen"
    )
    assert all(torch.equal(_bits(a), _bits(b)) for r in (0, 2) for a, b in zip(got[r], want[r]))


def test_a_restore_whose_slot_holds_another_page_is_refused_before_any_copy(world):
    """The tags: a save stamps the chain hash it was offloaded under; a restore that expects another one raises on every
    rank before a byte is copied (the host cache's map drifted from what the slot holds)."""
    w = world(8, 15, save_once=True, cap=16)
    w.drv.kv_save_many([1, 2], [2, 3], tags=[HA.tag_of(11), HA.tag_of(22)])
    w.sync()
    before = w.snapshot()
    with pytest.raises(RuntimeError, match="does not hold the page"):
        w.drv.kv_load_many([1, 2], [10, 11], tags=[HA.tag_of(11), HA.tag_of(33)])
    w.sync()
    assert {r for r, e in w.errors if "does not hold the page" in str(e)} == {1, 2, 3}
    assert all(torch.equal(_bits(a), _bits(b)) for r in range(WORLD) for a, b in zip(before[r], w.snapshot()[r]))
    w.errors.clear()
    w.drv.kv_load_many([1, 2], [10, 11], tags=[HA.tag_of(11), HA.tag_of(22)])  # the right tags: restored
    w.sync()
    assert not w.errors and w.drv.so_stats()["tag_checked"] == 2


def test_a_load_waits_for_a_lagging_owners_save_and_reads_its_bytes(world):
    """The generation fence at the runner: the owner row group (ranks 2 and 3 own the odd slots) has not run the save when
    the driver reaches the load; the driver must wait, not read the fresh arena's zeros (a timer releases the owners)."""
    w = world(8, 11, save_once=True, cap=16)
    src, dst, slots = [2, 3, 4], [10, 11, 12], [1, 3, 5]  # all owned by row group 1
    saved = _rows(w, src)
    w.gates[2].clear()
    w.gates[3].clear()
    w.drv.kv_save_many(slots, src)
    w.sync(skip=(2, 3))
    threading.Timer(0.3, lambda: [w.gates[2].set(), w.gates[3].set()]).start()
    t0 = time.time()
    w.drv.kv_load_many(slots, dst)
    waited = time.time() - t0
    w.sync()
    assert waited >= 0.25, f"the load did not wait for the owner's save ({waited:.3f} s)"
    assert not w.errors, w.errors
    back = _rows(w, dst)
    for r in range(WORLD):
        for a, b in zip(saved[r], back[r]):
            assert torch.equal(_bits(a), _bits(b)), f"rank {r}"
    assert any(_bits(c).any() for c in back[0])


def test_a_slot_rewritten_under_a_copy_is_an_error_and_not_a_torn_restore(world):
    """The owner begins the next generation between the reader's wait for its generation and its copies (the _settle hook
    runs exactly there): the load raises. The busy rule makes it impossible in service; this is the net under it."""
    w = world(8, 13, save_once=True, cap=16)
    slots, src, dst = [1, 3], [2, 3], [10, 11]  # both owned by row group 1 (ranks 2, 3)
    w.drv.kv_save_many(slots, src)
    w.sync()
    gens = [w.drv._so_gen[s] for s in slots]
    owner = w.ranks[2]._so
    w.drv._settle = lambda: owner.begin(slots[0], gens[0] + 1)
    with pytest.raises(RuntimeError, match="rewritten during the copy"):
        w.drv._kv_load_many_so(slots, dst, gens, 999)


def test_the_owner_marks_a_slot_before_it_writes_the_payload(world, monkeypatch):
    seen = []
    orig = HA.HostStore.commit

    def spy(self, slot, gen, *rest):
        seen.append((slot, gen, int(self.arenas[self.mine].gen[slot // self.R])))  # the cell as the commit finds it
        return orig(self, slot, gen, *rest)

    monkeypatch.setattr(HA.HostStore, "commit", spy)
    w = world(8, 14, save_once=True, cap=16)
    w.drv.kv_save_many([0, 1, 2, 3], [2, 3, 4, 5])
    w.sync()
    assert len(seen) == 8 and all(c == (HA.BUSY | g) for _, g, c in seen), seen  # 4 ranks, 2 owned slots each


def _tier_slots(w: World, n: int):
    """A batched HostTier over the driver, full: `n` pages of known ids saved from device pages 1 .. n (all ranks done)."""
    tier = HostTier(w.drv, page_size=8, capacity_pages=n, capacity_ckpts=1, batch=True)
    ids = list(range(100, 100 + 8 * n))
    tier.offload(ids, 0, list(range(1, 1 + n)))
    w.sync()
    return tier, ids


def test_a_lagging_rank_keeps_its_slot_busy_and_the_host_cache_does_not_evict_it(world):
    w = world(8, 5, save_once=True, cap=4)
    tier, ids = _tier_slots(w, 4)
    w.gates[3].clear()  # rank 3 lags: it will not run the next command until released
    slots = tier.lookup(ids, 0, 4)
    assert len(slots) == 4
    tier.load(slots, [10, 11, 12, 13])
    w.sync(skip=(3,))
    assert all(w.drv.so_busy(s) for s in slots), "a load rank 3 has not finished must keep its slots busy"
    assert tier._slot() is None, "the host cache evicted a slot a lagging rank still reads"
    w.gates[3].set()
    w.sync()
    assert not any(w.drv.so_busy(s) for s in slots)
    assert tier._slot() is not None  # released: evictable again
    assert not w.errors


def test_without_the_busy_rule_a_reused_slot_is_a_loud_error_on_the_lagging_rank(world):
    w = world(8, 6, save_once=True, cap=4)
    tier, ids = _tier_slots(w, 4)
    w.drv.so_busy = lambda slot: False  # the plant: the pin-through-copy rule off
    w.gates[3].clear()
    slots = tier.lookup(ids, 0, 4)
    tier.load(slots, [10, 11, 12, 13])
    w.sync(skip=(3,))
    victim = tier._slot()
    assert victim is not None  # evicted though rank 3 still has to read it
    w.drv.kv_save_many([victim], [20])  # reused: the owner stamps the next generation
    w.sync(skip=(3,))
    w.gates[3].set()
    w.sync()
    msgs = [str(e) for r, e in w.errors if r == 3]
    assert any("reused under a pending load" in m for m in msgs), (msgs, w.errors)


def _flush_under_a_lagging_load(w: World, plant: bool):
    """A full host cache; rank 3 lags a restore of slots 0 and 1; the cache is flushed (HostTier.clear) and a new path
    offloaded. plant: clear() as it was, every slot free at once."""
    tier, ids = _tier_slots(w, 4)
    w.gates[3].clear()
    slots = tier.lookup(ids, 0, 2)
    assert slots == [0, 1]
    tier.load(slots, [10, 11])
    w.sync(skip=(3,))
    tier.clear()
    if plant:
        tier.free_slots, tier.held = list(range(3, -1, -1)), set()
    tier.offload(list(range(700, 700 + 8 * 4)), 0, [12, 13, 14, 15])
    w.sync(skip=(3,))
    return tier, slots


def test_a_flush_keeps_the_slots_a_lagging_rank_still_reads_out_of_use(world):
    """A cache flush while rank 3 still copies a restore: the slots it reads are held, not freed, so the next offload
    cannot stamp a newer generation under the copy (its other slots are used and reused); they come back once it is done."""
    w = world(8, 16, save_once=True, cap=4)
    tier, slots = _flush_under_a_lagging_load(w, plant=False)
    assert tier.held == set(slots) and set(tier.entries.values()).isdisjoint(slots)
    assert tier.saved == 8  # the four pages of the new path all went to the two other slots
    w.gates[3].set()
    w.sync()
    assert not w.errors, w.errors
    assert tier._slot() in slots and not tier.held  # rank 3 is done: the held slots are back


def test_plant_a_flush_that_frees_every_slot_at_once_fails_the_lagging_rank(world):
    w = world(8, 17, save_once=True, cap=4)
    _flush_under_a_lagging_load(w, plant=True)
    w.gates[3].set()
    w.sync()
    msgs = [str(e) for r, e in w.errors if r == 3]
    assert any("reused under a pending load" in m for m in msgs), (msgs, w.errors)


def _tier_history(w: World):
    """tests/test_host_cache_batch.py's history on a world: offloads that fill the host cache and drop and reuse slots
    inside one offload, write-through, restores."""
    tier = HostTier(w.drv, page_size=8, capacity_pages=5, capacity_ckpts=1, batch=True)
    ids = list(range(100, 100 + 8 * 12))
    tier.offload(ids, 0, [1, 2, 3])
    tier.offload(ids, 0, [1, 2, 3, 4, 5, 6])
    other = list(range(500, 500 + 8 * 4))
    tier.offload(other, 0, [8, 9, 10, 11])
    slots = tier.lookup(other, 0, 4)
    tier.load(slots, [15, 16, 18][: len(slots)])
    slots = tier.lookup(ids, 3, 6)
    tier.load(slots, [20, 21, 22][: len(slots)])
    w.sync()
    return tier


@pytest.mark.parametrize("lps", [8, 4])
def test_a_host_cache_history_ends_the_same_under_save_once_and_write_through(world, lps):
    wt, so = world(lps, 7, save_once=False), world(lps, 7, save_once=True, cap=5)
    t_wt, t_so = _tier_history(wt), _tier_history(so)
    assert not wt.errors and not so.errors, (wt.errors, so.errors)
    a, b = wt.snapshot(), so.snapshot()
    for r in range(WORLD):
        for i, (x, y) in enumerate(zip(a[r], b[r])):
            assert torch.equal(_bits(x), _bits(y)), f"rank {r} cache {i}"
    assert (t_wt.saved, t_wt.restored, t_wt.dropped) == (t_so.saved, t_so.restored, t_so.dropped)
    assert t_so.restored > 0 and t_so.dropped > 0  # (the history evicts and restores)
    assert so.drv.so_stats()["tag_checked"] == t_so.restored  # every restored slot's tag was checked


def test_the_same_bytes_a_rank_hold_r_times_the_pages(world):
    n = 6  # pages a write-through host cache keeps in each rank's own heap
    w = world(8, 8, save_once=True, cap=n * R)
    st = w.drv._so
    raw = sum(c[:8].numel() * c.element_size() for c in w.drv._paged_caches())  # one page's bytes on a rank
    arena = st.arenas[st.mine]
    assert st.R == R and arena.capacity == n  # n slots in this rank's arena, n * R in the CP index: R times the pages
    assert raw <= arena.slot_bytes < raw + 64 * len(w.drv._paged_caches())  # per-cache alignment is the only padding
    assert arena.capacity * arena.slot_bytes < n * raw * 1.05  # the bytes of the write-through host cache's n pages
    for r in range(WORLD):
        assert w.ranks[r]._so.capacity == n * R


def test_with_save_once_off_nothing_is_busy_and_the_host_cache_evicts_as_before(world):
    """so_busy is a method of every runner and the host cache asks it on each eviction: with save-once off it must answer
    False, not raise."""
    w = world(8, 10, save_once=False)
    assert w.drv.so_busy(0) is False and not w.drv.so_active()
    tier = HostTier(w.drv, page_size=8, capacity_pages=2, capacity_ckpts=1, batch=True)
    tier.offload(list(range(100, 100 + 8 * 4)), 0, [1, 2, 3, 4])  # four pages into two slots: evicts
    w.sync()
    assert tier.dropped > 0 and not w.errors
    assert w.sent and all(m[0] in ("kv_save_many", "kv_drop") for m in w.sent), [m[0] for m in w.sent]


def test_closing_the_runner_unlinks_its_arena_and_the_board(world):
    w = world(8, 12, save_once=True, cap=4)
    names = [HA.arena_name(w.tag, 0), HA.board_name(w.tag)]
    for n in names:
        HA._Seg.attach(n, timeout=1.0).close()  # there while it runs
    w.drv.kv_arena_close()
    for n in names:
        with pytest.raises(TimeoutError):
            HA._Seg.attach(n, timeout=0.05)
    assert w.drv._so is None and w.drv.so_busy(0) is False
    w.drv.kv_arena_close()  # idempotent


def test_the_per_page_calls_are_refused_and_drops_send_nothing_under_save_once(world):
    w = world(8, 9, save_once=True, cap=4)
    sent = len(w.sent)
    with pytest.raises(RuntimeError, match="in batches only"):
        w.drv.kv_save(0, 1)
    with pytest.raises(RuntimeError, match="in batches only"):
        w.drv.kv_load(0, 1)
    w.drv.kv_drop(0)
    assert len(w.sent) == sent  # nothing broadcast: the next save of the slot overwrites it
    w.drv.kv_save_many([], [])
    w.drv.kv_load_many([], [])
    assert len(w.sent) == sent  # empty batches send nothing either


def test_save_once_needs_row_groups():
    r = _rank(0, 8, 1)
    assert r.so_possible()
    r.model.cp_rows = 1
    assert not r.so_possible()
