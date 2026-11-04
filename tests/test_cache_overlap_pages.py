"""A KV page is never owned twice under overlap scheduling. Host-only: the real Scheduler, RadixCache, PagePool and
HostTier, driven in the engine's overlap order (schedule step N + 1, advance it, then commit step N), with token ids
for samples and no model.

Two deterministic cases, both with (prompt length + 1) a page multiple, so the newest token of step 2's decode fills a
page while step 1 is still unread (1 in 256 prompts at the served page size):
- step 1 commits its prefill (whose in-chunk checkpoint attaches) after step 2's decode has advanced past it: no
  placeholder may enter a radix key and the tree must free every page exactly once (Scheduler._cacheable);
- an abort while a step is in flight: _drop_pending cuts num_computed below the page the last prefill chunk's _cache
  gave the tree, and _release must not free that page as its tail while the tree still holds it.

And a randomized one in the served configuration (state checkpoints, the host cache with running-path offload,
free-dominated, reserve admission, 8 seats, agentic multi-turn sessions with re-rendered replies, client aborts, small
pools), checking after every scheduler call that no page is in two nodes, no tree or running page is free, the pool's
pages add up, and every checkpoint row is accounted for.
"""

from __future__ import annotations

import collections
import random

import pytest

from trnrt.engine.host_cache import HostTier
from trnrt.engine.kv_pool import PagePool
from trnrt.engine.radix_cache import RadixCache
from trnrt.engine.request import Request, SamplingParams, Status
from trnrt.engine.scheduler import PENDING, NeedSync, Scheduler, SchedulerConfig

PS = 4


class Moves:
    """The host cache's copies (engine/host_cache.py), as bookkeeping only."""

    def kv_save(self, slot, page):
        pass

    def kv_load(self, slot, page):
        pass

    def kv_drop(self, slot):
        pass

    def state_save(self, slot, row):
        pass

    def state_load(self, slot, row):
        pass

    def state_drop(self, slot):
        pass


class Rows:
    """StatePool's checkpoint rows (alloc_ckpt / free_ckpt), with its double-free check."""

    def __init__(self, n: int):
        self.n = n
        self.free = list(range(100, 100 + n))

    def alloc_ckpt(self):
        return self.free.pop() if self.free else None

    def free_ckpt(self, row: int) -> None:
        assert row not in self.free, f"double free of checkpoint row {row}"
        self.free.append(row)


def nodes(radix):
    st = [radix.root]
    while st:
        n = st.pop()
        yield n
        st.extend(n.children.values())


def _depth(node) -> int:
    d = 0
    while node.parent is not None:
        d, node = d + len(node.key), node.parent
    return d


def check(s: Scheduler) -> None:
    pool, owner = s.pool, {}
    ckpts = 0
    for n in nodes(s.radix):
        assert PENDING not in n.key, "a placeholder in a radix key"
        ckpts += n.ckpt is not None
        for p in n.pages:
            assert p not in owner, f"page {p} in two radix nodes"
            assert not pool._is_free[p], f"radix page {p} is free in the pool"
            owner[p] = n
    private = collections.Counter(p for r in s.running for p in r.pages if p not in owner)
    for r in s.running:
        assert not any(pool._is_free[p] for p in r.pages), f"{r.rid} holds a free page"
    assert all(c == 1 for c in private.values()), "a private page held by two requests"
    assert pool.num_free + len(owner) + len(private) == pool.num_usable, "pages leaked or counted twice"
    pending = sum(len(r.ckpt_pending) for r in s.running)
    assert len(s.states.free) + ckpts + pending == s.states.n, "a checkpoint row leaked or is held twice"


def scheduler(num_pages=64, seats=8, budget=16, rows=8, host=None) -> Scheduler:
    pool = PagePool(num_pages)
    radix = RadixCache(pool, PS)
    cfg = SchedulerConfig(
        page_size=PS, max_num_seqs=seats, max_prefill_tokens=budget, max_model_len=400, admission_decode_tokens=8
    )
    s = Scheduler(cfg, pool, radix)
    s.states = Rows(rows)
    radix.free_ckpt = s.states.free_ckpt
    if host is not None:
        tier = HostTier(Moves(), PS, *host)
        radix.offload, radix.offload_ckpt, s.host = tier.offload, tier.offload_ckpt, tier
    return s


def tokens(plan, t=11):
    return [t] * len(plan.seqs())


def launch(s: Scheduler):
    """schedule + advance, as the engine's overlapped step does before it reads the step in flight back."""
    plan = s.schedule()
    for e in plan.prefills:  # the engine launches the restore copies
        if e.restore is not None:
            e.req.ckpt_restore = None
    s.advance(plan)
    return plan


def test_a_prefill_committed_after_the_next_decode_advanced_never_puts_a_placeholder_in_a_radix_key():
    """Prompt 15 (16 a page multiple), one step for all of it with its prompt-end checkpoint at 12 written in the
    chunk (one entry, mid). Step 2's decode advances num_computed to 16 before step 1 commits."""
    s = scheduler()
    r = Request("a", list(range(20, 35)), SamplingParams(max_new_tokens=4, ignore_eos=True))
    s.add(r)
    p1 = launch(s)
    assert [(e.start, e.end, e.sample) for e in p1.prefills] == [(0, 15, True)] and p1.prefills[0].mid[0] == 12
    s.in_flight = True
    p2 = launch(s)
    assert r.num_computed == 16 and r.token_ids[15:] == [PENDING, PENDING]
    s.commit(p1, tokens(p1))
    check(s)
    # the prompt-end checkpoint (12, anchor), and step 2's decode checkpoint (16), whose page step 1 gave the tree
    assert sorted((d, n.anchor) for n in nodes(s.radix) if n.ckpt is not None for d in [_depth(n)]) == [
        (12, True),
        (16, False),
    ]
    s.commit(p2, tokens(p2))
    check(s)
    while r.status is Status.RUNNING:
        s.in_flight = False
        s.commit(launch(s), [12])
    check(s)
    s.radix.evict(s.radix.total_pages())
    assert s.pool.num_free == s.pool.num_usable and len(s.states.free) == s.states.n


def test_an_abort_in_flight_never_frees_a_radix_page():
    """Prompt 15 in one chunk; step 2's decode advanced num_computed to 16, so step 1's commit gives the tree page 3
    (positions 12 .. 15). An abort before step 2 commits cuts num_computed to 15: page 3 is the tree's, not a tail."""
    s = scheduler()
    r = Request("a", list(range(20, 35)), SamplingParams(max_new_tokens=4, ignore_eos=True))
    s.add(r)
    p1 = launch(s)
    s.in_flight = True
    p2 = launch(s)
    s.commit(p1, tokens(p1))
    check(s)
    s.abort(r)
    check(s)
    s.commit(p2, tokens(p2))
    s.radix.evict(s.radix.total_pages())
    assert s.pool.num_free == s.pool.num_usable


def run(seed, num_pages=60, host=(40, 6), rows=8, seats=8, budget=16, abort_p=0.01, steps=1500):
    rng = random.Random(seed)
    s = scheduler(num_pages, seats, budget, rows, host=host)
    sys_prompt = [rng.randrange(1, 50) for _ in range(rng.choice([8, 12, 16]))]
    history, live, n = {}, {}, [0]

    def turn(sid):
        h = history.get(sid)
        h = (list(sys_prompt) if h is None else h) + [rng.randrange(1, 500) for _ in range(rng.randrange(1, 30))]
        n[0] += 1
        req = Request(
            f"r{n[0]}", h, SamplingParams(max_new_tokens=rng.randrange(1, 14), ignore_eos=True), session_id=sid
        )
        s.add(req)
        live[req.rid] = (sid, req)

    for i in range(seats):
        turn(f"s{i}")
    inflight = None

    def drain():
        nonlocal inflight
        if inflight is None:
            return
        plan, inflight = inflight, None
        s.in_flight = False
        for r in s.commit(plan, [rng.randrange(1, 500) for _ in plan.seqs()]):
            sid, _ = live.pop(r.rid)
            h = list(r.prompt_ids) + list(r.output_ids)
            if rng.random() < 0.3 and len(r.output_ids) > 1:  # the next turn re-renders the reply
                k = rng.randrange(len(r.prompt_ids), len(h))
                h[k] = h[k] % 500 + 1
            history[sid] = h
            if len(h) >= 300:
                history.pop(sid)
                sid += "+"
            turn(sid)
        check(s)

    for _ in range(steps):
        if live and rng.random() < abort_p:  # a client abort, between steps (the API's inbox, the engine thread)
            sid, req = live.pop(rng.choice(list(live)))
            s.abort(req)
            check(s)
            turn(sid)
        s.in_flight = inflight is not None
        try:
            plan = launch(s)
        except NeedSync:
            drain()
            continue
        check(s)
        if not plan:
            drain()
            continue
        drain()
        inflight = plan
    return s


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(num_pages=40, host=(30, 6)),
        dict(num_pages=80, rows=4, host=(40, 3)),
        dict(num_pages=50, budget=8, abort_p=0.05),
    ],
)
def test_served_configuration_under_pressure_keeps_every_page_owned_once(kw):
    hits = offloaded = 0
    for seed in range(6):
        s = run(seed, **kw)
        hits += s.host.ckpts_restored
        offloaded += s.radix.num_running_offloaded
    assert hits > 0 and offloaded > 0, "the host cache's checkpoint paths should have run"
