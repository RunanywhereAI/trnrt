"""The scheduling policies (engine/scheduler.py: SchedulerConfig.policy, priority scheduling, the queue limits).

The "model" is test_scheduler's: the next token is a pure function of the history, so every request's output is known
exactly however it was ordered, preempted or displaced. Each policy is checked by the order it admits requests in, and
every run also checks that the outputs are the reference and the pages and checkpoint rows balance."""

from __future__ import annotations

import dataclasses
import random
import zlib
from collections import deque

import pytest

from tests.test_scheduler import (
    VOCAB,
    check_invariants,
    make,
    next_token,
    reference_output,
    reference_output_no_eos,
    run,
)
from tests.test_scheduler_spec import K, emitted, speculating
from trnrt.engine.request import Request, SamplingParams, Status
from trnrt.engine.scheduler import DISPLACED, NeedSync, QueueFull, SchedulerConfig, Scheduling


def policy(sched, **kw):
    """`sched` (tests.test_scheduler.make) with these SchedulerConfig fields."""
    sched.cfg = dataclasses.replace(sched.cfg, **kw)
    return sched


def req(rid, n, max_new=2, priority=None):
    """A prompt of n copies of one token of its own (no two requests share a page: no junction reorders them)."""
    r = Request(rid, [8 + zlib.crc32(rid.encode()) % (VOCAB - 8)] * n, SamplingParams(max_new_tokens=max_new))
    r.priority = priority
    return r


def step(sched):
    plan = sched.schedule()
    sched.update(plan, [next_token(s.req.token_ids[: s.end]) if s.sample else -1 for s in plan.seqs()])
    check_invariants(sched)
    return plan


def admitted(sched, reqs) -> list[str]:
    """Queue every request, then run to the end: the order of their first prefill entries."""
    for r in reqs:
        sched.add(r)
    order: list[str] = []
    while sched.has_work():
        for s in step(sched).prefills:
            if s.req.rid not in order:
                order.append(s.req.rid)
    for r in reqs:
        assert r.output_ids == reference_output(r.prompt_ids, r.params.max_new_tokens), r.rid
    return order


def drive(sched, arrivals: dict[int, list[Request]], rng, depth=2, max_steps=20_000) -> list[Request]:
    """The engine's overlapped loop (tests/test_scheduler_spec.py run_overlapped), with arrivals: arrivals[t] are added
    before step t is scheduled, as the server's engine thread adds what its inbox holds before each step. Up to `depth`
    steps in flight; a NeedSync reads them all back and retries. Blind speculative entries emit 1 .. 1 + K tokens."""
    queue: deque = deque()
    added: list[Request] = []

    def commit_one():
        plan = queue.popleft()
        tokens = []
        for s in plan.seqs():
            if s.draft:
                tokens.append(emitted(s.req.token_ids, 1 + rng.randrange(K + 1)))
            else:
                tokens.append(next_token(s.req.token_ids[: s.end]) if s.sample else -1)
        sched.commit(plan, tokens)
        sched.in_flight = bool(queue)

    t = 0
    last = max(arrivals, default=0)
    while sched.has_work() or queue or t <= last:
        for r in arrivals.get(t, ()):
            sched.add(r)
            added.append(r)
        t += 1
        try:
            plan = sched.schedule()
        except NeedSync:
            while queue:
                commit_one()
            continue
        if not plan:
            if queue:
                commit_one()
            continue
        for s in plan.seqs():
            assert s.req.status is Status.RUNNING, "a planned entry belongs to a request that is not running"
        sched.advance(plan)
        queue.append(plan)
        sched.in_flight = True
        if len(queue) >= depth:
            commit_one()
        check_invariants(sched)
        assert t < max_steps
    return added


def _arrivals(rng, n, prios, spread=6) -> dict[int, list[Request]]:
    out: dict[int, list[Request]] = {}
    for i in range(n):
        r = Request(
            f"q{i}",
            [rng.randrange(8, VOCAB) for _ in range(rng.randrange(4, 30))],
            SamplingParams(max_new_tokens=rng.randrange(5, 40)),
        )
        r.priority = rng.choice(prios)
        out.setdefault(i * spread // 2 + rng.randrange(spread), []).append(r)
    return out


def test_config_validation():
    assert SchedulerConfig(1, 1, 1, 2, policy="priority").priority_scheduling  # fcfs with priority scheduling
    assert SchedulerConfig(1, 1, 1, 2, policy="priority").policy == "fcfs"
    with pytest.raises(ValueError, match="one of"):
        SchedulerConfig(1, 1, 1, 2, policy="random")
    with pytest.raises(ValueError, match="not lpm"):
        SchedulerConfig(1, 1, 1, 2, policy="lpm", priority_scheduling=True)
    with pytest.raises(ValueError, match="at least 1"):
        SchedulerConfig(1, 1, 1, 2, max_queued_requests=0)


def test_fcfs_spf_and_lof_admission_orders():
    mk = lambda: [req("a", 30, 9), req("bb", 5, 2), req("ccc", 12, 4)]  # noqa: E731
    assert admitted(make(max_seqs=1, budget=64), mk()) == ["a", "bb", "ccc"]
    assert admitted(policy(make(max_seqs=1, budget=64), policy="spf"), mk()) == ["bb", "ccc", "a"]
    assert admitted(policy(make(max_seqs=1, budget=64), policy="lof"), mk()) == ["a", "ccc", "bb"]


def test_lpm_admits_the_longest_cached_prefix_first():
    system = list(range(100, 132))  # 8 pages

    def order(sched) -> tuple[list[str], Request]:
        run(sched, [Request("warm", [*system, 9], SamplingParams(max_new_tokens=2))])  # caches it, with a checkpoint
        blocker = Request("blk", [700, 701, 702], SamplingParams(max_new_tokens=2))
        cold = Request("cold", [500 + i for i in range(20)], SamplingParams(max_new_tokens=2))
        hot = Request("hot", [*system, 11, 12], SamplingParams(max_new_tokens=2))
        return admitted(sched, [blocker, cold, hot]), hot

    got, hot = order(policy(make(max_seqs=1, budget=64), policy="lpm"))
    assert got == ["hot", "blk", "cold"] and hot.num_cached_tokens == len(system)
    assert order(make(max_seqs=1, budget=64))[0] == ["blk", "cold", "hot"]  # the control: arrival order


def test_priority_scheduling_orders_by_priority_both_ways_and_none_last():
    mk = lambda: [req("x", 3, priority=0), req("a", 4, priority=1), req("b", 4), req("c", 4, priority=5)]  # noqa: E731
    hi = policy(make(max_seqs=1, budget=64), priority_scheduling=True)
    assert admitted(hi, mk()) == ["c", "a", "x", "b"]  # higher values first (SGLang's default), no priority last
    lo = policy(make(max_seqs=1, budget=64), priority_scheduling=True, low_priority_first=True)
    assert admitted(lo, mk()) == ["x", "a", "c", "b"]
    assert admitted(make(max_seqs=1, budget=64), mk()) == ["x", "a", "b", "c"]  # off: arrival order
    # with lof the priority still leads, and max_new_tokens orders within one priority
    mk2 = lambda: [req("x", 3, 2), req("a", 4, 3, 1), req("bb", 4, 9, 1), req("c", 4, 2, 2)]  # noqa: E731
    lof = policy(make(max_seqs=1, budget=64), priority_scheduling=True, policy="lof")
    assert admitted(lof, mk2()) == ["c", "bb", "a", "x"]


def _seats_full(**kw):
    """Two seats taken by long requests of priority 0."""
    sched = policy(make(max_seqs=2, budget=64), priority_scheduling=True, **kw)
    sp = SamplingParams(max_new_tokens=30, ignore_eos=True)
    low = [req(f"low{i}", 6, 30, priority=0) for i in range(2)]
    for r in low:
        r.params = sp
        sched.add(r)
    for _ in range(3):
        step(sched)
    assert len(sched.running) == 2
    return sched, low


@pytest.mark.parametrize(("prio", "preempts"), [(20, True), (11, True), (10, False), (5, False)])
def test_a_waiting_request_better_by_more_than_the_threshold_takes_a_seat(prio, preempts):
    sched, low = _seats_full()
    urgent = req("urgent", 5, 30, priority=prio)
    urgent.params = low[0].params
    sched.add(urgent)
    plan = step(sched)
    assert (urgent in [s.req for s in plan.prefills]) == preempts
    assert sched.num_priority_preemptions == int(preempts)
    if preempts:  # the younger of the two equally bad ones goes, and waits behind the urgent one
        assert low[1].status is Status.WAITING and low[1].num_preemptions == 1 and low[0] in sched.running
    run(sched, [])
    for r in [*low, urgent]:
        assert r.output_ids == reference_output_no_eos(r.prompt_ids, 30), r.rid


def test_priority_preemption_can_be_disabled_and_needs_priority_scheduling():
    sched, _ = _seats_full(priority_preemption=False)
    urgent = req("urgent", 5, 30, priority=100)
    sched.add(urgent)
    assert urgent not in [s.req for s in step(sched).prefills] and sched.num_priority_preemptions == 0
    off = make(max_seqs=2, budget=64)  # no priority scheduling: the priority field orders nothing
    for r in [req("l0", 6, 30), req("l1", 6, 30)]:
        off.add(r)
    step(off)
    late = req("late", 5, 30, priority=100)
    off.add(late)
    assert late not in [s.req for s in step(off).prefills] and off.num_priority_preemptions == 0


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_priority_preemption_with_steps_in_flight_asks_for_a_sync_first(depth):
    """The overlapped loop with arrivals over time: a preemption for a better request never runs while a step is in
    flight (NeedSync first; drive() checks every plan's entries), and the outputs stay exact."""
    rng = random.Random(10 + depth)
    sched = policy(make(num_pages=200, max_seqs=3, budget=32), priority_scheduling=True)
    reqs = drive(sched, _arrivals(rng, 16, [None, 0, 5, 50, 100]), rng, depth)
    assert sched.num_priority_preemptions > 0
    for r in reqs:
        assert r.status is Status.FINISHED
        assert r.output_ids == reference_output(r.prompt_ids, r.params.max_new_tokens), r.rid


def test_kv_pressure_preempts_the_worst_priority_first():
    """The KV-short pool of test_scheduler's preemption test, with priorities: the victims are the worst priority, so the
    best request (a young one) is never preempted, where youngest-first preempts it (the control); every step's entries
    are of running requests and the outputs exact."""

    def victims(prio: bool) -> tuple[list[Request], list[Request]]:
        rng = random.Random(1)
        sched = make(num_pages=41, max_seqs=8, budget=32, decode_tokens=8)
        if prio:
            policy(sched, priority_scheduling=True)
        out: list[Request] = []
        real = sched._preempt
        sched._preempt = lambda r: (out.append(r), real(r))
        reqs = []
        for i in range(8):
            sp = SamplingParams(max_new_tokens=100, ignore_eos=True)
            r = Request(f"p{i}", [rng.randrange(8, VOCAB) for _ in range(20)], sp)
            r.priority = 1000 if i == 6 else i
            reqs.append(r)
        drive(sched, {0: reqs}, rng, depth=1)
        for r in reqs:
            assert r.output_ids == reference_output_no_eos(r.prompt_ids, 100), r.rid
        return out, reqs

    out, reqs = victims(True)
    assert out and reqs[6] not in out
    out, reqs = victims(False)
    assert reqs[6] in out, "the control: youngest-first preempts the young best request"


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_priority_scheduling_with_blind_speculative_steps_in_flight(depth, seed):
    """Priority scheduling under the blind speculative steps of tests/test_scheduler_spec.py (up to 3 in flight, any
    acceptance), KV pressure and arrivals over time: exact outputs, balanced pages and rows, preemptions of both
    kinds."""
    rng = random.Random(seed)
    sched = speculating(policy(make(num_pages=32, max_seqs=4, budget=24, decode_tokens=0), priority_scheduling=True))
    reqs = drive(sched, _arrivals(rng, 20, [None, 0, 3, 20, 40]), rng, depth)
    for r in reqs:
        assert r.status is Status.FINISHED and r.spec_inflight == 0
        assert r.output_ids == reference_output(r.prompt_ids, r.params.max_new_tokens), r.rid
    assert sched.num_preemptions > sched.num_priority_preemptions > 0


def test_max_queued_requests_refuses_past_the_limit_and_frees_room_as_requests_start():
    sched = policy(make(max_seqs=1, budget=64), max_queued_requests=2)
    a, b, c = req("a", 4), req("bb", 4), req("ccc", 4)
    sched.add(a)
    sched.add(b)
    with pytest.raises(QueueFull, match="The request queue is full"):
        sched.add(c)
    assert [r.rid for r in sched.waiting] == ["a", "bb"] and c.status is Status.WAITING
    step(sched)  # a starts: one waiting
    sched.add(c)
    assert [r.rid for r in sched.waiting] == ["bb", "ccc"]


def test_max_queued_tokens_counts_the_waiting_prompts():
    sched = policy(make(max_seqs=1, budget=64), max_queued_tokens=10)
    sched.add(req("a", 6))
    with pytest.raises(QueueFull, match="6 prompt tokens"):
        sched.add(req("bb", 5))
    sched.add(req("ccc", 4))  # 10 exactly


def test_a_full_queue_displaces_its_worst_request_for_a_better_one():
    sched = policy(make(max_seqs=1, budget=64), max_queued_requests=3, priority_scheduling=True)
    a, bb, cc = req("a", 4, priority=5), req("bb", 4, priority=1), req("cc", 4, priority=1)
    for r in (a, bb, cc):
        sched.add(r)
    with pytest.raises(QueueFull):  # no priority: never better than anyone
        sched.add(req("none", 4))
    with pytest.raises(QueueFull):  # equal to the worst: refused, as SGLang (strictly better only)
        sched.add(req("tie", 4, priority=1))
    best = req("best", 4, priority=9)
    assert sched.add(best) is cc  # of the two worst (1, 1) the newer
    assert cc.status is Status.FINISHED and cc.finish_reason == "abort" and cc.error == (503, DISPLACED)
    assert [r.rid for r in sched.waiting] == ["a", "bb", "best"]
    assert admitted(sched, []) == ["best", "a", "bb"]


def test_a_full_queue_never_displaces_a_request_that_has_generated():
    sched = policy(make(max_seqs=1, budget=64), max_queued_requests=1, priority_scheduling=True)
    w = req("w", 4, priority=0)
    sched.add(w)
    w.token_ids.append(55)  # as a preempted request that already streamed a token
    with pytest.raises(QueueFull):
        sched.add(req("x", 4, priority=99))


def test_scheduling_settings_and_the_priority_field():
    assert not Scheduling().takes_priority
    Scheduling().check_priority(None)
    with pytest.raises(ValueError, match="enable-priority-scheduling"):
        Scheduling().check_priority(3)
    for s in (
        Scheduling(priority_scheduling=True),
        Scheduling(policy="priority"),
        Scheduling(eviction_policy="priority"),
    ):
        assert s.takes_priority
        s.check_priority(-4)
    base = SchedulerConfig(4, 8, 16, 64)
    assert dataclasses.replace(base, **Scheduling().scheduler_fields()) == base  # the defaults are SchedulerConfig's


def test_the_eviction_bookkeeping_is_recorded_only_under_a_policy_that_reads_it():
    """lru (the default) records no hits and no importance; lfu and priority do, at admission and insert."""
    for ev, tracked in (("lru", False), ("lfu", True), ("priority", True)):
        sched = make()
        sched.radix.eviction_policy, sched.radix.tracks_use = ev, tracked
        system = list(range(100, 132))
        run(sched, [Request("a", [*system, 9], SamplingParams(max_new_tokens=2))])
        b = Request("b", [*system, 10], SamplingParams(max_new_tokens=2))
        b.priority = 7
        run(sched, [b])
        node = sched.radix.match_prefix(system).node
        assert (node.hit_count > 1) == tracked and (node.importance == 7) == tracked, ev
