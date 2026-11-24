"""Scheduler, blind speculative decode steps (engine/spec_async.py): what the host books for a step whose drafts, tokens and
progress live on the device until its results are read back.

The "model" is test_scheduler's: the next token is a pure function of the history, so what a verify emits is known exactly
whatever number of drafts it accepts. The property under test is the same as there: scheduling, here with 1 .. 3 blind steps
in flight per request and any acceptance, must never change what a request generates, and must keep the pages and the
checkpoint rows balanced."""

from __future__ import annotations

import random
from collections import deque

import pytest

from tests.test_scheduler import EOS, PS, VOCAB, check_invariants, make, next_token, reference_output
from trnrt.engine.request import Request, SamplingParams, Status
from trnrt.engine.scheduler import BLIND, NeedSync

K = 2  # drafts per verify (the shipped configurations' k)


def speculating(sched, plain=lambda: False):
    """Every decoding request speculates, as the engine's draft_fn does unless it is in plain mode."""
    sched.draft_fn = lambda req: [] if plain() else [BLIND] * K
    return sched


def emitted(history, n):
    """The n tokens a verify that accepts n - 1 drafts emits: the model's own continuation."""
    out = []
    for _ in range(n):
        out.append(next_token(history))
        history = [*history, out[-1]]
    return out


def run_overlapped(sched, reqs, rng, depth=2, max_steps=20_000, on_plan=None):
    """The engine's overlapped loop: a step is launched (advance) before the previous one's results are read back
    (commit), up to `depth` steps in flight; a NeedSync reads them all back and retries."""
    for r in reqs:
        sched.add(r)
    queue: deque = deque()

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

    steps = 0
    while sched.has_work() or queue:
        try:
            plan = sched.schedule()
        except NeedSync:
            while queue:
                commit_one()
            continue
        if not plan:
            assert queue, "scheduler made no progress"
            commit_one()
            continue
        if on_plan is not None:
            on_plan(plan)
        sched.advance(plan)
        queue.append(plan)
        sched.in_flight = True
        if len(queue) >= depth:
            commit_one()
        check_invariants(sched)
        steps += 1
        assert steps < max_steps
    return steps


def _decoding(sched, *prompts, max_new=40):
    """Requests prefilled, together and without speculation, to the point of decoding, results read back."""
    draft_fn, sched.draft_fn = sched.draft_fn, None
    reqs = [Request(f"r{i}", p, SamplingParams(max_new_tokens=max_new)) for i, p in enumerate(prompts)]
    for req in reqs:
        sched.add(req)
    while not all(r.is_decoding and r.output_ids for r in reqs):
        plan = sched.schedule()
        tokens = [next_token(s.req.token_ids[: s.end]) if s.sample else -1 for s in plan.seqs()]
        sched.update(plan, tokens)
    sched.draft_fn = draft_fn
    return reqs[0] if len(reqs) == 1 else reqs


def test_a_blind_entry_covers_the_newest_token_and_k_drafts_and_leaves_the_progress_to_the_commit():
    sched = speculating(make(budget=64))
    req = _decoding(sched, list(range(100, 106)))
    n0, t0 = req.num_computed, len(req.token_ids)

    (e,) = sched.schedule().decodes
    assert (e.start, e.num_tokens, e.draft, e.sample, e.save) == (n0, 1 + K, [BLIND] * K, True, None)
    assert len(req.pages) * PS >= e.end

    plan = sched.schedule()
    sched.advance(plan)
    assert (req.spec_inflight, req.num_computed, len(req.token_ids)) == (1, n0, t0), "a blind step moves nothing yet"

    # a second step launched before the first is read back starts past everything the first may have advanced
    sched.in_flight = True
    (e2,) = sched.schedule().decodes
    assert e2.start == n0 + (1 + K) * 1 and len(req.pages) * PS >= e2.end

    sched.commit(plan, [emitted(req.token_ids, 3)])
    assert (req.spec_inflight, len(req.token_ids)) == (0, t0 + 3) and req.num_computed == len(req.token_ids) - 1
    assert req.output_ids == reference_output(req.prompt_ids, len(req.output_ids))


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_outputs_match_the_reference_whatever_the_acceptance_and_the_steps_in_flight(depth, seed):
    rng = random.Random(seed)
    # a pool short enough that admission reserves tightly and running requests are preempted now and then
    sched = speculating(make(num_pages=48, max_seqs=4, budget=24, max_len=160, decode_tokens=8))
    system = [rng.randrange(8, VOCAB) for _ in range(12)]
    reqs = [
        Request(
            f"r{i}",
            ([*system] if i % 3 == 0 else []) + [rng.randrange(8, VOCAB) for _ in range(rng.randrange(1, 40))],
            SamplingParams(max_new_tokens=rng.randrange(1, 60)),
        )
        for i in range(24)
    ]
    entries = []
    run_overlapped(sched, reqs, rng, depth=depth, on_plan=lambda p: entries.extend(p.decodes))
    for r in reqs:
        assert r.status is Status.FINISHED and r.spec_inflight == 0
        assert r.output_ids == reference_output(r.prompt_ids, r.params.max_new_tokens)
    assert sched.pool.num_free + sched.radix.total_pages() == sched.pool.num_usable
    assert any(e.draft for e in entries), "no blind entry was ever planned"
    # not vacuous: the pool was short enough to preempt, and the overlap put steps in flight behind each other
    assert sched.num_preemptions > 0


def test_blind_entries_never_save_a_decode_checkpoint_and_plain_ones_do():
    """track() takes a checkpoint row when a plain decode crosses a page boundary; a blind step's state rows are the verify's
    own, so it takes none. The plain run is the positive control: the detector sees a save when there is one."""

    def saves(sched):
        found = []
        run_overlapped(
            sched,
            [Request("x", list(range(100, 109)), SamplingParams(max_new_tokens=30))],
            random.Random(0),
            depth=1,
            on_plan=lambda p: found.extend(s.save for s in p.decodes if s.save is not None),
        )
        return found

    assert saves(make(budget=64)), "plain decodes cross page boundaries without a checkpoint: the control is dead"
    assert saves(speculating(make(budget=64))) == []


def test_a_request_that_finished_ignores_the_blind_steps_launched_after_its_end():
    sched = speculating(make(budget=64))
    req = _decoding(sched, list(range(100, 106)), max_new=2)
    plans = []
    for _ in range(2):
        plans.append(sched.schedule())
        sched.advance(plans[-1])
        sched.in_flight = True
    assert req.spec_inflight == 2
    done = sched.commit(plans[0], [emitted(req.token_ids, 3)])
    assert done == [req] and req.finish_reason == "length" and len(req.output_ids) == 2
    n_tokens, free = len(req.token_ids), sched.pool.num_free
    assert sched.commit(plans[1], [emitted(req.token_ids, 3)]) == []
    assert (req.spec_inflight, len(req.token_ids), sched.pool.num_free) == (0, n_tokens, free)
    assert req.output_ids == reference_output(req.prompt_ids, 2)


def test_the_emitted_run_stops_at_the_token_that_finishes_the_request():
    sched = speculating(make(budget=64))
    req = _decoding(sched, list(range(100, 106)))
    plan = sched.schedule()
    sched.advance(plan)
    sched.commit(plan, [[11, EOS, 12]])  # the stop token is the second of three emitted
    assert req.finish_reason == "stop" and req.output_ids[-2:] == [11, EOS]


def test_pages_are_reserved_past_the_steps_in_flight_and_cut_at_max_model_len():
    sched = speculating(make(budget=64, max_len=24))
    req = _decoding(sched, list(range(100, 118)))  # 18 tokens, 6 below max_model_len
    req.spec_inflight = 5  # five steps in flight, each of which may have advanced 3 positions
    (e,) = sched.schedule().decodes
    assert e.start == 23, "the start is capped at max_model_len - 1: the request finishes by then"
    assert len(req.pages) == 24 // PS, "pages cover max_model_len exactly: not short of it, and none past it"


def test_a_pool_too_short_for_the_drafts_preempts_the_youngest_after_a_sync():
    sched = speculating(make(num_pages=7, max_seqs=2, budget=64, decode_tokens=0))
    a, b = _decoding(sched, list(range(100, 111)), list(range(200, 211)))  # 11 tokens, 3 pages each of the 6 usable
    assert sched.pool.num_free == 0 and a.num_computed == 11 and b.num_computed == 11
    sched.in_flight = True
    with pytest.raises(NeedSync):
        sched.schedule()  # a needs a 4th page for newest token + 2 drafts; only a preemption frees one
    sched.in_flight = False
    b.spec_ready = True
    plan = sched.schedule()
    assert b.status is Status.WAITING and b.spec_ready is False, "the preempted request's board row is rebuilt"
    assert [e.req for e in plan.decodes] == [a] and plan.decodes[0].draft == [BLIND] * K
