"""Priority scheduling and the queue limits through the engine's own step (LLMEngine.add_request and step), with blind
speculative steps, dynamic speculation and the state pool, against tests/test_spec_engine_fake.py's fake device: the
device checks every call's state rows and pages, so a preemption or a displacement the host books wrongly fails a call.
The outputs must be the reference whatever was preempted, displaced or reordered."""

from __future__ import annotations

import dataclasses
import random

import pytest

from tests.test_spec_engine_fake import check_done, make_engine, prompts
from trnrt.engine.request import SamplingParams, Status
from trnrt.engine.scheduler import DISPLACED, QueueFull, Scheduling


def drain(eng, limit=8000):
    """Step until the engine has no work, failing (not hanging) if it never drains."""
    steps = 0
    while eng.has_work():
        eng.step()
        steps += 1
        assert steps < limit, "the engine never drained its work"


def with_scheduling(eng, **kw):
    """make_engine's engine with these host-only settings, as LLMEngine.__init__ applies them."""
    eng.scheduling = Scheduling(**kw)
    eng.scheduler.cfg = dataclasses.replace(eng.scheduler.cfg, **eng.scheduling.scheduler_fields())
    return eng


@pytest.mark.parametrize("dyn", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_priority_scheduling_through_the_engine_generates_the_reference(dyn, seed):
    if dyn:
        eng = make_engine(num_pages=48, seats=8, ckpts=10, spec_seats=3, spec_max_batch=3, max_len=200, seed=seed)
    else:
        eng = make_engine(num_pages=40, seats=4, ckpts=6, max_len=120, seed=seed)
    with_scheduling(eng, priority_scheduling=True, priority_preemption_threshold=5)
    eng.scheduler.cfg = dataclasses.replace(eng.scheduler.cfg, max_running=3)  # host only: seats fill sooner
    rng = random.Random(seed)
    reqs = []
    pending = prompts(20, seed=seed + 3)
    while pending:  # arrivals in small bursts over the run
        n = rng.randrange(1, 4)
        burst, pending = pending[:n], pending[n:]
        for p in burst:
            prio = rng.choice([0, 3, 30])
            reqs.append(eng.add_request(p, SamplingParams(max_new_tokens=rng.randrange(6, 45)), priority=prio))
        for _ in range(rng.randrange(0, 4)):
            eng.step()
    drain(eng)
    check_done(eng, reqs)
    assert eng.scheduler.num_priority_preemptions > 0, "no request ever took a seat by priority"


def test_a_full_queue_refuses_or_displaces_through_the_engine_and_frees_what_it_held():
    eng = with_scheduling(make_engine(seats=2), priority_scheduling=True, max_queued_requests=2)
    sp = SamplingParams(max_new_tokens=8)
    running = [eng.add_request(p, sp, priority=5) for p in prompts(2, seed=5, shared=False)]
    eng.step()  # both admitted: the queue is empty again
    low = [eng.add_request(p, sp, priority=0) for p in prompts(2, seed=6, shared=False)]
    with pytest.raises(QueueFull):
        eng.add_request([10, 11, 12], sp, priority=0)  # no better than the worst waiting
    high = eng.add_request([10, 11, 12], sp, priority=9)
    gone = low[1]  # the newer of the two worst
    assert gone.status is Status.FINISHED and gone.error == (503, DISPLACED) and gone.finish_reason == "abort"
    drain(eng)
    check_done(eng, [*running, low[0], high])
    assert gone.output_ids == []


def test_the_engine_refuses_a_priority_it_would_ignore():
    eng = make_engine()
    with pytest.raises(ValueError, match="enable-priority-scheduling"):
        eng.add_request([10, 11, 12], SamplingParams(max_new_tokens=4), priority=1)
    assert not eng.scheduler.waiting
    r = eng.add_request([10, 11, 12], SamplingParams(max_new_tokens=4))  # no priority: the default engine as before
    drain(eng)
    check_done(eng, [r])
