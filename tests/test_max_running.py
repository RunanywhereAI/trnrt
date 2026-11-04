"""EngineConfig.max_running (--max-running): admission capped below the compiled seats, host only.

- The scheduler (tests/test_scheduler.py's pure-function model, so every output is known exactly): a burst of 5 requests
  on 4 seats capped at 2 never runs more than 2 at once, the rest wait and are admitted in arrival order as seats free,
  and every request completes with its reference output; uncapped, the same burst runs 4 at once (the control: the cap
  is what holds it at 2); a cap above the seats is the seats.
- The engine hands its configuration's cap to its scheduler (a tiny GLM-5.3-Flash on CPU: skipped without a transformers
  that has glm5_next): a burst on 4 seats capped at 2 runs at most 2, and its greedy outputs are the uncapped engine's.
"""

from __future__ import annotations

import pytest

from tests.test_scheduler import EOS, PS, Rows, check_invariants, next_token, reference_output
from trnrt.engine.kv_pool import PagePool
from trnrt.engine.radix_cache import RadixCache
from trnrt.engine.request import Request, SamplingParams, Status
from trnrt.engine.scheduler import Scheduler, SchedulerConfig


def make(max_seqs: int, max_running: int | None, budget: int = 16):
    pool = PagePool(256)
    radix = RadixCache(pool, PS)
    cfg = SchedulerConfig(
        page_size=PS,
        max_num_seqs=max_seqs,
        max_prefill_tokens=budget,
        max_model_len=512,
        eos_token_ids=(EOS,),
        max_running=max_running,
    )
    sched = Scheduler(cfg, pool, radix)
    sched.states = Rows(32)
    radix.free_ckpt = sched.states.free_ckpt
    return sched


def burst(sched, n: int = 5):
    """Every request added at once, then steps to the end: (requests, most running at once, whether any waited while
    others ran, the order requests first ran in)."""
    reqs = [Request(f"r{i}", [20 + i] * (3 + 2 * i), SamplingParams(max_new_tokens=6)) for i in range(n)]
    for r in reqs:
        sched.add(r)
    most, waited, admitted = 0, False, []
    while sched.has_work():
        plan = sched.schedule()
        most = max(most, len(sched.running))
        waited |= bool(sched.running) and len(sched.waiting) > 0
        admitted += [r.rid for r in sched.running if r.rid not in admitted]
        sched.update(plan, [next_token(s.req.token_ids[: s.end]) if s.sample else -1 for s in plan.seqs()])
        check_invariants(sched)
    return reqs, most, waited, admitted


def test_a_burst_runs_at_most_the_cap_and_every_request_completes():
    reqs, most, waited, admitted = burst(make(4, 2))
    assert most == 2 and waited
    assert admitted == [r.rid for r in reqs]  # first come, first served
    for r in reqs:
        assert r.status is Status.FINISHED
        assert r.output_ids == reference_output(r.prompt_ids, r.params.max_new_tokens)


def test_uncapped_the_burst_fills_the_seats():
    reqs, most, _, _ = burst(make(4, None))
    assert most == 4
    assert all(r.status is Status.FINISHED for r in reqs)


@pytest.mark.parametrize("seats,cap,want", [(4, 1, 1), (4, 4, 4), (4, 9, 4), (16, 12, 12)])
def test_the_cap_is_the_lesser_of_the_cap_and_the_seats(seats, cap, want):
    """A burst of more requests than seats, with a prefill budget that admits them all in one step."""
    _, most, _, _ = burst(make(seats, cap, budget=1024), n=seats + 2)
    assert most == want


def test_the_engine_hands_its_cap_to_the_scheduler(tmp_path):
    glm = pytest.importorskip("tests.test_glm5_next")
    from trnrt.engine.request import SamplingParams as SP

    glm.build(str(tmp_path), layers=4)
    ps = glm.prompts(3, (5, 9, 4, 7, 6))

    def run(**kw):
        eng = glm.engine(str(tmp_path), max_num_seqs=4, **kw)
        try:
            reqs = [eng.add_request(p, SP(max_new_tokens=5, temperature=0.0, ignore_eos=True)) for p in ps]
            most = 0
            while eng.has_work():
                eng.step()
                most = max(most, len(eng.scheduler.running))
            assert eng.scheduler.cfg.max_num_seqs == 4 and eng.scheduler.cfg.max_running == kw.get("max_running")
            return [r.output_ids for r in reqs], most
        finally:
            eng.close()

    want, most_plain = run()
    got, most = run(max_running=2)
    assert most == 2 and most_plain > 2
    assert got == want
