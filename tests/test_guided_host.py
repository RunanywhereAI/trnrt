"""The host side of guided decoding and thinking budgets (engine/guided.py HostSampling), driven token by token with a
stand-in tokenizer and matcher: no model, no xgrammar.

- a grammar's dead end ends its request as "stop", with or without ignore_eos;
- a thinking budget reached on a partial close tag forces only the rest of the tag;
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trnrt.engine.guided import HostSampling
from trnrt.engine.request import Request, SamplingParams, Status

EOS = 1
# "<think>" is two ids and "</think>" three, so a close tag can be cut short at the budget.
TAGS = {"<think>": [10, 11], "</think>": [12, 13, 14]}


def tokenizer(text, add_special_tokens=False):
    return {"input_ids": TAGS[text]}


class DeadMatcher:
    """A grammar at a dead end: it refuses every token and never terminates (host_inputs._mask_row then allows EOS)."""

    def __init__(self):
        self.calls = 0

    def is_terminated(self):
        return False

    def accept_token(self, t):
        self.calls += 1
        return False


def running(prompt, params, sampling: HostSampling | None = None) -> Request:
    r = Request("r", prompt, params)
    if sampling is not None:
        sampling.admit(r)
    r.status = Status.RUNNING
    return r


def emit(r: Request, *ids: int) -> list:
    r.token_ids.extend(ids)
    return [SimpleNamespace(req=r, sample=True, draft=False)]


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_a_grammar_dead_end_ends_the_request_as_stop(ignore_eos):
    hs = HostSampling(tokenizer, 64, (EOS,), runner=None)
    r = running([5, 6], SamplingParams(ignore_eos=ignore_eos))
    r.matcher, r.grammar_dead = DeadMatcher(), True
    finished = []

    def finish_now(req, why):
        finished.append((req, why))
        req.status = Status.FINISHED

    done = hs.advance(emit(r, EOS), finish_now)
    assert done == [r] and finished == [(r, "stop")]
    assert r.grammar_seen == 1 and r.matcher.calls == 1


def test_a_dead_end_the_commit_already_stopped_is_not_finished_twice():
    hs = HostSampling(tokenizer, 64, (EOS,), runner=None)
    r = running([5, 6], SamplingParams())
    r.matcher, r.grammar_dead = DeadMatcher(), True
    r.status = Status.FINISHED  # the commit stopped on the EOS (no ignore_eos)
    assert hs.advance(emit(r, EOS), lambda req, why: pytest.fail("finished twice")) == []


def test_a_token_its_grammar_refuses_without_a_dead_end_still_fails_loudly():
    hs = HostSampling(tokenizer, 64, (EOS,), runner=None)
    r = running([5, 6], SamplingParams())
    r.matcher = DeadMatcher()  # refuses, but no dead end was reached: the mask was not applied
    with pytest.raises(RuntimeError, match="rejected by its own grammar"):
        hs.advance(emit(r, EOS), lambda req, why: None)


def think(budget: int, *outs: list[int]) -> Request:
    """A request whose prompt opened a reasoning block, fed `outs` one commit at a time."""
    hs = HostSampling(tokenizer, 64, (EOS,), runner=None)
    r = running([5, *TAGS["<think>"]], SamplingParams(thinking_token_budget=budget), hs)
    assert r.think_in
    for ids in outs:
        hs.advance(emit(r, *ids), lambda req, why: None)
    return r


def test_a_budget_reached_on_a_partial_close_tag_forces_only_the_rest_of_it():
    r = think(3, [20], [21], [12])  # the third reasoning token is the close tag's first id
    assert r.think_force == [13, 14] and not r.think_done
    r = think(3, [20], [21], [12], [13], [14])  # the forced rest, one token a step: the tag is whole once
    assert r.think_done and not r.think_force and r.output_ids == [20, 21, 12, 13, 14]


def test_a_budget_reached_elsewhere_forces_the_whole_close_tag():
    r = think(2, [20], [21])
    assert r.think_force == [12, 13, 14]
    r = think(3, [20], [12, 21])  # a close tag's first id that the next token showed was no tag
    assert r.think_force == [12, 13, 14]
    r = think(6, [20], [21], [22], [12, 13, 14])  # the model closes the block itself, under the budget
    assert r.think_done and not r.think_force
