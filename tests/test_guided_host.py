"""The host side of guided decoding and thinking budgets (engine/guided.py HostSampling) and the stop-string check under
min_tokens (engine/engine.py), driven token by token with a stand-in tokenizer and matcher: no model, no xgrammar.

- a grammar's dead end ends its request as "stop", with or without ignore_eos;
- a thinking budget reached on a partial close tag forces only the rest of the tag;
- a stop string that ended among the first min_tokens tokens never ends the request later;
- the compiled-grammar cache keeps the most recently used grammars, CACHE_SIZE at most (a stand-in xgrammar).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from trnrt.engine import grammar
from trnrt.engine.engine import LLMEngine
from trnrt.engine.grammar import GrammarBackend, GrammarSpec
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


class _Tok:
    """Detokenizes each id to its own bracketed text, so a stop string names exactly one token."""

    @staticmethod
    def decode(ids):
        return "".join(f"<{i}>" for i in ids)


class StopEngine:
    """The engine's stop-string check and speculative trim, over _Tok, recording what it finishes."""

    _check_stop_strings = LLMEngine._check_stop_strings
    _stop_in = LLMEngine._stop_in
    _trim_past_stop = LLMEngine._trim_past_stop

    def __init__(self):
        self.tokenizer = _Tok()
        self.finished: list[str] = []
        self.scheduler = SimpleNamespace(finish_now=lambda req, why: self.finished.append(why))


def stop_check(r: Request, draft=False) -> list[Request]:
    eng = StopEngine()
    done = eng._check_stop_strings([SimpleNamespace(req=r, sample=True, draft=draft)])
    assert len(done) == len(eng.finished) and all(w == "stop" for w in eng.finished)
    return done


def test_a_stop_string_among_the_first_min_tokens_never_ends_the_request():
    """vLLM checks stop strings from min_tokens on, over the newly decoded text only: a stop string that ended among the
    protected tokens stays in the output and is not matched later."""
    r = running([5, 6], SamplingParams(stop=("<7>",), min_tokens=3))
    for t in (7, 8, 9, 20, 21):  # the stop token is the first output token, protected
        emit(r, t)
        assert stop_check(r) == [], r.output_ids
    emit(r, 7)  # past min_tokens: it ends the request
    assert stop_check(r) == [r]


def test_a_stop_string_completed_by_the_token_past_min_tokens_ends_the_request():
    r = running([5, 6], SamplingParams(stop=("<8><9>",), min_tokens=2))
    emit(r, 30, 8)
    assert stop_check(r) == []
    emit(r, 9)  # the stop string straddles the boundary and ends past it
    assert stop_check(r) == [r]


def test_a_speculative_step_past_min_tokens_trims_to_its_own_stop_string_not_a_protected_one():
    r = running([5, 6], SamplingParams(stop=("<7>",), min_tokens=3))
    emit(r, 7, 8, 9)
    assert stop_check(r) == []
    emit(r, 20, 7, 21)  # one speculative step of three tokens: its stop token is the middle one
    assert stop_check(r, draft=True) == [r]
    assert r.output_ids == [7, 8, 9, 20, 7], "trimmed past the protected stop token"


class FakeXgr:
    """The parts of xgrammar GrammarBackend calls, counting compilations."""

    def __init__(self):
        self.compiled: list[str] = []
        self.compiler_kw: dict = {}
        outer = self

        class TokenizerInfo:
            @staticmethod
            def from_huggingface(tokenizer, vocab_size, stop_token_ids):
                return "info"

        class GrammarCompiler:
            def __init__(self, info, **kw):
                outer.compiler_kw = kw

            def compile_regex(self, value):
                outer.compiled.append(value)
                return ("compiled", value)

        self.TokenizerInfo, self.GrammarCompiler = TokenizerInfo, GrammarCompiler


def test_the_compiled_grammar_cache_is_a_bounded_lru(monkeypatch):
    xgr = FakeXgr()
    monkeypatch.setitem(sys.modules, "xgrammar", xgr)
    monkeypatch.setattr(grammar, "CACHE_SIZE", 3)
    be = GrammarBackend(object(), 64, (EOS,))
    assert xgr.compiler_kw == {"cache_enabled": False}, "xgrammar's own unbounded cache stays off"
    spec = [GrammarSpec("regex", f"r{i}") for i in range(5)]
    for i in (0, 1, 2, 0, 3):  # r0 used again before r3 arrives: r1 is the least recent, and goes
        be.compile(spec[i])
    assert xgr.compiled == ["r0", "r1", "r2", "r3"] and len(be._cache) == 3
    be.compile(spec[0])
    be.compile(spec[1])  # recompiled: it was dropped
    assert xgr.compiled == ["r0", "r1", "r2", "r3", "r1"] and len(be._cache) == 3
