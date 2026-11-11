"""The HTTP layer's structured outputs, min_tokens, penalties and thinking budget (serve/api.py, serve/structured.py), on
tests/test_serve_api.py's scripted engine: each field reaches the engine's SamplingParams as the engine reads it, an
engine started without the graphs answers 400 naming the flag, and a malformed field is a 400 before the engine."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.test_serve_api import PLAIN, ScriptedEngine
from trnrt.engine.grammar import GrammarSpec
from trnrt.engine.request import Request, unsupported
from trnrt.serve.structured import OBJECT


class FlaggedEngine(ScriptedEngine):
    """The scripted engine started with --structured-outputs and --penalties (LLMEngine.add_request's refusal)."""

    def add_request(self, prompt_ids, params, rid=None, session_id=None):
        why = unsupported(params, structured=True, penalties=True)
        if why:
            raise ValueError(f"{why} is not supported; send the request without it")
        req = Request(rid, list(prompt_ids), params, session_id=session_id)
        req.num_cached_tokens = 0
        self.scheduler.running.append(req)
        self.added.append(req)
        return req


def _start(eng):
    from fastapi.testclient import TestClient

    from trnrt.serve import api

    c = TestClient(api.build_app(eng, "glm"))
    c.__enter__()
    yield c, eng
    c.__exit__(None, None, None)
    for lp in [lp for lp in api._LOOPS if lp.engine is eng]:
        lp.stop()


@pytest.fixture
def client():
    yield from _start(FlaggedEngine(PLAIN))


@pytest.fixture
def unflagged():
    yield from _start(ScriptedEngine(PLAIN))


MSG = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}}


@pytest.mark.parametrize(
    ("extra", "want"),
    [
        ({"response_format": {"type": "json_object"}}, GrammarSpec("json_schema", OBJECT)),
        (
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "s", "schema": SCHEMA, "strict": True},
                }
            },
            GrammarSpec("json_schema", json.dumps(SCHEMA)),
        ),
        ({"guided_choice": ["yes", "no"]}, GrammarSpec("regex", "(yes|no)")),
        ({"regex": "[0-9]+"}, GrammarSpec("regex", "[0-9]+")),
        ({"ebnf": 'root ::= "a"'}, GrammarSpec("ebnf", 'root ::= "a"')),
        (  # vLLM's whole StructuredOutputsParams, its other fields at their defaults
            {
                "structured_outputs": {
                    "json": SCHEMA,
                    "regex": None,
                    "choice": None,
                    "grammar": None,
                    "json_object": None,
                    "disable_any_whitespace": False,
                    "disable_additional_properties": False,
                    "whitespace_pattern": None,
                    "structural_tag": None,
                }
            },
            GrammarSpec("json_schema", json.dumps(SCHEMA)),
        ),
        ({"structured_outputs": {"json_object": True}}, GrammarSpec("json_schema", OBJECT)),
        ({"structured_outputs": {"json_object": False}, "regex": "a"}, GrammarSpec("regex", "a")),
    ],
)
def test_each_constraint_field_reaches_the_engine(client, extra, want):
    c, eng = client
    for path, body in (("/v1/chat/completions", MSG), ("/v1/completions", {"prompt": [1, 2], "max_tokens": 8})):
        r = c.post(path, json={**body, **extra})
        assert r.status_code == 200, r.text
        assert eng.added[-1].params.grammar == want


def test_min_tokens_penalties_and_a_thinking_budget_reach_the_engine(client):
    c, eng = client
    body = {
        **MSG,
        "min_tokens": 3,
        "repetition_penalty": 1.1,
        "frequency_penalty": 0.2,
        "presence_penalty": -0.3,
        "thinking_token_budget": 64,
    }
    assert c.post("/v1/chat/completions", json=body).status_code == 200
    p = eng.added[-1].params
    assert (p.min_tokens, p.repetition_penalty, p.frequency_penalty, p.presence_penalty) == (3, 1.1, 0.2, -0.3)
    assert p.thinking_token_budget == 64 and p.grammar is None


@pytest.mark.parametrize(
    "extra",
    [
        {"response_format": {"type": "json_object"}, "guided_regex": "a"},
        {"response_format": {"type": "structural_tag"}},
        {"guided_choice": "yes"},
        {"structured_outputs": {"json_object": "false"}},  # not a boolean: refused, not taken as true
        {"structured_outputs": {"json_object": 1}},
        {"structured_outputs": {"bogus": None}},  # an unknown key, whatever its value
        {"structured_outputs": {"bogus": False}},
        {"structured_outputs": {"disable_any_whitespace": True}},  # vLLM's option, not at its default
        {"min_tokens": 9},  # above max_tokens
        {"prompt_logprobs": 1},  # still not served (docs/API.md)
    ],
)
def test_a_malformed_or_unserved_field_is_a_400_before_the_engine(client, extra):
    c, eng = client
    r = c.post("/v1/chat/completions", json={**MSG, **extra})
    assert r.status_code == 400, r.text
    assert eng.added == []


def test_an_engine_without_the_flags_names_the_one_to_start_it_with(unflagged):
    c, eng = unflagged
    for extra, flag in (
        ({"response_format": {"type": "json_object"}}, "--structured-outputs"),
        ({"min_tokens": 2}, "--structured-outputs"),
        ({"thinking_token_budget": 5}, "--structured-outputs"),
        ({"presence_penalty": 0.5}, "--penalties"),
    ):
        r = c.post("/v1/chat/completions", json={**MSG, **extra})
        assert r.status_code == 400 and flag in r.text and "not supported" in r.text, (extra, r.text)
    assert eng.added == []
