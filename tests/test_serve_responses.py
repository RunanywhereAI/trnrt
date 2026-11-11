"""The OpenAI Responses API, POST /v1/responses (trnrt/serve/responses.py), over the scripted engine of
tests/test_serve_api.py: the same reasoning, answer and tool calls /v1/chat/completions gives, as Responses items and
events; what a stateless server cannot do is refused with 400."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.test_serve_api import PLAIN, TOOL_TURN, TOOLS, ScriptedEngine

# The chat TOOLS as Responses function tools.
R_TOOLS = [{"type": "function", **t["function"]} for t in TOOLS]


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from trnrt.serve import api

    started = []

    def start(script, setup=None):
        eng = ScriptedEngine(script)
        if setup is not None:
            setup(eng)
        c = TestClient(api.build_app(eng, "glm"))
        c.__enter__()
        started.append((c, eng))
        return c, eng

    yield start
    for c, eng in started:
        c.__exit__(None, None, None)
        for lp in [lp for lp in api._LOOPS if lp.engine is eng]:
            lp.stop()


def _sse(c, body) -> list[dict]:
    with c.stream("POST", "/v1/responses", json={**body, "stream": True}) as r:
        assert r.status_code == 200, r.read()
        lines = list(r.iter_lines())
    names = [ln[7:] for ln in lines if ln.startswith("event: ")]
    data = [json.loads(ln[6:]) for ln in lines if ln.startswith("data: ")]
    assert names == [d["type"] for d in data]
    assert [d["sequence_number"] for d in data] == list(range(len(data)))
    return data


def _strip_ids(items: list[dict]) -> list[dict]:
    return [{k: v for k, v in it.items() if k not in ("id", "call_id")} for it in items]


def test_a_tool_turn_matches_the_chat_api(client):
    """Reasoning, the answer and the call come back as reasoning, message and function_call items carrying what
    /v1/chat/completions returns for the same request; usage counts the reasoning tokens (those before </think>)."""
    c, eng = client(TOOL_TURN)
    r = c.post("/v1/responses", json={"input": "weather?", "tools": R_TOOLS, "instructions": "Be brief."})
    assert r.status_code == 200, r.text
    resp = r.json()
    chat = c.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "weather?"}],
            "tools": TOOLS,
        },
    )
    msg = chat.json()["choices"][0]["message"]
    assert eng.added[0].prompt_ids == eng.added[1].prompt_ids  # the same prompt
    reasoning, message, call = resp["output"]
    assert reasoning["type"] == "reasoning" and reasoning["content"] == [
        {"type": "reasoning_text", "text": msg["reasoning_content"]}
    ]
    assert reasoning["summary"] == [] and "encrypted_content" not in reasoning
    assert message["type"] == "message" and message["role"] == "assistant" and message["status"] == "completed"
    assert message["content"] == [{"type": "output_text", "text": msg["content"], "annotations": [], "logprobs": []}]
    (tc,) = msg["tool_calls"]
    assert call["type"] == "function_call" and call["name"] == tc["function"]["name"] == "get_weather"
    assert json.loads(call["arguments"]) == json.loads(tc["function"]["arguments"]) == {"city": "Paris"}
    assert call["call_id"].startswith("call_") and call["id"].startswith("fc_")
    assert resp["status"] == "completed" and resp["object"] == "response" and resp["id"].startswith("resp_")
    assert resp["store"] is False and resp["instructions"] == "Be brief." and resp["incomplete_details"] is None
    n_prompt = len(eng.added[0].prompt_ids)
    assert resp["usage"] == {
        "input_tokens": n_prompt,
        "input_tokens_details": {"cached_tokens": 2},
        "output_tokens": len(TOOL_TURN),
        "output_tokens_details": {"reasoning_tokens": 1},
        "total_tokens": n_prompt + len(TOOL_TURN),
    }


def test_the_stream_sends_the_responses_events_and_the_same_items(client):
    c, _ = client(TOOL_TURN)
    body = {"input": "weather?", "tools": R_TOOLS}
    whole = c.post("/v1/responses", json=body).json()
    ev = _sse(c, body)
    kinds = [e["type"] for e in ev]
    assert kinds[:2] == ["response.created", "response.in_progress"] and kinds[-1] == "response.completed"
    assert ev[0]["response"]["status"] == "in_progress" and ev[0]["response"]["output"] == []
    expected = [
        "response.output_item.added",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
    ]
    # each item's deltas in their place: a run of one delta kind counts once (the script sends one or more)
    mid = kinds[2:-1]
    assert [k for i, k in enumerate(mid) if i == 0 or not (k.endswith(".delta") and k == mid[i - 1])] == expected
    final = ev[-1]["response"]
    assert _strip_ids(final["output"]) == _strip_ids(whole["output"])  # each parse makes its own call id
    args = "".join(e["delta"] for e in ev if e["type"] == "response.function_call_arguments.delta")
    assert json.loads(args) == {"city": "Paris"}
    added = [e["item"] for e in ev if e["type"] == "response.output_item.added"]
    assert [a["status"] for a in added] == ["in_progress"] * 3
    assert final["usage"] == whole["usage"]


def test_length_is_incomplete(client):
    c, _ = client(PLAIN)
    r = c.post("/v1/responses", json={"input": "hi", "max_output_tokens": 2}).json()
    assert r["status"] == "incomplete" and r["incomplete_details"] == {"reason": "max_output_tokens"}
    assert r["usage"]["output_tokens"] == 2
    ev = _sse(c, {"input": "hi", "max_output_tokens": 2})
    assert ev[-1]["type"] == "response.incomplete" and ev[-1]["response"]["status"] == "incomplete"


def test_a_reasoning_summary_and_encrypted_content_mirror_the_reasoning(client):
    """reasoning.summary mirrors the reasoning into a summary part (as SGLang does: there is no summarizer), streamed
    as reasoning_summary events; include reasoning.encrypted_content returns the reasoning text there, for a stateless
    client to send back."""
    c, _ = client(TOOL_TURN)
    body = {
        "input": "weather?",
        "tools": R_TOOLS,
        "reasoning": {"summary": "auto"},
        "include": ["reasoning.encrypted_content"],
    }
    item = c.post("/v1/responses", json=body).json()["output"][0]
    assert item["summary"] == [{"type": "summary_text", "text": "The user wants weather."}]
    assert item["encrypted_content"] == "The user wants weather."
    kinds = [e["type"] for e in _sse(c, body)]
    assert "response.reasoning_summary_text.delta" in kinds and "response.reasoning_text.delta" not in kinds
    assert "response.reasoning_summary_part.added" in kinds and "response.reasoning_summary_part.done" in kinds


@pytest.mark.parametrize(
    ("extra", "why"),
    [
        ({"store": True}, "store is not supported"),
        ({"previous_response_id": "resp_1"}, "previous_response_id is not supported"),
        ({"background": True}, "background is not supported"),
        ({"truncation": "auto"}, "truncation is not supported"),
        ({"include": ["file_search_call.results"]}, "include supports only"),
        ({"include": ""}, "include supports only"),  # a falsy non-list is malformed, not an empty list
        ({"include": False}, "include supports only"),
        ({"include": {}}, "include supports only"),
        ({"text": {"format": {"type": "json_schema", "name": "x", "schema": {}}}}, "text.format"),
        ({"tools": [{"type": "web_search"}]}, "tool type 'web_search' is not supported"),
        ({"tool_choice": "required"}, "tool_choice"),
        ({"tool_choice": {"type": "function", "name": "get_weather"}}, "tool_choice"),
        ({"top_logprobs": 2}, "top_logprobs is not supported"),
        ({"max_tool_calls": 1}, "max_tool_calls is not supported"),
        ({"n": 2}, "n is not a /v1/responses field"),
        ({"max_tokens": 5}, "max_tokens is not a /v1/responses field"),
        ({"messages": []}, "messages is not a /v1/responses field"),
        ({"reasoning": {"effort": "medium"}}, "reasoning_effort"),
        ({"reasoning": {"summary": "long"}}, "reasoning.summary must be"),
        ({"reasoning": {"summary": "auto", "budget": 5}}, "reasoning.budget is not supported"),
        ({"service_tier": "flex"}, "service_tier is not supported"),
        ({"service_tier": "priority"}, "service_tier is not supported"),
        ({"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]}, "'input_image'"),
        ({"input": [{"type": "item_reference", "id": "x"}]}, "item_reference is not supported"),
        ({"input": []}, "input must be"),
        ({"input": 3}, "input must be"),
        ({"stream": "yes"}, "stream must be true or false"),
    ],
    ids=repr,
)
def test_what_cannot_be_served_is_refused(client, extra, why):
    """Each refusal is a 400 whose detail says why (AGENTS.md), and nothing reaches the engine."""
    c, eng = client(PLAIN)
    r = c.post("/v1/responses", json={"input": "hi", "tools": R_TOOLS, **extra})
    assert r.status_code == 400, r.text
    assert why in r.json()["detail"], r.text
    assert eng.added == []


def test_neutral_fields_are_served(client):
    """What an OpenAI SDK may send by default, or a stateless agent harness always sends."""
    c, _ = client(PLAIN)
    body = {
        "input": "hi",
        "store": False,
        "truncation": "disabled",
        "text": {"format": {"type": "text"}},
        "include": [],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "metadata": {"k": "v"},
        "user": "u-1",
        "service_tier": "auto",
        "top_logprobs": 0,
        "prompt_cache_key": "conversation-1",
        "safety_identifier": "user-hash-1",
        "reasoning": {"summary": None, "generate_summary": None},
    }
    r = c.post("/v1/responses", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["metadata"] == {"k": "v"}


def test_stored_responses_are_not_found(client):
    c, _ = client(PLAIN)
    for method, path in (("GET", "/v1/responses/resp_1"), ("DELETE", "/v1/responses/resp_1")):
        assert c.request(method, path).status_code == 404
    assert c.post("/v1/responses/resp_1/cancel").status_code == 404


def test_input_items_become_the_chat_messages():
    """An assistant turn's items (reasoning, its message, its calls) join into one assistant message; a call's output
    becomes a tool message; text parts join."""
    from fastapi import HTTPException

    from trnrt.serve.responses import input_messages

    body = {
        "instructions": "sys",
        "input": [
            {"role": "developer", "content": "dev"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "a"}, {"text": "b"}]},
            {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "think"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
            {"type": "function_call", "call_id": "call_1", "name": "f", "arguments": '{"x": 1}'},
            {"type": "function_call", "call_id": "call_2", "name": "g", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "r1"},
            {"type": "function_call_output", "call_id": "call_2", "output": [{"type": "input_text", "text": "r2"}]},
            {"type": "reasoning", "encrypted_content": "again"},
            {"role": "assistant", "content": "done"},
        ],
    }
    with pytest.raises(HTTPException, match="not supported") as e:  # a part without a type is not text
        input_messages(body)
    assert e.value.status_code == 400
    body["input"][1]["content"][1]["type"] = "input_text"
    assert input_messages(body) == [
        {"role": "system", "content": "sys"},
        {"role": "developer", "content": "dev"},
        {"role": "user", "content": "ab"},
        {
            "role": "assistant",
            "content": "ok",
            "reasoning_content": "think",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}},
                {"id": "call_2", "type": "function", "function": {"name": "g", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "r1"},
        {"role": "tool", "tool_call_id": "call_2", "content": "r2"},
        {"role": "assistant", "content": "done", "reasoning_content": "again"},
    ]


def _two_calls(eng) -> None:
    """The turn calls the tool twice, a newline between the calls (an added token of the stand-in tokenizer)."""
    eng.tokenizer.add_tokens(["\n"])
    nl = eng.tokenizer.convert_tokens_to_ids("\n")
    eng.script = [*TOOL_TURN, nl, *TOOL_TURN[3:]]


def test_whitespace_between_two_calls_opens_no_message_item(client):
    """The newline between two calls is content to the parser; streamed, it opens no message item, so the streamed
    items are the non-streamed ones (where parse_full strips it)."""
    c, _ = client(TOOL_TURN, _two_calls)
    body = {"input": "weather?", "tools": R_TOOLS}
    whole = c.post("/v1/responses", json=body).json()["output"]
    assert [it["type"] for it in whole] == ["reasoning", "message", "function_call", "function_call"]
    final = _sse(c, body)[-1]["response"]["output"]
    assert _strip_ids(final) == _strip_ids(whole)


def test_a_whitespace_only_answer_makes_no_message_item(client):
    """An answer of whitespace alone (thinking off, so parse_full keeps it) makes no message item, streamed or not."""

    def newline(eng) -> None:
        eng.tokenizer.add_tokens(["\n"])
        eng.script = [eng.tokenizer.convert_tokens_to_ids("\n")]

    c, _ = client(PLAIN, newline)
    body = {"input": "hi", "chat_template_kwargs": {"enable_thinking": False}}
    whole = c.post("/v1/responses", json=body).json()["output"]
    assert whole == []
    assert _sse(c, body)[-1]["response"]["output"] == []
