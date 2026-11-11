"""The Anthropic Messages API, POST /v1/messages and /v1/messages/count_tokens (trnrt/serve/anthropic.py), over the
scripted engine of tests/test_serve_api.py: the same reasoning, answer and tool calls /v1/chat/completions gives, as
thinking, text and tool_use blocks and Anthropic's events; errors in Anthropic's shape."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.test_serve_api import PLAIN, TOOL_TURN, TOOLS, ScriptedEngine

A_TOOLS = [
    {"name": t["function"]["name"], "description": "Weather.", "input_schema": t["function"]["parameters"]}
    for t in TOOLS
]
ASK = {"messages": [{"role": "user", "content": "weather?"}], "max_tokens": 64}


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
    with c.stream("POST", "/v1/messages", json={**body, "stream": True}) as r:
        assert r.status_code == 200, r.read()
        lines = list(r.iter_lines())
    names = [ln[7:] for ln in lines if ln.startswith("event: ")]
    data = [json.loads(ln[6:]) for ln in lines if ln.startswith("data: ")]
    assert names == [d["type"] for d in data]
    return data


def _blocks(events: list[dict]) -> list[dict]:
    """The content blocks a client rebuilds from the events, as the Anthropic SDK accumulates them."""
    blocks: list[dict] = []
    for e in events:
        if e["type"] == "content_block_start":
            assert e["index"] == len(blocks)
            blocks.append(dict(e["content_block"]))
        elif e["type"] == "content_block_delta":
            b, d = blocks[e["index"]], e["delta"]
            if d["type"] == "thinking_delta":
                b["thinking"] += d["thinking"]
            elif d["type"] == "text_delta":
                b["text"] += d["text"]
            else:
                b["_json"] = b.get("_json", "") + d["partial_json"]
    for b in blocks:
        if b["type"] == "tool_use":
            b["input"] = json.loads(b.pop("_json", "{}"))
    return blocks


def test_a_tool_turn_matches_the_chat_api(client):
    """Thinking, text and tool_use blocks carry what /v1/chat/completions returns for the same request; a call ends the
    turn with stop_reason tool_use; usage splits the prompt into uncached input and cache reads, as Anthropic does."""
    c, eng = client(TOOL_TURN)
    r = c.post("/v1/messages", json={**ASK, "tools": A_TOOLS, "system": [{"type": "text", "text": "Be brief."}]})
    assert r.status_code == 200, r.text
    m = r.json()
    chat = c.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "weather?"}],
            "tools": [dict(t, function=dict(t["function"], description="Weather.")) for t in TOOLS],
            "max_tokens": 64,
        },
    )
    msg = chat.json()["choices"][0]["message"]
    assert eng.added[0].prompt_ids == eng.added[1].prompt_ids
    thinking, text, use = m["content"]
    assert thinking == {"type": "thinking", "thinking": msg["reasoning_content"], "signature": ""}
    assert text == {"type": "text", "text": msg["content"]}
    (tc,) = msg["tool_calls"]
    assert use["type"] == "tool_use" and use["name"] == "get_weather" and use["id"].startswith("call_")
    assert use["input"] == json.loads(tc["function"]["arguments"]) == {"city": "Paris"}
    assert m["stop_reason"] == "tool_use" and m["stop_sequence"] is None
    assert m["type"] == "message" and m["role"] == "assistant" and m["id"].startswith("msg_") and m["model"] == "glm"
    n_prompt = len(eng.added[0].prompt_ids)
    assert m["usage"] == {
        "input_tokens": n_prompt - 2,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 2,
        "output_tokens": len(TOOL_TURN),
    }


def test_the_stream_rebuilds_the_same_message(client):
    c, _ = client(TOOL_TURN)
    body = {**ASK, "tools": A_TOOLS}
    whole = c.post("/v1/messages", json=body).json()
    ev = _sse(c, body)
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "message_start" and kinds[-2:] == ["message_delta", "message_stop"]
    start = ev[0]["message"]
    assert start["content"] == [] and start["stop_reason"] is None and start["usage"]["output_tokens"] == 0
    assert start["usage"]["cache_read_input_tokens"] == 2
    got = _blocks(ev)
    for b in got:
        b.pop("id", None)
    want = [{k: v for k, v in b.items() if k != "id"} for b in whole["content"]]
    assert got == want
    assert kinds.count("content_block_start") == kinds.count("content_block_stop") == 3
    assert ev[-2]["delta"] == {"stop_reason": "tool_use", "stop_sequence": None}
    assert ev[-2]["usage"] == whole["usage"]


def test_stop_sequences_and_max_tokens(client):
    c, _ = client(PLAIN)
    # thinking disabled: the stand-in template has a thinking switch (GLM-5.3-Flash's has none, and refuses it)
    body = {**ASK, "stop_sequences": [" again"], "thinking": {"type": "disabled"}}
    m = c.post("/v1/messages", json=body).json()
    assert m["content"] == [{"type": "text", "text": "Hello there"}]
    assert m["stop_reason"] == "stop_sequence" and m["stop_sequence"] == " again"
    ev = _sse(c, body)
    assert _blocks(ev) == m["content"] and ev[-2]["delta"]["stop_sequence"] == " again"
    m = c.post("/v1/messages", json={**ASK, "max_tokens": 2, "thinking": {"type": "disabled"}}).json()
    assert m["stop_reason"] == "max_tokens" and m["content"] == [{"type": "text", "text": "Hello there"}]
    m = c.post("/v1/messages", json={**ASK, "thinking": {"type": "disabled"}}).json()
    assert m["stop_reason"] == "end_turn" and m["content"] == [{"type": "text", "text": "Hello there again!"}]
    m = c.post("/v1/messages", json={**ASK, "thinking": {"type": "adaptive"}}).json()
    assert m["content"] == [{"type": "thinking", "thinking": "Hello there again!", "signature": ""}]


@pytest.mark.parametrize(
    "body",
    [
        {"messages": [{"role": "user", "content": "x"}]},  # no max_tokens
        {**ASK, "max_tokens": 0},
        {**ASK, "messages": []},
        {**ASK, "messages": [{"role": "system", "content": "x"}]},
        {**ASK, "messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]},
        {**ASK, "messages": [{"role": "assistant", "content": [{"type": "redacted_thinking", "data": "x"}]}]},
        {
            **ASK,
            "messages": [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": None}, {"type": "text", "text": "a"}],
                },
                {"role": "user", "content": "y"},
            ],
        },
        {**ASK, "tools": A_TOOLS, "tool_choice": {"type": "any"}},
        {**ASK, "tools": A_TOOLS, "tool_choice": {"type": "tool", "name": "get_weather"}},
        {**ASK, "tools": [{"type": "web_search_20250305", "name": "web_search"}]},
        {**ASK, "thinking": {"type": "enabled", "budget_tokens": 2048}},
        {**ASK, "thinking": {"type": "enabled"}},
        {**ASK, "thinking": {"type": "sometimes"}},
        {**ASK, "output_config": {"effort": "medium"}},
        {**ASK, "mcp_servers": [{"type": "url", "url": "x"}]},
        {**ASK, "stream": "yes"},
        {**ASK, "temperature": -1},
    ],
    ids=repr,
)
def test_refusals_are_400_in_anthropics_error_shape(client, body):
    c, eng = client(PLAIN)
    r = c.post("/v1/messages", json=body)
    assert r.status_code == 400, r.text
    err = r.json()
    assert err["type"] == "error" and err["error"]["type"] == "invalid_request_error" and err["error"]["message"]
    assert all(q.status.name == "FINISHED" for q in eng.added)  # a refusal leaves nothing running
    assert eng.scheduler.running == []


def test_a_malformed_body_and_an_engine_failure_answer_in_anthropics_shape(client):
    c, eng = client(PLAIN)
    r = c.post("/v1/messages", content=b"{oops", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"

    def failing_step():
        raise RuntimeError("device fault")

    eng.step = failing_step
    r = c.post("/v1/messages", json=ASK)
    assert r.status_code == 500 and r.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "engine step failed: device fault"},
    }


def test_count_tokens_counts_what_messages_would_send(client):
    """count_tokens renders the request exactly as /v1/messages does, so its count is that request's prompt length
    (and /tokenize's for the same chat messages). No request reaches the engine."""
    c, eng = client(PLAIN)
    body = {"messages": ASK["messages"], "system": "Be brief.", "tools": A_TOOLS}
    r = c.post("/v1/messages/count_tokens", json=body)
    assert r.status_code == 200, r.text
    assert eng.added == []
    assert c.post("/v1/messages", json={**body, "max_tokens": 4}).status_code == 200
    assert r.json() == {"input_tokens": len(eng.added[-1].prompt_ids)}
    tok = c.post(
        "/tokenize",
        json={
            "messages": [{"role": "system", "content": "Be brief."}, *ASK["messages"]],
            "tools": [dict(t, function=dict(t["function"], description="Weather.")) for t in TOOLS],
        },
    )
    assert tok.json()["count"] == r.json()["input_tokens"]
    bad = c.post("/v1/messages/count_tokens", json={"messages": []})
    assert bad.status_code == 400 and bad.json()["type"] == "error"


def test_the_messages_become_the_chat_messages():
    """A user's tool results become tool messages in their place among its text; an assistant's thinking becomes its
    reasoning_content and its tool_use blocks its tool_calls."""
    from trnrt.serve.anthropic import messages

    body = {
        "system": "sys",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "s"},
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"x": 1}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "r"}]},
                    {"type": "text", "text": "after"},
                ],
            },
        ],
    }
    assert messages(body) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a"},
        {
            "role": "assistant",
            "content": "ok",
            "reasoning_content": "hmm",
            "tool_calls": [{"id": "toolu_1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}],
        },
        {"role": "user", "content": "before"},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "r"},
        {"role": "user", "content": "after"},
    ]


def test_whitespace_between_two_calls_opens_no_text_block(client):
    """The newline between two calls is content to the parser; streamed, it opens no text block, so the streamed
    blocks are the non-streamed ones (where parse_full strips it)."""
    from tests.test_serve_responses import _two_calls

    c, _ = client(TOOL_TURN, _two_calls)
    body = {**ASK, "tools": A_TOOLS}
    whole = c.post("/v1/messages", json=body).json()["content"]
    assert [b["type"] for b in whole] == ["thinking", "text", "tool_use", "tool_use"]
    got = _blocks(_sse(c, body))
    assert [{k: v for k, v in b.items() if k != "id"} for b in got] == [
        {k: v for k, v in b.items() if k != "id"} for b in whole
    ]


def test_a_whitespace_only_answer_makes_no_text_block(client):
    """An answer of whitespace alone (thinking off, so parse_full keeps it) makes no text block, streamed or not."""

    def newline(eng) -> None:
        eng.tokenizer.add_tokens(["\n"])
        eng.script = [eng.tokenizer.convert_tokens_to_ids("\n")]

    c, _ = client(PLAIN, newline)
    body = {**ASK, "thinking": {"type": "disabled"}}
    assert c.post("/v1/messages", json=body).json()["content"] == []
    assert _blocks(_sse(c, body)) == []
