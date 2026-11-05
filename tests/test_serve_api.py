"""The HTTP layer (trnrt/serve/api.py) over a scripted stand-in for the engine, so it runs anywhere in seconds.

The stand-in implements exactly the engine API the server uses (add_request, has_work, step, abort, dead_ranks,
close_failed, close, tokenizer, cfg.tp, cfg.max_model_len, mcfg.vocab_size, scheduler, pool, radix) with real Request
objects, and generates a fixed GLM-format output one token per step. Its tokenizer is a word-level one whose words are
GLM's tags and a few phrases, decoded by plain concatenation, and whose chat template opens `<think>` in the prompt
unless enable_thinking is false. That switch is the stand-in's: GLM-5.3-Flash's own template has no enable_thinking and
always opens `<think>` (tests/test_glm53_template.py runs the server on it). The real engine behind the same server:
tests/test_api*.py.
"""

from __future__ import annotations

import json
import time
import types

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # fastapi.testclient

from trnrt.engine.request import Request, Status, unsupported

WORDS = [
    "<unk>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "The user wants weather.",
    "Let me check.",
    "get_weather",
    "city",
    "Paris",
    "Hello",
    " there",
    " again",
    "!",
]
ID = {w: i for i, w in enumerate(WORDS)}
TEMPLATE = (
    "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    "{% if enable_thinking is defined and not enable_thinking %}</think>{% else %}<think>{% endif %}"
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]
# Thinking on: the prompt opened the reasoning block, so the output starts inside it.
TOOL_TURN = [
    ID[w]
    for w in (
        "The user wants weather.",
        "</think>",
        "Let me check.",
        "<tool_call>",
        "get_weather",
        "<arg_key>",
        "city",
        "</arg_key>",
        "<arg_value>",
        "Paris",
        "</arg_value>",
        "</tool_call>",
    )
]
PLAIN = [ID[w] for w in ("Hello", " there", " again", "!")]


def _tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.WordLevel(ID, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tk.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, clean_up_tokenization_spaces=False)
    tok.chat_template = TEMPLATE
    return tok


class ScriptedEngine:
    """Every request generates `script` (finish "stop"), cut at its max_new_tokens (finish "length")."""

    def __init__(self, script: list[int]):
        self.tokenizer = _tokenizer()
        self.cfg = types.SimpleNamespace(tp=1, max_model_len=64)
        self.mcfg = types.SimpleNamespace(vocab_size=len(self.tokenizer))
        self.scheduler = types.SimpleNamespace(running=[], waiting=[], num_preemptions=0)
        self.pool = types.SimpleNamespace(num_usable=100, num_free=90)
        self.radix = types.SimpleNamespace(evictable_pages=4)
        self.script = script
        self.dead: list[int] = []
        self.aborted: list[str] = []
        self.added: list[Request] = []
        self.closed = False
        self.step_delay = 0.0

    def add_request(self, prompt_ids, params, rid=None, session_id=None):
        why = unsupported(params)  # the engine's own refusal (LLMEngine.add_request)
        if why:
            raise ValueError(f"{why} is not supported; send the request without it")
        if len(prompt_ids) >= self.cfg.max_model_len:
            raise ValueError(f"prompt of {len(prompt_ids)} tokens does not fit max_model_len=64")
        req = Request(rid, list(prompt_ids), params, session_id=session_id)
        req.num_cached_tokens = 2
        self.scheduler.running.append(req)
        self.added.append(req)
        return req

    def has_work(self) -> bool:
        return bool(self.scheduler.running)

    def step(self) -> None:
        time.sleep(self.step_delay)
        for req in list(self.scheduler.running):
            n = len(req.output_ids)
            t = self.script[n]
            req.token_ids.append(t)
            if req.params.logprobs is not None:
                req.logprobs.append((-0.25, [t, 0], [-0.25, -2.0]))
            req.status = Status.RUNNING
            now = time.monotonic()
            if req.first_token_time is None:
                req.first_token_time = now
            if n + 1 == len(self.script) or n + 1 == req.params.max_new_tokens:
                req.status, req.finish_time = Status.FINISHED, now
                req.finish_reason = "stop" if n + 1 == len(self.script) else "length"
                self.scheduler.running.remove(req)

    def abort(self, req) -> None:
        self.aborted.append(req.rid)
        self.scheduler.running.remove(req)
        req.status, req.finish_reason = Status.FINISHED, "abort"

    def dead_ranks(self) -> list[int]:
        return list(self.dead)

    def close_failed(self) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def serve():
    from fastapi.testclient import TestClient

    from trnrt.serve import api

    started = []

    def start(script):
        eng = ScriptedEngine(script)
        client = TestClient(api.build_app(eng, "glm"))
        client.__enter__()
        started.append((client, eng))
        return client, eng

    yield start
    for client, eng in started:
        client.__exit__(None, None, None)
        for lp in [lp for lp in api._LOOPS if lp.engine is eng]:
            lp.stop()


def _events(client, body) -> list:
    with client.stream("POST", "/v1/chat/completions" if "messages" in body else "/v1/completions", json=body) as r:
        assert r.status_code == 200, r.read()
        lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    return [json.loads(ln[6:]) for ln in lines[:-1]]


def test_chat_returns_reasoning_content_and_a_tool_call(serve):
    client, _ = serve(TOOL_TURN)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "weather?"}], "tools": TOOLS, "max_tokens": 64},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    (choice,) = body["choices"]
    msg = choice["message"]
    assert msg["reasoning_content"] == "The user wants weather." and msg["content"] == "Let me check."
    (call,) = msg["tool_calls"]
    assert call["function"]["name"] == "get_weather" and json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    assert choice["finish_reason"] == "tool_calls" and body["object"] == "chat.completion" and body["model"] == "glm"
    assert body["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": len(TOOL_TURN),
        "total_tokens": 2 + len(TOOL_TURN),
        "prompt_tokens_details": {"cached_tokens": 2},
    }


def test_streamed_chat_gives_the_same_message_and_a_usage_chunk_when_asked(serve):
    client, _ = serve(TOOL_TURN)
    body = {
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": TOOLS,
        "max_tokens": 64,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    events = _events(client, body)
    usage = events.pop()
    assert usage["choices"] == [] and usage["usage"]["completion_tokens"] == len(TOOL_TURN)
    deltas = [e["choices"][0]["delta"] for e in events]
    assert "".join(d.get("reasoning_content", "") for d in deltas) == "The user wants weather."
    assert "".join(d.get("content", "") for d in deltas) == "Let me check."
    calls = [c for d in deltas for c in d.get("tool_calls", ())]
    assert calls[0]["function"]["name"] == "get_weather" and calls[0]["id"].startswith("call_")
    assert json.loads("".join(c["function"]["arguments"] for c in calls)) == {"city": "Paris"}
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert all(e["object"] == "chat.completion.chunk" and "usage" not in e for e in events)
    body.pop("stream_options")
    assert all("usage" not in e for e in _events(client, body))


def test_thinking_off_is_plain_content(serve):
    client, _ = serve(PLAIN)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": {"enable_thinking": False}},
    )
    msg = r.json()["choices"][0]["message"]
    assert msg == {"role": "assistant", "content": "Hello there again!"}
    assert r.json()["choices"][0]["finish_reason"] == "stop"


def test_completion_logprobs_length_and_stop_string(serve):
    client, _ = serve(PLAIN)
    r = client.post("/v1/completions", json={"prompt": [3, 4, 5], "max_tokens": 3, "logprobs": 1})
    choice = r.json()["choices"][0]
    assert choice["text"] == "Hello there again" and choice["finish_reason"] == "length"
    lp = choice["logprobs"]
    assert lp["tokens"] == ["Hello", " there", " again"] and lp["token_logprobs"] == [-0.25] * 3
    assert lp["top_logprobs"][0] == {"Hello": -0.25, "<unk>": -2.0}
    r = client.post("/v1/completions", json={"prompt": "Hello", "stop": " again"})
    assert r.json()["choices"][0]["text"] == "Hello there"
    events = _events(client, {"prompt": [3], "stop": [" again"], "stream": True})
    assert "".join(e["choices"][0]["text"] for e in events) == "Hello there"  # never a partial stop string


def test_refused_requests_are_4xx_not_a_stream(serve):
    client, _ = serve(PLAIN)
    long = {"prompt": list(range(1, 17)) * 5, "stream": True}
    assert client.post("/v1/completions", json=long).status_code == 400  # the engine's ValueError
    assert client.post("/v1/completions", json={"prompt": []}).status_code == 400
    assert client.post("/v1/completions", json={"prompt": [1], "session_id": 7}).status_code == 400
    assert client.post("/v1/completions", json={"prompt": [1], "logprobs": 21}).status_code == 400
    bad_choice = {
        "messages": [{"role": "user", "content": "x"}],
        "tools": TOOLS,
        "tool_choice": {"type": "function", "function": {"name": "nope"}},
    }
    assert client.post("/v1/chat/completions", json=bad_choice).status_code == 400
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400
    # What no served graph computes is refused, not ignored; a plain-text response_format is no request at all.
    msg = {"messages": [{"role": "user", "content": "x"}]}
    for extra in (
        {"response_format": {"type": "json_object"}},
        {"guided_json": {"type": "object"}},
        {"prompt_logprobs": 1},
    ):
        r = client.post("/v1/chat/completions", json={**msg, **extra})
        assert r.status_code == 400 and "not supported" in r.text, (extra, r.text)
    assert client.post("/v1/chat/completions", json={**msg, "response_format": {"type": "text"}}).status_code == 200


# What an OpenAI SDK or an agent harness sends by default: every field at its neutral value.
SDK_CHAT = {
    "model": "glm",
    "temperature": 0.7,
    "top_p": 1,
    "n": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "logprobs": False,
    "top_logprobs": None,
    "logit_bias": None,
    "stop": None,
    "seed": None,
    "user": "u-1",
    "max_completion_tokens": 64,
    "response_format": {"type": "text"},
    "tool_choice": "auto",
    "parallel_tool_calls": True,
    "metadata": {},
    "store": False,
    "service_tier": "auto",
    "prompt_logprobs": None,
    "thinking_token_budget": None,
    "repetition_penalty": 1.0,
    "top_k": -1,
    "min_p": 0,
}
SDK_COMPLETION = {
    "model": "glm",
    "prompt": "Hello",
    "max_tokens": 8,
    "n": 1,
    "best_of": 1,
    "echo": False,
    "suffix": None,
    "logprobs": None,
    "logit_bias": {},
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "stop": None,
    "temperature": 1,
    "top_p": 1,
    "user": "u-1",
}


def test_an_sdk_request_with_every_neutral_field_is_served(serve):
    """Neutral values are accepted, so ordinary clients work; the same fields at a value that would change the output
    are refused. GLM's thinking switch works both ways; only a thinking budget is refused."""
    client, _ = serve(PLAIN)
    msgs = {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS}
    for thinking in (True, False):
        body = {**SDK_CHAT, **msgs, "chat_template_kwargs": {"enable_thinking": thinking}}
        r = client.post("/v1/chat/completions", json=body)
        assert r.status_code == 200, (thinking, r.text)
        msg = r.json()["choices"][0]["message"]
        assert (msg.get("reasoning_content"), msg.get("content")) == (
            ("Hello there again!", None) if thinking else (None, "Hello there again!")
        ), msg
        events = _events(client, {**body, "stream": True, "stream_options": {"include_usage": True}})
        assert events[-1]["usage"]["completion_tokens"] == len(PLAIN)
    r = client.post("/v1/completions", json=SDK_COMPLETION)
    assert r.status_code == 200 and r.json()["choices"][0]["text"] == "Hello there again!", r.text
    for field, value in (
        ("presence_penalty", 0.5),
        ("frequency_penalty", -0.2),
        ("repetition_penalty", 1.1),
        ("thinking_token_budget", 128),
        ("n", 2),
        ("logit_bias", {"13": 5}),
        ("response_format", {"type": "json_object"}),
        ("prompt_logprobs", 0),
    ):
        r = client.post("/v1/chat/completions", json={**SDK_CHAT, **msgs, field: value})
        assert r.status_code == 400 and "not supported" in r.text, (field, r.text)
    for field, value in (("echo", True), ("best_of", 2), ("suffix", "!")):
        r = client.post("/v1/completions", json={**SDK_COMPLETION, field: value})
        assert r.status_code == 400 and "not supported" in r.text, (field, r.text)


def test_health_stats_models_and_metrics(serve):
    client, _ = serve(PLAIN)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/v1/stats").json() == {"gauges": {"kv_pages_used": 6, "kv_pages_total": 100}}
    assert client.get("/v1/models").json()["data"][0]["id"] == "glm"
    client.post("/v1/completions", json={"prompt": [1, 2]})
    m = client.get("/metrics").text
    assert "trnrt:kv_cache_usage_perc 0.060000" in m and "trnrt:generation_tokens_total 4" in m
    assert 'trnrt:request_success_total{finished_reason="stop"} 1' in m
    assert "trnrt:time_to_first_token_seconds_count 1" in m


def test_a_dead_rank_turns_health_503_and_refuses_new_requests(serve):
    from trnrt.serve import api

    client, eng = serve(PLAIN)
    loop = next(lp for lp in api._LOOPS if lp.engine is eng)
    eng.dead = [1]
    h = client.get("/health")  # at once: /health asks the engine itself
    assert h.status_code == 503 and "rank(s) [1] exited" in h.json()["error"]
    assert client.post("/v1/completions", json={"prompt": [1]}).status_code == 503
    loop.thread.join(10)  # the loop's own check between steps fails the engine and ends the thread
    assert not loop.thread.is_alive() and loop.failed == "tensor-parallel rank(s) [1] exited"
    assert eng.closed and eng.aborted == []


def test_request_fields_are_checked_before_the_engine(serve):
    """api._params and EngineLoop.submit: a value the engine would only fail on in a step (numpy's seed, the int64
    sampling arrays, the embedding's rows, a NaN reaching the device's top-k) is a 400 before anything is queued, since
    at tp > 1 one failed step fails the engine for every request (tests/test_api_request_checks.py, on a real engine).
    A null is the field's default."""
    client, eng = serve(PLAIN)
    base = {"prompt": [1, 2], "temperature": 1.0}
    bad = [
        ("seed", -1),
        ("seed", "abc"),
        ("seed", 1.5),
        ("seed", 2**63),
        ("seed", True),
        ("top_k", 2**63),
        ("top_k", -2),
        ("top_k", "5"),
        ("temperature", -0.5),
        ("temperature", "hot"),
        ("top_p", 0),
        ("top_p", 1.5),
        ("min_p", 2),
        ("max_tokens", 0),
        ("max_tokens", -3),
        ("max_tokens", "16"),
        ("logprobs", 1.5),
        ("stop", ""),
        ("stop", [3]),
        ("stop_token_ids", [-1]),
        ("ignore_eos", "yes"),
        ("stream", "false"),
        ("prompt", [-1]),
        ("prompt", [len(WORDS)]),
        ("prompt", [1, True]),
        ("min_tokens", 4),
        ("include_stop_str_in_output", True),
        ("thinking_token_budget", "lots"),
        ("temperature", 10**400),
    ]
    for field, value in bad:
        r = client.post("/v1/completions", json={**base, field: value})
        assert r.status_code == 400, (field, value, r.text)
    for raw in (
        b'{"prompt": [1], "temperature": NaN}',
        b'{"prompt": [1], "top_p": Infinity}',
        b'{"prompt": [1],',
        b"[1, 2]",
    ):
        r = client.post("/v1/completions", content=raw, headers={"content-type": "application/json"})
        assert r.status_code == 400, (raw, r.text)
    assert eng.added == []  # none reached the engine
    ok = {
        **base,
        "seed": 2**63 - 1,
        "top_k": 2**63 - 1,
        "max_tokens": 4.0,
        "temperature": None,
        "top_p": None,
        "min_p": None,
        "frequency_penalty": None,
        "prompt": [0, len(WORDS) - 1],
    }
    r = client.post("/v1/completions", json=ok)
    assert r.status_code == 200, r.text
    p = eng.added[-1].params
    assert (p.seed, p.top_k, p.max_new_tokens, p.temperature, p.top_p, p.min_p) == (
        2**63 - 1,
        2**63 - 1,
        4,
        1.0,
        1.0,
        0.0,
    )
    for k in (-1, 0):  # both turn top-k off
        assert client.post("/v1/completions", json={**base, "top_k": k}).status_code == 200
        assert eng.added[-1].params.top_k == 0
    assert client.post("/v1/completions", json={"prompt": [1]}).status_code == 200
    assert eng.added[-1].params.max_new_tokens == 16  # the OpenAI completions default


def test_chat_fields_default_max_tokens_and_the_role_chunk(serve):
    """Chat: max_tokens defaults to the rest of the context (no hidden small default that reasoning could use up);
    top_logprobs needs logprobs; chat_template_kwargs is an object, cannot set apply_chat_template's own arguments and
    names only variables the template reads (a top-level reasoning_effort too: the stand-in's template has none); the
    first streamed chunk names the role, as OpenAI's does."""
    client, eng = serve(PLAIN)
    msg = {"messages": [{"role": "user", "content": "hi"}]}
    r = client.post("/v1/chat/completions", json=msg)
    assert r.status_code == 200, r.text
    req = eng.added[-1]
    assert req.params.max_new_tokens == eng.cfg.max_model_len - len(req.prompt_ids)
    for extra in (
        {"top_logprobs": 3},
        {"logprobs": "yes"},
        {"chat_template_kwargs": "x"},
        {"chat_template_kwargs": {"tools": []}},
        {"chat_template_kwargs": {"add_generation_prompt": False}},
        {"chat_template_kwargs": {"no_such_variable": 1}},
        {"reasoning_effort": "low"},
        {"add_generation_prompt": False},
        {"continue_final_message": True},
    ):
        r = client.post("/v1/chat/completions", json={**msg, **extra})
        assert r.status_code == 400, (extra, r.text)
    r = client.post("/v1/chat/completions", json={**msg, "top_logprobs": 0, "logprobs": False})
    assert r.status_code == 200 and "logprobs" not in r.json()["choices"][0]
    events = _events(client, {**msg, "stream": True})
    assert events[0]["choices"][0] == {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}


def test_tool_choice_required_or_named_is_refused(serve):
    """No guided decoding runs, so "required" and a named function cannot be enforced: 400, never a 200 with whatever
    the model chose. "auto" and "none" are served."""
    client, eng = serve(PLAIN)
    msg = {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS}
    for choice in ("required", {"type": "function", "function": {"name": "get_weather"}}):
        r = client.post("/v1/chat/completions", json={**msg, "tool_choice": choice})
        assert r.status_code == 400 and "guided decoding" in r.text, (choice, r.text)
    for choice in ("auto", "none", None):
        assert client.post("/v1/chat/completions", json={**msg, "tool_choice": choice}).status_code == 200
    assert len(eng.added) == 3


def test_a_tool_call_cut_off_by_max_tokens_finishes_length(serve):
    """A call the token limit cut short is returned with finish_reason "length", streamed or not, so an agent does not
    run a truncated call as complete; a call the model finished is "tool_calls"."""
    client, _ = serve(TOOL_TURN)
    body = {"messages": [{"role": "user", "content": "weather?"}], "tools": TOOLS, "max_tokens": len(TOOL_TURN) - 2}
    r = client.post("/v1/chat/completions", json=body)
    choice = r.json()["choices"][0]
    assert choice["message"].get("tool_calls") and choice["finish_reason"] == "length", choice
    events = _events(client, {**body, "stream": True})
    assert any(e["choices"][0]["delta"].get("tool_calls") for e in events)
    assert events[-1]["choices"][0]["finish_reason"] == "length"
    r = client.post("/v1/chat/completions", json={**body, "max_tokens": len(TOOL_TURN)})
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"


def test_engine_errors_are_5xx_and_refusals_400(serve):
    """An error inside the engine is the server's (500; 503 once the engine has failed: the dead-rank test above), so
    SDKs retry it and the router can fail over; a request the engine refuses stays 400."""
    client, eng = serve(PLAIN)
    real_step = eng.step

    def failing_step():
        raise RuntimeError("device fault")

    eng.step = failing_step
    r = client.post("/v1/completions", json={"prompt": [1], "max_tokens": 4})
    assert r.status_code == 500 and "device fault" in r.text, r.text
    eng.step = real_step
    real_add = eng.add_request

    def failing_add(*a, **k):
        raise RuntimeError("allocator bug")

    eng.add_request = failing_add
    assert client.post("/v1/completions", json={"prompt": [1]}).status_code == 500
    eng.add_request = real_add
    assert client.post("/v1/completions", json={"prompt": [1]}).status_code == 200  # tp 1: the engine still serves
    assert client.post("/v1/completions", json={"prompt": [1] * 64}).status_code == 400  # does not fit


@pytest.mark.parametrize("stream", [False, True])
def test_a_client_that_disconnects_aborts_its_request(stream):
    """Non-streaming too: nothing cancels a plain response's handler when its client goes away, so without the server's
    own watch the request would hold its seat until max_tokens. Real uvicorn (the TestClient never disconnects)."""
    import socket
    import threading

    import uvicorn

    from trnrt.serve import api

    eng = ScriptedEngine(PLAIN * 40)
    eng.step_delay = 0.05  # 160 tokens: 8 s to finish on its own
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(api.build_app(eng, "glm"), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    try:
        deadline = time.monotonic() + 30
        while not srv.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert srv.started
        body = json.dumps({"prompt": [1], "max_tokens": 160, "stream": stream}).encode()
        c = socket.create_connection(("127.0.0.1", port))
        c.sendall(
            b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n%s" % (len(body), body)
        )
        deadline = time.monotonic() + 10
        while not eng.scheduler.running and time.monotonic() < deadline:
            time.sleep(0.02)
        (req,) = eng.scheduler.running
        c.close()
        deadline = time.monotonic() + 3
        while not eng.aborted and time.monotonic() < deadline:
            time.sleep(0.02)
        assert eng.aborted == [req.rid] and len(req.output_ids) < 160
    finally:
        srv.should_exit = True
        th.join(timeout=30)
        for lp in [lp for lp in api._LOOPS if lp.engine is eng]:
            lp.stop()


@pytest.mark.parametrize(
    "extra",
    [
        {"bogus_field": 1},
        {"top_a": 0.3},
        {"prediction": {"type": "content", "content": "x"}},
        {"modalities": ["audio"]},
        {"audio": {"voice": "x"}},
        {"web_search_options": {}},
        {"max_token": 5},  # a typo of max_tokens
    ],
)
def test_an_unknown_field_is_refused_with_a_400_that_names_it(serve, extra):
    """AGENTS.md invariant 7: a field the server does not know was answered 200 as if it had been applied. Now 400,
    naming the field; no request reaches the engine. The fields an SDK sends by default (SDK_CHAT, SDK_COMPLETION)
    are still served (test_an_sdk_request_with_every_neutral_field_is_served)."""
    client, eng = serve(PLAIN)
    (name,) = extra
    for path, base in (
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/completions", {"prompt": [1]}),
    ):
        r = client.post(path, json={**base, "max_tokens": 2, **extra})
        assert r.status_code == 400 and name in r.json()["detail"] and "unknown" in r.json()["detail"], r.text
    assert eng.added == []


def test_a_field_of_the_other_endpoint_is_refused(serve):
    """`prompt` on a chat request, `messages` or `tools` on a completion: the endpoint would ignore them."""
    client, eng = serve(PLAIN)
    msgs = [{"role": "user", "content": "hi"}]
    r = client.post("/v1/chat/completions", json={"messages": msgs, "prompt": "x"})
    assert r.status_code == 400 and "prompt" in r.json()["detail"], r.text
    for extra in ({"messages": msgs}, {"tools": TOOLS}, {"reasoning_effort": "low"}):
        r = client.post("/v1/completions", json={"prompt": [1], **extra})
        assert r.status_code == 400 and next(iter(extra)) in r.json()["detail"], r.text
    assert eng.added == []
    for extra in ({"user": "u"}, {"model": "glm"}, {"metadata": {"a": "b"}}, {"prompt_cache_key": "k"}):
        assert client.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 2, **extra}).status_code == 200
