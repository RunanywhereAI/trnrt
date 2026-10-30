"""The HTTP layer (trnrt/serve/api.py) over a scripted stand-in for the engine, so it runs anywhere in seconds.

The stand-in implements exactly the engine API the server uses (add_request, has_work, step, abort, dead_ranks,
close_failed, close, tokenizer, cfg.tp, scheduler, pool, radix) with real Request objects, and generates a fixed
GLM-format output one token per step. Its tokenizer is a word-level one whose words are GLM's tags and a few phrases,
decoded by plain concatenation, and whose chat template opens `<think>` in the prompt unless enable_thinking is false,
as GLM-5.3-Flash's does. The real engine behind the same server: tests/test_api*.py.
"""

from __future__ import annotations

import json
import time
import types

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # fastapi.testclient

from trnrt.engine.request import Request, Status  # noqa: E402

WORDS = ["<unk>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
         "</arg_value>", "The user wants weather.", "Let me check.", "get_weather", "city", "Paris", "Hello", " there",
         " again", "!"]
ID = {w: i for i, w in enumerate(WORDS)}
TEMPLATE = ("{% for m in messages %}{{ m['content'] }} {% endfor %}"
            "{% if enable_thinking is defined and not enable_thinking %}</think>{% else %}<think>{% endif %}")
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}}}}]
# Thinking on: the prompt opened the reasoning block, so the output starts inside it.
TOOL_TURN = [ID[w] for w in ("The user wants weather.", "</think>", "Let me check.", "<tool_call>", "get_weather",
                             "<arg_key>", "city", "</arg_key>", "<arg_value>", "Paris", "</arg_value>",
                             "</tool_call>")]
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
        self.cfg = types.SimpleNamespace(tp=1)
        self.scheduler = types.SimpleNamespace(running=[], waiting=[], num_preemptions=0)
        self.pool = types.SimpleNamespace(num_usable=100, num_free=90)
        self.radix = types.SimpleNamespace(evictable_pages=4)
        self.script = script
        self.dead: list[int] = []
        self.aborted: list[str] = []
        self.closed = False

    def add_request(self, prompt_ids, params, rid=None, session_id=None):
        if len(prompt_ids) > 64:
            raise ValueError(f"prompt of {len(prompt_ids)} tokens does not fit max_model_len=64")
        req = Request(rid, list(prompt_ids), params, session_id=session_id)
        req.num_cached_tokens = 2
        self.scheduler.running.append(req)
        return req

    def has_work(self) -> bool:
        return bool(self.scheduler.running)

    def step(self) -> None:
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


@pytest.fixture()
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
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "weather?"}],
                                                  "tools": TOOLS, "max_tokens": 64})
    assert r.status_code == 200, r.text
    body = r.json()
    (choice,) = body["choices"]
    msg = choice["message"]
    assert msg["reasoning_content"] == "The user wants weather." and msg["content"] == "Let me check."
    (call,) = msg["tool_calls"]
    assert call["function"]["name"] == "get_weather" and json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    assert choice["finish_reason"] == "tool_calls" and body["object"] == "chat.completion" and body["model"] == "glm"
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": len(TOOL_TURN), "total_tokens": 2 + len(TOOL_TURN),
                             "prompt_tokens_details": {"cached_tokens": 2}}


def test_streamed_chat_gives_the_same_message_and_a_usage_chunk_when_asked(serve):
    client, _ = serve(TOOL_TURN)
    body = {"messages": [{"role": "user", "content": "weather?"}], "tools": TOOLS, "max_tokens": 64, "stream": True,
            "stream_options": {"include_usage": True}}
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
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}],
                                                  "chat_template_kwargs": {"enable_thinking": False}})
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
    bad_choice = {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS,
                  "tool_choice": {"type": "function", "function": {"name": "nope"}}}
    assert client.post("/v1/chat/completions", json=bad_choice).status_code == 400
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400
    # What no served graph computes is refused, not ignored; a plain-text response_format is no request at all.
    msg = {"messages": [{"role": "user", "content": "x"}]}
    for extra in ({"response_format": {"type": "json_object"}}, {"guided_json": {"type": "object"}},
                  {"prompt_logprobs": 1}):
        r = client.post("/v1/chat/completions", json={**msg, **extra})
        assert r.status_code == 400 and "not supported" in r.text, (extra, r.text)
    assert client.post("/v1/chat/completions", json={**msg, "response_format": {"type": "text"}}).status_code == 200


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
