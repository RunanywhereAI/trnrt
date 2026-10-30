"""The OpenAI-compatible server (trnrt/serve/api.py) over a real engine and a real GLM-5.3-Flash checkpoint.

Gated on TRNRT_TEST_MODEL: a GLM-5.3-Flash checkpoint directory, with its tokenizer and chat template, that the engine
can load with the configuration in the fixtures below. The answers are checked for sense ("Paris"), so a random-weight
checkpoint does not do.
"""

import json
import math
import os

import pytest
import torch

MODEL = os.environ.get("TRNRT_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set TRNRT_TEST_MODEL to run")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from trnrt.config import EngineConfig
    from trnrt.engine.engine import LLMEngine
    from trnrt.serve.api import build_app

    eng = LLMEngine(EngineConfig(model_path=MODEL, device="cpu", dtype=torch.float32, page_size=8,
                                 num_pages=512, max_num_seqs=4, max_model_len=1024, max_prefill_tokens=128))
    with TestClient(build_app(eng, "test-model")) as c:
        c.engine = eng
        yield c


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def test_completion_greedy_with_logprobs(client):
    r = client.post("/v1/completions", json={"prompt": "The capital of France is", "max_tokens": 4,
                                             "temperature": 0, "logprobs": 3})
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    assert "Paris" in choice["text"]
    lp = choice["logprobs"]
    assert len(lp["token_logprobs"]) == 4 and all(len(t) == 3 for t in lp["top_logprobs"])
    # Greedy: the chosen token is the most likely one.
    for chosen, top in zip(lp["token_logprobs"], lp["top_logprobs"]):
        assert math.isclose(chosen, max(top.values()), abs_tol=1e-5)
    assert r.json()["usage"]["completion_tokens"] == 4


def test_stop_string_is_cut(client):
    r = client.post("/v1/completions", json={"prompt": "1, 2, 3, 4,", "max_tokens": 30,
                                             "temperature": 0, "stop": [" 7"]})
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "7" not in choice["text"] and choice["text"].strip().endswith("6,")


def test_streaming_chat(client):
    body = {"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 12,
            "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    chunks, done = [], False
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                break
            chunks.append(json.loads(line[6:]))
    assert done and chunks
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text.strip()
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")


def test_tool_call_is_parsed(client):
    body = {"messages": [{"role": "user", "content": "What is the weather in Paris right now?"}],
            "tools": TOOLS, "max_tokens": 200, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = client.post("/v1/chat/completions", json=body)
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls", choice
    call = choice["message"]["tool_calls"][0]["function"]
    assert call["name"] == "get_weather" and "paris" in json.loads(call["arguments"])["city"].lower()


def test_reasoning_is_separated(client):
    body = {"messages": [{"role": "user", "content": "What is 2+3? Reply with the number."}],
            "max_tokens": 300, "temperature": 0}
    msg = client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]
    assert msg.get("reasoning_content"), msg
    assert "<think>" not in (msg["content"] or "") and "</think>" not in (msg["content"] or "")


def test_metrics_count_the_served_tokens(client):
    r = client.post("/v1/completions", json={"prompt": "The capital of France is", "max_tokens": 4,
                                             "temperature": 0})
    assert r.status_code == 200, r.text
    m = client.get("/metrics").text
    assert "trnrt:generation_tokens_total" in m and "trnrt:time_to_first_token_seconds_count" in m
    gen = [ln for ln in m.splitlines() if ln.startswith("trnrt:generation_tokens_total ")][0]
    assert int(gen.split()[-1]) >= 4
    # vllm-neuron's start-up and per-graph metrics (trnrt/metrics.py startup_lines)
    assert "trnrt:model_load_time_seconds " in m and "trnrt:model_load_size_bytes " in m
    execs = [ln for ln in m.splitlines() if ln.startswith("trnrt:neff_execution_count{")]
    assert execs and sum(int(ln.split()[-1]) for ln in execs) >= 4
    assert any(ln.startswith('trnrt:compilation_time_seconds{bucket_name="') for ln in m.splitlines())


def test_session_id(client):
    body = {"prompt": "A long agent conversation about capitals: France, Spain, Italy and Japan.",
            "session_id": "agent-7", "max_tokens": 2, "temperature": 0}
    assert client.post("/v1/completions", json=body).status_code == 200
    r = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 1, "session_id": "agent-7"})
    assert r.status_code == 200
    assert client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 1, "session_id": 5}).status_code == 400
