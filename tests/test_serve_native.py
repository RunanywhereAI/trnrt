"""SGLang's native API on one engine (trnrt/serve/native.py), over the scripted engine of tests/test_serve_api.py:
/generate in SGLang's request and response shapes, the info and load endpoints, /health_generate, /abort_request and
/flush_cache. The real engine behind the same routes: tests/test_api_native.py."""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.test_serve_api import ID, PLAIN, WORDS, ScriptedEngine
from trnrt.engine.request import Request, SamplingParams


@pytest.fixture
def serve():
    from fastapi.testclient import TestClient

    from trnrt.serve import api

    started = []

    def start(script, tp=1, preset=None):
        eng = ScriptedEngine(script)
        eng.cfg.tp = tp
        client = TestClient(api.build_app(eng, "glm", preset=preset))
        client.__enter__()
        started.append((client, eng))
        return client, eng

    yield start
    for client, eng in started:
        client.__exit__(None, None, None)
        for lp in [lp for lp in api._LOOPS if lp.engine is eng]:
            lp.stop()


@contextlib.contextmanager
def _uvicorn(eng, preset=None):
    """The app on a real server in a thread (the TestClient runs one request at a time): its base URL."""
    import uvicorn

    from trnrt.serve import api

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(
        uvicorn.Config(api.build_app(eng, "glm", preset=preset), host="127.0.0.1", port=port, log_level="warning")
    )
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    try:
        deadline = time.monotonic() + 30
        while not srv.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert srv.started
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.should_exit = True
        th.join(timeout=30)
        for lp in [lp for lp in api._LOOPS if lp.engine is eng]:
            lp.stop()


def _background(fn):
    """fn() in a thread; returns a getter that joins it and gives its result."""
    out: dict = {}
    th = threading.Thread(target=lambda: out.setdefault("r", fn()), daemon=True)
    th.start()

    def get():
        th.join(30)
        assert not th.is_alive()
        return out["r"]

    return get


def _wait_for(cond, s: float = 10.0) -> None:
    deadline = time.monotonic() + s
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert cond()


@pytest.mark.parametrize("method", ["POST", "PUT"])
def test_generate_answers_in_sglangs_shape(serve, method):
    client, eng = serve(PLAIN)
    r = client.request(method, "/generate", json={"text": "Hello", "rid": "a1"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["text"] == "Hello there again!" and out["output_ids"] == PLAIN
    meta = out["meta_info"]
    assert meta["id"] == "a1" and meta["finish_reason"] == {"type": "stop", "matched": PLAIN[-1]}
    assert (meta["prompt_tokens"], meta["completion_tokens"], meta["cached_tokens"]) == (1, 4, 2)
    assert meta["e2e_latency"] >= 0 and "output_token_logprobs" not in meta and "prompt_token_ids" not in out
    assert eng.added[-1].params.max_new_tokens == 128  # SGLang's default
    assert eng.added[-1].prompt_ids == [ID["Hello"]]

    r = client.post("/generate", json={"input_ids": [1, 2], "sampling_params": {"max_new_tokens": 2}})
    assert r.json()["meta_info"]["finish_reason"] == {"type": "length", "length": 2}
    assert r.json()["text"] == "Hello there"
    r = client.post("/generate", json={"input_ids": [1], "sampling_params": {"stop": [" again", "!"]}})
    assert r.json()["text"] == "Hello there" and r.json()["meta_info"]["finish_reason"] == {
        "type": "stop",
        "matched": " again",
    }
    p = eng.added[-1].params
    assert p.stop == (" again", "!")


def test_generate_maps_sglangs_sampling_params(serve):
    client, eng = serve(PLAIN)
    sp = {
        "max_new_tokens": 3,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 1 << 30,  # SGLang's "all"
        "min_p": 0.05,
        "stop_token_ids": [ID["!"]],
        "ignore_eos": True,
        "sampling_seed": 7,
        "n": 1,
        "skip_special_tokens": True,
        "spaces_between_special_tokens": True,
        "no_stop_trim": False,
        "min_new_tokens": 0,
        "json_schema": None,
        "stop_regex": None,
    }
    r = client.post("/generate", json={"input_ids": [1], "sampling_params": sp, "priority": 0, "image_data": None})
    assert r.status_code == 200, r.text
    p = eng.added[-1].params
    assert (p.max_new_tokens, p.temperature, p.top_p, p.top_k, p.min_p) == (3, 0.5, 0.9, 1 << 30, 0.05)
    assert (p.stop_token_ids, p.ignore_eos, p.seed) == ((ID["!"],), True, 7)


def test_generate_batch_rids_and_prompt_ids(serve):
    client, _ = serve(PLAIN)
    r = client.post(
        "/generate",
        json={
            "input_ids": [[1], [2, 3]],
            "rid": "b",
            "sampling_params": [{"max_new_tokens": 1}, {"max_new_tokens": 2}],
            "return_prompt_token_ids": True,
        },
    )
    assert r.status_code == 200, r.text
    a, b = r.json()
    assert (a["meta_info"]["id"], b["meta_info"]["id"]) == ("b_0", "b_1")  # SGLang's batch rids
    assert (a["text"], b["text"]) == ("Hello", "Hello there")
    assert (a["prompt_token_ids"], b["prompt_token_ids"]) == ([1], [2, 3])
    r = client.post("/generate", json={"text": ["Hello", "Hello !"], "rid": ["x", "y"]})
    assert [o["meta_info"]["id"] for o in r.json()] == ["x", "y"]
    assert [o["meta_info"]["prompt_tokens"] for o in r.json()] == [1, 2]


def test_generate_output_logprobs_in_sglangs_tuples(serve):
    client, _ = serve(PLAIN)
    body = {"input_ids": [1], "return_logprob": True, "top_logprobs_num": 1, "sampling_params": {"max_new_tokens": 2}}
    meta = client.post("/generate", json=body).json()["meta_info"]
    assert meta["input_token_logprobs"] == [] and meta["input_top_logprobs"] == []
    assert meta["output_token_logprobs"] == [[-0.25, ID["Hello"], None], [-0.25, ID[" there"], None]]
    assert meta["output_top_logprobs"][0] == [[-0.25, ID["Hello"], None], [-2.0, 0, None]]
    meta = client.post("/generate", json={**body, "return_text_in_logprobs": True, "logprob_start_len": -1}).json()[
        "meta_info"
    ]
    assert meta["output_token_logprobs"][1] == [-0.25, ID[" there"], " there"]
    assert meta["output_top_logprobs"][0][1] == [-2.0, 0, "<unk>"]
    meta = client.post("/generate", json={**body, "top_logprobs_num": 0}).json()["meta_info"]
    assert "output_top_logprobs" not in meta and len(meta["output_token_logprobs"]) == 2


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ({"input_ids": [1], "return_logprob": True, "logprob_start_len": 0}, "prompt logprobs"),
        ({"input_ids": [1], "return_logprob": True, "top_logprobs_num": 21}, "top_logprobs_num"),
        ({"input_ids": [1], "image_data": "cat.png"}, "not supported: image_data"),
        ({"input_ids": [1], "lora_path": "adapter"}, "not supported: lora_path"),
        ({"input_ids": [1], "priority": 2}, "not supported: priority"),
        ({"input_ids": [1], "sampling_params": {"n": 2}}, "not supported: n"),
        ({"input_ids": [1], "sampling_params": {"json_schema": "{}"}}, "not supported: json_schema"),
        ({"input_ids": [1], "sampling_params": {"no_stop_trim": True}}, "not supported: no_stop_trim"),
        ({"input_ids": [1], "sampling_params": {"max_new_tokens": 0}}, "max_new_tokens must be"),
        ({"input_ids": [1], "sampling_params": {"sampling_seed": -1}}, "sampling_seed must be"),
        ({"input_ids": [1], "sampling_params": {"bogus": 1}}, "unknown sampling parameters: bogus"),
        ({"input_ids": [1], "sampling_params": [{}]}, "sampling_params"),
        ({"input_ids": [1], "return_logprob": [True]}, "return_logprob"),
        ({"input_ids": [1], "rid": ["a"]}, "rid"),
        ({"input_ids": [[1], [2]], "rid": ["a"]}, "rid"),
        ({"input_ids": [[1], [2]], "rid": ["a", "a"]}, "must differ"),
        ({"input_ids": [[1], [2]], "sampling_params": [{}]}, "sampling_params"),
        ({"text": "Hello", "input_ids": [1]}, "exactly one of text or input_ids"),
        ({}, "exactly one of text or input_ids"),
        ({"input_ids": []}, "input_ids"),
        ({"text": [1]}, "text"),
        ({"input_ids": [len(WORDS)]}, "token ids"),
        ({"input_ids": [1], "stream": "yes"}, "stream"),
        ({"input_ids": [1], "session_id": 3}, "session_id"),
        ({"input_ids": [1], "sampling_params": {"frequency_penalty": 0.5}}, "penalties"),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_generate_refuses_what_it_cannot_serve_with_a_400(serve, body, why):
    client, eng = serve(PLAIN)
    r = client.post("/generate", json=body)
    assert r.status_code == 400 and why in r.json()["detail"], r.text
    assert eng.scheduler.running == [] and all(q.status.value == "finished" for q in eng.added)


def _sse(r) -> list:
    lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    return [json.loads(ln[6:]) for ln in lines[:-1]]


def test_generate_streams_cumulative_events(serve):
    client, _ = serve(PLAIN)
    with client.stream("POST", "/generate", json={"input_ids": [1], "stream": True}) as r:
        assert r.status_code == 200
        events = _sse(r)
    assert [e["text"] for e in events] == ["Hello", "Hello there", "Hello there again", "Hello there again!"]
    assert [e["output_ids"] for e in events][-1] == PLAIN and "index" not in events[0]
    assert [e["meta_info"]["finish_reason"] for e in events[:-1]] == [None] * 3
    assert events[-1]["meta_info"]["finish_reason"] == {"type": "stop", "matched": PLAIN[-1]}
    assert events[-1]["meta_info"]["completion_tokens"] == 4

    body = {"input_ids": [[1], [2]], "stream": True, "sampling_params": [{"max_new_tokens": 1}, {}]}
    with client.stream("POST", "/generate", json=body) as r:
        events = _sse(r)
    last = {e["index"]: e for e in events}
    assert last[0]["text"] == "Hello" and last[0]["meta_info"]["finish_reason"]["type"] == "length"
    assert last[1]["text"] == "Hello there again!" and last[1]["meta_info"]["finish_reason"]["type"] == "stop"


def test_abort_request_ends_a_request_by_rid_with_what_it_generated():
    """The client still waiting gets its output so far and finish_reason abort (a disconnect-only abort would leave it
    hanging); abort_all ends every request, an OpenAI one included; a rid in use is refused while it runs."""
    import httpx

    eng = ScriptedEngine(PLAIN * 40)
    eng.step_delay = 0.02  # 160 tokens: 3 s on its own
    with _uvicorn(eng) as base, httpx.Client(base_url=base, timeout=30) as c:
        body = {"input_ids": [1], "rid": "r1", "sampling_params": {"max_new_tokens": 160}}
        got = _background(lambda: c.post("/generate", json=body))
        _wait_for(lambda: any(len(q.output_ids) >= 3 for q in eng.scheduler.running))
        r = c.post("/generate", json=body)
        assert r.status_code == 400 and "already in use" in r.text
        assert c.post("/abort_request", json={"rid": "nope"}).status_code == 200  # no such rid: nothing happens
        assert eng.aborted == []
        assert c.post("/abort_request", json={"rid": "r1"}).status_code == 200
        r = got()
        assert r.status_code == 200, r.text
        meta = r.json()["meta_info"]
        assert meta["finish_reason"]["type"] == "abort" and 3 <= meta["completion_tokens"] < 160
        assert len(r.json()["output_ids"]) == meta["completion_tokens"] and eng.aborted == ["r1"]

        gen = _background(lambda: c.post("/generate", json={**body, "rid": "r2"}))
        cmpl = _background(lambda: c.post("/v1/completions", json={"prompt": [1], "max_tokens": 160}))
        _wait_for(lambda: len(eng.scheduler.running) == 2)
        assert c.post("/abort_request", json={"abort_all": True}).status_code == 200
        assert gen().json()["meta_info"]["finish_reason"]["type"] == "abort"
        assert cmpl().json()["choices"][0]["finish_reason"] == "abort"
        assert len(eng.aborted) == 3
        for bad in ({}, {"rid": ""}, {"rid": 3}, {"rid": "r1", "abort_message": "x"}, {"abort_all": "yes"}):
            assert c.post("/abort_request", json=bad).status_code == 400, bad


def test_flush_cache_flushes_only_an_idle_engine():
    """Idle: flushed at once. Busy with timeout 0 (SGLang's default): refused with the reason, nothing flushed. Busy
    with a timeout: flushed once the running request finishes, never while it runs; a second flush meanwhile is
    refused."""
    import httpx

    eng = ScriptedEngine(PLAIN * 40)
    eng.step_delay = 0.02
    with _uvicorn(eng) as base, httpx.Client(base_url=base, timeout=30) as c:
        r = c.get("/flush_cache")
        assert r.status_code == 200 and r.text.startswith("Cache flushed.") and eng.flushes == [0]
        assert c.post("/flush_cache").status_code == 200 and eng.flushes == [0, 0]

        busy = _background(lambda: c.post("/v1/completions", json={"prompt": [1], "max_tokens": 100}))  # 2 s
        _wait_for(lambda: eng.scheduler.running)
        r = c.post("/flush_cache")
        assert r.status_code == 400 and "1 requests are running or waiting" in r.text and eng.flushes == [0, 0]
        waiting = _background(lambda: c.post("/flush_cache?timeout=20"))
        time.sleep(0.1)
        r = c.get("/flush_cache?timeout=5")
        assert r.status_code == 400 and "Another flush_cache" in r.text
        assert busy().status_code == 200
        assert waiting().status_code == 200 and eng.flushes == [0, 0, 0]  # flushed after the request, while idle
        for t in ("-1", "x", "nan", "inf"):
            assert c.post(f"/flush_cache?timeout={t}").status_code == 400, t


def test_health_generate_runs_a_token_through_the_engine(monkeypatch):
    """Idle: the probe (prompt [0], one greedy token) runs, and /metrics does not count it. Stuck in a step past the
    timeout: 503, and the probe is aborted."""
    import httpx

    from trnrt.serve import native

    eng = ScriptedEngine(PLAIN)
    with _uvicorn(eng) as base, httpx.Client(base_url=base, timeout=30) as c:
        r = c.get("/health_generate")
        assert r.status_code == 200 and r.json() == {"status": "ok"}
        (probe,) = eng.added
        assert probe.prompt_ids == [0] and probe.params.max_new_tokens == 1 and probe.params.temperature == 0.0
        assert len(probe.output_ids) == 1 and eng.aborted == []
        assert "trnrt:generation_tokens_total 0" in c.get("/metrics").text  # a probe is not a request
        c.post("/v1/completions", json={"prompt": [1]})
        assert "trnrt:generation_tokens_total 4" in c.get("/metrics").text

        monkeypatch.setattr(native, "HEALTH_GENERATE_TIMEOUT_S", 0.3)
        eng.step_delay = 2.0  # no step ends within the timeout: the engine is stuck
        started = _background(lambda: c.post("/v1/completions", json={"prompt": [1], "max_tokens": 1}))
        _wait_for(lambda: eng.scheduler.running)
        time.sleep(0.1)
        r = c.get("/health_generate")
        assert r.status_code == 503 and "no engine step ended" in r.json()["error"], r.text
        assert started().status_code == 200
        _wait_for(lambda: len(eng.added) == 4 and eng.aborted == [eng.added[3].rid])


def test_health_generate_is_503_once_a_rank_died(serve):
    client, eng = serve(PLAIN, tp=2)
    eng.dead = [1]
    r = client.get("/health_generate")
    assert r.status_code == 503 and "rank(s) [1] exited" in r.json()["error"]


def test_load_and_info_endpoints(serve):
    client, eng = serve(PLAIN, preset="batch12")
    eng.scheduler.waiting.append(Request("w", [1, 2, 3], SamplingParams()))
    (load,) = client.get("/get_load").json()
    assert load["dp_rank"] == 0 and (load["num_reqs"], load["num_running_reqs"], load["num_waiting_reqs"]) == (1, 0, 1)
    # 6 KV pages in use (the stand-in's pool: 100 usable, 90 free, 4 evictable) of 4 tokens each, plus 3 waiting
    assert (load["num_tokens"], load["num_pending_tokens"], load["token_usage"]) == (27, 3, 0.06)

    for path in ("/model_info", "/get_model_info"):
        info = client.get(path).json()
        assert info["model_path"] == "org/glm-test" and info["served_model_name"] == "glm"
        assert info["is_generation"] is True and (info["reasoning_parser"], info["tool_call_parser"]) == (
            "glm45",
            "glm47",
        )
    for path in ("/server_info", "/get_server_info"):
        info = client.get(path).json()
        assert (info["preset"], info["tp_size"], info["dp_size"], info["context_length"]) == ("batch12", 1, 1, 64)
        assert info["max_model_len"] == 64 and info["page_size"] == 4 and info["model_path"] == "org/glm-test"
        assert info["internal_states"][0]["num_waiting_reqs"] == 1
        from trnrt import __version__

        assert info["version"] == __version__


def test_model_info_reads_the_checkpoint_but_never_names_its_download_path(tmp_path):
    """The engine's resolved checkpoint (a download cache under the server's home) is read for model_type and
    architectures; the path the clients see is the one the operator passed."""
    from fastapi.testclient import TestClient

    from trnrt.serve import api

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "glm5_next", "architectures": ["Glm5NextForCLM"]}))
    eng = ScriptedEngine(PLAIN)
    eng.model_path = str(tmp_path)
    with TestClient(api.build_app(eng, "glm")) as c:
        info = c.get("/model_info").json()
        assert (info["model_type"], info["architectures"]) == ("glm5_next", ["Glm5NextForCLM"])
        for path in ("/model_info", "/server_info"):
            assert str(tmp_path) not in c.get(path).text
    for lp in [lp for lp in api._LOOPS if lp.engine is eng]:
        lp.stop()


@pytest.mark.parametrize("stream", [False, True])
def test_a_generate_client_that_disconnects_aborts_its_requests(stream):
    """Every prompt of a batch is aborted when its client goes away, streamed or not, as on the OpenAI routes."""
    eng = ScriptedEngine(PLAIN * 40)
    eng.step_delay = 0.05  # 160 tokens: 8 s to finish on its own
    with _uvicorn(eng) as base:
        port = int(base.rsplit(":", 1)[1])
        body = json.dumps(
            {"input_ids": [[1], [2]], "sampling_params": {"max_new_tokens": 160}, "stream": stream}
        ).encode()
        c = socket.create_connection(("127.0.0.1", port))
        c.sendall(
            b"POST /generate HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n%s" % (len(body), body)
        )
        _wait_for(lambda: len(eng.scheduler.running) == 2)
        rids = sorted(q.rid for q in eng.scheduler.running)
        c.close()
        _wait_for(lambda: len(eng.aborted) == 2, 3)
        assert sorted(eng.aborted) == rids and all(len(q.output_ids) < 160 for q in eng.added)
