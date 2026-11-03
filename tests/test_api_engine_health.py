"""A tensor-parallel server that loses a rank stops serving, and says so (trnrt/serve/api.py EngineLoop._fail):
/health and every new request answer 503, an open request ends with the error instead of hanging, and the engine shuts
down without aborting its requests. Without this, the engine thread could die in the abort path (an abort's state
offload broadcast into the dead workers) while /health kept answering 200, and clients kept sending into a dead engine.

A tiny random GLM-5.3-Flash-architecture model (tests/test_glm5_next.py) at tp=2 over gloo on CPU; prompts are token
ids. At attention TP 2 the DSA caches are context-parallel, so the model takes the served form's constraints (hidden a
multiple of 512, pages of kpool x cp = 8 tokens, an FP8 KV cache; test_glm5_next.test_tp2_matches_tp1).
"""

from __future__ import annotations

import contextlib
import faulthandler
import json
import time

import pytest

from tests.test_api_cached_tokens import _save_tokenizer

PROMPT = [5, 17, 42, 99, 7, 3, 250, 11]
HANG_S = 240  # the whole test, engine start included: a hang here is the bug, not a slow host


@contextlib.contextmanager
def _server(tmp_path):
    from fastapi.testclient import TestClient

    from trnrt.serve import api

    glm = pytest.importorskip("tests.test_glm5_next")
    glm.build(str(tmp_path), hidden_size=512)
    _save_tokenizer(tmp_path, 384)
    faulthandler.dump_traceback_later(HANG_S, exit=True)
    eng = glm.engine(str(tmp_path), tp=2, page_size=8, kv_cache_dtype="fp8")
    try:
        with TestClient(api.build_app(eng, "test-model")) as c:
            loop = next(lp for lp in api._LOOPS if lp.engine is eng)
            c.workers = list(eng._workers)
            aborted = []
            eng.abort = lambda req: aborted.append(req.rid)  # the failure path must never take it
            yield c, eng, loop, aborted
    finally:
        eng.close()
        faulthandler.cancel_dump_traceback_later()


def _completion(c, max_tokens=4):
    return c.post("/v1/completions", json={"prompt": PROMPT, "max_tokens": max_tokens, "temperature": 0,
                                           "ignore_eos": True})


def _stream_events(c, max_tokens=200) -> list[dict]:
    body = {"prompt": PROMPT, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True, "stream": True}
    events = []
    with c.stream("POST", "/v1/completions", json=body) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            events.append(json.loads(line[6:]))
    return events


def _failed(c, loop, eng, thread_ends=True):
    """The state every failure must end in: /health 503 with the reason, new requests 503, every worker process ended
    and (unless it is blocked in a collective with the dead rank) the loop's thread gone."""
    h = c.get("/health")
    assert h.status_code == 503, h.text
    assert h.json()["status"] == "failed"
    assert _completion(c).status_code == 503
    if thread_ends:
        loop.thread.join(60)
        assert not loop.thread.is_alive()
    assert eng._workers == [] and all(w.exitcode is not None for w in c.workers)
    return h.json()["error"]


def _wait_503(c, timeout=60.0):
    end = time.monotonic() + timeout
    while c.get("/health").status_code != 503:
        assert time.monotonic() < end, "/health still answers 200 with a dead rank"
        time.sleep(0.1)


def test_a_worker_that_dies_while_idle_turns_health_503(tmp_path):
    """Turns red if /health or request admission stops looking at the workers' processes, or the loop leaves a dead
    engine's live workers running."""
    with _server(tmp_path) as (c, eng, loop, aborted):
        assert c.get("/health").status_code == 200
        assert _completion(c).status_code == 200
        eng._workers[0].kill()
        _wait_503(c)
        assert "rank(s) [1] exited" in _failed(c, loop, eng)
        assert aborted == []


def test_a_failed_step_under_tp_answers_the_request_and_never_aborts(tmp_path):
    """Rank 0's step raises first, its workers still alive. Turns red if the failure path goes back to aborting the
    open requests (an abort's state offload broadcasts into the workers), if the open request is left waiting, or if
    /health keeps answering 200."""
    with _server(tmp_path) as (c, eng, loop, aborted):
        real, calls = eng.step, [0]

        def step():
            calls[0] += 1
            if calls[0] == 5:
                raise RuntimeError("Failed to schedule neff execution. status=7 message=Execution Queue Full")
            return real()

        eng.step = step
        events = _stream_events(c)
        assert "Execution Queue Full" in events[-1]["error"]["message"]
        assert "Execution Queue Full" in _failed(c, loop, eng)
        assert aborted == []


def test_a_worker_killed_mid_request_ends_the_request_with_the_error(tmp_path):
    """The real thing on CPU: SIGKILL the worker between two steps of a 200-token request, so rank 0's next broadcast
    meets a dead peer. That broadcast blocks on macOS (gloo) and raises on Linux: either way the request ends with the
    error (EngineLoop._watch_ranks when it blocks). Turns red if it hangs, or if a blocked step keeps the server
    answering 200 (requests sent into a dead engine)."""
    with _server(tmp_path) as (c, eng, loop, aborted):
        real, calls = eng.step, [0]

        def step():  # from inside the step: the TestClient hands a stream over only once it has ended
            calls[0] += 1
            if calls[0] == 5:
                eng._workers[0].kill()
                eng._workers[0].join(10)
            return real()

        eng.step = step
        events = _stream_events(c)
        assert "error" in events[-1], events[-1]
        assert "rank(s) [1] exited" in _failed(c, loop, eng, thread_ends=False)
        assert aborted == []
