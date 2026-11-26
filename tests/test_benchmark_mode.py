"""The benchmark mode (synthetic acceptance) announces itself (trnrt/serve/benchmark_mode.py): the engine's and the router's
responses carry the x-trnrt-synthetic-acceptance header, /health carries a `benchmark_mode` block, and both warn in the
log. Without the mode nothing changes: no header, the same /health body."""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # fastapi.testclient

from trnrt.serve import benchmark_mode
from trnrt.serve.router import Router

AL = 2.518506
H = benchmark_mode.HEADER


def _engine_client(synth_al):
    from fastapi.testclient import TestClient

    from tests.test_serve_api import PLAIN, ScriptedEngine
    from trnrt.serve import api

    eng = ScriptedEngine(PLAIN)
    if synth_al is not None:
        eng.cfg.spec_synth_al = synth_al
    return TestClient(api.build_app(eng, "glm")), eng


def _stop(eng):
    from trnrt.serve import api

    for lp in [lp for lp in api._LOOPS if lp.engine is eng]:
        lp.stop()


def test_the_engine_announces_the_mode_on_every_response_and_in_health(caplog):
    caplog.set_level(logging.WARNING)
    client, eng = _engine_client(AL)
    try:
        with client:
            assert "BENCHMARK MODE" in caplog.text and "2.51851" in caplog.text
            h = client.get("/health")
            assert h.status_code == 200 and h.headers[H] == "2.51851"
            block = h.json()["benchmark_mode"]
            assert block["synthetic_acceptance_length"] == AL and block["drafts_verified"] is False
            assert client.get("/v1/models").headers[H] == "2.51851"
            r = client.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 3})
            assert r.status_code == 200 and r.headers[H] == "2.51851"
            with client.stream(
                "POST", "/v1/completions", json={"prompt": "Hello", "max_tokens": 3, "stream": True}
            ) as s:
                assert s.status_code == 200 and s.headers[H] == "2.51851"
                lines = [ln for ln in s.iter_lines() if ln.startswith("data: ")]
            assert lines[-1] == "data: [DONE]"  # the stream itself is untouched
    finally:
        _stop(eng)


def test_a_failed_engine_still_announces_the_mode_in_its_503():
    from trnrt.serve import api

    client, eng = _engine_client(AL)
    try:
        with client:
            loop = next(lp for lp in api._LOOPS if lp.engine is eng)
            loop.health_error = lambda: "rank 3 died"  # as EngineLoop._fail leaves it
            h = client.get("/health")
            assert h.status_code == 503 and h.json()["status"] == "failed" and h.json()["error"] == "rank 3 died"
            assert h.json()["benchmark_mode"]["synthetic_acceptance_length"] == AL and h.headers[H] == "2.51851"
    finally:
        _stop(eng)


def test_without_the_mode_the_engine_adds_nothing():
    client, eng = _engine_client(None)
    try:
        with client:
            h = client.get("/health")
            assert h.json() == {"status": "ok"} and H not in h.headers
            r = client.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 3})
            assert r.status_code == 200 and H not in r.headers
    finally:
        _stop(eng)


def _stats(al):
    return {"gauges": {"kv_pages_used": 10, "kv_pages_total": 100, "spec": {"synth_al": al}}}


def test_the_router_takes_the_mode_from_its_workers_stats(caplog):
    from fastapi.testclient import TestClient

    from trnrt.serve.router import build_router_app

    caplog.set_level(logging.WARNING)
    router = Router(["http://127.0.0.1:9", "http://127.0.0.1:10"])
    with TestClient(build_router_app(router, poll_s=0)) as c:
        h = c.get("/health")
        assert H not in h.headers and "benchmark_mode" not in h.json()  # before any poll saw the mode

        router.note_stats(0, _stats(None))
        router.note_stats(1, {"gauges": {"kv_pages_used": 1, "kv_pages_total": 4}})  # no spec block at all
        assert router.benchmark_lengths() == [] and router.kv_used == [0.1, 0.25]
        assert H not in c.get("/health").headers

        router.note_stats(1, _stats(AL))
        router.note_stats(1, _stats(AL))  # warned once per change, not per poll
        assert caplog.text.count("BENCHMARK MODE") == 1
        assert "trnrt.router: worker http://127.0.0.1:10: " in caplog.text  # the router's own log prefix
        h = c.get("/health")
        assert h.headers[H] == "2.51851" and h.json()["benchmark_mode"]["synthetic_acceptance_length"] == AL
        assert c.get("/v1/models").headers[H] == "2.51851"  # a 503 (no worker answers) says it too

        router.note_stats(0, _stats(3.0))  # two lengths at once: both named
        assert c.get("/health").headers[H] == "2.51851,3"

        router.note_stats(0, _stats(None))
        router.note_stats(1, _stats(None))  # restarted without the mode
        assert H not in c.get("/health").headers


def test_the_header_middleware_replaces_a_header_of_the_same_name():
    from fastapi import FastAPI, Response
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.get("/x")
    async def x():
        return Response("ok", headers={H: "stale"})

    app.add_middleware(benchmark_mode.HeaderMiddleware, value=lambda: "2.5")
    r = TestClient(app).get("/x")
    assert r.headers.get_list(H) == ["2.5"]


def test_the_banner_says_what_the_text_is_not():
    b = benchmark_mode.banner(AL)
    assert "unverified" in b and "not the model's" in b and "never to judge output quality" in b
