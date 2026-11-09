"""The router's SGLang native routes (trnrt/serve/router.py): /generate routed like a completion by its prompt or
session, and the info, load, health, abort and flush routes sent to one worker or to all and their answers combined."""

import asyncio
import json

import pytest

for _m in ("httpx", "fastapi", "uvicorn"):
    pytest.importorskip(_m)

from tests.test_router import _free_port, _start, _stop  # noqa: E402  after the importorskip of its stack
from trnrt.serve.router import Router, _request_text, _session_key, build_router_app  # noqa: E402  the same


def test_generate_prompts_are_the_routing_text():
    """/generate's `text` or `input_ids` (a batch's list too) is what the prefix trees match, as a completion's
    `prompt`; without this every /generate looked like an empty prompt."""
    assert _request_text("/generate", {"text": "You are helpful. Q: 2+2?"}) == "You are helpful. Q: 2+2?"
    assert _request_text("/generate", {"input_ids": [5, 6, 7]}) == "[5, 6, 7]"
    assert _request_text("/generate", {"text": ["a b", "c"]}) == json.dumps(["a b", "c"])
    assert _request_text("/generate", {"input_ids": [[1, 2], [3]]}) == "[[1, 2], [3]]"
    assert _request_text("/v1/completions", {"prompt": "x", "text": "y"}) == "x"
    assert _session_key("/generate", {"text": "abc"}) == _session_key("/v1/completions", {"prompt": "abc"})
    assert _session_key("/generate", {"text": "abc"}) != _session_key("/generate", {"text": "abd"})


def _native_worker(i: int, flush_status: int = 200, health_status: int = 200):
    """A worker with SGLang's native routes: it answers each with its index, and records what it was sent. (No
    `from __future__ import annotations` in this module: FastAPI resolves a handler's `req: Request` in the module's
    globals, where this local import is not, and would take `req` for a query parameter.)"""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, PlainTextResponse, Response

    app = FastAPI()
    app.state.seen = []

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/generate")
    async def generate(req: Request):
        body = await req.json()
        app.state.seen.append(("generate", body))
        return {"text": f"w{i}", "output_ids": [i], "meta_info": {"id": body.get("rid", "x")}}

    @app.get("/get_load")
    async def get_load():
        return [{"dp_rank": 0, "num_reqs": i + 1, "num_waiting_reqs": 0, "num_tokens": 10 * i}]

    @app.get("/model_info")
    async def model_info():
        app.state.seen.append(("model_info", None))
        return {"model_path": "org/glm", "worker": i}

    @app.get("/server_info")
    async def server_info():
        return {"tp_size": 32, "dp_size": 1, "worker": i, "internal_states": [{"dp_rank": 0, "num_reqs": i}]}

    @app.get("/health_generate")
    async def health_generate():
        if health_status != 200:
            return JSONResponse(status_code=health_status, content={"status": "failed", "error": "stuck"})
        return {"status": "ok"}

    @app.post("/abort_request")
    async def abort_request(req: Request):
        body = await req.json()
        app.state.seen.append(("abort", body))
        if not body.get("rid") and not body.get("abort_all"):
            return JSONResponse(status_code=400, content={"detail": "rid is required"})
        return Response(status_code=200)

    @app.api_route("/flush_cache", methods=["GET", "POST"])
    async def flush_cache(req: Request):
        app.state.seen.append(("flush", req.method, req.url.query))
        if flush_status != 200:
            return PlainTextResponse("Cache not flushed: 2 requests are running or waiting.\n", status_code=400)
        return PlainTextResponse("Cache flushed.\n")

    return app


def _run(workers, check, router_kw=None) -> None:
    async def main() -> None:
        ports = [_free_port() for _ in range(len(workers) + 1)]
        router = Router([f"http://127.0.0.1:{p}" for p in ports[:-1]], **(router_kw or {}))
        apps = [(w, p) for w, p in zip(workers, ports) if w is not None]
        servers, tasks = await _start([*apps, (build_router_app(router, poll_s=0), ports[-1])])
        try:
            import httpx

            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{ports[-1]}", timeout=60) as c:
                await check(c, router)
        finally:
            await _stop(servers, tasks)

    asyncio.run(main())


def test_generate_goes_through_the_prefix_routed_proxy():
    """A /generate is routed like a completion: a repeated long prompt (one prompt or a batch) stays on the worker
    that has its prefix, a session's requests on its pin; the body reaches the worker whole."""
    w0, w1 = _native_worker(0), _native_worker(1)

    async def check(c, router) -> None:
        long = "You are a careful assistant. " * 40
        r = await c.post("/generate", json={"text": long + "Q1", "sampling_params": {"max_new_tokens": 4}})
        assert r.status_code == 200, r.text
        first = r.json()["text"]
        for q in range(2, 6):
            r = await c.put("/generate", json={"text": long + f"Q{q}"})
            assert r.json()["text"] == first
        assert router.routed["cache"] == 4 and router.load == [0, 0]
        batch = {"input_ids": [[7] * 300 + [1], [7] * 300 + [2]], "rid": "b"}
        w = (await c.post("/generate", json=batch)).json()["text"]
        assert (await c.post("/generate", json={**batch, "rid": "c"})).json()["text"] == w
        assert ("generate", batch) in (w0 if w == "w0" else w1).state.seen
        r = await c.post("/generate", json={"text": "hi", "session_id": "s9"})
        assert router.sessions["s9"] == int(r.json()["text"][1])

    _run([w0, w1], check)


def test_info_and_load_routes_combine_the_workers():
    """model_info: one live worker's answer. get_load: every worker's entry with dp_rank its index. server_info: the
    first answer, dp_size the workers, internal_states every worker's. A dead worker is skipped (and marked down)."""
    w1, w2 = _native_worker(1), _native_worker(2)

    async def check(c, router) -> None:
        r = await c.get("/get_load")
        assert r.status_code == 200, r.text
        assert [(e["dp_rank"], e["num_reqs"]) for e in r.json()] == [(1, 2), (2, 3)]
        assert router.up == [False, True, True]
        for path in ("/server_info", "/get_server_info"):
            info = (await c.get(path)).json()
            assert info["dp_size"] == 3 and info["tp_size"] == 32 and info["worker"] == 1
            assert [(s["dp_rank"], s["num_reqs"]) for s in info["internal_states"]] == [(1, 1), (2, 2)]
        for path in ("/model_info", "/get_model_info"):
            assert (await c.get(path)).json()["model_path"] == "org/glm"
        assert len([s for s in w1.state.seen + w2.state.seen if s[0] == "model_info"]) == 2  # one worker each time

    _run([None, w1, w2], check)


def test_health_generate_follows_every_worker():
    async def check(c, router) -> None:
        r = await c.get("/health_generate")
        assert r.status_code == 200 and r.json()["status"] == "degraded", r.text
        assert r.json()["down"] == [router.workers[1]] and router.up == [True, False]

    _run([_native_worker(0), _native_worker(1, health_status=503)], check)

    async def none(c, router) -> None:
        r = await c.get("/health_generate")
        assert r.status_code == 503 and r.json()["status"] == "down"

    _run([None, _native_worker(1, health_status=503)], none)

    async def both(c, router) -> None:
        router.up = [False, False]
        r = await c.get("/health_generate")
        assert r.status_code == 200 and r.json() == {"status": "ok", "down": []}
        assert router.up == [True, True]  # each one generated: up

    _run([_native_worker(0), _native_worker(1)], both)


def test_abort_request_reaches_every_worker():
    """A rid's request is on one worker, which the router need not know: every worker gets the abort. A malformed abort
    is refused as the workers refuse it."""
    w0, w1 = _native_worker(0), _native_worker(1)

    async def check(c, router) -> None:
        for body in ({"rid": "r1"}, {"abort_all": True}):
            r = await c.post("/abort_request", json=body)
            assert r.status_code == 200, r.text
            assert ("abort", body) in w0.state.seen and ("abort", body) in w1.state.seen
        r = await c.post("/abort_request", json={})
        assert r.status_code == 400 and "rid" in r.text
        r = await c.post("/abort_request", content=b"[1]", headers={"content-type": "application/json"})
        assert r.status_code == 400

    _run([w0, w1], check)


def test_flush_cache_flushes_every_worker_and_their_prefix_trees():
    """200 only when every worker flushed; the router forgets a flushed worker's prefixes (its tree) but keeps the
    others', and the query (timeout) reaches the workers."""
    w0, w1 = _native_worker(0), _native_worker(1, flush_status=400)

    async def check(c, router) -> None:
        for _ in range(2):
            await c.post("/generate", json={"text": "x" * 500})
        await c.post("/generate", json={"text": "y" * 500})
        router.trees[0].insert("z" * 100)
        router.trees[1].insert("z" * 100)
        r = await c.get("/flush_cache?timeout=3")
        assert r.status_code == 400 and "engine 1: Cache not flushed: 2 requests" in r.text, r.text
        assert router.trees[0].size == 0 and router.trees[1].size > 0
        assert ("flush", "GET", "timeout=3") in w0.state.seen and ("flush", "GET", "timeout=3") in w1.state.seen

    _run([w0, w1], check)

    async def ok(c, router) -> None:
        router.trees[0].insert("z" * 100)
        r = await c.post("/flush_cache")
        assert r.status_code == 200 and r.text == "Cache flushed.\n"
        assert all(t.size == 0 for t in router.trees)

    _run([_native_worker(0), _native_worker(1)], ok)
