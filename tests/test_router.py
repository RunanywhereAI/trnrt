"""The router's prefix tree and its sessionless policy (trnrt/serve/router.py), and its connection pool under many
concurrent streams."""

import random

import pytest

for _m in ("httpx", "fastapi", "uvicorn"):  # the router's stack
    pytest.importorskip(_m)

from trnrt.serve.router import PrefixTree, Router  # noqa: E402


def test_prefix_tree_match_insert_and_prune():
    t = PrefixTree(max_chars=1000)
    t.insert("You are a helpful assistant. Q: what is 2+2?")
    t.insert("You are a helpful assistant. Q: name a color")
    assert t.match("You are a helpful assistant. Q: what is 3+3?") == len("You are a helpful assistant. Q: what is ")
    assert t.match("Something else") == 0
    small = PrefixTree(max_chars=50)
    for i in range(20):
        small.insert(f"prefix-{i}-" + "x" * 10)
    assert small.size <= 50


def test_router_keeps_a_shared_prefix_on_one_worker():
    r = Router(["a", "b", "c"], cache_threshold=0.5)
    system = "SYSTEM: you are an expert SQL assistant. " * 10
    first = r.pick(system + "query 0")
    for i in range(1, 30):
        assert r.pick(system + f"query {i}") == first
    assert r.routed["cache"] == 29


def test_router_spreads_unrelated_requests_and_respects_load():
    rng = random.Random(0)
    r = Router(["a", "b"], cache_threshold=0.5, balance_abs=4)
    seen = {r.pick("".join(rng.choice("abcdefghij") for _ in range(40))) for _ in range(20)}
    assert seen == {0, 1}
    r.load = [10, 0]
    assert r.pick("anything") == 1 and r.routed["balance"] == 1


def test_router_does_not_cap_streams_in_flight():
    """Every concurrent stream reaches a worker: the router's connection pool is not a cap on requests in flight
    (httpx's default, 100 connections shared by every worker, held a whole host at 100 in flight and queued the rest
    inside the router). Two real HTTP workers on loopback hold each stream open until all of them have arrived; 130
    streams go through the router at once."""
    import asyncio
    import socket

    import httpx
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    from trnrt.serve.router import build_router_app

    n = 130

    def free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    async def main() -> int:
        seen = {"open": 0, "max": 0}
        release = asyncio.Event()
        worker = FastAPI()

        @worker.post("/v1/completions")
        async def completions():
            seen["open"] += 1
            seen["max"] = max(seen["max"], seen["open"])
            if seen["open"] == n:
                release.set()

            async def body():
                try:
                    yield b"data: {}\n\n"
                    await release.wait()
                    yield b"data: [DONE]\n\n"
                finally:
                    seen["open"] -= 1

            return StreamingResponse(body(), media_type="text/event-stream")

        ports = [free_port() for _ in range(3)]
        apps = [worker, worker, build_router_app(Router([f"http://127.0.0.1:{p}" for p in ports[:2]]))]
        servers = [uvicorn.Server(uvicorn.Config(a, host="127.0.0.1", port=p, log_level="warning"))
                   for a, p in zip(apps, ports)]
        tasks = [asyncio.create_task(s.serve()) for s in servers]
        try:
            while not all(s.started for s in servers):
                await asyncio.sleep(0.05)
            lim = httpx.Limits(max_connections=None, max_keepalive_connections=None)
            async with httpx.AsyncClient(timeout=60, limits=lim) as c:

                async def one(i: int) -> str:
                    body = {"prompt": f"request {i} " + "x" * (i % 7), "max_tokens": 1, "stream": True}
                    async with c.stream("POST", f"http://127.0.0.1:{ports[2]}/v1/completions", json=body) as r:
                        return [line async for line in r.aiter_lines() if line.startswith("data: ")][-1]

                calls = asyncio.gather(*(one(i) for i in range(n)))
                try:
                    await asyncio.wait_for(release.wait(), timeout=20)
                except asyncio.TimeoutError:
                    release.set()  # let the streams that did arrive finish, then report how many there were
                lasts = await calls
            assert lasts == ["data: [DONE]"] * n
            return seen["max"]
        finally:
            for s in servers:
                s.should_exit = True
            await asyncio.gather(*tasks)

    assert asyncio.run(main()) == n


def test_router_spreads_unrelated_requests_once_the_trees_are_full():
    """Requests that share no prefix go to the least-loaded worker even after both prefix trees reached
    max_tree_chars. By tree size alone the worker that just received a request was pruned back below the other
    one, won every following request, and only an imbalance over balance_abs moved one."""
    import json

    rng = random.Random(1)
    r = Router(["a", "b"], cache_threshold=0.5, max_tree_chars=200_000)
    for _ in range(64):  # 64 in flight, none finished: the router's load counts every one of them
        w = r.pick(json.dumps([rng.randrange(1000, 150_000) for _ in range(8192)]))
        r.load[w] += 1
    assert all(t.size <= 200_000 for t in r.trees) and max(t.size for t in r.trees) > 150_000  # the trees are full
    assert abs(r.load[0] - r.load[1]) <= 1, r.load


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _start(apps_ports):
    import asyncio

    import uvicorn

    servers = [uvicorn.Server(uvicorn.Config(a, host="127.0.0.1", port=p, log_level="warning")) for a, p in apps_ports]
    tasks = [asyncio.create_task(s.serve()) for s in servers]
    while not all(s.started for s in servers):
        await asyncio.sleep(0.05)
    return servers, tasks


async def _stop(servers, tasks, i=None):
    import asyncio

    for j, s in enumerate(servers):
        if i is None or j == i:
            s.should_exit = True
    await asyncio.gather(*(tasks if i is None else [tasks[i]]), return_exceptions=True)


def _stub_worker(i: int, chat_status: int = 200):
    """A worker that answers /health, /v1/models and /v1/chat/completions (with chat_status: 503 is a failed engine,
    whose serve/api.py refuses every request before it queues it), streaming or not."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse

    app = FastAPI()
    app.state.chats, app.state.status = 0, chat_status

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "glm", "object": "model", "owned_by": "trnrt"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        app.state.chats += 1
        if body.get("bad"):
            raise HTTPException(status_code=400, detail="bad request")
        if app.state.status != 200:
            raise HTTPException(status_code=app.state.status, detail="engine failed: rank 3 exited")
        if body.get("stream"):
            async def stream():
                yield ('data: {"choices": [{"delta": {"content": "w%d"}}]}\n\n' % i).encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")
        return {"choices": [{"message": {"role": "assistant", "content": f"w{i}"}}]}

    return app


def test_router_status_follows_its_view_of_the_workers():
    r = Router(["a", "b"])
    assert r.status() == "ok"
    r.mark_down(0)
    assert r.status() == "degraded"
    r.mark_down(1)
    assert r.status() == "down"
    r.note_ok(1)
    assert r.status() == "degraded"


def test_router_health_and_models_follow_the_workers():
    """/health: 200 "ok" with both workers up, 200 "degraded" naming the one down, 503 "down" with neither. /v1/models
    answers from a live worker when the one it would ask first is dead (the router not knowing yet: no poller here), and
    503 once none answers."""
    import asyncio

    import httpx

    from trnrt.serve.router import build_router_app

    async def main() -> None:
        ports = [_free_port() for _ in range(3)]
        router = Router([f"http://127.0.0.1:{p}" for p in ports[:2]])
        servers, tasks = await _start([(_stub_worker(0), ports[0]), (_stub_worker(1), ports[1]),
                                       (build_router_app(router, poll_s=0), ports[2])])
        base = f"http://127.0.0.1:{ports[2]}"
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                h = await c.get(base + "/health")
                assert h.status_code == 200 and h.json()["status"] == "ok" and h.json()["down"] == [], h.json()
                m = await c.get(base + "/v1/models")
                assert m.status_code == 200 and m.json()["data"][0]["id"] == "glm"
                await _stop(servers, tasks, 0)  # worker 0, the first asked, dies
                m = await c.get(base + "/v1/models")
                assert m.status_code == 200 and m.json()["data"][0]["id"] == "glm", m.text
                h = await c.get(base + "/health")  # the refused connection marked worker 0 down
                assert h.status_code == 200 and h.json()["status"] == "degraded", h.json()
                assert h.json()["down"] == [router.workers[0]] and h.json()["up"] == [False, True]
                await _stop(servers, tasks, 1)
                m = await c.get(base + "/v1/models")
                assert m.status_code == 503, m.text
                h = await c.get(base + "/health")
                assert h.status_code == 503 and h.json()["status"] == "down", h.json()
                assert h.json()["down"] == router.workers
        finally:
            await _stop(servers, tasks)

    asyncio.run(main())


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
@pytest.mark.parametrize("body", [b"{not json", b"[1, 2]", b'"text"', b"null", b"true", b"3", b"", b"\xff\xfe"])
def test_a_body_that_is_not_a_json_object_is_answered_400_and_reaches_no_worker(path, body):
    """The router refuses a malformed body, or JSON that is not an object, with 400, as each engine does, instead of
    raising inside the proxy (500). No worker sees the request."""
    import asyncio

    import httpx

    from trnrt.serve.router import build_router_app

    async def main() -> None:
        ports = [_free_port() for _ in range(2)]
        router, worker = Router([f"http://127.0.0.1:{ports[0]}"]), _stub_worker(0)
        servers, tasks = await _start([(worker, ports[0]), (build_router_app(router, poll_s=0), ports[1])])
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(f"http://127.0.0.1:{ports[1]}{path}", content=body,
                                 headers={"Content-Type": "application/json"})
                assert r.status_code == 400, r.text
                assert r.json()["detail"] == "the request body must be a JSON object"
                assert router.load == [0] and worker.state.chats == 0
        finally:
            await _stop(servers, tasks)

    asyncio.run(main())


@pytest.mark.parametrize("stream", [False, True])
def test_a_503_from_a_request_marks_its_worker_down_and_the_request_goes_to_another(stream):
    """A failed engine answers every request 503 (its /health too, but the poller is off here): the router marks it down
    at once and sends the request to the live worker, so the client gets its answer and the session fails over. A 400
    (the client's error) passes through and marks nothing; a 503 from every worker reaches the client."""
    import asyncio
    import json

    import httpx

    from trnrt.serve.router import build_router_app

    async def main() -> None:
        ports = [_free_port() for _ in range(3)]
        router = Router([f"http://127.0.0.1:{p}" for p in ports[:2]])
        failed, live = _stub_worker(0, chat_status=503), _stub_worker(1)
        servers, tasks = await _start([(failed, ports[0]), (live, ports[1]),
                                       (build_router_app(router, poll_s=0), ports[2])])
        base = f"http://127.0.0.1:{ports[2]}"

        def body(sid: str, **kw) -> dict:
            return {"model": "glm", "stream": stream, "session_id": sid,
                    "messages": [{"role": "user", "content": sid}], **kw}

        async def ask(c, b) -> tuple[int, str]:
            r = await c.post(base + "/v1/chat/completions", json=b)
            if r.status_code != 200:
                return r.status_code, r.text
            if stream:
                data = [x[6:] for x in r.text.splitlines() if x.startswith("data: ") and x != "data: [DONE]"]
                return 200, json.loads(data[0])["choices"][0]["delta"]["content"]
            return 200, r.json()["choices"][0]["message"]["content"]

        try:
            async with httpx.AsyncClient(timeout=30) as c:
                code, _ = await ask(c, body("s0", bad=True))  # control: a client error marks no worker down
                assert code == 400 and router.up == [True, True] and failed.state.chats == 1
                router.sessions.clear()
                assert await ask(c, body("s1")) == (200, "w1")  # pinned to worker 0 (both idle), which answers 503
                assert router.up == [False, True] and failed.state.chats == 2 and live.state.chats == 1
                assert router.sessions["s1"] == 1 and router.routed["failover"] == 1, router.routed
                assert await ask(c, body("s1")) == (200, "w1") and failed.state.chats == 2  # and stays there
                assert router.load == [0, 0]
                live.state.status = 503  # the other engine fails too: the second worker's 503 reaches the client
                router.note_ok(0)
                code, text = await ask(c, body("s2"))
                assert code == 503 and "engine failed" in text and router.up == [False, False], (code, text)
                assert router.load == [0, 0]
        finally:
            await _stop(servers, tasks)

    asyncio.run(main())
