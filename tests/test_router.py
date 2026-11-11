"""The router's prefix tree and its sessionless policy (trnrt/serve/router.py), and its connection pool under many
concurrent streams."""

import contextlib
import random
import sys

import pytest

for _m in ("httpx", "fastapi", "uvicorn"):  # the router's stack
    pytest.importorskip(_m)

from trnrt.serve.router import STATELESS_ROUTES, PrefixTree, Router  # noqa: E402


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


@pytest.mark.skipif(sys.platform == "darwin", reason="130 connections at once exceed macOS's listen backlog of 128")
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
        servers = [
            uvicorn.Server(uvicorn.Config(a, host="127.0.0.1", port=p, log_level="warning"))
            for a, p in zip(apps, ports)
        ]
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
                except TimeoutError:
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
    """A worker that answers /health, /v1/models, /v1/chat/completions and STATELESS_ROUTES (with chat_status: 503 is a
    failed engine, whose serve/api.py refuses every request before it queues it), streaming or not."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse

    app = FastAPI()
    app.state.chats, app.state.toks, app.state.status = 0, 0, chat_status

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
                yield ('data: {"choices": [{"delta": {"content": "w' + str(i) + '"}}]}\n\n').encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")
        return {"choices": [{"message": {"role": "assistant", "content": f"w{i}"}}]}

    async def stateless(req: Request):
        app.state.toks += 1
        if app.state.status != 200:
            raise HTTPException(status_code=app.state.status, detail="engine failed: rank 3 exited")
        return {"worker": i, "body": await req.json()}

    for path in STATELESS_ROUTES:
        app.add_api_route(path, stateless, methods=["POST"])
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
        servers, tasks = await _start(
            [(_stub_worker(0), ports[0]), (_stub_worker(1), ports[1]), (build_router_app(router, poll_s=0), ports[2])]
        )
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


@pytest.mark.parametrize(
    "path", ["/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages", "/tokenize", "/v1/detokenize"]
)
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
                r = await c.post(
                    f"http://127.0.0.1:{ports[1]}{path}", content=body, headers={"Content-Type": "application/json"}
                )
                assert r.status_code == 400, r.text
                assert r.json()["detail"] == "the request body must be a JSON object"
                assert router.load == [0] and worker.state.chats == 0 and worker.state.toks == 0
        finally:
            await _stop(servers, tasks)

    asyncio.run(main())


@pytest.mark.parametrize("path", STATELESS_ROUTES)
def test_tokenize_and_detokenize_go_to_a_live_worker_and_change_no_routing_state(path):
    """STATELESS_ROUTES go to the least loaded live worker and touch no prefix tree, no session and no load count: a
    /tokenize of an agent's context must not steer where its next turn goes. A refused connection or a 503 marks the
    worker down and the next one is asked; 503 once none answers."""
    import asyncio

    import httpx

    from trnrt.serve.router import build_router_app

    async def main() -> None:
        ports = [_free_port() for _ in range(4)]  # ports[0] serves nothing: a dead worker
        router = Router([f"http://127.0.0.1:{p}" for p in ports[:3]])
        failed, ok = _stub_worker(1, chat_status=503), _stub_worker(2)
        servers, tasks = await _start(
            [(failed, ports[1]), (ok, ports[2]), (build_router_app(router, poll_s=0), ports[3])]
        )
        body = {"messages": [{"role": "user", "content": "x" * 64}], "session_id": "s1"}
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(f"http://127.0.0.1:{ports[3]}{path}", json=body)
                assert r.status_code == 200, r.text
                assert r.json() == {"worker": 2, "body": body}
                assert router.up == [False, False, True]  # refused, then 503: both marked down
                assert failed.state.toks == 1 and ok.state.toks == 1
                assert router.sessions == {} and all(t.size == 0 for t in router.trees) and router.load == [0, 0, 0]
                await _stop(servers, tasks, 1)
                r = await c.post(f"http://127.0.0.1:{ports[3]}{path}", json=body)
                assert r.status_code == 503 and r.json()["detail"] == f"no engine answered {path}", r.text
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
        servers, tasks = await _start(
            [(failed, ports[0]), (live, ports[1]), (build_router_app(router, poll_s=0), ports[2])]
        )
        base = f"http://127.0.0.1:{ports[2]}"

        def body(sid: str, **kw) -> dict:
            return {
                "model": "glm",
                "stream": stream,
                "session_id": sid,
                "messages": [{"role": "user", "content": sid}],
                **kw,
            }

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


def test_pick_off_loop_routes_as_pick_does():
    """The same sequence of sessionless requests and loads gives the same workers, routing counters and trees through
    pick (on the event loop) and pick_off_loop (the trees in a worker thread)."""
    import asyncio

    rng = random.Random(3)
    shared = ["".join(rng.choices("abcdefgh", k=rng.randint(200, 40_000))) for _ in range(6)]
    reqs = []
    for _ in range(300):
        text = rng.choice(shared) + "".join(rng.choices("xyz", k=rng.randint(0, 3000)))
        reqs.append((text, rng.choice((0, 0, 100)), [rng.randint(0, 40) for _ in range(3)]))

    def new() -> Router:
        return Router(["a", "b", "c"], max_tree_chars=200_000, sticky_chars=1_000)

    a, b = new(), new()
    want = []
    for text, head, load in reqs:
        a.load = list(load)
        want.append(a.pick(text, head=head))

    async def main() -> list[int]:
        got = []
        for text, head, load in reqs:
            b.load = list(load)
            got.append(await b.pick_off_loop(text, head=head))
        return got

    assert asyncio.run(main()) == want
    assert a.routed == b.routed and [t.size for t in a.trees] == [t.size for t in b.trees]
    assert len(set(want)) > 1  # the sequence exercises more than one rule


def test_the_prefix_trees_are_walked_off_the_event_loop(monkeypatch):
    """A sessionless request's match and insert walk its text character by character (~300 ms for 1.1 MB, ~250K
    tokens): on the router's event loop, they stalled every stream it relays. Through the app, both run in another
    thread than the loop's."""
    import asyncio
    import threading

    import httpx

    from trnrt.serve import router as router_mod

    threads: list[tuple[str, int | None]] = []
    real_match, real_insert = router_mod.PrefixTree.match, router_mod.PrefixTree.insert

    def match(self, s):
        threads.append(("match", threading.get_ident()))
        return real_match(self, s)

    def insert(self, s):
        threads.append(("insert", threading.get_ident()))
        return real_insert(self, s)

    monkeypatch.setattr(router_mod.PrefixTree, "match", match)
    monkeypatch.setattr(router_mod.PrefixTree, "insert", insert)

    async def main() -> int:
        ports = [_free_port() for _ in range(3)]
        router = Router([f"http://127.0.0.1:{p}" for p in ports[:2]])
        servers, tasks = await _start(
            [
                (_stub_worker(0), ports[0]),
                (_stub_worker(1), ports[1]),
                (router_mod.build_router_app(router, poll_s=0), ports[2]),
            ]
        )
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                for stream in (False, True):
                    body = {"messages": [{"role": "user", "content": "hello " * 1000}], "stream": stream}
                    r = await c.post(f"http://127.0.0.1:{ports[2]}/v1/chat/completions", json=body)
                    assert r.status_code == 200, r.text
        finally:
            await _stop(servers, tasks)
        return threading.get_ident()

    loop_thread = asyncio.run(main())
    assert {k for k, _ in threads} == {"match", "insert"}
    assert all(t != loop_thread for _, t in threads), threads


def test_a_plain_client_that_disconnects_from_the_router_aborts_its_request_on_the_worker():
    """A non-streaming request: the worker aborts a request whose client disconnects (serve/api.py), but behind the
    router its client is the router's pooled connection, which stayed open, so the worker generated to max_tokens and
    held a seat. The router now watches its own client and closes the upstream request. The stub worker watches for
    the disconnect as the engine's API does; the control is the same request sent straight to it."""
    import asyncio
    import time

    import httpx
    from fastapi import FastAPI, Request

    from trnrt.serve.router import build_router_app

    seen: list[tuple[str, float]] = []
    worker = FastAPI()

    @worker.get("/health")
    async def health():
        return {"status": "ok"}

    @worker.post("/v1/chat/completions")
    async def chat(req: Request):
        await req.json()
        t0 = time.monotonic()

        async def watch():
            while (await req.receive())["type"] != "http.disconnect":
                pass

        gone, work = asyncio.ensure_future(watch()), asyncio.ensure_future(asyncio.sleep(10))
        await asyncio.wait({gone, work}, return_when=asyncio.FIRST_COMPLETED)
        seen.append(("disconnect" if gone.done() else "ran to the end", time.monotonic() - t0))
        gone.cancel()
        work.cancel()
        return {"choices": [{"message": {"role": "assistant", "content": "x"}}]}

    async def leave_after(url: str, s: float) -> None:
        body = {"messages": [{"role": "user", "content": "hi"}]}
        async with httpx.AsyncClient(timeout=30) as c:  # the client itself gives up after `s`
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(c.post(url + "/v1/chat/completions", json=body), s)

    async def wait_seen(n: int) -> None:
        deadline = time.monotonic() + 5
        while len(seen) < n and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

    async def main() -> None:
        ports = [_free_port() for _ in range(2)]
        router = Router([f"http://127.0.0.1:{ports[0]}"])
        servers, tasks = await _start([(worker, ports[0]), (build_router_app(router, poll_s=0), ports[1])])
        try:
            await leave_after(f"http://127.0.0.1:{ports[0]}", 0.5)  # the control: straight to the worker
            await wait_seen(1)
            assert seen and seen[0][0] == "disconnect", seen
            await leave_after(f"http://127.0.0.1:{ports[1]}", 0.5)  # through the router
            await wait_seen(2)
            assert len(seen) == 2 and seen[1][0] == "disconnect" and seen[1][1] < 3, seen
            deadline = time.monotonic() + 5
            while router.load != [0] and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            assert router.load == [0]
        finally:
            await _stop(servers, tasks)

    asyncio.run(main())


# One conversation in each API that carries one: a long shared system part, then the turns.
SYSTEM = "You are a coding agent. Environment and file tree follow. " + "src/x.py " * 4500
TURNS = [
    {"role": "user", "content": "task 0: fix the bug"},
    {"role": "assistant", "content": "reading files " * 3000},
]


def _bodies(turns: list) -> dict[str, dict]:
    return {
        "/v1/chat/completions": {"messages": [{"role": "system", "content": SYSTEM}, *turns]},
        "/v1/responses": {"instructions": SYSTEM, "input": turns},
        "/v1/messages": {"system": SYSTEM, "messages": turns, "max_tokens": 8},
    }


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses", "/v1/messages"])
def test_each_conversation_api_routes_by_its_conversation(path):
    """A Responses request's instructions and input, and a Messages request's system and messages, are its
    conversation as a chat's messages are: its routing text grows at its end as the conversation does (so its prefix
    sticks to the worker holding it), its head is the system part (so a shared system prompt does not pin new sessions),
    and its session key is stable while it grows and differs by system prompt."""
    from trnrt.serve.router import _request_head, _request_text, _session_key

    first, later = _bodies(TURNS[:1])[path], _bodies([*TURNS, {"role": "user", "content": "next"}])[path]
    t1, t2 = _request_text(path, first), _request_text(path, later)
    head = _request_head(path, first)
    assert head > 40_000 and SYSTEM[:1000] in t1[:head] and "task 0" not in t1[:head]
    assert t2.startswith(t1[:-1])  # "[A, B" is a prefix of "[A, B, C]"
    assert _request_head(path, later) == head
    assert _session_key(path, first) == _session_key(path, later)
    other = dict(first, **({"instructions": "Write a title."} if path == "/v1/responses" else {}))
    if path == "/v1/messages":
        other["system"] = "Write a title."
    elif path == "/v1/chat/completions":
        other = {"messages": [{"role": "system", "content": "Write a title."}, *TURNS[:1]]}
    assert _session_key(path, other)[0] != _session_key(path, first)[0]
    assert _session_key(path, other)[1] == _session_key(path, first)[1]
    r = Router(["a", "b"], balance_abs=1, balance_rel=1.0, sticky_chars=32_768)
    home = r.pick(_request_text(path, _bodies(TURNS)[path]), head=head)
    r.load[home] += 20  # far past balance_abs: only the sticky rule keeps the conversation home
    assert r.pick(t2, head=head) == home and r.routed["sticky"] == 1


def test_a_session_key_reads_only_the_start_of_a_long_prompt(monkeypatch):
    """A session request's key hashes the first SESSION_KEY_CHARS characters of its prompt's JSON text, on the event
    loop: written from the prompt's start only, never the whole prompt (a 200K-token input_ids list), with the same key
    as the whole prompt's text cut there."""
    import hashlib
    import json

    from trnrt.serve import router as R

    K = R.SESSION_KEY_CHARS
    h = lambda b: hashlib.sha1(b.encode("utf-8", "replace"), usedforsecurity=False).hexdigest()  # noqa: E731
    ids = list(range(200_000))
    prompts = [
        ("/generate", {"input_ids": ids}, ids),
        ("/generate", {"input_ids": [ids, ids]}, [ids, ids]),
        ("/generate", {"input_ids": [[7] * 3, ids]}, [[7] * 3, ids]),  # a short first prompt: the next one is read too
        ("/generate", {"text": ["x" * 100_000, "y"]}, ["x" * 100_000, "y"]),
        ("/v1/completions", {"prompt": ["\u00e9\U0001f642" * 50_000]}, ["\u00e9\U0001f642" * 50_000]),
    ]
    want = [(h(""), h(json.dumps(p)[:K])) for _, _, p in prompts]
    content = [{"type": "text", "text": "z" * 100_000}, {"type": "text", "text": "w"}]
    chat = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": content}]}
    want_chat = h("user\x00" + json.dumps(content, ensure_ascii=False)[:K])
    sizes: list[int] = []
    real = json.dumps
    monkeypatch.setattr(R.json, "dumps", lambda *a, **kw: sizes.append(len(out := real(*a, **kw))) or out)
    assert [R._session_key(path, body) for path, body, _ in prompts] == want
    assert R._session_key("/v1/chat/completions", chat)[1] == want_chat
    assert max(sizes) < 16 * K  # a cut prompt's text (up to 12 characters an item): never the 1.3 MB of the list


def test_the_router_proxies_responses_and_messages_with_their_sessions():
    """/v1/responses and /v1/messages go through the proxy: a session_id pins them as it pins a chat."""
    import asyncio

    import httpx
    from fastapi import FastAPI, Request

    from trnrt.serve.router import build_router_app

    def worker(i: int) -> FastAPI:
        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        async def answer(req: Request):
            return {"worker": i, "body": await req.json()}

        for path in ("/v1/responses", "/v1/messages"):
            app.add_api_route(path, answer, methods=["POST"])
        return app

    async def main() -> None:
        ports = [_free_port() for _ in range(3)]
        router = Router([f"http://127.0.0.1:{p}" for p in ports[:2]])
        servers, tasks = await _start(
            [(worker(0), ports[0]), (worker(1), ports[1]), (build_router_app(router, poll_s=0), ports[2])]
        )
        base = f"http://127.0.0.1:{ports[2]}"
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                for path, conversation in _bodies(TURNS[:1]).items():
                    if path == "/v1/chat/completions":
                        continue
                    body = {**conversation, "session_id": "s-" + path}
                    got = {(await c.post(base + path, json=body)).json()["worker"] for _ in range(3)}
                    assert got == {router.sessions["s-" + path]}, (path, got)
                assert router.load == [0, 0] and all(t.size == 0 for t in router.trees)  # session requests: no tree
        finally:
            await _stop(servers, tasks)

    asyncio.run(main())
