"""Session affinity in trnrt/serve/router.py: an agent session's turns stay on the worker holding its context.

A continuing turn never moves under load imbalance (a move is a cold prefill of a 150-230K-token context), new sessions
balance over the workers, and a session whose worker dies fails over to a live one. Unit tests on Router, then two stub
HTTP workers behind the real router app, one of which is shut down mid-session.
"""

import asyncio
import json
import socket

import pytest

for _m in ("httpx", "fastapi", "uvicorn"):  # the router's stack
    pytest.importorskip(_m)

from trnrt.serve.router import Router  # noqa: E402


def test_a_continuing_session_never_moves_under_imbalance():
    r = Router(["a", "b"], balance_abs=0, balance_rel=1.0)
    first = r.pick("", session="agent-01")
    r.load[first] += 50  # far past any balance rule
    for _ in range(20):
        assert r.pick("", session="agent-01") == first
    assert r.routed["session"] == 20 and r.routed["new_session"] == 1 and r.routed["balance"] == 0
    # the same imbalance does move a request without a session: the old rule, unchanged
    assert r.pick("no session here") != first and r.routed["balance"] == 1


def test_new_sessions_balance_over_the_workers():
    r = Router(["a", "b"])
    seats = [r.pick("", session=f"agent-{i:02d}") for i in range(10)]
    assert seats.count(0) == 5 and seats.count(1) == 5  # pinned, active sessions count as load
    r.load = [3, 0]  # in flight
    assert r.pick("", session="agent-10") == 1
    r2 = Router(["a", "b"])
    r2.kv_used = [0.9, 0.1]  # equal load and sessions: the KV in use breaks the tie
    assert r2.pick("", session="new") == 1


def test_idle_sessions_stop_counting_toward_load():
    now = [0.0]
    r = Router(["a", "b"], session_idle_s=600, clock=lambda: now[0])
    for i in range(4):
        r.sessions[f"old{i}"], r.seen[f"old{i}"] = 0, 0.0
    assert r.active_sessions(0) == 4 and r.pick("", session="fresh") == 1
    now[0] = 601.0  # the old sessions idled out
    assert r.active_sessions(0) == 0
    assert r.pick("", session="fresh2") == 0  # worker 1 now holds an active session, worker 0 none


def test_a_session_fails_over_when_its_worker_is_down():
    r = Router(["a", "b"])
    w = r.pick("", session="s")
    r.mark_down(w)
    moved = r.pick("", session="s")
    assert moved != w and r.routed["failover"] == 1 and r.sessions["s"] == moved
    r.turn_done("s", moved)  # a turn completed there: its context now lives on the new worker
    r.note_ok(w)  # home is back up: the session stays where its context is
    assert r.pick("", session="s") == moved and r.routed["failover"] == 1 and r.routed["home"] == 0
    # a sessionless request never goes to a worker known to be down while another is up
    r.mark_down(moved)
    assert all(r.pick(f"text {i}") == w for i in range(5))


def test_a_failed_over_session_goes_home_if_it_completed_no_turn_away():
    r = Router(["a", "b"])
    w = r.pick("", session="s")
    r.mark_down(w)
    moved = r.pick("", session="s")  # failed over; this request then fails (no turn_done)
    assert moved != w
    r.note_ok(w)
    assert r.pick("", session="s") == w and r.routed["home"] == 1  # its context is still only at home
    assert r.pick("", session="s") == w and r.routed["session"] == 1


def test_one_slow_health_poll_moves_no_session():
    r = Router(["a", "b"], down_after=3)
    w = r.pick("", session="s")
    for _ in range(2):  # slow or failed polls: soft
        r.note_failure(w, hard=False)
        assert r.up[w] and r.pick("", session="s") == w
    r.note_ok(w)  # an answer resets the count
    r.note_failure(w, hard=False)
    r.note_failure(w, hard=False)
    assert r.up[w] and r.routed["failover"] == 0
    r.note_failure(w, hard=False)  # the third in a row: down
    assert not r.up[w] and r.pick("", session="s") != w and r.routed["failover"] == 1
    r2 = Router(["a", "b"])
    w2 = r2.pick("", session="t")
    r2.note_failure(w2, hard=True)  # a refused connection or a 503: down at once
    assert not r2.up[w2] and r2.pick("", session="t") != w2


def test_a_long_shared_prefix_stays_on_its_worker_under_imbalance():
    """No session id (most OpenAI clients send none): the prefix tree, locality first."""
    r = Router(["a", "b"], balance_abs=0, balance_rel=1.0, sticky_chars=32_768)
    convo = "system prompt and a long agent conversation " * 2000  # ~88K chars
    first = r.pick(convo)
    r.load[first] += 50
    for i in range(5):
        assert r.pick(convo + f" turn {i}") == first
    assert r.routed["sticky"] == 5 and r.routed["balance"] == 0
    # a short or new prefix under the same imbalance balances
    assert r.pick("short new prefix") != first and r.routed["balance"] == 1


def test_the_session_map_is_pruned():
    now = [0.0]
    r = Router(["a", "b"], session_idle_s=100, session_ttl_s=400, max_sessions=5, clock=lambda: now[0])
    for i in range(5):
        r.pick("", session=f"s{i}")
    now[0] = 300.0
    r.pick("", session="s0")  # s0 is recent again
    now[0] = 450.0  # s1..s4 idle past the TTL; s0 not
    r.prune()
    assert set(r.sessions) == {"s0"} and set(r.seen) == {"s0"}
    for i in range(10):
        r.pick("", session=f"n{i}")
    assert len(r.sessions) <= 5 + 1  # bounded: least recently used forgotten on each new pin
    assert "n9" in r.sessions and "n0" not in r.sessions


def test_session_requests_skip_the_prefix_tree():
    r = Router(["a", "b"])
    r.pick("x" * 10_000, session="s")
    assert [t.size for t in r.trees] == [0, 0]
    r.pick("y" * 100)
    assert sum(t.size for t in r.trees) == 100


def test_session_of_reads_a_non_empty_string_session_id_only():
    r = Router(["a"])
    assert r.session_of({"session_id": "agent-01"}) == "agent-01"
    assert r.session_of({"session_id": ""}) is None and r.session_of({"session_id": 7}) is None
    assert r.session_of({"messages": []}) is None and r.session_of({"session": "agent-01"}) is None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_end_to_end_affinity_and_failover_through_the_router():
    """Two stub workers behind the real router app. Each session's turns land on one worker; the worker holding a
    session is shut down; its next turn fails over to the other worker and completes; the router reports it."""
    import httpx
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    from trnrt.serve.router import build_router_app

    async def main() -> None:
        got = {0: [], 1: []}

        def worker(i: int) -> FastAPI:
            app = FastAPI()

            @app.get("/health")
            async def health():
                return {"status": "ok"}

            @app.post("/v1/chat/completions")
            async def chat(req: Request):
                body = await req.json()
                got[i].append(body.get("session_id"))

                async def stream():
                    yield ("data: " + json.dumps({"choices": [{"delta": {"content": f"w{i}"}}]}) + "\n\n").encode()
                    yield b"data: [DONE]\n\n"

                return StreamingResponse(stream(), media_type="text/event-stream")

            return app

        ports = [_free_port() for _ in range(3)]
        router = Router([f"http://127.0.0.1:{p}" for p in ports[:2]])
        apps = [worker(0), worker(1), build_router_app(router, poll_s=0)]
        servers = [
            uvicorn.Server(uvicorn.Config(a, host="127.0.0.1", port=p, log_level="warning"))
            for a, p in zip(apps, ports)
        ]
        tasks = [asyncio.create_task(s.serve()) for s in servers]
        try:
            while not all(s.started for s in servers):
                await asyncio.sleep(0.05)
            base = f"http://127.0.0.1:{ports[2]}"
            async with httpx.AsyncClient(timeout=30) as c:

                async def turn(sid: str) -> str:
                    body = {
                        "model": "glm",
                        "stream": True,
                        "session_id": sid,
                        "messages": [{"role": "user", "content": f"{sid} " + "context " * 2000}],
                    }
                    async with c.stream("POST", base + "/v1/chat/completions", json=body) as r:
                        assert r.status_code == 200
                        lines = [x async for x in r.aiter_lines() if x.startswith("data: ") and x != "data: [DONE]"]
                    return json.loads(lines[0][6:])["choices"][0]["delta"]["content"]

                sids = [f"agent-{i:02d}" for i in range(4)]
                first = {s: await turn(s) for s in sids}
                assert sorted(first.values()) == ["w0", "w0", "w1", "w1"]  # new sessions balanced
                for _ in range(3):  # continuing turns stay put
                    for s in sids:
                        assert await turn(s) == first[s]
                victim = next(s for s in sids if first[s] == "w0")
                servers[0].should_exit = True  # worker 0 dies
                await tasks[0]
                assert await turn(victim) == "w1"  # failed over, and completed
                assert await turn(victim) == "w1"  # and stays there
                h = (await c.get(base + "/health")).json()
                assert h["routed"]["failover"] == 1 and h["up"] == [False, True], h
                survivor = next(s for s in sids if first[s] == "w1")
                assert await turn(survivor) == "w1"
            assert set(got[0]) | set(got[1]) == set(sids)
        finally:
            for s in servers:
                s.should_exit = True
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(main())


def test_a_health_poll_that_times_out_once_moves_no_session_but_a_refused_port_does():
    """The router's own poller against stub workers: worker 0's /health times out once (as under a long step or a GC
    pause) and then answers: no session moves (routed failover 0). Then worker 0's port is closed: the poll's refused
    connection marks it down at once and its session fails over."""
    import httpx
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    from trnrt.serve.router import build_router_app

    async def main() -> None:
        slow = {"left": 0}  # armed after the sessions are pinned

        def worker(i: int) -> FastAPI:
            app = FastAPI()

            @app.get("/health")
            async def health():
                if i == 0 and slow["left"] > 0:
                    slow["left"] -= 1
                    await asyncio.sleep(1.0)  # past the router's health timeout
                return {"status": "ok"}

            @app.post("/v1/chat/completions")
            async def chat(req: Request):
                await req.json()

                async def stream():
                    yield ("data: " + json.dumps({"choices": [{"delta": {"content": f"w{i}"}}]}) + "\n\n").encode()
                    yield b"data: [DONE]\n\n"

                return StreamingResponse(stream(), media_type="text/event-stream")

            return app

        ports = [_free_port() for _ in range(3)]
        router = Router([f"http://127.0.0.1:{p}" for p in ports[:2]], down_after=3)
        apps = [worker(0), worker(1), build_router_app(router, poll_s=0.2, health_timeout_s=0.3)]
        servers = [
            uvicorn.Server(uvicorn.Config(a, host="127.0.0.1", port=p, log_level="warning"))
            for a, p in zip(apps, ports)
        ]
        tasks = [asyncio.create_task(s.serve()) for s in servers]
        try:
            while not all(s.started for s in servers):
                await asyncio.sleep(0.05)
            base = f"http://127.0.0.1:{ports[2]}"
            async with httpx.AsyncClient(timeout=30) as c:

                async def turn(sid: str) -> str:
                    body = {
                        "model": "glm",
                        "stream": True,
                        "session_id": sid,
                        "messages": [{"role": "user", "content": sid}],
                    }
                    async with c.stream("POST", base + "/v1/chat/completions", json=body) as r:
                        lines = [x async for x in r.aiter_lines() if x.startswith("data: ") and x != "data: [DONE]"]
                    return json.loads(lines[0][6:])["choices"][0]["delta"]["content"]

                for _ in range(100):  # the first polls may have run before the workers listened (refused: down)
                    if (await c.get(base + "/health")).json()["up"] == [True, True]:
                        break
                    await asyncio.sleep(0.05)
                pins = {s: await turn(s) for s in ("a", "b", "c", "d")}
                on0 = [s for s, w in pins.items() if w == "w0"]
                assert on0
                slow["left"] = 1
                loop = asyncio.get_running_loop()
                t0, seen = loop.time(), []
                while loop.time() - t0 < 2.0:  # turns all through the slow poll, its timeout and the next polls
                    seen.append(await turn(on0[0]))
                assert slow["left"] == 0 and set(seen) == {"w0"}, (slow, set(seen))
                h = (await c.get(base + "/health")).json()
                assert h["up"] == [True, True] and h["routed"]["failover"] == 0, h
                for _ in range(2):  # and every session still lands where it was pinned
                    for s in pins:
                        assert await turn(s) == pins[s]
                servers[0].should_exit = True  # control: worker 0's port now refuses
                await tasks[0]
                for _ in range(40):
                    if not (await c.get(base + "/health")).json()["up"][0]:
                        break
                    await asyncio.sleep(0.1)
                assert await turn(on0[0]) == "w1"
                assert (await c.get(base + "/health")).json()["routed"]["failover"] == 1
        finally:
            for s in servers:
                s.should_exit = True
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(main())


def test_a_long_shared_system_prompt_does_not_pin_new_sessions():
    """An agent client's system message (instructions, environment, file tree) is shared by every session it starts. New
    sessions (that system message + a distinct first user turn) spread over the workers under load: the shared head
    does not count toward stickiness. A continuing session's later turn, its own conversation long, still sticks."""
    from trnrt.serve.router import _request_head, _request_text

    path = "/v1/chat/completions"
    system = {
        "role": "system",
        "content": "You are a coding agent. Environment and file tree follow. " + "src/x.py " * 4500,
    }
    r = Router(["a", "b"], balance_abs=1, balance_rel=1.0, sticky_chars=32_768)
    picks = []
    for i in range(8):
        body = {"messages": [system, {"role": "user", "content": f"task {i}: fix the bug in module {i}"}]}
        text, head = _request_text(path, body), _request_head(path, body)
        assert head > 40_000 and text[:head] == _request_text(path, {"messages": [system]})[:head]
        w = r.pick(text, head=head)
        r.load[w] += 1  # in flight: the load the router sees
        picks.append(w)
    assert set(picks) == {0, 1} and r.routed["sticky"] == 0, (picks, r.routed)
    # the positive control: one session's conversation grows past sticky_chars beyond the system message
    convo = [system, {"role": "user", "content": "task 0: fix the bug in module 0"}]
    convo += [
        {"role": "assistant", "content": "reading files " * 3000},
        {"role": "tool", "content": "def f(): pass " * 1000},
    ]
    home = r.pick(_request_text(path, {"messages": convo}), head=_request_head(path, {"messages": convo}))
    r.load[home] += 20  # far past balance_abs
    nxt = convo + [{"role": "assistant", "content": "next step"}]
    assert r.pick(_request_text(path, {"messages": nxt}), head=_request_head(path, {"messages": nxt})) == home
    assert r.routed["sticky"] == 1


MAIN, TITLE = ("sys-main", "task"), ("sys-title", "task")  # (sys_hash, first_hash): a title request has its own system


def test_a_side_request_goes_to_the_least_loaded_worker_and_the_pin_stays():
    r = Router(["a", "b"])
    w = r.pick("", session="s", key=MAIN, tools=True)
    r.load[w] += 1  # the main turn in flight (prefilling)
    side = r.pick("", session="s", key=TITLE, tools=False)  # the client's title request, its own system prompt
    assert side != w and r.routed["side"] == 1 and r.sessions["s"] == w and r.keys["s"] == MAIN
    r.load[w] += 5
    for _ in range(3):  # main turns stay pinned whatever the load
        assert r.pick("", session="s", key=MAIN, tools=True) == w
    assert r.routed["new_session"] == 1 and r.routed["rekey"] == 0


def _pin_side_failover(r: Router) -> tuple[int, int]:
    """A pin, a side request, a continuing turn, then the pinned worker down and the session's next turn."""
    w = r.pick("", session="s", key=MAIN, tools=True)
    r.load[w] += 1
    side = r.pick("", session="s", key=TITLE, tools=False)
    r.pick("", session="s", key=MAIN, tools=True)
    r.note_failure(w, hard=True)
    r.pick("", session="s", key=MAIN, tools=True)
    return w, side


def test_with_the_debug_switch_each_pin_and_side_request_logs_its_worker(capsys, monkeypatch):
    """TRNRT_ROUTER_LOG_SESSIONS=1: a log reader attributes requests to engines by these lines, which name the session:
    a new pin and a side request each print the worker; a continuing turn prints nothing (the last pin line per
    session is its worker); a failover names the session."""
    monkeypatch.setenv("TRNRT_ROUTER_LOG_SESSIONS", "1")
    r = Router(["a", "b"])
    w, side = _pin_side_failover(r)
    out = [ln for ln in capsys.readouterr().out.splitlines() if "is down" not in ln]
    assert out == [
        f"trnrt.router: session 's' pinned to worker {w} ({r.workers[w]})",
        f"trnrt.router: side request of session 's' to worker {side} ({r.workers[side]})",
        f"trnrt.router: session 's' failed over from worker {w} ({r.workers[w]}, down) to worker {1 - w} "
        f"({r.workers[1 - w]})",
    ]


def test_by_default_no_line_names_a_session(capsys, monkeypatch):
    """A session id is the client's own identifier: by default nothing is printed per request, and a failover is
    logged without the id."""
    monkeypatch.delenv("TRNRT_ROUTER_LOG_SESSIONS", raising=False)
    r = Router(["a", "b"])
    w, _ = _pin_side_failover(r)
    out = [ln for ln in capsys.readouterr().out.splitlines() if "is down" not in ln]
    assert out == [
        f"trnrt.router: a session failed over from worker {w} ({r.workers[w]}, down) to worker {1 - w} "
        f"({r.workers[1 - w]})"
    ]
    assert Router(["a"], log_sessions=True).log_sessions and not Router(["a"]).log_sessions


def test_a_side_request_that_comes_first_does_not_hold_the_main_history():
    r = Router(["a", "b"])
    t = r.pick("", session="s", key=TITLE, tools=False)  # the title request, before the first main turn
    r.load[t] += 1  # still in flight
    m = r.pick("", session="s", key=MAIN, tools=True)  # the main agent's first turn: placed like a new session
    assert m != t and r.routed["rekey"] == 1 and r.sessions["s"] == m and r.keys["s"] == MAIN
    r.load[m] += 3
    assert r.pick("", session="s", key=MAIN, tools=True) == m  # pinned there from now on
    assert r.pick("", session="s", key=TITLE, tools=False) == t and r.sessions["s"] == m  # later titles: by load


def test_a_compaction_keeps_an_established_pin_and_rekeys():
    r = Router(["a", "b"])
    w = r.pick("", session="s", key=("s", "f1"), tools=True)
    r.load[w] += 5
    assert (
        r.pick("", session="s", key=("s", "f2"), tools=True) == w
        and r.routed["rekey"] == 1
        and r.keys["s"] == ("s", "f2")
    )
    assert r.pick("", session="s", key=("s", "f2"), tools=True) == w and r.routed["side"] == 0


def test_a_client_without_tools_keeps_its_pin():
    r = Router(["a", "b"])
    w = r.pick("", session="s", key=("s", "f"), tools=False)
    r.load[w] += 5
    assert all(r.pick("", session="s", key=("s", "f"), tools=False) == w for _ in range(5)) and r.routed["side"] == 0
    r._forget("s")
    assert "s" not in r.keys and "s" not in r.tentative


def test_the_session_key_is_stable_while_a_conversation_grows_and_differs_by_system_prompt():
    from trnrt.serve.router import _session_key

    main = {"role": "system", "content": "You are a coding agent. " * 200}
    task = {"role": "user", "content": "Fix the bug in parser.py. " + "x" * 10000}
    k = lambda b: _session_key("/v1/chat/completions", b)  # noqa: E731
    t1 = {"messages": [main, task]}
    t2 = {"messages": [main, task, {"role": "assistant", "content": "ok"}, {"role": "tool", "content": "r"}]}
    title = {"messages": [{"role": "system", "content": "You are a title generator."}, task]}
    assert k(t1) == k(t2) and k(title)[0] != k(t1)[0] and k(title)[1] == k(t1)[1]
    other = k({"messages": [main, {"role": "user", "content": "Another task"}]})
    assert other[0] == k(t1)[0] and other[1] != k(t1)[1]  # same agent, another history: a re-key, not a side request


def test_end_to_end_a_title_request_is_not_pinned_with_its_session():
    """Through the real router app: the proxy keys session requests and passes their tools; a title request of a
    pinned session goes to the other (idle) worker, and the session's main turns stay home."""
    import httpx
    import uvicorn
    from fastapi import FastAPI, Request

    from trnrt.serve.router import build_router_app

    async def main() -> None:
        def worker(i: int) -> FastAPI:
            app = FastAPI()

            @app.get("/health")
            async def health():
                return {"status": "ok"}

            @app.post("/v1/chat/completions")
            async def chat(req: Request):
                await req.json()
                return {"choices": [{"message": {"role": "assistant", "content": f"w{i}"}}]}

            return app

        ports = [_free_port() for _ in range(3)]
        router = Router([f"http://127.0.0.1:{p}" for p in ports[:2]])
        apps = [worker(0), worker(1), build_router_app(router, poll_s=0)]
        servers = [
            uvicorn.Server(uvicorn.Config(a, host="127.0.0.1", port=p, log_level="warning"))
            for a, p in zip(apps, ports)
        ]
        tasks = [asyncio.create_task(s.serve()) for s in servers]
        try:
            while not all(s.started for s in servers):
                await asyncio.sleep(0.05)
            base = f"http://127.0.0.1:{ports[2]}"
            sysmsg = {"role": "system", "content": "You are a coding agent."}
            task = {"role": "user", "content": "Fix the parser."}
            tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
            async with httpx.AsyncClient(timeout=30) as c:

                async def ask(messages, with_tools):
                    body = {"model": "glm", "session_id": "s1", "messages": messages}
                    if with_tools:
                        body["tools"] = tools
                    r = await c.post(base + "/v1/chat/completions", json=body)
                    return r.json()["choices"][0]["message"]["content"]

                home = await ask([sysmsg, task], True)
                title = await ask([{"role": "system", "content": "Generate a title."}, task], False)
                again = await ask([sysmsg, task, {"role": "assistant", "content": "ok"}], True)
                h = (await c.get(base + "/health")).json()
            assert title != home and again == home, (home, title, again)
            assert h["routed"]["side"] == 1 and h["routed"]["session"] == 1 and h["routed"]["new_session"] == 1, h
        finally:
            for s in servers:
                s.should_exit = True
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(main())


def test_a_compaction_without_tools_rekeys_and_keeps_the_pin():
    """A tool-less client: its new history under the same system block (a compaction, a front-truncated history) is a
    re-key, not a side request; another system block is a side request."""
    r = Router(["a", "b"])
    w = r.pick("", session="s", key=("s", "f1"), tools=False)
    r.load[w] += 5
    assert r.pick("", session="s", key=("s", "f2"), tools=False) == w and r.routed["rekey"] == 1
    assert r.keys["s"] == ("s", "f2") and r.routed["side"] == 0
    assert all(r.pick("", session="s", key=("s", "f2"), tools=False) == w for _ in range(3))
    assert r.pick("", session="s", key=("t", "f2"), tools=False) != w and r.routed["side"] == 1  # a title: by load
    assert r.sessions["s"] == w and r.keys["s"] == ("s", "f2")
