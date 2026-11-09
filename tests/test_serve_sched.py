"""The HTTP side of scheduling (serve/api.py, serve/router.py): a request's `priority`, and the 503 of a full queue.

The engine is tests/test_serve_api.py's scripted stand-in, extended with what LLMEngine.add_request does for these: the
priority check of its Scheduling settings, scheduler.QueueFull for a full queue, and a waiting request displaced by a
better one (finished, Request.error set)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.test_serve_api import PLAIN, ScriptedEngine
from trnrt.engine.request import Status
from trnrt.engine.scheduler import DISPLACED, QueueFull, Scheduling

MSG = {"messages": [{"role": "user", "content": "Hello"}]}


class QueueEngine(ScriptedEngine):
    """ScriptedEngine with LLMEngine.add_request's scheduling: `scheduling` checks a request's priority; with `limit`
    set, a request of priority below 5 waits (it never steps) and one that finds `limit` waiting is refused with
    QueueFull, unless its priority is 5 or more, when it displaces the oldest waiting request and runs."""

    def __init__(self, script, scheduling=None, limit=None):
        super().__init__(script)
        self.scheduling = scheduling or Scheduling()
        self.limit = limit
        self.parked: list = []
        self.priorities: list = []

    def add_request(self, prompt_ids, params, rid=None, session_id=None, priority=None):
        self.scheduling.check_priority(priority)
        self.priorities.append(priority)
        if self.limit is None:
            return super().add_request(prompt_ids, params, rid=rid, session_id=session_id)
        urgent = priority is not None and priority >= 5
        if len(self.parked) >= self.limit and not urgent:
            raise QueueFull("The request queue is full.")
        req = super().add_request(prompt_ids, params, rid=rid, session_id=session_id)
        if not urgent:
            self.scheduler.running.remove(req)
            req.status = Status.WAITING
            self.parked.append(req)
        elif len(self.parked) >= self.limit:
            gone = self.parked.pop(0)
            gone.status, gone.finish_reason, gone.error = Status.FINISHED, "abort", (503, DISPLACED)
        return req


@pytest.fixture
def app():
    from fastapi.testclient import TestClient

    from trnrt.serve import api

    started = []

    def start(eng):
        client = TestClient(api.build_app(eng, "glm"))
        client.__enter__()
        started.append((client, eng))
        return client

    yield start
    for client, eng in started:
        client.__exit__(None, None, None)
        for lp in [lp for lp in api._LOOPS if lp.engine is eng]:
            lp.stop()


def test_a_priority_the_engine_would_ignore_is_refused_with_400(app):
    eng = QueueEngine(PLAIN)
    client = app(eng)
    r = client.post("/v1/chat/completions", json={**MSG, "priority": 3})
    assert r.status_code == 400 and "--enable-priority-scheduling" in r.json()["detail"], r.text
    assert client.post("/v1/chat/completions", json={**MSG, "priority": None}).status_code == 200  # null is absent
    assert eng.priorities == [None]  # the refused one never reached the queue


@pytest.mark.parametrize("value", ["high", 1.5, True, [1], 2**63])
def test_a_priority_that_is_not_an_integer_is_refused_before_the_engine(app, value):
    eng = QueueEngine(PLAIN, Scheduling(priority_scheduling=True))
    client = app(eng)
    r = client.post("/v1/completions", json={"prompt": [1, 2], "priority": value})
    assert r.status_code == 400 and "priority must be an integer" in r.text, r.text
    assert eng.priorities == []


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
def test_the_priority_reaches_the_engine_on_both_endpoints(app, path):
    eng = QueueEngine(PLAIN, Scheduling(priority_scheduling=True))
    client = app(eng)
    body = {**MSG} if "chat" in path else {"prompt": [1, 2]}
    for p in (7, -3, 2.0):
        assert client.post(path, json={**body, "priority": p}).status_code == 200
    assert eng.priorities == [7, -3, 2]


@pytest.mark.parametrize("stream", [False, True])
def test_a_full_queue_answers_503_with_the_queue_full_header(app, stream):
    from trnrt.serve.api import QUEUE_FULL_HEADER

    eng = QueueEngine(PLAIN, limit=0)
    client = app(eng)
    r = client.post("/v1/chat/completions", json={**MSG, "stream": stream})
    assert r.status_code == 503 and r.json()["detail"] == "The request queue is full.", r.text
    assert r.headers[QUEUE_FULL_HEADER] == "1"
    h = client.get("/health")  # the engine is serving: only that request was refused
    assert h.status_code == 200
    eng.limit = None
    assert client.post("/v1/chat/completions", json=MSG).status_code == 200


def test_an_engine_failure_503_has_no_queue_full_header(app):
    from trnrt.serve.api import QUEUE_FULL_HEADER

    eng = QueueEngine(PLAIN)
    eng.cfg.tp = 2
    client = app(eng)
    eng.dead = [1]
    r = client.post("/v1/chat/completions", json=MSG)
    assert r.status_code == 503 and QUEUE_FULL_HEADER not in r.headers


@pytest.mark.parametrize("stream", [False, True])
def test_a_displaced_request_gets_a_503_and_the_better_one_runs(app, stream):
    import threading

    from trnrt.serve.api import QUEUE_FULL_HEADER

    eng = QueueEngine(PLAIN, Scheduling(priority_scheduling=True), limit=1)
    client = app(eng)
    out: dict = {}

    def waiting_request():
        if stream:
            with client.stream("POST", "/v1/chat/completions", json={**MSG, "stream": True, "priority": 0}) as r:
                out["status"] = r.status_code
                out["lines"] = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
        else:
            r = client.post("/v1/chat/completions", json={**MSG, "priority": 0})
            out["status"], out["body"], out["headers"] = r.status_code, r.json(), r.headers

    t = threading.Thread(target=waiting_request)
    t.start()
    for _ in range(500):  # until the engine holds it waiting
        if eng.parked:
            break
        t.join(0.01)
    assert eng.parked
    r = client.post("/v1/chat/completions", json={**MSG, "priority": 9})
    assert r.status_code == 200 and r.json()["choices"][0]["finish_reason"] == "stop", r.text
    t.join(10)
    assert not t.is_alive()
    if stream:  # accepted with a 200 before it was displaced: the stream ends with the error
        assert out["status"] == 200
        err = json.loads(out["lines"][-1][6:])["error"]
        assert err == {"message": DISPLACED, "code": 503}
    else:
        assert out["status"] == 503 and out["body"]["detail"] == DISPLACED and out["headers"][QUEUE_FULL_HEADER] == "1"


def test_the_router_passes_a_queue_full_503_on_and_keeps_the_engine_up():
    """The real engine app behind the real router: a full queue's 503 reaches the client with its header, and the router
    neither marks the engine down nor retries elsewhere; an engine failure's 503 still marks it down (the control)."""
    import asyncio

    import httpx

    from tests.test_router import _free_port, _start, _stop
    from trnrt.serve import api
    from trnrt.serve.router import QUEUE_FULL_HEADER, Router, build_router_app

    assert api.QUEUE_FULL_HEADER == QUEUE_FULL_HEADER

    async def main() -> None:
        ports = [_free_port() for _ in range(3)]
        full, ok = QueueEngine(PLAIN, limit=0), QueueEngine(PLAIN)
        full.cfg.tp = 2  # its rank watch: the control below kills a rank
        router = Router([f"http://127.0.0.1:{p}" for p in ports[:2]])
        servers, tasks = await _start(
            [
                (api.build_app(full, "glm"), ports[0]),
                (api.build_app(ok, "glm"), ports[1]),
                (build_router_app(router, poll_s=0), ports[2]),
            ]
        )
        base = f"http://127.0.0.1:{ports[2]}"
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                for stream in (False, True):
                    body = {**MSG, "session_id": "s1", "stream": stream}
                    router.sessions["s1"] = 0  # pinned to the engine whose queue is full
                    r = await c.post(base + "/v1/chat/completions", json=body)
                    assert r.status_code == 503 and r.headers.get(QUEUE_FULL_HEADER) == "1", r.text
                    assert r.json()["detail"] == "The request queue is full."
                    assert router.up == [True, True] and router.load == [0, 0] and ok.added == []
                full.dead = [1]  # the control: the same engine failing answers 503 without the header
                r = await c.post(base + "/v1/chat/completions", json={**MSG, "session_id": "s1"})
                assert r.status_code == 200 and router.up == [False, True] and len(ok.added) == 1
        finally:
            await _stop(servers, tasks)
            for lp in [lp for lp in api._LOOPS if lp.engine in (full, ok)]:
                lp.stop()

    asyncio.run(main())
