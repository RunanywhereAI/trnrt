"""GET /v1/stats's statistics (serve/stats.py), the production log (serve/prodlog.py), GET /dashboard
(serve/dashboard.py), the prompt token cache behind the chat endpoint (serve/prompt_cache.py), and the router's views
of all of them (serve/router.py).

The statistics run on fake clocks, so every rate and percentile is checked against a hand count. The HTTP tests use
tests/test_serve_api.py's scripted engine."""

from __future__ import annotations

import gzip
import json
import os
import time
import types

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.test_serve_api import PLAIN, ScriptedEngine
from trnrt.serve import prodlog as PL
from trnrt.serve import stats as S

T0 = 1_800_000_000.0  # a wall-clock second on a bucket boundary


class Clock:
    def __init__(self):
        self.mono = 1000.0

    def wall(self) -> float:
        return T0 + self.mono - 1000.0

    def mono_(self) -> float:
        return self.mono


def req(rid, arrival, first, done, prompt=100, cached=40, output=11, finish="stop"):
    return types.SimpleNamespace(
        rid=rid,
        arrival_time=arrival,
        first_token_time=first,
        finish_time=done,
        num_prompt=prompt,
        num_cached_tokens=cached,
        output_ids=[1] * output,
        finish_reason=finish,
        spec_proposed=0,
        spec_accepted=0,
    )


def stats():
    c = Clock()
    return S.Stats(clock=c.wall, mono=c.mono_), c


def test_percentiles_and_derived_times():
    assert S.pct([], 0.5) is None and S.pct([3.0], 0.9) == 3.0
    assert S.pct([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5 and S.pct([0.0, 10.0], 0.9) == 9.0  # numpy's linear
    rec = S.derive({"arrival": 10.0, "first": 10.5, "done": 12.5, "output": 11})
    assert (rec["ttft"], rec["e2e"], rec["decode_tps"], rec["itl"]) == (0.5, 2.5, 5.0, 0.2)
    one = S.derive({"arrival": 1.0, "first": 2.0, "done": 2.0, "output": 1})
    assert one["decode_tps"] is None and one["itl"] is None


def test_windows_count_requests_by_their_finish_and_tokens_as_handed_out():
    st, c = stats()
    # three finished requests: two ok in the last minute, one ok 4 minutes ago; one aborted, one failed
    c.mono = 1000.0 + 600  # 10 minutes after start
    for rid, arr, first, done, out in (("a", 1590, 1590.5, 1592.5, 11), ("b", 1580, 1581, 1585, 21)):
        st.record(req(rid, arr, first, done, output=out), stream=True, at_arrival=(1, 0))
    st.record(req("old", 1350, 1351, 1360, output=5), stream=False, at_arrival=None)
    st.record(req("gone", 1595, None, 1596, output=0, finish="abort"), stream=True, at_arrival=(2, 1))
    st.failed("queue_full")
    for k in range(6):  # 6 ticks of 10 tokens in the last minute
        st.tick(1550.0 + 10 * k, 10, 2, 1)
    snap = st.snapshot({"x": 1}, now=c.wall(), ttl=0)
    w1, w5 = snap["windows"]["1m"], snap["windows"]["5m"]
    assert (w1["n"], w1["ok"], w1["rpm"]) == (3, 2, 2.0)
    assert (w5["ok"], w5["rpm"]) == (3, 3 / 5)
    assert w1["tpm"]["input"] == 300 and w1["tpm"]["output"] == 32 and w1["cache_hit_pct"] == 40.0
    assert w1["ttft"]["n"] == 2 and w1["ttft"]["p50"] == pytest.approx(0.75)  # 0.5 and 1.0
    assert w1["decode_tps"]["mean"] == pytest.approx((5.0 + 5.0) / 2)  # 10 / 2 s and 20 / 4 s
    assert w1["out_tok_s"] == pytest.approx(60 / 60)
    assert snap["totals"]["requests"] == {"ok": 3, "aborted": 1, "failed": 1}
    assert snap["totals"]["failed_kinds"] == {"queue_full": 1} and snap["gauges"] == {"x": 1}
    assert [r["rid"] for r in snap["recent"]] == ["gone", "old", "b", "a"]  # newest finished first
    assert snap["recent"][0]["running_at_arrival"] == 2 and snap["recent"][0]["stream"] is True
    series = snap["series"]
    assert len(series) == S.SERIES_BUCKETS and series[-1]["t"] == (c.wall() // S.BUCKET_S) * S.BUCKET_S
    assert sum(p["rpm"] or 0 for p in series) * S.BUCKET_S / 60 == 3  # each ok request in its finish bucket


def test_the_start_and_today_windows_count_past_the_ring():
    """ "start" and "today" count every request, as their ok and token totals do, not only the ring's newest."""
    c = Clock()
    st = S.Stats(ring=3, clock=c.wall, mono=c.mono_)
    earlier = [
        S.derive(
            {
                "rid": f"e{i}",
                "arrival": T0 - 50,
                "first": None,
                "done": T0 - 40,
                "prompt": 1,
                "cached": 0,
                "output": 0,
                "finish": "abort",
            }
        )
        for i in range(4)
    ]
    st.merge_day(earlier, [], T0 - 3600)
    c.mono = 1060.0
    for i in range(5):
        st.record(req(f"r{i}", 1001, 1002, 1003 + i), stream=False, at_arrival=None)
    w = st.snapshot({}, now=c.wall(), ttl=0)["windows"]
    assert len(st.records) == 3 and w["1m"]["n"] == 3  # a window inside the ring counts what it holds
    assert (w["start"]["n"], w["start"]["ok"]) == (5, 5)
    assert (w["today"]["n"], w["today"]["ok"]) == (9, 5)


def test_the_snapshot_is_shared_for_its_ttl_but_the_gauges_are_live():
    st, _c = stats()
    a = st.snapshot({"g": 1}, now=T0 + 5, ttl=1.0)
    st.record(req("a", 1001, 1002, 1003), stream=False, at_arrival=None)
    b = st.snapshot({"g": 2}, now=T0 + 5.5, ttl=1.0)
    assert b["recent"] == a["recent"] == [] and b["gauges"] == {"g": 2}
    assert len(st.snapshot({}, now=T0 + 6.5, ttl=1.0)["recent"]) == 1


def test_a_closed_bucket_becomes_a_gauges_record():
    st, _ = stats()
    out = []
    st.on_bucket = out.append
    st.kv = lambda: (25, 100)
    st.record(req("a", 1001, 1002, 1005, prompt=60, output=6), stream=False, at_arrival=None)
    st.tick(1001.0, 30, 3, 1)
    st.tick(1004.0, 10, 1, 3)
    st.tick(1012.0, 0, 0, 0)  # the next bucket: the first one closes
    (g,) = out
    assert g["t"] == T0 and (g["running"], g["running_max"], g["waiting"], g["waiting_max"]) == (2, 3, 2, 3)
    assert (g["out_tok_s"], g["rpm"], g["input_tpm"], g["output_tpm"], g["kv_used_pct"]) == (
        4.0,
        6.0,
        360.0,
        36.0,
        25.0,
    )


def test_the_production_log_writes_reloads_and_rotates(tmp_path):
    st, c = stats()
    log = PL.ProdLog(str(tmp_path / "engine-8001"), stats=st, clock=c.wall, rotate_bytes=400)
    for i in range(6):
        log.put("requests", st.record(req(f"r{i}", 1001, 1002, 1003 + i), stream=False, at_arrival=(0, 0)))
    log.put("gauges", {"t": T0, "running": 1.0, "out_tok_s": 3.0})
    log.close()
    day = tmp_path / "engine-8001" / PL.utc_day(T0)
    reqs = [r for p in PL.day_files(str(day), "requests") for r in PL.read_jsonl(p)]
    assert [r["rid"] for r in reqs] == [f"r{i}" for i in range(6)]
    assert any(n.endswith(".gz") for n in os.listdir(day)), "a rotated segment is gzipped"
    assert set(reqs[0]) >= {"prompt", "cached", "output", "ttft", "finish"} and "prompt_ids" not in reqs[0]
    assert log.written == {"requests": 6, "gauges": 1} and log.write_errors == 0
    # a new process: today's records come back into the "today" window
    st2, c2 = stats()
    c2.mono = 1100.0
    log2 = PL.ProdLog(str(tmp_path / "engine-8001"), stats=st2, clock=c2.wall)
    log2.close()
    today = st2.snapshot({}, now=c2.wall(), ttl=0)["windows"]["today"]
    assert today["ok"] == 6 and today["tpm"]["input"] is not None
    assert log2.status()["on"] and "requests.jsonl" in "".join(log2.status()["files"])


def test_a_torn_line_and_a_full_queue_lose_only_themselves(tmp_path):
    p = tmp_path / "x.jsonl.gz"
    with gzip.open(p, "wt") as f:
        f.write('{"a": 1}\n{"b": 2')
    assert PL.read_jsonl(str(p)) == [{"a": 1}]
    log = PL.ProdLog.__new__(PL.ProdLog)
    import queue

    log.q, log.dropped = queue.Queue(1), dict.fromkeys(PL.NAMES, 0)
    assert log.put("requests", {}) and not log.put("requests", {}) and log.dropped["requests"] == 1


def test_the_production_log_is_off_unless_set(monkeypatch, tmp_path):
    monkeypatch.delenv(PL.ENV, raising=False)
    assert PL.ProdLog.from_env("engine-8001") is None
    monkeypatch.setenv(PL.ENV, str(tmp_path))
    log = PL.ProdLog.from_env("engine-8001")
    assert log is not None and log.root == str(tmp_path / "engine-8001")
    log.close()


# -- over HTTP ---------------------------------------------------------------------------------------------------------


@pytest.fixture
def app():
    from fastapi.testclient import TestClient

    from trnrt.serve import api

    started = []

    def start(eng, **kw):
        client = TestClient(api.build_app(eng, "glm", **kw))
        client.__enter__()
        started.append((client, eng))
        return client

    yield start
    for client, eng in started:
        client.__exit__(None, None, None)
        for lp in [lp for lp in api._LOOPS if lp.engine is eng]:
            lp.stop()


def _wait(cond, t=10.0):
    end = time.monotonic() + t
    while not cond():
        assert time.monotonic() < end
        time.sleep(0.02)


def test_stats_over_http_keep_the_routers_gauges_and_count_what_ran(app):
    eng = ScriptedEngine(PLAIN)
    client = app(eng)
    before = client.get("/v1/stats").json()
    g = before["gauges"]
    assert (g["kv_pages_used"], g["kv_pages_total"]) == (6, 100)  # serve/router.py reads these two
    assert g["running"] == 0 and g["waiting"] == 0 and g["model"] == "glm" and g["prompt_cache"] == {"on": False}
    assert g["prod"] == {"on": False} and g["stats_errors"] == 0
    assert client.post("/v1/completions", json={"prompt": [1, 2], "max_tokens": 3}).status_code == 200
    events = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "stream": True})
    assert events.status_code == 200
    assert client.post("/v1/completions", json={"prompt": [1] * 70}).status_code == 400  # the engine refuses it
    _wait(lambda: client.get("/v1/stats").json()["totals"]["requests"]["ok"] == 2)
    snap = client.get("/v1/stats").json()
    assert snap["totals"]["failed_kinds"] == {"refused": 1}
    recent = snap["recent"]
    assert len(recent) == 2 and {r["stream"] for r in recent} == {True, False}
    plain = next(r for r in recent if not r["stream"])
    assert (plain["prompt"], plain["output"], plain["cached"], plain["finish"]) == (2, 3, 2, "length")
    assert snap["windows"]["start"]["ok"] == 2 and snap["windows"]["1m"]["tpm"]["output"] is not None


def test_the_dashboard_is_one_self_contained_page(app):
    client = app(ScriptedEngine(PLAIN))
    r = client.get("/dashboard")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    page = r.text
    assert "<title>TrnRT serving</title>" in page and "v1/stats" in page
    assert "http://" not in page and "https://" not in page and "<script src" not in page and "<link" not in page


def test_the_production_log_through_the_server(app, monkeypatch, tmp_path):
    monkeypatch.setenv(PL.ENV, str(tmp_path))
    from trnrt.serve import api

    eng = ScriptedEngine(PLAIN)
    client = app(eng, name="engine-8001")
    assert client.post("/v1/completions", json={"prompt": [1, 2], "max_tokens": 2}).status_code == 200
    assert client.get("/v1/stats").json()["gauges"]["prod"]["on"]
    (lp,) = [lp for lp in api._LOOPS if lp.engine is eng]
    _wait(lambda: lp.prodlog.written["requests"] == 1)
    lp.stop()
    day = tmp_path / "engine-8001" / PL.utc_day(time.time())
    (rec,) = PL.read_jsonl(str(day / "requests.jsonl"))
    assert rec["prompt"] == 2 and rec["output"] == 2 and rec["finish"] == "length"


def test_the_chat_endpoint_serves_the_same_ids_through_the_prompt_cache(app, monkeypatch):
    """TRNRT_PROMPT_CACHE_MB on: a session's turns reach the engine with exactly the ids the server sends without it,
    and the cache hit (GET /v1/stats shows it)."""
    from tests.test_prompt_cache import TAGS, _tokenizer

    def run(cache_mb):
        if cache_mb:
            monkeypatch.setenv("TRNRT_PROMPT_CACHE_MB", cache_mb)
        else:
            monkeypatch.delenv("TRNRT_PROMPT_CACHE_MB", raising=False)
        eng = ScriptedEngine(PLAIN)
        eng.tokenizer = _tokenizer()
        eng.mcfg.vocab_size = len(eng.tokenizer)
        eng.cfg.max_model_len = 100_000
        client = app(eng)
        msgs = [{"role": "system", "content": "You are an agent. " * 30}]
        for k in range(6):
            msgs = [*msgs, {"role": "user", "content": f"turn {k}: do the next thing {TAGS[k]} please"}]
            body = {"messages": msgs, "session_id": "s1", "max_tokens": 2}
            r = client.post("/v1/chat/completions", json=body)
            assert r.status_code == 200, r.text
            msgs = [*msgs, {"role": "assistant", "content": "ok " * k, "reasoning_content": "thinking"}]
        return [list(r.prompt_ids) for r in eng.added], client.get("/v1/stats").json()["gauges"]["prompt_cache"]

    plain, off = run(None)
    cached, on = run("8")
    assert cached == plain and off == {"on": False}
    assert on["on"] and on["hits"] >= 4 and on["audits"] == on["hits"], on  # the first hits are all audited


# -- the router --------------------------------------------------------------------------------------------------------


def test_the_router_serves_each_engines_stats_and_the_dashboard():
    import asyncio

    import httpx

    from tests.test_router import _free_port, _start, _stop
    from trnrt.serve import api
    from trnrt.serve.router import Router, build_router_app

    async def main() -> None:
        ports = [_free_port() for _ in range(3)]
        engs = [ScriptedEngine(PLAIN), ScriptedEngine(PLAIN)]
        router = Router([f"http://127.0.0.1:{p}" for p in ports[:2]])
        servers, tasks = await _start(
            [
                (api.build_app(engs[0], "glm"), ports[0]),
                (api.build_app(engs[1], "glm"), ports[1]),
                (build_router_app(router, poll_s=0), ports[2]),
            ]
        )
        base = f"http://127.0.0.1:{ports[2]}"
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(f"http://127.0.0.1:{ports[1]}/v1/completions", json={"prompt": [1], "max_tokens": 1})
                assert r.status_code == 200
                for _ in range(200):  # until engine 1 has recorded it
                    one = (await c.get(base + "/v1/stats?engine=1")).json()
                    if one["totals"]["requests"]["ok"] == 1:
                        break
                    await asyncio.sleep(0.02)
                assert one["router"] == {"engine": 1, "engines": [{"url": u, "up": True} for u in router.workers]}
                assert one["gauges"]["kv_pages_total"] == 100
                both = (await c.get(base + "/v1/stats")).json()
                assert [e["stats"]["totals"]["requests"]["ok"] for e in both["engines"]] == [0, 1]
                assert both["router"]["workers"] == router.workers
                assert (await c.get(base + "/v1/stats?engine=2")).status_code == 400
                assert (await c.get(base + "/v1/stats?engine=x")).status_code == 400
                d = await c.get(base + "/dashboard", follow_redirects=False)
                assert d.status_code == 307 and d.headers["location"] == "dashboard?engine=0"
                d = await c.get(base + "/dashboard?engine=1")
                assert d.status_code == 200 and "<title>TrnRT serving</title>" in d.text
                assert router.load == [0, 0] and router.sessions == {}  # read only: no routing state changed
                await _stop(servers, tasks, 0)  # engine 0 goes away: its entry says so, the other still answers
                both = (await c.get(base + "/v1/stats")).json()
                assert both["engines"][0]["stats"] is None and "did not answer" in both["engines"][0]["error"]
                assert both["engines"][1]["stats"] is not None
                assert (await c.get(base + "/v1/stats?engine=0")).status_code == 503
        finally:
            await _stop(servers, tasks)
            for lp in [lp for lp in api._LOOPS if lp.engine in engs]:
                lp.stop()

    asyncio.run(main())


def test_json_of_a_snapshot_round_trips():
    st, _ = stats()
    st.record(req("a", 1001, 1002, 1003), stream=False, at_arrival=None)
    snap = st.snapshot({}, now=T0 + 10, ttl=0)
    assert json.loads(json.dumps(snap)) == snap
