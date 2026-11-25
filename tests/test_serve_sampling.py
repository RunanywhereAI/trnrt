"""The sampling extras of /v1/chat/completions and /v1/completions (trnrt/serve/api.py): `n` choices,
stop_regex, keeping the stop text (include_stop_str_in_output / no_stop_trim), skip_special_tokens, the stop token left
out of the text, and the refusal of a minimum length. Over the scripted engine of tests/test_serve_api.py."""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.test_serve_api import ID, PLAIN, TOOL_TURN, TOOLS, ScriptedEngine, _events

MSG = {"messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def app():
    from fastapi.testclient import TestClient

    from trnrt.serve import api

    started = []

    def start(script, setup=None):
        eng = ScriptedEngine(script)
        if setup is not None:
            setup(eng)
        client = TestClient(api.build_app(eng, "glm"))
        client.__enter__()
        started.append((client, eng))
        return client, eng

    yield start
    for client, eng in started:
        client.__exit__(None, None, None)
        for lp in [lp for lp in api._LOOPS if lp.engine is eng]:
            lp.stop()


def test_n_choices_share_the_prompt_and_count_it_once(app):
    """n engine requests for one HTTP request, one prompt; each choice its index; usage counts the prompt once and
    every choice's output. A seed gives choice i seed + i, so seeded choices differ and reproduce."""
    client, eng = app(TOOL_TURN)
    r = client.post("/v1/chat/completions", json={**MSG, "tools": TOOLS, "n": 3, "seed": 7})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [c["index"] for c in body["choices"]] == [0, 1, 2]
    for c in body["choices"]:
        assert c["finish_reason"] == "tool_calls" and c["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    reqs = eng.added[-3:]
    assert len({tuple(q.prompt_ids) for q in reqs}) == 1 and len({q.rid for q in reqs}) == 3
    assert [q.params.seed for q in reqs] == [7, 8, 9]
    n_prompt = len(reqs[0].prompt_ids)
    assert body["usage"] == {
        "prompt_tokens": n_prompt,
        "completion_tokens": 3 * len(TOOL_TURN),
        "total_tokens": n_prompt + 3 * len(TOOL_TURN),
        "prompt_tokens_details": {"cached_tokens": 2},
    }
    assert client.post("/v1/chat/completions", json={**MSG, "n": 2}).status_code == 200
    assert [q.params.seed for q in eng.added[-2:]] == [None, None]  # unseeded: each its own noise (its request id)


def test_n_choices_stream_per_choice_chunks(app):
    """Streamed, each choice opens with its role chunk, its deltas carry its index, each ends with its finish reason,
    and the usage chunk sums the choices."""
    client, _ = app(PLAIN)
    events = _events(
        client, {**MSG, "n": 2, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 3}
    )
    usage = events.pop()
    assert usage["usage"]["completion_tokens"] == 6 and usage["choices"] == []
    by = {0: [], 1: []}
    for e in events:
        (c,) = e["choices"]
        by[c["index"]].append(c)
    for i in (0, 1):
        assert by[i][0]["delta"] == {"role": "assistant", "content": ""}
        assert "".join(c["delta"].get("reasoning_content", "") for c in by[i]) == "Hello there again"
        assert [c["finish_reason"] for c in by[i]][-1] == "length"
        assert sum(c["finish_reason"] is not None for c in by[i]) == 1


def test_n_completions_with_logprobs(app):
    client, _ = app(PLAIN)
    r = client.post("/v1/completions", json={"prompt": [1], "n": 2, "max_tokens": 2, "logprobs": 1})
    choices = r.json()["choices"]
    assert [c["index"] for c in choices] == [0, 1]
    for c in choices:
        assert c["text"] == "Hello there" and c["logprobs"]["tokens"] == ["Hello", " there"]
        assert c["finish_reason"] == "length" and c["matched_stop"] is None


@pytest.mark.parametrize(
    "extra",
    [
        {"n": 0},
        {"n": 129},
        {"n": "2"},
        {"n": 2, "best_of": 3},
        {"best_of": 2},
        {"n": 2, "seed": 2**63 - 1},
        {"min_tokens": 4},
        {"min_new_tokens": 4},
        {"spaces_between_special_tokens": False},
        {"stop_regex": ""},
        {"stop_regex": [3]},
        {"stop_regex": "("},
        {"stop_regex": ["a"] * 17},
        {"no_stop_trim": "yes"},
        {"include_stop_str_in_output": True, "no_stop_trim": "yes"},  # checked even when the other keeps the stop
        {"skip_special_tokens": "no"},
        {"service_tier": "flex"},
        {"service_tier": "priority"},
    ],
    ids=repr,
)
def test_what_cannot_be_served_is_refused_before_the_engine(app, extra):
    client, eng = app(PLAIN)
    for path, body in (("/v1/chat/completions", MSG), ("/v1/completions", {"prompt": [1]})):
        r = client.post(path, json={**body, **extra})
        assert r.status_code == 400, (path, r.text)
    assert eng.added == []
    ok = {
        "n": 2,
        "best_of": 2,
        "seed": 2**63 - 2,
        "min_tokens": 0,
        "spaces_between_special_tokens": True,
        "service_tier": "auto",
    }
    assert client.post("/v1/completions", json={"prompt": [1], **ok}).status_code == 200


def test_a_choice_the_engine_refuses_aborts_the_others():
    """The second choice is refused after the first was accepted: the client gets the refusal, and the first is aborted
    rather than left running for nobody."""
    from fastapi.testclient import TestClient

    from trnrt.serve import api

    eng = ScriptedEngine(PLAIN * 40)
    eng.step_delay = 0.01
    real, calls = eng.add_request, []

    def second_refused(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("no room")
        return real(*a, **k)

    eng.add_request = second_refused
    with TestClient(api.build_app(eng, "glm")) as client:
        r = client.post("/v1/completions", json={"prompt": [1], "n": 2, "max_tokens": 100})
        assert r.status_code == 400 and "no room" in r.text
        deadline = time.monotonic() + 5
        while eng.scheduler.running and time.monotonic() < deadline:
            time.sleep(0.01)
        assert eng.scheduler.running == [] and eng.aborted == [eng.added[0].rid]
    for lp in [lp for lp in api._LOOPS if lp.engine is eng]:
        lp.stop()


def test_a_client_that_disconnects_aborts_every_choice():
    """Real uvicorn (the TestClient never disconnects): every one of the n engine requests is aborted."""
    import socket
    import threading

    import uvicorn

    from trnrt.serve import api

    eng = ScriptedEngine(PLAIN * 40)
    eng.step_delay = 0.05
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(api.build_app(eng, "glm"), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    try:
        deadline = time.monotonic() + 30
        while not srv.started and time.monotonic() < deadline:
            time.sleep(0.02)
        body = json.dumps({"prompt": [1], "max_tokens": 160, "stream": True, "n": 3}).encode()
        c = socket.create_connection(("127.0.0.1", port))
        c.sendall(
            b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n%s" % (len(body), body)
        )
        deadline = time.monotonic() + 10
        while len(eng.scheduler.running) < 3 and time.monotonic() < deadline:
            time.sleep(0.02)
        rids = sorted(q.rid for q in eng.scheduler.running)
        assert len(rids) == 3
        c.close()
        deadline = time.monotonic() + 3
        while len(eng.aborted) < 3 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert sorted(eng.aborted) == rids
    finally:
        srv.should_exit = True
        th.join(timeout=30)
        for lp in [lp for lp in api._LOOPS if lp.engine is eng]:
            lp.stop()


@pytest.mark.parametrize("stream", [False, True])
def test_stop_regex_ends_the_output_after_its_match_and_aborts_the_request(app, stream):
    """stop_regex: the output ends at the end of the first match, the matched text kept, finish "stop", matched_stop the
    pattern. The engine knows nothing of it, so the request is aborted there. The match spans two tokens (" there",
    " again"), so it is only found by searching back into text already pushed: a bounded pattern's search starts
    its longest match before the new text. The engine takes 10 ms a token, so the abort reaches it before it ends the
    request on its own."""
    client, eng = app(PLAIN * 10, setup=lambda e: setattr(e, "step_delay", 0.01))
    body = {"prompt": [1], "max_tokens": 40, "stop_regex": [r"x{3}", r"re\s?ag"]}
    if stream:
        events = _events(client, {**body, "stream": True})
        text = "".join(e["choices"][0]["text"] for e in events)
        last = events[-1]["choices"][0]
    else:
        last = client.post("/v1/completions", json=body).json()["choices"][0]
        text = last["text"]
    assert text == "Hello there ag"
    assert last["finish_reason"] == "stop" and last["matched_stop"] == r"re\s?ag"
    deadline = time.monotonic() + 3
    while not eng.aborted and time.monotonic() < deadline:
        time.sleep(0.01)
    assert eng.aborted == [eng.added[-1].rid]


def test_a_stop_regex_match_is_recorded_as_a_stop(app, monkeypatch, tmp_path):
    """The server ends a stop_regex match by aborting the engine request, but the request finished as a stop: /v1/stats
    counts it ok (so it is in the RPM and the latency percentiles), the production log records "stop", and /metrics
    counts its finish reason and its latencies. An abort the client asks for is still an abort."""
    from trnrt.serve import api
    from trnrt.serve import prodlog as PL

    monkeypatch.setenv(PL.ENV, str(tmp_path))
    client, eng = app(PLAIN * 10, setup=lambda e: setattr(e, "step_delay", 0.01))
    r = client.post("/v1/completions", json={"prompt": [1], "max_tokens": 40, "stop_regex": r"re\s?ag"})
    assert r.json()["choices"][0]["finish_reason"] == "stop"
    (lp,) = [lp for lp in api._LOOPS if lp.engine is eng]
    deadline = time.monotonic() + 5
    while not (eng.aborted and lp.prodlog.written["requests"]) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert eng.aborted == [eng.added[-1].rid]  # the engine request was aborted, as before
    assert lp.stats.counts == {"ok": 1, "aborted": 0, "failed": 0}
    (rec,) = lp.stats.records
    assert rec["finish"] == "stop" and rec["e2e"] is not None and rec["output"] >= 3
    w = client.get("/v1/stats").json()["windows"]["1m"]
    assert w["ok"] == 1 and w["e2e"]["n"] == 1 and w["ttft"]["n"] == 1
    m = client.get("/metrics").text
    assert 'trnrt:request_success_total{finished_reason="stop"} 1' in m and 'finished_reason="abort"' not in m
    assert "trnrt:e2e_request_latency_seconds_count 1" in m and "trnrt:time_per_output_token_seconds_count 1" in m
    lp.stop()
    (path,) = tmp_path.rglob("requests.jsonl")
    assert [x["finish"] for x in PL.read_jsonl(str(path))] == ["stop"]


def test_stop_regex_unbounded_and_anchored(app):
    """A pattern with an unbounded repeat is searched back over the text (STOP_REGEX_SPAN characters before the new
    text); an anchor keeps its meaning (the search starts at a position, never at a slice)."""
    client, _ = app(PLAIN * 4)
    r = client.post("/v1/completions", json={"prompt": [1], "max_tokens": 16, "stop_regex": r"!.*again"})
    assert r.json()["choices"][0]["text"] == "Hello there again!Hello there again"
    r = client.post("/v1/completions", json={"prompt": [1], "max_tokens": 16, "stop_regex": r"^there"})
    assert r.json()["choices"][0]["text"] == "Hello there again!" * 4  # "there" never starts the text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("stops", "text", "matched"),
    [
        ({"stop": ["Hello", "again"]}, "Hello there ", "again"),
        ({"stop_regex": [r"Hel+o", r"ag\w+"]}, "Hello there again", r"ag\w+"),
    ],
    ids=["stop", "stop_regex"],
)
def test_a_stop_among_the_min_tokens_does_not_end_the_output(app, monkeypatch, stream, stops, text, matched):
    """min_tokens 2: "Hello" ends in the first output token, so it stays in the text and the output runs on to the next
    stop, as the engine checks stop strings (engine._stop_in). The scripted engine stands in for one started
    with --structured-outputs, which serves min_tokens."""
    import tests.test_serve_api as scripted

    monkeypatch.setattr(scripted, "unsupported", lambda params: None)
    client, _ = app(PLAIN * 2)
    body = {"prompt": [1], "max_tokens": 8, "min_tokens": 2, **stops}
    if stream:
        events = _events(client, {**body, "stream": True})
        got = "".join(e["choices"][0]["text"] for e in events)
        last = events[-1]["choices"][0]
    else:
        last = client.post("/v1/completions", json=body).json()["choices"][0]
        got = last["text"]
    assert got == text
    assert last["finish_reason"] == "stop" and last["matched_stop"] == matched


class _Bytes:
    """A tokenizer stand-in: token i decodes to vocab[i]."""

    def __init__(self, vocab):
        self.vocab = vocab

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.vocab[i] for i in ids)


def test_zero_width_stop_regexes_match_and_terminate():
    """A zero-width pattern matches where it should, with or without min_tokens, and a search past the protected text
    ends: a search position past the end is clamped to the end, where `$` would match again forever."""
    import re

    from trnrt.serve.api import _Detok

    tok = _Bytes(["abc", ""])
    end, start = ((re.compile("$"), 0),), ((re.compile("^"), 0),)
    d = _Detok(tok, regexes=end)  # an empty output: `$` matches at its start
    assert d.push([], final=True) == "" and d.matched == "$"
    d = _Detok(tok, regexes=start)
    assert d.push([0], final=False) == "" and d.matched == "^"
    d = _Detok(tok, regexes=end, protect=1)  # the text ends inside the protected token: no stop, and the search ends
    assert d.push([0, 1], final=True) == "abc" and d.matched is None


@pytest.mark.parametrize(
    "pattern",
    [
        r"^(a+)+b$",  # exponential: a repeat over a part that matches more than one way
        r"(a|a)*b",
        r"(a{1,3}){1,10}b",
        r"(?:\w+\s?)+$",
        r"a*a*b",  # a power of the length: two repeats in a row that take the same characters
        r"\d+\s*\d+",
        r".*x.*y",
        r"(x)?a*(?(1)b)a*c",  # a*a*c when group 1 is unset: a conditional with no "no" branch can match the empty text
        r"(?:)+",
        "(?:.|.)" * 5 + "x",  # alternations that can start alike, in a row: 32 ways to try
        "(?:a|ab)" * 3 + "(?:a|ab)(?:a|ab)",
        "".join(f"(?:{c}?|x)" for c in "abcde") + "y",  # branches that can be empty: as many ways as overlapping ones
        "(?:(?:a{0}|b{0})x)+y",
        "(?:(?=x)?x)+y",  # an optional empty part, repeated
        "(?:(?=x))?" * 5 + "y",
        "(?:){1000000}X",  # an exact count of an empty part: every iteration runs
        "".join(f"(?:{c}?)?" for c in "bcdefghijklmnopq") + "X",  # optional parts that can match empty: 65,536 ways
        # a lookaround that can match more than one way or length: the matcher runs it again at every position a repeat
        # around it tries, cubic or worse (each about a second, or minutes, over 2048 characters of prose)
        r"(?:[^\n](?=(?:[^\n](?=[^\n]*\n))+))+Y",
        r"[^~]+(?=[^~]*~~)~",
        r"(?:[^~](?![^~]*~~))+~~",
        r"(?:.(?!.*@@))+@@",
        r"[^\n]+(?=[^\n]*\n)Y",
        r"\.(?=\s|$)",
        # a possessive repeat or an atomic group after a repeat that can take its characters: cubic
        r".+.*+@",
        r".+(?>.*)@",
        # quadratic, but walked again at every character a repeat gives back: past the cost bound
        ".+" + "." * 1000 + "@",
        r"(?:\w(?!\w{31}@))+@",
        r"\w+(?!\w{15}@)@",
        r"\w*(?:aa|a)(?:aa|a)(?:aa|a)(?:aa|a)@",
        r"(?:a|a)(?:a|a)(?:a|a)(?:a|a)(\w)+@",  # a capturing group, marked at every iteration: 3 times \w+@'s steps
    ],
)
def test_stop_regex_refuses_patterns_that_backtrack(app, pattern):
    """CPython's re holds the GIL through a search and cannot be interrupted, so a pattern whose search can take
    exponential time, a power of the text's length, or many times a single repeat's quadratic time, is refused with the
    reason before it runs on the event loop."""
    client, eng = app(PLAIN)
    r = client.post("/v1/completions", json={"prompt": [1], "stop_regex": ["again", pattern]})
    assert r.status_code == 400 and r.json()["detail"].startswith(f"stop_regex {pattern!r} is refused: it "), r.text
    assert eng.added == []


def test_stop_regex_admits_ordinary_patterns():
    """Ordinary patterns, a lookaround that matches one way at one length, and every pattern this suite's requests send
    and expect served, all pass the check."""
    from trnrt.serve.api import _backtracking

    ordinary = [
        r"\n\n+",
        r"</answer>",
        r"Answer:\s*\d+",
        r"\d+\.\d+",
        r"\$\d+(?:\.\d\d)?",
        r"\bEND\b",
        r"(?:Human|User):",
        r"(?:foo|bar)+",
        r"[.!?]\s*$",
        r"\w+@\w+\.\w+",
        r"(?s)<think>.*</think>",
        r"(?i)stop here",
        r"[a-f0-9]{40}",
        r"(\w+)\s\1",
        r"re\s?ag",
        r"!.*again",
        r"(?:Answer|Ans):\s*(?:yes|y)\b",  # two alternations that start alike: 4 ways
        r"colou?r",
        r"https?://\S+",
        r"foo(?!bar)",  # lookarounds of one width that match one way
        r"(?<=\.)\s",
        r"\n(?=\n)",
        r'(?<!\\)"',
        r"<tool_call>.*?</tool_call>",
        r".*Final Answer:",
        r"(<)?\w+(?(1)>)@",  # a conditional with no "no" branch, nothing after it that \w+ could take
    ]
    served = [r"x{3}", r"^there", r"Hel+o", r"ag\w+", r"<.*>", "$", "^", "again"]  # the other tests' patterns
    assert {p: _backtracking(p) for p in ordinary + served} == dict.fromkeys(ordinary + served)


def test_the_reported_lookaround_patterns_are_refused_for_their_lookaround():
    """The shapes that got past the check: a lookaround's part was checked on its own, without the repeats around it, so
    a repeat over a lookahead that scans to the end of the text was admitted, and its search was cubic or worse (about a
    second, or minutes, over 2048 characters of prose)."""
    from trnrt.serve.api import _backtracking

    reported = [
        r"(?:[^\n](?=(?:[^\n](?=[^\n]*\n))+))+Y",
        r"[^~]+(?=[^~]*~~)~",
        r"(?:[^~](?![^~]*~~))+~~",
        r"(?:.(?!.*@@))+@@",
    ]
    for p in reported:
        assert (_backtracking(p) or "").startswith("it has a lookaround that can match more than one way"), p


def test_an_optional_part_that_can_match_the_empty_text_matches_it_two_ways():
    """(?:b?)? matches the empty text two ways, taken empty and skipped, and the matcher tries both: sixteen such parts
    in a row, admitted at cost 16, took 4.5 s a search over 2048 characters of "z" (65,536 tries at every position;
    CPython 3.13, one Apple M3 Max core). A possessive one gives nothing back, so it matches one way."""
    from trnrt.serve.api import _search_cost

    letters = "bcdefghijklmnopq"
    chain = "".join(f"(?:{c}?)?" for c in letters) + "X"
    assert (_search_cost(chain)[0] or "").startswith("it has optional parts that can match the empty text"), chain
    assert _search_cost("(?:b?)?X") == (None, 2 * _search_cost("(?:b)?X")[1])
    assert _search_cost("".join(f"(?:{c}?)?+" for c in letters) + "X") == (None, 1)


def test_a_capturing_group_in_a_repeat_costs_its_marks():
    """The matcher marks a capturing group at both ends at every iteration, so (\\w)+@ took 2.8 times as long as \\w+@
    over 2048 characters of "a" (CPython 3.13, one Apple M3 Max core), and after 16 ways 0.94 s against 0.34 s, where
    both were priced 16. A group's two marks are now two steps of the repeat's walk: (\\w)+@ costs 3."""
    from trnrt.serve.api import _search_cost

    assert _search_cost(r"\w+@") == (None, 1) and _search_cost(r"(?:\w)+@") == (None, 1)
    assert _search_cost(r"(\w)+@") == (None, 3)
    why, cost = _search_cost(r"(?:a|a)(?:a|a)(?:a|a)(?:a|a)(\w)+@")
    assert cost == 48 and (why or "").startswith("it can take 48 times as long to search"), why


def test_a_requests_stop_regex_patterns_share_one_cost_bound(app):
    """Each pattern is searched at every streamed piece, so the bound is the request's: two patterns at the
    per-pattern bound (16 ways each) are served, a third is refused, though each alone would be served."""
    from trnrt.serve.api import MAX_STOP_REGEX_COST, MAX_STOP_REGEX_REQUEST_COST, _search_cost

    ways16 = [
        r"(?:a|a)(?:a|a)(?:a|a)(?:a|a)\w+@",
        r"(?:b|b)(?:b|b)(?:b|b)(?:b|b)\w+@",
        r"(?:c|c)(?:c|c)(?:c|c)(?:c|c)\w+@",
    ]
    assert [_search_cost(p) for p in ways16] == [(None, MAX_STOP_REGEX_COST)] * 3
    assert 2 * MAX_STOP_REGEX_COST <= MAX_STOP_REGEX_REQUEST_COST < 3 * MAX_STOP_REGEX_COST
    client, eng = app(PLAIN)
    assert client.post("/v1/completions", json={"prompt": [1], "stop_regex": ways16[:2]}).status_code == 200
    r = client.post("/v1/completions", json={"prompt": [1], "stop_regex": ways16})
    assert r.status_code == 400 and r.json()["detail"].startswith("stop_regex patterns are refused: together"), r.text
    assert len(eng.added) == 1


def test_a_requests_stop_regex_cost_counts_every_choice(app):
    """Each of a request's `n` choices has its own detokenizer, which searches every pattern at each of its streamed
    pieces, so the request's bound counts its patterns' cost once per choice: (?s)<think>.*</think> (cost 9) is served
    with n = 3 (27) and refused with n = 4 (36), though alone it is served."""
    from trnrt.serve.api import MAX_STOP_REGEX_REQUEST_COST, _search_cost

    think = r"(?s)<think>.*</think>"
    assert _search_cost(think) == (None, 9) and 3 * 9 <= MAX_STOP_REGEX_REQUEST_COST < 4 * 9
    client, eng = app(PLAIN)
    assert client.post("/v1/completions", json={"prompt": [1], "n": 3, "stop_regex": think}).status_code == 200
    r = client.post("/v1/completions", json={"prompt": [1], "n": 4, "stop_regex": [think]})
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert detail.startswith("stop_regex patterns are refused: together they can take 9 times as long"), detail
    assert "for each of the request's 4 choices, 36 in all, more than 32" in detail and "or fewer choices" in detail
    assert len(eng.added) == 3  # the served request's three choices, none of the refused one's


def test_stop_regex_search_reaches_back_at_most_the_span(monkeypatch):
    """The search for an unbounded pattern starts STOP_REGEX_SPAN (+ 8) characters before the text already searched,
    which bounds its cost: a match starting further back is not found (the limitation, pinned), a later one is."""
    import re

    from trnrt.serve import api

    class Chars:  # token i is the character chr(i)
        def decode(self, ids, skip_special_tokens=True):
            return "".join(map(chr, ids))

    def run(span):
        monkeypatch.setattr(api, "STOP_REGEX_SPAN", span)
        d = api._Detok(Chars(), regexes=((re.compile(r"<.*>"), None),))
        text = "<" + "abcdefghijklmnopqrstuvwxyz" + "> then <ab> and more"
        ids = [ord(x) for x in text]
        out = "".join(d.push(ids[i : i + 2], i + 2 >= len(ids)) for i in range(0, len(ids), 2))
        return out, d.matched

    assert run(2048) == ("<abcdefghijklmnopqrstuvwxyz>", r"<.*>")
    assert run(4) == ("<abcdefghijklmnopqrstuvwxyz> then <ab>", r"<.*>")


@pytest.mark.parametrize("key", ["include_stop_str_in_output", "no_stop_trim"])
def test_the_stop_text_is_kept_when_asked(app, key):
    client, _ = app(PLAIN)
    base = {"prompt": [1], "stop": " again"}
    assert client.post("/v1/completions", json=base).json()["choices"][0]["text"] == "Hello there"
    r = client.post("/v1/completions", json={**base, key: True})
    assert r.json()["choices"][0]["text"] == "Hello there again"
    assert r.json()["choices"][0]["matched_stop"] == " again"
    events = _events(client, {**base, key: True, "stream": True})
    assert "".join(e["choices"][0]["text"] for e in events) == "Hello there again"


def test_the_stop_token_is_not_in_the_text(app):
    """The EOS (or a stop_token_ids entry) that ends a request is left out of its text even when the tokenizer does not
    mark it special; matched_stop names it. Kept with no_stop_trim."""

    def eos(eng):
        eng.mcfg.eos_token_ids = (ID["!"],)

    client, _ = app(PLAIN, eos)
    choice = client.post("/v1/completions", json={"prompt": [1]}).json()["choices"][0]
    assert choice["text"] == "Hello there again" and choice["matched_stop"] == ID["!"]
    assert client.post("/v1/completions", json={"prompt": [1], "no_stop_trim": True}).json()["choices"][0]["text"] == (
        "Hello there again!"
    )
    client2, _ = app(PLAIN)
    choice = client2.post("/v1/completions", json={"prompt": [1], "stop_token_ids": [ID["!"]]}).json()["choices"][0]
    assert choice["text"] == "Hello there again" and choice["matched_stop"] == ID["!"]
    events = _events(client2, {"prompt": [1], "stop_token_ids": [ID["!"]], "stream": True})
    assert "".join(e["choices"][0]["text"] for e in events) == "Hello there again"


def test_skip_special_tokens_false_keeps_special_tokens(app):
    def special(eng):
        eng.tokenizer.add_special_tokens({"unk_token": "<unk>"})  # id 0 becomes special

    client, _ = app([ID["Hello"], ID["<unk>"], ID["!"]], special)
    assert client.post("/v1/completions", json={"prompt": [1]}).json()["choices"][0]["text"] == "Hello!"
    r = client.post("/v1/completions", json={"prompt": [1], "skip_special_tokens": False})
    assert r.json()["choices"][0]["text"] == "Hello<unk>!"
    events = _events(client, {"prompt": [1], "skip_special_tokens": False, "stream": True})
    assert "".join(e["choices"][0]["text"] for e in events) == "Hello<unk>!"
