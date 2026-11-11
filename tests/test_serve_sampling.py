"""The sampling extras of /v1/chat/completions and /v1/completions (trnrt/serve/api.py): `n` choices, SGLang's
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
    every choice's output. A seed gives choice i seed + i (vLLM), so seeded choices differ and reproduce."""
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
    ],
    ids=repr,
)
def test_what_cannot_be_served_is_refused_before_the_engine(app, extra):
    client, eng = app(PLAIN)
    for path, body in (("/v1/chat/completions", MSG), ("/v1/completions", {"prompt": [1]})):
        r = client.post(path, json={**body, **extra})
        assert r.status_code == 400, (path, r.text)
    assert eng.added == []
    ok = {"n": 2, "best_of": 2, "seed": 2**63 - 2, "min_tokens": 0, "spaces_between_special_tokens": True}
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
    """SGLang's stop_regex: the output ends at the end of the first match, the matched text kept (SGLang's trim looks
    for the pattern itself), finish "stop", matched_stop the pattern. The engine knows nothing of it, so the request is
    aborted there. The match spans two tokens (" there", " again"), so it is only found by searching back into text
    already pushed: a bounded pattern's search starts its longest match before the new text. The engine takes 10 ms a
    token, so the abort reaches it before it ends the request on its own."""
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


def test_stop_regex_unbounded_and_anchored(app):
    """A pattern with an unbounded repeat is searched back over the text (STOP_REGEX_SPAN characters before the new
    text); an anchor keeps its meaning (the search starts at a position, never at a slice)."""
    client, _ = app(PLAIN * 4)
    r = client.post("/v1/completions", json={"prompt": [1], "max_tokens": 16, "stop_regex": r"!.*again"})
    assert r.json()["choices"][0]["text"] == "Hello there again!Hello there again"
    r = client.post("/v1/completions", json={"prompt": [1], "max_tokens": 16, "stop_regex": r"^there"})
    assert r.json()["choices"][0]["text"] == "Hello there again!" * 4  # "there" never starts the text


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
        r"(?:)+",
    ],
)
def test_stop_regex_refuses_patterns_that_backtrack(app, pattern):
    """CPython's re holds the GIL through a search and cannot be interrupted, so a pattern whose search can take
    exponential time, or a power of the text's length, is refused with the reason before it runs on the event loop."""
    client, eng = app(PLAIN)
    r = client.post("/v1/completions", json={"prompt": [1], "stop_regex": ["again", pattern]})
    assert r.status_code == 400 and r.json()["detail"].startswith(f"stop_regex {pattern!r} is refused: it "), r.text
    assert eng.added == []


def test_stop_regex_admits_ordinary_patterns():
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
    ]
    assert {p: _backtracking(p) for p in ordinary} == dict.fromkeys(ordinary)


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
    mark it special, as vLLM and SGLang leave it out; matched_stop names it. Kept with no_stop_trim."""

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
