"""Request values the engine would fail on in a step are refused before it sees them (trnrt/serve/api.py _params and
EngineLoop.submit), on a real engine: a tiny random GLM-5.3-Flash-architecture model (tests/test_glm5_next.py) on CPU.

numpy refuses a negative or fractional seed, the sampling inputs are int64 arrays, and a prompt id outside the
vocabulary is outside the embedding: each fails the step that samples the request. At tp 1 a failed step fails every
request in it; at tp > 1 it fails the engine for good (EngineLoop._fail, tests/test_api_engine_health.py). So:
- each bad value is a 400 and never reaches add_request;
- a long request decoding meanwhile finishes untouched, and the next good request is served;
- positive control: the same seed handed straight to the engine loop does fail the step, and its request with it.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from tests.test_api_cached_tokens import _save_tokenizer

PROMPT = [5, 17, 42, 99, 7, 3, 250, 11]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    from trnrt.serve import api

    glm = pytest.importorskip("tests.test_glm5_next")
    d = tmp_path_factory.mktemp("api_checks")
    glm.build(str(d), seed=3)
    cfg = json.loads((d / "config.json").read_text())
    vocab = cfg.get("vocab_size") or cfg["text_config"]["vocab_size"]
    _save_tokenizer(d, vocab)
    eng = glm.engine(str(d))
    added, real_add = [], eng.add_request

    def add(prompt_ids, params, **kw):
        added.append(params)
        return real_add(prompt_ids, params, **kw)
    eng.add_request = add
    with TestClient(api.build_app(eng, "test-model")) as c:
        c.engine, c.vocab, c.added = eng, vocab, added
        c.loop = next(lp for lp in api._LOOPS if lp.engine is eng)
        yield c


def test_bad_values_are_400_and_a_running_request_is_untouched(client):
    c = client
    good = {"prompt": PROMPT, "max_tokens": 40, "temperature": 0, "ignore_eos": True}
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("r", c.post("/v1/completions", json=good)))
    th.start()
    deadline = time.monotonic() + 120
    while not c.engine.scheduler.running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert c.engine.scheduler.running, "the long request never started"
    n_added = len(c.added)
    sampled = {"prompt": PROMPT, "max_tokens": 4, "temperature": 1.0}
    for field, value in (("seed", -1), ("seed", 1.5), ("seed", "abc"), ("seed", 2 ** 63), ("top_k", 2 ** 63),
                         ("top_k", -5), ("prompt", [c.vocab]), ("prompt", PROMPT + [-1]), ("top_p", 0)):
        r = c.post("/v1/completions", json={**sampled, field: value})
        assert r.status_code == 400, (field, value, r.text)
        assert c.get("/health").status_code == 200
    r = c.post("/v1/completions", content=b'{"prompt": [5, 17], "temperature": NaN}',
               headers={"content-type": "application/json"})
    assert r.status_code == 400, r.text
    assert len(c.added) == n_added  # none reached the engine
    th.join(120)
    r = out["r"]
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["finish_reason"] == "length" and r.json()["usage"]["completion_tokens"] == 40
    r = c.post("/v1/completions", json={**sampled, "seed": 7, "top_k": 2 ** 63 - 1})
    assert r.status_code == 200 and r.json()["usage"]["completion_tokens"] == 4, r.text


def test_positive_control_the_seed_fails_a_step_when_it_gets_there(client):
    from trnrt.engine.request import SamplingParams

    async def direct():
        sub = client.loop.submit(PROMPT, SamplingParams(max_new_tokens=4, temperature=1.0, seed=-1))
        first = await asyncio.wait_for(sub.out.get(), 60)
        return first, await asyncio.wait_for(sub.out.get(), 120)

    first, then = asyncio.run(direct())
    assert first[0] == "accepted" and then[0] == "error" and then[2] == 500, then
    assert "engine step failed" in then[1]
    # tp 1: the engine still serves (at tp > 1 this step would have failed it for good)
    r = client.post("/v1/completions", json={"prompt": PROMPT, "max_tokens": 2, "temperature": 0})
    assert r.status_code == 200, r.text
