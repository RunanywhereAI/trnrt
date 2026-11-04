"""A chat request's prompt is rendered and tokenized on a worker thread with its own tokenizer (trnrt/serve/api.py
build_app prompt_pool), not on the event loop: the prompt ids are the ones chat_prompt gives on the engine's tokenizer
and the greedy output equals a completion over them; and while one chat prompt takes 2 s to render, another request's
stream keeps arriving (on the loop, its tokens waited the whole 2 s). A tiny random GLM-5.3-Flash-architecture model
(tests/test_glm5_next.py) on CPU, a WordLevel tokenizer whose chat template joins the messages' words (word t<i> is
token i), served by uvicorn on a local port (Starlette's TestClient hands a streamed body over only once it is
complete, so it cannot time a stream)."""

from __future__ import annotations

import socket
import threading
import time
import types
from itertools import pairwise

import pytest

TEMPLATE = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
CHAT = {
    "messages": [{"role": "system", "content": "t5 t17 t42"}, {"role": "user", "content": "t99 t7 t3 t250 t11"}],
    "max_tokens": 8,
    "temperature": 0,
    "ignore_eos": True,
}


def _save_tokenizer(path, vocab: int) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.WordLevel({f"t{i}": i for i in range(vocab)}, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    t = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="t0")
    t.chat_template = TEMPLATE
    t.save_pretrained(str(path))


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    import uvicorn

    from trnrt.serve import api

    glm = pytest.importorskip("tests.test_glm5_next")
    d = tmp_path_factory.mktemp("prompt_thread")
    glm.build(str(d))
    _save_tokenizer(d, 384)
    eng = glm.engine(str(d), max_model_len=512, max_prefill_tokens=16)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(
        uvicorn.Config(api.build_app(eng, "test-model"), host="127.0.0.1", port=port, log_level="warning")
    )
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    deadline = time.perf_counter() + 60
    while not srv.started and time.perf_counter() < deadline:
        time.sleep(0.05)
    assert srv.started
    yield types.SimpleNamespace(url=f"http://127.0.0.1:{port}", engine=eng)
    srv.should_exit = True
    th.join(timeout=30)


def test_the_prompt_and_the_output_are_the_loops(server, monkeypatch):
    from trnrt.serve import api

    want, _ = api.chat_prompt(server.engine.tokenizer, CHAT)
    assert want == [5, 17, 42, 99, 7, 3, 250, 11]
    seen, real = [], api.chat_prompt

    def spy(tok, *a):
        seen.append((threading.current_thread().name, tok is server.engine.tokenizer))
        return real(tok, *a)

    monkeypatch.setattr(api, "chat_prompt", spy)
    import httpx

    r = httpx.post(server.url + "/v1/chat/completions", json=CHAT, timeout=60)
    assert r.status_code == 200, r.text
    assert seen == [("trnrt-prompt_0", False)]  # the worker thread, its own tokenizer
    assert r.json()["usage"]["prompt_tokens"] == len(want)
    c = httpx.post(
        server.url + "/v1/completions",
        json={"prompt": want, "max_tokens": 8, "temperature": 0, "ignore_eos": True},
        timeout=60,
    )
    assert r.json()["choices"][0]["message"]["content"] == c.json()["choices"][0]["text"]


def test_another_stream_flows_while_a_prompt_renders(server, monkeypatch):
    import httpx

    from trnrt.serve import api

    real, rendering = api.chat_prompt, threading.Event()

    def slow(*a):
        rendering.set()
        time.sleep(2.0)
        return real(*a)

    monkeypatch.setattr(api, "chat_prompt", slow)
    step = server.engine.step

    def paced():  # the stream outlasts the render (a tiny model on CPU decodes 500 tokens in well under 2 s)
        time.sleep(0.01)
        return step()

    monkeypatch.setattr(server.engine, "step", paced)
    times = []

    def stream():
        body = {"prompt": [5, 17, 42, 99], "max_tokens": 450, "temperature": 0, "ignore_eos": True, "stream": True}
        with httpx.stream("POST", server.url + "/v1/completions", json=body, timeout=120) as r:
            times.extend(
                time.perf_counter() for line in r.iter_lines() if line.startswith("data: ") and line != "data: [DONE]"
            )

    th = threading.Thread(target=stream)
    th.start()
    deadline = time.perf_counter() + 60
    while len(times) < 5 and time.perf_counter() < deadline:
        time.sleep(0.01)
    t0 = time.perf_counter()
    r = httpx.post(server.url + "/v1/chat/completions", json=CHAT, timeout=60)
    t1 = time.perf_counter()
    th.join(timeout=120)
    assert r.status_code == 200 and rendering.is_set() and t1 - t0 >= 2.0
    during = [t for t in times if t0 + 0.2 < t < t0 + 1.8]
    assert times[-1] > t0 + 1.8, (len(times), times[-1] - times[0], times[-1] - t0)
    assert len(during) >= 20, f"{len(during)} stream events while another prompt rendered for 2 s"
    gap = max(b - a for a, b in pairwise(times) if t0 < b < t1)
    assert gap < 0.5, gap
