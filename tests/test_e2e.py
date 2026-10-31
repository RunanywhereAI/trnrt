"""End to end on CPU: two engines of a tiny GLM-5.3-Flash-architecture model behind the session-affine router, over HTTP.

The model is tests/test_glm5_next.py's small random one, scripted so that greedy decoding writes a GLM tool call: every
residual branch's output projection is zeroed (attention, KDA, dense MLP, experts), so each position's final hidden
state is a positive multiple of its own token's embedding (the mHC stream mixing and head only scale it), and the
embedding and lm_head rows of a short chain of tokens make each one's argmax the next:

    ASSISTANT -> <tool_call> -> get_weather -> <arg_key> -> city -> </arg_key> -> <arg_value> -> Paris
              -> </arg_value> -> </tool_call> -> EOS

A WordLevel tokenizer whose chat template ends every prompt with ASSISTANT, and whose decoder joins tokens with no
space, turns that into `<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>`. So the
engine's real graphs, sampler and overlap scheduling produce the call, and the server's detokenizer and GLM parser
read it back as an OpenAI tool call.

Checked through the router:
- a chat completion returns the tool call (finish_reason "tool_calls"), non-streaming and streamed as SSE deltas;
- a session's second, identical request goes to the same engine and resumes from the prefix cache (cached_tokens > 0);
- a second session goes to the other engine, so both engines served.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest
import torch

pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
glm = pytest.importorskip("tests.test_glm5_next")
httpx = pytest.importorskip("httpx")

CHAIN = ["ASSISTANT", "<tool_call>", "get_weather", "<arg_key>", "city", "</arg_key>", "<arg_value>", "Paris",
         "</arg_value>", "</tool_call>"]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "the weather in a city",
                                           "parameters": {"type": "object",
                                                          "properties": {"city": {"type": "string"}},
                                                          "required": ["city"]}}}]
PROMPT = " ".join(f"t{i}" for i in range(5, 41))  # 36 words: the prompt spans several 4-token pages


def _script(path: str) -> dict:
    """Build the tiny model at `path`, script it to write CHAIN then EOS, save it with its tokenizer. Returns the
    word -> id map of the chain."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from trnrt.config import ModelConfig

    model = glm.build(path, seed=3)
    mcfg = ModelConfig.from_pretrained(path)
    vocab, eos = mcfg.vocab_size, mcfg.eos_token_ids[0]
    free = [i for i in range(200, vocab) if i != eos]
    ids = dict(zip(CHAIN, free))
    nxt = [ids[w] for w in CHAIN[1:]] + [eos]
    params = dict(model.named_parameters())
    emb = next(p for n, p in params.items() if n.endswith("embed_tokens.weight"))
    head = params["lm_head.weight"]
    D = emb.shape[1]
    assert len(CHAIN) <= D
    with torch.no_grad():
        for n, p in params.items():
            if any(k in n for k in ("o_proj", "down_proj", "out_proj")):
                p.zero_()  # every residual branch writes nothing back
            elif n.endswith("norm.weight") and ".layers." not in n:
                p.fill_(1.0)  # the final norm keeps the embedding's sign
        head.zero_()
        for i, w in enumerate(CHAIN):
            emb[ids[w]].zero_()
            emb[ids[w], i] = 10.0
            head[nxt[i], i] = 50.0
    model.save_pretrained(path, safe_serialization=True)

    words = {f"t{i}": i for i in range(vocab)}
    for w, i in ids.items():
        words.pop(f"t{i}")
        words[w] = i
    tk = Tokenizer(models.WordLevel(words, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.decoder = decoders.Fuse()
    # The chain ends on the model's EOS, which the engine keeps in a request's output ids and the server's detokenizer drops
    # only as a special token (skip_special_tokens), as GLM-5.3-Flash's own tokenizer marks every EOS. Unmarked, its word
    # ("t1") came back as the message's content.
    t = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="t0", eos_token=f"t{eos}")
    t.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}ASSISTANT"
    t.save_pretrained(path)
    return ids


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(app):
    """uvicorn on a thread: (base URL, server)."""
    import uvicorn

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    t0 = time.monotonic()
    while not server.started:
        assert time.monotonic() - t0 < 60, "server did not start"
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}", server


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    from trnrt.serve.api import build_app
    from trnrt.serve.router import Router, build_router_app

    path = str(tmp_path_factory.mktemp("e2e"))
    _script(path)
    engines, servers, urls = [], [], []
    try:
        for _ in range(2):
            eng = glm.engine(path, max_prefill_tokens=16)
            engines.append(eng)
            url, srv = _start(build_app(eng, "glm"))
            urls.append(url)
            servers.append(srv)
        rurl, rsrv = _start(build_router_app(Router(urls), poll_s=0.2))
        servers.append(rsrv)
        with httpx.Client(base_url=rurl, timeout=120) as client:
            t0 = time.monotonic()
            while client.get("/health").status_code != 200:
                assert time.monotonic() - t0 < 30, "router not healthy"
                time.sleep(0.1)
            yield client, urls
    finally:
        for srv in servers:
            srv.should_exit = True
        for eng in engines:
            eng.close()


def _body(session: str, stream: bool = False) -> dict:
    return {"model": "glm", "messages": [{"role": "user", "content": PROMPT}], "tools": TOOLS, "temperature": 0,
            "max_tokens": 16, "session_id": session, "stream": stream,
            **({"stream_options": {"include_usage": True}} if stream else {})}


def _served(url: str) -> float:
    """Requests the engine finished, over every finish reason (one labelled line each)."""
    return sum(float(line.split()[-1]) for line in httpx.get(url + "/metrics").text.splitlines()
               if line.startswith("trnrt:request_success_total{"))


def test_tool_call_prefix_hit_and_both_engines(stack):
    client, urls = stack
    first = client.post("/v1/chat/completions", json=_body("s1")).json()
    choice = first["choices"][0]
    assert choice["finish_reason"] == "tool_calls", first
    (call,) = choice["message"]["tool_calls"]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    assert not choice["message"].get("content")

    again = client.post("/v1/chat/completions", json=_body("s1")).json()
    assert again["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    cached = again["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
    assert 0 < cached < again["usage"]["prompt_tokens"], again["usage"]  # the same engine: a prefix-cache hit

    # A second session goes to the other engine (the first holds session s1), streamed.
    events = []
    with client.stream("POST", "/v1/chat/completions", json=_body("s2", stream=True)) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(line[len("data: "):])
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    name, args, finish = None, "", None
    for c in chunks:
        for ch in c.get("choices", []):
            for tc in ch.get("delta", {}).get("tool_calls") or []:
                name = tc.get("function", {}).get("name") or name
                args += tc.get("function", {}).get("arguments") or ""
            finish = ch.get("finish_reason") or finish
    assert name == "get_weather" and json.loads(args) == {"city": "Paris"} and finish == "tool_calls", chunks
    assert chunks[-1]["usage"]["completion_tokens"] == len(CHAIN)  # the chain's tokens after ASSISTANT, then EOS

    served = [_served(u) for u in urls]
    assert sorted(served) == [1.0, 2.0], served  # s1's two requests on one engine, s2's on the other
