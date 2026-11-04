"""OpenAI usage.prompt_tokens_details.cached_tokens from the server (trnrt/serve/api.py), on a GLM-5.3-Flash engine on
CPU (a tiny random model, tests/test_glm5_next.py: its KDA layers resume a prefix from a state checkpoint).

- non-streaming: cold 0; a session's next turn (the earlier prompt, its reply and a new message) reports exactly the
  engine's hit (its num_cached_tokens: the checkpoint the earlier turn left at its last page boundary), and the
  engine's trnrt:prefix_cache_hits_total moves by the same;
- streaming with stream_options.include_usage: one last chunk with no choices and the same usage, before [DONE];
  without it, no usage chunk (as OpenAI).
A tiny WordLevel tokenizer is saved next to the checkpoint so the server can decode; prompts are token ids.

The hit is a next turn's, not a second prompt sharing only a prefix: like the served configuration, the engine takes no
periodic prefill checkpoints, so a shared prefix is resumable only from a junction checkpoint, and the request that
leaves it is the second one to compute that prefix, not the first.
"""

from __future__ import annotations

import json

import pytest
import torch


def _save_tokenizer(path, vocab: int) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.WordLevel({f"t{i}": i for i in range(vocab)}, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="t0").save_pretrained(str(path))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    from trnrt.serve.api import build_app

    glm = pytest.importorskip("tests.test_glm5_next")
    d = tmp_path_factory.mktemp("api_cached")
    glm.build(str(d), seed=1, index_topk=16)
    cfg = json.loads((d / "config.json").read_text())
    vocab = cfg.get("vocab_size") or cfg["text_config"]["vocab_size"]
    _save_tokenizer(d, vocab)
    eng = glm.engine(str(d), max_prefill_tokens=64)
    with TestClient(build_app(eng, "test-model")) as c:
        c.engine, c.vocab = eng, vocab
        yield c


def _hits(c) -> float:
    for line in c.get("/metrics").text.splitlines():
        if line.startswith("trnrt:prefix_cache_hits_total"):
            return float(line.split()[-1])
    raise AssertionError("no trnrt:prefix_cache_hits_total")


def _random(c, g, n):
    return torch.randint(1, c.vocab, (n,), generator=g).tolist()


def _turn(c, prompt, max_tokens):
    """One greedy turn: (the response, its reply's token ids, read back from its logprobs' tokens)."""
    r = c.post("/v1/completions", json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "logprobs": 0})
    assert r.status_code == 200, r.text
    r = r.json()
    return r, c.engine.tokenizer.convert_tokens_to_ids(r["choices"][0]["logprobs"]["tokens"])


def test_cached_tokens_in_usage_cold_then_hit(client):
    c = client
    g = torch.Generator().manual_seed(1)
    first = _random(c, g, 25)
    h0 = _hits(c)
    r1, reply = _turn(c, first, 4)
    h1 = _hits(c)
    second = first + reply + _random(c, g, 5)  # the session's next turn
    r2, _ = _turn(c, second, 4)
    h2 = _hits(c)
    u1, u2 = r1["usage"], r2["usage"]
    assert u1["prompt_tokens"] == len(first) and u1["prompt_tokens_details"]["cached_tokens"] == 0 and h1 == h0
    cached = u2["prompt_tokens_details"]["cached_tokens"]
    assert cached == h2 - h1, "usage must report the engine's own hit for the request"
    ps = c.engine.cfg.page_size
    assert 0 < cached <= len(first) + len(reply) and cached % ps == 0, (
        "the next turn resumes from the checkpoint the first left at its last page boundary"
    )


def test_streaming_usage_chunk_only_when_asked(client):
    c = client
    g = torch.Generator().manual_seed(2)
    first = _random(c, g, 25)
    _, reply = _turn(c, first, 3)
    second = first + reply + _random(c, g, 5)

    def stream(body):
        with c.stream("POST", "/v1/completions", json=body) as r:
            lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
        return [json.loads(ln[6:]) for ln in lines if ln != "data: [DONE]"], lines[-1]

    h1 = _hits(c)
    evs, last = stream(
        {"prompt": second, "max_tokens": 3, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    )
    h2 = _hits(c)
    assert last == "data: [DONE]"
    tail = evs[-1]
    assert tail["choices"] == [] and tail["usage"]["completion_tokens"] == 3
    assert tail["usage"]["prompt_tokens_details"]["cached_tokens"] == h2 - h1 > 0
    assert all("usage" not in e for e in evs[:-1])
    evs, _ = stream({"prompt": second + [5], "max_tokens": 3, "temperature": 0, "stream": True})
    assert all("usage" not in e for e in evs)
