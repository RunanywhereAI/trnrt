"""SGLang's native API (trnrt/serve/native.py) on a GLM-5.3-Flash engine on CPU (a tiny random model,
tests/test_glm5_next.py), with the host-memory tier on: /generate gives the tokens and logprobs /v1/completions gives,
/flush_cache empties the prefix cache of both tiers so the next turn is cold and the one after it hits again, and
/health_generate runs its token through the engine.
"""

from __future__ import annotations

import json

import pytest
import torch

HOST = 0.05  # GiB per rank of host-memory tier: ample for the tiny model's pages and rows


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    from tests.test_api_cached_tokens import _save_tokenizer
    from trnrt.serve.api import build_app

    glm = pytest.importorskip("tests.test_glm5_next")
    d = tmp_path_factory.mktemp("api_native")
    glm.build(str(d), seed=1, index_topk=16)
    cfg = json.loads((d / "config.json").read_text())
    vocab = cfg.get("vocab_size") or cfg["text_config"]["vocab_size"]
    _save_tokenizer(d, vocab)
    eng = glm.engine(str(d), max_prefill_tokens=64, host_cache_gb=HOST)
    with TestClient(build_app(eng, "test-model")) as c:
        c.engine, c.vocab = eng, vocab
        yield c


def _random(c, g, n):
    return torch.randint(1, c.vocab, (n,), generator=g).tolist()


def _completion(c, prompt, n):
    r = c.post(
        "/v1/completions", json={"prompt": prompt, "max_tokens": n, "temperature": 0, "logprobs": 0, "ignore_eos": True}
    )
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    ids = c.engine.tokenizer.convert_tokens_to_ids(choice["logprobs"]["tokens"])
    return ids, choice["logprobs"]["token_logprobs"], r.json()["usage"]["prompt_tokens_details"]["cached_tokens"]


def _generate(c, prompt, n):
    body = {
        "input_ids": prompt,
        "sampling_params": {"max_new_tokens": n, "temperature": 0, "ignore_eos": True},
        "return_logprob": True,
    }
    r = c.post("/generate", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_generate_matches_completions(client):
    c = client
    prompt = _random(c, torch.Generator().manual_seed(11), 21)
    ids, lps, _ = _completion(c, prompt, 5)
    out = _generate(c, prompt, 5)
    meta = out["meta_info"]
    assert out["output_ids"] == ids and meta["completion_tokens"] == 5 and meta["prompt_tokens"] == 21
    assert [t for _, t, _ in meta["output_token_logprobs"]] == ids
    assert [lp for lp, _, _ in meta["output_token_logprobs"]] == pytest.approx(lps, abs=1e-5)
    assert meta["finish_reason"] == {"type": "length", "length": 5}
    assert out["text"] == c.engine.tokenizer.decode(ids, skip_special_tokens=True)


def test_flush_cache_empties_both_tiers_and_the_cache_works_after(client):
    c = client
    eng = c.engine
    g = torch.Generator().manual_seed(12)
    first = _random(c, g, 25)
    reply, _, cold = _completion(c, first, 4)
    second = first + reply + _random(c, g, 5)  # the next turn: resumes from the first's checkpoint
    _, _, hit = _completion(c, second, 4)
    assert cold == 0 and hit > 0
    assert eng.radix.total_pages() > 0 and (eng.host_tier.entries or eng.radix.num_ckpts)

    r = c.post("/flush_cache?timeout=30")  # the engine may still be reading back its last step: wait for idle
    assert r.status_code == 200 and r.text.startswith("Cache flushed."), r.text
    assert eng.radix.total_pages() == 0 and eng.radix.num_ckpts == 0
    assert eng.pool.num_free == eng.pool.num_usable
    assert not eng.host_tier.entries and not eng.host_tier.ckpts

    again, _, cached = _completion(c, second, 4)
    assert cached == 0  # cold: nothing of the first turn is left
    third = second + again + _random(c, g, 5)
    _, _, hit = _completion(c, third, 4)
    assert hit > 0  # the cache serves again after the flush


def test_health_generate_on_the_engine(client):
    r = client.get("/health_generate")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
