"""Speculative decoding on the real model code: GLM-5.3-Flash's architecture at a few random layers plus an MTP layer (a
copy of layer 3's weights with its own norms and eh_proj), fp32 on the CPU, the engine's own scheduler, runner and board
graphs. The invariant is the one MTP is for: **greedy output does not change**, with or without speculation, whatever
the draft layer predicts. Then the MTP layer's own state:

- greedy tokens with speculation equal plain greedy, in the static form and in the dynamic one (speculation only while
  at most 2 requests decode, the plain decode step above, with the plain steps writing the MTP layer's KV);
- the plain steps' KV write (DYN_MTP_KV "1": only the MTP layer's cache writes) leaves the MTP caches bitwise equal to what
  running the whole layer (DYN_MTP_KV "full": the reference) leaves, and with the write off ("0") they differ: the control
  that the comparison can see a missing write;
- the MTP pass over a prefill chunk in the chunk form (mtp_chunk_q), with and without KV-only windows, changes no token;
- a prefix restored from the host tier and then speculated equals a cold request.

Needs transformers with glm5_next, as the other model tests."""

from __future__ import annotations

import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests import test_glm5_next as T
from trnrt.engine.request import SamplingParams
from trnrt.models import decoder, mla

MTP_SEED = 52


def add_mtp(path: str, seed: int = MTP_SEED) -> None:
    """Add an MTP layer to the checkpoint at `path`: layer `n` is layer 3's weights (no hyper-connection parameters) with
    its own enorm / hnorm / final norm and eh_proj, and num_nextn_predict_layers 1."""
    f = os.path.join(path, "model.safetensors")
    t = load_file(f)
    with open(os.path.join(path, "config.json")) as fh:
        cfg = json.load(fh)
    tc = cfg["text_config"]
    n, H = tc["num_hidden_layers"], tc["hidden_size"]
    src, dst = "model.language_model.layers.3.", f"model.language_model.layers.{n}."
    for k in [k for k in t if k.startswith(src) and ".hc_" not in k]:
        t[dst + k[len(src) :]] = t[k].clone()
    g = torch.Generator().manual_seed(seed)
    t[dst + "enorm.weight"] = 1 + 0.1 * torch.randn(H, generator=g)
    t[dst + "hnorm.weight"] = 1 + 0.1 * torch.randn(H, generator=g)
    t[dst + "shared_head.norm.weight"] = 1 + 0.1 * torch.randn(H, generator=g)
    t[dst + "eh_proj.weight"] = torch.randn(H, 2 * H, generator=g) * (2 * H) ** -0.5
    save_file(t, f, metadata={"format": "pt"})
    tc["num_nextn_predict_layers"] = 1
    with open(os.path.join(path, "config.json"), "w") as fh:
        json.dump(cfg, fh)


@pytest.fixture(scope="module")
def mtp_model(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("glm5_next_mtp"))
    T.build(path, seed=2, index_topk=16)
    add_mtp(path)
    return path


BASE = {"max_num_seqs": 4, "max_prefill_tokens": 16, "max_model_len": 512, "num_pages": 512}
SPEC = {"spec_method": "mtp", "spec_k": 2}


@pytest.fixture(autouse=True)
def restore_routes(monkeypatch):
    """The engine writes the decoder's MTP routes when it is built (engine.set_decode_routes); undo them after the test."""
    monkeypatch.setattr(decoder, "MTP_CHUNK_Q", decoder.MTP_CHUNK_Q)
    monkeypatch.setattr(decoder, "DYN_MTP_KV", decoder.DYN_MTP_KV)


def generate(path, prompts, n, **kw):
    params = SamplingParams(max_new_tokens=n, temperature=0.0, ignore_eos=True)
    eng = T.engine(path, **{**BASE, **kw})
    try:
        return eng, [r.output_ids for r in eng.generate(prompts, params)]
    except BaseException:
        eng.close()
        raise


def test_speculative_greedy_equals_plain_greedy(mtp_model):
    ps = T.prompts(11, (9, 23, 14))
    plain, want = generate(mtp_model, ps, 20)
    plain.close()
    eng, got = generate(mtp_model, ps, 20, **SPEC)
    try:
        assert got == want and all(len(w) == 20 for w in want)
        assert eng.spec_proposed > 0 and eng.spec_verifies > 0, "the engine never speculated"
        assert eng.spec_accepted <= eng.spec_proposed
    finally:
        eng.close()


def test_dynamic_speculation_with_the_plain_steps_writing_the_mtp_kv_changes_no_token(mtp_model):
    ps = T.prompts(11, (9, 23, 14))
    plain, want = generate(mtp_model, ps, 20)
    plain.close()
    eng, got = generate(mtp_model, ps, 20, **SPEC, spec_max_batch=2, dyn_mtp_kv=True)
    try:
        assert decoder.DYN_MTP_KV == "1", "the engine's route reached the decoder"
        assert got == want
        assert eng.dyn_flips >= 1 and eng.dyn_plain_steps > 0 and eng.dyn_spec_steps > 0
    finally:
        eng.close()


def test_a_plain_step_leaves_the_mtp_layers_kv_the_always_on_layer_would_have(mtp_model, monkeypatch):
    """DYN_MTP_KV "1" writes only the layer's K/V/pool keys on a plain step; "full" runs the whole layer (the reference whose
    writes the always-on pass makes). Their caches are bitwise equal; with the write off ("0") they are not."""
    ps = T.prompts(11, (9, 23, 14))
    calls = {"n": 0}
    orig = mla.write_kv_only

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(mla, "write_kv_only", counted)
    got = {}
    for mode in ("1", "full", "0"):
        calls["n"] = 0
        params = SamplingParams(max_new_tokens=16, temperature=0.0, ignore_eos=True)
        eng = T.engine(mtp_model, **BASE, **SPEC, spec_max_batch=2, dyn_mtp_kv=mode != "0")
        decoder.DYN_MTP_KV = mode  # "full" has no engine flag: the reference form, set after the engine's own routes
        try:
            out = [r.output_ids for r in eng.generate(ps, params)]
            m = eng.runner.model.mtp
            got[mode] = (out, m.k_cache.clone(), m.v_cache.clone(), calls["n"])
        finally:
            eng.close()
    assert got["1"][0] == got["full"][0] == got["0"][0], "greedy tokens do not depend on the write"
    assert got["1"][3] > 0, "the KV-only write was not exercised"
    for i, name in ((1, "k_cache"), (2, "v_cache")):
        assert torch.equal(got["1"][i], got["full"][i]), f"{name}: the KV-only write differs from the whole layer's"
    assert (got["1"][1].float() - got["0"][1].float()).abs().max() > 1e-3, (
        "with the write off the cache should differ (control)"
    )


@pytest.mark.parametrize("kv_only", [False, True])
def test_the_mtp_pass_over_a_prefill_chunk_in_the_chunk_form_changes_no_token(mtp_model, kv_only, monkeypatch):
    ps = T.prompts(13, (41, 23, 57))
    plain, want = generate(mtp_model, ps, 14)
    plain.close()
    count = {}
    for name in ("forward_mtp_kv", "prep_extend_chunk", "_mtp_pass"):
        f = getattr(decoder.DecoderForCausalLM, name)

        def wrapped(self, *a, _f=f, _n=name, **k):
            count[_n] = count.get(_n, 0) + 1
            return _f(self, *a, **k)

        monkeypatch.setattr(decoder.DecoderForCausalLM, name, wrapped)
    eng, got = generate(mtp_model, ps, 14, **SPEC, mtp_chunk_q=4, mtp_kv_only=kv_only)
    try:
        assert decoder.MTP_CHUNK_Q == 4 and got == want
        assert count.get("prep_extend_chunk", 0) > 0, "the chunk form never ran"
        assert (count.get("forward_mtp_kv", 0) > 0) is kv_only, count
    finally:
        eng.close()


def test_a_prefix_restored_from_the_host_tier_then_speculated_equals_a_cold_request(mtp_model):
    ps = T.prompts(17, (37,))
    params = SamplingParams(max_new_tokens=12, temperature=0.0, ignore_eos=True)
    plain, (want,) = generate(mtp_model, ps, 12)
    plain.close()
    eng = T.engine(mtp_model, **BASE, **SPEC, host_cache_gb=0.01)
    try:
        (cold,) = eng.generate(ps, params)
        assert cold.output_ids == want
        eng.radix.evict(eng.radix.total_pages())  # the prefix and its checkpoints go to the host tier
        assert eng.radix.total_pages() == 0 and eng.host_tier.entries and eng.host_tier.ckpts
        (again,) = eng.generate(ps, params)
        assert again.num_cached_tokens > 0, "nothing was restored"
        assert again.output_ids == want
        assert again.spec_proposed > 0
    finally:
        eng.close()
