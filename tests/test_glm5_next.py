"""GLM-5.3-Flash (`glm5_next`: KDA + NoPE DSA with a pooled indexer, mHC hyper-connections, clamped SwiGLU,
shared-expert MoE) against transformers' Glm5NextForConditionalGeneration, on CPU in fp32 with identical weights.
Needs transformers >= 5.18 (skipped below it).

Every model is built by transformers itself from the REAL config.json (tests/fixtures/GLM-5.3-Flash-config.json,
https://huggingface.co/zai-org/GLM-5.3-Flash/raw/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json), truncated
to its first layers (3 KDA layers with dense MLPs, a DSA layer with MoE, KDA layers with MoE, ...), with fewer heads
and experts, a small hidden size and vocabulary, and everything else as published: MLA geometry (q_lora 1536,
kv_lora 512, 256 nope + 256 value dims, no RoPE), the indexer (128-dim heads, kpool 4, tail selection), KDA (128-dim
heads, lower bound -5), mHC (4 streams, 20 Sinkhorn iterations), swiglu_limit 10, routing (sigmoid, bias, scale 2.5,
one shared expert). The reference for generation is a full forward per step without a cache.

The helpers (build, engine, prompts) are shared by the engine and server tests.
"""

import json
import os

import pytest
import torch

pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

CONFIG = os.path.join(os.path.dirname(__file__), "fixtures", "GLM-5.3-Flash-config.json")
# The indexer keeps its 32 heads: with few heads a pool's score is exactly 0 (every head's ReLU zero) often
# enough to tie at the top-k boundary, where any tie-break is a valid top-k but transformers' and ours differ
# (measured with 4 heads: 1 query in 70 at layer 3).
SMALL = dict(hidden_size=128, intermediate_size=256, moe_intermediate_size=128, num_attention_heads=4,
             num_key_value_heads=4, vocab_size=384, n_routed_experts=8, num_experts_per_tok=2,
             max_position_embeddings=1024, pad_token_id=None, eos_token_id=1, num_nextn_predict_layers=0)
VISION = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=128,
              projection_intermediate_size=64)


def text_config(layers=8, linear_heads=4, **extra):
    with open(CONFIG) as f:
        c = json.load(f)
    t = c["text_config"]
    t.pop("dtype", None)
    t.update(SMALL)
    t["linear_attn_config"] = dict(t["linear_attn_config"], num_heads=linear_heads)
    for k in ("layer_types", "mlp_layer_types", "indexer_types"):
        t[k] = t[k][:layers]
    t["num_hidden_layers"] = layers
    t.update(extra)
    return c, t


def randomize(model, seed=0):
    """Weights at scales that exercise every piece: unit-variance matmul outputs, SwiGLU inputs
    large enough that swiglu_limit clamps some of them, KDA decays from fast to slow."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".visual." in n or n.startswith("model.visual"):
                continue
            r = torch.randn(p.shape, generator=g)
            if n.endswith("A_log"):
                p.copy_(torch.empty(p.shape).uniform_(-4.0, 1.0, generator=g))
            elif n.endswith("dt_bias"):
                p.copy_(r)
            elif "norm" in n and n.endswith("weight") or n.endswith("hc_attn.scale") or n.endswith("hc_ffn.scale") \
                    or n.endswith("_hc.scale"):
                p.copy_(1 + 0.1 * r)
            elif n.endswith("bias") or n.endswith(".base"):
                p.copy_((0.5 if "correction" in n else 0.1) * r)
            elif "embed_tokens" in n:
                p.copy_(r)
            elif "kpool_compress_ape" in n:
                p.copy_(0.5 * r)
            elif "mlp" in n and ("gate_up" in n or "gate_proj" in n or "up_proj" in n):
                p.copy_(r * 3 * p.shape[-1] ** -0.5)  # SwiGLU inputs ~N(0, 9): some cross +-10
            elif "mlp" in n and "down" in n:
                p.copy_(r * p.shape[-1] ** -0.5 / 3)
            else:
                p.copy_(r * p.shape[-1] ** -0.5)
        for n, b in model.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.copy_(0.5 * torch.randn(b.shape, generator=g))


def build(path, seed=0, layers=8, linear_heads=4, **extra):
    """A random GLM-5.3-Flash-architecture model saved to `path` by transformers; returns the HF model."""
    import transformers as tf

    c, t = text_config(layers, linear_heads, **extra)
    cfg = tf.Glm5NextConfig(text_config=t, vision_config=VISION)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    model = tf.Glm5NextForConditionalGeneration(cfg)
    randomize(model, seed)
    model.eval()
    model.save_pretrained(path, safe_serialization=True)
    return model


def hf_greedy(model, ids, n):
    out = list(ids)
    with torch.no_grad():
        for _ in range(n):
            out.append(int(model(torch.tensor([out]), use_cache=False).logits[0, -1].argmax()))
    return out[len(ids):]


def engine(path, **kw):
    """A CPU engine on the model at `path`: fp32, pages of 4 tokens, prefill chunks of 8."""
    from trnrt.config import EngineConfig
    from trnrt.engine.engine import LLMEngine

    base = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256, max_num_seqs=3,
                max_model_len=256, max_prefill_tokens=8)
    base.update(kw)
    return LLMEngine(EngineConfig(**base))


def prompts(seed, lengths, vocab=384):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (n,), generator=g).tolist() for n in lengths]


@pytest.fixture(scope="module")
def dense(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm5_next")
    return str(path), build(str(path))


@pytest.fixture(scope="module")
def sparse(tmp_path_factory):
    """index_topk 16: four pools of four keys, so prompts past 16 tokens are in the sparse regime."""
    path = tmp_path_factory.mktemp("glm5_next_sparse")
    return str(path), build(str(path), seed=1, index_topk=16)


def test_config_is_parsed(dense):
    from trnrt.models.config import LinearSpec, ModelConfig
    from trnrt.models.decoder import DecoderForCausalLM

    path, hf = dense
    cfg = ModelConfig.from_pretrained(path)
    hc = hf.config.text_config
    kinds = ["kda" if isinstance(s, LinearSpec) else "dsa" for s in cfg.attn_layers]
    assert kinds == ["kda", "kda", "kda", "dsa", "kda", "kda", "kda", "dsa"]
    assert list(cfg.moe_layers) == [i for i, l in enumerate(hf.model.language_model.layers) if hasattr(l.mlp, "experts")]
    kda = cfg.attn_layers[0]
    assert (kda.num_k_heads, kda.head_k_dim, kda.lower_bound, kda.gate_rank) == (4, 128, -5.0, 128)
    m = cfg.attn_layers[3].mla
    assert (m.q_lora_rank, m.kv_lora_rank, m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim) == (1536, 512, 256, 0, 256)
    assert abs(m.softmax_scale - hf.model.language_model.layers[3].self_attn.scaling) < 1e-12
    assert m.norm_eps == hc.rms_norm_eps and (m.dsa.kpool, m.dsa.kpool_tail, m.dsa.head_dim) == (4, True, 128)
    hy = cfg.hybrid
    assert (hy.hc, hy.sinkhorn_iters, hy.swiglu_limit) == (4, 20, 10.0)
    assert cfg.routed_scaling_factor == 2.5 and cfg.n_shared_experts == 1 and cfg.router_scoring == "sigmoid"
    # The MLA cache: the latent, then the indexer key and its pool gate logits (no rope key).
    model = DecoderForCausalLM(cfg, torch.float32, 64)
    assert model.kv_shapes() == [((1, 512), (1, 256))] * 2


def test_logits_match(dense):
    """Dense regime (index_topk 2048): 70 tokens through the sequence form."""
    from trnrt.models.config import ModelConfig
    from trnrt.models.loader import load_model

    path, hf = dense
    ours = load_model(path, ModelConfig.from_pretrained(path), torch.float32, torch.device("cpu"), 256)
    ids = torch.randint(0, 384, (70,), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0]
        got = ours.forward_logits(ids)
    err = (got - want).abs().max().item()
    print(f"glm5_next dense: logits max abs diff {err:.2e} (|logits| max {want.abs().max().item():.2f})")
    assert err < 1e-4


def test_paged_chunked_greedy_matches(dense):
    """Prompts of 1 to 61 tokens prefilled in chunks of 8, four requests for three running slots,
    batched decode; the KDA state and the MLA latent cache carried across chunks."""
    from trnrt.engine.request import SamplingParams

    path, hf = dense
    eng = engine(path)
    ps = prompts(1, (5, 19, 61, 1))
    want = [hf_greedy(hf, p, 10) for p in ps]
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=10, ignore_eos=True))
    assert [r.output_ids for r in reqs] == want
    assert eng.pool.num_free + eng.radix.total_pages() == eng.pool.num_usable  # nothing leaked (the cache holds the rest)


def test_sparse_logits_and_greedy_match(sparse):
    """index_topk 16 (4 pools of 4): the pooled indexer selects per query (kernels/dsa_topk.py's selection, its
    emulation on the host), plus its tail."""
    from trnrt.engine.request import SamplingParams

    path, hf = sparse
    eng = engine(path, max_prefill_tokens=16)
    ids = torch.randint(0, 384, (70,), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0]
        got = eng.model.forward_logits(ids)
    err = (got - want).abs().max().item()
    print(f"glm5_next sparse: logits max abs diff {err:.2e}")
    assert err < 1e-4
    ps = prompts(2, (37, 70, 9))
    want = [hf_greedy(hf, p, 12) for p in ps]
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=12, ignore_eos=True))
    assert [r.output_ids for r in reqs] == want


def test_fp8_kv_keeps_a_separate_pool_key_cache(sparse):
    """Under an FP8 KV cache (the served form) the pool keys are a separate bf16 token-slot state (an FP8 row would
    round the key), bound by the engine for every DSA layer."""
    from trnrt.engine.request import SamplingParams

    path, hf = sparse
    eng = engine(path, kv_cache_dtype="fp8")
    layers = eng.model.kv_layers()
    assert layers and all(getattr(l, "pool_key", None) is not None for l in layers)
    (got,) = eng.generate(prompts(7, (41,)), SamplingParams(max_new_tokens=6, ignore_eos=True))
    assert len(got.output_ids) == 6


def test_tp2_matches_tp1(tmp_path):
    """KDA heads, MLA heads, the MLP and experts' intermediate split over two ranks; the latent, the indexer and the
    hyper-connections replicated. At attention TP 2 the DSA caches are context-parallel over the two ranks (each holds
    1 / 2 of every sequence's latent, indexer rows and pool keys), so the model takes the served form's constraints:
    hidden a multiple of 512 (the fused decode layer's tiles), pages of kpool x cp = 8 tokens, an FP8 KV cache (the
    separate pool-key cache). tp=1 runs the same model replicated."""
    from trnrt.engine.request import SamplingParams

    path = str(tmp_path)
    build(path, seed=1, index_topk=16, hidden_size=512)
    ps = prompts(3, (11, 41))
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    kw = dict(max_num_seqs=2, page_size=8, kv_cache_dtype="fp8")
    want = [r.output_ids for r in engine(path, **kw).generate(ps, sp)]
    two = engine(path, tp=2, **kw)
    try:
        assert two.model.cp == 2
        got = [r.output_ids for r in two.generate(ps, sp)]
    finally:
        two.close()
    assert got == want


def test_real_width_layers(tmp_path):
    """The first 4 layers (3 KDA + 1 DSA) at GLM-5.3-Flash's real widths: hidden 4096, 64 KDA heads, 64 MLA heads
    (and the indexer's 32, as everywhere here). The MLP and expert widths, expert count and vocabulary are cut (about
    2.4 GB per fp32 copy); they are not what this test is about."""
    from trnrt.engine.request import SamplingParams

    hf = build(str(tmp_path), seed=11, layers=4, linear_heads=64, hidden_size=4096, num_attention_heads=64,
               num_key_value_heads=64, intermediate_size=256, moe_intermediate_size=128, vocab_size=512)
    ps = prompts(8, (70, 9), vocab=512)
    with torch.no_grad():
        want_logits = hf(torch.tensor([ps[0]])).logits[0]
    want = [hf_greedy(hf, p, 6) for p in ps]
    del hf
    eng = engine(str(tmp_path), max_prefill_tokens=32, max_num_seqs=2)
    with torch.no_grad():
        got = eng.model.forward_logits(torch.tensor(ps[0]))
    err = (got - want_logits).abs().max().item()
    print(f"GLM-5.3-Flash widths: logits max abs diff {err:.2e} (|logits| max {want_logits.abs().max().item():.2f})")
    assert err < 1e-3
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=6, ignore_eos=True))
    assert [r.output_ids for r in reqs] == want
