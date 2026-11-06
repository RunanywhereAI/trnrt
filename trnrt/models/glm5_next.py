"""GLM-5.3-Flash (`glm5_next`, Glm5NextForConditionalGeneration): the text model.

45 layers: 34 Kimi Delta Attention layers (models/linear_attn.py, the "safe gate" lower bound
-5) and 11 DSA layers, every 4th from layer 3 (config layer_types). A DSA layer is MLA without
RoPE (qk_rope_head_dim 0: 64 heads of 256 nope + 256 value dims over a 512 latent, q_lora 1536)
with a lightning indexer whose keys are scored in POOLS of index_kpool = 4 consecutive tokens
(below). The residual is 4 streams mixed by manifold-constrained hyper-connections
(models/hybrid.py), the MLP a SwiGLU clamped at swiglu_limit 10, first 3 layers dense
(intermediate 12288), then 288 routed experts (sigmoid routing with a selection bias, top 8,
routed_scaling_factor 2.5) plus one shared expert. The MTP layer (layers.45) and the vision
tower are not loaded.

Numerics follow transformers v5.17.0 src/transformers/models/glm5_next/modeling_glm5_next.py
(https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/glm5_next/modeling_glm5_next.py),
config keys its configuration_glm5_next.py; the checkpoint's config.json:
https://huggingface.co/zai-org/GLM-5.3-Flash/raw/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json

The pooled indexer (Glm5NextTextIndexer.forward, get_pooled_states, append_visible_tail): each
token keeps its indexer key k (LayerNorm(wk x)) and gate logits g = index_kpool_compress_gate x.
Pool p holds tokens kpool p .. kpool p + kpool - 1 (pooling starts at the first real token,
which is position 0 here), its key is the per-channel softmax-weighted mean
sum_j softmax_j(g_j + ape_j) k_j, and it is a candidate for a query only once its LAST token is
visible. A query scores every candidate pool with the usual index (sum_h w_h relu(q_h . k_p /
sqrt(Di))), keeps the index_topk / kpool best, expands them back to their tokens, and with
index_kpool_always_select_tail also attends to the tokens of its own incomplete pool. When the
context holds at most index_topk / kpool pools every pool is kept, so below index_topk tokens the
layer is exactly dense causal MLA. Above it, the host path's bucketed form (models/mla.py
attention) applies the selection as an additive mask; the long path and the context-parallel
device path (models/dsa_long.py) attend the selected pools' tokens only.
"""

from __future__ import annotations

import torch

from .config import AttnSpec, LinearSpec, ModelConfig
from .decoder import NEG_INF
from .mla import DSASpec, MLASpec, cache_widths

ARCHITECTURES = ("Glm5NextForConditionalGeneration",)

# configuration_utils.py (v5.18.0) normalises these layer types to "indexed_attention".
_DSA_TYPES = ("deepseek_sparse_attention", "indexed_attention", "full_attention")


def _indexer_types(c: dict, n: int) -> list[str]:
    """Each layer's indexer type: explicit indexer_types, else index_topk_pattern ("F" / "S"), else "full"
    every index_topk_freq layers from index_skip_topk_offset (transformers GlmMoeDsaConfig.__post_init__'s rule)."""
    if c.get("indexer_types"):
        return list(c["indexer_types"])
    pattern = c.get("index_topk_pattern")
    if pattern is not None:
        return [{"F": "full", "S": "shared"}[x] for x in pattern] if isinstance(pattern, str) else list(pattern)
    freq, offset = max(c.get("index_topk_freq", 1), 1), c.get("index_skip_topk_offset", 2)
    return ["full" if (max(i - offset + 1, 0) % freq) == 0 else "shared" for i in range(n)]


def config_from_hf(cls, c: dict, eos_ids: tuple[int, ...]) -> ModelConfig:
    """Keys of Glm5NextTextConfig (configuration_glm5_next.py, v5.18.0); a flat text-only file
    is accepted as well as the composite one (text fields under text_config)."""
    from .hybrid import HybridSpec

    t = c.get("text_config") or c
    n = t["num_hidden_layers"]
    types = t.get("layer_types") or ["linear_attention" if i % 4 != 3 else "indexed_attention" for i in range(n)]
    mlp = t.get("mlp_layer_types") or ["dense"] * min(3, n) + ["sparse"] * (n - 3)
    if t.get("hidden_act", "silu") != "silu":
        raise NotImplementedError(f"hidden_act {t.get('hidden_act')!r}")
    if t.get("qk_rope_head_dim", 0):
        raise NotImplementedError("glm5_next DSA layers are NoPE (transformers rejects qk_rope_head_dim > 0)")
    if t.get("q_lora_rank") is None:
        raise NotImplementedError("glm5_next DSA needs q_lora_rank")
    if t.get("scoring_func", "sigmoid") != "sigmoid" or t.get("topk_method", "noaux_tc") != "noaux_tc":
        raise NotImplementedError("only sigmoid / noaux_tc routing (Glm5NextTextTopkRouter)")
    # Linear attention: linear_attn_config, with transformers' defaults (Glm5NextTextConfig:
    # linear_lower_bound -5.0, also when safe_gate is set without a bound).
    la = dict(t.get("linear_attn_config") or {})
    h, d = la.get("num_heads", t.get("linear_num_heads", 64)), la.get("head_dim", t.get("linear_head_dim", 128))
    lb = la.get("gate_lower_bound", t.get("linear_lower_bound", -5.0))
    if lb is None and la.get("safe_gate", True):
        lb = -5.0
    kda = LinearSpec("kda", h, h, d, d, la.get("short_conv_kernel_size", t.get("linear_conv_kernel_dim", 4)),
                     "sigmoid", d, float(lb) if lb is not None else None)
    # DSA layers. The indexer of every one is "full" in the checkpoint (indexer_types); IndexShare
    # over pooled selections is not implemented.
    itypes = _indexer_types(t, n)
    dn, dv = t["qk_nope_head_dim"], t["v_head_dim"]
    kpool = int(t.get("index_kpool", 16))
    if kpool < 2:
        raise NotImplementedError(f"index_kpool {kpool}: the DSA layers here are pooled (index_kpool > 1)")
    if t.get("index_topk", 2048) % kpool:
        raise ValueError("index_topk must be divisible by index_kpool")
    if not t.get("index_kpool_compress", True):
        raise NotImplementedError("index_kpool_compress=false")
    specs = []
    for i in range(n):
        if types[i] == "linear_attention":
            specs.append(kda)
            continue
        if types[i] not in _DSA_TYPES:
            raise NotImplementedError(f"layer type {types[i]!r}")
        if itypes[i] != "full":
            raise NotImplementedError("glm5_next IndexShare (indexer_types 'shared') over pooled selections")
        dsa = DSASpec(t["index_n_heads"], t["index_head_dim"], t["index_topk"], kpool=kpool,
                      kpool_tail=bool(t.get("index_kpool_always_select_tail", True)))
        # q_a_layernorm / kv_a_layernorm are built with rms_norm_eps (Glm5NextTextAttention).
        m = MLASpec(t["q_lora_rank"], t["kv_lora_rank"], dn, 0, dv, dn ** -0.5, norm_eps=t.get("rms_norm_eps", 1e-5),
                    dsa=dsa)
        kw, vw = cache_widths(m)
        specs.append(AttnSpec(t["num_attention_heads"], 1, kw, vw, 0, 10000.0, mla=m))
    experts = t.get("n_routed_experts") or 0
    # The MTP layer (num_nextn_predict_layers, model.language_model.layers.<n> in zai-org/GLM-5.3-Flash's
    # model.safetensors.index.json) is a pooled-DSA MLA layer with the MoE and NO hyper-connections (vLLM v0.30.0
    # models/glm5next/nvidia/model.py Glm5NextDecoderLayer is_mtp_layer: "MTP layers use the non-mHC path";
    # nvidia/mtp.py Glm5NextMultiTokenPredictorLayer: enorm, hnorm, eh_proj, shared_head). Its later passes run their own
    # indexer, as in the builds behind the published numbers: index_share_for_mtp_iteration (reusing the first pass's
    # pooled selection) is left unparsed, and under context-parallel DSA every bucket takes the long path, which has no
    # shared-selection form. So drafts can differ from vLLM's there (never the tokens: verification keeps the target's).
    mtp = {}
    if int(t.get("num_nextn_predict_layers") or 0):
        mtp_dsa = DSASpec(t["index_n_heads"], t["index_head_dim"], t["index_topk"], kpool=kpool,
                          kpool_tail=bool(t.get("index_kpool_always_select_tail", True)))
        mtp_m = MLASpec(t["q_lora_rank"], t["kv_lora_rank"], dn, 0, dv, dn ** -0.5,
                        norm_eps=t.get("rms_norm_eps", 1e-5), dsa=mtp_dsa)
        mtp_kw, mtp_vw = cache_widths(mtp_m)
        mtp = dict(mtp_layers=int(t["num_nextn_predict_layers"]),
                   mtp_spec=AttnSpec(t["num_attention_heads"], 1, mtp_kw, mtp_vw, 0, 10000.0, mla=mtp_m),
                   mtp_prefix=f"model.layers.{n}", mtp_moe=bool(experts))
    hy = HybridSpec("glm5_next", hc=int(t.get("hc_mult", 4)), sinkhorn_iters=int(t.get("hc_sinkhorn_iters", 20)),
                    hc_eps=float(t.get("hc_eps", 1e-6)), swiglu_limit=t.get("swiglu_limit"), block_norms=True)
    first = next((s for s in specs if not isinstance(s, LinearSpec)), None)
    eos = eos_ids or _ids(t.get("eos_token_id"))
    return cls(
        architecture=c["architectures"][0], vocab_size=t["vocab_size"], hidden_size=t["hidden_size"],
        intermediate_size=t["intermediate_size"], num_layers=n, num_heads=t["num_attention_heads"], num_kv_heads=1,
        head_dim=first.head_dim if first is not None else d, rms_norm_eps=t.get("rms_norm_eps", 1e-5),
        rope_theta=10000.0, max_position_embeddings=t.get("max_position_embeddings", 1048576),
        tie_word_embeddings=bool(t.get("tie_word_embeddings", False)), eos_token_ids=eos,
        num_experts=experts, num_experts_per_tok=t.get("num_experts_per_tok", 0) or 0,
        moe_intermediate_size=t.get("moe_intermediate_size", 0) or 0, norm_topk_prob=t.get("norm_topk_prob", True),
        moe_layers=tuple(i for i in range(n) if experts and mlp[i] == "sparse"), attn_layers=tuple(specs),
        router_scoring="sigmoid", router_bias=bool(experts), routed_scaling_factor=float(t.get("routed_scaling_factor") or 1.0),
        n_shared_experts=(t.get("n_shared_experts") or 0) if experts else 0, n_group=t.get("n_group") or 1,
        topk_group=t.get("topk_group") or 1, hybrid=hy, **mtp, **cls._quant(c if "quantization_config" in c else t))


def _ids(e) -> tuple[int, ...]:
    return tuple(e) if isinstance(e, list) else ((e,) if e is not None else ())


def pool_keys(layer, d: DSASpec, rows: torch.Tensor) -> torch.Tensor:
    """Each pool's key [..., Di] from its kpool tokens' indexer rows [..., kpool, 2 Di] (key, then
    gate logits): the per-channel softmax over the pool of gate + ape, weighting the keys
    (get_pooled_states). The same arithmetic wherever a pool key is made: from the gathered context
    (models/mla.py context_pool_keys, without a pool-key cache) or once per written token into the
    pool-key cache (models/mla.py write_pool_keys and the context-parallel writes)."""
    Di = d.head_dim
    k = rows[..., :Di]
    logits = rows[..., Di : 2 * Di].float() + layer.idx_pool_ape.float()
    return (torch.softmax(logits, dim=-2).to(k.dtype) * k).sum(dim=-2)


def pooled_selection(layer, d: DSASpec, q, w: torch.Tensor, kI: torch.Tensor | None, vis: torch.Tensor,
                     pk: torch.Tensor | None = None) -> torch.Tensor:
    """Additive mask [B, Q, L] (0 = selected, NEG_INF = not) of the pooled indexer for B x Q
    queries over L keys: q = (None, [B, Q, Hi, Di]), w [B, Q, Hi] fp32, kI [B, L, 2 Di] each token's
    key then gate logits, or pk [B, L / kpool, Di] the pool keys themselves (read from the pool-key
    cache, models/mla.py), vis [B, Q, L] the causal visibility (0 / NEG_INF, a prefix of the context
    for every query). Glm5NextTextIndexer.forward, with get_pooled_states and append_visible_tail, as
    one batched computation: no per-query loop and no comparison against a float literal (those can
    lower to f64 in neuronx-cc)."""
    kp, Di = d.kpool, d.head_dim
    B, Q, L = vis.shape
    pad = -L % kp  # the sequence form (forward_logits): the last pool may be incomplete
    if pad:
        vis = torch.cat([vis, torch.full((B, Q, pad), NEG_INF, dtype=vis.dtype, device=vis.device)], dim=-1)
    if pk is None:
        if pad:
            kI = torch.cat([kI, kI.new_zeros(B, pad, kI.shape[-1])], dim=1)
        P = (L + pad) // kp
        pk = pool_keys(layer, d, kI[..., : 2 * Di].reshape(B, P, kp, 2 * Di))  # [B, P, Di]
    index = pool_index(d, q, w, pk)
    return block_mask(index, vis, kp, d.topk // kp, d.kpool_tail)[..., :L]


def pool_index(d: DSASpec, q, w: torch.Tensor, pk: torch.Tensor) -> torch.Tensor:
    """The pooled indexer's scores [B, Q, P] of every pool: sum_h w_h relu(q_h . k_p / sqrt(Di)) (q = (None,
    [B, Q, Hi, Di]), w [B, Q, Hi] fp32, pk [B, P, Di] the pool keys)."""
    _, q_p = q
    s = torch.einsum("bqhd,bpd->bqhp", q_p.float(), pk.float())
    return torch.einsum("bqh,bqhp->bqp", w, torch.relu(s * d.head_dim ** -0.5))


def block_mask(index: torch.Tensor, vis: torch.Tensor, kp: int, keep: int, tail: bool) -> torch.Tensor:
    """Additive mask [B, Q, L] keeping, per query, the `keep` best-scored blocks of kp tokens
    among those whose last token it can see (index [B, Q, L / kp] the block scores, vis [B, Q, L]
    0 / NEG_INF with L a multiple of kp), expanded to their tokens, plus (tail) its own incomplete
    block. kernels/dsa_topk.py: the keep best of the visible candidates, the lowest indices among
    those tied with the keep-th best (all visible ones if fewer), exact by construction on the host
    and on the device; selection, expansion to tokens and tail are one NKI kernel on the device (its
    emulation on the host). (transformers' own choice among exact ties is torch.topk's on its own
    layout, which none reproduces: CPU torch.topk is not lowest-index-first among ties.)"""
    B, Q, L = vis.shape
    P = L // kp
    cand = vis.reshape(B, Q, P, kp)[..., kp - 1]  # a block is a candidate once its last token is visible
    from ..kernels import dsa_topk

    return dsa_topk.select(index + cand, keep, vis_only=True, kp=kp, tail=tail)
