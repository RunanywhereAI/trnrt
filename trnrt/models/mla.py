"""GLM-5.3-Flash's DSA layers: multi-head latent attention (MLA) without RoPE, with a pooled
lightning indexer, as a per-layer attention kind of DecoderForCausalLM.

MLA, per layer:

    q      = q_b(rmsnorm(q_a(x)))                                    [T, H, dn]
    c      = rmsnorm(kv_a(x))                                        latent [T, r]
    k_h    = W_UK[h] c,  v_h = W_UV[h] c       (kv_b_proj rows of head h: dn rows of W_UK, dv of W_UV)

The paged cache holds ONLY the latent c per token (kv_lora_rank values), against 2 * H * head_dim
for MHA. In the cache's terms an MLA layer has one "KV head" whose K is the latent (k_cache
[slots, 1, r]) and whose V is the indexer row (v_cache [slots, 1, 2 Di]: the indexer key, then
its pool gate logits). Decode attends over the latent directly by weight absorption (q_lat =
W_UK^T q_nope; out = W_UV (p @ c)), so the per-head keys and values are never formed; a prefill
chunk on the host decompresses its context.

Tensor parallelism is SGLang v0.5.21's and vLLM v0.30.0's layout (srt/models/deepseek_v2.py
DeepseekV2AttentionMLA; vllm/model_executor/models/deepseek_v2.py DeepseekV2MLAAttention):
q_a_proj and kv_a_proj_with_mqa are ReplicatedLinear, q_b_proj and kv_b_proj are
ColumnParallelLinear over heads, o_proj is RowParallelLinear; so every rank computes the same
latent, and only the heads are split, over the attention TP.

The indexer scores pools of index_kpool consecutive tokens (models/glm5_next.py): index[t, p] =
sum_h w[t, h] relu(q_I[t, h] . k_P[p] / sqrt(Di)), with q_I = wq_b(q latent), each pool key k_P
the softmax-gated mean of its tokens' keys LayerNorm(wk(x)), and w = weights_proj(x) / sqrt(Hi);
the query then attends only to the tokens of its index_topk / kpool best pools plus its own
incomplete pool. Numerics follow transformers v5.17.0 (models/glm5_next/modeling_glm5_next.py):
scores in fp32 and no Hadamard rotation or FP8 activation quantisation of q_I / k_I.

On the device every DSA layer is context-parallel (models/dsa_long.py "context parallelism"):
each rank of a CP group holds 1 / cp of every sequence's latent, indexer rows and pool keys. A
decode batch runs attention_cp, a prefill chunk attention_cp_rows (the attention group split into
row groups of cp ranks). The host path at attention TP 1 (attention below) is the CPU reference:
a bucket of at most dsa_long.LONG_KEYS keys gathers its context and selects with an additive mask,
a longer one takes the long path (attention_long), which never forms anything of the context's
size per query. When the context holds at most index_topk / kpool pools every pool is selected,
so there the layer is exactly dense causal MLA.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..kernels import shell as SH
from . import dsa_long as _dsa_long
from .config import AttnSpec, ModelConfig
from .decoder import NEG_INF, rms_norm
from .quant import dequant

# The pool-key cache (pool_cache_mode): a pooled indexer keeps every pool's key, written when the
# pool's tokens are written and read by the selection, instead of rebuilding all L / kpool pool keys
# from their tokens' keys and gates in every call. Under an FP8 KV cache (the served one) it is
# "separate": kpool pieces in a bf16 token-slot state beside K and V (pool_key_width; 704 bytes per
# token per rank, 8% fewer KV pages in FP8); an FP8 cached row would round the key to e4m3. A bf16 /
# fp32 KV cache (the host tests) keeps it "inplace": the key replaces the indexer key in the cached
# row of the pool's LAST token, no extra bytes (write_pool_keys_inplace).

# A context-parallel prefill chunk's slot attention sizes each row's buffer by its real local share
# (_cp_attend_classes): CP_SLOTS_SMALL slots for the rows that fit (keep / cp + the tail is ~65 at
# cp 8) and the full 640 for the rest, each buffer's kernel call running its live rows only. Exact
# for any split.
CP_SLOTS_SMALL = 128
# attention_cp_rows' local selection keeps each rank's top CP_LOCAL_K + 8 pools instead of keep (the
# kernel extracts in rounds of 8), the merge runs over each rank's top CP_LOCAL_K, and a row's merge
# is exact unless a CERTIFICATE fails: some rank's (K + 1)-th value is valid and at or above the
# merged keep-th value, or the merge selects fewer than keep while some rank dropped a valid pool
# (every pool not in the union is at or below its rank's (K + 1)-th value, so with the certificate it
# is strictly below the threshold, ties included). Rows that fail it, up to CP_LOCAL_F per chunk and
# row group, are selected again with the whole keep list on every rank, merged and attended from
# that, so the chunk is exact; rows past CP_LOCAL_F are not, and each layer counts them (its
# cp_localk buffer: [rows, failed, past F]). On real text at 128K and 1M contexts, K = 120 + 8 failed
# no row of over 3.2M; K = 96 failed 63. Must be a multiple of 8 with cp K >= keep; 0 or K >= keep
# turns it off (_local_k).
CP_LOCAL_K = 120
CP_LOCAL_F = 128
VISIBLE_SCORE = -5e29  # kernels/dsa_topk.VISIBLE: scores at or below it are not candidates
# A context-parallel layer's FP8 w_a (q_a + kv_a), q_b and o_proj are also kept as model-dtype copies
# (w_a_dec, q_b_dec, o_dec: dequant of the stored FP8 and scales at load, so the same values), which
# attention_cp's decode branch reads instead of dequantizing every weight in every decode graph (0.40
# ms of a 4.1 ms DSA layer, measured on trn1). Prefill keeps the FP8 weights. +22.5 MB per DSA layer
# per rank at tp 32, +0.23 GiB over the 11 layers.
DECODE_BF16_NAMES = ("w_a", "q_b", "o")
# The decode shell's selection (kernels/dsa_shell.py select2): its page groups gathered ahead, its
# page tiles in flight, and its compaction with each program's rows stacked up to 4 a tile.
SEL2_PF = 5
SEL2_NR = 6
SEL2_CS = 1
# A decode step of at most SHELL_MAXB rows merges and attends in the DSA shell's attend call
# (kernels/dsa_shell.py attend); a larger one merges in dsa_long.cp_merge. A decode step of at most
# LAYER_MK_MAXB rows runs the fused layer calls (kernels/dsa_sat.py, kernels/dsa_satwf.py).
# LAYER_MK_MAXB is the engine's --dsa-layer-max-rows (EngineConfig.dsa_layer_max_rows, set on every rank by
# engine.build_shard before it builds the shard; 8 here and under --preset batch8, 16 served).
SHELL_MAXB = 6
LAYER_MK_MAXB = 8
# _cp_scores' kernel scores (kernels/dsa_index.py) of a call of more than CP_INDEX_ROWS rows run as one kernel call
# per CP_INDEX_ROWS rows, each with its rows' weights, page groups and candidates; 0 (batch8's): one call. Rows are
# independent in the kernel, so the scores are the same. Each call builds the kernel's query layout [D, rows Hi] by an
# XLA transpose of [rows Hi, D], which neuronx-cc lowers as an ordinary transpose at 8 rows but at 16 as a DMA transpose
# of 16,384 16-byte descriptors per DSA layer. The engine's --dsa-index-rows (EngineConfig.dsa_index_rows, set by
# build_shard; 8 served). Only graphs whose calls reach _cp_scores with more rows move.
CP_INDEX_ROWS = 0


def _shell_rows_on(rows: int) -> bool:
    """Whether a decode step of `rows` rows takes the shell's merge / attend call."""
    return rows <= SHELL_MAXB


def _mk_rows_on(rows: int) -> bool:
    """Whether a decode step of `rows` rows takes the fused layer call."""
    return rows <= LAYER_MK_MAXB


def _cp_scores_rows(q, w, pkc, table, cand):
    """CP_INDEX_ROWS: dsa_index.scores of q [T, Hi, D] in calls of CP_INDEX_ROWS rows (each with its rows' weights,
    page groups and candidates), concatenated: [T, Pl] fp32."""
    from ..kernels import dsa_index

    n = CP_INDEX_ROWS
    return torch.cat([dsa_index.scores(q[i : i + n], w[i : i + n], pkc, dsa_index.page_groups(table[i : i + n]),
                                       cand[i : i + n]) for i in range(0, q.shape[0], n)])


def pool_cache_mode(kv_fp8: bool) -> str:
    """The pool-key cache form for a KV cache stored in FP8 (kv_fp8) or not."""
    return "separate" if kv_fp8 else "inplace"


@dataclass(frozen=True)
class DSASpec:
    """One layer's pooled DSA indexer (models/glm5_next.py): it scores pools of `kpool` (> 1) consecutive
    keys and selects topk / kpool of them (plus, with kpool_tail, the current incomplete pool)."""

    n_heads: int  # index_n_heads
    head_dim: int  # index_head_dim
    topk: int  # index_topk
    kpool: int  # index_kpool
    kpool_tail: bool  # index_kpool_always_select_tail


@dataclass(frozen=True)
class MLASpec:
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int  # 0: NoPE
    v_head_dim: int
    softmax_scale: float
    norm_eps: float = 1e-6  # q_a_layernorm / kv_a_layernorm
    dsa: DSASpec | None = None


def _pooled(spec) -> bool:
    m = getattr(spec, "mla", None)
    d = m.dsa if m is not None else None
    return d is not None and d.kpool > 1


def pool_key_width(spec, kv_fp8: bool = False) -> int:
    """Per-token width of the separate pool-key cache of an attention layer (0: none, also in the
    inplace form, pool_cache_mode). A pooled indexer's pool key (head_dim values) is stored as kpool
    pieces of head_dim / kpool, piece i in the slot of the pool's i-th token, so the cache is paged
    with the tokens (prefix sharing, the host tier and the page gathers need nothing else) and a page
    gather of the context reads [L, head_dim / kpool] = [L / kpool, head_dim], the pool keys in order.
    Kept in the model dtype even under an FP8 KV cache (a token-slot state,
    DecoderForCausalLM.token_state_shapes): the pool keys are computed in bf16 from the cached rows
    exactly as the per-call form computes them."""
    if pool_cache_mode(kv_fp8) != "separate" or not _pooled(spec):
        return 0
    d = spec.mla.dsa
    if d.head_dim % d.kpool:
        raise ValueError(f"index_head_dim {d.head_dim} is not divisible by index_kpool {d.kpool}")
    return d.head_dim // d.kpool


def cache_widths(m: MLASpec) -> tuple[int, int]:
    """(K, V) widths of the paged cache: the latent, and the indexer key with each token's
    pool-compression gate logits beside it."""
    return m.kv_lora_rank, m.qk_rope_head_dim + 2 * m.dsa.head_dim


# -- parameters ---------------------------------------------------------------------


def init_layer(layer, cfg: ModelConfig, spec: AttnSpec, p, lin, tp: int) -> None:
    """MLA attention parameters of one attention rank of `tp` (the attention TP,
    DecoderForCausalLM; DecoderLayer.__init__)."""
    m = spec.mla
    H, nh = cfg.hidden_size, spec.num_heads // tp
    dn, dr, dv, r = m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim, m.kv_lora_rank
    layer.nh, layer.nkv, layer.kv_offset = nh, 1, 0
    # q_a_proj and kv_a_proj_with_mqa read the same input and are replicated: one matmul (vLLM's
    # fused_qkv_a_proj).
    lin("w_a", "self_attn.q_a_proj", m.q_lora_rank + r + dr, H)
    layer.q_a_norm = p(m.q_lora_rank)
    lin("q_b", "self_attn.q_b_proj", nh * (dn + dr), m.q_lora_rank)
    layer.kv_a_norm = p(r)
    lin("w_uk", "self_attn.kv_b_proj", nh * dn, r)  # W_UK of this rank's heads, [nh * dn, r]
    lin("w_uv", "self_attn.kv_b_proj", nh * dv, r)  # W_UV, [nh * dv, r]
    lin("o", "self_attn.o_proj", H, nh * dv)
    if _dsa_long.cp_enabled() and tp > 1 and long_capable(spec):  # attention_cp's decode copies (DECODE_BF16_NAMES)
        dec = [n for n in DECODE_BF16_NAMES if getattr(layer, n + "_scale", None) is not None]
        for n in dec:
            setattr(layer, n + "_dec", p(*getattr(layer, n).shape))
        # kept out of the layer graphs' explicit tensors (decoder._layer_split: every prefill piece); the
        # whole-step decode traces the module and reads them
        layer._decode_only = getattr(layer, "_decode_only", ()) + tuple(n + "_dec" for n in dec)
        # the fused layer calls' pre-tiled decode weights (filled in load_layer)
        if H % 512 or r % 128 or (nh * dv) % 128:
            raise ValueError(f"context-parallel DSA (attention TP {tp} > 1) needs hidden_size a multiple of 512 and "
                             f"kv_lora_rank and the rank's value width multiples of 128 (the fused decode layer's "
                             f"pre-tiled o_proj and W_UV); have hidden {H}, kv_lora {r}, {nh} heads x {dv}")
        layer.w_uv_t = p(128, nh * r // 128, dv)
        layer.o_t = p(128, H // 512 * (nh * dv // 128), 512)
        layer._decode_only = layer._decode_only + ("w_uv_t", "o_t")
    rows_cp = _dsa_long.cp_enabled() and tp > 1 and _dsa_long.cp_degree(tp) < tp  # CP row groups (attention_cp_rows)
    if rows_cp and long_capable(spec):
        # every head: a prefill chunk's row group attends its own rows with all of them (attention_cp_rows)
        layer.qshard = True
        Hh = spec.num_heads
        lin("q_b_all", "self_attn.q_b_proj", Hh * (dn + dr), m.q_lora_rank)
        lin("w_uk_all", "self_attn.kv_b_proj", Hh * dn, r)
        lin("w_uv_all", "self_attn.kv_b_proj", Hh * dv, r)
        lin("o_all", "self_attn.o_proj", H, Hh * dv)
    d = m.dsa
    lin("idx_wq", "self_attn.indexer.wq_b", d.n_heads * (d.head_dim - dr), m.q_lora_rank)
    lin("idx_wk", "self_attn.indexer.wk", d.head_dim, H)
    layer.idx_knorm_w, layer.idx_knorm_b = p(d.head_dim), p(d.head_dim)
    # transformers keeps weights_proj in fp32 (_keep_in_fp32_modules)
    layer.idx_wproj = nn.Parameter(torch.empty(d.n_heads, H, dtype=torch.float32), requires_grad=False)
    # Glm5NextTextIndexer.index_kpool_compress_ape / _gate
    layer.idx_pool_ape, layer.idx_pool_gate = p(d.kpool, d.head_dim), p(d.head_dim, H)
    layer.dsa_topk = layer.dsa_mask = None  # the host path's selection scratch (bind_scratch)
    layer.pool_key = None  # [slots, head_dim / kpool] pool-key pieces, bound by the runner (pool_key_width)
    layer.pool_inplace = False  # pool keys in the pool's last cached row (bind_scratch, pool_cache_mode)
    layer.dsa_stage = False  # the host path's selection goes through the scratch (bind_scratch)


def bind_scratch(model, rows: int, device, keys: int | None = None) -> None:
    """The selection scratch of the DSA layers: rows query rows (the most one call carries) by keys
    (the longest context a bucket reads). On the host path a layer writes its selection there and
    attends with what it reads back (layer.dsa_stage, attention below)."""
    layers = [l for l in model.kv_layers() if l.spec.mla is not None and l.spec.mla.dsa is not None]
    if not layers:
        return
    if keys and all(long_capable(l.spec) for l in layers):
        # Buckets past dsa_long.LONG_KEYS run the long path, which stages nothing (attention_long).
        keys = min(keys, max(_dsa_long.LONG_KEYS, 1))
    # int32 with explicit casts on both sides: an int64 topk index written into an int64 buffer
    # failed in LNL's tracer with "Check failed: self.scalar_type() == values.scalar_type()".
    top = torch.zeros(max(rows, 1), layers[0].spec.mla.dsa.topk, dtype=torch.int32, device=device)
    mask = torch.zeros(max(rows, 2), keys, dtype=torch.bfloat16, device=device) if keys else None
    inplace = pool_cache_mode(getattr(model, "fp8_max", None) is not None) == "inplace"
    for l in layers:
        l.pool_inplace = inplace and _pooled(l.spec)
        l.dsa_topk, l.dsa_mask = top, mask
        l.dsa_stage = mask is not None


def load_layer(layer, ck, pre: str, cfg: ModelConfig, rank: int, dtype, assign, concat, put) -> None:
    """Fill an MLA layer's attention parameters from the checkpoint (loader._load_decoder), as
    attention rank `rank` (DecoderForCausalLM.attn_rank). assign(param, scale_param, w, s) and
    concat(parts) are the loader's _assign / _concat."""
    m = layer.spec.mla
    a = pre + "self_attn."
    blk = cfg.quant_block
    nh, dn, dr, dv, r = layer.nh, m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim, m.kv_lora_rank
    parts = [ck.linear(a + "q_a_proj", block=blk), ck.linear(a + "kv_a_proj_with_mqa", block=blk)]
    assign(layer.w_a, layer.w_a_scale, *concat(parts))
    put(layer.q_a_norm, a + "q_a_layernorm.weight")
    put(layer.kv_a_norm, a + "kv_a_layernorm.weight")
    qd = dn + dr
    w, s = ck.linear(a + "q_b_proj", slice(rank * nh * qd, (rank + 1) * nh * qd), block=blk)
    assign(layer.q_b, layer.q_b_scale, w, s)
    kd = dn + dv
    w, s = ck.linear(a + "kv_b_proj", slice(rank * nh * kd, (rank + 1) * nh * kd), block=blk)
    w3 = w.view(nh, kd, -1)
    s3 = s.view(nh, kd, -1) if s is not None else None
    assign(layer.w_uk, layer.w_uk_scale, w3[:, :dn].reshape(nh * dn, -1),
           s3[:, :dn].reshape(nh * dn, -1) if s3 is not None else None)
    assign(layer.w_uv, layer.w_uv_scale, w3[:, dn:].reshape(nh * dv, -1),
           s3[:, dn:].reshape(nh * dv, -1) if s3 is not None else None)
    w, s = ck.linear(a + "o_proj", cols=slice(rank * nh * dv, (rank + 1) * nh * dv), block=blk)
    assign(layer.o, layer.o_scale, w, s)
    for n in DECODE_BF16_NAMES:  # the decode copies: the stored FP8 and scales dequantized once
        if hasattr(layer, n + "_dec"):
            getattr(layer, n + "_dec").data.copy_(dequant(getattr(layer, n).data, getattr(layer, n + "_scale").data, dtype))
    if hasattr(layer, "w_uv_t"):  # the fused layer calls: the same values the einsum / F.linear read, re-laid once
        from ..kernels import dsa_sat

        sc = getattr(layer, "w_uv_scale", None)
        wuv = dequant(layer.w_uv.data, sc.data if sc is not None else None, dtype)
        layer.w_uv_t.data.copy_(dsa_sat.tile_wuv(wuv.view(nh, dv, -1)))
        od = layer.o_dec.data if hasattr(layer, "o_dec") else dequant(
            layer.o.data, layer.o_scale.data if getattr(layer, "o_scale", None) is not None else None, dtype)
        layer.o_t.data.copy_(dsa_sat.tile_o(od))
    if getattr(layer, "qshard", False):  # every head's copies (attention_cp_rows)
        Hh = layer.spec.num_heads
        w, s = ck.linear(a + "q_b_proj", block=blk)
        assign(layer.q_b_all, layer.q_b_all_scale, w, s)
        w, s = ck.linear(a + "kv_b_proj", block=blk)
        w3 = w.view(Hh, kd, -1)
        s3 = s.view(Hh, kd, -1) if s is not None else None
        assign(layer.w_uk_all, layer.w_uk_all_scale, w3[:, :dn].reshape(Hh * dn, -1),
               s3[:, :dn].reshape(Hh * dn, -1) if s3 is not None else None)
        assign(layer.w_uv_all, layer.w_uv_all_scale, w3[:, dn:].reshape(Hh * dv, -1),
               s3[:, dn:].reshape(Hh * dv, -1) if s3 is not None else None)
        w, s = ck.linear(a + "o_proj", block=blk)
        assign(layer.o_all, layer.o_all_scale, w, s)
    # Tensor names: zai-org/GLM-5.3-Flash model.safetensors.index.json (self_attn.indexer.wq_b, .wk,
    # .k_norm.weight / .bias, .weights_proj, .index_kpool_compress_ape / _gate).
    w, s = ck.linear(a + "indexer.wq_b", block=blk)
    assign(layer.idx_wq, layer.idx_wq_scale, w, s)
    w, s = ck.linear(a + "indexer.wk", block=blk)
    assign(layer.idx_wk, layer.idx_wk_scale, w, s)
    layer.idx_knorm_w.data.copy_(ck.get(a + "indexer.k_norm.weight").to(dtype))
    layer.idx_knorm_b.data.copy_(ck.get(a + "indexer.k_norm.bias").to(dtype))
    w, s = ck.linear(a + "indexer.weights_proj", block=blk)
    layer.idx_wproj.data.copy_(w.float() if s is None else dequant(w, s, torch.float32))
    layer.idx_pool_ape.data.copy_(ck.get(a + "indexer.index_kpool_compress_ape").to(dtype))
    layer.idx_pool_gate.data.copy_(ck.get(a + "indexer.index_kpool_compress_gate").to(dtype))


# -- forward ------------------------------------------------------------------------


def _wd(model, layer, name: str, dec: bool) -> torch.Tensor:
    """model._w(layer, name), or in a decode call its model-dtype copy when the layer keeps one."""
    w = getattr(layer, name + "_dec", None) if dec else None
    return model._w(layer, name) if w is None else w


def _project(model, layer, x, positions, dec: bool = False):
    """q_nope [T, nh, dn], q_pe [T, nh, 0], latent c [T, r], k_pe [T, 0] and the normalised q latent
    (the indexer's input). dec: attention_cp's decode branch (the model-dtype weight copies)."""
    m = layer.spec.mla
    T = x.shape[0]
    dn, dr, r = m.qk_nope_head_dim, m.qk_rope_head_dim, m.kv_lora_rank
    a = F.linear(x, _wd(model, layer, "w_a", dec))
    # Explicit slices, not torch.split (LNL miscompiles an unequal split).
    ql = m.q_lora_rank
    q_resid = rms_norm(a[:, :ql], layer.q_a_norm, m.norm_eps)
    kv = a[:, ql:]
    q = F.linear(q_resid, _wd(model, layer, "q_b", dec))
    q = q.view(T, layer.nh, dn + dr)
    q_nope, q_pe = q[..., :dn], q[..., dn:]
    c = rms_norm(kv[:, :r], layer.kv_a_norm, m.norm_eps)
    k_pe = kv[:, r:]
    return q_nope, q_pe, c, k_pe, q_resid


def _indexer(model, layer, x, q_resid, positions):
    """The lightning indexer's per-token query (None, [T, Hi, Di]), key row [T, 2 Di] (the key, then
    the token's pool gate logits) and head weights [T, Hi] (Glm5NextTextIndexer.forward before
    scoring: gate_scores = index_kpool_compress_gate(x))."""
    d = layer.spec.mla.dsa
    T = x.shape[0]
    q_p = F.linear(q_resid, model._w(layer, "idx_wq")).view(T, d.n_heads, d.head_dim)
    k = F.linear(x, model._w(layer, "idx_wk"))
    k = F.layer_norm(k.float(), (d.head_dim,), layer.idx_knorm_w.float(), layer.idx_knorm_b.float(), 1e-6).to(x.dtype)
    w = F.linear(x.float(), layer.idx_wproj) * d.n_heads ** -0.5
    k = torch.cat([k, F.linear(x, layer.idx_pool_gate)], dim=-1)
    return (None, q_p), k, w


def _layer_selection(layer, d: DSASpec, index, vis, B: int, Q: int):
    """The additive selection mask of a sparse bucket (models/glm5_next.py pooled_selection): index
    (q, w, kI[, pk]) as _core takes it (pk: the pool keys from the cache)."""
    (q_r, q_p), wi, kI, *pk = index
    q4 = (None, q_p.reshape(B, Q, d.n_heads, -1))
    from .glm5_next import pooled_selection

    return pooled_selection(layer, d, q4, wi.view(B, Q, d.n_heads), kI, vis, pk[0] if pk else None)


def _core(model, layer, q_nope, kc, vis, B: int, Q: int, top=None, index=None, absorb=True):
    """The host path's attention of B x Q queries over L latents each: kc [B, L, r], vis [B, Q, L]
    additive visibility. index: (q, w, kI[, pk]) to select with in a sparse bucket (L > index_topk);
    top: a selection already made (the staged one, attention below), an additive mask. absorb: attend
    over the latent (decode), else decompress the context into per-head keys and values (a chunk,
    reference). Returns the o_proj input [B * Q, nh * dv]."""
    m = layer.spec.mla
    d = m.dsa
    nh, dn, dv, r = layer.nh, m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
    L = kc.shape[1]
    if L > d.topk and index is not None and top is None:  # static per bucket: below it every visible pool is kept
        top = _layer_selection(layer, d, index, vis, B, Q)
    if top is not None:
        vis = vis + top
    w_uk = model._w(layer, "w_uk").view(nh, dn, r)
    w_uv = model._w(layer, "w_uv").view(nh, dv, r)
    qn = q_nope.reshape(B, Q, nh, dn)
    if not absorb:  # decompress the context into per-head keys and values
        k_nope = torch.einsum("blr,hdr->blhd", kc, w_uk)
        v = torch.einsum("blr,hvr->blhv", kc, w_uv)
        s = torch.einsum("bqhd,blhd->bhql", qn, k_nope)
        p = torch.softmax(s.float() * m.softmax_scale + vis.unsqueeze(1), dim=-1).to(model.dtype)
        o = torch.einsum("bhql,blhv->bqhv", p, v)
    else:  # weight absorption: attend over the latent itself
        q_lat = torch.einsum("bqhd,hdr->bqhr", qn, w_uk)
        s = torch.einsum("bqhr,blr->bhql", q_lat, kc)
        p = torch.softmax(s.float() * m.softmax_scale + vis.unsqueeze(1), dim=-1).to(model.dtype)
        o = torch.einsum("bqhr,hvr->bqhv", torch.einsum("bhql,blr->bqhr", p, kc), w_uv)
    return o.reshape(B * Q, nh * dv)


def attention(model, layer, x, positions, slot_mapping, table, bias, mixed=None, ffn=None):
    """The attention block of a DSA layer over the paged cache, for a decode batch (table [B, P],
    positions [B]) or a prefill chunk (table [P], positions [C]); x is already normalised. Returns
    the o_proj output before the tensor-parallel all-reduce, or (the fused DSA layer with its FFN
    half, ffn given) the FFN half's (S_n, post, comb, y).

    Context-parallel layers (the device) run attention_cp / attention_cp_rows. At attention TP 1 (the
    host reference) a bucket of at most dsa_long.LONG_KEYS keys gathers its context: the latent, the
    indexer rows and the pool keys, the selection through the scratch as an additive mask; a longer
    one takes attention_long."""
    if mixed is not None:
        raise NotImplementedError("mixed prefill + decode batches")
    if getattr(model, "cp", 1) > 1 and long_capable(layer.spec):
        if getattr(model, "cp_rows", 1) > 1 and table.dim() == 1:  # a chunk over context-parallel row groups
            return attention_cp_rows(model, layer, x, positions, slot_mapping, table)
        return attention_cp(model, layer, x, positions, slot_mapping, table, ffn=ffn)
    m = layer.spec.mla
    d = m.dsa
    T = x.shape[0]
    q_nope, q_pe, c, k_pe, q_resid = _project(model, layer, x, positions)
    qi, ki, wi = _indexer(model, layer, x, q_resid, positions)
    vrow = ki  # NoPE: no zero-width cat with k_pe
    model._store(layer.k_cache, slot_mapping, c.unsqueeze(1))
    model._store(layer.v_cache, slot_mapping, vrow.unsqueeze(1))
    pooled = layer.pool_key is not None
    if pooled:
        write_pool_keys(model, layer, d, positions, table, slot_mapping)
    elif layer.pool_inplace:
        write_pool_keys_inplace(model, layer, d, positions, table, slot_mapping, vrow)
    if long_capable(layer.spec) and _dsa_long.enabled(table.shape[-1] * model.page_size):
        return attention_long(model, layer, q_nope, qi, wi, positions, table)
    kc = model._load(layer.k_cache, table).squeeze(-2)  # [(B,) L, r]
    kv = model._load(layer.v_cache, table).squeeze(-2)  # [(B,) L, 2 Di]
    pk = model._gather(layer.pool_key.unsqueeze(1), table).squeeze(-2) if pooled else None  # [(B,) L, Di / kp]
    if table.dim() == 1:  # chunk: one sequence, C queries
        B, Q = 1, T
        kc, kv = kc.unsqueeze(0), kv.unsqueeze(0)
        pk = pk.unsqueeze(0) if pooled else None
    else:  # decode
        B, Q = T, 1
    L = kc.shape[1]
    vis = bias.reshape(B, Q, L)
    index = (qi, wi, kv[..., : 2 * d.head_dim])
    if pooled:  # the pool keys in order: [B, L / kp, Di]
        index = (*index, pk.reshape(B, L // d.kpool, d.head_dim))
    elif layer.pool_inplace:  # the key part of each pool's last row: [B, L / kp, Di]
        index = (*index, _inplace_keys(model, layer, d, table).reshape(B, L // d.kpool, d.head_dim))
    top = None
    if layer.dsa_stage and L > d.topk and T <= layer.dsa_mask.shape[0] and L <= layer.dsa_mask.shape[1]:
        # Select, write it to the scratch and attend with what is read back: the selection is
        # materialised once. Attending with the selection itself made a decode layer 10.0 ms at B=8
        # over 4K keys against 4.4 ms this way (trn1).
        _share(layer, _layer_selection(layer, d, index, vis, B, Q), T)
        top = _shared(layer, T, L).view(B, Q, -1)
    out = _core(model, layer, q_nope, kc, vis, B, Q, top, index, absorb=table.dim() != 1)
    return F.linear(out, model._w(layer, "o"))


def long_capable(spec) -> bool:
    """Whether an attention layer can run the long-context path (models/dsa_long.py): a pooled DSA indexer
    layer without RoPE that attends its own incomplete pool."""
    return _pooled(spec) and not spec.mla.qk_rope_head_dim and spec.mla.dsa.kpool_tail


def context_pool_keys(model, layer, d: DSASpec, table: torch.Tensor) -> torch.Tensor:
    """[(B,) P, Di]: the pool keys of the context a block table addresses, P = L / kpool, from the pool-key cache
    (separate or in place) or, without one, rebuilt from the cached indexer rows."""
    kp, Di = d.kpool, d.head_dim
    if layer.pool_key is not None:
        pk = model._gather(layer.pool_key.unsqueeze(1), table).squeeze(-2)  # [(B,) L, Di / kp]
        return pk.reshape(*pk.shape[:-2], -1, Di)
    if layer.pool_inplace:
        return _inplace_keys(model, layer, d, table)
    from .glm5_next import pool_keys

    kv = model._load(layer.v_cache, table).squeeze(-2)  # [(B,) L, 2 Di]
    return pool_keys(layer, d, kv.reshape(*kv.shape[:-2], -1, kp, kv.shape[-1])[..., : 2 * Di])


def attention_long(model, layer, q_nope, qi, wi, positions, table):
    """The host path's chunk (table [P], positions [C]) or decode batch (table [B, P], positions [B]) over a long
    context, after its latent, indexer rows and pool keys were written (attention): the pooled indexer's scores of
    the complete candidate pools, their exact top index_topk / kpool, and absorbed MLA over the selected pools'
    tokens plus the query's tail (models/dsa_long.py). Nothing of the context's size per query is formed. Returns
    the o_proj output."""
    m = layer.spec.mla
    d = m.dsa
    kp = d.kpool
    T = q_nope.shape[0]
    nh, dn, dv, r = layer.nh, m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
    keep = d.topk // kp
    scale = d.head_dim ** -0.5
    pk = context_pool_keys(model, layer, d, table)  # [P, Di] (chunk) or [B, P, Di]
    npool = _dsa_long.npools(positions, kp)
    q_p = qi[1].reshape(T, d.n_heads, -1)
    w2 = wi.view(T, d.n_heads)
    pools, cnt = _dsa_long.select(_dsa_long.scores(q_p, w2, pk, npool, scale), keep)
    rows, sbias = _dsa_long.slots(pools, cnt, npool, positions, table, model.page_size, kp, keep + 1)
    q_lat = torch.einsum("thd,hdr->thr", q_nope.reshape(T, nh, dn), model._w(layer, "w_uk").view(nh, dn, r))
    ol = _dsa_long.attend(q_lat, layer.k_cache, rows, sbias, m.softmax_scale)
    out = torch.einsum("thr,hvr->thv", ol.to(model.dtype), model._w(layer, "w_uv").view(nh, dv, r)).reshape(T, nh * dv)
    return F.linear(out, model._w(layer, "o"))


def _cp_pg(model):
    """The context-parallel collectives' process group: the row group (cp_group), else the attention group."""
    pg = getattr(model, "cp_group", None)
    return model.attn_group if pg is None else pg


def _cp_gather(model, x: torch.Tensor) -> torch.Tensor:
    """[A, ...]: every member of this rank's context-parallel group's x, in rank order (a zero-padded group
    all-reduce: exact), A = cp of them."""
    A = model.cp
    full = x.unsqueeze(0) * model.long_grp_onehot.to(x.dtype).view(A, *[1] * x.dim())
    return model._all_reduce(full.contiguous(), _cp_pg(model))


def _cp_merge_gather(model, lval: torch.Tensor, cpool: torch.Tensor, keep: int):
    """(gv [A, T, keep] as lval, gc [A, T, keep] fp32) from ONE gather of [lval, cpool] in fp32 instead of two: the
    same values, as each slot sums one rank's value with the others' x * 0, element for element the same zeros in
    both forms (a sum of zeros is -0 only when every term is, in any order), and fp32 holds every pool index below
    2^24. One group all-reduce fewer per DSA layer."""
    packed = _cp_gather(model, torch.cat([lval.to(torch.float32), cpool.to(torch.float32)], dim=-1))  # [A, T, 2 keep]
    return packed[..., :keep].to(lval.dtype), packed[..., keep:]


def _idx_q_lat_rows(model, layer, q_resid: torch.Tensor, q_lat: torch.Tensor):
    """A decode batch's (indexer query [T, n_heads, head_dim], q_all [T, cp nh, r]) from ONE gather over the
    context-parallel group.

    The indexer query projection is split by output head: every rank projects its n_heads / cp heads (a contiguous
    block of idx_wq's rows a hs Di .. (a + 1) hs Di - 1, with their block scales, so 1 / cp of the weight's bytes:
    the replicated 12.6 MB bf16 idx_wq was ~384 of the ~1,040 weight tiles of a DSA layer's front), and the query is
    gathered back, because every rank scores its local pools over every head. Exact: each output's reduction over
    the input is unchanged. The same gather carries this rank's latent queries q_lat [T, nh, r] (attention_cp's),
    concatenated: each part is element for element the value of its own gather (zero-padded: one value plus zeros
    per position). Head a nh + j of q_all is rank a's head j."""
    d = layer.spec.mla.dsa
    A, Hi, Di = model.cp, d.n_heads, d.head_dim
    hs, T = Hi // A, q_resid.shape[0]
    nh, r = q_lat.shape[1], q_lat.shape[2]
    w, s = layer.idx_wq, getattr(layer, "idx_wq_scale", None)
    w = w.view(A, hs * Di, w.shape[-1]).index_select(0, model.long_grp_index)[0]
    if s is not None:
        s = s.view(A, hs * Di, s.shape[-1]).index_select(0, model.long_grp_index)[0]
    q = F.linear(q_resid, dequant(w, s, model.dtype))  # [T, hs Di]: this rank's heads
    if q_lat.dtype != q.dtype:  # one gather of one dtype: a cast here would change q_lat's bits, so refuse it
        raise ValueError(f"q_lat {q_lat.dtype} and the indexer query {q.dtype} differ")
    g = _cp_gather(model, torch.cat([q, q_lat.reshape(T, nh * r)], dim=-1))  # [A, T, hs Di + nh r]
    qi = g[..., : hs * Di].permute(1, 0, 2).reshape(T, Hi, Di)
    q_all = g[..., hs * Di :].reshape(A, T, nh, r).permute(1, 0, 2, 3).reshape(T, A * nh, r)
    return qi, q_all


def _cp_reduce_scatter(model, x: torch.Tensor) -> torch.Tensor:
    """Block rank of A equal row blocks of x [A r, ...] summed over the context-parallel group."""
    if x.device.type == "cpu":
        y = model._all_reduce(x, _cp_pg(model))
        return y.reshape(model.cp, -1, *x.shape[1:]).index_select(0, model.long_grp_index)[0]
    import torch.distributed._functional_collectives as funcol

    return funcol.reduce_scatter_tensor(x, "sum", 0, _cp_pg(model))


def _lnc2() -> bool:
    """Whether this graph is traced for trn2 at LNC=2 (False on the host, where no runtime is configured)."""
    from .. import platform

    try:
        return platform.nki_grid() == 2
    except RuntimeError:
        return False


def _pool_row_flat(table: torch.Tensor, pg: torch.Tensor, mm: torch.Tensor, ppl: int) -> torch.Tensor:
    """pool_row of a chunk's one block table [P] at LNC=2: the page of each local pool by a 1-D gather of the table,
    not an element gather from the table broadcast to [rows, P]. Equal values. On trn2 the broadcast form computed
    wrong in the 4096-page prefill piece graphs (a 32 MB broadcast int64 [1024, 4096]): with it, and the selection
    kernel's shared scratch at grid 2 (kernels/dsa_long_select.py), the 4096-page bucket answered a long-context
    needle prompt wrong. Alone on one core both forms are exact, and the 1024-page graphs are right with either."""
    pgc = pg.clamp(max=table.shape[-1] - 1)
    return table.view(-1)[pgc.reshape(-1)].view(pgc.shape) * ppl + (mm - pg * ppl)


def _cp_scores(model, layer, q_p, w2, pk, table, nloc, scale: float) -> torch.Tensor:
    """[T, Pl] fp32: attention_cp's local pool scores (NEG_INF at local pools m >= nloc): dsa_long.scores over the
    gathered pool keys pk [T, Pl, Di], or (a decode call of whole 128-page groups, 128-dim keys) kernels/dsa_index.py,
    which reads the pool keys' page rows in place and scores them on the chip instead of gathering them into
    [B, Pl, Di] (8,192 x 128 per row at the 1024-page bucket) for XLA's einsum. The page rows' pool order is
    attention_cp's local order (page j's ppk pools at j ppk ..), the arithmetic dsa_long.scores' up to rounding
    (w scale relu(q k) against w relu(scale q k)); pk then has no reader."""
    d = layer.spec.mla.dsa
    if not (table.dim() == 2 and table.shape[1] % 128 == 0 and d.head_dim == 128):
        return _dsa_long.scores(q_p, w2, pk, nloc, scale)
    from ..kernels import dsa_index

    T, Pl, Di = pk.shape
    ppk = Pl // table.shape[1]  # local pools per page row (page_size / (kpool cp))
    cand = torch.where(torch.arange(Pl, device=q_p.device).view(1, Pl) < nloc.view(-1, 1), 0.0, NEG_INF)
    if CP_INDEX_ROWS and T > CP_INDEX_ROWS:  # one kernel call per piece of rows
        return _cp_scores_rows(q_p, w2.float() * scale, layer.pool_key.view(-1, ppk * Di), table, cand)
    return dsa_index.scores(q_p, w2.float() * scale, layer.pool_key.view(-1, ppk * Di), dsa_index.page_groups(table),
                            cand)


def _shell_sel_takes(layer, table, keep: int, Pl: int) -> bool:
    """Whether attention_cp's decode branch takes the DSA shell's selection call (kernels/dsa_shell.py select2):
    whole 128-page groups, 128-dim keys, the separate bf16 pool-key cache at 8 local pools per page row, more local
    pools than keep (a multiple of 128 on the device: the kernel's k tiles)."""
    from ..kernels import dsa_shell

    d = layer.spec.mla.dsa
    return (table.dim() == 2 and table.shape[1] % dsa_shell.UNIT == 0
            and d.head_dim == dsa_shell.D and layer.pool_key is not None
            and Pl == table.shape[1] * dsa_shell.PPP and Pl > keep
            and ((layer.pool_key.dtype == torch.bfloat16 and keep % 128 == 0)
                 or table.device.type == "cpu"))  # the host runs emulate(): any keep, the CPU model's fp32 cache


def _shell_merge_takes(table, keep: int, A: int, kp: int, ppl: int) -> bool:
    """Whether attention_cp's decode branch takes the DSA shell's attend call (kernels/dsa_shell.py attend): 8 local
    pools per page row, and on the device the kernel's tile (A keep a multiple of 128, A dividing 128)."""
    from ..kernels import dsa_shell

    pow2 = A & (A - 1) == 0 and kp & (kp - 1) == 0
    return (table.dim() == 2 and ppl == dsa_shell.PPP and pow2
            and ((A * keep) % 128 == 0 and 128 % A == 0 or table.device.type == "cpu"))


def attention_cp(model, layer, x, positions, slot_mapping, table, ffn=None):
    """attention() of a pooled DSA layer under context parallelism (models/dsa_long.py "context parallelism"): this
    rank holds the context's pools m A + rank only. A decode batch (table [B, P], positions [B]) on the device; on
    the host also a chunk (table [P], positions [C]) when the CP group is the whole attention group. Returns the
    o_proj output of this rank's heads (the head partial _attn_all_reduce sums), or with ffn (the fused DSA layer
    that runs the FFN half too) the FFN half's (S_n, post, comb, y).

    A decode step on the device runs the whole section from the selection to o_proj as one NKI call
    (kernels/dsa_sat.py; kernels/dsa_satwf.py with the FFN half). The step-by-step form below it is what that call
    computes, run on the host and in decode steps the call does not take."""
    m = layer.spec.mla
    d = m.dsa
    kp, Di = d.kpool, d.head_dim
    A = model.cp
    rank = model.long_grp_index
    T = x.shape[0]
    nh, dn, dv, r = layer.nh, m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
    keep = d.topk // kp
    scale = Di ** -0.5
    ps = model.page_size
    if layer.pool_key is None:
        raise NotImplementedError("context-parallel DSA needs the separate pool-key cache (an FP8 KV cache)")
    q_nope, q_pe, c, k_pe, q_resid = _project(model, layer, x, positions, dec=table.dim() == 2)
    qi, ki, wi = _indexer(model, layer, x, q_resid, positions)
    q_pre = None  # a decode batch's q_all, from the indexer query's gather
    if table.dim() == 2 and d.n_heads % A == 0:
        q_lat = torch.einsum("thd,hdr->thr", q_nope.reshape(T, nh, dn), model._w(layer, "w_uk").view(nh, dn, r))
        q_idx, q_pre = _idx_q_lat_rows(model, layer, q_resid, q_lat)
        qi = (None, q_idx)  # the replicated projection above is then dead code

    # A padded row (slot_mapping in the null page, ModelRunner.pad_slots) sits at position 0 of a real sequence's table
    # in a chunk: it must not write that sequence's first pool, so it writes the null page.
    real = slot_mapping >= ps
    dump = _dsa_long.cp_dump_slot(ps, A)
    ls = _dsa_long.cp_local_slots(positions, table, ps, kp, A, rank)
    ls = torch.where(real, ls, torch.full_like(ls, dump))
    model._store(layer.k_cache, ls, c.unsqueeze(1))
    _cp_zero_dump(model, layer.k_cache, dump, c)
    model._store(layer.v_cache, ls, ki.unsqueeze(1))
    _cp_zero_dump(model, layer.v_cache, dump, ki)
    # every written token's pool, from its kpool cached rows (all on the pool's owner; the others write their null
    # page), as write_pool_keys does
    first = positions - positions % kp
    pos4 = (first.unsqueeze(-1) + torch.arange(kp, device=positions.device, dtype=positions.dtype)).reshape(-1)
    tb4 = table if table.dim() == 1 else table.repeat_interleave(kp, dim=0)
    slots4 = _dsa_long.cp_local_slots(pos4, tb4, ps, kp, A, rank)
    slots4 = torch.where(real.repeat_interleave(kp), slots4, torch.full_like(slots4, dump))
    rows = layer.v_cache[slots4]
    if model.fp8_max is not None:
        rows = rows.to(model.dtype)
    from .glm5_next import pool_keys

    keys = pool_keys(layer, d, rows.reshape(T, kp, -1)[..., : 2 * Di])
    layer.pool_key.index_put_((slots4,), keys.reshape(T * kp, Di // kp).to(layer.pool_key.dtype))
    _cp_zero_dump(None, layer.pool_key, dump, keys)
    # this rank's pools of the context(s): local pool m = context pool m A + rank
    prow = _dsa_long.cp_pool_rows(table, ps, kp, A)  # [(B,) Pl]
    if table.dim() == 2:
        # the local pool keys as whole page rows: row page of [pages, ppl Di] is pool rows page ppl .. (2 KiB
        # descriptors at ppl 8, Di 128, against 256 B rows), the same values
        ppk = ps // (kp * A)
        pk = layer.pool_key.view(-1, ppk * Di)[table].view(T, prow.shape[-1], Di)
    else:
        pk = layer.pool_key.view(-1, Di)[prow]  # [(B,) Pl, Di]
    if model.fp8_max is not None and pk.dtype != model.dtype:
        pk = pk.to(model.dtype)
    npool = _dsa_long.npools(positions, kp)
    nloc = _dsa_long.cp_local_count(npool, A, rank)
    q_p = qi[1].reshape(T, d.n_heads, -1)
    w2 = wi.view(T, d.n_heads)
    # A rank may hold fewer local pools than keep (a short context's bucket: 33 pages x 8 local pools = 264 < 512 at
    # A = 8); all of them are then candidates. Those shapes take the scores and the decode form's selection (the kernel
    # needs keep <= its pools), with the local list padded to keep.
    few = prow.shape[-1] < keep
    shell_sel = _shell_sel_takes(layer, table, keep, prow.shape[-1])
    if shell_sel and q_pre is not None and hasattr(layer, "w_uv_t") and (q_nope.device.type != "cpu" or SH.HOST_SIM):
        # the path below, from the selection to o_proj, as one NKI call
        ppl_ = ps // (kp * A)
        if _shell_merge_takes(table, keep, A, kp, ppl_) and _mk_rows_on(T) and ffn is not None:
            # + the world sum and the FFN half (the fused DSA layer); hybrid.layer reads the tuple
            from ..kernels import dsa_decode, dsa_satwf, dsa_shell
            from ..kernels import ffn_half as FH

            ident = torch.eye(T, dtype=torch.bfloat16, device=q_nope.device)
            S_, post_a, comb_a = ffn
            blobs, fkw = FH.decode_args(model, layer, S_.shape[1])
            r_ = dsa_satwf.satwf(q_p, w2.float() * scale, layer.pool_key.view(-1, dsa_shell.PPP * Di), table,
                                 nloc, positions, q_pre, layer.k_cache, layer.w_uv_t, layer.o_t, ident, keep, A,
                                 rank, model.tp_size, m.softmax_scale, S=S_, post_a=post_a, comb_a=comb_a,
                                 blobs=blobs, w_blob=layer.w_blob, ffn=fkw, kp=kp, ns=dsa_decode.NCH * 128,
                                 nr=SEL2_NR, pf=SEL2_PF, il=1, fz=1, ob=1, pw=1, rp=0, yb=1, pq=1, cs=SEL2_CS)
            # fz 1 + ob 1 + pw 1 + rp 0 + yb 1: the measured-exact form of the join, every cross-program hand-off
            # through outputs or program 0, none through internal shared scratch
            Sn, post, comb, rp, sp = r_[10:15]
            return Sn, post, comb, FH.decode_out(model, rp, sp, S_.dtype)
    if shell_sel:  # the scores, the selection and its compaction in one call
        from ..kernels import dsa_shell

        lval, cpool_sh, rows_sh = dsa_shell.select2(q_p, w2.float() * scale,
                                                    layer.pool_key.view(-1, dsa_shell.PPP * Di), table, nloc, keep, A,
                                                    rank, nr=SEL2_NR, pf=SEL2_PF, cs=SEL2_CS)
    else:
        if q_nope.device.type == "cpu" and not SH.HOST_SIM:
            sc = _cp_scores(model, layer, q_p, w2, pk, table, nloc, scale)  # [T, Pl]
            lp, lc = _dsa_long.select(sc, keep)
        else:
            sc = _cp_scores(model, layer, q_p, w2, pk, table, nloc, scale)  # [T, Pl]
            lp, lc = _dsa_long.select_device(sc, keep)
        k = torch.arange(keep, device=lp.device).view(1, keep)
        fill = sc.new_full((T, keep), NEG_INF) if few else torch.full_like(sc[:, :keep], NEG_INF)
        lval = torch.where(k < lc.view(T, 1), sc.gather(1, lp), fill)
        if table.dim() == 1:  # a chunk: the selection kernel's value order (_cp_attend_classes)
            from ..kernels import dsa_long_select

            lp, lc, lval = dsa_long_select.value_order(lp, lc, lval)
    cpool = cpool_sh if shell_sel else lp * A + rank.view(())  # context pool indices of the local list
    ppl = ps // (kp * A)
    # the merge, the slot table and the attention in one call (kernels/dsa_shell.py attend)
    shell_attend = _shell_merge_takes(table, keep, A, kp, ppl) and _shell_rows_on(T)
    if shell_attend:  # the two lists in one gather
        gv, gc = _cp_merge_gather(model, lval, cpool, keep)
        every_val = gv.permute(1, 0, 2)  # [T, A, keep]
        every_c = gc.permute(1, 0, 2)
    else:
        every_val = _cp_gather(model, lval).permute(1, 0, 2)  # [T, A, keep]
        every_c = _cp_gather(model, cpool.to(torch.float32)).permute(1, 0, 2)
        # a decode batch above CP_MERGE_ROWS rows merges in pieces of it; a prefill chunk's merge is one piece
        sel = _dsa_long.cp_merge(every_val, every_c, keep, _dsa_long.CP_MERGE_ROWS if table.dim() == 2 else None,
                                 prow.shape[-1] * A, two_k=table.dim() == 2 and T >= 2)  # [T, A, keep]
        mine = sel.permute(1, 0, 2).reshape(A, T * keep)
        mine = mine.to(torch.float32).index_select(0, rank).view(T, keep) > 0  # this rank's selected entries
    # slots: the local list (selected ones attended), then the tail pool if this rank owns it
    tb = table.view(1, -1).expand(T, -1) if table.dim() == 1 else table
    flat = table.dim() == 1 and _lnc2()  # see _pool_row_flat
    def pool_row(mm):
        pg = torch.floor(mm.to(torch.float32) * (1.0 / ppl)).to(torch.int64)
        if flat:
            return _pool_row_flat(table, pg, mm, ppl)
        return tb.gather(1, pg.clamp(max=tb.shape[1] - 1)) * ppl + (mm - pg * ppl)
    tail_m = torch.floor(npool.to(torch.float32) * (1.0 / A)).to(torch.int64)
    tail_own = (npool - tail_m * A) == rank.view(())
    Pl = prow.shape[-1]
    rows_sel = rows_sh.to(torch.int64) if shell_sel else pool_row(lp)
    rows_tail = pool_row(tail_m.clamp(max=Pl - 1).view(T, 1))
    cpu = q_nope.device.type == "cpu" and not SH.HOST_SIM
    from ..kernels import dsa_decode

    n_slots = keep + 1 if cpu else dsa_decode.NCH * 128
    if table.dim() == 1:  # a chunk: the slot classes (decode batches failed neuronx-cc 2.27 with them, NCC_ILSM901)
        q_lat = torch.einsum("thd,hdr->thr", q_nope.reshape(T, nh, dn), model._w(layer, "w_uk").view(nh, dn, r))
        q_all = _cp_gather(model, q_lat).permute(1, 0, 2, 3).reshape(T, A * nh, r)
        o_r, lse_r = _cp_attend_classes(q_all, layer.k_cache, rows_sel, rows_tail, mine, tail_own, npool, positions,
                                        m.softmax_scale, kp, CP_SLOTS_SMALL, n_slots)
        return _cp_combine(model, layer, o_r, lse_r, T, A, nh, r, dv)
    if not shell_attend:
        # by broadcasting, not concatenation (dsa_long.slots: NCC_IFML902 on a concatenate of these pieces)
        sl = torch.arange(n_slots, device=positions.device).view(1, n_slots)
        cl = sl.clamp(max=keep - 1).expand(T, n_slots)
        srows = torch.where(sl < keep, torch.gather(rows_sel, 1, cl),
                            torch.where(sl == keep, rows_tail, torch.zeros_like(rows_tail)))
        t4 = torch.arange(kp, device=positions.device).view(1, 1, kp)
        sel_ok = ((sl < keep) & torch.gather(mine, 1, cl)).unsqueeze(-1)
        tail_ok = ((sl == keep) & tail_own.view(T, 1)).unsqueeze(-1) & (npool.view(T, 1, 1) * kp + t4
                                                                         <= positions.view(T, 1, 1))
        sbias = torch.where(sel_ok | tail_ok, 0.0, NEG_INF).to(torch.float32)
    # every head's latent query (the heads are split over the group: gathered with the indexer query), partial
    # attention, combine
    if q_pre is not None:
        q_all = q_pre
    else:
        q_lat = torch.einsum("thd,hdr->thr", q_nope.reshape(T, nh, dn), model._w(layer, "w_uk").view(nh, dn, r))
        q_all = _cp_gather(model, q_lat).permute(1, 0, 2, 3).reshape(T, A * nh, r)
    if shell_attend:
        # the merge (its rows stacked, one interleaved selection), the slot table built in SBUF, every row's tail at
        # once, and the attention with its over-128-slot fallback inside the call: o and lse are final
        from ..kernels import dsa_shell

        o_r, lse_r, srows, sbias, over = dsa_shell.attend(gv, gc, rows_sel, table, positions, q_all, layer.k_cache,
                                                          keep, A, rank, m.softmax_scale, kp=kp, ns=n_slots, il=1,
                                                          st=1, ss=1, tl=1, fo=1)
    else:  # each row over its live slots (dsa_slots_c compacts each row's ~65 live slots of 640 into one 128 chunk)
        from ..kernels import dsa_slots_c

        o_r, lse_r = dsa_slots_c.attend(q_all, layer.k_cache, srows, sbias, m.softmax_scale, lse=True)
    lse_all = _cp_gather(model, lse_r.float())  # [A, T, H]
    LSE = torch.logsumexp(lse_all, dim=0)  # [T, H]
    part = o_r * torch.exp(lse_r - LSE).unsqueeze(-1)  # [T, H, R]
    part = part.view(T, A, nh, r).permute(1, 0, 2, 3).reshape(A * T, nh, r).contiguous()
    o = _cp_reduce_scatter(model, part)  # [T, nh, R]: this rank's heads
    out = torch.einsum("thr,hvr->thv", o.to(model.dtype), model._w(layer, "w_uv").view(nh, dv, r)).reshape(T, nh * dv)
    return F.linear(out, _wd(model, layer, "o", table.dim() == 2))


def attention_cp_rows(model, layer, x, positions, slot_mapping, table):
    """attention_cp of a prefill chunk (table [P], positions [T]) over context-parallel ROW GROUPS
    (models/dsa_long.py cp_degree). The attention group (attention TP = TP, DP attention 1) is R = cp_rows row
    groups of A = cp consecutive ranks; each row group holds a context-parallel copy of the sequence's DSA cache (its
    rank c holds context pools m A + c), so every rank writes ITS share of all T rows' latent, indexer rows and pool
    keys, while row group g selects and attends only the chunk's rows g n .. (g + 1) n - 1 (n = T / R) with every
    head (the whole-head q_b / W_UK / W_UV / o_proj copies, layer.qshard). Per rank the selection, the merge (over the
    A ranks of the row group) and the slot attention (the slot classes, _cp_attend_classes, over the rank's own
    selected pools and its tail) are those of an n-row chunk at CP A, while the chunk has T rows. The row group's
    partials are combined by log-sum-exp with a reduce-scatter onto n / A-row sub-blocks: rank c of row group g ends
    with rows g n + c n / A .., i.e. block g A + c of T / (R A) rows, its tensor-parallel rank's sequence-parallel
    block. Returns the o_proj output of those rows in their place of a zero [T, H]: the token mixer's world reduction
    (_attn_all_reduce: a reduce-scatter with sequence-parallel streams) sums each row's one value with zeros,
    exactly."""
    m = layer.spec.mla
    d = m.dsa
    kp, Di = d.kpool, d.head_dim
    A, R = model.cp, model.cp_rows
    rank = model.long_grp_index  # this rank within its row group
    T = x.shape[0]
    Hh, dn, dv, r = layer.spec.num_heads, m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
    if T % (A * R):
        raise ValueError(f"context-parallel row groups: a {T}-row chunk does not divide into {R} x {A} row blocks")
    if m.qk_rope_head_dim or not getattr(layer, "qshard", False) or kp < 2 or layer.pool_key is None:
        raise NotImplementedError("context-parallel row groups need a NoPE pooled DSA layer, the whole-head copies and "
                                  "the separate pool-key cache (an FP8 KV cache)")
    n = T // R
    keep = d.topk // kp
    scale = Di ** -0.5
    ps = model.page_size
    gi = model.cp_row_index

    def mine(t):  # this row group's rows
        return t.reshape(R, n, *t.shape[1:]).index_select(0, gi)[0]

    # every row's latent and indexer key rows (_project / _indexer's arithmetic), this row group's queries
    a = F.linear(x, model._w(layer, "w_a"))
    ql = m.q_lora_rank
    q_resid = rms_norm(a[:, :ql], layer.q_a_norm, m.norm_eps)
    c = rms_norm(a[:, ql:ql + r], layer.kv_a_norm, m.norm_eps)
    qrb = mine(q_resid)
    q_p = F.linear(qrb, model._w(layer, "idx_wq")).view(n, d.n_heads, Di)
    ki = F.linear(x, model._w(layer, "idx_wk"))
    ki = F.layer_norm(ki.float(), (Di,), layer.idx_knorm_w.float(), layer.idx_knorm_b.float(), 1e-6).to(x.dtype)
    w2 = F.linear(mine(x).float(), layer.idx_wproj) * d.n_heads ** -0.5
    ki = torch.cat([ki, F.linear(x, layer.idx_pool_gate)], dim=-1)

    # this rank's share of every row (attention_cp's writes; padded rows to the dump slot)
    real = slot_mapping >= ps
    dump = _dsa_long.cp_dump_slot(ps, A)
    ls = _dsa_long.cp_local_slots(positions, table, ps, kp, A, rank)
    ls = torch.where(real, ls, torch.full_like(ls, dump))
    model._store(layer.k_cache, ls, c.unsqueeze(1))
    _cp_zero_dump(model, layer.k_cache, dump, c)
    model._store(layer.v_cache, ls, ki.unsqueeze(1))
    _cp_zero_dump(model, layer.v_cache, dump, ki)
    first = positions - positions % kp
    pos4 = (first.unsqueeze(-1) + torch.arange(kp, device=positions.device, dtype=positions.dtype)).reshape(-1)
    slots4 = _dsa_long.cp_local_slots(pos4, table, ps, kp, A, rank)
    slots4 = torch.where(real.repeat_interleave(kp), slots4, torch.full_like(slots4, dump))
    vrows = layer.v_cache[slots4]
    if model.fp8_max is not None:
        vrows = vrows.to(model.dtype)
    from .glm5_next import pool_keys

    keys = pool_keys(layer, d, vrows.reshape(T, kp, -1)[..., : 2 * Di])
    layer.pool_key.index_put_((slots4,), keys.reshape(T * kp, Di // kp).to(layer.pool_key.dtype))
    _cp_zero_dump(None, layer.pool_key, dump, keys)

    # this row group's rows: the local selection in the kernel's value order, the exact merge over the row group
    prow = _dsa_long.cp_pool_rows(table, ps, kp, A)  # [Pl]
    pk = layer.pool_key.view(-1, Di)[prow]  # [Pl, Di]
    if model.fp8_max is not None and pk.dtype != model.dtype:
        pk = pk.to(model.dtype)
    pos_b = mine(positions)
    npool = _dsa_long.npools(pos_b, kp)
    nloc = _dsa_long.cp_local_count(npool, A, rank)
    Pl = prow.shape[-1]
    cpu = x.device.type == "cpu"
    from ..kernels import dsa_long_select

    K = _local_k(keep, A)
    if not K:  # keep at most CP_LOCAL_K (small configs): every rank's whole keep list
        if not cpu and not Pl < keep:
            lp, lc, lval = _select_tiles(q_p, w2, pk, nloc, keep, scale, vorder=True)
        else:
            sc = _dsa_long.scores(q_p, w2, pk, nloc, scale)  # [n, Pl]
            lp, lc = _dsa_long.select(sc, keep) if cpu else _dsa_long.select_device(sc, keep)
            k = torch.arange(keep, device=lp.device).view(1, keep)
            fill = sc.new_full((n, keep), NEG_INF) if Pl < keep else torch.full_like(sc[:, :keep], NEG_INF)
            lval = torch.where(k < lc.view(n, 1), sc.gather(1, lp), fill)
            lp, lc, lval = dsa_long_select.value_order(lp, lc, lval)
        cpool = lp * A + rank.view(())
        every_val = _cp_gather(model, lval).permute(1, 0, 2)  # [n, A, keep]
        every_c = _cp_gather(model, cpool.to(torch.float32)).permute(1, 0, 2)
        sel = _dsa_long.cp_merge(every_val, every_c, keep, None, Pl * A)  # [n, A, keep]
        chosen = sel.permute(1, 0, 2).reshape(A, n * keep)
        chosen = chosen.to(torch.float32).index_select(0, rank).view(n, keep) > 0  # this rank's selected entries
    else:  # CP_LOCAL_K: each rank's top K + 8, the merge over the top K, the certificate's (K + 1)-th values
        lp, lc, lval = _cp_local_lists(q_p, w2, pk, nloc, K + 8, scale, Pl, cpu)
        lp = lp[:, :K]
        cpool = lp * A + rank.view(())
        every_val = _cp_gather(model, lval[:, :K + 1].contiguous()).permute(1, 0, 2)  # [n, A, K + 1]
        every_c = _cp_gather(model, cpool.to(torch.float32)).permute(1, 0, 2)  # [n, A, K]
        vk = every_val[:, :, :K]
        glue = not cpu and _cp_glue_takes(n, A, K, keep, ps // (kp * A))  # ints only
        if glue:  # the main rows' mask through their slot buffer in one call; its rows / bias feed the attention below
            g_rows, g_bias, fail = _cp_glue(model, table, every_val, every_c, vk, lp, npool, pos_b, rank, keep, A, K,
                                            Pl, ps // (kp * A), kp)
        else:
            sel = _dsa_long.cp_merge(vk, every_c, keep, None, Pl * A)  # [n, A, K]
            fail = _local_k_fail(vk, every_val[:, :, K], sel, keep)  # [n], the same on every rank of the row group
            chosen = _cp_chosen(sel, rank, A, n, K)
        # the failed rows again from every rank's whole keep list (up to nF of them, the first ones)
        nF = min(CP_LOCAL_F, n)
        fi, nf = _dsa_long.compact(fail.to(torch.float32).view(1, n), nF)
        fi = fi.view(nF)
        live = torch.arange(nF, device=fi.device) < nf.view(())
        nloc_f = torch.where(live, nloc[fi], torch.zeros_like(nloc[fi]))
        lp_f, _, lval_f = _cp_local_lists(q_p[fi], w2[fi], pk, nloc_f, keep, scale, Pl, cpu,
                                          trip=(nf > 0).to(torch.int32))  # an empty tile does not run (~0.24 ms)
        lp_f = torch.where(live.view(nF, 1), lp_f, torch.zeros_like(lp_f))  # (stale memory when it did not run)
        lval_f = torch.where(live.view(nF, 1), lval_f, torch.full_like(lval_f, NEG_INF))
        cpool_f = lp_f * A + rank.view(())
        ev_f = _cp_gather(model, lval_f).permute(1, 0, 2)  # [nF, A, keep]
        ec_f = _cp_gather(model, cpool_f.to(torch.float32)).permute(1, 0, 2)
        chosen_f = _cp_chosen(_dsa_long.cp_merge(ev_f, ec_f, keep, None, Pl * A), rank, A, nF, keep) & live.view(nF, 1)
        if getattr(layer, "cp_localk", None) is not None:
            nff = fail.to(torch.float32).sum().view(1)  # every failed row: compact's count stops at nF
            layer.cp_localk.add_(torch.cat([torch.full_like(nff, float(n)), nff, (nff - float(nF)).clamp(min=0.0)]))
    ppl = ps // (kp * A)
    tb = table.view(1, -1).expand(n, -1)
    flat = _lnc2()  # see _pool_row_flat

    def pool_row(mm):
        pg = torch.floor(mm.to(torch.float32) * (1.0 / ppl)).to(torch.int64)
        if flat:
            return _pool_row_flat(table, pg, mm, ppl)
        return tb.gather(1, pg.clamp(max=tb.shape[1] - 1)) * ppl + (mm - pg * ppl)

    tail_m = torch.floor(npool.to(torch.float32) * (1.0 / A)).to(torch.int64)
    tail_own = (npool - tail_m * A) == rank.view(())
    rows_sel = pool_row(lp)
    rows_tail = pool_row(tail_m.clamp(max=Pl - 1).view(n, 1))
    from ..kernels import dsa_decode

    n_slots = keep + 1 if cpu else dsa_decode.NCH * 128
    # every head's latent query of this row group's rows (whole-head q_b, W_UK), the partial attention over this
    # rank's slots
    q = F.linear(qrb, model._w(layer, "q_b_all")).view(n, Hh, dn)
    q_lat = torch.einsum("thd,hdr->thr", q, model._w(layer, "w_uk_all").view(Hh, dn, r))
    if not K:
        o_r, lse_r = _cp_attend_classes(q_lat, layer.k_cache, rows_sel, rows_tail, chosen, tail_own, npool, pos_b,
                                        m.softmax_scale, kp, CP_SLOTS_SMALL, n_slots)
    else:  # every row in one buffer of CP_SLOTS_SMALL slots (its K entries and the tail), the failed rows replaced
        if glue:  # the buffer the glue kernel built (_cp_attend_classes single's, bit for bit)
            from ..kernels import dsa_slots_n

            o_r, lse_r = dsa_slots_n.attend(q_lat, layer.k_cache, g_rows, g_bias, m.softmax_scale,
                                            torch.full((1,), n, dtype=torch.int64, device=q_lat.device), lse=True)
        else:
            o_r, lse_r = _cp_attend_classes(q_lat, layer.k_cache, rows_sel, rows_tail, chosen, tail_own, npool, pos_b,
                                            m.softmax_scale, kp, CP_SLOTS_SMALL, n_slots, single=True)
        tb_f = table.view(1, -1).expand(nF, -1)
        pg_f = torch.floor(lp_f.to(torch.float32) * (1.0 / ppl)).to(torch.int64)
        rows_f = tb_f.gather(1, pg_f.clamp(max=tb_f.shape[1] - 1)) * ppl + (lp_f - pg_f * ppl)
        o_f, lse_f = _cp_attend_classes(q_lat[fi], layer.k_cache, rows_f, rows_tail[fi], chosen_f,
                                        tail_own[fi] & live, npool[fi], pos_b[fi], m.softmax_scale, kp,
                                        CP_SLOTS_SMALL, n_slots)
        slot = (torch.cumsum(fail.to(torch.float32), 0) - 1.0).clamp(min=0.0).to(torch.int64)  # its place in fi
        use = fail & (slot < nF)
        slot = slot.clamp(max=nF - 1)
        o_r = torch.where(use.view(n, 1, 1), o_f[slot], o_r)
        lse_r = torch.where(use.view(n, 1), lse_f[slot], lse_r)
    # combined over the row group onto n / A-row sub-blocks, then W_UV and o_proj of every head for this rank's rows
    lse_all = _cp_gather(model, lse_r.float())  # [A, n, Hh]
    LSE = torch.logsumexp(lse_all, dim=0)  # [n, Hh]
    part = (o_r * torch.exp(lse_r - LSE).unsqueeze(-1)).contiguous()  # [n, Hh, r] = [A n / A, Hh, r]
    o = _cp_reduce_scatter(model, part)  # [n / A, Hh, r]: this rank's sub-block
    out = torch.einsum("thr,hvr->thv", o.to(model.dtype), model._w(layer, "w_uv_all").view(Hh, dv, r))
    out = F.linear(out.reshape(n // A, Hh * dv), model._w(layer, "o_all"))  # [n / A, H]
    oh = (model.cp_row_onehot.view(R, 1) * model.long_grp_onehot.view(1, A)).reshape(R * A).to(out.dtype)
    full = out.unsqueeze(0) * oh.view(R * A, 1, 1)
    return full.reshape(T, -1)


def _cp_local_lists(q_, w_, pk, nloc, L: int, scale: float, Pl: int, cpu: bool, trip=None):
    """(pools, counts, values) of this rank's local top L of rows q_ in the selection kernel's value order
    (attention_cp_rows' selection; a module function, not a closure: a closure would turn the caller's variables
    into cells, which renames their nodes in the trace). trip (an int tensor of one element, 0 or 1, at most 128
    rows): kernels/dsa_long_select_t.py, the tile run only when trip is 1 (its outputs are stale memory at 0: the
    caller masks them)."""
    from ..kernels import dsa_long_select

    rr = q_.shape[0]
    if not cpu and not Pl < L:
        if trip is not None and rr <= 128:
            from ..kernels import dsa_long_select_t

            return dsa_long_select_t.select(q_, w_, pk, nloc, L, scale, trip, vorder=True)
        return _select_tiles(q_, w_, pk, nloc, L, scale, vorder=True)
    sc_ = _dsa_long.scores(q_, w_, pk, nloc, scale)  # [rr, Pl]
    lp_, lc_ = _dsa_long.select(sc_, L) if cpu else _dsa_long.select_device(sc_, L)
    k_ = torch.arange(L, device=lp_.device).view(1, L)
    fill_ = sc_.new_full((rr, L), NEG_INF) if Pl < L else torch.full_like(sc_[:, :L], NEG_INF)
    lv = torch.where(k_ < lc_.view(rr, 1), sc_.gather(1, lp_), fill_)
    return dsa_long_select.value_order(lp_, lc_, lv)


def _cp_glue_takes(n: int, A: int, K: int, keep: int, ppl: int) -> bool:
    """Whether the glue kernel (kernels/dsa_cp_glue.py) holds attention_cp_rows' local-K call: its shapes
    (dsa_cp_glue.takes, the single buffer of CP_SLOTS_SMALL slots), a merge that selects (A K > keep) in one dsa_topk
    call. Ints only."""
    from .. import platform
    from ..kernels import dsa_cp_glue, dsa_topk

    return (A * K > keep and dsa_topk.supported(n, A * K)
            and dsa_cp_glue.takes(n, A, K, CP_SLOTS_SMALL, ppl, platform.nki_grid()))


def _cp_glue(model, table, every_val, every_c, vk, lp, npool, pos_b, rank, keep: int, A: int, K: int, Pl: int,
             ppl: int, kp: int):
    """(rows [n, CP_SLOTS_SMALL] int32, bias [n, CP_SLOTS_SMALL, kp], fail [n] bool) of the main rows, as cp_merge ->
    _local_k_fail / _cp_chosen -> pool_row -> _cp_attend_classes(single=True) give them: the merge's dsa_topk selection
    (cp_merge_rows' call, on the same scores), the tail's row and ownership (attention_cp_rows' lines, here before the
    kernel that needs them), then kernels/dsa_cp_glue.py with the merge's TIE_BITS. A module function, not a
    closure."""
    from ..kernels import dsa_cp_glue, dsa_topk

    n = vk.shape[0]
    s = vk.reshape(n, A * K).float()  # cp_merge_rows' scores: dsa_topk's input and the kernel's, row-major
    sel0 = dsa_topk.select(s, keep, vis_only=True)
    tail_m = torch.floor(npool.to(torch.float32) * (1.0 / A)).to(torch.int64)
    tail_own = (npool - tail_m * A) == rank.view(())
    pg = torch.floor(tail_m.clamp(max=Pl - 1).to(torch.float32) * (1.0 / ppl)).to(torch.int64)
    rows_tail = _pool_row_flat(table, pg, tail_m.clamp(max=Pl - 1), ppl)
    npos = torch.stack([npool.to(torch.float32), pos_b.to(torch.float32)], dim=1)
    return dsa_cp_glue.glue(s, every_c.reshape(n, A * K).float(), every_val[:, :, K].float(), sel0, lp, table, rows_tail,
                            tail_own, npos, model.long_grp_onehot, keep, ppl, kp, CP_SLOTS_SMALL, _dsa_long.TIE_BITS)


def _cp_chosen(sel_, rank, A: int, rr: int, L: int):
    """This rank's selected entries [rr, L] of a merge's [rr, A, L] selection."""
    c_ = sel_.permute(1, 0, 2).reshape(A, rr * L)
    return c_.to(torch.float32).index_select(0, rank).view(rr, L) > 0


def _local_k(keep: int, A: int) -> int:
    """CP_LOCAL_K for a merge of A ranks' lists into keep pools (0: not below keep)."""
    K = CP_LOCAL_K
    if K <= 0 or K >= keep:
        return 0
    if K % 8 or A * K < keep:
        raise ValueError(f"CP_LOCAL_K={K}: a multiple of 8 with {A} x K >= keep {keep}")
    return K


def _local_k_fail(vk, nxt, sel, keep: int):
    """[n] bool, CP_LOCAL_K's certificate failed: vk [n, A, K] every rank's top K (value order, NEG_INF past its
    count), nxt [n, A] each rank's (K + 1)-th, sel the merge of vk. A row fails when some rank's (K + 1)-th is a
    candidate and reaches the merged keep-th value, or the merge selected fewer than keep (then every candidate
    belongs, and a rank with a (K + 1)-th dropped one)."""
    n = vk.shape[0]
    t = torch.where(sel, vk, torch.full_like(vk, 3.0e38)).amin(dim=-1).amin(dim=-1)  # [n] the merged keep-th value
    # (against a tensor, not a float literal: a comparison with one lowers to f64, NCC_ESPP004)
    short = sel.to(torch.float32).sum(dim=-1).sum(dim=-1) < torch.full_like(t, float(keep))
    vis = torch.full_like(nxt, VISIBLE_SCORE)
    bad = (nxt > vis) & ((nxt >= t.view(n, 1)) | short.view(n, 1))
    return bad.to(torch.float32).amax(dim=-1) > 0


def _cp_zero_dump(model, cache: torch.Tensor, dump: int, like: torch.Tensor) -> None:
    """Zero the context-parallel dump slot's row of cache (attention_cp / attention_cp_rows: the tokens another rank
    owns, and padded ones, write the null page's last local slot, cp_dump_slot), as its own one-row write after the
    call's store into cache, so every later read sees it. Why: one request's NaN reached that slot on every rank that
    does not own its token, and every row whose block table pads with the null page gathers its pool keys into
    masked score lanes; kernels/dsa_index.py's transposes spread the NaN to that row's candidate lanes and changed its
    selection (in nki.simulate at trn2: 118 candidate lanes NaN in each healthy row, 26-30 of 120 selections moved).
    No real token lives on the null page, and a finite value in a masked lane scores NEG_INF either way (-1e30 + x
    rounds to -1e30 in fp32 for any |x| below ~3.8e22). The stores themselves stay as they are: a select on the stored
    rows moved the device rounding of the live k_cache rows on the bf16 -> e4m3 store. like: the stored rows (dtype
    only; model: its _store's cast, None for a cache written with index_put_ as is)."""
    idx = torch.full((1,), dump, dtype=torch.int64, device=cache.device)
    zero = torch.zeros((1, *cache.shape[1:]), dtype=like.dtype, device=cache.device)
    if model is None:
        cache.index_put_((idx,), zero.to(cache.dtype))
    else:
        model._store(cache, idx, zero)


def _cp_combine(model, layer, o_r, lse_r, T: int, A: int, nh: int, r: int, dv: int):
    """attention_cp's tail: the ranks' partial attentions combined by their log-sum-exps, this rank's heads, o_proj."""
    lse_all = _cp_gather(model, lse_r.float())  # [A, T, H]
    LSE = torch.logsumexp(lse_all, dim=0)  # [T, H]
    part = o_r * torch.exp(lse_r - LSE).unsqueeze(-1)  # [T, H, R]
    part = part.view(T, A, nh, r).permute(1, 0, 2, 3).reshape(A * T, nh, r).contiguous()
    o = _cp_reduce_scatter(model, part)  # [T, nh, R]: this rank's heads
    out = torch.einsum("thr,hvr->thv", o.to(model.dtype), model._w(layer, "w_uv").view(nh, dv, r)).reshape(T, nh * dv)
    return F.linear(out, model._w(layer, "o"))


def _cp_attend_classes(q_all, kc, rows_sel, rows_tail, mine, tail_own, npool, positions, scale: float, kp: int,
                       small: int, full: int, single: bool = False):
    """(o [T, H, R] fp32, lse [T, H] fp32): this rank's partial attention of every row over its selected local pools
    and its tail, as attention_cp's fixed slots give it, with each row in a buffer sized by where its selected entries
    end. The local lists come in the selection kernel's value order (score descending, pool ascending:
    dsa_long_select vorder), where the selected entries of a row are its first ones (the merge takes every value above
    the threshold and the ties with the smallest pools), so a row whose last selected entry sits before `small` - 1
    takes the small buffer: the list's first `small` - 1 slots as they are and the tail pool last, the others masked by
    `mine` as in the fixed slots; every other row the full one (attention_cp's fixed layout, `full` >= keep + 1
    slots). Masking by `mine` keeps it exact whatever the order. Each buffer holds its rows first (compact over the
    rows) and kernels/dsa_slots_n.py runs only those: n_small c(small) + n_full c(full). No gather of the local list
    (a per-element gather of [1024, 512] cost ~60 ms on trn1)."""
    from ..kernels import dsa_slots_n

    T, keep = mine.shape
    dev = q_all.device
    S1 = min(small - 1, keep)  # list slots of the small buffer; the tail takes slot S1
    j = torch.arange(keep, device=dev).view(1, keep)
    bound = torch.where(mine, j + 1, torch.zeros_like(j)).amax(-1)  # [T] one past the last selected entry
    fits = bound <= S1
    t4 = torch.arange(kp, device=dev).view(1, 1, kp)
    tail_tok = tail_own.view(T, 1, 1) & (npool.view(T, 1, 1) * kp + t4 <= positions.view(T, 1, 1))  # [T, 1, kp]

    def build(C: int, L: int):
        """C slots: the local list's first L entries, the tail at slot L, the rest padding."""
        sl = torch.arange(C, device=dev).view(1, C)
        if C <= keep:  # the list's first C columns as a slice (no gather), the tail and padding by broadcasting
            rs, ms = rows_sel[:, :C], mine[:, :C]
        else:
            cl = sl.clamp(max=keep - 1).expand(T, C)
            rs, ms = torch.gather(rows_sel, 1, cl), torch.gather(mine, 1, cl)
        rows = torch.where(sl < L, rs, torch.where(sl == L, rows_tail, torch.zeros_like(rows_tail)))
        ok = ((sl < L) & ms).unsqueeze(-1) | ((sl == L).unsqueeze(-1) & tail_tok)
        return rows, torch.where(ok, 0.0, NEG_INF).to(torch.float32)

    if single:  # every row fits by construction (CP_LOCAL_K): its keep entries, the tail, zero padding
        if small < keep + 1:
            raise ValueError(f"one slot buffer of {small} slots for lists of {keep} and the tail")
        C = small
        sl = torch.arange(C, device=dev).view(1, C)
        rs = torch.cat([rows_sel, rows_sel.new_zeros(T, C - keep)], dim=1) if C > keep else rows_sel
        ms = torch.cat([mine, mine.new_zeros(T, C - keep)], dim=1) if C > keep else mine
        rows = torch.where(sl < keep, rs, torch.where(sl == keep, rows_tail, torch.zeros_like(rows_tail)))
        ok = ((sl < keep) & ms).unsqueeze(-1) | ((sl == keep).unsqueeze(-1) & tail_tok)
        bias = torch.where(ok, 0.0, NEG_INF).to(torch.float32)
        return dsa_slots_n.attend(q_all, kc, rows, bias, scale, torch.full((1,), T, dtype=torch.int64, device=dev),
                                  lse=True)
    out_o, out_l = [], []
    for want, (C, L) in ((fits, (small, S1)), (~fits, (full, keep))):
        idx, n = _dsa_long.compact(want.to(torch.float32).view(1, T), T)  # this buffer's rows first
        idx = idx.view(T)
        rows, bias = build(C, L)
        o, ls = dsa_slots_n.attend(q_all[idx], kc, rows[idx], bias[idx], scale, n.view(1), lse=True)
        # row b's place in its buffer: the rows of its class before it (an exact fp32 count)
        pos = (torch.cumsum(want.to(torch.float32), 0) - 1.0).clamp(min=0.0).to(torch.int64)
        out_o.append(o[pos])
        out_l.append(ls[pos])
    o = torch.where(fits.view(T, 1, 1), out_o[0], out_o[1])
    lse = torch.where(fits.view(T, 1), out_l[0], out_l[1])
    return o, lse


def _select_tiles(q_p, w, pk, npool, keep: int, scale: float, vorder: bool = False):
    """kernels/dsa_long_select.py's selection of a chunk's queries over one context. A chunk of more than one 128-query
    tile runs kernels/dsa_long_pipe.py, every tile in one software-pipelined call (the selection of tile t overlapping
    the scores of tile t + 1; pools, order and scores bit-identical to one dsa_long_select call per tile), in blocks of
    MAX_TILES tiles. Where neither kernel takes the shape, one dsa_long_select call per tile: the kernel's own device
    loop over 128-query tiles returned wrong selections past the first tile on trn1 (112 / 496 of 1024 rows), so a call
    never holds more than one tile."""
    from ..kernels import dsa_long_select

    n = q_p.shape[0]
    kw = {"vorder": True} if vorder else {}  # (vorder: the kernel's value order, CP slot classes)
    if n <= 128:
        return dsa_long_select.select(q_p, w, pk, npool, keep, scale, **kw)
    from ..kernels import dsa_long_pipe

    P = pk.shape[0]
    sub = dsa_long_select.pick_sub(P, keep)
    if dsa_long_pipe.supported(n, q_p.shape[1], q_p.shape[2], P, keep, sub):
        return dsa_long_pipe.select(q_p, w, pk, npool, keep, scale, **kw)
    blk = dsa_long_pipe.MAX_TILES * 128
    if n > blk and dsa_long_pipe.supported(blk, q_p.shape[1], q_p.shape[2], P, keep, sub):
        # A chunk past MAX_TILES tiles per rank: the pipelined call over each block of MAX_TILES tiles, a tail under
        # two tiles one call per tile, rather than every tile its own call.
        parts = []
        for i in range(0, n, blk):
            m = min(blk, n - i)
            if dsa_long_pipe.supported(m, q_p.shape[1], q_p.shape[2], P, keep, sub):
                parts.append(dsa_long_pipe.select(q_p[i:i + m], w[i:i + m], pk, npool[i:i + m], keep, scale, **kw))
            else:
                parts += [dsa_long_select.select(q_p[j:j + 128], w[j:j + 128], pk, npool[j:j + 128], keep, scale,
                                                 **kw) for j in range(i, i + m, 128)]
        return tuple(torch.cat([p[j] for p in parts]) for j in range(3))
    parts = [dsa_long_select.select(q_p[i:i + 128], w[i:i + 128], pk, npool[i:i + 128], keep, scale, **kw)
             for i in range(0, n, 128)]
    return tuple(torch.cat([p[j] for p in parts]) for j in range(3))


def _pool_key_slots(model, d: DSASpec, positions: torch.Tensor, table: torch.Tensor,
                    slot_mapping: torch.Tensor) -> torch.Tensor:
    """[T * kpool]: the slots of each written token's pool through the block table, except that a padded row
    (slot_mapping < page_size: the null page, ModelRunner.pad_slots) targets its own slot kpool times. A
    padded row sits at position 0, so through a real sequence's table (a chunk's padded tail) its pool is
    that sequence's pool 0: it must not rewrite those slots, beside the real rows that own them."""
    kp = d.kpool
    T = positions.shape[0]
    first = positions - positions % kp  # kpool is a power of two here (4): exact
    pos = first.unsqueeze(-1) + torch.arange(kp, device=positions.device, dtype=positions.dtype)  # [T, kp]
    slots = model._slots(pos, table).reshape(T, kp)
    real = (slot_mapping >= model.page_size).unsqueeze(-1)
    return torch.where(real, slots, slot_mapping.unsqueeze(-1).expand(T, kp)).reshape(-1)


def write_pool_keys(model, layer, d: DSASpec, positions: torch.Tensor, table: torch.Tensor,
                    slot_mapping: torch.Tensor) -> None:
    """Recompute the pool key of every written token's pool from the cached indexer rows of its
    kpool tokens (after this call's rows were stored) and write it to the pool-key cache as kpool
    pieces, one per slot of the pool (pool_key_width). A pool's key is therefore rewritten whenever
    one of its tokens is, and holds the key of its current rows: once its last token is written it
    is the key the per-call form would compute, and the selection only scores complete pools (the
    query's own incomplete pool is the tail, attended whole). Rows of several tokens of one pool
    write the same values to the same slots. Padded rows write only their own null-page slot
    (_pool_key_slots): no slot a real row owns, and not the null page's slot 0 that padded rows read."""
    kp, Di = d.kpool, d.head_dim
    T = positions.shape[0]
    slots = _pool_key_slots(model, d, positions, table, slot_mapping)  # [T * kp]
    rows = layer.v_cache[slots]  # [T * kp, 1, 2 Di]
    if model.fp8_max is not None:
        rows = rows.to(model.dtype)
    from .glm5_next import pool_keys

    keys = pool_keys(layer, d, rows.reshape(T, kp, -1)[..., : 2 * Di])  # [T, Di]
    layer.pool_key.index_put_((slots,), keys.reshape(T * kp, Di // kp).to(layer.pool_key.dtype))


def _inplace_keys(model, layer, d: DSASpec, table: torch.Tensor) -> torch.Tensor:
    """[(B,) L / kpool, Di]: the pool keys of the context a block table addresses (inplace form): the key part of
    every pool's last row of the whole cache (a strided view, [slots / kpool, Di]) gathered by the block table, so
    the context's gates and other keys are never read."""
    kp, Di = d.kpool, d.head_dim
    ps = model.page_size
    v = layer.v_cache
    keys = v.view(-1, kp, v.shape[-1])[:, kp - 1, :Di]  # [slots / kp, Di]
    pages = keys.reshape(-1, ps // kp, Di)[table]  # [(B,) P, ps / kp, Di]
    pages = pages.to(model.dtype) if model.fp8_max is not None else pages
    return pages.flatten(-3, -2)


def write_pool_keys_inplace(model, layer, d: DSASpec, positions: torch.Tensor, table: torch.Tensor,
                            slot_mapping: torch.Tensor, vrow: torch.Tensor) -> None:
    """The inplace pool-key cache (pool_cache_mode): after this call's indexer rows vrow [T, 2 Di] were
    stored, every token whose position is the last of its pool (position % kpool == kpool - 1) rewrites
    its own row with the pool's key (glm5_next.pool_keys of the pool's kpool cached rows) in place of its
    indexer key; the other tokens rewrite their rows unchanged. So a complete pool's last row holds
    [key | that token's gate logits], its other rows their own key and gates, and the selection reads
    the key part of every pool's last row. Only the last row is ever overwritten, so a pool can always
    be recomputed from its first kpool - 1 rows plus a fresh last row: a later write of an incomplete
    pool's tokens finds the rows it needs. Padded rows (position 0) are no pool's last token. No extra
    bytes: the gate logits and keys of a complete pool are dead once its key exists, and the last row's
    key is one of them."""
    kp, Di = d.kpool, d.head_dim
    T = positions.shape[0]
    first = positions - positions % kp  # kpool is a power of two here (4): exact
    pos = first.unsqueeze(-1) + torch.arange(kp, device=positions.device, dtype=positions.dtype)  # [T, kp]
    rows = layer.v_cache[model._slots(pos, table).reshape(-1)]  # [T * kp, 1, 2 Di]
    if model.fp8_max is not None:
        rows = rows.to(model.dtype)
    from .glm5_next import pool_keys

    keys = pool_keys(layer, d, rows.reshape(T, kp, -1)[..., : 2 * Di])  # [T, Di]
    last = (positions % kp == kp - 1).unsqueeze(-1)  # [T, 1]
    key = torch.where(last, keys.to(vrow.dtype), vrow[:, :Di])
    model._store(layer.v_cache, slot_mapping, torch.cat([key, vrow[:, Di:]], dim=-1).unsqueeze(1))


def _shared(layer, T: int, L: int) -> torch.Tensor:
    """The selection a layer staged for T query rows over L keys (_share), as its additive mask."""
    return layer.dsa_mask[:T, :L].float()


def _share(layer, top, T: int) -> None:
    """Write a selection's additive mask [B, Q, L] to the scratch as whole scratch rows, a one-row
    write as two rows (on trn1 a ONE-row index_put_ of a computed value is the slow path)."""
    rows = torch.arange(T, device=top.device)
    val = top.reshape(T, -1)
    keys = layer.dsa_mask.shape[1]
    if val.shape[1] < keys:
        val = torch.cat([val, torch.full((T, keys - val.shape[1]), NEG_INF, dtype=val.dtype, device=val.device)], 1)
    from .linear_attn import _write_rows

    _write_rows(layer.dsa_mask, rows, val.to(layer.dsa_mask.dtype))


def reference(model, layer, x, positions, state: dict):
    """The attention block over a whole sequence without a cache (forward_logits, tests): decompressed
    keys and values, the indexer over every earlier token. state: the per-sequence carry of the caller
    (unused by DSA layers)."""
    T = x.shape[0]
    q_nope, q_pe, c, k_pe, q_resid = _project(model, layer, x, positions)
    j = positions.unsqueeze(0)
    vis = torch.where(j <= positions.unsqueeze(1), 0.0, NEG_INF).view(1, T, T)
    qi, ki, wi = _indexer(model, layer, x, q_resid, positions)
    out = _core(model, layer, q_nope, c.unsqueeze(0), vis, 1, T, None, (qi, wi, ki.unsqueeze(0)), absorb=False)
    return F.linear(out, model._w(layer, "o"))
