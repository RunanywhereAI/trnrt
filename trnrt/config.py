from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttnSpec:
    """One layer's attention shape. Uniform models repeat one; hybrids mix kinds."""

    num_heads: int
    num_kv_heads: int
    head_dim: int  # query / key
    v_head_dim: int
    rope_dim: int  # leading dims of q / k that RoPE rotates
    rope_theta: float
    window: int | None = None  # sliding window: a query sees the last `window` keys
    sink: bool = False  # per-query-head attention-sink logit in the softmax
    # Rotate only the first rope_freqs of the rope_dim / 2 frequency pairs (pair j is dims j and
    # j + rope_dim / 2); the other pairs pass through. None: all of them. K2-Horizon's
    # rope_head_dim 64 of head_dim 128 is rope_dim 128, rope_freqs 32 (see _k2_horizon).
    rope_freqs: int | None = None
    scale: float | None = None  # score scale (None: head_dim ** -0.5)
    # Hidden-conditioned relative position logits (Inkling): bias for key distances below
    # rel_extent, from a per-head d_rel projection of the layer input.
    rel_extent: int = 0
    log_scaled: bool = False  # queries and position logits scaled by ModelConfig.log_scaling
    # Multi-head latent attention (models/mla.py MLASpec). For an MLA layer num_kv_heads,
    # head_dim and v_head_dim describe its paged cache: one latent "head" whose K is the
    # kv_lora_rank latent and whose V is the rope key (plus the DSA indexer key).
    mla: object | None = None


@dataclass(frozen=True)
class LinearSpec:
    """One linear-attention layer (models/linear_attn.py): a short causal conv over q, k, v
    and a gated delta rule whose state is a fixed-size matrix per head instead of a KV cache.

    kind "gdn": Gated DeltaNet (Qwen3.5 / Qwen3.8 / Qwen3-Next), one decay per head and token,
    num_v_heads >= num_k_heads (each k head serves num_v_heads / num_k_heads v heads).
    kind "kda": Kimi Delta Attention (Kimi Linear / Kimi K3, GLM-5.3-Flash), one decay per key
    channel, q, k and v heads all equal."""

    kind: str
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel: int
    gate_act: str = "silu"  # activation of the output gate in the gated RMSNorm
    gate_rank: int = 0  # KDA output gate: 0 = full-rank g_proj, else g_a_proj / g_b_proj of this rank
    lower_bound: float | None = None  # KDA "safe gate": decay = lower_bound * sigmoid(...)


@dataclass(frozen=True)
class ModelConfig:
    """The subset of a Hugging Face config.json that trnRT's model code reads."""

    architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    eos_token_ids: tuple[int, ...]
    qk_norm: bool = False  # per-head RMSNorm on q and k before RoPE (Qwen3)
    qkv_bias: bool = False  # bias on q/k/v projections (Qwen2, attention_bias=True Llamas)
    rope_scaling: tuple | None = None  # sorted (key, value) items, e.g. llama3 scaling
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True
    moe_layers: tuple[int, ...] = ()  # indices of layers whose MLP is a sparse MoE
    # Per-layer token mixer (None = uniform attention). An entry's type is the layer's kind:
    # AttnSpec (softmax attention over the paged KV cache) or LinearSpec (recurrent state).
    attn_layers: tuple[AttnSpec | LinearSpec, ...] | None = None
    attn_output_gate: bool = False  # attention output * sigmoid(gate), gate rows in q_proj (Qwen3.5)
    norm_offset: bool = False  # RMSNorm weights are stored zero-centred, applied as (1 + w) (Qwen3.5)
    v_scale: float | None = None  # values scaled before attention (MiMo-V2)
    # "softmax" (Qwen3-MoE), "sigmoid" (DeepSeek-V3 style) or "topk_softmax" (gpt-oss: top-k of
    # the logits, softmax over just those k)
    router_scoring: str = "softmax"
    router_bias: bool = False  # e_score_correction_bias added for expert choice only
    routed_scaling_factor: float = 1.0
    n_shared_experts: int = 0  # DeepSeek-V3 style: a dense MLP of n x moe_intermediate_size beside the experts
    n_group: int = 1  # DeepSeek-V3 node-limited routing: experts in n_group groups, topk_group kept
    topk_group: int = 1
    fused_qkv_weights: bool = False  # checkpoint stores self_attn.qkv_proj (MiMo-V2)
    # Multi-token-prediction layers in the checkpoint (num_nextn_predict_layers) and the
    # attention they use; MiMo-V2's MTP layer uses the sliding-window configuration
    # (vLLM model_executor/models/mimo_v2_mtp.py, v0.30.0).
    mtp_layers: int = 0
    mtp_spec: AttnSpec | None = None
    # DeepSeek-V3 / GLM-5 MTP (models/mtp.py): the layer's checkpoint prefix
    # (model.layers.<num_hidden_layers>; None = MiMo's model.mtp.layers.0), whether its MLP is the
    # MoE, and index_share_for_mtp_iteration (later draft passes reuse the first pass's DSA selection).
    mtp_prefix: str | None = None
    mtp_moe: bool = False
    mtp_index_share: bool = False
    # Checkpoint names of the MTP head's own weights beside its layer: "mimo" / DeepSeek
    # (enorm, hnorm, eh_proj, the final norm; models/mtp.py names) or "qwen3_5" (mtp.{pre_fc_norm_embedding,
    # pre_fc_norm_hidden, fc, norm}: vLLM v0.30.0 model_executor/models/qwen3_5_mtp.py
    # Qwen3_5MultiTokenPredictor).
    mtp_names: str = "mimo"
    # Checkpoint weight quantization (quantization_config): FP8 with block_size scales for
    # dense weights, optionally MXFP4 experts; `quant_ignored` modules stay unquantized.
    quant_block: int | None = None
    quant_expert_block: int | None = None
    quant_expert_mxfp4: bool = False  # experts are stored MXFP4 (store_dtype: mxfp4)
    quant_ignored: tuple[str, ...] = ()
    # The checkpoint's scales cover blocks of input columns (FP8 weight_block_size, MXFP4), so a
    # tensor-parallel column shard must not split a block. Per-tensor scales (Hy3-FP8) and FP8
    # made at load (quantized_at_load, after sharding) have no such constraint.
    quant_blockwise: bool = False
    o_bias: bool = False  # bias on o_proj (gpt-oss attention_bias)
    # Routed experts: "silu" (SwiGLU) or "swiglu_oai" (gpt-oss's clamped GLU, see
    # decoder.swiglu_oai); expert_bias: gate_up / down biases per expert (gpt-oss).
    moe_act: str = "silu"
    swiglu_limit: float = 7.0
    expert_bias: bool = False
    router_logit_bias: bool = False  # the router linear has a bias added to its logits (gpt-oss)
    # Inkling (see _inkling): the router also scores router_shared_rows shared experts, whose
    # weights scale them; a scalar global scale on the router and on the dense MLPs; RMSNorm of
    # the embeddings; logits of h / logits_divisor; causal depthwise short convolutions of kernel
    # sconv_kernel (with a residual) on k, v, the attention output and the MLP output; relative
    # position logits of d_rel dims; log_scaling = (n_floor, alpha) for log_scaled layers.
    router_shared_rows: int = 0
    router_global_scale: bool = False
    dense_mlp_scale: bool = False
    embed_norm: bool = False
    logits_divisor: float = 1.0
    sconv_kernel: int = 0
    d_rel: int = 0
    log_scaling: tuple[float, float] | None = None
    # Hyper-connection hybrids (GLM-5.3-Flash, Qwen3.8-Flash-Next): their residual streams and
    # per-model extras, a models/hybrid.py HybridSpec.
    hybrid: object | None = None

    def truncated(self, n: int) -> "ModelConfig":
        """The first n layers only (debugging a large checkpoint layer by layer)."""
        import dataclasses

        return dataclasses.replace(
            self, num_layers=n, moe_layers=tuple(i for i in self.moe_layers if i < n),
            attn_layers=self.attn_layers[:n] if self.attn_layers is not None else None)

    def quantized_at_load(self, dense: bool, block: int = 128) -> "ModelConfig":
        """A BF16 checkpoint's experts (and, with dense, its other linears) quantized to FP8 by
        trnRT at load: per (row, block input columns) absmax scales onto e4m3 max 240
        (models/quant.py quantize_fp8_rows). Embeddings, lm_head, norms and routers stay."""
        import dataclasses

        if self.quant_block is not None or self.quant_expert_block is not None:
            raise ValueError("the checkpoint is already quantized; --weight-dtype fp8 is for BF16 checkpoints")
        return dataclasses.replace(self, quant_block=block if dense else None,
                                   quant_expert_block=block if self.num_experts else None)

    def is_quantized(self, module: str, expert: bool = False) -> bool:
        """`module` like "model.layers.3.self_attn.qkv_proj"."""
        if module in self.quant_ignored:
            return False
        return (self.quant_expert_block if expert else self.quant_block) is not None

    @classmethod
    def from_pretrained(cls, path: str) -> "ModelConfig":
        with open(os.path.join(path, "config.json")) as f:
            c = json.load(f)
        eos = c.get("eos_token_id")
        gen_path = os.path.join(path, "generation_config.json")
        if os.path.exists(gen_path):
            with open(gen_path) as f:
                eos = json.load(f).get("eos_token_id", eos)
        eos_ids = tuple(eos) if isinstance(eos, list) else ((eos,) if eos is not None else ())
        return cls.from_config_dict(c, eos_ids)

    @classmethod
    def from_config_dict(cls, c: dict, eos_ids: tuple[int, ...] = ()) -> "ModelConfig":
        """from_pretrained on a config.json already read (an EAGLE-3 draft's embedded layer config,
        models/eagle3.py)."""
        from .models import hybrid, mla

        arch = (c.get("architectures") or [None])[0]
        if arch in mla.ARCHITECTURES:
            return mla.model_config(cls, c, eos_ids)
        if arch in GQA_MOE:  # these parse their own RoPE (gpt-oss: YaRN)
            return GQA_MOE[arch](cls, c, eos_ids)
        from .models import deepseek_v41

        if arch in hybrid.ARCHITECTURES or arch in deepseek_v41.ARCHITECTURES:
            return hybrid.config_from_hf(cls, c, eos_ids)
        # transformers 5 writes rope settings under "rope_parameters"; older files use
        # a top-level "rope_theta".
        rope = dict(c.get("rope_parameters") or {})
        rope_theta = rope.get("rope_theta", c.get("rope_theta", 10000.0))
        scaling = dict(c.get("rope_scaling") or {})
        rope_type = scaling.get("rope_type", scaling.get("type", rope.get("rope_type", "default")))
        if rope_type not in ("default", "llama3"):
            raise NotImplementedError(f"rope scaling {rope_type!r} is not supported yet")
        if rope_type == "llama3":
            scaling = {**rope, **scaling, "rope_type": "llama3"}
        if arch == "MiMoV2ForCausalLM":
            return cls._mimo_v2(c, eos_ids)
        from .models import linear_attn

        if arch in linear_attn.ARCHITECTURES:  # linear-attention hybrids
            return linear_attn.config_from_hf(cls, arch, c, eos_ids)
        known = {
            "LlamaForCausalLM": (False, bool(c.get("attention_bias", False))),
            "MistralForCausalLM": (False, False),
            "Qwen2ForCausalLM": (False, True),
            "Qwen3ForCausalLM": (True, False),
            "Qwen3MoeForCausalLM": (True, False),
        }
        if arch not in known:
            raise NotImplementedError(f"architecture {arch} is not supported (have {sorted([*known, *GQA_MOE])} "
                                      "+ MiMoV2ForCausalLM)")
        qk_norm, qkv_bias = known[arch]
        if c.get("use_sliding_window") or (arch == "MistralForCausalLM" and c.get("sliding_window")):
            raise NotImplementedError("sliding-window attention is only wired for MiMo-V2 so far")
        n_layers = c["num_hidden_layers"]
        # Hub configs say num_experts; transformers 5 writes num_local_experts.
        experts = c.get("num_experts") or c.get("num_local_experts") or 0
        moe_layers = ()
        if arch == "Qwen3MoeForCausalLM" and experts:
            step, dense = c.get("decoder_sparse_step", 1), set(c.get("mlp_only_layers") or [])
            moe_layers = tuple(i for i in range(n_layers) if i not in dense and (i + 1) % step == 0)
        num_heads = c["num_attention_heads"]
        return cls(
            architecture=arch,
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_layers=n_layers,
            num_heads=num_heads,
            num_kv_heads=c.get("num_key_value_heads", num_heads),
            head_dim=c.get("head_dim") or c["hidden_size"] // num_heads,
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=float(rope_theta),
            max_position_embeddings=c["max_position_embeddings"],
            tie_word_embeddings=c.get("tie_word_embeddings", False),
            eos_token_ids=eos_ids,
            qk_norm=qk_norm,
            qkv_bias=qkv_bias,
            rope_scaling=tuple(sorted(scaling.items())) if rope_type == "llama3" else None,
            num_experts=experts,
            num_experts_per_tok=c.get("num_experts_per_tok", 0) or 0,
            moe_intermediate_size=c.get("moe_intermediate_size", 0) or 0,
            norm_topk_prob=c.get("norm_topk_prob", True),
            moe_layers=moe_layers,
            **cls._quant(c),
        )

    @staticmethod
    def _quant(c: dict) -> dict:
        q = c.get("quantization_config") or {}
        if not q:
            return {}
        if q.get("quant_method") == "mxfp4":
            # gpt-oss: experts MXFP4 (`*_blocks` / `*_scales`, one E8M0 exponent per 32), every
            # other module ("model.layers.*.self_attn", "model.layers.*.mlp.router",
            # "model.embed_tokens", "lm_head", modules_to_not_convert in
            # https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json) BF16.
            return dict(quant_expert_block=32, quant_expert_mxfp4=True, quant_blockwise=True)
        if q.get("quant_method") != "fp8" or q.get("fmt", "e4m3") != "e4m3":
            raise NotImplementedError(f"quantization {q.get('quant_method')}/{q.get('fmt')} is not supported")
        if not q.get("weight_block_size"):
            # Per-tensor FP8 (tencent/Hy3-FP8: quant_method fp8, no weight_block_size, a scalar
            # BF16 `weight_scale` per weight; https://huggingface.co/tencent/Hy3-FP8/raw/main/config.json
            # and its shard headers). trnRT keeps per-row scales in 128-column blocks, so that
            # rescaling onto trn2's e4m3 max 240 (fit_e4m3_max) touches only the blocks that need it.
            return dict(quant_block=128, quant_expert_block=128, quant_ignored=tuple(q.get("ignored_layers") or ()))
        bn, bk = q.get("weight_block_size")
        if bn is None or bn != bk:
            raise NotImplementedError("only square FP8 weight blocks are supported")
        expert = q.get("mxfp4_block_size", 32) if q.get("store_dtype") == "mxfp4" else bk
        # GLM-5.3 names the unquantized modules modules_to_not_convert
        # (https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json), MiMo-V2 ignored_layers.
        # Vision-language checkpoints may spell them under model.language_model. (Qwen/Qwen3.8-
        # Flash-Next-FP8 config.json); trnRT names modules model.layers..., as the loader reads them.
        vl = "model.language_model."
        ignored = [m if not m.startswith(vl) else "model." + m[len(vl):]
                   for m in q.get("ignored_layers") or q.get("modules_to_not_convert") or ()]
        return dict(quant_block=int(bk), quant_expert_block=int(expert), quant_blockwise=True,
                    quant_expert_mxfp4=q.get("store_dtype") == "mxfp4", quant_ignored=tuple(ignored))

    @classmethod
    def _mimo_v2(cls, c: dict, eos_ids: tuple[int, ...]) -> "ModelConfig":
        """Xiaomi MiMo-V2 (modeling_mimo_v2.py in the model repo): hybrid full / sliding-window
        attention per hybrid_layer_pattern (1 = sliding), partial RoPE, value scale, attention
        sinks, and DeepSeek-V3-style sigmoid MoE routing with a correction bias."""
        if (c.get("n_group") or 1) != 1 or (c.get("topk_group") or 1) != 1:
            raise NotImplementedError("grouped expert routing (n_group > 1) is not supported yet")
        if c.get("scoring_func", "sigmoid") != "sigmoid" or c.get("topk_method", "noaux_tc") != "noaux_tc":
            raise NotImplementedError("only sigmoid / noaux_tc MiMo-V2 routing is supported")
        prf = float(c.get("partial_rotary_factor", (c.get("rope_parameters") or {}).get("partial_rotary_factor", 1.0)))
        heads, dk = c["num_attention_heads"], c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
        full = AttnSpec(heads, c["num_key_value_heads"], dk, c.get("v_head_dim", dk), int(dk * prf),
                        float(c.get("rope_theta", 10000.0)), None, bool(c.get("add_full_attention_sink_bias")))
        sdk = c.get("swa_head_dim", dk)
        swa = AttnSpec(c.get("swa_num_attention_heads", heads), c.get("swa_num_key_value_heads", c["num_key_value_heads"]),
                       sdk, c.get("swa_v_head_dim", c.get("v_head_dim", sdk)), int(sdk * prf),
                       float(c.get("swa_rope_theta", c.get("rope_theta", 10000.0))), c.get("sliding_window"),
                       bool(c.get("add_swa_attention_sink_bias")))
        n = c["num_hidden_layers"]
        pattern = c.get("hybrid_layer_pattern") or [0] * n
        specs = tuple(swa if pattern[i] == 1 else full for i in range(n))
        freq = c.get("moe_layer_freq") or [0] * n
        experts = c.get("n_routed_experts") or 0
        if c.get("n_shared_experts"):
            raise NotImplementedError("MiMo-V2 shared experts are not supported yet")
        return cls(
            architecture="MiMoV2ForCausalLM", vocab_size=c["vocab_size"], hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"], num_layers=n, num_heads=full.num_heads,
            num_kv_heads=full.num_kv_heads, head_dim=dk, rms_norm_eps=c.get("layernorm_epsilon", c.get("rms_norm_eps", 1e-6)),
            rope_theta=full.rope_theta, max_position_embeddings=c["max_position_embeddings"],
            tie_word_embeddings=c.get("tie_word_embeddings", False), eos_token_ids=eos_ids,
            qkv_bias=bool(c.get("attention_bias", False)), num_experts=experts,
            num_experts_per_tok=c.get("num_experts_per_tok", 0) or 0,
            moe_intermediate_size=c.get("moe_intermediate_size", 0) or 0,
            norm_topk_prob=c.get("norm_topk_prob", True),
            moe_layers=tuple(i for i in range(n) if experts and freq[i]),
            attn_layers=specs, v_scale=c.get("attention_value_scale"), router_scoring="sigmoid",
            router_bias=c.get("topk_method", "noaux_tc") == "noaux_tc",
            routed_scaling_factor=float(c.get("routed_scaling_factor") or 1.0),
            fused_qkv_weights=c.get("attention_projection_layout") == "fused_qkv",
            **cls._quant(c),
            mtp_layers=int(c.get("num_nextn_predict_layers") or 0), mtp_spec=swa
        )

    def kv_bytes_per_token(self, dtype: torch.dtype) -> int:
        """KV bytes per token for a cache stored in `dtype` (linear-attention layers hold none;
        an MLA layer holds its own K and V widths, models/mla.py)."""
        specs = self.attn_layers or ()
        v41 = getattr(getattr(self, "hybrid", None), "v41", None)
        if v41 is not None:  # DeepSeek-V4.1: compressed-latent and index-key PIECES of its kv sources only
            from .models.deepseek_v41 import kv_elements_per_token

            return kv_elements_per_token(v41, self.num_layers) * torch.finfo(dtype).bits // 8
        if any(getattr(s, "mla", None) is not None for s in specs):
            return sum(s.num_kv_heads * (s.head_dim + s.v_head_dim) for s in specs
                       if not isinstance(s, LinearSpec)) * torch.finfo(dtype).bits // 8
        n = self.num_layers - sum(isinstance(s, LinearSpec) for s in specs)
        aux = sum(getattr(s, "aux_kv_width", 0) for s in specs)  # e.g. a QSA indexer key (models/qwen4_exp.py)
        return (2 * n * self.num_kv_heads * self.head_dim + aux) * torch.finfo(dtype).bits // 8


def _rope_theta(c: dict, default: float = 10000.0) -> float:
    return float((c.get("rope_parameters") or {}).get("rope_theta", c.get("rope_theta", default)))


def _gpt_oss(cls, c: dict, eos_ids: tuple[int, ...]) -> ModelConfig:
    """openai/gpt-oss-120b / -20b. Keys from https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json;
    semantics from transformers 5.15 models/gpt_oss/modeling_gpt_oss.py: layer_types alternates
    sliding_attention (window 128) and full_attention, every layer has per-head sinks, q/k/v/o
    carry biases (attention_bias), the router is a biased linear whose top-k logits are
    softmaxed, experts (num_local_experts, intermediate_size each) have gate_up and down biases
    and the clamped GLU (swiglu_limit), and RoPE is YaRN (rope_scaling)."""
    n, heads = c["num_hidden_layers"], c["num_attention_heads"]
    D = c.get("head_dim") or c["hidden_size"] // heads
    theta = _rope_theta(c)
    rs = dict(c.get("rope_scaling") or c.get("rope_parameters") or {})
    if rs.get("rope_type", rs.get("type")) != "yarn":
        raise NotImplementedError(f"gpt-oss with rope {rs.get('rope_type')!r} (expected yarn)")
    rs.pop("rope_theta", None)
    types = c.get("layer_types") or ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(n)]
    specs = tuple(AttnSpec(heads, c["num_key_value_heads"], D, D, D, theta,
                           c["sliding_window"] if t == "sliding_attention" else None, True) for t in types[:n])
    experts = c.get("num_local_experts") or c.get("num_experts")
    return cls(
        architecture="GptOssForCausalLM", vocab_size=c["vocab_size"], hidden_size=c["hidden_size"],
        intermediate_size=c["intermediate_size"], num_layers=n, num_heads=heads,
        num_kv_heads=c["num_key_value_heads"], head_dim=D, rms_norm_eps=c["rms_norm_eps"], rope_theta=theta,
        max_position_embeddings=c["max_position_embeddings"], tie_word_embeddings=c.get("tie_word_embeddings", False),
        eos_token_ids=eos_ids, qkv_bias=bool(c.get("attention_bias", True)), o_bias=bool(c.get("attention_bias", True)),
        rope_scaling=tuple(sorted({**rs, "rope_type": "yarn"}.items())), num_experts=experts,
        num_experts_per_tok=c.get("num_experts_per_tok") or c.get("experts_per_token"),
        moe_intermediate_size=c["intermediate_size"], moe_layers=tuple(range(n)), attn_layers=specs,
        router_scoring="topk_softmax", router_logit_bias=True, expert_bias=True, moe_act="swiglu_oai",
        swiglu_limit=float(c.get("swiglu_limit", 7.0)), **cls._quant(c))


def _hy_v3(cls, c: dict, eos_ids: tuple[int, ...]) -> ModelConfig:
    """tencent/Hy3 and Hy3-FP8. Keys from https://huggingface.co/tencent/Hy3/raw/main/config.json;
    semantics from transformers 5.15 models/hy_v3/modeling_hy_v3.py: GQA with per-head q/k RMSNorm
    before RoPE, the first layer(s) dense (mlp_layer_types, else first_k_dense_replace), the rest
    sigmoid-routed with a selection bias (expert_bias), weights renormalised (+1e-20) and scaled
    by router_scaling_factor, plus num_shared_experts x moe_intermediate_size of shared MLP. The
    MTP layer (num_nextn_predict_layers, stored as model.layers.{num_hidden_layers}) is not loaded."""
    if c.get("attention_bias") or c.get("mlp_bias"):
        raise NotImplementedError("Hy3 with attention_bias / mlp_bias")
    if not c.get("moe_router_use_sigmoid", True) or c.get("hidden_act", "silu") != "silu":
        raise NotImplementedError("Hy3 routing other than sigmoid, or an activation other than silu")
    n, heads = c["num_hidden_layers"], c["num_attention_heads"]
    kinds = c.get("mlp_layer_types") or ["dense"] * c.get("first_k_dense_replace", 1) + ["sparse"] * n
    experts = c.get("num_experts") or c.get("num_local_experts")
    return cls(
        architecture="HYV3ForCausalLM", vocab_size=c["vocab_size"], hidden_size=c["hidden_size"],
        intermediate_size=c["intermediate_size"], num_layers=n, num_heads=heads,
        num_kv_heads=c["num_key_value_heads"], head_dim=c.get("head_dim") or c["hidden_size"] // heads,
        rms_norm_eps=c["rms_norm_eps"], rope_theta=_rope_theta(c, 11158840.0),
        max_position_embeddings=c["max_position_embeddings"], tie_word_embeddings=c.get("tie_word_embeddings", False),
        eos_token_ids=eos_ids, qk_norm=bool(c.get("qk_norm", True)), num_experts=experts,
        num_experts_per_tok=c["num_experts_per_tok"], moe_intermediate_size=c["moe_intermediate_size"],
        norm_topk_prob=bool(c.get("route_norm", True)), moe_layers=tuple(i for i in range(n) if kinds[i] == "sparse"),
        router_scoring="sigmoid", router_bias=bool(c.get("moe_router_enable_expert_bias", True)),
        routed_scaling_factor=float(c.get("router_scaling_factor") or 1.0),
        n_shared_experts=c.get("num_shared_experts") or 0, **cls._quant(c))


def _k2_horizon(cls, c: dict, eos_ids: tuple[int, ...]) -> ModelConfig:
    """IFM/K2-Horizon-375B-A23B. Keys from https://huggingface.co/IFM/K2-Horizon-375B-A23B/raw/main/config.json;
    semantics from that repo's modeling_k2_horizon.py (Qwen3-MoE layout): mlp_only_layers dense,
    sigmoid routing whose gate bias (moe_gate_bias) only steers the top-k choice, renormalised and
    scaled by router_scaling_factor, num_shared_experts x moe_intermediate_size of shared MLP, and
    partial RoPE over rope_head_dim: the head is split into halves and RoPE rotates the first
    rope_head_dim / 2 dims of each half together (split_to_interleaved, then the first
    rope_head_dim interleaved dims), i.e. a full-width table whose upper frequencies are zero."""
    unsupported = {"attention_gate_func": c.get("attention_gate_func") is not None,
                   "mova_num_experts": bool(c.get("mova_num_experts")),
                   "layernorm_num_groups": (c.get("layernorm_num_groups") or 1) != 1,
                   "query_key_norm": bool(c.get("query_key_norm")), "attention_bias": bool(c.get("attention_bias")),
                   "use_sliding_window": bool(c.get("use_sliding_window"))}
    if any(unsupported.values()):
        raise NotImplementedError(f"K2-Horizon options not supported: {[k for k, v in unsupported.items() if v]}")
    if c.get("router_score_func", "softmax") != "sigmoid":
        raise NotImplementedError("K2-Horizon routing other than sigmoid")
    n, heads = c["num_hidden_layers"], c["num_attention_heads"]
    D = c.get("head_dim") or c["hidden_size"] // heads
    rd = c.get("rope_head_dim") or D
    theta = _rope_theta(c)
    spec = AttnSpec(heads, c["num_key_value_heads"], D, D, D, theta, None, False, rd // 2 if rd != D else None)
    step, dense = c.get("decoder_sparse_step", 1), set(c.get("mlp_only_layers") or [])
    experts = c.get("num_experts") or 0
    return cls(
        architecture="K2HorizonForCausalLM", vocab_size=c["vocab_size"], hidden_size=c["hidden_size"],
        intermediate_size=c["intermediate_size"], num_layers=n, num_heads=heads, num_kv_heads=c["num_key_value_heads"],
        head_dim=D, rms_norm_eps=c["rms_norm_eps"], rope_theta=theta,
        max_position_embeddings=c["max_position_embeddings"], tie_word_embeddings=c.get("tie_word_embeddings", False),
        eos_token_ids=eos_ids, num_experts=experts,
        num_experts_per_tok=c["num_experts_per_tok"], moe_intermediate_size=c["moe_intermediate_size"],
        norm_topk_prob=bool(c.get("norm_topk_prob", False)),
        moe_layers=tuple(i for i in range(n) if experts and i not in dense and (i + 1) % step == 0),
        attn_layers=(spec,) * n, router_scoring="sigmoid", router_bias=bool(c.get("moe_gate_bias")),
        routed_scaling_factor=float(c.get("router_scaling_factor") or 1.0),
        n_shared_experts=c.get("num_shared_experts") or 0, **cls._quant(c))


def _inkling(cls, c: dict, eos_ids: tuple[int, ...]) -> ModelConfig:
    """thinkingmachines/Inkling-Small and Inkling, text model only. Keys from text_config in
    https://huggingface.co/thinkingmachines/Inkling-Small/raw/main/config.json; semantics from
    transformers 5.15 models/inkling/modeling_inkling.py: no RoPE; per-head q / k RMSNorm with
    1 / head_dim score scaling; a relative position bias from r_proj(x) @ rel_logits_proj over
    key distances below rel_extent (sliding_window_size on local layers); log scaling of queries
    and that bias on global layers; causal depthwise short convolutions (kernel sconv_kernel_size,
    plus the input) on k and v before k_norm and on the attention and MLP outputs; layers
    local_layer_ids slide over sliding_window_size keys; the first dense_mlp_idx layers dense;
    sigmoid routing with a selection bias, where the top-k and the n_shared_experts shared
    experts' weights are sigmoid(logit) normalised over all of them, times route_scale and the
    gate's global_scale; embeddings RMS-normalised; logits of h / logits_mup_width_multiplier
    over the first unpadded_vocab_size rows. The MTP layers (mtp_config) are not loaded.

    The experts' intermediate size is the text config's intermediate_size when
    dense_intermediate_size is given: Inkling-Small's experts.w2_weight is [256, 4096, 2048]
    (shard headers), while transformers 5.15's InklingTextConfig keeps moe_intermediate_size at
    its default 3072 in that case (it matches the large Inkling's 3072 only by coincidence)."""
    t = c["text_config"]
    unsupported = {"q_bias": bool(t.get("q_bias")), "o_bias": bool(t.get("o_bias")),
                   "final_logit_softcapping": t.get("final_logit_softcapping") is not None,
                   "gate_activation": t.get("gate_activation", "sigmoid") != "sigmoid",
                   "norm_after_topk": not t.get("norm_after_topk", True)}
    if any(unsupported.values()):
        raise NotImplementedError(f"Inkling options not supported: {[k for k, v in unsupported.items() if v]}")
    n, H = t["num_hidden_layers"], t["hidden_size"]
    local = set(t.get("local_layer_ids") if t.get("local_layer_ids") is not None
                else [i for i in range(n) if (i + 1) % 6])
    log = None
    if t.get("log_scaling_n_floor"):
        log = (float(t["log_scaling_n_floor"]), float(t.get("log_scaling_alpha", 0.1)))
    w = t.get("sliding_window_size", 512)
    D, sD = t.get("head_dim", 128), t.get("swa_head_dim", t.get("head_dim", 128))
    glob = AttnSpec(t["num_attention_heads"], t["num_key_value_heads"], D, D, 0, 0.0, None, False, None, 1.0 / D,
                    t.get("rel_extent", 1024), log is not None)
    swa = AttnSpec(t.get("swa_num_attention_heads", t["num_attention_heads"]),
                   t.get("swa_num_key_value_heads", t["num_key_value_heads"]), sD, sD, 0, 0.0, w, False, None, 1.0 / sD,
                   w, False)
    dense_I = t.get("dense_intermediate_size")
    moe_I = t.get("moe_intermediate_size") or (t["intermediate_size"] if dense_I else 3072)
    dense_n = t.get("dense_mlp_idx", 0)
    shared = t.get("n_shared_experts") or 0
    experts = t.get("n_routed_experts") or t.get("num_local_experts")
    V = t.get("unpadded_vocab_size") or t["vocab_size"]
    eos = c.get("eos_token_id", t.get("eos_token_id"))
    return cls(
        architecture="InklingForConditionalGeneration", vocab_size=V, hidden_size=H,
        intermediate_size=dense_I or t["intermediate_size"], num_layers=n, num_heads=glob.num_heads,
        num_kv_heads=glob.num_kv_heads, head_dim=D, rms_norm_eps=t.get("rms_norm_eps", 1e-6), rope_theta=0.0,
        max_position_embeddings=t.get("model_max_length", t.get("max_position_embeddings", 131072)),
        tie_word_embeddings=False, eos_token_ids=eos_ids or ((eos,) if isinstance(eos, int) else tuple(eos or ())),
        qk_norm=True, num_experts=experts, num_experts_per_tok=t["num_experts_per_tok"], moe_intermediate_size=moe_I,
        moe_layers=tuple(range(dense_n, n)), attn_layers=tuple(swa if i in local else glob for i in range(n)),
        router_scoring="sigmoid_shared", router_bias=bool(t.get("use_gate_bias", True)),
        routed_scaling_factor=float(t.get("route_scale", 1.0)), n_shared_experts=shared,
        router_shared_rows=shared, router_global_scale=bool(t.get("use_global_scale", True)),
        dense_mlp_scale=bool(t.get("use_global_scale", True)), embed_norm=bool(t.get("use_embed_norm", False)),
        logits_divisor=float(t.get("logits_mup_width_multiplier") or 1.0),
        sconv_kernel=t.get("sconv_kernel_size", 4) if t.get("use_sconv", True) else 0, d_rel=t.get("d_rel", 16),
        log_scaling=log)


# GQA + MoE architectures beyond Qwen3-MoE and MiMo-V2, parsed by their own functions.
GQA_MOE = {"GptOssForCausalLM": _gpt_oss, "HYV3ForCausalLM": _hy_v3, "K2HorizonForCausalLM": _k2_horizon,
           "InklingForConditionalGeneration": _inkling}


def _geo_ladder(lo: int, hi: int, ratio: float = 1.5) -> tuple[int, ...]:
    """lo, ~lo*ratio, ... hi, as distinct ints. With ratio 1.5 a step reads at most 1.5x the
    KV its longest sequence needs (a power-of-two ladder allows 2x)."""
    out, b = [], float(lo)
    while round(b) < hi:
        out.append(int(round(b)))
        b *= ratio
    out.append(hi)
    return tuple(sorted(set(out)))


def _pow2_ladder(lo: int, hi: int) -> tuple[int, ...]:
    out, b = [], lo
    while b < hi:
        out.append(b)
        b *= 2
    out.append(hi)
    return tuple(sorted(set(out)))


@dataclass
class EngineConfig:
    """One engine: a model on `tp` ranks, its KV / state pools, and the static shapes its graphs are compiled at.

    The engine always runs the served configuration: overlap scheduling, piecewise layer graphs with each decode call
    compiled whole, the radix prefix cache with state checkpoints for the linear-attention layers (at every page
    boundary in decode, inside prefill chunks, and at a finished request's end), and FCFS scheduling with reserve
    admission. Only the shapes and sizes below vary."""
    model_path: str
    device: str = "cpu"  # "cpu" or "neuron"
    dtype: torch.dtype = torch.bfloat16
    # "auto" stores KV in the model dtype; "fp8" in float8 e4m3 (vLLM / SGLang --kv-cache-dtype fp8), halving the KV
    # bytes every decode step reads.
    kv_cache_dtype: str = "auto"
    page_size: int = 32
    num_pages: int | None = None  # derived from kv_cache_gb when None
    kv_cache_gb: float = 4.0  # per rank
    max_num_seqs: int = 8
    max_model_len: int = 4096
    max_prefill_tokens: int = 512  # prefill tokens per step (chunk budget)
    sampling_candidates: int = 64  # top-k candidate set the on-device sampler draws from
    # Static-shape buckets. None derives power-of-two ladders from the limits above.
    decode_batch_buckets: tuple[int, ...] | None = None
    prefill_token_buckets: tuple[int, ...] | None = None
    page_buckets: tuple[int, ...] | None = None
    # The page buckets of prefill calls, apart from decode's: such a call takes the smallest rung covering its context.
    # Under context-parallel DSA the long path's selection scans every pool of the bucket, so a short prefill in a large
    # bucket pays for keys it never reads. Each rung is one more set of prefill graphs. The largest rung must be
    # page_buckets' largest. None: page_buckets.
    prefill_page_buckets: tuple[int, ...] | None = None
    # The smaller prefill page buckets only for calls of at least this many rows (a prefill bucket of n or more);
    # smaller calls take the largest page bucket, so their graphs are not duplicated. 0: every prefill call takes every
    # prefill page bucket.
    prefill_page_min_rows: int = 0
    seed: int = 0
    # Reserve admission (engine/scheduler.py): a request is admitted only when the free and evictable pages cover its
    # whole prompt and the pages the running requests still need (their prompts plus up to this many decode tokens
    # each). A KV-short pool then runs fewer requests instead of admitting prompts it must later preempt.
    admission_decode_tokens: int = 256
    # Host memory under the radix cache, GiB per rank (engine/host_cache.py): half for state checkpoints, the rest for
    # KV pages. 0: off.
    host_cache_gb: float = 0.0
    # Rows of the state pool kept for state checkpoints (None: 2 x max_num_seqs).
    state_checkpoints: int | None = None
    tp: int = 1  # tensor-parallel ranks, one process and one logical NeuronCore each
    tp_core_base: int = 0  # first logical NeuronCore of this engine (a second engine on the host offsets it)
    # Selected-expert MoE: "nki" (trnrt/kernels) or "xla" (the gather in DecoderForCausalLM._moe). None: "nki" on a
    # NeuronCore and in a capture of one (device "meta"), the served path; "xla" on CPU, where NKI kernels do not run.
    moe_kernel: str | None = None
    # Layers per piecewise graph, MoE layers included, for decode and for prefill chunks.
    piecewise_moe_group: int = 12
    piecewise_prefill_moe_group: int = 12
    num_layers: int | None = None  # debugging: run only the first n layers of the checkpoint

    def resolved_moe_kernel(self) -> str:
        return self.moe_kernel or ("nki" if self.device in ("neuron", "meta") else "xla")

    @property
    def max_pages_per_seq(self) -> int:
        return -(-self.max_model_len // self.page_size)

    def resolved_decode_batch_buckets(self) -> tuple[int, ...]:
        return self.decode_batch_buckets or _pow2_ladder(1, self.max_num_seqs)

    def resolved_prefill_token_buckets(self) -> tuple[int, ...]:
        n = self.max_prefill_tokens
        return self.prefill_token_buckets or _pow2_ladder(min(32, n), n)

    def resolved_page_buckets(self) -> tuple[int, ...]:
        return self.page_buckets or _geo_ladder(min(4, self.max_pages_per_seq), self.max_pages_per_seq)

    def resolved_prefill_page_buckets(self) -> tuple[int, ...]:
        """prefill_page_buckets sorted, or the page buckets. Its largest is theirs: every context has a rung, and DSA's
        selection scratch is sized by the largest page bucket."""
        pages = self.resolved_page_buckets()
        if not self.prefill_page_buckets:
            return pages
        rungs = tuple(sorted(set(self.prefill_page_buckets)))
        if rungs[0] < 1 or rungs[-1] != max(pages):
            raise ValueError(f"prefill_page_buckets={self.prefill_page_buckets}: positive, the largest the page "
                             f"buckets' largest ({max(pages)})")
        return rungs


def prefill_pages(rungs: tuple[int, ...], min_rows: int, rows: int) -> tuple[int, ...]:
    """The page buckets a prefill (or mixed) call of a prefill bucket of `rows` rows may take: every one of `rungs`
    (ascending) from min_rows rows on (EngineConfig.prefill_page_min_rows; 0: always), else only the largest."""
    return rungs if not min_rows or rows >= min_rows else rungs[-1:]


def pick_bucket(n: int, ladder: tuple[int, ...]) -> int:
    for b in ladder:
        if b >= n:
            return b
    raise ValueError(f"{n} exceeds the largest bucket {ladder[-1]}")
