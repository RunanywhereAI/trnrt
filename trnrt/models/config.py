"""The model configuration: the subset of GLM-5.3-Flash's config.json the model code reads."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttnSpec:
    """One softmax-attention layer's shape. GLM-5.3-Flash's are pooled-DSA MLA layers (models/mla.py
    MLASpec in `mla`): num_kv_heads, head_dim and v_head_dim then describe the layer's paged cache, one
    latent "head" whose K is the kv_lora_rank latent and whose V holds the indexer key and the pool
    gate logits."""

    num_heads: int
    num_kv_heads: int
    head_dim: int  # query / key
    v_head_dim: int
    rope_dim: int  # leading dims of q / k that RoPE rotates (0: NoPE)
    rope_theta: float
    scale: float | None = None  # score scale (None: head_dim ** -0.5)
    mla: object | None = None


@dataclass(frozen=True)
class LinearSpec:
    """One linear-attention layer (models/linear_attn.py): a short causal conv over q, k, v and a
    gated delta rule whose state is a fixed-size matrix per head instead of a KV cache.

    kind "kda": Kimi Delta Attention, one decay per key channel, q, k and v heads all equal."""

    kind: str
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel: int
    gate_act: str = "silu"  # activation of the output gate in the gated RMSNorm
    gate_rank: int = 0  # output gate: 0 = full-rank g_proj, else g_a_proj / g_b_proj of this rank
    lower_bound: float | None = None  # "safe gate": decay = lower_bound * sigmoid(...)


@dataclass(frozen=True)
class ModelConfig:
    """The subset of a Hugging Face config.json that the model code reads."""

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
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True
    moe_layers: tuple[int, ...] = ()  # indices of layers whose MLP is a sparse MoE
    # Per-layer token mixer: AttnSpec (attention over the paged KV cache) or LinearSpec (recurrent state).
    attn_layers: tuple[AttnSpec | LinearSpec, ...] | None = None
    # Sigmoid routing with a selection bias: the experts are chosen on score + e_score_correction_bias and
    # weighted by the uncorrected score.
    router_scoring: str = "sigmoid"
    router_bias: bool = False  # e_score_correction_bias added for expert choice only
    routed_scaling_factor: float = 1.0
    n_shared_experts: int = 0  # a dense MLP of n x moe_intermediate_size beside the routed experts
    n_group: int = 1
    topk_group: int = 1
    # Checkpoint weight quantization (quantization_config): FP8 with 128 x 128 block scales; `quant_ignored`
    # modules stay unquantized. The scales cover blocks of input columns, so a tensor-parallel column shard
    # must not split a block (quant_blockwise).
    quant_block: int | None = None
    quant_expert_block: int | None = None
    quant_ignored: tuple[str, ...] = ()
    quant_blockwise: bool = False
    router_logit_bias: bool = False  # the router linear has no bias
    # The hyper-connection residual streams and the model's extras, a models/hybrid.py HybridSpec.
    hybrid: object | None = None
    # Multi-token prediction (models/mtp.py): the checkpoint's MTP layers (num_nextn_predict_layers; the first one is
    # applied recursively for every draft), the attention spec of that layer, its checkpoint prefix
    # (model.layers.<num_hidden_layers>) and whether its MLP is the MoE.
    mtp_layers: int = 0
    mtp_spec: AttnSpec | None = None
    mtp_prefix: str | None = None
    mtp_moe: bool = False

    def truncated(self, n: int) -> "ModelConfig":
        """The first n layers only (debugging a large checkpoint layer by layer)."""
        import dataclasses

        return dataclasses.replace(
            self, num_layers=n, moe_layers=tuple(i for i in self.moe_layers if i < n),
            attn_layers=self.attn_layers[:n] if self.attn_layers is not None else None)

    def is_quantized(self, module: str, expert: bool = False) -> bool:
        """`module` like "model.layers.3.mlp.shared_experts.gate_proj"."""
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
        """from_pretrained on a config.json already read."""
        from . import glm5_next

        arch = (c.get("architectures") or [None])[0]
        if arch not in glm5_next.ARCHITECTURES:
            raise NotImplementedError(f"architecture {arch} is not supported (have {list(glm5_next.ARCHITECTURES)})")
        return glm5_next.config_from_hf(cls, c, eos_ids)

    @staticmethod
    def _quant(c: dict) -> dict:
        q = c.get("quantization_config") or {}
        if not q:
            return {}
        if q.get("quant_method") != "fp8" or q.get("fmt", "e4m3") != "e4m3" or not q.get("weight_block_size"):
            raise NotImplementedError(f"quantization {q.get('quant_method')}/{q.get('fmt')} is not supported "
                                      "(block-scaled FP8 e4m3 only)")
        bn, bk = q.get("weight_block_size")
        if bn is None or bn != bk:
            raise NotImplementedError("only square FP8 weight blocks are supported")
        if q.get("store_dtype") == "mxfp4":
            raise NotImplementedError("MXFP4 experts are not supported")
        # The unquantized modules are named modules_to_not_convert (or ignored_layers), possibly under
        # model.language_model.; the loader reads them as model.layers....
        vl = "model.language_model."
        ignored = [m if not m.startswith(vl) else "model." + m[len(vl):]
                   for m in q.get("ignored_layers") or q.get("modules_to_not_convert") or ()]
        return dict(quant_block=int(bk), quant_expert_block=int(bk), quant_blockwise=True, quant_ignored=tuple(ignored))

    def kv_bytes_per_token(self, dtype: torch.dtype) -> int:
        """KV bytes per token for a cache stored in `dtype`: each attention layer's latent K and its V
        (linear-attention layers hold none)."""
        specs = self.attn_layers or ()
        return sum(s.num_kv_heads * (s.head_dim + s.v_head_dim) for s in specs
                   if not isinstance(s, LinearSpec)) * torch.finfo(dtype).bits // 8
