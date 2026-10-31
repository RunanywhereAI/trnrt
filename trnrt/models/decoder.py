"""GLM-5.3-Flash's decoder for static-shape execution over a paged KV cache.

Numerics follow Hugging Face transformers' modeling code so the CPU path can be checked against it
directly: RMSNorm in fp32 then cast back, softmax in fp32, MoE routing in fp32.

Every layer is a hyper-connection layer (models/hybrid.py) whose token mixer is described PER LAYER
by its spec: Kimi Delta Attention (a LinearSpec, models/linear_attn.py) or pooled-DSA MLA (an AttnSpec
with an MLASpec, models/mla.py).

Graph entry points, each compiled once per bucket:
- `forward_decode`: B sequences with one query token each, context up to P pages, as one graph.
- `forward_prefill`: one sequence's chunk of C tokens, with up to P pages of context (cached prefix
  included), so a radix-cache hit skips straight to the uncached tail. Served piecewise: prep_prefill,
  one graph per run of layers (group_fn), post_prefill.
Both write new KV into the paged cache in place and return packed sampler output, so a step reads back
a few floats per sequence instead of a vocab-sized row.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..engine.sampler import NUM_TOP_LOGPROBS, logsumexp_large, sample, sample_candidates, topk_large
from . import linear_attn
from .config import AttnSpec, LinearSpec, ModelConfig
from .quant import FP8, dequant, dequant_t

NEG_INF = -1e30
# KV gather granularity: below this many tokens a cache is gathered by token slots, from it on by whole
# pages. neuronx-cc lowers the two differently per shape: on trn1 at B=6 decode, token 8.80 / 14.59 /
# 21.99 ms and page 10.65 / 21.57 / 17.39 ms at 128 / 512 / 2048 context.
PAGE_GATHER_MIN_TOKENS = 2048


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dtype)


def moe_inter_per_rank(cfg: ModelConfig, tp: int, quantized: bool) -> int:
    """Expert intermediate size on one tensor-parallel rank. A block-FP8 shard that sits inside one scale
    block takes that block's scale (the loader's concern), so the shard needs no padding."""
    I = cfg.moe_intermediate_size
    if not cfg.num_experts:
        return 0
    if I % tp:
        raise ValueError(f"tp={tp} does not divide moe_intermediate_size {I}")
    return I // tp


def kv_heads_per_rank(num_kv_heads: int, tp: int) -> int:
    """KV heads split across ranks, or, with more ranks than KV heads, replicated so each
    rank holds the ONE head its query heads attend to (an MLA latent is one head, replicated)."""
    return num_kv_heads // tp if tp <= num_kv_heads else 1


def layer_specs(cfg: ModelConfig) -> tuple[AttnSpec | LinearSpec, ...]:
    return cfg.attn_layers


def heads_fit(spec, atp: int) -> bool:
    """Whether a token mixer's heads split over `atp` attention ranks: linear attention by k and v heads,
    MLA by query heads (its latent is replicated)."""
    if isinstance(spec, LinearSpec):
        return spec.num_k_heads % atp == 0 and spec.num_v_heads % atp == 0
    return spec.num_heads % atp == 0


def attention_tp(cfg: ModelConfig, tp: int, requested: int | None = None) -> int:
    """The tensor-parallel degree of the token mixers: `requested` when given (it must divide tp and fit
    every mixer's heads), else the largest divisor of tp that every mixer of the model can split."""
    specs = list(layer_specs(cfg))
    if requested is not None:
        bad = [s for s in specs if not heads_fit(s, requested)]
        if requested < 1 or tp % requested or bad:
            why = f"does not divide tp={tp}" if requested < 1 or tp % requested else f"does not fit {_heads(bad[0])}"
            raise ValueError(f"attention_tp={requested} {why}")
        return requested
    return max(a for a in range(1, tp + 1) if tp % a == 0 and all(heads_fit(s, a) for s in specs))


def _heads(spec) -> str:
    if isinstance(spec, LinearSpec):
        return f"{spec.num_k_heads} k / {spec.num_v_heads} v linear-attention heads"
    return f"{spec.num_heads} MLA query heads"


def state_bytes_per_token_rank(cfg: ModelConfig, atp: int, dtype: torch.dtype, kv_fp8: bool = False) -> int:
    """Token-slot state bytes one token occupies on ONE rank (DecoderForCausalLM.token_state_shapes): the
    pooled DSA indexer's separate pool-key cache (models/mla.py pool_key_width)."""
    b = torch.finfo(dtype).bits // 8
    return sum(_mla.pool_key_width(s, kv_fp8) for s in layer_specs(cfg)) * b


# Prefill chunks of a packed MoE layer from this many tokens on run kernels/moe_prefill.py (a grouped GEMM
# over the whole chunk) instead of kernels/moe_dedupe.py's 128-token calls (models/hybrid.py _moe_clamped).
MOE_PREFILL_MIN_TOKENS = 512


# Sequence-parallel residual streams for prefill: between the layers of a prefill chunk every
# tensor-parallel rank holds only its own 1 / tp of the rows of the [R, hc * H] streams instead of all
# of them, so the hyper-connection arithmetic (models/hybrid.py, fp32 over [R, hc, H]: replicated on
# every rank otherwise) runs on R / tp rows. Each block's normalised input is gathered to every rank by
# a zero-padded world all-reduce (_sp_gather; all_gather_into_tensor NEFFs do not reload in a later
# process, see _head), the block runs as before over all R rows, and each rank keeps its rows of the
# reduced output (_sp_rows). Measured at a 1024-row prefill: the collapse and the output mix were 0.95
# and 1.8 ms per block, twice per layer. Same per-row arithmetic either way; decode keeps the
# replicated streams. On trn2 the streams' Veltkamp rounding is bit-identical to the host's at 1 to 1024
# rows, and wikitext-2 scores -0.549 with them against -0.552 without.
#
# Sampling on each rank's vocabulary shard (DecoderForCausalLM._sample_vp): with the lm_head
# vocab-parallel, the decode and prefill graphs sample from each rank's top candidates and log-sum-exp
# instead of gathering the logits over the whole vocabulary first. At tp 32 that gather is bf16
# [2, 154,880] (~620 KB) at two decode rows and the sampler then reads it in >= 3 full passes from HBM:
# 0.44 ms at B1, 1.70 ms at B2 from the gather to the step's end. Each rank's top max(K, N) and log-sum-exp
# instead: B x tp x (1 + 2 max(K, N)) floats (~33 KB at B2). Exact: the same candidates, token and top N
# as sampler.sample (sampler.sample_candidates); the log-sum-exp is combined from the shards', so a
# logprob may differ in its last fp32 bits.


def moe_prefill_down(layer):
    """The prefill kernel's per-column down scales of a layer (DecoderLayer.pack_experts), or None."""
    dsc = getattr(layer, "moe_prefill_dsc", None)
    return None if dsc is None else (dsc, layer.moe_prefill_dfr)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, spec: AttnSpec | LinearSpec, dtype: torch.dtype, tp: int, tp_rank: int,
                 moe: bool, index: int = 0, keep_fp8: bool = False, prefix: str | None = None,
                 moe_kernel: str = "xla", attn_tp: int | None = None, attn_rank: int | None = None):
        """tp / tp_rank shard the MLP (dense or experts); attn_tp / attn_rank (default: the same) shard the
        token mixer's heads. keep_fp8: weights the checkpoint quantizes stay FP8 (with fp32 block scales, see
        models/quant.py) and are dequantized inside the graph right before their matmul. moe_kernel "nki":
        the experts are repacked into w_blob after loading (pack_experts) for kernels/moe_dedupe.py and
        kernels/moe_prefill.py; "xla" (the CPU) keeps them as loaded."""
        super().__init__()
        self.spec = spec
        H = cfg.hidden_size

        def p(*shape):
            return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

        def lin(name: str, module: str, *shape: int, expert: bool = False, transposed: bool = False):
            """A weight [..., out, in], FP8 + scale when the checkpoint quantizes it.
            transposed: stored [..., in, out] (scale [..., in / bk, out]); see w_down below."""
            quant = keep_fp8 and cfg.is_quantized(f"{prefix or f'model.layers.{index}'}.{module}", expert)
            if quant:
                bk = cfg.quant_expert_block if expert else cfg.quant_block
                *lead, n_out, n_in = shape
                nb = -(-n_in // bk)
                setattr(self, name, nn.Parameter(torch.empty(*lead, *((n_in, n_out) if transposed else (n_out, n_in)),
                                                             dtype=FP8), requires_grad=False))
                setattr(self, name + "_scale", nn.Parameter(
                    torch.empty(*lead, *((nb, n_out) if transposed else (n_out, nb)), dtype=torch.float32),
                    requires_grad=False))
            else:
                *lead, n_out, n_in = shape
                setattr(self, name, p(*lead, *((n_in, n_out) if transposed else (n_out, n_in))))
                setattr(self, name + "_scale", None)

        atp = tp if attn_tp is None else attn_tp
        if not heads_fit(spec, atp):
            raise ValueError(f"attention tp={atp} does not fit {_heads(spec)}")
        self.in_norm = p(H)
        if isinstance(spec, LinearSpec):  # recurrent-state token mixer (models/linear_attn.py)
            linear_attn.init_mixer(self, cfg, spec, atp, p, lin)
        else:  # multi-head latent attention (models/mla.py)
            _mla.init_layer(self, cfg, spec, p, lin, atp)
        self.post_norm = p(H)
        self.moe = moe
        self.moe_blob = False  # experts packed into w_blob (pack_experts)
        self.moe_tiles = False  # ... in kernels/moe_dedupe.py's tile-scale layout
        self.pack_moe = False  # the loader packs them (an NKI moe_kernel)
        self.pack_tiles = False  # ... into the tile-scale layout
        if moe:
            E = cfg.num_experts
            Im = moe_inter_per_rank(cfg, tp, keep_fp8)
            # Shared experts: a dense MLP beside the routed experts (intermediate moe_intermediate_size *
            # n_shared_experts), sharded like the dense MLP.
            if (cfg.moe_intermediate_size * cfg.n_shared_experts) % tp:
                raise ValueError(f"tp={tp} does not divide the shared experts' intermediate size")
            Is = cfg.moe_intermediate_size * cfg.n_shared_experts // tp
            if Is:
                lin("shared_gate_up", "mlp.shared_experts.gate_proj", 2 * Is, H)
                lin("shared_down", "mlp.shared_experts.down_proj", H, Is)
            else:
                self.shared_gate_up = self.shared_gate_up_scale = self.shared_down = self.shared_down_scale = None
            self.router = p(E, H)
            self.router_bias = (nn.Parameter(torch.empty(E, dtype=torch.float32), requires_grad=False)
                                if cfg.router_bias else None)
            lin("w_gu", "mlp.experts", E, 2 * Im, H, expert=True)  # per expert: gate rows, then up
            # Stored [E, Im, H] (input dim first): dequantizing [E, H, Im] with its 64-wide inner dim took
            # 11.4 ms for 32 experts on trn1 against 0.35 ms transposed.
            self.down_t = True
            lin("w_down", "mlp.experts", E, H, Im, expert=True, transposed=self.down_t)
            if moe_kernel == "nki":
                from ..kernels import moe_dedupe

                if not moe_dedupe.supports(self.w_gu, self.w_gu_scale, self.w_down, self.w_down_scale, self.down_t):
                    raise ValueError(f"layer {index}: the NKI MoE kernels need FP8 experts with fp32 128 x 128 block "
                                     f"scales, 2 x moe_intermediate / tp = 128 and w_down stored [E, Im, H]; have "
                                     f"w_gu {tuple(self.w_gu.shape)} {self.w_gu.dtype}, down_t={self.down_t}")
                self.pack_tiles = True
                self.pack_moe = True
                from ..kernels import moe_half

                moe_half.register(self, cfg)  # the decode FFN half's packed blobs, decode-only
            elif moe_kernel != "xla":
                raise ValueError(f"moe_kernel must be xla or nki, not {moe_kernel!r}")
        else:
            I = cfg.intermediate_size // tp
            lin("gate_up", "mlp.gate_proj", 2 * I, H)  # gate_proj, up_proj fused
            lin("down", "mlp.down_proj", H, I)
            from ..kernels import dense_half

            dense_half.register(self, cfg, I)  # the dense FFN half's blob, decode-only
        self.k_cache: torch.Tensor | None = None  # [num_pages * page_size, nkv, Dk]
        self.v_cache: torch.Tensor | None = None  # [num_pages * page_size, nkv, Dv]
        _hybrid.init_layer(self, cfg, spec, index, tp, p)  # the hyper-connection streams' parameters

    def pack_experts(self) -> None:
        """Replace w_gu, w_gu_scale, w_down, w_down_scale (filled, on the host) by w_blob, the layout of
        kernels/moe_dedupe.py; the XLA paths read it back through that module's unpack."""
        from ..kernels import moe_dedupe

        H = self.w_gu.shape[-1]
        self.w_blob = nn.Parameter(moe_dedupe.pack(self.w_gu.data, self.w_gu_scale.data, self.w_down.data,
                                                   self.w_down_scale.data), requires_grad=False)
        for n in ("w_gu", "w_gu_scale", "w_down", "w_down_scale"):
            setattr(self, n, None)
        self.moe_blob = True
        self.moe_tiles = self.pack_tiles
        from ..kernels.moe_prefill import check_blob, down_factors

        # True: its dequantize-first path. down_factors: the down scales per output column as a chunk
        # scale times a power of two (fp32 scales that fit_e4m3_max doubled per row), or None; it raises
        # for a layout the kernel cannot take.
        self.moe_prefill_dq = check_blob(self.w_blob.data, H)
        down = down_factors(self.w_blob.data, H)
        self.moe_prefill_dsc, self.moe_prefill_dfr = (
            (None, None) if down is None else (nn.Parameter(t, requires_grad=False) for t in down))


class DecoderForCausalLM(nn.Module):
    """Tensor parallelism is Megatron-style over `tp` ranks: the token mixers' and the MLP's input
    projections (each expert's, for MoE layers) are split on their output dim (heads, intermediate), the
    output projections on their input dim followed by an all-reduce. The final norm is replicated. With
    vocab_parallel (vLLM VocabParallelEmbedding / ParallelLMHead) each rank holds 1 / tp of the
    vocabulary rows of the embedding and lm_head: a lookup is a masked local lookup plus an all-reduce,
    and the logits are the local shard gathered, which equals the replicated matmul column for column.
    Every rank samples the same tokens from the same noise and only rank 0's output is read.

    The token mixers run at the attention TP (attention_tp), which is the TP itself for GLM-5.3-Flash:
    64 DSA query heads and 64 KDA heads split over the tp ranks.

    Context parallelism of the DSA caches (models/dsa_long.py): each rank holds 1 / cp of every
    sequence's latent, indexer rows and pool keys, the attention group split into cp_rows row groups of
    cp consecutive ranks (models/mla.py attention_cp_rows)."""

    def __init__(self, cfg: ModelConfig, dtype: torch.dtype, max_positions: int,
                 tp_rank: int = 0, tp_size: int = 1, tp_group=None, keep_fp8: bool = False,
                 vocab_parallel: bool = False, moe_kernel: str = "xla", attn_tp: int | None = None, attn_group=None):
        """attn_tp: the attention TP (None: attention_tp's default), which must equal tp_size; attn_group
        is then the tensor-parallel group."""
        super().__init__()
        self.keep_fp8 = keep_fp8
        if cfg.intermediate_size % tp_size:
            raise ValueError(f"tp={tp_size} does not divide the MLP intermediate size")
        self.cfg = cfg
        self.dtype = dtype
        self.tp_rank, self.tp_size, self.tp_group = tp_rank, tp_size, tp_group
        self.attn_tp = attention_tp(cfg, tp_size, attn_tp)
        if self.attn_tp != tp_size:
            raise NotImplementedError(f"attention TP {self.attn_tp} below tp={tp_size}")
        self.attn_rank = tp_rank
        self.attn_group = tp_group
        self._sp_rs = False  # set while a sequence-parallel block is traced (_sp_rs_call): see _out_reduce
        # Sequence-parallel prefill streams (see above).
        self.prefill_sp = tp_size > 1
        self.rank_buffers()
        self.moe_inter = moe_inter_per_rank(cfg, tp_size, keep_fp8)
        specs = layer_specs(cfg)
        self.vocab_parallel = vocab_parallel and tp_size > 1
        V = cfg.vocab_size
        self.vocab_rows = -(-V // tp_size) if self.vocab_parallel else V  # rows on this rank
        self.vocab_start = tp_rank * self.vocab_rows if self.vocab_parallel else 0
        self.embed = nn.Parameter(torch.empty(self.vocab_rows, cfg.hidden_size, dtype=dtype), requires_grad=False)
        # A tensor, not a Python int, so that every rank traces the SAME graph: a per-rank
        # constant gave each of 32 ranks its own compile cache key (32 compiles of every graph).
        self.register_buffer("vocab_start_t", torch.tensor(self.vocab_start, dtype=torch.int64), persistent=False)
        self.layers = nn.ModuleList(DecoderLayer(cfg, specs[i], dtype, tp_size, tp_rank, moe=i in cfg.moe_layers,
                                                 index=i, keep_fp8=keep_fp8, moe_kernel=moe_kernel,
                                                 attn_tp=self.attn_tp, attn_rank=self.attn_rank)
                                    for i in range(cfg.num_layers))
        self.norm = nn.Parameter(torch.empty(cfg.hidden_size, dtype=dtype), requires_grad=False)
        self.lm_head = (
            None if cfg.tie_word_embeddings
            else nn.Parameter(torch.empty(self.vocab_rows, cfg.hidden_size, dtype=dtype), requires_grad=False)
        )
        self.window = None  # no sliding-window layers
        self.page_size = 0
        self.fp8_max = None
        # Long-context pooled DSA (models/dsa_long.py): every attention layer is a pooled DSA indexer layer
        # without RoPE, so a bucket past dsa_long.LONG_KEYS keys runs every layer's long path and the prep
        # graphs build no [C, L] visibility for it (long_ctx).
        self.long_dsa = bool(self.layers) and all(
            isinstance(l.spec, LinearSpec) or _mla.long_capable(l.spec) for l in self.layers) and any(
            not isinstance(l.spec, LinearSpec) for l in self.layers)
        # Context parallelism of the DSA caches over the attention group (models/dsa_long.py): each rank
        # holds 1 / cp of every sequence's latent, indexer rows and pool keys, and every bucket runs the long
        # path. Needs page_size / kpool to be a multiple of cp (checked in bind_kv_cache). cp below the
        # attention TP splits the attention group into cp_rows row groups of cp consecutive ranks
        # (models/mla.py attention_cp_rows). cp_group: the row group's process group (the engine sets it;
        # None: the attention group).
        self.cp = self.attn_tp if (_dsa_long.cp_enabled() and self.long_dsa and self.attn_tp > 1) else 1
        self.cp_rows = 1
        self.cp_group = None
        if self.cp > 1:
            self.cp = _dsa_long.cp_degree(self.attn_tp)
            self.cp_rows = self.attn_tp // self.cp
        # The first attention layer's head counts (the engine reads nkv).
        first = next(iter(self.kv_layers()), None)
        self.nh, self.nkv, self.kv_offset = (first.nh, first.nkv, first.kv_offset) if first is not None else (0, 0, 0)

    def rank_buffers(self, device=None) -> None:
        """This rank's position as buffers, so that every rank traces the same graph (a per-rank Python int
        would give each rank its own compile, see vocab_start_t): its row block of the sequence-parallel
        prefill streams (prefill_sp) as an index [1] and a one-hot [tp_size] row, and its block of a
        long-context DSA chunk's queries. Called at build and again on the device after loading."""
        if getattr(self, "prefill_sp", False) and self.tp_size > 1:
            sp_onehot = torch.zeros(self.tp_size, dtype=self.dtype, device="cpu")
            sp_onehot[self.tp_rank] = 1
            self.register_buffer("sp_index", torch.tensor([self.tp_rank], dtype=torch.int64, device="cpu").to(device),
                                 persistent=False)
            self.register_buffer("sp_onehot", sp_onehot.to(device), persistent=False)
        if getattr(self, "long_dsa", False) and self.attn_tp > 1:
            # Long-context DSA prefill (models/mla.py _long_prefill_select): this rank's block of a chunk's queries
            # within its attention group, as an index [1] and an fp32 one-hot [attn_tp] row (fp32: the gathered
            # pool indices go up to 262,144, which bf16 cannot hold). With row groups (cp_rows > 1): the rank
            # within its row group of cp ranks, and the row group as cp_row_index [1] / cp_row_onehot [cp_rows].
            rows = getattr(self, "cp_rows", 1)
            A = self.cp if rows > 1 else self.attn_tp
            g2 = torch.zeros(A, dtype=torch.float32, device="cpu")
            g2[self.attn_rank % A] = 1
            self.register_buffer("long_grp_index", torch.tensor([self.attn_rank % A], dtype=torch.int64,
                                                                device="cpu").to(device), persistent=False)
            self.register_buffer("long_grp_onehot", g2.to(device), persistent=False)
            if rows > 1 and _mla.CP_LOCAL_K > 0:  # the local-list selection's counts: [rows, failed, past CP_LOCAL_F]
                for l in self.kv_layers():
                    if _mla.long_capable(l.spec):
                        l.register_buffer("cp_localk", torch.zeros(3, dtype=torch.float32, device="cpu").to(device),
                                          persistent=False)
            if rows > 1:
                g3 = torch.zeros(rows, dtype=torch.float32, device="cpu")
                g3[self.attn_rank // A] = 1
                self.register_buffer("cp_row_index", torch.tensor([self.attn_rank // A], dtype=torch.int64,
                                                                  device="cpu").to(device), persistent=False)
                self.register_buffer("cp_row_onehot", g3.to(device), persistent=False)

    def kv_layers(self) -> list:
        """Every layer with a KV cache. Linear-attention layers hold a recurrent state instead (state_layers)."""
        return [l for l in self.layers if not isinstance(l.spec, LinearSpec)]

    def state_layers(self) -> list:
        """Layers whose per-sequence state is a fixed-size recurrent state (linear attention)."""
        return [l for l in self.layers if isinstance(l.spec, LinearSpec)]

    def state_shapes(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Per state layer, this rank's per-sequence (conv state, recurrent state) shapes."""
        return [linear_attn.state_shapes(l) for l in self.state_layers()]

    def bind_state(self, conv_states: list[torch.Tensor], rec_states: list[torch.Tensor]) -> None:
        """Device pools [slots, ...] of every state layer (engine/state_pool.py)."""
        for layer, c, r in zip(self.state_layers(), conv_states, rec_states):
            layer.conv_state = c
            layer.rec_state = r

    def aux_kv_shapes(self) -> list[tuple[int, ...]]:
        """Extra per-token caches (shape per token) the runner allocates beside K and V: none (the pooled
        indexer's rows live in the DSA layers' own caches and the pool keys in a token-slot state)."""
        return []

    def bind_aux_kv(self, caches: list[torch.Tensor]) -> None:
        if caches:
            raise ValueError(f"GLM-5.3-Flash has no per-token caches beside K and V; got {len(caches)}")

    def kv_shapes(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Per layer: ((nkv, Dk), (nkv, Dv)) of the K and V caches on this rank."""
        return [((l.nkv, l.spec.head_dim), (l.nkv, l.spec.v_head_dim)) for l in self.kv_layers()]

    def token_state_shapes(self) -> list[dict[str, tuple[int, ...]]]:
        """Per layer: token-slot state caches beside K and V on this rank, name -> per-token shape: a pooled
        DSA indexer's pool-key pieces (models/mla.py pool_key_width), kept per token so prefix caching and
        the host cache need nothing more."""
        out = []
        for l in self.kv_layers():
            d = {}
            w = _mla.pool_key_width(l.spec, self.fp8_max is not None)
            if w:
                d["pool_key"] = (w,)
            out.append(d)
        return out

    def bind_token_states(self, caches: list[dict[str, torch.Tensor]]) -> None:
        for layer, d in zip(self.kv_layers(), caches):
            for n, t in d.items():
                setattr(layer, n, t)

    def bind_kv_cache(self, k_caches: list[torch.Tensor], v_caches: list[torch.Tensor], page_size: int,
                      fp8_max: float | None = None, max_rows: int = 1, max_keys: int | None = None) -> None:
        """fp8_max: the cache is float8 e4m3 and values are clamped to +-fp8_max on write.
        max_rows / max_keys: the most query tokens one call carries and the longest context one
        reads (size DSA's selection scratch; max_keys defaults to the cache's slots)."""
        for layer, k, v in zip(self.kv_layers(), k_caches, v_caches):
            layer.k_cache = k
            layer.v_cache = v
        self.page_size = page_size
        self.fp8_max = fp8_max
        if self.cp > 1:
            if fp8_max is None:
                raise ValueError("context-parallel DSA (attention TP > 1) reads its pool keys from the separate "
                                 "token-slot cache, which exists under an FP8 KV cache only (kv_cache_dtype fp8)")
            kp = next(l.spec.mla.dsa.kpool for l in self.kv_layers())
            if page_size % kp or (page_size // kp) % self.cp:
                raise ValueError(f"context-parallel DSA: {page_size // kp} pools per page do not divide over "
                                 f"cp={self.cp} (page_size must be a multiple of kpool x cp)")
            max_keys = 1  # every bucket runs the long path, which stages nothing
        _mla.bind_scratch(self, max_rows, k_caches[0].device if k_caches else None,
                          max_keys or (k_caches[0].shape[0] if k_caches else None))

    def _store(self, cache: torch.Tensor, slots: torch.Tensor, x: torch.Tensor) -> None:
        if self.fp8_max is not None:
            # KV scale 1.0 (vLLM's default without calibration); clamp first, because a
            # finite value past the format's max would become inf / NaN on the cast.
            x = x.float().clamp(-self.fp8_max, self.fp8_max).to(cache.dtype)
        cache.index_put_((slots,), x)

    def _load(self, cache: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        kv = self._gather(cache, block_table)
        return kv.to(self.dtype) if self.fp8_max is not None else kv

    # -- shared pieces --------------------------------------------------------------

    def _all_reduce(self, x: torch.Tensor, group=None) -> torch.Tensor:
        """Sum over the tensor-parallel ranks (or over `group`, a subgroup of them)."""
        if self.tp_size == 1:
            return x
        group = self.tp_group if group is None else group
        if x.device.type == "cpu":
            import torch.distributed as dist

            y = x.clone()
            dist.all_reduce(y, group=group)
            return y
        # Lowered by libtorch_neuronx_lite to an in-graph collective (_c10d_functional.all_reduce); compile
        # flow only. Its replica groups are the group's ranks, which also enter the compile cache key, so
        # subgroups each get their own NEFF of a graph holding a group all-reduce.
        import torch.distributed._functional_collectives as funcol

        return funcol.all_reduce(x, "sum", group)

    def _attn_all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """A token mixer's output projection summed over the ranks (the block output's reduction,
        _out_reduce); nothing at tp 1."""
        if self.tp_size == 1:
            return x
        return self._out_reduce(x)

    def _reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        """Block tp_rank of tp_size equal row blocks of x [tp_size * r, ...] summed over the world: the rows
        _sp_rows would keep of _all_reduce(x), from one world reduce-scatter (half the bytes). A graph
        holding one reloads from the compile cache in a later process, unlike an all-gather. On the host
        (gloo) it is that all-reduce and those rows."""
        if self.tp_size == 1:
            return x
        if x.device.type == "cpu":
            return self._sp_rows(self._all_reduce(x))
        import torch.distributed._functional_collectives as funcol

        return funcol.reduce_scatter_tensor(x, "sum", 0, self.tp_group)

    def _out_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """A block output's reduction over the ranks: the all-reduce, or while a sequence-parallel block is
        traced (_sp_rs_call) the reduce-scatter that hands each rank its own rows."""
        return self._reduce_scatter(x) if self._sp_rs else self._all_reduce(x)

    def _sp_rs_call(self, fn, *args):
        """fn(*args) with its block output reduced by a reduce-scatter (_out_reduce): this rank's rows."""
        self._sp_rs = True
        try:
            return fn(*args)
        finally:
            self._sp_rs = False

    def _sp_rs_attn(self) -> bool:
        """Whether the token mixers' output reduction can be a world reduce-scatter (_attn_all_reduce)."""
        return self.tp_size > 1

    def _sp_on(self) -> bool:
        """Whether a prefill chunk's streams are sequence-parallel (see above)."""
        return self.prefill_sp and self.tp_size > 1

    def _sp_rows(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's rows of a batch x [R, ...] held by every rank: block tp_rank of tp_size equal
        blocks (an index_select by the sp_index buffer)."""
        return x.reshape(self.tp_size, -1, *x.shape[1:]).index_select(0, self.sp_index)[0]

    def _sp_gather(self, x: torch.Tensor) -> torch.Tensor:
        """Every rank's rows [tp_size * r, ...] from each rank's own x [r, ...], in rank order: x in its
        block of a zero [tp_size, r, ...] (a one-hot multiply: index_copy gave NaN on the device, see _head)
        summed over the world. Exact: each element is one rank's value plus zeros. (An NKI all_gather here
        failed neuronx-cc 2.27 on the prefill pieces with tensor-parallel experts, NCC_ISCH719 "topological
        order violations".)"""
        full = x.unsqueeze(0) * self.sp_onehot.view(self.tp_size, *[1] * x.dim())
        return self._all_reduce(full.reshape(-1, *x.shape[1:]))

    def _w(self, layer: DecoderLayer, name: str) -> torch.Tensor:
        return dequant(getattr(layer, name), getattr(layer, name + "_scale"), self.dtype)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        """The weight is read where the embedding is taken, AFTER vocab_start_t: dynamo lifts graph inputs in
        the order the trace touches them, and reading self.embed first swapped two inputs of every graph holding
        the embedding."""
        if not self.vocab_parallel:
            return F.embedding(ids, self.embed)
        local = ids - self.vocab_start_t
        mine = (local >= 0) & (local < self.vocab_rows)
        e = F.embedding(local.clamp(0, self.vocab_rows - 1), self.embed)
        return self._all_reduce(torch.where(mine.unsqueeze(-1), e, torch.zeros_like(e)))

    def _final(self, h: torch.Tensor) -> torch.Tensor:
        """The last hidden state, its residual streams collapsed and normalised for the lm_head (models/hybrid.py)."""
        return _hybrid.final(self, h)

    def _hidden_in(self, input_ids: torch.Tensor, sp: bool = False) -> torch.Tensor:
        """The residual streams of the input tokens; sp: only this rank's rows (sequence-parallel prefill
        streams)."""
        return _hybrid.hidden_in(self, input_ids, sp)

    def _head(self, h: torch.Tensor) -> torch.Tensor:
        """lm_head over hidden states that are already normalised."""
        w = self.embed if self.lm_head is None else self.lm_head
        if not self.vocab_parallel:
            return F.linear(h, w).float()
        # Gathered with an all-reduce of zero-padded shards, not all_gather_into_tensor: a
        # cached NEFF holding an all-gather, loaded by a LATER process, failed with "replica
        # group signature mismatch ... mismatched collectives between peers" on all 32 ranks
        # (SDK 2.32), while all-reduce graphs reload fine. The shard's column offset is a buffer,
        # so every rank traces one graph. The shard is placed by a one-hot multiply, not index_copy:
        # with index_copy every prompt logprob came back NaN on the device. The shard is found by
        # multiplying, not dividing: with vocab_start_t // Vr, a tp=32 model on trn1 returned its top
        # tokens one shard too low (in-graph integer division is not exact on the device).
        T, Vr = h.shape[0], self.vocab_rows
        local = F.linear(h, w)  # [T, Vr]
        mine = (torch.arange(self.tp_size, device=h.device) * Vr == self.vocab_start_t).to(local.dtype)
        full = (local.unsqueeze(1) * mine.view(1, -1, 1)).reshape(T, self.tp_size * Vr)
        return self._all_reduce(full)[:, : self.cfg.vocab_size].float()

    def _sample_vp_on(self, bitmask=None, penalties=None) -> bool:
        """Whether a sampling call runs _sample_vp (a vocab-parallel head, no bitmask or penalties)."""
        return self.vocab_parallel and self.tp_size > 1 and bitmask is None and penalties is None

    def _sample_vp(self, hn: torch.Tensor, temperature, top_p, top_k, min_p, noise) -> torch.Tensor:
        """sampler.sample(self._head(hn), ...) from each rank's shard of the logits (see above): every rank's top
        max(K, N) values and their ids, and its shard's log-sum-exp, gathered with one all-reduce of zero-padded
        rows (as _head gathers, without the vocabulary), then sampler.sample_candidates on their union. The
        shard's values are _head's (bf16 matmul, then fp32). Ids are fp32 (exact below 2^24): a cast of an int64
        id sum to fp32 failed to lower ("shift-left with different element types: s64[] and s32[]"), so the
        shard's first id is a float from the one-hot."""
        w = self.embed if self.lm_head is None else self.lm_head
        h = hn
        B, Vr, tp = h.shape[0], self.vocab_rows, self.tp_size
        M = max(noise.shape[1], NUM_TOP_LOGPROBS)
        local = F.linear(h, w).float()  # [B, Vr]
        mine = (torch.arange(tp, device=h.device) * Vr == self.vocab_start_t).to(torch.float32)  # [tp] one-hot
        start = (mine * torch.arange(tp, device=h.device, dtype=torch.float32)).sum() * float(Vr)
        col = torch.arange(Vr, device=h.device, dtype=torch.float32).view(1, Vr) + start
        real = col < torch.full_like(col, float(self.cfg.vocab_size))  # the last shard's padding columns are not ids
        local = torch.where(real, local, torch.full_like(local, -1e30))
        vals, idx = topk_large(local, M)
        part = torch.cat([logsumexp_large(local), vals, idx.float() + start], dim=1)  # [B, 1 + 2M]
        both = self._all_reduce((part.unsqueeze(1) * mine.view(1, tp, 1)).reshape(B, tp * (1 + 2 * M)))
        both = both.view(B, tp, 1 + 2 * M)
        lse = torch.logsumexp(both[:, :, 0], dim=-1, keepdim=True)
        return sample_candidates(both[:, :, 1 : 1 + M].reshape(B, tp * M), both[:, :, 1 + M :].reshape(B, tp * M),
                                 lse, temperature, top_p, top_k, min_p, noise)

    def _sample(self, hn, temperature, top_p, top_k, min_p, noise, bitmask, penalties) -> torch.Tensor:
        """The sampler over normalised hidden states hn [B, H] (packed sampler output [B, *])."""
        if self._sample_vp_on(bitmask, penalties):
            return self._sample_vp(hn, temperature, top_p, top_k, min_p, noise)
        return sample(self._head(hn), temperature, top_p, top_k, min_p, noise, bitmask, penalties)

    def _slots(self, pos: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """Cache slots of positions pos [T, n] (non-negative) through a block table: [P] for one
        sequence's chunk, [B, P] for B sequences whose T = B * Q rows are in order."""
        ps = self.page_size
        if table.dim() == 1:
            pages = table[pos // ps]
        else:
            B = table.shape[0]
            pages = torch.gather(table, 1, (pos // ps).reshape(B, -1)).reshape(pos.shape)
        return pages * ps + pos % ps

    # Up to this many (token, expert) pairs, a MoE layer on the host gathers just the selected experts'
    # weights; above it, every expert runs over every token and the routing weights zero the rest. Both
    # are exact. On the device a packed layer always runs the NKI kernels: LNL rejects the in-graph uint8
    # -> float8 view the blob's unpack needs ("Expected XLA tensor. Got: XLAFloat8_e4m3fnType", SDK 2.32),
    # so the XLA paths serve blob layers on the CPU only.
    MOE_GATHER_MAX_PAIRS = 512

    def _route(self, layer: DecoderLayer, x: torch.Tensor):
        """Sigmoid routing with a selection bias (noaux_tc): select on score + correction bias, weight by the
        uncorrected score, renormalise, scale."""
        cfg = self.cfg
        k = cfg.num_experts_per_tok
        logits = F.linear(x.float(), layer.router.float(), None)  # [T, E] fp32
        scores = logits.sigmoid()
        choice = scores + layer.router_bias if layer.router_bias is not None else scores
        # The device-safe top-k (dsa_select.topk_ids): NaN and -inf ranked as finfo.min and the indices clamped
        # into [0, E). On trn2 (SDK 2.32, LNC=2) torch.topk over a row holding NaN returns 0xFFFFFFFF for every
        # index, and the gather at them faults ("scatter/gather (indirect memory copy via vector DGE)
        # out-of-bound access"): a bucket warmup's all-zero inputs give NaN rows. The ranking matters as much
        # as the clamp: a clamp alone sends a NaN row to expert E - 1 eight times, past moe_dedupe's plan of
        # ceil(T / lanes) slots per expert; ranked as finfo.min, a NaN row is all tied and takes k distinct
        # in-range experts. Every finite row routes bit for bit as torch.topk.
        topi = _dsa_select.topk_ids(choice, k)
        topv = torch.gather(scores, 1, topi)
        if cfg.norm_topk_prob and k > 1:
            topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-20)
        topv = topv * cfg.routed_scaling_factor
        return topv.to(self.dtype), topi

    def _experts(self, layer: DecoderLayer, name: str, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Expert weights [E or len(idx), ...] in the model dtype: w_gu as [out, in], w_down as
        [in, out] (Im, H). FP8 + scale (or the packed blob) is dequantized."""
        if layer.moe_blob:
            from ..kernels import moe_dedupe

            if name == "w_gu":
                return dequant(*moe_dedupe.unpack_gu(layer.w_blob, self.cfg.hidden_size, idx), self.dtype)
            return dequant_t(*moe_dedupe.unpack_down(layer.w_blob, self.cfg.hidden_size, idx), self.dtype)
        w, sc = getattr(layer, name), getattr(layer, name + "_scale")
        if idx is not None:
            w, sc = w[idx], (sc[idx] if sc is not None else None)
        if name == "w_down" and layer.down_t:
            return dequant_t(w, sc, self.dtype)
        return dequant(w, sc, self.dtype)

    def _gather(self, cache: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        """Paged cache [num_pages * page_size, nkv, D] gathered by a block table [..., P] into
        [..., P * page_size, nkv, D], by token slots below PAGE_GATHER_MIN_TOKENS, else by whole pages."""
        ps = self.page_size
        tokens = block_table.shape[-1] * ps
        if tokens < PAGE_GATHER_MIN_TOKENS:
            offs = torch.arange(ps, device=block_table.device, dtype=block_table.dtype)
            return cache[(block_table.unsqueeze(-1) * ps + offs).flatten(-2)]
        pages = cache.view(-1, ps, cache.shape[1], cache.shape[2])[block_table]  # [..., P, ps, nkv, D]
        return pages.flatten(-4, -3)

    @staticmethod
    def _bias(visible: torch.Tensor) -> torch.Tensor:
        return torch.where(visible, 0.0, NEG_INF)

    # -- one layer over a batch, shared by every forward ------------------------------
    #
    # Two batch forms, told apart by the block table and the bias:
    #   decode  table [B, P], bias [B, 1, 1, L]     h [B, hc * H]   (one token per sequence)
    #   chunk   table [P],    bias [1, 1, C, L]     h [C, hc * H]   (one sequence, C tokens)
    # Under context-parallel DSA every bucket is a long-context one and the bias is a placeholder
    # (_long_attn).

    def _layer(self, layer, h, positions, slot_mapping, table, bias, state_slot=None, **mixin):
        """state_slot: each sequence's row in the recurrent-state pool, read by linear-attention layers
        (models/linear_attn.py); attention layers ignore it. mixin: hybrid.layer's pend / defer (_run_layers)."""
        return _hybrid.layer(self, layer, h, positions, slot_mapping, table, bias, state_slot, **mixin)

    def long_ctx(self, L: int) -> bool:
        """Whether a bucket of L keys runs every attention layer's long-context path (models/dsa_long.py; static
        per bucket)."""
        return self.long_dsa and (self.cp > 1 or _dsa_long.enabled(L))

    def _long_attn(self, block_table, shape):
        """_attn_inputs of a long-context bucket: the long path reads the positions and the block table only, so
        the bias is a [*shape, 1] placeholder (its rank tells the batch forms apart, models/mla.py attention). It is
        made from the block table (times 0) so that every page bucket's prep graph is its own: without the table the
        prep graphs of all long buckets were one graph, one cache key loaded once per bucket, and at tp=2 the ranks'
        loaded copies failed the collective barrier ("MPMD execution is not supported. Most likely some ranks
        recompiled/reloaded a graph")."""
        zero = block_table.reshape(-1)[:1].to(torch.float32) * 0.0
        return {None: (zero.expand(int(torch.Size(shape).numel())).reshape(*shape, 1).contiguous(), block_table)}

    def _attn_inputs(self, vis_full, block_table, shape):
        """(bias, table) of the attention layers, under key None. vis_full: visibility over block_table's keys
        [..., L]; shape: the view of a bias row set."""
        return {None: (self._bias(vis_full).view(*shape, -1), block_table)}

    def _run_layers(self, h, positions, slot_mapping, attn, state_slot=None):
        """Every layer in order. A layer may leave its FFN block's stream mix to the next layer's KDA shell call
        (hybrid.defers), which mixes it in first."""
        pend = None
        bias, table = attn[None]
        for n, layer in enumerate(self.layers, 1):
            h, pend = self._layer_mixin(n, layer, h, positions, slot_mapping, table, bias, state_slot, attn, pend)
        return h

    def _layer_mixin(self, n, layer, h, positions, slot_mapping, table, bias, state_slot, attn, pend):
        """_run_layers' layer n (1-based): given the previous layer's deferred FFN triple (pend, or None), it defers
        its own when the next layer's attention block is a KDA shell call (hybrid.defers). (h, its deferred triple
        or None); the last layer never defers."""
        defer = False
        if n < len(self.layers):
            nxt = self.layers[n]
            nbias, ntable = attn[None]
            defer = _hybrid.defers(self, layer, nxt, h, positions, ntable, nbias, state_slot)
        kw = {}
        if pend is not None:
            kw["pend"] = pend
        if defer:
            kw["defer"] = True
        out = self._layer(layer, h, positions, slot_mapping, table, bias, state_slot, **kw)
        return out if defer else (out, None)

    # -- piecewise execution: one compiled graph per KIND of layer ---------------------
    #
    # vLLM compiles its model piecewise (split at attention) and SGLang captures per batch
    # size; here the unit is a run of whole layers. layer_fn(i) is a pure function of tensors, so
    # every layer of one kind (same static attributes, same tensor shapes) traces to the
    # same graph and shares one NEFF.

    def _layer_split(self, i: int):
        layer = self.layers[i]
        skip = getattr(layer, "_decode_only", ())  # tensors only the whole-step decode reads
        tensors = {n: t for n, t in [*layer._parameters.items(), *layer._buffers.items(),
                                     *vars(layer).items()] if isinstance(t, torch.Tensor) and n not in skip}
        static = {n: v for n, v in vars(layer).items()
                  if not n.startswith("_") and n not in tensors and not isinstance(v, (nn.Module, torch.Tensor))}
        static.update({n: None for n, t in [*layer._parameters.items(), *layer._buffers.items()] if t is None})
        return static, tensors

    def layer_kind(self, i: int) -> tuple:
        static, tensors = self._layer_split(i)
        return (tuple(sorted((n, repr(v)) for n, v in static.items())),
                tuple((n, tuple(t.shape), t.dtype) for n, t in tensors.items()))

    def layer_tensors(self, i: int) -> tuple:
        return tuple(self._layer_split(i)[1].values())

    def group_fn(self, idxs):
        """f(h, positions, slot_mapping, table, bias, table_w, bias_w, *tensors, state_slot=None, mixed=None)
        -> h running layers idxs in order, where tensors is layer_tensors of each in turn and state_slot serves
        the linear-attention ones. Valid for every run of layers with the same sequence of kinds. table_w /
        bias_w (sliding-window layers) and mixed (mixed batches) are the engine's calling convention: there are
        none, so they must be None."""
        parts = []
        for i in idxs:
            static, tensors = self._layer_split(i)
            parts.append((static, tuple(tensors)))

        def f(h, positions, slot_mapping, table, bias, table_w, bias_w, *ts, state_slot=None, mixed=None):
            if mixed is not None:
                raise NotImplementedError("mixed batches")
            o = 0
            for static, names in parts:
                view = _LayerView(static, dict(zip(names, ts[o : o + len(names)])))
                o += len(names)
                h = self._layer(view, h, positions, slot_mapping, table, bias, state_slot)
            return h

        return f

    # -- decode: B sequences x 1 token ------------------------------------------------

    def prep_decode(self, input_ids, positions, block_table, context_lens, board, read_slot,
                    swa_table=None, swa_first=None, ngram_ids=None):
        input_ids = torch.where(read_slot >= 0, board[read_slot.clamp(min=0)].long(), input_ids)
        # (Shapes come from input_ids, not block_table: the order a graph first touches its inputs orders its
        # placeholders, which is part of the compile cache key.)
        B = input_ids.shape[0]
        if self.long_ctx(block_table.shape[1] * self.page_size):  # models/dsa_long.py: no [B, L] visibility
            return self._hidden_in(input_ids), self._long_attn(block_table, (B, 1, 1))
        j = torch.arange(block_table.shape[1] * self.page_size, device=block_table.device).unsqueeze(0)
        attn = self._attn_inputs(j < context_lens.unsqueeze(1), block_table, (B, 1, 1))
        return self._hidden_in(input_ids), attn

    def post_decode(self, h, temperature, top_p, top_k, min_p, noise, board, write_slot,
                    bitmask=None, penalties=None):
        out = self._sample(self._final(h), temperature, top_p, top_k, min_p, noise, bitmask, penalties)
        board.index_put_((write_slot,), out[:, 0])
        return out

    def forward_decode(self, input_ids, positions, block_table, context_lens, slot_mapping,
                       temperature, top_p, top_k, min_p, noise, board, read_slot, write_slot,
                       bitmask=None, penalties=None, swa_table=None, swa_first=None, state_slot=None,
                       ngram_ids=None):
        """board [S] fp32: each request's last sampled token, kept on the device. A row with
        read_slot >= 0 takes its input token from the board instead of input_ids (overlap
        scheduling: the previous step's token was never read back); every row writes its
        sampled token to board[write_slot].

        state_slot [B]: each sequence's row in the recurrent-state pool (padded rows use the
        scratch row). swa_table / swa_first / ngram_ids are the engine's calling convention (no
        sliding-window layers, no n-gram embedding): None."""
        h, attn = self.prep_decode(input_ids, positions, block_table, context_lens, board, read_slot)
        h = self._run_layers(h, positions, slot_mapping, attn, state_slot)
        return self.post_decode(h, temperature, top_p, top_k, min_p, noise, board, write_slot, bitmask, penalties)

    # -- prefill: one sequence x C tokens ---------------------------------------------

    def prep_prefill(self, input_ids, positions, block_table, swa_table=None, swa_first=None, ngram_ids=None):
        """block_table [P] covers the sequence's whole context."""
        C = input_ids.shape[0]
        if self.long_ctx(block_table.shape[0] * self.page_size):  # models/dsa_long.py: no [C, L] visibility
            attn = self._long_attn(block_table, (1, 1, C))
        else:
            j = torch.arange(block_table.shape[0] * self.page_size, device=block_table.device).unsqueeze(0)
            attn = self._attn_inputs(j <= positions.unsqueeze(1), block_table, (1, 1, C))
        sp = self._sp_on()
        if sp and input_ids.shape[0] % self.tp_size:
            raise ValueError(f"sequence-parallel prefill streams need the chunk's {input_ids.shape[0]} rows to "
                             f"divide over tp={self.tp_size} (ModelRunner turns them off for such buckets)")
        return self._hidden_in(input_ids, sp), attn

    def post_prefill(self, h, last_index, temperature, top_p, top_k, min_p, noise, board, write_slot,
                     bitmask=None, penalties=None, plp_targets=None):
        if plp_targets is not None:
            raise NotImplementedError("prompt logprobs")
        hn = self._final(h)
        if self._sp_on():  # sequence-parallel streams: every rank's rows of the final hidden state
            hn = self._sp_gather(hn)
        out = self._sample(hn.index_select(0, last_index), temperature, top_p, top_k, min_p, noise, bitmask,
                           penalties)
        board.index_put_((write_slot,), out[:, 0])
        return out

    def forward_prefill(self, input_ids, positions, block_table, slot_mapping, last_index,
                        temperature, top_p, top_k, min_p, noise, board, write_slot,
                        bitmask=None, penalties=None, swa_table=None, swa_first=None, plp_targets=None,
                        state_slot=None, ngram_ids=None):
        """state_slot [1]: the sequence's row in the recurrent-state pool. The whole-graph form of the
        piecewise prefill (prep_prefill, group_fn, post_prefill)."""
        h, attn = self.prep_prefill(input_ids, positions, block_table)
        h = self._run_layers(h, positions, slot_mapping, attn, state_slot)
        return self.post_prefill(h, last_index, temperature, top_p, top_k, min_p, noise, board, write_slot,
                                 bitmask, penalties, plp_targets)

    # -- reference: logits for a whole sequence, no cache (tests only) -----------------

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return _hybrid.forward_logits(self, input_ids)


class _LayerView:
    """A DecoderLayer's static attributes plus tensors passed in as graph inputs."""

    def __init__(self, static: dict, tensors: dict):
        for d in (static, tensors):  # setattr, not __dict__.update: dynamo traces this
            for k, v in d.items():
                setattr(self, k, v)


# Imported last: models/mla.py and models/hybrid.py import this module's helpers.
from . import hybrid as _hybrid  # noqa: E402
from . import mla as _mla  # noqa: E402
from . import dsa_select as _dsa_select  # noqa: E402
from . import dsa_long as _dsa_long  # noqa: E402
