"""The engine configuration: shapes, pools and buckets. The model configuration is models/config.py."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .models.config import AttnSpec, LinearSpec, ModelConfig  # noqa: F401  (re-exported)


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
    admission. Only the shapes and sizes below vary, with the decode kernel routes' row caps (graph structure, like
    the shapes) and the admission cap max_running (host only)."""
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
    # At most this many requests running at once, the rest waiting in arrival order (None: max_num_seqs). Host only: no
    # graph or cache key changes, since max_num_seqs is the graph shape (the state pool's rows) and the decode buckets
    # are compiled whatever runs. A deployment that wants fewer concurrent requests than its compiled seats caps them
    # here.
    max_running: int | None = None
    # The decode kernel routes' row caps (engine.build_shard sets them on every rank before the shard is built, so the
    # ranks of an engine and of a capture trace the same routes). Each one decides which graph a decode step of a given
    # row count traces, so a change needs a new capture. The defaults here are the model modules' own import-time caps,
    # for decode steps of up to 8 rows (--preset batch8); the command line's defaults, the served configuration
    # (trnrt/__main__.py), raise them for steps of up to 12 rows: 12, 16, 17 and 8.
    # - kda_shell_max_rows (models/hybrid.py KDA_SHELL_MAXB): a KDA layer's decode attention block runs as the KDA
    #   shell's kernel at most this many rows, the XLA half above.
    # - dsa_layer_max_rows (models/mla.py LAYER_MK_MAXB): a DSA layer's decode attention block and FFN half run as the
    #   fused layer's one call at most this many rows.
    # - ffn_half_below (models/hybrid.py MOE_HALF_BELOW): a decode FFN block runs as the FFN half's one call below this
    #   many rows.
    # - dsa_index_rows (models/mla.py CP_INDEX_ROWS): context-parallel DSA indexer scores of a call of more rows than
    #   this run as one kernel call per this many rows (0: one call).
    kda_shell_max_rows: int = 8
    dsa_layer_max_rows: int = 8
    ffn_half_below: int = 16
    dsa_index_rows: int = 0

    def __post_init__(self):
        if self.max_running is not None and self.max_running < 1:
            raise ValueError(f"max_running={self.max_running}: at least 1, or None (max_num_seqs)")
        for name in ("kda_shell_max_rows", "dsa_layer_max_rows", "ffn_half_below"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name}={getattr(self, name)}: at least 1")
        if self.dsa_index_rows < 0:
            raise ValueError(f"dsa_index_rows={self.dsa_index_rows}: 0 (one call) or more")

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
