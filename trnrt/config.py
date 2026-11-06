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
    # With a Hugging Face repo id as model_path: the commit (or branch, tag) to download (None: the default branch).
    model_revision: str | None = None
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
    # - dense_half_skip_rows (kernels/dense_half.py SKIP_T): the decode row counts a dense MLP layer runs as the XLA graph
    #   instead of the dense half's kernel.
    dense_half_skip_rows: tuple[int, ...] = (2,)
    # Speculative decoding (engine/spec_async.py): the model's own MTP layer drafts spec_k tokens a step, the target
    # verifies them in the next step's call, and the accepted count, the newest token, the positions and the drafts stay
    # on the device, so a step is scheduled before the previous one is read back. None: plain decoding.
    spec_method: str | None = None
    spec_k: int = 2  # drafts per step: a verify call is 1 + spec_k rows a sequence
    # Benchmark mode (engine/spec_synth.py): the accepted draft count of every verify is drawn from a schedule whose mean
    # acceptance length (1 + the accepted drafts a step) is this value, in [1, spec_k + 1], instead of the draft head's
    # own. Verification stays real; accepted positions still take their drafts. For comparing serving throughput at a
    # published acceptance length, not for quality. The command line reads TRNRT_SPEC_SYNTH_AL when --spec-synth-al is absent.
    spec_synth_al: float | None = None
    # Dynamic speculation (vLLM's speculative_disable_by_batch_size, with hysteresis): while at most this many requests
    # decode they speculate, above it every decode step is the plain graph (0: always speculate). Once plain, speculation
    # returns when at most spec_max_batch - spec_hysteresis requests decode. A flip drains the step in flight.
    spec_max_batch: int = 0
    spec_hysteresis: int = 1
    # With dynamic speculation, the decode buckets each step family is warmed and picked at: the verify family (verify,
    # the speculative board graphs, the draft pass) only at buckets of at most spec_bucket_max, the plain decode only at
    # buckets of at least plain_bucket_min (a smaller batch is padded up). 0: every bucket.
    spec_bucket_max: int = 0
    plain_bucket_min: int = 0
    # The page buckets of every graph the MTP layer or a verify runs in (None: page_buckets). Its largest is theirs.
    spec_page_buckets: tuple[int, ...] | None = None
    # The MTP pass over a prefill chunk fills the draft layer's KV: in calls of at most this many rows (None: one call
    # of the chunk's bucket), taking the main layers' prefill form from mtp_chunk_q rows on (0: never), and, with
    # mtp_kv_only, a window that holds no row of the last position writes only the draft layer's KV.
    mtp_fill_rows: int | None = None
    mtp_chunk_q: int = 0
    mtp_kv_only: bool = False
    # A plain decode step (dynamic speculation) writes the draft layer's KV for its rows, as a speculative step's draft
    # pass does, so a request that returns to speculation finds no holes in it.
    dyn_mtp_kv: bool = False
    # The recurrent-state pool sized apart from max_num_seqs: state_seats seats (None: max_num_seqs), of which
    # state_spec_seats hold 1 + spec_k rows each (the ones that speculate) and the rest one row (None: all of them).
    state_seats: int | None = None
    state_spec_seats: int | None = None

    def __post_init__(self):
        if self.max_running is not None and self.max_running < 1:
            raise ValueError(f"max_running={self.max_running}: at least 1, or None (max_num_seqs)")
        for name in ("kda_shell_max_rows", "dsa_layer_max_rows", "ffn_half_below"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name}={getattr(self, name)}: at least 1")
        if self.dsa_index_rows < 0:
            raise ValueError(f"dsa_index_rows={self.dsa_index_rows}: 0 (one call) or more")
        if self.spec_method not in (None, "mtp"):
            raise ValueError(f"spec_method={self.spec_method!r}: only 'mtp' (or None) is served")
        if self.spec_method is None:
            if self.spec_synth_al is not None:
                raise ValueError("spec_synth_al (--spec-synth-al, or TRNRT_SPEC_SYNTH_AL) needs spec_method 'mtp'")
            for name in ("spec_page_buckets", "mtp_fill_rows", "state_spec_seats"):
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} needs spec_method 'mtp'")
            for name in ("spec_max_batch", "spec_bucket_max", "plain_bucket_min", "mtp_chunk_q", "mtp_kv_only",
                         "dyn_mtp_kv"):
                if getattr(self, name):
                    raise ValueError(f"{name} needs spec_method 'mtp'")
        else:
            if self.spec_k < 1:
                raise ValueError(f"spec_k={self.spec_k}: at least 1")
            if self.page_size & (self.page_size - 1):
                raise ValueError(f"speculative decoding needs a power-of-two page size, not {self.page_size}")
            if self.spec_synth_al is not None and not 1.0 <= self.spec_synth_al <= self.spec_k + 1:
                raise ValueError(f"spec_synth_al={self.spec_synth_al}: in [1, spec_k + 1] = [1, {self.spec_k + 1}]")
            if self.spec_max_batch < 0 or self.spec_hysteresis < 0:
                raise ValueError("spec_max_batch and spec_hysteresis: 0 or more")
            if (self.spec_bucket_max or self.plain_bucket_min) and not self.spec_max_batch:
                raise ValueError("spec_bucket_max and plain_bucket_min need spec_max_batch (dynamic speculation)")
            if self.state_spec_seats is not None and not self.spec_max_batch:
                raise ValueError("state_spec_seats needs spec_max_batch (dynamic speculation)")
            if self.dyn_mtp_kv and not self.spec_max_batch:
                raise ValueError("dyn_mtp_kv needs spec_max_batch (dynamic speculation)")
            if self.mtp_kv_only and not self.mtp_chunk_q:
                raise ValueError("mtp_kv_only needs mtp_chunk_q (the chunk form of the draft layer's windows)")
            if self.mtp_fill_rows is not None and self.mtp_fill_rows < 1:
                raise ValueError(f"mtp_fill_rows={self.mtp_fill_rows}: a positive row count, or None")
        if self.state_seats is not None:
            cap = self.max_running or self.max_num_seqs
            if self.state_seats < cap:
                raise ValueError(
                    f"state_seats={self.state_seats} is below the running cap {cap} (max_running, else max_num_seqs): "
                    f"a running request must hold its state rows")
            if self.state_seats > self.max_num_seqs:
                raise ValueError(f"state_seats={self.state_seats} above max_num_seqs={self.max_num_seqs}")
        if self.state_spec_seats is not None and not self.spec_max_batch <= self.state_spec_seats <= (
                self.state_seats or self.max_num_seqs):
            raise ValueError(
                f"state_spec_seats={self.state_spec_seats} must cover spec_max_batch {self.spec_max_batch} and fit the "
                f"{self.state_seats or self.max_num_seqs} seats")

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
