"""The TrnRT command line.

    python -m trnrt serve --model DIR [--port 8000]       # two engines and the router in front of them
    python -m trnrt engine --model DIR --core-base 0       # one engine and its OpenAI API, no router
    python -m trnrt compile {shapes,decisions,capture,compile,strip} ...   # the compile cache (trnrt/compile/farm.py)

Every default is the served configuration of GLM-5.3-Flash on one trn2.48xlarge: two engines of 32 tensor-parallel ranks
each (logical NeuronCores 0-31 and 32-63 at LNC 2), 1M-token context, FP8 KV cache, 256-token pages, decode batches of
up to 12 per engine (16 seats, 12 running), 4,096-token prefill chunks, and 4 GiB per rank of host memory under the
prefix cache (`--preset batch12` names it). `--preset batch8` is the other supported configuration: decode batches of up
to 8 per engine (PRESETS below). The bucket ladders and the decode kernel routes are graph structure: a server finds its
graphs in the compile cache only if `compile capture` ran with the same arguments (the same --preset included).
--max-running (fewer requests running than the compiled seats) is host only. An engine compiles nothing on the serving
host: a graph missing from the cache ends its start-up with the likely cause (NEURON_LIBTORCH_ASSERT_CACHE_HIT=1;
--allow-compile compiles it there instead, minutes to hours a graph). Before it starts, an engine on Neuron checks its
host (engine/start_checks.py): its devices free of other processes, and enough contiguous free host memory for the
runtime.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

from . import log

GiB = 2**30
# libtorch-neuronx-lite raises on a graph missing from the compile cache instead of compiling it (neuronx-cc on the
# serving host: minutes to hours per graph, while the engine is not serving).
CACHE_HIT_ENV = "NEURON_LIBTORCH_ASSERT_CACHE_HIT"
CACHE_MISS_STATUS = 3  # an engine's exit status when its start-up stopped at a graph missing from the compile cache
START_REFUSED_STATUS = 4  # an engine's exit status when a check of its host refused its start (engine/start_checks.py)
HEALTH_POLL_S = 5.0  # serve: how often it asks a starting engine's /health


def _ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x)


def _at_least(lo: int):
    def parse(s: str) -> int:
        v = int(s)
        if v < lo:
            raise argparse.ArgumentTypeError(f"{v}: at least {lo}")
        return v

    return parse


# Named configurations: each sets the engine arguments below (argparse dests) as defaults, and an argument given
# explicitly still wins. A preset's graphs are its own, so `compile capture -- --preset NAME ...` must run with the
# preset the server runs with.
# - batch12: the served configuration, the defaults below; the preset names them. Decode batches of up to 12 requests
#   per engine: 16 seats (the state pool's rows), decode buckets 1, 2, 4, 6, 8 and 12, at most 12 requests running, and
#   every decode step of up to 12 rows on the fused decode kernels: the KDA shell and the split KDA layer, the fused
#   DSA layer and the FFN half.
# - batch8: decode batches of up to 8 requests per engine. 8 seats (every graph that carries the linear-attention state
#   is re-keyed against the defaults), decode buckets 1, 2, 4, 6 and 8, all 8 seats running (--max-running unset), and
#   the fused decode kernels' row caps of steps of up to 8 rows: the model modules' own (hybrid.KDA_SHELL_MAXB,
#   mla.LAYER_MK_MAXB, ...) and EngineConfig's.
# The mtp-* presets serve with speculative decoding (docs/AGENTX.md): the model's MTP layer drafts 2 tokens a step.
# - mtp-batch8: decode batches of up to 8 requests per engine. The decode buckets 1, 2, 4, 6 and 8, 8 seats running,
#   speculation while at most 4 requests decode and the plain decode step above, page buckets 1024, 3072 and 4096
#   (the speculative graphs' own: 1024 and 4096).
# - mtp-batch2: batches of up to 8 with speculation at every batch size (no plain decode step): the configuration
#   of the lowest concurrency points, page buckets 1024 and 4096.
# - mtp-batch12: up to 12 requests: speculation while at most 4 decode (4 of the 12 state seats hold the rows a
#   verify writes), the plain step above; decode buckets 1, 2, 4, 8 and 12.
# - mtp-batch24: up to 24 requests, 32 seats: decode buckets 1, 2, 4, 16 and 24, speculation up to 4 requests.
PRESETS: dict[str, dict] = {
    "batch12": {
        "max_num_seqs": 16,
        "decode_buckets": (1, 2, 4, 6, 8, 12),
        "max_running": 12,
        "kda_shell_max_rows": 12,
        "dsa_layer_max_rows": 16,
        "ffn_half_below": 17,
        "dsa_index_rows": 8,
    },
    "batch8": {
        "max_num_seqs": 8,
        "decode_buckets": (1, 2, 4, 6, 8),
        "max_running": None,
        "kda_shell_max_rows": 8,
        "dsa_layer_max_rows": 8,
        "ffn_half_below": 16,
        "dsa_index_rows": 0,
    },
}

_MTP = {
    "sp_gather": "nki",
    "sp_gather_tp_experts": True,
    "dsa_sel2g": True,
    "spec_method": "mtp",
    "spec_k": 2,
    "max_num_seqs": 16,
    "prefill_buckets": (512, 1024, 2048, 4096),
    "mtp_chunk_q": 64,
    "mtp_fill_rows": 1024,
    "kda_shell_max_rows": 12,
    "dsa_layer_max_rows": 16,
    "ffn_half_below": 17,
    "dsa_index_rows": 8,
}
_MTP_DYN = {
    **_MTP,
    "page_buckets": (1024, 3072, 4096),
    "spec_max_batch": 4,
    "spec_hysteresis": 1,
    "spec_bucket_max": 4,
    "plain_bucket_min": 4,
    "spec_page_buckets": (1024, 4096),
    "mtp_kv_only": True,
    "dyn_mtp_kv": True,
}
PRESETS["mtp-batch8"] = {
    **_MTP_DYN,
    "decode_buckets": (1, 2, 4, 6, 8),
    "max_running": 8,
    "state_seats": 8,
    "state_checkpoints": 8,
}
PRESETS["mtp-batch2"] = {
    **_MTP,
    "page_buckets": (1024, 4096),
    "prefill_buckets": (512, 1024, 4096),
    "decode_buckets": (1, 2, 4, 6, 8),
    "max_running": 8,
    "state_seats": 8,
    "state_checkpoints": 8,
}
PRESETS["mtp-batch12"] = {
    **_MTP_DYN,
    "decode_buckets": (1, 2, 4, 8, 12),
    "max_running": 12,
    "state_seats": 12,
    "state_spec_seats": 4,
    "state_checkpoints": 12,
    "host_cache_gb": 8.0,
}
PRESETS["mtp-batch24"] = {
    **_MTP_DYN,
    "max_num_seqs": 32,
    "decode_buckets": (1, 2, 4, 16, 24),
    "max_running": 24,
    "state_seats": 24,
    "state_spec_seats": 4,
    "state_checkpoints": 24,
    "plain_bucket_min": 16,
    "host_cache_gb": 12.0,
    "dsa_layer_max_rows": 24,
    "ffn_half_below": 25,
    "dense_half_skip_rows": (2, 18, 24),
}


def add_engine_args(ap: argparse.ArgumentParser) -> None:
    """The engine's arguments, with the served values as defaults."""
    g = ap.add_argument_group("engine")
    g.add_argument(
        "--preset",
        default=None,
        choices=sorted(PRESETS),
        help="a named configuration (docs/CONFIGURATION.md): its values become the defaults of the "
        "arguments it sets; an argument given explicitly still wins",
    )
    g.add_argument("--model", required=True, help="the checkpoint directory (or a Hugging Face repo id)")
    g.add_argument(
        "--revision",
        default=None,
        help="with a Hugging Face repo id as --model: the commit (or branch, tag) to download; default: the "
        "repo's default branch. Host only: no graph changes",
    )
    g.add_argument("--device", default="neuron", choices=["neuron", "cpu"])
    g.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    g.add_argument("--tp", type=int, default=32, help="tensor-parallel ranks per engine, one logical NeuronCore each")
    g.add_argument("--max-model-len", type=int, default=1_048_576)
    g.add_argument("--page-size", type=int, default=256)
    g.add_argument(
        "--max-num-seqs",
        type=int,
        default=16,
        help="request seats per engine: the state pool's rows, a graph shape (--max-running caps how many run at once)",
    )
    g.add_argument(
        "--max-running",
        type=_at_least(1),
        default=12,
        help="at most this many requests running at once per engine, the rest waiting in arrival order "
        "(default 12; --preset batch8: --max-num-seqs). Host only: no graph changes",
    )
    g.add_argument("--max-prefill-tokens", type=int, default=4096, help="prefill tokens per step (the chunk size)")
    g.add_argument("--decode-buckets", type=_ints, default=(1, 2, 4, 6, 8, 12))
    g.add_argument("--prefill-buckets", type=_ints, default=(512, 1024, 2048, 4096))
    g.add_argument("--page-buckets", type=_ints, default=(512, 1024, 4096))
    g.add_argument("--prefill-page-buckets", type=_ints, default=(1024, 4096))
    g.add_argument("--prefill-page-min-rows", type=int, default=2048)
    g.add_argument("--kv-cache-gb", type=float, default=4.0, help="KV pool per rank, GiB")
    g.add_argument("--kv-cache-dtype", default="fp8", choices=["auto", "fp8"])
    g.add_argument("--state-checkpoints", type=int, default=16, help="state-pool rows kept for prefix checkpoints")
    g.add_argument(
        "--host-cache-gb", type=float, default=4.0, help="host memory under the prefix cache per rank, GiB (0: off)"
    )
    g.add_argument(
        "--moe-kernel",
        default=None,
        choices=["nki", "xla"],
        help="the selected experts' MoE: the NKI kernels (the served path) or XLA's gather (default: nki on "
        "Neuron, xla on --device cpu, where NKI kernels do not run)",
    )
    g.add_argument("--num-layers", type=int, default=None, help="debugging: the first n layers only")
    sp = ap.add_argument_group(
        "speculative decoding",
        "the model's MTP layer drafts --spec-k tokens a step and the target verifies them in the next step "
        "(docs/AGENTX.md). Graph structure: a change needs a new capture. Supported: the --preset mtp-* values",
    )
    sp.add_argument("--spec-method", default=None, choices=["mtp"], help="speculative decoding with the MTP layer")
    sp.add_argument("--spec-k", type=_at_least(1), default=2, help="drafts per step")
    sp.add_argument(
        "--spec-synth-al",
        type=float,
        default=None,
        help="benchmark mode: draw each verify's accepted drafts from a schedule with this mean acceptance length "
        "(1 + the accepted drafts a step, in [1, spec-k + 1]) instead of the draft head's own; for comparing serving "
        "throughput at a published acceptance length, not for quality. Default: the TRNRT_SPEC_SYNTH_AL "
        "environment variable, else off",
    )
    sp.add_argument(
        "--spec-max-batch",
        type=_at_least(0),
        default=0,
        help="speculate while at most this many requests decode, plain decode steps above it (0: always speculate)",
    )
    sp.add_argument(
        "--spec-hysteresis",
        type=_at_least(0),
        default=1,
        help="once plain, speculate again only when at most spec-max-batch minus this many requests decode",
    )
    sp.add_argument(
        "--spec-bucket-max",
        type=_at_least(0),
        default=0,
        help="with --spec-max-batch: the verify family's graphs only at decode buckets up to this (0: all)",
    )
    sp.add_argument(
        "--plain-bucket-min",
        type=_at_least(0),
        default=0,
        help="with --spec-max-batch: the plain decode graphs only at decode buckets from this (0: all)",
    )
    sp.add_argument(
        "--spec-page-buckets",
        type=_ints,
        default=None,
        help="the page buckets of every graph the MTP layer or a verify runs in (default: --page-buckets)",
    )
    sp.add_argument(
        "--mtp-fill-rows",
        type=_at_least(1),
        default=None,
        help="the MTP pass over a prefill chunk in calls of this many rows",
    )
    sp.add_argument(
        "--mtp-chunk-q", type=_at_least(0), default=0, help="the MTP pass takes the prefill form from this many rows"
    )
    sp.add_argument(
        "--mtp-kv-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="MTP windows of a prefill chunk that hold no last position write only the draft layer's KV",
    )
    sp.add_argument(
        "--dyn-mtp-kv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="plain decode steps write the draft layer's KV for their rows",
    )
    sp.add_argument(
        "--state-seats",
        type=_at_least(1),
        default=None,
        help="seats of the recurrent-state pool, apart from --max-num-seqs (default: --max-num-seqs)",
    )
    sp.add_argument(
        "--state-spec-seats",
        type=_at_least(1),
        default=None,
        help="of the state seats, how many hold the rows a verify writes (1 + spec-k); the rest hold one",
    )
    k = ap.add_argument_group(
        "decode kernel routes",
        "the row counts at which a decode step takes each fused kernel. "
        "Graph structure: a change needs a new capture. Supported: the defaults (the served "
        "configuration) and --preset batch8's values",
    )
    k.add_argument(
        "--kda-shell-max-rows",
        type=_at_least(1),
        default=12,
        help="KDA layers' decode attention as the KDA shell's kernel up to this many rows",
    )
    k.add_argument(
        "--dsa-layer-max-rows",
        type=_at_least(1),
        default=16,
        help="DSA layers' decode attention and FFN as the fused layer's one call up to this many rows",
    )
    k.add_argument(
        "--ffn-half-below",
        type=_at_least(1),
        default=17,
        help="decode FFN blocks as the FFN half's one call below this many rows",
    )
    k.add_argument(
        "--dsa-index-rows",
        type=_at_least(0),
        default=8,
        help="the context-parallel DSA indexer's decode scores in kernel calls of at most this many rows (0: one call)",
    )
    k.add_argument(
        "--sp-gather",
        default="xla",
        choices=["xla", "nki"],
        help="the sequence-parallel stream's all-gather of a decode block's output: the XLA collective or the NKI kernel",
    )
    k.add_argument(
        "--sp-gather-tp-experts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="the NKI all-gather over the tensor-parallel group the experts run in",
    )
    k.add_argument(
        "--dsa-sel2g",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="the DSA layer's two-stage selection in one kernel (needed by a page ladder with a 3072-page rung)",
    )
    k.add_argument(
        "--dense-half-skip-rows",
        type=_ints,
        default=(2,),
        help="the decode row counts at which a dense MLP layer runs as the XLA graph instead of the dense half's kernel",
    )


def _env_synth_al() -> float | None:
    """TRNRT_SPEC_SYNTH_AL: the benchmark mode's mean acceptance length, when --spec-synth-al is not given."""
    v = os.environ.get("TRNRT_SPEC_SYNTH_AL")
    return float(v) if v else None


def parse_engine_args(argv: list[str]) -> argparse.Namespace:
    """The engine arguments of `argv` (serve's, engine's, or capture's after its `--`), its --preset applied: the
    preset's values are the defaults, arguments given explicitly win."""
    # No abbreviations: the server's own arguments pass through here unparsed, and one that is a prefix of an engine
    # argument would be read as it (serve's `--host 127.0.0.1` for each engine as --host-cache-gb).
    ap = argparse.ArgumentParser(prog="trnrt", allow_abbrev=False)
    add_engine_args(ap)
    a, _ = ap.parse_known_args(argv)  # the server's own arguments (--port, --engines, ...) are not the engine's
    if a.preset is not None:
        ap.set_defaults(**PRESETS[a.preset])
        a, _ = ap.parse_known_args(argv)
    return a


def engine_config(argv: list[str], core_base: int = 0):
    """The EngineConfig of the serve or engine arguments `argv` (also `compile capture`'s, after its `--`)."""
    import torch

    from .config import EngineConfig

    a = parse_engine_args(argv)
    return EngineConfig(
        model_path=a.model,
        model_revision=a.revision,
        device=a.device,
        dtype=getattr(torch, a.dtype),
        kv_cache_dtype=a.kv_cache_dtype,
        page_size=a.page_size,
        kv_cache_gb=a.kv_cache_gb,
        max_num_seqs=a.max_num_seqs,
        max_model_len=a.max_model_len,
        max_prefill_tokens=a.max_prefill_tokens,
        decode_batch_buckets=a.decode_buckets,
        prefill_token_buckets=a.prefill_buckets,
        page_buckets=a.page_buckets,
        prefill_page_buckets=a.prefill_page_buckets,
        prefill_page_min_rows=a.prefill_page_min_rows,
        host_cache_gb=a.host_cache_gb,
        state_checkpoints=a.state_checkpoints,
        tp=a.tp,
        tp_core_base=core_base,
        moe_kernel=a.moe_kernel,
        num_layers=a.num_layers,
        max_running=a.max_running,
        kda_shell_max_rows=a.kda_shell_max_rows,
        dsa_layer_max_rows=a.dsa_layer_max_rows,
        ffn_half_below=a.ffn_half_below,
        dsa_index_rows=a.dsa_index_rows,
        dense_half_skip_rows=tuple(a.dense_half_skip_rows),
        sp_gather=a.sp_gather,
        sp_gather_tp_experts=a.sp_gather_tp_experts,
        dsa_sel2g=a.dsa_sel2g,
        spec_method=a.spec_method,
        spec_k=a.spec_k,
        spec_synth_al=a.spec_synth_al if a.spec_synth_al is not None else _env_synth_al(),
        spec_max_batch=a.spec_max_batch,
        spec_hysteresis=a.spec_hysteresis,
        spec_bucket_max=a.spec_bucket_max,
        plain_bucket_min=a.plain_bucket_min,
        spec_page_buckets=a.spec_page_buckets,
        mtp_fill_rows=a.mtp_fill_rows,
        mtp_chunk_q=a.mtp_chunk_q,
        mtp_kv_only=a.mtp_kv_only,
        dyn_mtp_kv=a.dyn_mtp_kv,
        state_seats=a.state_seats,
        state_spec_seats=a.state_spec_seats,
    )


def add_compile_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--allow-compile",
        action="store_true",
        help="compile a graph missing from the compile cache here (neuronx-cc: minutes to hours per graph) "
        "instead of failing start-up",
    )


def compile_env(allow_compile: bool) -> dict[str, str]:
    """The engines' environment for the compile cache: NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 unless --allow-compile (then
    0, which also overrides a 1 in the caller's environment)."""
    return {CACHE_HIT_ENV: "0" if allow_compile else "1"}


def cache_miss(e: BaseException | None) -> str | None:
    """The line of `e`, or of an exception it chains (torch.compile wraps its backend's), in which libtorch-neuronx-lite
    refuses to compile a graph missing from the compile cache ("NEURON_LIBTORCH_ASSERT_CACHE_HIT: compile cache miss for
    key ..."); None if there is none."""
    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        for line in str(e).splitlines():
            if CACHE_HIT_ENV in line and "miss" in line.lower():
                return line.strip()
        e = e.__cause__ or e.__context__
    return None


def cache_miss_help() -> str:
    root = os.environ.get("NEURON_LIBTORCH_CACHE_ROOT")
    return (
        "A graph this engine needs is not in the compile cache "
        f"(NEURON_LIBTORCH_CACHE_ROOT={root or 'unset: ~/.cache/neuron_libtorch'}), and the engine does "
        "not compile on the serving host. The cache was most likely built for other engine arguments (the "
        "arguments after `compile capture --` must be the server's), another checkpoint revision, another trnrt "
        "version or another Neuron SDK; or NEURON_LIBTORCH_CACHE_ROOT does not point at it. Rebuild the cache for "
        "this server (python -m trnrt compile), or pass --allow-compile to compile the missing graphs here "
        "(minutes to hours per graph)."
    )


def cmd_engine(argv: list[str]) -> int:
    # allow_abbrev=False here too: an abbreviation this parser took would reach engine_config, which ignores it
    ap = argparse.ArgumentParser(
        prog="python -m trnrt engine", description="One engine and its OpenAI API.", allow_abbrev=False
    )
    add_engine_args(ap)
    ap.add_argument("--core-base", type=int, default=0, help="the engine's first logical NeuronCore")
    ap.add_argument("--host", default="0.0.0.0")  # noqa: S104  an API server: reachable by design
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--served-model-name", default="glm")
    add_compile_arg(ap)
    a = ap.parse_args(argv)
    cfg = engine_config(argv, a.core_base)
    from .engine import start_checks

    why = start_checks.refusal(cfg.device, a.core_base, cfg.tp)  # its devices free, enough contiguous host memory
    if why is not None:
        print(f"trnrt engine: not starting: {why}", file=sys.stderr, flush=True)
        return START_REFUSED_STATUS
    os.environ.update(compile_env(a.allow_compile))  # before the engine spawns its ranks: they inherit it
    from .engine.engine import LLMEngine
    from .serve.api import build_app, serve

    t = time.perf_counter()
    try:
        eng = LLMEngine(cfg)
        print(
            f"engine up in {time.perf_counter() - t:.1f} s (cores {a.core_base}-{a.core_base + cfg.tp - 1})", flush=True
        )
        w = eng.warmup()
    except Exception as e:
        miss = cache_miss(e)
        if miss is None:
            raise
        print(f"trnrt engine: {miss}\n{cache_miss_help()}", file=sys.stderr, flush=True)
        return CACHE_MISS_STATUS
    print(f"bucket warmup {w['seconds']:.1f} s over {w['graphs']} graphs", flush=True)
    # The router keeps its connections to the engines alive and drops idle ones after 30 s: uvicorn closing them first
    # (its default keep-alive is 5 s) raced a request onto a closing connection.
    serve(build_app(eng, a.served_model_name), host=a.host, port=a.port, log_level="warning", timeout_keep_alive=600)
    return 0


def _dies_with_parent() -> None:
    """In a child before exec: SIGTERM when the parent exits (Linux PR_SET_PDEATHSIG), so an engine holding its
    NeuronCores never outlives the server that started it."""
    if sys.platform.startswith("linux"):
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def _without(argv: list[str], flag: str) -> list[str]:
    """argv without `flag VALUE` / `flag=VALUE`."""
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
        elif x == flag:
            skip = True
        elif not x.startswith(flag + "="):
            out.append(x)
    return out


def _start_failed(i: int, status: int) -> str:
    if status == CACHE_MISS_STATUS:
        return (
            f"engine {i} stopped at start-up: a graph it needs is not in the compile cache (its message above "
            "names the key and the likely cause). Rebuild the cache for these arguments, or pass --allow-compile."
        )
    if status == START_REFUSED_STATUS:
        return (
            f"engine {i} did not start: a check of the host refused it (its message above says what it found and "
            "how to fix it)"
        )
    return f"engine {i} exited during start-up (status {status}); see its log above"


def _wait_healthy(url: str, procs: list[subprocess.Popen], timeout_s: float, poll_s: float) -> float:
    """Wait for GET `url`/health to answer 200 and return the seconds that took. SystemExit (status 1, with the reason)
    if one of `procs` (the engines started so far, engine i at index i) exits first, or after `timeout_s` (0: never)."""
    import httpx

    t0 = time.monotonic()
    while True:
        for i, p in enumerate(procs):
            if p.poll() is not None:
                raise SystemExit(_start_failed(i, p.returncode))
        try:
            if httpx.get(url + "/health", timeout=2).status_code == 200:
                return time.monotonic() - t0
        except httpx.HTTPError:
            pass
        if timeout_s and time.monotonic() - t0 >= timeout_s:
            raise SystemExit(
                f"{url} did not answer /health within --startup-timeout {timeout_s:g} s: stopping the engines"
            )
        time.sleep(poll_s)


def cmd_serve(argv: list[str]) -> int:
    # allow_abbrev=False: the engines get this argv (less --engines and --startup-timeout, by their full names), and an
    # abbreviation would pass through to them unrecognized
    ap = argparse.ArgumentParser(
        prog="python -m trnrt serve",
        allow_abbrev=False,
        description="Engines on disjoint NeuronCores and the session-affine router in front.",
    )
    add_engine_args(ap)
    ap.add_argument("--engines", type=int, default=2, help="engines of --tp ranks each, on consecutive cores")
    ap.add_argument("--host", default="0.0.0.0")  # noqa: S104  an API server: reachable by design
    ap.add_argument("--port", type=int, default=8000, help="the router's port; engine i listens on port + 1 + i")
    ap.add_argument("--served-model-name", default="glm")
    ap.add_argument(
        "--timeout-keep-alive",
        type=int,
        default=75,
        help="seconds the router keeps an idle client connection open. Above common client pools' "
        "keep-alive (aiohttp: 15 s): at uvicorn's default 5 s a client could send on a connection the "
        "router was closing",
    )
    ap.add_argument(
        "--startup-timeout",
        type=float,
        default=3600.0,
        help="seconds each engine has, from its start, to load and warm every graph and answer /health "
        "(0: no limit); past it every engine is stopped and serve exits 1",
    )
    add_compile_arg(ap)
    a = ap.parse_args(argv)
    # The engines start one at a time, each healthy (its weights loaded, every graph warmed) before the next starts. The
    # host-wide start lock stays on too (engine/start_lock.py: each graph's first call holds it): two engines whose
    # ranks load collective graphs at the same moment collide in the runtime's collectives bootstrap.
    env = dict(os.environ, **compile_env(a.allow_compile))
    # A fixed path: the engines' processes coordinate through it, so it cannot be a private temp file.
    env.setdefault("TRNRT_START_LOCK", os.path.join("/tmp", f"trnrt-start-lock-{a.port}"))  # noqa: S108
    # --engines, --startup-timeout and --timeout-keep-alive are serve's own; --allow-compile goes on: engine takes it
    engine_argv = _without(_without(_without(argv, "--engines"), "--startup-timeout"), "--timeout-keep-alive")
    procs: list[subprocess.Popen] = []
    urls = []
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))  # unwind through the finally below
    try:
        for i in range(a.engines):
            port = a.port + 1 + i
            cmd = [
                sys.executable,
                "-m",
                "trnrt",
                "engine",
                *engine_argv,
                "--core-base",
                str(i * a.tp),
                "--port",
                str(port),
                "--host",
                "127.0.0.1",
            ]
            procs.append(
                subprocess.Popen(
                    cmd,
                    env=dict(env, TRNRT_START_LOCK_OWNER=f"engine{i}"),
                    preexec_fn=_dies_with_parent,  # noqa: PLW1509  PR_SET_PDEATHSIG has no subprocess option; no thread runs here
                )
            )
            urls.append(f"http://127.0.0.1:{port}")
            s = _wait_healthy(urls[-1], procs, a.startup_timeout, HEALTH_POLL_S)
            print(f"engine {i} ({urls[-1]}) ready after {s:.0f} s", flush=True)
        import uvicorn

        from .serve.router import Router, build_router_app

        uvicorn.run(build_router_app(Router(urls)), host=a.host, port=a.port, timeout_keep_alive=a.timeout_keep_alive)
    finally:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in procs:
            try:
                p.wait(120)  # an engine's close stops its 31 workers
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


def cmd_compile(argv: list[str]) -> int:
    from .compile import farm

    return farm.main(argv, engine_config=engine_config)


def main(argv: list[str] | None = None) -> int:
    log.configure()  # once per process: `serve` starts each engine as a process of its own, which runs this main too
    argv = sys.argv[1:] if argv is None else argv
    cmds = {"serve": cmd_serve, "engine": cmd_engine, "compile": cmd_compile}
    if not argv or argv[0] not in cmds:
        print(__doc__.strip())
        return 0 if argv[:1] in ([], ["-h"], ["--help"]) else 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":  # spawned capture ranks re-import this module: only a direct run starts a command
    sys.exit(main())
