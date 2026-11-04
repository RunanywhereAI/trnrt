"""Prometheus text exposition for the server's GET /metrics, no client library needed.

Names follow vLLM's metric set (docs/design/metrics.md) under a `trnrt:` prefix, so an existing vLLM dashboard maps
by replacing the prefix. Start-up and per-graph metrics follow vllm-neuron's ("Neuron-specific metrics" in its
features guide), with the engine's graph keys as the bucket label.
"""

from __future__ import annotations

import bisect
import threading

LATENCY_BUCKETS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96)


class Histogram:
    def __init__(self, buckets=LATENCY_BUCKETS):
        self.buckets = buckets
        self.counts = [0] * (len(buckets) + 1)
        self.sum = 0.0
        self.n = 0

    def observe(self, v: float) -> None:
        self.counts[bisect.bisect_left(self.buckets, v)] += 1
        self.sum += v
        self.n += 1

    def lines(self, name: str) -> list[str]:
        out, acc = [], 0
        for b, c in zip(self.buckets, self.counts):
            acc += c
            out.append(f'{name}_bucket{{le="{b}"}} {acc}')
        out.append(f'{name}_bucket{{le="+Inf"}} {self.n}')
        out.append(f"{name}_sum {self.sum}")
        out.append(f"{name}_count {self.n}")
        return out


def kv_pages(engine) -> tuple[int, int]:
    """(KV pages in use, usable pages) of the engine's page pool; pages that hold only an evictable cached prefix count
    as free, since a new request can take them."""
    pool = engine.pool
    return pool.num_usable - pool.num_free - engine.radix.evictable_pages, pool.num_usable


class Metrics:
    """Request counters and latency histograms, updated by the engine thread as requests finish (on_finish) and read
    by the HTTP handler (render)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.prompt_tokens = 0
        self.generation_tokens = 0
        self.prefix_cache_queries = 0  # prompt tokens looked up
        self.prefix_cache_hits = 0  # of which served from the cache
        self.finished: dict[str, int] = {}
        self.ttft = Histogram()
        self.tpot = Histogram()
        self.e2e = Histogram((*LATENCY_BUCKETS, 81.92, 163.84))
        # The local-K selection's counters (LLMEngine.local_k_counts) as the engine thread last read them, and whether
        # a render asked for them since. They are device buffers the graphs write, so only the engine thread reads
        # them, and only while it is idle (serve/api.py EngineLoop): a scrape shows the counts of the last idle moment
        # after the scrape before it.
        self.local_k: list[float] | None = None
        self.local_k_wanted = False

    def set_local_k(self, counts: list[float] | None) -> None:
        with self.lock:
            self.local_k, self.local_k_wanted = counts, False

    def on_finish(self, req) -> None:
        with self.lock:
            n_out = len(req.output_ids)
            self.prompt_tokens += req.num_prompt
            self.generation_tokens += n_out
            self.prefix_cache_queries += req.num_prompt
            self.prefix_cache_hits += max(req.num_cached_tokens, 0)
            reason = req.finish_reason or "unknown"
            self.finished[reason] = self.finished.get(reason, 0) + 1
            if req.first_token_time is not None:
                self.ttft.observe(req.first_token_time - req.arrival_time)
                if req.finish_time is not None and n_out > 1:
                    self.tpot.observe((req.finish_time - req.first_token_time) / (n_out - 1))
            if req.finish_time is not None:
                self.e2e.observe(req.finish_time - req.arrival_time)

    def render(self, engine) -> str:
        s = engine.scheduler
        used, usable = kv_pages(engine)
        lines = [
            "# TYPE trnrt:num_requests_running gauge",
            f"trnrt:num_requests_running {len(s.running)}",
            "# TYPE trnrt:num_requests_waiting gauge",
            f"trnrt:num_requests_waiting {len(s.waiting)}",
            "# TYPE trnrt:kv_cache_usage_perc gauge",
            f"trnrt:kv_cache_usage_perc {used / usable:.6f}",
            "# TYPE trnrt:num_preemptions_total counter",
            f"trnrt:num_preemptions_total {s.num_preemptions}",
        ]
        lines += startup_lines(engine)
        with self.lock:
            lines += [
                "# TYPE trnrt:prompt_tokens_total counter",
                f"trnrt:prompt_tokens_total {self.prompt_tokens}",
                "# TYPE trnrt:generation_tokens_total counter",
                f"trnrt:generation_tokens_total {self.generation_tokens}",
                "# TYPE trnrt:prefix_cache_queries_total counter",
                f"trnrt:prefix_cache_queries_total {self.prefix_cache_queries}",
                "# TYPE trnrt:prefix_cache_hits_total counter",
                f"trnrt:prefix_cache_hits_total {self.prefix_cache_hits}",
                "# TYPE trnrt:request_success_total counter",
            ]
            lines += [
                f'trnrt:request_success_total{{finished_reason="{k}"}} {v}' for k, v in sorted(self.finished.items())
            ]
            for name, h in (
                ("trnrt:time_to_first_token_seconds", self.ttft),
                ("trnrt:time_per_output_token_seconds", self.tpot),
                ("trnrt:e2e_request_latency_seconds", self.e2e),
            ):
                lines.append(f"# TYPE {name} histogram")
                lines += h.lines(name)
            lines += local_k_lines(self.local_k)
            self.local_k_wanted = True
        return "\n".join(lines) + "\n"


def local_k_lines(counts: list[float] | None) -> list[str]:
    """The context-parallel prefill's local-K selection (models/mla.py CP_LOCAL_K; engine.local_k_counts), over the
    engine's run so far, its warmup included, on rank 0's row group: rows selected, rows that failed the certificate
    (selected again from every rank's whole list), and failed rows past CP_LOCAL_F, whose selection was not exact (0 on
    an exact run). None (an engine without the counters, or none read yet): no lines."""
    if counts is None:
        return []
    out = []
    for name, v in zip(
        ("trnrt:dsa_local_k_rows_total", "trnrt:dsa_local_k_failed_rows_total", "trnrt:dsa_local_k_inexact_rows_total"),
        counts,
    ):
        out += [f"# TYPE {name} counter", f"{name} {v:.0f}"]
    return out


def _bucket(key) -> str:
    """A graph key as vllm-neuron's bucket_name label (e.g. prefill_s1024): the key's parts joined by "_"."""
    return "_".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))


def startup_lines(engine) -> list[str]:
    """Start-up and per-graph metrics, each emitted only when the engine records it:
    - startup_time_seconds: engine construction to the end of warmup (engine.startup_seconds);
    - model_load_time_seconds: the shard load (engine.load_seconds);
    - model_load_size_bytes: rank 0's parameters on the device (engine.model);
    - compilation_time_seconds{bucket_name}: a graph's first call, a compile or the load of a cached NEFF
      (engine.runner.compile_seconds);
    - neff_execution_count{bucket_name}: graph executions (engine.runner.exec_counts)."""
    runner = getattr(engine, "runner", None)
    out = []
    if getattr(engine, "startup_seconds", None) is not None:
        out += ["# TYPE trnrt:startup_time_seconds gauge", f"trnrt:startup_time_seconds {engine.startup_seconds:.3f}"]
    if getattr(engine, "load_seconds", None) is not None:
        out += [
            "# TYPE trnrt:model_load_time_seconds gauge",
            f"trnrt:model_load_time_seconds {engine.load_seconds:.3f}",
        ]
    model = getattr(engine, "model", None)
    if model is not None and hasattr(model, "parameters"):
        n = sum(p.numel() * p.element_size() for p in model.parameters())
        out += ["# TYPE trnrt:model_load_size_bytes gauge", f"trnrt:model_load_size_bytes {n}"]
    if runner is not None:
        cs = dict(getattr(runner, "compile_seconds", {}))
        if cs:
            out.append("# TYPE trnrt:compilation_time_seconds gauge")
            out += [f'trnrt:compilation_time_seconds{{bucket_name="{_bucket(k)}"}} {v:.3f}' for k, v in cs.items()]
        ec = dict(getattr(runner, "exec_counts", {}))
        if ec:
            out.append("# TYPE trnrt:neff_execution_count counter")
            out += [f'trnrt:neff_execution_count{{bucket_name="{_bucket(k)}"}} {v}' for k, v in ec.items()]
    return out
