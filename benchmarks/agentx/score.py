"""Score one AgentX run (agentx-harness's AIPerf artifact directory) with InferenceX's AgentX metric definitions.

    python3 -I benchmarks/agentx/score.py ARTIFACT_DIR [--chips 8] [--price NAME=USD_PER_CHIP_HOUR ...] [--label L]

A restatement of InferenceX's scoring (SemiAnalysisAI/InferenceX, inferencex-e2e/infx/results/agentic:
artifacts.load_records_with_accounting, request_metrics.compute_latency_stats / compute_throughput_stats,
common.percentile), not its code. InferenceX's own code is the authority; this exists so a run can be scored without it,
and it gave the same numbers as the scores we published for the same artifacts.

- Records come from `profile_export.jsonl`. A record is dropped if its `metadata.benchmark_phase` is set and is not
  "profiling" (warmup), or if it carries an `error`; the rest are the profiled records.
- duration = the last profiled request's end - the first profiled request's start; requests per minute = profiled /
  duration x 60.
- Throughput = the profiled records' input_sequence_length (cached input included) and output_sequence_length summed, /
  duration; per chip = / --chips (8 for one engine, half a trn2.48xlarge).
- Time to first token, end-to-end latency and inter-token latency: per record, ms -> s, finite and > 0 only; their
  mean and linearly interpolated p50 / p75 / p90 / p95.
- Interactivity (tokens / s / user) at a statistic = 1 / that statistic of the inter-token latency, so P90
  interactivity = 1 / p90 ITL: the slow decile, the x axis of InferenceX's AgentX page.
- Tokens per dollar (millions) at each --price = total tokens / s / chip x 3,600 / price per chip-hour / 10^6, with
  the defaults of benchmarks/prices.py (SemiAnalysis's $0.58 and AWS's Capacity Block $2.235 per chip-hour).

Prints one JSON object.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prices import DEFAULT_PRICES, parse_price, positive_int, tokens_per_usd_millions  # after the path is set


def percentile(data: list[float], p: float) -> float:
    """Linear interpolation between closest ranks (numpy's default), as InferenceX's common.percentile."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    return s[f] if f + 1 >= len(s) else s[f] + (k - f) * (s[f + 1] - s[f])


def stats(v: list[float]) -> dict[str, float]:
    if not v:
        return {}
    return {"mean": statistics.mean(v), **{f"p{p}": percentile(v, p) for p in (50, 75, 90, 95)}}


def _metric(rec: dict, key: str):
    m = rec.get("metrics", {}).get(key)
    return m.get("value") if isinstance(m, dict) else m


def _seconds(recs: list[dict], key: str) -> list[float]:
    out = []
    for r in recs:
        try:
            v = float(_metric(r, key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0:
            out.append(v / 1000.0)
    return out


def _total(recs: list[dict], key: str) -> int:
    """The sum of a token count over the profiled records; a record without one is refused, not counted as 0 (that
    would understate the throughput with no sign)."""
    n = 0
    for r in recs:
        v = _metric(r, key)
        try:
            n += int(v)
        except (TypeError, ValueError, OverflowError):  # OverflowError: an infinite count (JSON's Infinity)
            raise SystemExit(f"a profiled record has no {key} ({v!r}): {json.dumps(r)[:200]}") from None
    return n


def load(artifact_dir: Path) -> tuple[list[dict], dict]:
    """The profiled records and the accounting of the dropped ones."""
    recs: list[dict] = []
    acc: dict = {"total": 0, "warmup_dropped": 0, "error_dropped": 0, "errors": {}}
    lines = (artifact_dir / "profile_export.jsonl").read_text().splitlines()
    records = [(i, ln) for i, ln in enumerate(lines, 1) if ln.strip()]  # the file's own line numbers, blanks skipped
    for i, ln in records:
        try:
            o = json.loads(ln)
        except ValueError:
            if i == records[-1][0]:  # a half-written last record of a file still being written
                continue
            raise SystemExit(f"profile_export.jsonl line {i} of {len(lines)} is not JSON: {ln[:120]!r}") from None
        acc["total"] += 1
        phase = o.get("metadata", {}).get("benchmark_phase")
        warm = phase is not None and phase != "profiling"
        err = o.get("error")
        if warm:
            acc["warmup_dropped"] += 1
        if err:
            acc["error_dropped"] += 1
            kind = (
                (err.get("type") or err.get("error_type") or err.get("code") or "unknown")
                if isinstance(err, dict)
                else str(err)[:40]
            )
            acc["errors"][str(kind)] = acc["errors"].get(str(kind), 0) + 1
        if not (err or warm):
            recs.append(o)
    return recs, acc


def score(artifact_dir: Path, chips: int, prices: dict[str, float], label: str | None = None) -> dict:
    recs, acc = load(artifact_dir)
    if not recs:
        raise SystemExit(f"{artifact_dir}: no profiled records")
    starts = [int(r["metadata"]["request_start_ns"]) for r in recs if r.get("metadata", {}).get("request_start_ns")]
    ends = [int(r["metadata"]["request_end_ns"]) for r in recs if r.get("metadata", {}).get("request_end_ns")]
    dur = (max(ends) - min(starts)) / 1e9
    isl, osl = _total(recs, "input_sequence_length"), _total(recs, "output_sequence_length")
    ttft = stats(_seconds(recs, "time_to_first_token"))
    e2e = stats(_seconds(recs, "request_latency"))
    itl = stats(_seconds(recs, "inter_token_latency"))
    intv = {k: 1.0 / v for k, v in itl.items() if v > 0}
    tps_chip = (isl + osl) / dur / chips
    return {
        "label": label or artifact_dir.name,
        "chips": chips,
        "profiled": len(recs),
        **acc,
        "duration_s": round(dur, 1),
        "requests_per_min": round(len(recs) / dur * 60, 3),
        "input_tok_s": round(isl / dur, 1),
        "output_tok_s": round(osl / dur, 2),
        "total_tok_s_per_chip": round(tps_chip, 1),
        "output_tok_s_per_chip": round(osl / dur / chips, 3),
        "p90_interactivity_tok_s_per_user": round(intv.get("p90", 0.0), 3),
        "mean_interactivity_tok_s_per_user": round(intv.get("mean", 0.0), 2),
        "ttft_p50_s": round(ttft.get("p50", 0.0), 2),
        "ttft_p90_s": round(ttft.get("p90", 0.0), 2),
        "e2e_p50_s": round(e2e.get("p50", 0.0), 2),
        "e2e_p90_s": round(e2e.get("p90", 0.0), 2),
        "itl_p50_ms": round(itl.get("p50", 0.0) * 1000, 2),
        "itl_p90_ms": round(itl.get("p90", 0.0) * 1000, 3),
        "mean_input_tokens": round(isl / len(recs)),
        "mean_output_tokens": round(osl / len(recs)),
        "price_usd_per_chip_hour": prices,
        "total_tokens_per_usd_millions": tokens_per_usd_millions(tps_chip, prices),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("artifact_dir", type=Path)
    ap.add_argument("--chips", type=positive_int, default=8, help="chips behind the served endpoint (8: one engine)")
    ap.add_argument("--price", type=parse_price, action="append", default=None, help="NAME=USD_PER_CHIP_HOUR (repeat)")
    ap.add_argument("--label", default=None)
    a = ap.parse_args(argv)
    prices = dict(a.price) if a.price else dict(DEFAULT_PRICES)
    print(json.dumps(score(a.artifact_dir, a.chips, prices, a.label)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
