"""Score one fixed-length run (InferenceX's fixed_seq recipe, benchmark_serving's result JSON) per chip and per dollar.

    python3 -I benchmarks/fixed_seq/score.py RESULT_JSON [--chips 8] [--price NAME=USD_PER_CHIP_HOUR ...] [--label L]

- Throughput per chip = benchmark_serving's total_token_throughput (input + output tokens / s over the run) / --chips
  (8 for one engine, half a trn2.48xlarge); output per chip likewise from output_throughput.
- Tokens / s / user at a percentile = 1,000 / that percentile of TPOT (ms): P90 per-user speed = 1000 / p90_tpot_ms,
  the slow decile, the x axis of InferenceX's fixed-length pages.
- Tokens per dollar (millions) at each --price = total tokens / s / chip x 3,600 / price per chip-hour / 10^6, with the
  defaults of benchmarks/prices.py (SemiAnalysis's $0.58 and AWS's Capacity Block $2.235 per chip-hour).

Prints one JSON object. A run with failed requests is reported with them (`failed`), never dropped silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prices import DEFAULT_PRICES, parse_price, positive_int, tokens_per_usd_millions  # after the path is set


def score(result: dict, chips: int, prices: dict[str, float], label: str) -> dict:
    tps_chip = result["total_token_throughput"] / chips
    return {
        "label": label,
        "chips": chips,
        "max_concurrency": result.get("max_concurrency"),
        "num_prompts": result.get("num_prompts"),
        "completed": result.get("completed"),
        "failed": (result.get("num_prompts") or 0) - (result.get("completed") or 0),
        "duration_s": round(result["duration"], 1),
        "mean_input_tokens": round(result["total_input_tokens"] / max(1, result["completed"])),
        "mean_output_tokens": round(result["total_output_tokens"] / max(1, result["completed"])),
        "total_tok_s_per_chip": round(tps_chip, 2),
        "output_tok_s_per_chip": round(result["output_throughput"] / chips, 3),
        "p90_tok_s_per_user": round(1000.0 / result["p90_tpot_ms"], 2),
        "median_tok_s_per_user": round(1000.0 / result["median_tpot_ms"], 2),
        "ttft_p50_s": round(result["median_ttft_ms"] / 1000, 3),
        "ttft_p90_s": round(result["p90_ttft_ms"] / 1000, 3),
        "tpot_p50_ms": round(result["median_tpot_ms"], 2),
        "tpot_p90_ms": round(result["p90_tpot_ms"], 2),
        "price_usd_per_chip_hour": prices,
        "total_tokens_per_usd_millions": tokens_per_usd_millions(tps_chip, prices, digits=4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("result_json", type=Path)
    ap.add_argument("--chips", type=positive_int, default=8)
    ap.add_argument("--price", type=parse_price, action="append", default=None, help="NAME=USD_PER_CHIP_HOUR (repeat)")
    ap.add_argument("--label", default=None)
    a = ap.parse_args(argv)
    prices = dict(a.price) if a.price else dict(DEFAULT_PRICES)
    result = json.loads(a.result_json.read_text())
    print(json.dumps(score(result, a.chips, prices, a.label or a.result_json.stem)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
