"""Check a server's acceptance mode before a run: refuse a quality run on the benchmark mode, and a golden-length
throughput run on real acceptance.

    python3 -I benchmarks/accuracy/acceptance_mode.py --expect real|golden (--url URL | --health-json JSON)

The benchmark mode (`--spec-synth-al`, docs/AGENTX.md#benchmark-mode) emits accepted drafts unverified, so its text is
not the model's: an accuracy run on it would score the draft head, and must not run. The server says which mode it is
in on GET /health (trnrt/serve/benchmark_mode.py): a `benchmark_mode` block with the synthetic acceptance length, on
the engine at once and on the router from its first poll of the engines (2 s). Exit status 0: the mode is the expected
one; 2: it is not, or /health does not answer.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

GOLDEN = 2.518506  # InferenceX's golden acceptance length for GLM-5.3-Flash with 2 draft tokens


def lengths(health: dict) -> list[float]:
    block = health.get("benchmark_mode")
    if not block:
        return []
    v = block.get("synthetic_acceptance_length")
    return [float(x) for x in (v if isinstance(v, list) else [v])]


def check(health: dict, expect: str) -> str | None:
    """None if the server runs the expected mode, else why not."""
    found = lengths(health)
    if expect == "real" and found:
        return (
            f"the server runs the benchmark mode (synthetic acceptance length {found}): its drafts are not verified "
            "and its text is not the model's. Serve without --spec-synth-al / TRNRT_SPEC_SYNTH_AL for any quality run"
        )
    if expect == "golden" and found != [GOLDEN]:
        return (
            f"the point needs the benchmark mode at the golden length {GOLDEN}, the server reports "
            f"{found or 'real acceptance'}: serve its build (benchmarks/builds.csv, latency-golden)"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("--expect", choices=["real", "golden"], required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="the server (router or engine), e.g. http://127.0.0.1:8000")
    src.add_argument("--health-json", help="a /health body already fetched")
    a = ap.parse_args(argv)
    try:
        if a.url:
            with urllib.request.urlopen(a.url.rstrip("/") + "/health", timeout=30) as r:  # noqa: S310  an operator-given URL
                health = json.loads(r.read())
        else:
            health = json.loads(a.health_json)
    except (OSError, ValueError) as e:
        print(f"acceptance mode: /health unreadable: {e}", file=sys.stderr)
        return 2
    why = check(health, a.expect)
    if why:
        print(f"refused: {why}", file=sys.stderr)
        return 2
    print(f"acceptance mode: {a.expect} ({lengths(health) or 'no benchmark mode'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
