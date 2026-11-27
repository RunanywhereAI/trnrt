"""Check a server's acceptance mode before a run: refuse a quality run on the benchmark mode, and a golden-length
throughput run on real acceptance.

    python3 -I benchmarks/accuracy/acceptance_mode.py --expect real|golden (--url URL [--wait S] | --body-json JSON)

The benchmark mode (`--spec-synth-al`, docs/AGENTX.md#benchmark-mode) emits accepted drafts unverified, so its text is
not the model's: an accuracy run on it would score the draft head, and must not run. Each engine reports its mode in
GET /v1/stats (gauges.spec.synth_al, trnrt/metrics.py: the configured length, None for real acceptance) from its start.
The router's /v1/stats asks every engine at the moment of the request, so it shows each engine's mode, a router over
engines in two modes included; that is the body this check reads. Its /health is no substitute: the router adds a
`benchmark_mode` block there only after its poll has read the engines' /v1/stats, and the block does not say which
engine it came from. An engine's own /health carries its block from the start, and is accepted as a body too.
Expect `real`: every engine runs real acceptance. Expect `golden`: every engine runs the benchmark mode at the golden
length. A body the mode cannot be read from (an engine that did not answer, a malformed body) is retried for --wait
seconds (default 60), then refused; a wrong mode is refused at once. Exit status 0: the mode is the expected one; 2:
it is not, or it could not be read.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.request

GOLDEN = 2.518506  # InferenceX's golden acceptance length for GLM-5.3-Flash with 2 draft tokens


def _number(v: object) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _from_stats(stats: dict) -> float | str | None:
    """One engine's mode from its /v1/stats: the synthetic length, None for real acceptance, or why it cannot be read."""
    gauges = stats.get("gauges")
    if not isinstance(gauges, dict):
        return "its /v1/stats has no gauges"
    spec = gauges.get("spec")
    if spec is None:  # no speculation: real acceptance (the benchmark mode needs MTP, config.py)
        return None
    if not isinstance(spec, dict) or "synth_al" not in spec:
        return "its /v1/stats has a malformed spec block"
    v = spec["synth_al"]
    if v is None:
        return None
    n = _number(v)
    return n if n is not None else f"its synth_al {v!r} is not a number"


def modes(body: object) -> list[float | None] | str:
    """Each engine's synthetic acceptance length (None: real acceptance), or a string that says why the body cannot show
    them (module docstring: the bodies this reads)."""
    if not isinstance(body, dict):
        return "the body is not a JSON object"
    if "engines" in body:  # the router's /v1/stats
        engines = body["engines"]
        if not isinstance(engines, list) or not engines:
            return "the router's /v1/stats lists no engine"
        out: list[float | None] = []
        for i, e in enumerate(engines):
            stats = e.get("stats") if isinstance(e, dict) else None
            if not isinstance(stats, dict):
                return f"engine {i} did not answer the router's /v1/stats"
            m = _from_stats(stats)
            if isinstance(m, str):
                return f"engine {i}: {m}"
            out.append(m)
        return out
    if "gauges" in body:  # one engine's /v1/stats
        m = _from_stats(body)
        return m if isinstance(m, str) else [m]
    if "workers" in body:  # the router's /health
        return "a router's /health does not show each engine's mode; read its /v1/stats"
    block = body.get("benchmark_mode")  # one engine's /health
    if block is None:
        return [None]
    v = block.get("synthetic_acceptance_length") if isinstance(block, dict) else None
    found = [_number(x) for x in (v if isinstance(v, list) else [v])]
    if not found or any(x is None for x in found):
        return f"a malformed benchmark_mode block: {block!r}"
    return found


def check(body: object, expect: str) -> str | None:
    """None if every engine runs the expected mode, else why not (a body the mode cannot be read from included)."""
    found = modes(body)
    if isinstance(found, str):
        return f"the acceptance mode cannot be read: {found}"
    synthetic = [x for x in found if x is not None]
    if expect == "real" and synthetic:
        return (
            f"the server runs the benchmark mode (synthetic acceptance length {synthetic}, of engines {found}): its "
            "drafts are not verified and its text is not the model's. Serve without --spec-synth-al / "
            "TRNRT_SPEC_SYNTH_AL for any quality run"
        )
    if expect == "golden" and not all(x is not None and abs(x - GOLDEN) < 1e-9 for x in found):
        shown = ["real acceptance" if x is None else x for x in found]
        return (
            f"the point needs the benchmark mode at the golden length {GOLDEN} on every engine, the server reports "
            f"{shown}: serve its build (benchmarks/builds.csv, latency-golden)"
        )
    return None


def fetch(url: str) -> object:
    with urllib.request.urlopen(url.rstrip("/") + "/v1/stats", timeout=30) as r:  # noqa: S310  an operator-given URL
        return json.loads(r.read())


def main(argv: list[str] | None = None, get=fetch, sleep=time.sleep) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("--expect", choices=["real", "golden"], required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="the server (router or engine), e.g. http://127.0.0.1:8000")
    src.add_argument("--body-json", help="a /v1/stats (or an engine's /health) body already fetched")
    ap.add_argument("--wait", type=float, default=60.0, help="seconds to retry a mode that cannot be read (with --url)")
    a = ap.parse_args(argv)
    deadline = time.monotonic() + (a.wait if a.url else 0.0)
    while True:
        try:
            body = get(a.url) if a.url else json.loads(a.body_json)
            found = modes(body)
            unreadable = found if isinstance(found, str) else None
        # urllib's errors are OSErrors, a response cut off mid-body an HTTPException, a body that is not JSON a ValueError
        except (OSError, http.client.HTTPException, ValueError) as e:
            body, found, unreadable = None, None, f"/v1/stats unreadable: {e}"
        if unreadable is None or time.monotonic() >= deadline:
            break
        sleep(2.0)
    if unreadable is not None:
        print(f"refused: the acceptance mode cannot be read: {unreadable}", file=sys.stderr)
        return 2
    why = check(body, a.expect)
    if why:
        print(f"refused: {why}", file=sys.stderr)
        return 2
    print(f"acceptance mode: {a.expect} (engines: {found})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
