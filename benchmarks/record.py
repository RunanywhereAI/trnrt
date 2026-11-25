"""Write one run's result file: the score, the point and build that produced it, the versions, the client, scrubbed.

    python3 -I benchmarks/record.py --point POINT --score score.json [--versions versions.json]
                                    [--client-cpus N] [--commit SHA] [--note TEXT] --out result.json

The result file is what a published number cites. It records the point (benchmarks/points.csv) with its build and engine
arguments, the score that a scorer printed, the versions (benchmarks/versions.sh), the load client's core count, the
repository commit and the date. Every string is scrubbed before it is written: URLs lose their host, absolute paths
keep only their last component, and IPv4 addresses are replaced, so a result names no machine. The versions are the
exception to the address rule: versions.sh writes package versions only, and the Neuron runtime's are four-part
numbers (2.34.10.0) that read as addresses. With
TRNRT_TRACE_PATTERNS set (tests/test_no_trace.py's private patterns), a result that still matches one is refused.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from points import points  # after the path is set

_URL = re.compile(r"\b(https?://)[^/\s\"']+([^\s\"']*)")  # the host goes, the URL's path stays
# not inside a longer dotted number: version strings such as 2.11.0.1.0.1284 keep their digits
_IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w)(?!\.\d)")
_PATH = re.compile(r"(?<![\w.:/])/(?:[\w.@+-]+/)+([\w.@+-]+)")


def scrub(x, ips: bool = True):
    """The same structure with every string scrubbed (module docstring); `ips=False` keeps dotted numbers."""
    if isinstance(x, dict):
        return {scrub(k, ips): scrub(v, ips) for k, v in x.items()}
    if isinstance(x, list):
        return [scrub(v, ips) for v in x]
    if isinstance(x, str):
        out, i = [], 0
        for m in _URL.finditer(x):
            out += [_plain(x[i : m.start()], ips), f"{m.group(1)}<server>{m.group(2)}"]
            i = m.end()
        return "".join([*out, _plain(x[i:], ips)])
    return x


def _plain(s: str, ips: bool) -> str:
    """Text outside URLs: addresses and absolute paths."""
    return _PATH.sub(r"<path>/\1", _IPV4.sub("<ip>", s) if ips else s)


def private_hits(text: str) -> list[str]:
    """The private patterns (TRNRT_TRACE_PATTERNS) the text still matches: their labels."""
    f = os.environ.get("TRNRT_TRACE_PATTERNS")
    if not f:
        return []
    pats = json.loads(Path(f).read_text())["patterns"]
    return sorted({label for label, rx, *_ in pats if re.search(rx, text, re.IGNORECASE)})


def _commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return r.stdout.strip()


def build_record(
    point: str, score: dict, versions: dict | None, client_cpus: int | None, commit: str | None, note: str | None
) -> dict:
    p = points().get(point)
    if p is None:
        raise SystemExit(f"no point {point!r}: `python3 -I benchmarks/points.py list`")
    rec = scrub(
        {
            "point": point,
            "benchmark": p["benchmark"],
            "concurrency": int(p["concurrency"]),
            "build": p["build"],
            "engine_args": p["engine_args"],
            "unit": "one engine of 8 trn2 chips (half a trn2.48xlarge), the other half idle",
            "instance": "trn2.48xlarge",
            "precision": "FP8 weights as released, FP8 KV cache",
            "synthetic_acceptance": p["build"] == "latency-golden",
            "score": score,
            "versions": None,
            "client_logical_cpus": client_cpus,
            "commit": commit,
            "date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
            "note": note,
        }
    )
    rec["versions"] = scrub(versions, ips=False)
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("--point", required=True)
    ap.add_argument("--score", type=Path, required=True)
    ap.add_argument("--versions", type=Path, default=None)
    ap.add_argument("--client-cpus", type=int, default=None)
    ap.add_argument("--commit", default=None, help="default: this checkout's HEAD")
    ap.add_argument("--note", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rec = build_record(
        a.point,
        json.loads(a.score.read_text()),
        json.loads(a.versions.read_text()) if a.versions else None,
        a.client_cpus,
        a.commit or _commit(),
        a.note,
    )
    text = json.dumps(rec, indent=1) + "\n"
    hits = private_hits(text)
    if hits:
        print(f"refused: the result still matches private patterns {hits}", file=sys.stderr)
        return 2
    a.out.write_text(text)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
