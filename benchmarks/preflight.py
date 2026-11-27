"""Check a server against a benchmark point before its run: one engine, of the point's build, in its acceptance mode.

    python3 -I benchmarks/preflight.py POINT URL [--wait S]

Every score divides by 8 chips: one engine, tensor-parallel over 32 logical NeuronCores (the unit, README.md). A server
of two engines (`trnrt serve`'s default) or of another build would be scored with no error, so the clients run this
first. GET /server_info: the router of `trnrt serve` answers with its first engine's configuration and dp_size, the
number of its engines (serve/router.py); an engine with its own (serve/native.py). It must show dp_size 1, tp_size 32,
the build's --preset and its --spec-synth-al as spec_synth_al (None without it; benchmarks/builds.csv). Then every
engine's acceptance mode, from /v1/stats (accuracy/acceptance_mode.py): golden for a build with --spec-synth-al, real
acceptance otherwise. A server that does not answer is retried for --wait seconds (default 120, for a server still
starting); a server that answers with the wrong configuration is refused at once. Exit status 0: the server matches
the point; 2: it does not, or it did not answer.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "accuracy"))
import acceptance_mode  # after the path is set
from points import points

TP = 32  # one engine of 8 trn2 chips at logical NeuronCore 2: the unit every score divides by


def expected(engine_args: str) -> dict:
    """The /server_info fields a build's engine arguments set: its preset and its synthetic acceptance length."""
    a = engine_args.split()

    def value(flag: str) -> str | None:
        return a[a.index(flag) + 1] if flag in a and a.index(flag) + 1 < len(a) else None

    al = value("--spec-synth-al")
    return {"preset": value("--preset"), "spec_synth_al": float(al) if al is not None else None}


def mismatch(info: object, engine_args: str) -> str | None:
    """None if /server_info shows one engine of the build, else what differs."""
    if not isinstance(info, dict):
        return "its /server_info is not a JSON object"
    want, why = expected(engine_args), []
    if info.get("dp_size") != 1:
        why.append(
            f"it serves {info.get('dp_size')} engines (dp_size) where a point is one: start it with benchmarks/serve.sh"
        )
    if info.get("tp_size") != TP:
        why.append(f"its tp_size is {info.get('tp_size')}, a point's engine is {TP} ranks (8 chips)")
    if info.get("preset") != want["preset"]:
        why.append(f"its preset is {info.get('preset')!r}, the build's is {want['preset']!r}")
    got = info.get("spec_synth_al")
    n = acceptance_mode._number(got)
    same = got is None if want["spec_synth_al"] is None else n is not None and abs(n - want["spec_synth_al"]) < 1e-9
    if not same:
        why.append(f"its spec_synth_al is {got!r}, the build's is {want['spec_synth_al']!r}")
    return "; ".join(why) or None


def fetch(url: str, path: str) -> object:
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=30) as r:  # noqa: S310  an operator-given URL
        return json.loads(r.read())


def main(argv: list[str] | None = None, get=fetch, sleep=time.sleep) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("point", help="a row of benchmarks/points.csv")
    ap.add_argument("url", help="the server, e.g. http://127.0.0.1:8000")
    ap.add_argument("--wait", type=float, default=120.0, help="seconds to retry a server that does not answer")
    a = ap.parse_args(argv)
    p = points().get(a.point)
    if p is None:
        print(f"preflight: no point {a.point!r}: `python3 -I benchmarks/points.py list`", file=sys.stderr)
        return 2
    deadline = time.monotonic() + a.wait
    while True:
        try:
            info = get(a.url, "/server_info")
            break
        # urllib's errors are OSErrors, a response cut off mid-body (IncompleteRead, BadStatusLine) an HTTPException, a
        # body that is not JSON a ValueError
        except (OSError, http.client.HTTPException, ValueError) as e:
            if time.monotonic() >= deadline:
                print(f"preflight: {a.url}/server_info did not answer: {e}", file=sys.stderr)
                return 2
            sleep(2.0)
    why = mismatch(info, p["engine_args"])
    if why:
        print(f"refused: {a.url} is not one engine of build {p['build']} ({p['engine_args']}): {why}", file=sys.stderr)
        return 2
    expect = "golden" if expected(p["engine_args"])["spec_synth_al"] is not None else "real"
    wait = str(max(0.0, deadline - time.monotonic()))
    rc = acceptance_mode.main(
        ["--expect", expect, "--url", a.url, "--wait", wait], get=lambda u: get(u, "/v1/stats"), sleep=sleep
    )
    if rc == 0:
        print(f"preflight: {a.url} is one engine of build {p['build']} ({p['engine_args']}), as {a.point} needs")
    return rc


if __name__ == "__main__":
    sys.exit(main())
