"""The benchmark points (points.csv) and the builds that serve them (builds.csv).

    python3 -I benchmarks/points.py list                 # every point: benchmark, concurrency, build, engine arguments,
                                                         # and when its published number was measured
    python3 -I benchmarks/points.py get POINT FIELD      # one field of a point; FIELD may be a build field
    python3 -I benchmarks/points.py build-args BUILD     # the engine arguments of a build

A point is one measurement: one benchmark at one concurrency on one engine of 8 chips (half a trn2.48xlarge, the other
half idle), served by one build. The build is the engine arguments, the same for `trnrt compile capture` and for `trnrt
serve`, so a point's compile cache is its build's. `published` is the date of the run behind the point's published
number (docs/RESULTS.md), empty for a point the recipe defines that has not been measured on this repository.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _rows(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(newline="") as f:
        return list(csv.DictReader(f))


def _unique(rows: list[dict[str, str]], key: str, name: str) -> dict[str, dict[str, str]]:
    """Rows by `key`; a repeated name is refused: the later row would replace the earlier with no error."""
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        if r[key] in out:
            raise SystemExit(f"{name}: {key} {r[key]!r} appears twice")
        out[r[key]] = r
    return out


def builds() -> dict[str, dict[str, str]]:
    return _unique(_rows("builds.csv"), "build", "builds.csv")


def points() -> dict[str, dict[str, str]]:
    b = builds()
    out = {}
    for name, r in _unique(_rows("points.csv"), "point", "points.csv").items():
        if r["build"] not in b:
            raise SystemExit(f"points.csv: {name}: unknown build {r['build']!r} (builds.csv)")
        out[name] = {**b[r["build"]], **r}
    return out


def main(argv: list[str]) -> int:
    if argv[:1] == ["list"]:
        for name, p in points().items():
            when = f"published {p['published']}" if p["published"] else "not measured on this repository"
            print(f"{name}\t{p['benchmark']}\tC {p['concurrency']}\t{p['build']}\t{p['engine_args']}\t{when}")
        return 0
    if len(argv) == 3 and argv[0] == "get":
        p = points().get(argv[1])
        if p is None or argv[2] not in p:
            print(f"no point {argv[1]!r} or no field {argv[2]!r}: `points.py list` lists the points", file=sys.stderr)
            return 2
        print(p[argv[2]])
        return 0
    if len(argv) == 2 and argv[0] == "build-args":
        b = builds().get(argv[1])
        if b is None:
            print(f"no build {argv[1]!r}: one of {', '.join(builds())}", file=sys.stderr)
            return 2
        print(b["engine_args"])
        return 0
    print(__doc__.strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
