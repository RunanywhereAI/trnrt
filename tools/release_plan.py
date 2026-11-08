"""Decide what a push to main releases (`.github/workflows/release.yml`): nothing, a new release, or a rerun of one.

A promotion merged into main releases the version in `trnrt/__init__.py`, once. The tag is created by the release, on
the merge commit, so this decides from the tags that already exist:

- `release`: the version is a final `X.Y.Z`, has no tag, and is newer than every existing tag.
- `rerun`: the version's tag already points at this commit (a failed release run started again). The publishing steps
  are idempotent, so they run again.
- `skip`: the version's tag points at another commit (a merge into main that did not bump the version), or the version
  is not a final `X.Y.Z` (a pre-release is never released from main).

A version that has no tag but is not newer than the latest tag is an error: a release must not go backwards. Exit 1.

    python tools/release_plan.py --sha "$GITHUB_SHA" --github-output "$GITHUB_OUTPUT"
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_FINAL = re.compile(r"\d+\.\d+\.\d+")
_TAG = re.compile(r"v(\d+\.\d+\.\d+)")


@dataclass(frozen=True)
class Plan:
    action: str  # "release", "rerun" or "skip"
    version: str
    tag: str
    why: str


def package_version(repo: Path) -> str:
    """`__version__` from trnrt/__init__.py, read without importing the package."""
    tree = ast.parse((repo / "trnrt" / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError("trnrt/__init__.py has no string __version__")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split("."))


def plan(repo: Path, sha: str) -> Plan:
    version = package_version(repo)
    tag = f"v{version}"
    if not _FINAL.fullmatch(version):
        return Plan("skip", version, tag, f"{version} is not a final X.Y.Z version, so main does not release it")
    existing = {m.group(1): t for t in _git(repo, "tag", "--list", "v*").split() if (m := _TAG.fullmatch(t))}
    if version in existing:
        at = _git(repo, "rev-parse", f"{tag}^{{commit}}")
        if at == _git(repo, "rev-parse", f"{sha}^{{commit}}"):
            return Plan("rerun", version, tag, f"{tag} already points at this commit: publishing again")
        return Plan("skip", version, tag, f"{tag} already exists at {at[:7]}: bump the version to release")
    if existing and _key(version) <= max(_key(v) for v in existing):
        latest = max(existing, key=_key)
        raise ValueError(f"{version} is not newer than the latest release, {latest}")
    return Plan("release", version, tag, f"{tag} is new: releasing it")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--sha", required=True, help="the commit being released (GITHUB_SHA)")
    ap.add_argument("--repo", type=Path, default=ROOT, help="the checkout (default: this repository)")
    ap.add_argument("--github-output", help="append action, version and tag here (GITHUB_OUTPUT)")
    a = ap.parse_args(argv)
    try:
        p = plan(a.repo, a.sha)
    except (ValueError, subprocess.CalledProcessError) as e:
        print(f"::error::{e}", file=sys.stderr)
        return 1
    print(f"{p.action}: {p.why}")
    if a.github_output:
        with open(a.github_output, "a") as f:
            f.write(f"action={p.action}\nversion={p.version}\ntag={p.tag}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
