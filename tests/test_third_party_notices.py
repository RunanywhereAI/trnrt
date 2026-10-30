"""THIRD_PARTY_NOTICES.md lists exactly the third-party source in the tree.

Every tracked file that carries a third-party marker (another holder's copyright line, a license header, or an
"adapted from" / "portions" note) must be covered by a path listed in the notices' table of included source, and every
listed path must exist. A positive control checks that a planted header is caught and that listing it clears it.
"""

import os
import re
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTICES = "THIRD_PARTY_NOTICES.md"
OWN = re.compile(rb"RunanywhereAI", re.I)
MARKERS = re.compile(
    rb"(copyright\s+(\(c\)\s*)?(\d{4}[-\d, ]*)?\s*(?!runanywhere)[a-z][^\n]{2,}"
    rb"|spdx-license-identifier"
    rb"|licensed under the (apache|mit|bsd)"
    rb"|portions\s+(adapted|copied|derived|taken)\s+from"
    rb"|\badapted from\s+[a-z0-9_.-]+/[a-z0-9_.-]+)",
    re.I,
)
SKIP = {"LICENSE", "NOTICE", NOTICES, "tests/test_third_party_notices.py"}


def listed_paths(text):
    """Backticked paths in the first column of the 'Source included' table."""
    sec = text.split("## Source included in this repository", 1)[-1].split("\n## ", 1)[0]
    out = []
    for line in sec.splitlines():
        if line.startswith("|") and not line.startswith("|---") and not line.startswith("| Path"):
            first = line.split("|")[1]
            out += [p.rstrip("/") for p in re.findall(r"`([^`]+)`", first) if "/" in p or "." in p]
    return out


def marked_files(root, paths):
    hits = []
    for p in paths:
        if p in SKIP or p.endswith((".png", ".svg", ".gz", ".npy", ".neff")):
            continue
        full = os.path.join(root, p)
        if not os.path.isfile(full):
            continue
        with open(full, "rb") as f:
            head = f.read(64 << 10)
        m = MARKERS.search(head)
        if m and not OWN.search(m.group(0)):
            hits.append(p)
    return hits


def covered(path, listed):
    return any(path == l or path.startswith(l + "/") for l in listed)


def check(root, paths):
    text = open(os.path.join(root, NOTICES)).read() if os.path.exists(os.path.join(root, NOTICES)) else ""
    listed = listed_paths(text)
    problems = [f"{NOTICES} lists {l}, which does not exist" for l in listed if not os.path.exists(os.path.join(root, l))]
    problems += [f"{p} carries a third-party marker but {NOTICES} does not list it" for p in marked_files(root, paths)
                 if not covered(p, listed)]
    return problems


def _tracked(root):
    return [p for p in subprocess.run(["git", "-C", root, "ls-files", "-z"], capture_output=True, check=True).stdout
            .decode().split("\0") if p]


@pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, ".git")) and not os.path.isfile(os.path.join(ROOT, ".git")),
                    reason="needs the git checkout")
def test_notices_match_the_tree():
    problems = check(ROOT, _tracked(ROOT))
    assert not problems, "\n".join(problems)


def test_planted_header_is_caught(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "own.py").write_text("# Copyright 2026 RunanywhereAI\n")
    (tmp_path / "pkg" / "theirs.py").write_text("# SPDX-License-Identifier: Apache-2.0\n# Copyright contributors to Foo\n")
    head = ("# Third-party notices\n\n## Source included in this repository\n\n| Path | Origin | License | Changes |\n"
            "|---|---|---|---|\n")
    (tmp_path / NOTICES).write_text(head)
    paths = ["pkg/own.py", "pkg/theirs.py", NOTICES]
    assert check(str(tmp_path), paths) == [f"pkg/theirs.py carries a third-party marker but {NOTICES} does not list it"]
    (tmp_path / NOTICES).write_text(head + "| `pkg/theirs.py` | Foo | Apache-2.0 | none |\n| `pkg/gone.py` | x | y | z |\n")
    assert check(str(tmp_path), paths) == [f"{NOTICES} lists pkg/gone.py, which does not exist"]
