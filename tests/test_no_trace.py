"""The published tree carries no trace of where it was built.

The repository must not name the upstream project it derives from, except in NOTICE; nor internal hosts, home paths,
agent names, tool or working-notes directories, or the region, price or billing terms of the instances it ran on.
This test scans every tracked file (binaries included: compiled artifacts carry absolute paths), every tracked path,
and every commit's message and author / committer identity.

The patterns are stored encoded, so this file holds none of the strings it looks for. A positive control plants each
pattern in a file, a path and a commit message of a throwaway repository and requires the scan to report it.
"""

import base64
import json
import os
import re
import subprocess
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = json.loads(base64.b64decode(
    "eyJwYXR0ZXJucyI6IFtbInVwc3RyZWFtIHByb2plY3QgbmFtZSIsICJraWxuIiwgdHJ1ZSwgImF0dHJpYnV0aW9uIl0sIFsi"
    "dXBzdHJlYW0gb3JnIG5hbWUiLCAiZm94bCIsIHRydWUsICJhdHRyaWJ1dGlvbiJdLCBbImludGVybmFsIGhvc3QiLCAiXFxi"
    "dHJuMltzYl1cXGIiLCBmYWxzZSwgImhvc3QiXSwgWyJpbnRlcm5hbCBob3N0IiwgIlxcYmNjWzU3XVxcYiIsIGZhbHNlLCAi"
    "aG9zdCJdLCBbImludGVybmFsIGhvbWUgcGF0aCIsICIvaG9tZS91YnVudHUiLCBmYWxzZSwgInBhdGgiXSwgWyJpbnRlcm5h"
    "bCBob21lIHBhdGgiLCAiL1VzZXJzL2FtYW4iLCBmYWxzZSwgInBhdGgiXSwgWyJhZ2VudCBuYW1lIiwgIlxcYm1hc3Rlcl9h"
    "d3NcXGIiLCBmYWxzZSwgImFnZW50Il0sIFsiYWdlbnQgbmFtZSIsICJcXGJyZXNlYXJjaF9hd3NcXGIiLCBmYWxzZSwgImFn"
    "ZW50Il0sIFsiYWdlbnQgbmFtZSIsICJcXGJkZXZbMTIzXVxcYiIsIGZhbHNlLCAiYWdlbnQiXSwgWyJ0b29sIGRpcmVjdG9y"
    "eSIsICJcXC5jbGF1ZGUvIiwgZmFsc2UsICJwYXRoIl0sIFsid29ya2luZy1ub3RlcyBkaXJlY3RvcnkiLCAiZGV2L3Byb2dy"
    "ZXNzIiwgZmFsc2UsICJwYXRoIl0sIFsicmVnaW9uIiwgImh5ZGVyYWJhZCIsIHRydWUsICJyZWdpb24iXSwgWyJyZWdpb24i"
    "LCAiYXAtc291dGgtMiIsIGZhbHNlLCAicmVnaW9uIl0sIFsicHJpY2UgcGFpZCIsICIoPzwhW1xcZC5dKTEwXFwuMzIoPyFc"
    "XGQpIiwgZmFsc2UsICJwcmljZSJdLCBbImNyZWRpdHMiLCAiXFxiY3JlZGl0c1xcYiIsIHRydWUsICJjcmVkaXRzIl1dLCAi"
    "c2FtcGxlcyI6IFsia2lsbiIsICJmb3hsIiwgInRybjJzIiwgImNjNSIsICIvaG9tZS91YnVudHUiLCAiL1VzZXJzL2FtYW4i"
    "LCAibWFzdGVyX2F3cyIsICJyZXNlYXJjaF9hd3MiLCAiZGV2MSIsICIuY2xhdWRlLyIsICJkZXYvcHJvZ3Jlc3MiLCAiSHlk"
    "ZXJhYmFkIiwgImFwLXNvdXRoLTIiLCAiMTAuMzIiLCAiY3JlZGl0cyJdfQ=="
))
PATTERNS = [(label, re.compile(rx.encode(), re.I if nocase else 0), kind) for label, rx, nocase, kind in _SPEC["patterns"]]
SAMPLES = _SPEC["samples"]
# NOTICE carries the one attribution line the license requires; only the attribution patterns are exempt there.
EXEMPT = {"NOTICE": {"attribution"}}


def _git(root, *args):
    return subprocess.run(["git", "-C", root, *args], capture_output=True, check=True).stdout


def _hits(data, where, exempt=()):
    out = []
    for label, rx, kind in PATTERNS:
        if kind in exempt:
            continue
        for m in rx.finditer(data):
            line = data.count(b"\n", 0, m.start()) + 1
            out.append(f"{where}:{line}: {label}")
            break
    return out


def scan(root):
    """Every hit in root's tracked files, tracked paths and commit history, as 'where: what' strings."""
    hits = []
    paths = [p for p in _git(root, "ls-files", "-z").decode().split("\0") if p]
    for p in paths:
        hits += _hits(p.encode(), f"path {p}")
        full = os.path.join(root, p)
        if os.path.isfile(full) and not os.path.islink(full):
            with open(full, "rb") as f:
                hits += _hits(f.read(), p, EXEMPT.get(p, ()))
    log = _git(root, "log", "--all", "--format=%H%x00%an <%ae>%x00%cn <%ce>%x00%B%x01")
    for rec in log.split(b"\x01"):
        rec = rec.strip(b"\n")
        if rec:
            sha, rest = rec.split(b"\x00", 1)
            hits += _hits(rest, f"commit {sha[:12].decode()}")
    return hits


def _is_git(root):
    try:
        _git(root, "rev-parse", "--git-dir")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


@pytest.mark.skipif(not _is_git(ROOT), reason="needs the git checkout (scans tracked files and history)")
def test_tree_and_history_are_clean():
    hits = scan(ROOT)
    assert not hits, "\n".join(hits)


def test_planted_hits_trip():
    """Positive control: each pattern, planted in a file, a path and a commit message, is reported; NOTICE's exemption
    covers the attribution patterns only."""
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        subprocess.run(["git", "init", "-q", d], check=True)
        with open(os.path.join(d, "clean.txt"), "w") as f:
            f.write("nothing to see\n")
        subprocess.run(["git", "-C", d, "add", "."], check=True)
        subprocess.run(["git", "-C", d, "commit", "-qm", "clean"], check=True, env=env)
        assert scan(d) == []
        for i, s in enumerate(SAMPLES):
            with open(os.path.join(d, f"f{i}.txt"), "w") as f:
                f.write(f"line one\nsee {s} here\n")
        os.makedirs(os.path.join(d, "x"), exist_ok=True)
        with open(os.path.join(d, "x", f"{SAMPLES[0]}_file.txt"), "w") as f:
            f.write("ok\n")
        with open(os.path.join(d, "NOTICE"), "w") as f:
            f.write(f"derived from {SAMPLES[0]} by {SAMPLES[1]}\n")
        subprocess.run(["git", "-C", d, "add", "."], check=True)
        subprocess.run(["git", "-C", d, "commit", "-qm", f"planted {SAMPLES[6]}"], check=True, env=env)
        hits = scan(d)
        for i in range(len(SAMPLES)):
            assert any(h.startswith(f"f{i}.txt:2: ") for h in hits), (i, hits)
        assert any(h.startswith("path x/") for h in hits), hits
        assert any(h.startswith("commit ") for h in hits), hits
        assert not any(h.startswith("NOTICE:") for h in hits), hits
        with open(os.path.join(d, "NOTICE"), "a") as f:
            f.write(f"built on {SAMPLES[2]}\n")
        subprocess.run(["git", "-C", d, "commit", "-qam", "notice"], check=True, env=env)
        assert any(h.startswith("NOTICE:") for h in scan(d))
