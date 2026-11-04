"""The published tree carries no trace of where it was built.

Each scan covers every tracked file (binaries included: compiled artifacts carry absolute paths), every tracked path,
and every commit's message and author / committer identity.

- **The upstream name, always.** The upstream project this code derives from is credited on NOTICE's attribution line
  and nowhere else. The scan reads the project's name and GitHub organization from that line, so this file spells
  neither.
- **Private patterns, when given.** The maintainers keep the other names that must stay out of the repository
  (internal hosts, paths and the like) in a file outside it, and point ``TRNRT_TRACE_PATTERNS`` at that file. Without
  it, this scan is skipped. The file is JSON: ``patterns``, a list of ``[label, regex, ignore_case, kind]``, and
  ``samples``, strings that between them trip every pattern. NOTICE's attribution line may carry the patterns of kind
  ``attribution``, and no other line may.

Positive controls plant each name in a file, a path and a commit message of a throwaway repository and require the scan
to report it.
"""

import json
import os
import re
import subprocess
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERNS_ENV = "TRNRT_TRACE_PATTERNS"
# NOTICE's attribution line names the upstream and links its repository, as "derived from <Name> (<GitHub URL>)".
ATTRIBUTION = re.compile(rb"derived from (\S+) \(https://github\.com/([\w.-]+)/([\w.-]+)\)")
# Commits whose message predates this scan. History is published unchanged, so these are exempt by full hash, each with
# the reason; a new commit gets no exemption.
EXEMPT_COMMITS = {
    "e47040be9903990aa0f4c6bc18eac686043656cf": "a merge of two working branches, named in its message",
}


def _git(root, *args):
    return subprocess.run(["git", "-C", root, *args], capture_output=True, check=True).stdout


def _is_git(root):
    try:
        _git(root, "rev-parse", "--git-dir")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


needs_git = pytest.mark.skipif(not _is_git(ROOT), reason="needs the git checkout (scans tracked files and history)")
needs_patterns = pytest.mark.skipif(
    not os.environ.get(PATTERNS_ENV),
    reason=f"{PATTERNS_ENV} is unset: the private patterns are kept outside the repository, by its maintainers",
)


def attribution(root):
    """NOTICE's attribution line, and the upstream's names read from it as patterns of kind 'attribution'.

    The names are the project, its repository and the first word of its organization, each matched anywhere, in any
    case."""
    with open(os.path.join(root, "NOTICE"), "rb") as f:
        lines = [ln for ln in f.read().splitlines() if ATTRIBUTION.search(ln)]
    if len(lines) != 1:
        raise ValueError(f"NOTICE must carry exactly one attribution line, found {len(lines)}")
    name, org, repo = ATTRIBUTION.search(lines[0]).groups()
    words = {name.lower(), repo.lower(), re.split(rb"[^a-z0-9]", org.lower())[0]}
    if min(len(w) for w in words) < 3:
        raise ValueError("an upstream name read from NOTICE is under 3 characters: too short to scan for")
    return lines[0], [("upstream name", re.compile(re.escape(w), re.IGNORECASE), "attribution") for w in sorted(words)]


def load_patterns(path):
    """The private patterns file at path, as ([(label, regex, kind)], samples).

    Each pattern must trip on a sample and each sample must trip a pattern, so every pattern has a positive control."""
    with open(path) as f:
        spec = json.load(f)
    patterns = [
        (label, re.compile(rx.encode(), re.IGNORECASE if nocase else 0), kind)
        for label, rx, nocase, kind in spec["patterns"]
    ]
    samples = spec["samples"]
    assert patterns, f"{PATTERNS_ENV} names a file with no patterns"
    untripped = [j for j, (_, rx, _) in enumerate(patterns) if not any(rx.search(s.encode()) for s in samples)]
    assert not untripped, f"patterns no sample trips: {untripped}"
    unmatched = [i for i, s in enumerate(samples) if not any(rx.search(s.encode()) for _, rx, _ in patterns)]
    assert not unmatched, f"samples that trip no pattern: {unmatched}"
    return patterns, samples


def _hits(data, where, patterns, allow=None):
    """The first hit of each pattern in data, as 'where:line: label'. allow = (line, kinds): that exact line may carry
    patterns of those kinds."""
    out = []
    for label, rx, kind in patterns:
        for m in rx.finditer(data):
            start = data.rfind(b"\n", 0, m.start()) + 1
            end = data.find(b"\n", m.start())
            text = data[start : len(data) if end < 0 else end].rstrip(b"\r")
            if allow is not None and kind in allow[1] and text == allow[0]:
                continue
            line = data.count(b"\n", 0, m.start()) + 1
            out.append(f"{where}:{line}: {label}")
            break
    return out


def scan(root, patterns, allow=None):
    """Every hit of patterns in root's tracked files, tracked paths and commit history, as 'where: what' strings.

    allow maps a tracked file to (line, kinds): that line of that file may carry patterns of those kinds."""
    allow = allow or {}
    hits = []
    paths = [p for p in _git(root, "ls-files", "-z").decode().split("\0") if p]
    for p in paths:
        hits += _hits(p.encode(), f"path {p}", patterns)
        full = os.path.join(root, p)
        if os.path.isfile(full) and not os.path.islink(full):
            with open(full, "rb") as f:
                hits += _hits(f.read(), p, patterns, allow.get(p))
    log = _git(root, "log", "--all", "--format=%H%x00%an <%ae>%x00%cn <%ce>%x00%B%x01")
    for raw in log.split(b"\x01"):
        rec = raw.strip(b"\n")
        if rec:
            sha, rest = rec.split(b"\x00", 1)
            if sha.decode() in EXEMPT_COMMITS:
                continue
            hits += _hits(rest, f"commit {sha[:12].decode()}", patterns)
    return hits


def _notice_allow(root):
    """The upstream patterns and the allowance for NOTICE's attribution line, both read from root's NOTICE."""
    line, upstream = attribution(root)
    return upstream, {"NOTICE": (line, {"attribution"})}


@needs_git
def test_upstream_named_only_in_notice():
    upstream, allow = _notice_allow(ROOT)
    hits = scan(ROOT, upstream, allow)
    assert not hits, "\n".join(hits)


@needs_git
@needs_patterns
def test_private_patterns_absent():
    patterns, _ = load_patterns(os.environ[PATTERNS_ENV])
    _, allow = _notice_allow(ROOT)
    hits = scan(ROOT, patterns, allow)
    assert not hits, "\n".join(hits)


# A throwaway repository's NOTICE line, naming a made-up upstream.
NOTICE_LINE = "This product includes software derived from Widget (https://github.com/acme-labs/widget), and modified."


def _write(d, files):
    for name, text in files.items():
        full = os.path.join(d, name)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "a") as f:
            f.write(text)


def _commit(d, message):
    """Commit everything in d with message kept verbatim; returns the full hash."""
    env = dict(
        os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"
    )
    subprocess.run(["git", "-C", d, "add", "."], check=True)
    subprocess.run(
        ["git", "-C", d, "commit", "-q", "--cleanup=verbatim", "-F", "-"], input=message.encode(), check=True, env=env
    )
    return _git(d, "rev-parse", "HEAD").decode().strip()


def _repo(d, notice_line):
    subprocess.run(["git", "init", "-q", d], check=True)
    _write(d, {"NOTICE": f"Example\n\n{notice_line}\n", "clean.txt": "nothing to see\n"})
    _commit(d, "clean\n")


def test_notice_needs_one_attribution_line(tmp_path):
    (tmp_path / "NOTICE").write_text("Example\n")
    with pytest.raises(ValueError, match="exactly one attribution line"):
        attribution(str(tmp_path))
    (tmp_path / "NOTICE").write_text(f"{NOTICE_LINE}\n{NOTICE_LINE}\n")
    with pytest.raises(ValueError, match="exactly one attribution line"):
        attribution(str(tmp_path))


def test_upstream_planted_hits_trip():
    """Positive control for the upstream scan: the names come from NOTICE's attribution line, which may carry them; a
    file, a path, a commit message or another NOTICE line that names them is reported."""
    with tempfile.TemporaryDirectory() as d:
        _repo(d, NOTICE_LINE)
        upstream, allow = _notice_allow(d)
        assert allow["NOTICE"][0] == NOTICE_LINE.encode()
        assert [rx.pattern for _, rx, _ in upstream] == [b"acme", b"widget"]
        assert scan(d, upstream, allow) == []
        _write(d, {"docs.txt": "line one\nbuilt on WIDGET\n", "x/acme-notes.txt": "ok\n"})
        sha = _commit(d, "port from Acme\n")
        hits = scan(d, upstream, allow)
        assert "docs.txt:2: upstream name" in hits
        assert "path x/acme-notes.txt:1: upstream name" in hits
        assert f"commit {sha[:12]}:1: upstream name" in hits
        assert not any(h.startswith("NOTICE:") for h in hits), hits
        _write(d, {"NOTICE": "Widget, again\n"})
        _commit(d, "notice\n")
        assert "NOTICE:4: upstream name" in scan(d, upstream, allow)


@needs_patterns
def test_private_planted_hits_trip():
    """Positive control for the private scan: every pattern is reported from a file and from a commit message, and from
    a path for each sample that makes a file name. NOTICE's attribution line may carry the attribution patterns and no
    others; NOTICE's other lines may carry none. Failures name indices only, never a sample."""
    patterns, samples = load_patterns(os.environ[PATTERNS_ENV])
    kinds = [{k for _, rx, k in patterns if rx.search(s.encode())} for s in samples]
    attr = [s for s, k in zip(samples, kinds) if k == {"attribution"}]
    other = [s for s, k in zip(samples, kinds) if "attribution" not in k]
    assert attr, "no sample trips only the attribution patterns"
    assert other, "no sample trips only non-attribution patterns"

    # The allowance, on bytes: attribution patterns pass on the allowed line only, other kinds never do. Each check is
    # bound to a name first, so that a failure shows no pattern or sample.
    line = (NOTICE_LINE + "".join(f" {s}" for s in attr)).encode()
    passes = not _hits(line, "NOTICE", patterns, (line, {"attribution"}))
    assert passes, "the attribution line's own patterns were reported"
    off_line = bool(_hits(b"x\n" + attr[0].encode(), "NOTICE", patterns, (line, {"attribution"})))
    assert off_line, "an attribution pattern off the attribution line passed"
    tainted = line + b" " + other[0].encode()
    other_kind = bool(_hits(tainted, "NOTICE", patterns, (tainted, {"attribution"})))
    assert other_kind, "a non-attribution pattern on the attribution line passed"

    with tempfile.TemporaryDirectory() as d:
        _repo(d, line.decode())
        _, allow = _notice_allow(d)
        clean = not scan(d, patterns, allow)
        assert clean, "the clean throwaway repository was reported"
        safe = [i for i, s in enumerate(samples) if re.fullmatch(r"[\w.-]+", s)]
        assert safe, "no sample makes a file name"
        files = {f"f{i}.txt": f"line one\nsee {s} here\n" for i, s in enumerate(samples)}
        files.update({f"x/p{i}-{samples[i]}.txt": "ok\n" for i in safe})
        _write(d, files)
        sha = _commit(d, "planted\n\n" + "".join(f"{s}\n" for s in samples))
        hits = set(scan(d, patterns, allow))

        def unreported(where):
            """Indices of the patterns not reported at where(i, label) for any sample i that trips them."""
            return [
                j
                for j, (label, rx, _) in enumerate(patterns)
                if not any(where(i, label) in hits for i, s in enumerate(samples) if rx.search(s.encode()))
            ]

        assert not unreported(lambda i, label: f"f{i}.txt:2: {label}"), "patterns not reported from a file"
        assert not unreported(lambda i, label: f"commit {sha[:12]}:{3 + i}: {label}"), (
            "patterns not reported from a commit"
        )
        missed = [i for i in safe if not any(h.startswith(f"path x/p{i}-") for h in hits)]
        assert not missed, f"samples not reported from a path: {missed}"
        assert not any(h.startswith("NOTICE:") for h in hits), "NOTICE's attribution line was reported"

        _write(d, {"NOTICE": f"see {attr[0]} here\nsee {other[0]} here\n"})
        _commit(d, "notice\n")
        notice = [h for h in scan(d, patterns, allow) if h.startswith("NOTICE:")]
        assert any(h.startswith("NOTICE:4: ") for h in notice), "an attribution pattern off the attribution line passed"
        assert any(h.startswith("NOTICE:5: ") for h in notice), "a non-attribution pattern in NOTICE passed"
