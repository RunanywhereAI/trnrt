"""Check the repository's Markdown: every relative link and image resolves, and CLAUDE.md is a symlink to AGENTS.md.

A relative link is a Markdown link or image (`[text](path)`, `![alt](path)`) or an HTML `src`/`srcset`/`href` whose
target is not a URL. A link with a fragment (`path#anchor`, or `#anchor` alone) must name a heading of the file it
points at, in GitHub's anchor form. Fenced code blocks are skipped. Exits 1 and names every broken link.

    python tools/check_docs.py            # every tracked *.md file
    python tools/check_docs.py README.md  # just these
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MD_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HTML_ATTR = re.compile(r"""\b(?:src|srcset|href)=["']([^"']+)["']""")
_FENCE = re.compile(r"^\s*(```|~~~)")
_EXTERNAL = ("http://", "https://", "mailto:", "data:")


def anchor(heading: str) -> str:
    """GitHub's anchor for a heading: lowercase, punctuation dropped (hyphens and underscores kept), spaces to hyphens."""
    text = re.sub(r"`([^`]*)`", r"\1", heading.strip()).lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path: str) -> set[str]:
    out, seen, fenced = set(), {}, False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if _FENCE.match(line):
                fenced = not fenced
                continue
            m = None if fenced else re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
            if m:
                a = anchor(m.group(1))
                n = seen.get(a, 0)
                out.add(a if n == 0 else f"{a}-{n}")
                seen[a] = n + 1
    return out


def links(path: str):
    """(line number, target) of every link in the Markdown file at path, outside fenced code."""
    fenced = False
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if _FENCE.match(line):
                fenced = not fenced
                continue
            if fenced:
                continue
            for m in _MD_LINK.finditer(line):
                yield i, m.group(1)
            for m in _HTML_ATTR.finditer(line):
                for target in m.group(1).split(","):
                    yield i, target.strip().split(" ")[0]


def broken(path: str, root: str = ROOT) -> list[str]:
    """`file:line: target` for every relative link in path that does not resolve."""
    out, rel = [], os.path.relpath(path, root)
    for line, target in links(path):
        if not target or target.startswith(_EXTERNAL):
            continue
        file_part, _, frag = target.partition("#")
        dest = os.path.normpath(os.path.join(os.path.dirname(path), file_part)) if file_part else path
        if not os.path.exists(dest):
            out.append(f"{rel}:{line}: {target} (no such file)")
        elif frag and dest.endswith(".md") and frag not in anchors(dest):
            out.append(f"{rel}:{line}: {target} (no heading #{frag})")
    return out


def agents_pair(root: str = ROOT) -> list[str]:
    """AGENTS.md is the canonical agent guide; CLAUDE.md must be a symlink to it, never a second copy."""
    claude = os.path.join(root, "CLAUDE.md")
    if not os.path.isfile(os.path.join(root, "AGENTS.md")):
        return ["AGENTS.md is missing"]
    if not os.path.islink(claude) or os.readlink(claude) != "AGENTS.md":
        return ["CLAUDE.md must be a symlink to AGENTS.md (ln -sf AGENTS.md CLAUDE.md)"]
    return []


def tracked_markdown(root: str = ROOT) -> list[str]:
    files = subprocess.run(["git", "-C", root, "ls-files", "-z", "*.md"], capture_output=True, check=True).stdout
    return [
        os.path.join(root, p) for p in files.decode().split("\0") if p and not os.path.islink(os.path.join(root, p))
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="check_docs", description=__doc__.split("\n")[0], allow_abbrev=False)
    ap.add_argument("files", nargs="*", help="Markdown files to check (default: every tracked *.md)")
    args = ap.parse_args(argv)
    files = [os.path.abspath(f) for f in args.files if f.endswith(".md")] or tracked_markdown()
    problems = agents_pair() + [p for f in files if not os.path.islink(f) for p in broken(f)]
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
