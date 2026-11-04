"""tools/check_docs.py: the repository's Markdown links resolve, and the checker itself catches what it must."""

import importlib.util
import os
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("check_docs", os.path.join(ROOT, "tools", "check_docs.py"))
cd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cd)


def _is_git(root):
    return subprocess.run(["git", "-C", root, "rev-parse", "--git-dir"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _is_git(ROOT), reason="needs the git checkout (lists the tracked Markdown); the image has none")
def test_the_repository_docs_have_no_broken_links():
    assert cd.main([]) == 0


def test_github_anchor_form():
    assert cd.anchor("Batch 4: `--max-running 4`") == "batch-4---max-running-4"
    assert cd.anchor("Adding a model family") == "adding-a-model-family"
    assert cd.anchor("What you need") == "what-you-need"


def test_broken_links_and_anchors_are_reported(tmp_path):
    (tmp_path / "other.md").write_text("# Title\n\n## A section\n")
    (tmp_path / "img.svg").write_text("<svg/>")
    doc = tmp_path / "doc.md"
    doc.write_text(
        "# Doc\n"
        "[ok](other.md#a-section) [ok](#doc) ![ok](img.svg) [web](https://example.com)\n"
        '<img src="img.svg"> <source srcset="img.svg">\n'
        "[missing](nope.md) [bad anchor](other.md#nope)\n"
        '<img src="missing.svg">\n'
        "```\n[inside a fence](ignored.md)\n```\n"
    )
    found = cd.broken(str(doc), root=str(tmp_path))
    assert found == [
        "doc.md:4: nope.md (no such file)",
        "doc.md:4: other.md#nope (no heading #nope)",
        "doc.md:5: missing.svg (no such file)",
    ]


def test_claude_md_must_be_a_symlink_to_agents_md(tmp_path):
    assert cd.agents_pair(str(tmp_path)) == ["AGENTS.md is missing"]
    (tmp_path / "AGENTS.md").write_text("# Agents\n")
    (tmp_path / "CLAUDE.md").write_text("# a copy\n")
    assert cd.agents_pair(str(tmp_path)) != []
    (tmp_path / "CLAUDE.md").unlink()
    os.symlink("AGENTS.md", tmp_path / "CLAUDE.md")
    assert cd.agents_pair(str(tmp_path)) == []
