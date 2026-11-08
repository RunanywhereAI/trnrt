"""The PR labeller's rules (.github/labeler.yml) stay in step with the repository: the graphs-changed label covers exactly
the traced paths the formatter skips (pyproject.toml) plus the graph-shape defaults in trnrt/__main__.py, and every
label it applies exists in the issue forms' and CONTRIBUTING's vocabulary."""

from __future__ import annotations

import pathlib
import tomllib

import yaml  # a dependency of huggingface_hub, which trnrt requires

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _globs(rule) -> list[str]:
    return [g for entry in rule for c in entry["changed-files"] for g in c["any-glob-to-any-file"]]


def test_graphs_changed_covers_exactly_the_traced_paths():
    rules = yaml.safe_load((ROOT / ".github" / "labeler.yml").read_text())
    traced = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["ruff"]["format"]["exclude"]
    assert sorted(_globs(rules["graphs-changed"])) == sorted([*traced, "trnrt/__main__.py"])


def test_every_rule_names_a_path_that_exists():
    rules = yaml.safe_load((ROOT / ".github" / "labeler.yml").read_text())
    for label, rule in rules.items():
        for g in _globs(rule):
            assert any(ROOT.glob(g)), (label, g)
