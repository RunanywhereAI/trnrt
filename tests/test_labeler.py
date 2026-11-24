"""The PR labeller's rules (.github/labeler.yml) stay in step with the repository: the graphs-changed label covers exactly
the traced paths the formatter skips (pyproject.toml) plus the graph-shape defaults in trnrt/__main__.py, every rule
names files that exist, and its labels are the area labels CodeRabbit is told about (.coderabbit.yaml), so the two
vocabularies can't drift."""

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


def test_the_labels_are_the_area_labels_coderabbit_knows():
    rules = yaml.safe_load((ROOT / ".github" / "labeler.yml").read_text())
    cr = yaml.safe_load((ROOT / ".coderabbit.yaml").read_text())
    assert set(rules) == {x["label"] for x in cr["reviews"]["labeling_instructions"]}
