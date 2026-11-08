"""tools/release_plan.py: what a push to main releases. Each case builds a small git repository with the package version
and tags it has, and asks for the plan at its latest commit."""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("release_plan", ROOT / "tools" / "release_plan.py")
rp = importlib.util.module_from_spec(_spec)
sys.modules["release_plan"] = rp  # a dataclass resolves its annotations through its module
_spec.loader.exec_module(rp)


def git(repo, *args):
    env = {
        **os.environ,  # PATH: git may live outside the default search path
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env)
    return out.stdout.strip()


def commit(repo, version):
    (repo / "trnrt").mkdir(exist_ok=True)
    (repo / "trnrt" / "__init__.py").write_text(f'"""trnrt."""\n\n__version__ = "{version}"\n')
    git(repo, "add", "-A")
    git(repo, "-c", "commit.gpgsign=false", "commit", "-q", "-m", f"release {version}")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    return tmp_path


def test_a_new_version_with_no_tags_is_released(repo):
    sha = commit(repo, "0.1.0")
    p = rp.plan(repo, sha)
    assert (p.action, p.version, p.tag) == ("release", "0.1.0", "v0.1.0")


def test_a_newer_version_than_the_latest_tag_is_released(repo):
    first = commit(repo, "0.9.0")
    git(repo, "tag", "v0.9.0", first)
    git(repo, "tag", "v0.10.0", first)  # compared as numbers, not text: 0.10.0 is newer than 0.9.0
    sha = commit(repo, "0.11.0")
    assert rp.plan(repo, sha).action == "release"


def test_the_tag_at_this_commit_means_a_failed_release_run_started_again(repo):
    sha = commit(repo, "0.1.0")
    git(repo, "tag", "v0.1.0", sha)
    p = rp.plan(repo, sha)
    assert (p.action, p.tag) == ("rerun", "v0.1.0")


def test_a_merge_that_did_not_bump_the_version_releases_nothing(repo):
    first = commit(repo, "0.1.0")
    git(repo, "tag", "v0.1.0", first)
    git(repo, "-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty", "-m", "docs")  # merged into main later
    p = rp.plan(repo, git(repo, "rev-parse", "HEAD"))
    assert p.action == "skip" and "v0.1.0 already exists" in p.why


@pytest.mark.parametrize("version", ["0.2.0rc1", "0.2.0.dev3", "0.2", "1.0.0+local"])
def test_a_version_that_is_not_a_final_release_is_never_released(repo, version):
    sha = commit(repo, version)
    p = rp.plan(repo, sha)
    assert p.action == "skip" and "not a final" in p.why


def test_a_version_that_goes_backwards_is_an_error(repo):
    first = commit(repo, "0.2.0")
    git(repo, "tag", "v0.2.0", first)
    sha = commit(repo, "0.1.5")
    with pytest.raises(ValueError, match=re.escape("not newer than the latest release, 0.2.0")):
        rp.plan(repo, sha)
    assert rp.main(["--repo", str(repo), "--sha", sha]) == 1


def test_tags_that_are_not_releases_are_ignored(repo):
    first = commit(repo, "0.1.0")
    git(repo, "tag", "v9.9.9-rc1", first)
    git(repo, "tag", "vnext", first)
    sha = commit(repo, "0.1.1")
    assert rp.plan(repo, sha).action == "release"


def test_the_plan_is_written_for_the_workflow(repo, tmp_path):
    sha = commit(repo, "1.2.3")
    out = tmp_path / "out.txt"
    assert rp.main(["--repo", str(repo), "--sha", sha, "--github-output", str(out)]) == 0
    assert out.read_text() == "action=release\nversion=1.2.3\ntag=v1.2.3\n"


def test_this_repository_reads_its_own_version():
    import trnrt

    assert rp.package_version(ROOT) == trnrt.__version__
