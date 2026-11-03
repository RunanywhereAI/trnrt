"""--model as a Hugging Face repo id (trnrt/models/loader.py resolve_model_path): the download takes the chat template
(GLM-5.3-Flash ships it as chat_template.jinja alone) with the configs, weights and tokenizer files, at the --revision
given (the default branch without one); a local directory is used as it is, and refuses a revision. The engine
resolves its configuration's model_revision, which the command line sets from --revision (host only: nothing else in
the configuration moves). Nothing is downloaded: snapshot_download is a stand-in."""

from __future__ import annotations

import fnmatch
import sys
import types

import pytest

from trnrt import __main__ as cli
from trnrt.models import loader

GLM_FILES = ["config.json", "generation_config.json", "model.safetensors.index.json", "model-00001-of-00093.safetensors",
             "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md", "assets/figure.png"]


@pytest.fixture
def hub(monkeypatch):
    calls = []

    def snapshot_download(repo_id, revision=None, allow_patterns=None):
        calls.append({"repo": repo_id, "revision": revision, "patterns": list(allow_patterns)})
        return f"/cache/{repo_id}/{revision or 'main'}"

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    return calls


def test_a_repo_id_downloads_the_chat_template_at_the_revision(hub):
    assert loader.resolve_model_path("zai-org/GLM-5.3-Flash", "eb9eb208") == "/cache/zai-org/GLM-5.3-Flash/eb9eb208"
    assert loader.resolve_model_path("zai-org/GLM-5.3-Flash") == "/cache/zai-org/GLM-5.3-Flash/main"
    assert [c["revision"] for c in hub] == ["eb9eb208", None]
    taken = [f for f in GLM_FILES if any(fnmatch.fnmatch(f, p) for p in hub[0]["patterns"])]
    assert taken == ["config.json", "generation_config.json", "model.safetensors.index.json",
                     "model-00001-of-00093.safetensors", "tokenizer.json", "tokenizer_config.json",
                     "chat_template.jinja"]


def test_a_local_directory_is_used_as_it_is_and_refuses_a_revision(hub, tmp_path):
    assert loader.resolve_model_path(str(tmp_path)) == str(tmp_path)
    with pytest.raises(ValueError, match="local checkpoint directory"):
        loader.resolve_model_path(str(tmp_path), "eb9eb208")
    assert hub == []


def test_the_command_line_sets_only_the_revision():
    import dataclasses

    plain, pinned = cli.engine_config(["--model", "m"]), cli.engine_config(["--model", "m", "--revision", "eb9eb208"])
    assert plain.model_revision is None and pinned.model_revision == "eb9eb208"
    assert {f.name for f in dataclasses.fields(plain) if getattr(plain, f.name) != getattr(pinned, f.name)} == \
        {"model_revision"}


def test_the_engine_resolves_its_configured_revision(monkeypatch):
    from trnrt.config import EngineConfig
    from trnrt.engine import engine as E

    seen = []

    class Resolved(Exception):
        pass

    def resolve(model, revision=None):
        seen.append((model, revision))
        raise Resolved  # nothing past the resolution: no model to build here

    monkeypatch.setattr(E, "resolve_model_path", resolve)
    with pytest.raises(Resolved):
        E.LLMEngine(EngineConfig(model_path="zai-org/GLM-5.3-Flash", model_revision="eb9eb208"))
    assert seen == [("zai-org/GLM-5.3-Flash", "eb9eb208")]
