"""A shape directory written from the Hugging Face hub (capture.write_shape_dir_hub: metadata, no weights) holds
the same JSON files as one written from the downloaded checkpoint (capture.write_shape_dir): one rule,
capture.shape_json, applied to the repo's listing at the pinned revision. The tokenizer's files must be among
them, since at capture the engine's model path is the shape directory.

The hub test reads the repo's listing, a few small JSON files and the safetensors headers by range reads; it runs
only with TRNRT_TEST_NETWORK=1.
"""

from __future__ import annotations

import json
import os

import pytest

from trnrt.compile import capture

# zai-org/GLM-5.3-Flash at the commit its reference config was read at.
REPO, REV = "zai-org/GLM-5.3-Flash", "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"


def test_shape_json_takes_every_top_level_json_but_the_index():
    names = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "README.md",
        "inference/config.json",
        "special_tokens_map.json",
    ]
    assert capture.shape_json(names) == [
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]


def _online() -> bool:
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRNRT_TEST_NETWORK") != "1":  # opted in only
        return False
    try:
        from huggingface_hub import model_info

        model_info(REPO, revision=REV, timeout=10)
        return True
    except Exception:  # noqa: BLE001 -- no network, or the hub refused
        return False


def test_the_hub_shape_dir_holds_the_checkpoints_json_files(tmp_path):
    if not _online():
        pytest.skip("needs the Hugging Face hub: TRNRT_TEST_NETWORK=1")
    from huggingface_hub import list_repo_files
    from transformers import AutoTokenizer

    dest = str(tmp_path / "shapes")
    capture.write_shape_dir_hub(REPO, dest, REV)
    want = capture.shape_json(list_repo_files(REPO, revision=REV))
    got = sorted(f for f in os.listdir(dest) if f != capture.HEADERS)
    assert got == want, "the hub shape directory's JSON files must be the downloaded checkpoint's"
    assert "tokenizer.json" in got and "config.json" in got
    tok = AutoTokenizer.from_pretrained(dest)
    assert len(tok("GLM-5.3-Flash on Trainium2")["input_ids"]) > 0
    with open(os.path.join(dest, capture.HEADERS)) as f:
        assert capture.moe_prefixes(json.load(f))
