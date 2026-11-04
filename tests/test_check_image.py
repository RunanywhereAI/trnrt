"""docker/check_image.py's version checks, without a Neuron install: the compiler versions it requires are the ones
README.md states, a wrong version fails, and transformers must have glm5_next (the Dockerfile pins one that does)."""

import importlib.util
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def ci():
    spec = importlib.util.spec_from_file_location("check_image", os.path.join(ROOT, "docker", "check_image.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_required_compiler_is_the_one_the_readme_states(ci):
    with open(os.path.join(ROOT, "README.md")) as f:
        readme = f.read()
    stated = re.search(r"neuronx-cc (\d+\.\d+)\)", readme)
    assert stated and ci.WANT_VERSION["neuronx-cc"] == stated.group(1) + ".", (stated, ci.WANT_VERSION)
    assert "Neuron SDK 2.32" in readme and ci.WANT_VERSION["libtorch-neuronx-lite"] == "2.11."  # SDK 2.32's


def test_dist_version_fails_another_release(ci):
    have = {"neuronx-cc": "2.27.5334.0+f702b353", "libtorch-neuronx-lite": "2.11.0.1.0.1284", "numpy": "9.9"}
    for d, v in have.items():
        assert ci.dist_version(d, version=have.get) == v
    for d, v in (("neuronx-cc", "2.28.1.0"), ("neuronx-cc", "2.2.1"), ("libtorch-neuronx-lite", "2.13.0.1.0.3112")):
        with pytest.raises(RuntimeError, match=f"{d} {v}, want"):
            ci.dist_version(d, version={d: v}.get)


def test_glm5_next_is_required(ci, monkeypatch):
    pytest.importorskip("transformers.models.glm5_next")
    assert ci.glm5_next().startswith("transformers ")
    monkeypatch.setitem(sys.modules, "transformers.models.glm5_next", None)  # as in a transformers without it
    with pytest.raises(ImportError):
        ci.glm5_next()


def test_the_image_pins_a_transformers_with_glm5_next():
    with open(os.path.join(ROOT, "docker", "Dockerfile")) as f:
        df = f.read()
    pins = dict(re.findall(r"\b(transformers|tokenizers)==(\d+\.\d+\.\d+)\b", df))
    assert set(pins) == {"transformers", "tokenizers"}, pins
    # 5.17.0 has glm5_next; the base image's 5.15.0 does not
    assert tuple(int(x) for x in pins["transformers"].split(".")) >= (5, 17, 0), pins
