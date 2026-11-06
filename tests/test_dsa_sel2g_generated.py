"""The 3,072-page decode bucket's generated kernels are the generators' output for the current sources:
trnrt/kernels/dsa_sel2g.py (tools/make_sel2g.py, from dsa_shell.py), dsa_satwfg.py and dsa_satwfcg.py
(tools/make_satwf_sel2g.py, from dsa_satwf.py and dsa_satwfc.py). A source edit without a regeneration fails here, so a copy
can't drift from the kernel it was made from."""

from __future__ import annotations

import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dsa_sel2g_is_the_transform_of_dsa_shell():
    m = _tool("make_sel2g")
    with open(m.SRC) as f:
        want = m.transform(f.read())
    with open(m.DST) as f:
        assert f.read() == want, "trnrt/kernels/dsa_sel2g.py is stale: run python tools/make_sel2g.py"


@pytest.mark.parametrize("name", ["satwfg", "satwfcg"])
def test_satwf_sel2g_copies_are_the_transform_of_their_sources(name):
    m = _tool("make_satwf_sel2g")
    src, dst = m.paths(name)
    with open(src) as f:
        want = m.transform(name, f.read())
    with open(dst) as f:
        assert f.read() == want, f"{os.path.basename(dst)} is stale: run python tools/make_satwf_sel2g.py"


def test_the_copies_hold_no_dispatch_to_themselves():
    """dsa_satwf's SEL2G block (its dispatch to dsa_satwfg) is cut from the copy, so the copy never calls itself."""
    with open(os.path.join(ROOT, "trnrt", "kernels", "dsa_satwfg.py")) as f:
        src = f.read()
    assert "_satwf_pre_sel2g" not in src and "from . import dsa_satwfcg as dsa_satwfc" in src


def test_takes_only_the_partial_chunk_buckets():
    """dsa_sel2g.takes routes a decode call to the copies only where sel2 cannot build: whole 128-page groups whose score
    tile (NT = 8 per group) is past one chunk and not whole 128-column chunks (P3072: NT 192)."""
    torch = pytest.importorskip("torch")
    from trnrt.kernels import dsa_sel2g

    def t(npg):
        return torch.zeros(2, npg, dtype=torch.int32)

    assert dsa_sel2g.takes(t(3072))  # NT 192
    assert not dsa_sel2g.takes(t(1024))  # NT 64: one chunk
    assert not dsa_sel2g.takes(t(4096))  # NT 256: whole chunks
    assert not dsa_sel2g.takes(t(512))
