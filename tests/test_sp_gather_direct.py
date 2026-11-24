"""The sequence-parallel gather's direct form (kernels/sp_gather.py DIRECT, EngineConfig.sp_gather_direct): gather()
takes the direct kernel only with the route, with its own revision (which covers the helpers its kernel traces), and the
copy-out kernel's revision does not move with the direct section."""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest
import torch

from trnrt import platform
from trnrt.config import EngineConfig
from trnrt.kernels import sp_gather


@pytest.fixture
def fake_wrap(monkeypatch):
    """wrap_nki(kernel)[grid](**kw) -> (kernel, kw), without the Neuron runtime."""
    hop = types.ModuleType("libtorch_neuronx_lite.nki.nki_hop")

    class _W:
        def __init__(self, k):
            self.k = k

        def __getitem__(self, grid):
            return lambda **kw: (self.k, grid, kw)

    hop.wrap_nki = _W
    for name in ("libtorch_neuronx_lite", "libtorch_neuronx_lite.nki"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "libtorch_neuronx_lite.nki.nki_hop", hop)
    monkeypatch.setattr(platform, "nki_grid", lambda: 2)
    monkeypatch.setattr(sp_gather, "sp_gather_dupf_kernel", "dupf")
    monkeypatch.setattr(sp_gather, "sp_gather_dupfd_kernel", "dupfd")


@pytest.mark.parametrize(("direct", "kernel", "rev"), [(False, "dupf", "REV"), (True, "dupfd", "REV_DIRECT")])
def test_gather_takes_the_direct_kernel_only_with_the_route(monkeypatch, fake_wrap, direct, kernel, rev):
    monkeypatch.setattr(sp_gather, "DIRECT", direct)
    k, grid, kw = sp_gather.gather(torch.zeros(4, 8), 32, 32)
    assert (k, grid, kw["rev"], kw["world"], kw["gsize"]) == (kernel, 2, getattr(sp_gather, rev), 32, 32)


def test_the_direct_form_has_its_own_revision():
    assert sp_gather.REV_DIRECT != sp_gather.REV
    with open(sp_gather.__file__) as f:
        src = f.read()
    assert src.index("    sp_gather_dupf_kernel = None") < src.index("# -- the direct form")  # REV's span ends before


def _revs(tmp_path, name: str, old: str | None = None, new: str = ""):
    """(REV, REV_DIRECT) of a copy of kernels/sp_gather.py, with `old` (once) replaced by `new`: each rev hashes its file."""
    with open(sp_gather.__file__) as f:
        src = f.read()
    if old is not None:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    path = tmp_path / f"{name}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"sp_gather_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.REV, mod.REV_DIRECT


def test_the_direct_revision_covers_the_helpers_its_kernel_traces(tmp_path):
    """The direct kernel calls _groups (above its section): an edit there moves REV_DIRECT, as an edit of its own section
    does, and an edit of the direct section leaves the copy-out kernel's REV where it is."""
    base = _revs(tmp_path, "base")
    assert base == (sp_gather.REV, sp_gather.REV_DIRECT)
    groups = _revs(tmp_path, "groups", "ranks.append(g * gsize + i)", "ranks.append(i + g * gsize)")
    assert groups[1] != base[1] and groups[0] != base[0]
    direct = _revs(tmp_path, "direct", "ncc.all_gather(srcs=[xs1], dsts=[z1]", "ncc.all_gather(srcs=[xs1],  dsts=[z1]")
    assert direct[1] != base[1] and direct[0] == base[0]


def test_direct_needs_the_kernel_gather():
    with pytest.raises(ValueError, match="sp_gather_direct needs"):
        EngineConfig(model_path="m", sp_gather_direct=True)
    assert EngineConfig(model_path="m", sp_gather="nki", sp_gather_direct=True).sp_gather_direct
