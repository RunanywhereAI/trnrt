"""The sequence-parallel gather's direct form (kernels/sp_gather.py DIRECT, EngineConfig.sp_gather_direct): gather()
takes the direct kernel only with the route, with its own revision, and the copy-out kernel's revision does not move
with the direct section."""

from __future__ import annotations

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


def test_direct_needs_the_kernel_gather():
    with pytest.raises(ValueError, match="sp_gather_direct needs"):
        EngineConfig(model_path="m", sp_gather_direct=True)
    assert EngineConfig(model_path="m", sp_gather="nki", sp_gather_direct=True).sp_gather_direct
