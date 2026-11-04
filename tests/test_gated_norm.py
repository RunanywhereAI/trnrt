"""kernels/gated_norm.py: KDA's gated RMSNorm as an NKI kernel at LNC = 2. reference() against _mix_rows's tail on the
host, the dispatch (takes2) and the REV span; the kernel in the NKI CPU simulator at grid 2 against reference() (Neuron
venv only)."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from trnrt.kernels import gated_norm as gn


def _inputs(T, Hv, Dv=128, pad=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    scale = torch.pow(10.0, torch.rand(T + pad, Hv, 1, generator=g) * 2 - 1)
    o = torch.randn(T + pad, Hv, Dv, generator=g) * scale
    z = (torch.randn(T, Hv, Dv, generator=g) * 2).to(torch.bfloat16)
    w = 1 + 0.1 * torch.randn(Dv, generator=g)
    return o, z, w


def test_reference_is_mix_rows_tail():
    """reference() is linear_attn._mix_rows's gated norm, element for element."""
    o, z, w = _inputs(64, 3)
    eps = 1e-5
    of = o.to(torch.bfloat16).float()
    of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    of = w * of
    want = (of * torch.sigmoid(z.float())).to(torch.bfloat16).reshape(64, -1)
    assert torch.equal(gn.reference(o, z, w, eps), want)


def test_takes2_needs_a_split_grid_and_whole_tiles(monkeypatch):
    """dv <= 512 at a grid above 1 with T whole tiles for every program; at grid 1 nothing takes."""
    from trnrt import platform

    monkeypatch.setattr(platform, "nki_grid", lambda: 2)
    assert gn.takes2(128, 4096) and gn.takes2(128, 1024) and gn.takes2(128, 256)
    assert not gn.takes2(128, 128)  # one tile: not one for each of two programs
    assert not gn.takes2(128, 300) and not gn.takes2(128, 0)
    assert not gn.takes2(1024, 4096)
    monkeypatch.setattr(platform, "nki_grid", lambda: 1)
    assert not gn.takes2(128, 4096)


def test_rev_is_the_kernel_span_and_p():
    """REV is a CRC of the span holding the kernel (its def inside it) and of P, the constant it reads."""
    import zlib

    src = open(gn.__file__).read()
    a = src.index("# -- the kernel (its span is what REV hashes)")
    b = src.index("    gated_norm2_kernel = None", a)
    assert "def gated_norm2_kernel(" in src[a:b] and "nisa.activation_reduce" in src[a:b]
    assert zlib.crc32(repr((gn.P,)).encode(), zlib.crc32(src[a:b].encode())) == gn.REV
    assert zlib.crc32(repr((gn.P * 2,)).encode(), zlib.crc32(src[a:b].encode())) != gn.REV  # P re-keys it


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("T", [256, 512, 4096])
def test_nki_simulator_grid2_matches_reference(T, monkeypatch):
    """The served form (fix 4, fp32 o) at grid 2 is within one bf16 ulp of reference() on bf16-representable o (where
    reference()'s bf16 copy is exact), every row written, at the served heads (2 a rank) and Dv 128."""
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    o, z, w = _inputs(T, 2, seed=T)
    o[3] = 0.0  # a zero row (rsqrt(eps)), and huge / tiny rows
    o[T - 1] *= 1e6
    o[T // 2] *= 1e-6
    o = o.to(torch.bfloat16).float()
    got = torch.as_tensor(nki.simulate(gn.gated_norm2_kernel[2])(o=o, z=z, w=w, eps=1e-5, rev=gn.REV, fix=gn.FIX))
    want = gn.reference(o, z, w, 1e-5)
    assert got.shape == want.shape and got.dtype == torch.bfloat16
    assert not torch.isnan(got.float()).any()  # the simulator's unwritten elements are NaN
    diff = (got.float() - want.float()).abs()
    spacing = torch.pow(2.0, torch.floor(torch.log2(want.float().abs().clamp_min(1e-30))) - 7)
    assert (diff <= spacing * 1.01).all()
