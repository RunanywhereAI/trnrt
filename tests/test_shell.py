"""kernels/shell.py, the decode-shell frame's shared helpers: the host side on CPU; the NKI helpers in the simulator
(Neuron venv) inside a small kernel, against numpy."""

from __future__ import annotations

import importlib.util
import os
import tempfile

import numpy as np
import pytest
import torch

from trnrt.kernels import shell as SH


def test_program_units_split_evenly_or_refuse():
    assert [SH.program_units(2, 2, p) for p in (0, 1)] == [(0, 1), (1, 1)]
    assert SH.program_units(8, 1, 0) == (0, 8)
    with pytest.raises(ValueError):
        SH.program_units(3, 2, 0)


def test_ring_rows():
    """A ring of per-row tiles needs PD + ceil(SKEW / units per row) + 1 entries: 1 unit per row 5, 2 units 4."""
    assert SH.ring_rows(2, 2, 1) == 5 and SH.ring_rows(2, 2, 2) == 4 and SH.ring_rows(2, 2, 8) == 4


def test_register_decode_only_lists_every_tensor():
    from torch import nn

    layer = nn.Module()
    layer._decode_only = ("w_dec",)
    SH.register_decode_only(layer, "xh_", {"wp": ((2, 4, 3), torch.bfloat16), "s": ((2,), torch.float32)})
    assert layer._decode_only == ("w_dec", "xh_wp", "xh_s")
    assert tuple(layer.xh_wp.shape) == (2, 4, 3) and layer.xh_wp.dtype == torch.bfloat16
    assert {n for n, _ in layer.named_parameters()} == {"xh_wp", "xh_s"}


def test_slots_reads_the_given_row_at_position_0():
    s = SH.slots(torch.tensor([3, 5]), torch.tensor([0, 9]), rows=7)
    assert s.dtype == torch.int32 and s.tolist() == [[7, 5], [3, 5]]


def test_a_sequence_start_reads_the_zero_row():
    """The state pools have one more row, the last, zeros and in no free list: a row starting its sequence (warmup's
    rows, padded rows) reads it, in range (an out-of-range read, which the hardware-DGE reads skip, hung a serve
    warmup)."""
    assert SH.start_row(9) == 8
    s = SH.slots(torch.tensor([3, 5]), torch.tensor([0, 9]), rows=SH.start_row(9))
    assert s.tolist() == [[8, 5], [3, 5]]


def test_add_partials_rounds_the_fp32_sum_once():
    a = torch.tensor([[[1.0, 0.00390625]], [[0.00390625, 1.0]]])
    y = SH.add_partials(a, torch.bfloat16)
    assert y.dtype == torch.bfloat16 and torch.equal(y, (a[0] + a[1]).to(torch.bfloat16))


def test_kernel_rev_follows_the_source_between_its_markers():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("head\n# BEGIN\nbody 1\n# END\ntail\n")
    try:
        r1 = SH.kernel_rev(f.name, "# BEGIN", "# END")
        open(f.name, "w").write("head 2\n# BEGIN\nbody 1\n# END\ntail 2\n")
        assert SH.kernel_rev(f.name, "# BEGIN", "# END") == r1  # outside the markers: the same rev
        open(f.name, "w").write("head\n# BEGIN\nbody 2\n# END\ntail\n")
        assert SH.kernel_rev(f.name, "# BEGIN", "# END") != r1
    finally:
        os.unlink(f.name)


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
def test_the_nki_helpers_in_the_simulator(monkeypatch):
    """rsqrt_bcast and sigmoid_exp inside a kernel, with dma_static in and out, against numpy."""
    import nki
    import nki.isa as nisa
    import nki.language as nl

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

    @nki.jit
    def k(x):
        out = nl.ndarray((2, 128, 8), dtype=nl.float32, buffer=nl.shared_hbm)
        xt = nl.ndarray((128, 8), dtype=nl.float32, buffer=nl.sbuf)
        SH.dma_static(dst=xt, src=x)
        oc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=oc, value=1.0)
        orr = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=orr, value=1.0)
        sq = nl.ndarray((128, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=xt, data2=xt, op=nl.multiply, engine=nisa.vector_engine)
        rb = SH.rsqrt_bcast(oc, orr, sq, 8, 1.0 / 128, 1e-5)
        sg = nl.ndarray((128, 8), dtype=nl.float32, buffer=nl.sbuf)
        e = nl.ndarray((128, 8), dtype=nl.float32, buffer=nl.sbuf)
        SH.sigmoid_exp(sg, xt, e)
        SH.dma_static(dst=out[0], src=rb)
        SH.dma_static(dst=out[1], src=sg)
        return out

    x = np.random.default_rng(0).standard_normal((128, 8)).astype(np.float32)
    got = np.asarray(nki.simulate(k)(x=x))
    want_r = 1.0 / np.sqrt((x * x).mean(0, keepdims=True) + 1e-5)
    assert np.allclose(got[0], np.broadcast_to(want_r, (128, 8)), rtol=1e-5)
    assert np.allclose(got[1], 1.0 / (1.0 + np.exp(-x)), rtol=1e-5, atol=1e-6)


def test_packed_layout_is_aligned_disjoint_and_in_order():
    shapes = {"wp": ((2, 4, 300), torch.bfloat16), "dtb": ((2, 128, 1), torch.float32),
              "outT": ((128, 2, 8), torch.bfloat16), "o_norm": ((128,), torch.float32)}
    lay = SH.packed_layout(shapes)
    assert set(lay) == {torch.bfloat16, torch.float32}
    n, idx = lay[torch.bfloat16]
    assert idx == {"wp": (0, (2, 4, 300)), "outT": (2560, (128, 2, 8))}  # 2400 elements -> next multiple of 256
    assert n == 2560 + 2048 and lay[torch.float32] == (512, {"dtb": (0, (2, 128, 1)), "o_norm": (256, (128,))})


def test_register_packed_views_fill_their_own_bytes():
    from torch import nn

    shapes = {"a": ((3, 5), torch.bfloat16), "b": ((7,), torch.float32), "c": ((2, 2), torch.bfloat16)}
    layer = nn.Module()
    SH.register_packed(layer, "xh_", shapes)
    assert layer._decode_only == ("xh_blob_bf16", "xh_blob_f32")
    layer.xh_blob_bf16.data.zero_()
    layer.xh_blob_f32.data.zero_()
    v = SH.packed_views(layer, "xh_", shapes)
    v["a"].copy_(torch.arange(15.0).view(3, 5))
    v["c"].fill_(-1)
    v["b"].fill_(2)
    blob = layer.xh_blob_bf16.data
    assert torch.equal(blob[:15], torch.arange(15.0).to(torch.bfloat16)) and blob[15:256].abs().sum() == 0
    assert torch.equal(blob[256:260], torch.full((4,), -1.0, dtype=torch.bfloat16))
    assert torch.equal(SH.packed_views(layer, "xh_", shapes)["b"], torch.full((7,), 2.0))
