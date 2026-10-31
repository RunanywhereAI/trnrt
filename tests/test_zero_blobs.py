"""The shells' packed decode-only blobs (shell.register_packed: kda_shell, moe_half, ffn_half.pack) hold zeros wherever
no packed tensor lies, after the shell's own fill(). A fill writes only its views, so the alignment padding is whatever
the blob was created with; created empty, its bytes vary run to run and break any byte-level digest of a step's inputs.
The allocator is made to hand back NaN-filled memory here, so a blob left empty cannot pass by landing on fresh,
OS-zeroed pages."""

from __future__ import annotations

import math

import pytest
import torch

from trnrt.kernels import kda_shell as KS
from trnrt.kernels import moe_half as MH
from trnrt.kernels import shell as SH
from tests import test_kda_shell as TKS
from tests.test_ffn_half import fp8_layer


def _nan_empty(monkeypatch):
    real = torch.empty

    def empty(*a, **k):
        t = real(*a, **k)
        return t.fill_(float("nan")) if t.is_floating_point() else t

    monkeypatch.setattr(torch, "empty", empty)


def _padding(shapes: dict) -> dict:
    """{dtype: bool mask of the blob's elements that no packed tensor covers}."""
    out = {}
    for dt, (n, idx) in SH.packed_layout(shapes).items():
        m = torch.ones(n, dtype=torch.bool)
        for off, shape in idx.values():
            m[off:off + math.prod(shape)] = False
        out[dt] = m
    return out


def _filled(shell: str, monkeypatch):
    """(layer, prefix, shapes) of a layer whose blobs the shell registered on the host and its fill() filled."""
    if shell == "kda_shell":
        layer, _ = TKS._layer(monkeypatch, 2)
        KS.fill(layer)
        return layer, KS.PREFIX, KS.shapes(*layer.ks_dims)
    monkeypatch.setattr(SH, "programs", lambda: 2)
    E, H, Is = 8, 256, 8
    layer = fp8_layer(E, H, Is)
    SH.register_packed(layer, MH.PREFIX, MH.shapes(E, H, Is, 2))  # moe_half.register's call (its takes() needs a cfg)
    MH.fill(layer, torch.bfloat16)
    return layer, MH.PREFIX, MH.shapes(E, H, Is, 2)


def _check(layer, prefix: str, shapes: dict):
    pads = _padding(shapes)
    assert pads[torch.float32].any()  # the fp32 blob pads (the bias, A_log, ... are short); the bf16 one may not
    for dt, pad in pads.items():
        blob = getattr(layer, prefix + "blob_" + SH.DT_NAMES[dt]).data
        assert torch.count_nonzero(blob[pad]) == 0, (dt, int(torch.count_nonzero(blob[pad])), int(pad.sum()))


@pytest.mark.parametrize("shell", ["kda_shell", "moe_half"])
def test_host_registered_blobs_pad_with_zeros(shell, monkeypatch):
    _nan_empty(monkeypatch)
    _check(*_filled(shell, monkeypatch))


@pytest.mark.parametrize("shell", ["kda_shell", "moe_half"])
def test_meta_built_blobs_materialize_with_zero_padding(shell, monkeypatch):
    """The served path: the model is built on meta (loader.load_model), so the blobs are meta parameters until
    loader._materialize gives them host storage; then the fill, then done(). The stand-in layer's other weights are
    host tensors, as the checkpoint's would be; only its blobs are re-registered on meta. A meta parameter that is not
    decode-only still materializes empty (the allocator's NaN here): only _decode_only is zeroed."""
    from torch import nn

    from trnrt.models import loader as LD

    layer, prefix, shapes = _filled(shell, monkeypatch)
    with torch.device("meta"):
        SH.register_packed(layer, prefix, shapes)  # replaces the host blobs (_decode_only lists them twice: harmless)
        layer.not_decode_only = nn.Parameter(torch.zeros(4), requires_grad=False)
    assert getattr(layer, prefix + "blob_f32").device.type == "meta"
    _nan_empty(monkeypatch)
    done = LD._materialize(layer, torch.device("cpu"))
    KS.fill(layer) if shell == "kda_shell" else MH.fill(layer, torch.bfloat16)
    done()
    _check(layer, prefix, shapes)
    assert layer.not_decode_only.isnan().all()


def test_ffn_half_pack_pads_with_zeros(monkeypatch):
    """ffn_half builds its kernel's blobs on the host through shell.register_packed and moe_half.fill
    (ffn_half.pack), so its fp32 blob's padding is zero too."""
    from trnrt.kernels import ffn_half as FH

    _nan_empty(monkeypatch)
    E, H, Is, hc = 8, 256, 8, 4
    L = fp8_layer(E, H, Is)
    O = 2 * hc + hc * hc
    b = FH.pack(L.router, L.router_bias, L.shared_gate_up, L.shared_gate_up_scale, L.shared_down, L.shared_down_scale,
                torch.zeros(O, hc * H), torch.ones(H), torch.zeros(O), torch.ones(3), npg=2)
    pads = _padding(MH.shapes(E, H, Is, 2))
    assert pads[torch.float32].any()
    for dt, pad in pads.items():
        assert torch.count_nonzero(b[dt][pad]) == 0, (dt, int(torch.count_nonzero(b[dt][pad])), int(pad.sum()))
