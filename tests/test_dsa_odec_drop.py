"""models/mla.drop_unread_o_dec (EngineConfig.dsa_drop_unread_o_dec): o_dec goes only where the fused layer calls' o_t
exists, every other parameter stays, and it leaves _decode_only; a layer without o_t (the MTP layer is never passed)
keeps it. load_layer then fills o_t from dequant(o, o_scale), the value o_dec held."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from trnrt.models import mla

pytestmark = pytest.mark.usefixtures("decode_routes")  # set_decode_routes's writes are undone


def _layer(with_ot: bool) -> nn.Module:
    m = nn.Module()
    for n in ("o", "o_dec", "w_a_dec", *(("o_t",) if with_ot else ())):
        setattr(m, n, nn.Parameter(torch.zeros(2), requires_grad=False))
    m._decode_only = ("w_a_dec", "o_dec", *(("o_t",) if with_ot else ()))
    return m


def test_drops_o_dec_only_beside_o_t():
    a, b = _layer(True), _layer(False)
    assert mla.drop_unread_o_dec([a, b]) == 1
    assert not hasattr(a, "o_dec") and "o_dec" not in a._decode_only
    assert hasattr(a, "o_t") and hasattr(a, "w_a_dec") and hasattr(a, "o")
    assert hasattr(b, "o_dec") and "o_dec" in b._decode_only
    assert mla.drop_unread_o_dec([a, b]) == 0  # idempotent


def test_the_xla_decode_branch_falls_back_to_the_stored_weight():
    """_wd reads model._w(layer, "o") once the copy is gone (what attention_cp's XLA decode branch reads)."""
    layer = _layer(True)
    mla.drop_unread_o_dec([layer])
    model = type("M", (), {"_w": staticmethod(lambda lyr, name: ("stored", name))})()
    assert mla._wd(model, layer, "o", True) == ("stored", "o")


def test_route_defaults_off():
    assert mla.ODEC_DROP is False
