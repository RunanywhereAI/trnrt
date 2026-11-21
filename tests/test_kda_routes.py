"""The KDA, mHC and FFN-half fields of EngineConfig and their routes: engine.set_decode_routes copies each into the
module that reads it when a graph is traced, every rank before it builds its shard, and sets every one, so a later
engine in the process inherits none of an earlier engine's."""

from __future__ import annotations

import pytest

from trnrt.config import EngineConfig
from trnrt.engine import engine as E
from trnrt.models import hybrid

pytestmark = pytest.mark.usefixtures("decode_routes")  # set_decode_routes's writes are undone


@pytest.fixture(autouse=True)
def _routes(monkeypatch):
    """Whatever set_decode_routes writes is undone after each test: every module constant (an upper-case name) of the
    modules it writes."""
    from trnrt.kernels import dense_half, dsa_satwf, shell, sp_gather
    from trnrt.models import decoder, linear_attn, mla

    for mod in (hybrid, mla, sp_gather, dsa_satwf, dense_half, decoder, shell, linear_attn):
        for name, v in list(vars(mod).items()):
            if name.isupper():
                monkeypatch.setattr(mod, name, v)


def test_the_kda_shell_switch():
    assert EngineConfig(model_path="m").kda_shell is True  # the default: today's behaviour
    E.set_decode_routes(EngineConfig(model_path="m", kda_shell=False))
    assert hybrid.KDA_SHELL is False
    E.set_decode_routes(EngineConfig(model_path="m"))
    assert hybrid.KDA_SHELL is True


def test_the_mhc_seam_switch():
    assert EngineConfig(model_path="m").mhc_seam is False
    E.set_decode_routes(EngineConfig(model_path="m", mhc_seam=True))
    assert hybrid.MHC_SEAM is True
    E.set_decode_routes(EngineConfig(model_path="m"))
    assert hybrid.MHC_SEAM is False


def test_the_mix_row_split_switch():
    from trnrt.kernels import mix_wide, shell

    assert EngineConfig(model_path="m").shell_mix_row_split is False
    assert shell.mix_rev_term() == 0
    E.set_decode_routes(EngineConfig(model_path="m", shell_mix_row_split=True))
    assert shell.MIX_ROW_SPLIT is True and shell.mix_rev_term() == mix_wide.RSPLIT_REV != 0
    E.set_decode_routes(EngineConfig(model_path="m"))
    assert shell.MIX_ROW_SPLIT is False and shell.mix_rev_term() == 0


def test_the_mid_export_switch():
    from trnrt.models import linear_attn

    assert EngineConfig(model_path="m").state_ckpt_mid_export is False
    E.set_decode_routes(EngineConfig(model_path="m", state_ckpt_mid_export=True))
    assert linear_attn.MID_EXPORT is True
    E.set_decode_routes(EngineConfig(model_path="m"))
    assert linear_attn.MID_EXPORT is False
