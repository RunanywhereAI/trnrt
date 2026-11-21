"""The KDA, mHC and FFN-half fields of EngineConfig and their routes: engine.set_decode_routes copies each into the
module that reads it when a graph is traced, every rank before it builds its shard, and sets every one, so a later
engine in the process inherits none of an earlier engine's."""

from __future__ import annotations

import pytest

from trnrt.config import EngineConfig
from trnrt.engine import engine as E
from trnrt.models import hybrid


@pytest.fixture(autouse=True)
def _routes(monkeypatch):
    """Whatever set_decode_routes writes is undone after each test: every module constant (an upper-case name) of the
    modules it writes."""
    from trnrt.kernels import dense_half, dsa_satwf, sp_gather
    from trnrt.models import decoder, mla

    for mod in (hybrid, mla, sp_gather, dsa_satwf, dense_half, decoder):
        for name, v in list(vars(mod).items()):
            if name.isupper():
                monkeypatch.setattr(mod, name, v)


def test_the_kda_shell_switch():
    assert EngineConfig(model_path="m").kda_shell is True  # the default: today's behaviour
    E.set_decode_routes(EngineConfig(model_path="m", kda_shell=False))
    assert hybrid.KDA_SHELL is False
    E.set_decode_routes(EngineConfig(model_path="m"))
    assert hybrid.KDA_SHELL is True
