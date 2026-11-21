"""The DSA layers' routes in EngineConfig (trnrt/config.py, the DSA block): engine.set_decode_routes copies each field
into the module constant the traced code reads, every engine sets every one (a later engine inherits nothing), and each
route changes the call it names and no other."""

from __future__ import annotations

import pytest
import torch

from trnrt.config import EngineConfig
from trnrt.engine import engine as E
from trnrt.kernels import dsa_slots_n_p, dsa_topk, sp_gather
from trnrt.models import mla

# field: (module, constant, a non-default value, the constant it then holds); set_decode_routes writes each
FIELDS = {
    "dsa_topk_rowsplit_big": (dsa_topk, "ROWSPLIT_BIG", True, True),
    "dsa_cp_fail_trip": (mla, "CP_FAIL_TRIP", True, True),
    "dsa_cp_fail_attn_trip": (mla, "CP_FAIL_ATTN_TRIP", True, True),
    "dsa_slots_pipe_rows": (dsa_slots_n_p, "PIPE_ROWS", 64, 64),
    "sp_gather_direct": (sp_gather, "DIRECT", True, True),
    "dsa_drop_unread_o_dec": (mla, "ODEC_DROP", True, True),
}
# what a field needs beside it
NEEDS = {"sp_gather_direct": {"sp_gather": "nki"}}
ROUTES = sorted({(m, n) for m, n, _, _ in FIELDS.values()}, key=lambda r: (r[0].__name__, r[1]))


@pytest.fixture
def routes(monkeypatch):
    for mod, name in ROUTES:
        monkeypatch.setattr(mod, name, getattr(mod, name))


def _cfg(**kw) -> EngineConfig:
    return EngineConfig(model_path="m", **kw)


def test_defaults_are_the_modules_own(routes):
    """The fields' defaults are the module constants at import: an engine built without them traces what the modules
    alone would."""
    before = {name: getattr(mod, name) for mod, name in ROUTES}
    E.set_decode_routes(_cfg())
    assert {name: getattr(mod, name) for mod, name in ROUTES} == before


@pytest.mark.parametrize("field", sorted(FIELDS))
def test_each_field_reaches_its_module_and_resets(routes, field):
    mod, name, value, want = FIELDS[field]
    default = getattr(mod, name)
    E.set_decode_routes(_cfg(**{field: value}, **NEEDS.get(field, {})))
    assert getattr(mod, name) == want
    E.set_decode_routes(_cfg())  # a later engine inherits nothing
    assert getattr(mod, name) == default


def test_set_decode_routes_writes_every_field():
    """Every DSA field's constant is one set_decode_routes assigns (its source names it)."""
    import inspect

    src = inspect.getsource(E.set_decode_routes)
    for field, (mod, name, _, _) in FIELDS.items():
        assert f"{mod.__name__.rsplit('.', 1)[1]}.{name} = " in src and f"cfg.{field}" in src, field


@pytest.mark.parametrize(
    ("big", "rows", "want"),
    [(False, 128, (1, 2)), (True, 128, (2, 1)), (True, 32, (4, 2)), (False, 8, (32, 1)), (True, 8, (32, 1))],
)
def test_rowsplit_big_splits_only_the_large_one_tile_calls(monkeypatch, big, rows, want):
    """(g, tiles) of a split grid-2 call over 4,096 scores a row: a 128-row call (a prefill chunk's local selection)
    takes the two-tile form only with the route; a 32-row call (neither a decode call nor 64 rows) never; an 8-row
    decode call always (the row split's own form)."""
    monkeypatch.setattr(dsa_topk, "ROWSPLIT_BIG", big)
    k = dsa_topk.kernel_inputs(torch.zeros(rows, 4096), 512, rowsplit=True)
    assert (k["g"], k["tiles"]) == want
