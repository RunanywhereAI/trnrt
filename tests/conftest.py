"""Suite-wide fixtures."""

import logging

import pytest

from trnrt import log


@pytest.fixture(autouse=True)
def _restore_trnrt_logging():
    """Entry points (`main()` of the CLI, the router, check_mixed) configure trnrt's logger for their process; tests run
    several in one process, so each test gets the logger back as it found it."""
    root = logging.getLogger(log.ROOT)
    handlers, level, propagate = list(root.handlers), root.level, root.propagate
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    root.propagate = propagate


# Every module engine.set_decode_routes writes (tests/test_decode_route_modules.py checks the list against its imports).
ROUTE_MODULES = (
    "trnrt.kernels.dense_half",
    "trnrt.kernels.dsa_satwf",
    "trnrt.kernels.dsa_satwfcga",
    "trnrt.kernels.dsa_sel2gc",
    "trnrt.kernels.dsa_slots_n_p",
    "trnrt.kernels.dsa_topk",
    "trnrt.kernels.shell",
    "trnrt.kernels.sp_gather",
    "trnrt.models.decoder",
    "trnrt.models.hybrid",
    "trnrt.models.linear_attn",
    "trnrt.models.mla",
)


@pytest.fixture
def decode_routes(monkeypatch):
    """Whatever engine.set_decode_routes writes is undone after the test: every module constant (an upper-case name,
    including the private ones set_layouts rebinds) of every module it writes. A test that builds an engine or calls
    set_decode_routes uses it, so a later test in the process sees the modules' own values."""
    import importlib

    for name in ROUTE_MODULES:
        mod = importlib.import_module(name)
        for attr in [a for a in vars(mod) if a.isupper()]:
            monkeypatch.setattr(mod, attr, getattr(mod, attr))
