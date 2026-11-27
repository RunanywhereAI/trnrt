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


@pytest.fixture(autouse=True)
def _off_neuron_host(monkeypatch):
    """The CPU suite sees an off-Neuron host wherever it runs. trnrt.platform.target() reads the EC2 instance type and the
    driver's device names, so on a trn2 host it says trn2 and the kernels' program count becomes the runtime's LNC (2):
    the tiny test models then fail kda_half.takes_spec (H // programs must be a multiple of 512) and the shells never run.
    A test that needs the trn2 target sets NEURON_PLATFORM_TARGET_OVERRIDE, which target() reads first."""
    from trnrt import platform

    monkeypatch.setattr(platform, "DMI_PRODUCT", "/nonexistent/trnrt-tests/product_name")
    monkeypatch.setattr(platform, "SYSFS_DEVICE_NAME", "/nonexistent/trnrt-tests/neuron*/device_name")


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
