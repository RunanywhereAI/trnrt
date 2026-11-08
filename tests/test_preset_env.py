"""The environment a preset needs before trnrt.kernels is imported (trnrt/__main__.py PRESET_ENV, apply_preset_env).

TRNRT_DMA_HW is read at kernel import (tests/test_dma_hw.py). So the CLI puts it into the process environment of
`trnrt engine`, and through it of every engine `trnrt serve` starts, and of `trnrt compile capture`, before anything
imports trnrt.kernels; the ranks they spawn inherit it. The CLI cases run in a fresh interpreter, since the knob is read
once per process, and each records the environment and the loaded kernel modules at the point the command hands on (the
start checks, the compile farm), then imports the kernels and records shell's REV: the import-time effect itself."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from trnrt import __main__ as cli

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MTP = sorted(p for p in cli.PRESETS if p.startswith("mtp-"))
SEEN = (
    "def seen():\n"
    "    return {'env': os.environ.get('TRNRT_DMA_HW'),\n"
    "            'kernels': sorted(m for m in sys.modules if m.startswith('trnrt.kernels'))}\n"
    "def after(rec, rc):\n"
    "    from trnrt.kernels import shell\n"
    "    print(json.dumps(dict(rec, rc=rc, rev=shell.REV, span=shell._dma_hw_rev())))\n"
)
ENGINE = (
    "import json, os, sys\n"
    "from trnrt import __main__ as cli\n"
    "from trnrt.engine import start_checks\n" + SEEN + "rec = {}\n"
    "def refusal(*_):\n"
    "    rec.update(seen())\n"
    "    return 'stopped by the test'\n"
    "start_checks.refusal = refusal\n"
    "after(rec, cli.cmd_engine(['--model', 'm', *sys.argv[1:]]))\n"
)
COMPILE = (
    "import json, os, sys\n"
    "from trnrt import __main__ as cli\n"
    "from trnrt.compile import farm\n" + SEEN + "rec = {}\n"
    "def main(argv, engine_config=None):\n"
    "    rec.update(seen())\n"
    "    return 0\n"
    "farm.main = main\n"
    "after(rec, cli.cmd_compile(sys.argv[1:]))\n"
)


def _run(code: str, args: list[str], env_value: str | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "TRNRT_DMA_HW"}
    if env_value is not None:
        env["TRNRT_DMA_HW"] = env_value
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", code, *args], env=env, capture_output=True, text=True, check=True, cwd=ROOT
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def plain_rev() -> int:
    """shell.REV in a process that never had the knob: what an unset run imports."""
    return _run(ENGINE, [])["rev"]


def test_every_mtp_preset_and_only_they_need_the_hardware_dge_knob():
    assert MTP == ["mtp-always", "mtp-batch12", "mtp-batch24", "mtp-batch8"]
    assert set(cli.PRESET_ENV) == set(MTP)
    assert all(cli.PRESET_ENV[p] == {"TRNRT_DMA_HW": "1"} for p in MTP)


@pytest.mark.parametrize("preset", MTP)
def test_engine_sets_it_for_an_mtp_preset_before_any_kernel_import(preset, plain_rev):
    r = _run(ENGINE, ["--preset", preset])
    assert r["rc"] == cli.START_REFUSED_STATUS  # the test's start check stopped it, after the environment was set
    assert r["env"] == "1"
    assert r["kernels"] == []
    assert r["rev"] == plain_rev ^ r["span"]  # the kernels imported afterwards are the hardware-DGE form


@pytest.mark.parametrize("preset", [None, "batch8"])
def test_engine_leaves_it_unset_for_the_other_configurations(preset, plain_rev):
    r = _run(ENGINE, [] if preset is None else ["--preset", preset])
    assert r["env"] is None
    assert r["kernels"] == []
    assert r["rev"] == plain_rev


def test_engine_keeps_a_value_the_caller_set(plain_rev):
    r = _run(ENGINE, ["--preset", "mtp-batch8"], env_value="0")
    assert r["env"] == "0"
    assert r["rev"] == plain_rev


@pytest.mark.parametrize("preset", MTP)
def test_compile_capture_sets_it_from_the_engine_arguments_before_the_farm_runs(preset, plain_rev):
    r = _run(COMPILE, ["capture", "--out", "o", "--", "--model", "m", "--preset", preset])
    assert r["rc"] == 0
    assert r["env"] == "1"
    assert r["kernels"] == []
    assert r["rev"] == plain_rev ^ r["span"]


def test_compile_capture_without_a_preset_and_the_other_subcommands_leave_it_unset(plain_rev):
    for args in (["capture", "--out", "o", "--", "--model", "m"], ["strip", "--out", "o"]):
        r = _run(COMPILE, args)
        assert r["env"] is None
        assert r["rev"] == plain_rev


def test_a_process_that_imported_the_kernels_without_it_is_refused(monkeypatch):
    import trnrt.kernels.shell  # noqa: F401  this session's kernels, bound without the knob

    monkeypatch.setenv("TRNRT_DMA_HW", "x")  # so that teardown restores the variable's absence
    monkeypatch.delenv("TRNRT_DMA_HW")
    with pytest.raises(RuntimeError, match=r"before trnrt\.kernels is imported"):
        cli.apply_preset_env(["--model", "m", "--preset", "mtp-batch8"])
    assert "TRNRT_DMA_HW" not in os.environ


def test_a_value_already_set_is_kept_and_not_refused_after_a_kernel_import(monkeypatch):
    import trnrt.kernels.shell  # noqa: F401

    monkeypatch.setenv("TRNRT_DMA_HW", "1")
    assert cli.apply_preset_env(["--model", "m", "--preset", "mtp-batch8"]) == {}
    assert cli.apply_preset_env(["--model", "m"]) == {}
    assert os.environ["TRNRT_DMA_HW"] == "1"
