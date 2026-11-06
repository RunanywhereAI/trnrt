"""TRNRT_DMA_HW: the static DMA helpers (shell.dma_static, kda_half._dma, mix_wide._dma) in hardware DGE, read at import.

Unset (the default presets), each module's REV is its hashed span's CRC exactly as before. Set (every MTP preset, as in every build
behind the published MTP numbers), REV also folds the hardware-DGE span's CRC, so every kernel folding REV re-keys, and the helper
a kernel binds by name at import (dsa_shell's `from .shell import dma_static`) is the hardware-DGE form. Each case imports the
modules in a fresh interpreter, since the knob is read once at import."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODS = ("shell", "kda_half", "mix_wide")
needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")


def _probe(env_value: str | None, code: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "TRNRT_DMA_HW"}
    if env_value is not None:
        env["TRNRT_DMA_HW"] = env_value
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(out.stdout.strip().splitlines()[-1])


REVS = (
    "import json\n"
    "from trnrt.kernels import kda_half, mix_wide, shell\n"
    "m = {'shell': shell, 'kda_half': kda_half, 'mix_wide': mix_wide}\n"
    "print(json.dumps({n: [x.REV, x._dma_hw_rev()] for n, x in m.items()} | {'mix_rev': mix_wide.MIX_REV}))\n"
)


def test_unset_and_zero_leave_rev_as_the_hashed_span():
    off, zero = _probe(None, REVS), _probe("0", REVS)
    assert off == zero
    for n in MODS:
        assert off[n][0] != off[n][0] ^ off[n][1]  # the span's CRC is not zero, so a fold would show


def test_set_folds_the_span_into_every_rev():
    off, on = _probe(None, REVS), _probe("1", REVS)
    for n in MODS:
        assert on[n][1] == off[n][1]  # the span's CRC itself does not depend on the knob
        assert on[n][0] == off[n][0] ^ off[n][1], n
    assert on["mix_rev"] != off["mix_rev"]  # a mix_in_h caller's fold moves with mix_wide's REV


@needs_nki
def test_set_rebinds_the_helpers_to_hardware_dge_before_any_kernel_binds_them():
    code = (
        "import inspect, json\n"
        "from trnrt.kernels import dsa_shell, kda_half, mix_wide, shell\n"
        "src = lambda f: inspect.getsource(f)\n"
        "print(json.dumps({'shell': 'hwdge' in src(shell.dma_static), 'kda_half': 'hwdge' in src(kda_half._dma),\n"
        "                  'mix_wide': 'hwdge' in src(mix_wide._dma), 'bound': dsa_shell.dma_static is shell.dma_static}))\n"
    )
    on, off = _probe("1", code), _probe(None, code)
    assert on == {"shell": True, "kda_half": True, "mix_wide": True, "bound": True}
    assert off["shell"] is False and off["kda_half"] is False and off["mix_wide"] is False and off["bound"] is True
