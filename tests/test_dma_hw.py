"""TRNRT_DMA_HW: the static DMA helpers (shell.dma_static, kda_half._dma, mix_wide._dma, cc_tail._dma) in hardware DGE,
read at import.

Unset (the default presets), each module's REV is its hashed span's CRC exactly as before. Set (every MTP preset, as in every build
behind the published MTP numbers), REV also folds the hardware-DGE span's CRC, so every kernel folding REV re-keys, and the helper
a kernel binds by name at import (dsa_shell's `from .shell import dma_static`, dsa_sat's `from .cc_tail import _dma`) is the
hardware-DGE form. Each case imports the modules in a fresh interpreter, since the knob is read once at import."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODS = ("shell", "kda_half", "mix_wide", "cc_tail")
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
    "from trnrt.kernels import cc_tail, kda_half, mix_wide, shell\n"
    "m = {'shell': shell, 'kda_half': kda_half, 'mix_wide': mix_wide, 'cc_tail': cc_tail}\n"
    "from trnrt.kernels import dsa_sat\n"
    "print(json.dumps({n: [x.REV, x._dma_hw_rev()] for n, x in m.items()}\n"
    "                 | {'mix_rev': mix_wide.MIX_REV, 'sat_rev': dsa_sat.REV, 'sat_rev_k': dsa_sat.REV_K}))\n"
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
    # cc_tail's callers (the J4 kernel and its section) fold cc_tail.REV
    assert on["sat_rev"] != off["sat_rev"]
    assert on["sat_rev_k"] != off["sat_rev_k"]


@needs_nki
def test_set_rebinds_the_helpers_to_hardware_dge_before_any_kernel_binds_them():
    code = (
        "import inspect, json\n"
        "from trnrt.kernels import cc_tail, dsa_sat, dsa_satwf, dsa_shell, kda_half, mix_wide, shell\n"
        "src = lambda f: inspect.getsource(f)\n"
        "print(json.dumps({'shell': 'hwdge' in src(shell.dma_static), 'kda_half': 'hwdge' in src(kda_half._dma),\n"
        "                  'mix_wide': 'hwdge' in src(mix_wide._dma), 'cc_tail': 'hwdge' in src(cc_tail._dma),\n"
        "                  'bound': dsa_shell.dma_static is shell.dma_static,\n"
        "                  'bound_tail': dsa_sat._dma is cc_tail._dma and dsa_satwf._dma is cc_tail._dma}))\n"
    )
    on, off = _probe("1", code), _probe(None, code)
    bound = {"bound": True, "bound_tail": True}  # every binder holds its module's helper, set or not
    assert on == {"shell": True, "kda_half": True, "mix_wide": True, "cc_tail": True} | bound
    assert off == {"shell": False, "kda_half": False, "mix_wide": False, "cc_tail": False} | bound
