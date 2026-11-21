"""The fused DSA layer's merge selection decides 2 bits a round (kernels/dsa_sat.py sat_args' rbits): every fused call
(dsa_satwf, its copies and the plain MTP layer's dsa_sat.sat) passes none, so the default is the served value. The
selection is the same set at any width (tests/test_dsa_topk.py, tests/test_dsa_tieskip.py run widths 2 and 4)."""

from __future__ import annotations

import inspect

from trnrt.kernels import dsa_sat


def test_sat_args_and_sat_default_to_two_bits():
    assert inspect.signature(dsa_sat.sat_args).parameters["rbits"].default == 2
    assert inspect.signature(dsa_sat.sat).parameters["rbits"].default == 2


def test_no_fused_caller_passes_its_own_rbits():
    """models/mla.py's fused calls leave rbits to sat_args (a caller passing 4 would silently take the wider rounds)."""
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "trnrt" / "models" / "mla.py").read_text()
    for call in ("dsa_satwf.satwf(", "dsa_sat.sat("):
        a = src.index(call)
        body = src[a : src.index("\n\n", a)]
        assert "rbits=" not in body.replace("rbits2=", ""), call
