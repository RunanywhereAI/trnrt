"""The fused DSA layer's merge selection decides 2 bits a round (kernels/dsa_sat.py sat_args' rbits): every fused call
(dsa_satwf, its copies and the plain MTP layer's dsa_sat.sat) passes none, so the default is the served value. The
selection is the same set at any width (tests/test_dsa_topk.py, tests/test_dsa_tieskip.py run widths 2 and 4)."""

from __future__ import annotations

import ast
import inspect
import pathlib

from trnrt.kernels import dsa_sat

TRNRT = pathlib.Path(__file__).resolve().parents[1] / "trnrt"
FUSED = {("dsa_satwf", "satwf"), ("dsa_sat", "sat")}  # (module, function) of the fused calls


def test_sat_args_and_sat_default_to_two_bits():
    assert inspect.signature(dsa_sat.sat_args).parameters["rbits"].default == 2
    assert inspect.signature(dsa_sat.sat).parameters["rbits"].default == 2
    # keyword-only in both, so a positional argument cannot reach it (satwf hands its *args to sat_args)
    assert inspect.signature(dsa_sat.sat_args).parameters["rbits"].kind is inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(dsa_sat.sat).parameters["rbits"].kind is inspect.Parameter.KEYWORD_ONLY


def _dicts(e: ast.expr) -> list[ast.Dict] | None:
    """The dict literals a `**` argument can be (a literal, or a conditional of literals), or None when it is anything
    else, whose keys cannot be read from the source."""
    if isinstance(e, ast.Dict):
        return [e]
    if isinstance(e, ast.IfExp):
        a, b = _dicts(e.body), _dicts(e.orelse)
        return None if a is None or b is None else a + b
    return None


def fused_calls(src: str) -> tuple[int, list[str]]:
    """(how many fused calls `src` makes, the ones that pass rbits or might): every call, wherever it sits, each keyword
    by its parsed name, and a `**` mapping by its literal keys (one that is not a literal counts as passing it)."""
    n, bad = 0, []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (f.value.id, f.attr) in FUSED):
            continue
        n += 1
        for kw in node.keywords:
            if kw.arg == "rbits":
                bad.append(f"line {node.lineno}: rbits=")
            elif kw.arg is None:
                ds = _dicts(kw.value)
                keys = None if ds is None else [k for d in ds for k in d.keys]
                if keys is None or any(not isinstance(k, ast.Constant) for k in keys):
                    bad.append(f"line {node.lineno}: a ** mapping whose keys are not literals")
                elif any(isinstance(k, ast.Constant) and k.value == "rbits" for k in keys):
                    bad.append(f"line {node.lineno}: **{{'rbits': ...}}")
    return n, bad


def test_no_fused_caller_passes_its_own_rbits():
    """Every fused call outside the kernels (models/mla.py's today) leaves rbits to sat_args: a caller passing 4 would
    silently take the wider rounds."""
    found = 0
    for p in sorted(TRNRT.rglob("*.py")):
        if p.parent.name == "kernels":
            continue  # the kernels' own wrappers hand their keywords on unchanged
        n, bad = fused_calls(p.read_text())
        assert not bad, (str(p.relative_to(TRNRT)), bad)
        found += n
    assert found >= 2, "no fused call found: the detector reads nothing"  # mla.py's satwf and sat calls


def test_the_detector_finds_every_spelling_and_every_call():
    """Planted sources: the check must catch rbits however it is spelt and in any call, not just the first."""
    ok = "a = dsa_satwf.satwf(q, w, kp=4, **({'pq': 1} if c else {}))\nb = dsa_sat.sat(q, w, nr=2)\n"
    assert fused_calls(ok) == (2, [])
    for plant in (
        "dsa_sat.sat(q, w, rbits = 4)",
        "dsa_sat.sat(q, w,\n             rbits=4)",
        "dsa_satwf.satwf(q, **{'rbits': 4})",
        "dsa_satwf.satwf(q, **({'pq': 1} if c else {'rbits': 4}))",
        "dsa_satwf.satwf(q, **kw)",
        "def f():\n    return [dsa_satwf.satwf(q, rbits=4)]",
    ):
        n, bad = fused_calls(ok + plant + "\n")
        assert n == 3 and len(bad) == 1, (plant, n, bad)
    assert fused_calls("dsa_sat.sat(q, rbits2=4)\n") == (1, [])  # the first stage's own width is another argument
