"""kernels/ffn_half_pb.py's two exact reorders of the FFN half, served (IDS1 and XC1 True), on the host by the source:
- IDS1: one early block, right after the id clamp, whose e_sb statements are the late block's verbatim and whose prologue
  loads are the loop's load body verbatim for groups 0 .. min(PD, NL) - 1; the loop then loads PD .. NL - 1 with the
  same body; together they issue every group's loads once (checked by enumeration);
- XC1: _norm_xc's k-major branch runs the h-layout ops on xk (the multiply by r, the optional bf16 rounding, the weight
  product, the row isolation), with post_norm read k-major, and builds xc with the h-layout path's copy pattern;
- the ring's loads are hardware-DGE gathers (GATHER_HW) in both places.
The simulator's byte-equality against the forms without them, and the trace control (XC1 drops one masked expansion's T
start-only matmuls), need the NKI package (the last test here)."""

from __future__ import annotations

import ast
import importlib.util
import pathlib

import pytest

from trnrt.kernels import ffn_half_pb as FB

SRC = pathlib.Path(FB.__file__).read_text()
NKI = importlib.util.find_spec("nki") is not None
needs_nki = pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")


def _fn(name: str) -> ast.FunctionDef:
    return next(f for f in ast.walk(ast.parse(SRC)) if isinstance(f, ast.FunctionDef) and f.name == name)


def _ifs(f, test: str) -> list[ast.If]:
    return [s for s in ast.walk(f) if isinstance(s, ast.If) and ast.unparse(s.test) == test]


def test_the_served_forms():
    assert FB.IDS1 is True and FB.XC1 is True and FB.GATHER_HW is True
    assert FB.SS_XLA  # XC1 reuses that expansion


def _late_esb(f) -> list[str]:
    """The late e_sb statements: mi's (in its `not IDS1` guard) and ones_c .. e_sb's copy (in the other)."""
    gs = [g for g in _ifs(f, "not IDS1") if not any(isinstance(s, ast.If) for s in g.body)]
    assert len(gs) == 3  # mi, ones_c .. e_sb, bufs
    return [ast.unparse(s) for g in gs[:2] for s in g.body]


def test_ids1_block_is_the_late_statements_moved():
    f = _fn("half_body_pb")
    early = _ifs(f, "IDS1")
    assert len(early) == 1 and not early[0].orelse
    body = early[0].body
    seq = [ast.unparse(s) for s in f.body]  # placed right after the id clamp, before tv
    i = seq.index(ast.unparse(early[0]))
    assert seq[i - 1].startswith("nisa.tensor_scalar(dst=i8f, data=i8f, op0=nl.maximum")
    assert seq[i + 1].startswith("tv = nl.ndarray((T, KTOP)")
    got = [ast.unparse(s) for s in body]
    assert got[:9] == _late_esb(f)  # mi, ones_c, pe, e_sb: verbatim
    late_bufs = [g for g in _ifs(f, "not IDS1") if not any(isinstance(s, ast.If) for s in g.body)][2]
    assert ast.unparse(ast.parse(got[9]).body[0].value) == ast.unparse(late_bufs.body[0].value)  # the five buffers[:NR]
    loop = body[10]
    assert (
        isinstance(loop, ast.For)
        and ast.unparse(loop.iter) == "range(min(PD, NL))"
        and ast.unparse(loop.target) == "gl"
    )
    guard = next(g for g in _ifs(f, "not IDS1") if any(isinstance(s, ast.If) for s in g.body))
    late_load = guard.body[0]
    assert ast.unparse(late_load.test) == "0 <= gl < NL"
    elif_ = guard.orelse[0]
    assert ast.unparse(elif_.test) == "PD <= gl < NL"

    def norm(stmts):  # comments are not in the AST; compare statements
        return [ast.unparse(s) for s in stmts]

    assert norm(loop.body) == norm(late_load.body) == norm(elif_.body)
    assert "dge_mode=nisa.dge_mode.hwdge" in ast.unparse(loop)


@pytest.mark.parametrize(("NL", "PD"), [(1, 1), (2, 1), (12, 1), (12, 3), (2, 3), (24, 3)])
def test_ids1_issues_every_group_once(NL, PD):
    late = [it + PD for it in range(-PD, NL + 1) if 0 <= it + PD < NL]
    ids1 = list(range(min(PD, NL))) + [it + PD for it in range(-PD, NL + 1) if PD <= it + PD < NL]
    assert late == ids1 == list(range(NL))


def test_xc1_branch_mirrors_the_h_layout_ops():
    f = _fn("_norm_xc")
    br = _ifs(f, "XC1")
    assert len(br) == 2  # the k-major post_norm load (first), the k-major ops (after r)
    first, ops = br
    assert "pnorm.ap(pattern=[[1, P], [P, C]], offset=0)" in ast.unparse(first.body[1])
    assert [ast.unparse(s) for s in first.orelse] == [
        "xf = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)",
        "nisa.tensor_copy(dst=xf, src=xb, engine=VE)",
    ]
    k = [ast.unparse(s) for s in ops.body]
    assert k[0] == (
        "nisa.tensor_tensor(dst=xk, data1=xk, data2=r.ap(pattern=[[T, P], [1, T], [0, C]], offset=0), "
        "op=nl.multiply, engine=VE)"
    )
    src = ast.unparse(f)
    # the h-layout multiply is the same op with W for C
    assert (
        "nisa.tensor_tensor(dst=xf, data1=xf, data2=r.ap(pattern=[[T, P], [1, T], [0, W]], offset=0), op=nl.multiply, "
        "engine=VE)"
    ) in src
    assert any("data1=wk.ap(pattern=[[C, P], [0, T], [1, C]], offset=0), data2=xqk, op=nl.multiply" in s for s in k)
    assert any("RI.finite_or_zero3(xnk.view(u16), x16k, fzk)" in ast.unparse(s) for s in ast.walk(ops))
    # xc: the h-layout path's copy pattern, from SBUF
    assert (
        sum("nisa.tensor_copy(dst=xc.ap(pattern=[[C * T, P], [1, T], [T, C]], offset=0)" in s for s in src.splitlines())
        == 2
    )
    assert k[-1] == "return (xnk, xc)"


@needs_nki
@pytest.mark.parametrize("T", [16, 24])
def test_xc1_traces_one_expansion(monkeypatch, T):
    """The positive control: XC1 traces T fewer start-only matmuls (one masked expansion's)."""
    import collections
    import re

    import ml_dtypes
    import nki as _nki
    import nki.isa as nisa
    import nki.language as nl
    import numpy as np
    from nki.compiler.frontend import FrontendKind, source_to_mlir

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    H = 4096

    @_nki.jit
    def norm_probe(x, pnorm):
        xb = nl.ndarray((128, T, H // 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xb, src=x)
        _xn, xc = FB._norm_xc(xb, pnorm, 1e-5, 2, T, H)
        out = nl.ndarray((128, H // 128, T), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=xc)
        return out

    def ops(on: bool):
        monkeypatch.setattr(FB, "XC1", on)
        m = str(
            source_to_mlir(
                norm_probe,
                x=np.zeros((128, T, H // 128), ml_dtypes.bfloat16),
                pnorm=np.zeros((H,), ml_dtypes.bfloat16),
                target="trn2",
                frontend=FrontendKind.TRACER,
            )
        )
        return collections.Counter(re.findall(r"nisa\.[a-z_0-9]+", m))

    off, on = ops(False), ops(True)
    mm = [k for k in off if "matmul" in k]
    assert mm and sum(off[k] for k in mm) - sum(on[k] for k in mm) == T
