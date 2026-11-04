"""mk_kda_layer's chunked glue (_b5c, mk_ffn_b5c_kernel: the split KDA layer above 8 rows at grid 2), the path decode
steps of 12 rows take under --preset batch12:
- in nki.simulate (skipped without the NKI package): _b5c's rows of y = bf16((r0 + r1) + (s0 + s1)), fp32 throughout
  and one rounding, bit for bit at T 4 / 8 (one chunk), 12 (chunks of 4 and 2 rows a program) and 16 (two of 4); C = 32,
  as GLM-5.3-Flash's H 4096; at T 4 / 8, where both apply, _b5c's y equals _b5's bit for bit;
- on the host: the reference layout is the docstring's (a control for the simulator tests' reference); the served
  kernel's REV span holds none of the chunked code (it sits after that span, under its own REVC, which folds REV); and
  decode_layer_split picks the served kernel and REV at 8 rows or fewer, the chunked one and REVC only past one chunk.
"""

from __future__ import annotations

import ast
import importlib.util
import zlib

import pytest
import torch

from trnrt.kernels import mk_kda_layer as MK
from trnrt.kernels import shell as SH

NKI = importlib.util.find_spec("nki") is not None
P = 128

if NKI and MK.nki is not None:
    import nki

    @nki.jit
    def _b5_only(rp, sp):
        return MK._b5(rp, sp)

    @nki.jit
    def _b5c_only(rp, sp):
        return MK._b5c(rp, sp)


def want_y(rp: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
    """[T, H] bf16: rp fp32 [2, P, T, C] (element (p, t, c = 2 j + k) is h = k H/2 + j P + p), sp fp32 [2, T, H]."""
    _, _, T, C = rp.shape
    J = C // 2
    r = rp[0] + rp[1]  # program 0's first
    rh = r.view(P, T, J, 2).permute(1, 3, 2, 0).reshape(T, C * P)  # (t, k, j, p): h = k H/2 + j P + p
    return (rh + (sp[0] + sp[1])).to(torch.bfloat16)


def _case(T: int, C: int = 32):
    g = torch.Generator().manual_seed(500 + T)
    return torch.randn(2, P, T, C, generator=g), torch.randn(2, T, C * P, generator=g) * 0.5


def _sim(kernel, rp, sp):
    import numpy as np

    T, C = rp.shape[2], rp.shape[3]
    got = nki.simulate(kernel[2])(rp=rp.numpy(), sp=sp.numpy())
    return torch.from_numpy(np.asarray(got).astype(np.float32)).view(T, C * P).to(torch.bfloat16)


@pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("T", [4, 8, 12, 16])
def test_b5c_is_the_one_rounding_glue_bit_for_bit(T, monkeypatch):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    rp, sp = _case(T)
    got, want = _sim(_b5c_only, rp, sp), want_y(rp, sp)
    bad = (got != want).nonzero()
    assert bad.numel() == 0, (len(bad), bad[:4].tolist())


@pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("T", [4, 8])
def test_b5c_is_b5_where_both_apply(T, monkeypatch):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    rp, sp = _case(T)
    assert torch.equal(_sim(_b5c_only, rp, sp), _sim(_b5_only, rp, sp))


def test_the_reference_layout_is_the_docstrings():
    """want_y's h for (p, t, c) is k H/2 + j P + p with c = 2 j + k: one element placed by hand. Control for the
    reference the simulator tests compare against."""
    T, C = 2, 4
    rp = torch.zeros(2, P, T, C)
    rp[0, 5, 1, 3] = 1.0  # p 5, t 1, c 3: j 1, k 1 -> h = 1 * (4 * 128 / 2) + 1 * 128 + 5 = 389
    y = want_y(rp, torch.zeros(2, T, C * P))
    assert y[1, 389] == 1 and y.float().abs().sum() == 1


SRC = open(MK.__file__).read()


def test_the_served_rev_span_holds_none_of_the_chunked_code_and_revc_is_its_own():
    """The served kernel's REV covers "# -- the kernel source" to "    mk_ffn_b5_kernel = None"; the chunked code lies
    after that span (so it cannot move REV), and REVC covers its own span and folds REV."""
    a = SRC.index("# -- the kernel source")
    end = SRC.index("    mk_ffn_b5_kernel = None", a)
    c = SRC.index("# -- the chunked glue (_b5c")
    assert c > end and "_b5c" not in SRC[a:end] and "mk_ffn_b5c_kernel" not in SRC[a:end]
    own = SH.kernel_rev(MK.__file__, "# -- the chunked glue (_b5c", "# -- the chunked glue end")
    assert zlib.crc32(repr((own, MK.REV)).encode()) == MK.REVC and MK.REVC != MK.REV


def test_the_chunked_kernel_is_the_served_one_but_for_its_glue():
    """mk_ffn_b5c_kernel takes mk_ffn_b5_kernel's parameters and calls the FFN half's body with the same arguments; only
    its glue call differs (_b5c for _b5)."""

    def fn(name):
        return next(f for f in ast.walk(ast.parse(SRC)) if isinstance(f, ast.FunctionDef) and f.name == name)

    served, chunked = fn("mk_ffn_b5_kernel"), fn("mk_ffn_b5c_kernel")
    assert ast.unparse(served.args) == ast.unparse(chunked.args)
    body = [ast.unparse(s) for s in served.body if not isinstance(s, ast.Expr)]
    assert [ast.unparse(s) for s in chunked.body if not isinstance(s, ast.Expr)] == [
        s.replace("_b5(fo[3], fo[4])", "_b5c(fo[3], fo[4])") for s in body
    ]


@pytest.mark.parametrize("T,chunked", [(2, False), (4, False), (6, False), (8, False), (12, True), (16, True)])
def test_decode_layer_split_picks_the_served_kernel_at_eight_rows_or_fewer(T, chunked):
    """The dispatch rule as decode_layer_split computes it, at H 4096 and grid 2: a program's rows x H / 128 > 128
    takes the chunked kernel and REVC; T <= 8 keeps the served kernel and REV (no graph of 8 rows or fewer moves)."""
    body = SRC[SRC.index("def decode_layer_split") : SRC.index("# -- the kernel source")]
    assert "chunked = (T // npg) * (H // 128) > 128" in body
    assert "rev=REVC if chunked else REV" in body
    assert "mk_ffn_b5c_kernel if chunked else mk_ffn_b5_kernel" in body
    H, npg = 4096, 2
    assert ((T // npg) * (H // 128) > 128) == chunked
