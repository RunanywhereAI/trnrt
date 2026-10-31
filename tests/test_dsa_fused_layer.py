"""The fused DSA decode layer (kernels/dsa_satwf.py, kernels/dsa_satwfc.py): one call a DSA layer from the selection's
input to the FFN half's partials, built from the selection-attend section (kernels/dsa_sa.py), the context-parallel
tail (kernels/cc_tail.py) and the FFN half's body (kernels/ffn_half_shared.py).

On the host: every copied kernel body is its source's statement for statement (the section's sel2 and attend are
dsa_shell's served kernels' bodies; dsa_satwfc's kernel is dsa_satwf's but for its named edits), the argument mappings
between the parts are exact, and each REV carries its parts' REVs, so an edit to one part alone re-keys the call. The
pins compare code with its comments stripped: comments change no instruction. A planted change on either side breaks
each pin. In the NKI simulator (Neuron venv): the tail's heads-stacked form against its per-head form, bit for bit.
The whole call (its collectives) runs on the device only.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import re

import pytest
import torch

from trnrt.kernels import cc_tail as CT
from trnrt.kernels import cc_world as CW
from trnrt.kernels import dsa_sa as SA
from trnrt.kernels import dsa_sat as SAT
from trnrt.kernels import dsa_satwf as SATWF_M
from trnrt.kernels import dsa_satwfc as SATWFC_M
from trnrt.kernels import dsa_sel_s2 as S2
from trnrt.kernels import dsa_shell as dsh
from trnrt.kernels import ffn_half_shared as FS
from trnrt.kernels import mhc_front as MF
from trnrt.kernels import mhc_pre as MP
from trnrt.kernels import mhc_split as MS
from trnrt.kernels import shell as SH

KDIR = pathlib.Path(SA.__file__).parent
SATWF = (KDIR / "dsa_satwf.py").read_text()
SATWFC = (KDIR / "dsa_satwfc.py").read_text()
RENAME = {"scale": "fscale", "wb": "fwb"}
needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")


def _code(text: str) -> list[str]:
    """The code lines of a span: docstrings, comments and blank lines dropped, trailing whitespace stripped."""
    out = []
    for ln in re.sub(r'"""[\s\S]*?"""', "", text).splitlines():
        s = re.sub(r"\s+#.*$", "", ln).rstrip()
        if s.strip() and not s.lstrip().startswith("#"):
            out.append(s)
    return out


def _span(path, sig: str, first: str, last: str) -> str:
    src = pathlib.Path(path).read_text() if not isinstance(path, str) or "\n" not in path else path
    a = src.index(sig)
    b = src.index(first, a)
    return src[b:src.index(last, b) + len(last)]


def _fn(src: str, name: str) -> ast.FunctionDef:
    for f in ast.walk(ast.parse(src)):
        if isinstance(f, ast.FunctionDef) and f.name == name:
            return f
    raise AssertionError(f"no {name}")


# -- the section: dsa_shell's served bodies -------------------------------------------------------------------------


def test_sel2_helper_is_the_served_body():
    """dsa_sa._sel2 is dsa_shell's sel2 kernel body to and with its return, statement for statement."""
    first, last = "        Dk, BH = qT.shape\n", "        return out\n"
    served = _code(_span(dsh.__file__, "    def dsa_shell_sel2_kernel(", first, last))
    fused = _code(_span(SA.__file__, "    def _sel2(", first, last))
    assert len(served) > 80 and fused == served


def test_attend_helper_is_the_served_body():
    """dsa_sa._attend is dsa_shell's attend kernel body to and with its return (the over fallback's fold included)."""
    first, last = "        AA, T, K = ev.shape\n", "        return o, lse, srows, sbias, over\n"
    served = _code(_span(dsh.__file__, "    def dsa_shell_attend_kernel(", first, last))
    fused = _code(_span(SA.__file__, "    def _attend(", first, last))
    assert len(served) > 150 and fused == served
    assert any(ln.strip() == "if fo:" for ln in fused)  # the fold's loop is part of what is pinned


def test_section_pins_catch_a_planted_change():
    first, last = "        Dk, BH = qT.shape\n", "        return out\n"
    src = pathlib.Path(SA.__file__).read_text()
    body = _span(src, "    def _sel2(", first, last)
    lines = _code(body)
    drift = src.replace(body, body.replace(lines[5].strip(), lines[5].strip() + " + 0", 1), 1)
    assert _code(_span(drift, "    def _sel2(", first, last)) != _code(
        _span(dsh.__file__, "    def dsa_shell_sel2_kernel(", first, last))


def test_revs_carry_their_parts():
    """Each fused part's REV is its own kernel source's CRC XOR its parts' REVs: an edit to any part re-keys the
    fused call."""
    k = "# -- the kernel source"
    assert S2.REV == SH.kernel_rev(S2.__file__, k, k + " end") ^ dsh.REV2 != dsh.REV2
    assert SA.REV == SH.kernel_rev(SA.__file__, k, k + " end") ^ dsh.REV2 ^ dsh.REVA ^ S2.REV
    assert SAT.REV == SH.kernel_rev(SAT.__file__, k, k + " end") ^ SA.REV ^ CT.REV
    assert SAT.REV not in (SA.REV, CT.REV)
    assert SATWF_M.REV == SH.kernel_rev(SATWF_M.__file__, k, "    dsa_satwf_kernel = None") ^ SAT.REV ^ CW.REV ^ FS.REV
    assert SATWFC_M.REV == SH.kernel_rev(SATWFC_M.__file__, k, "    dsa_satwfc_kernel = None") ^ SATWF_M.REV ^ MS.REV_K
    assert CT.REV == (SH.kernel_rev(CT.__file__, "LP = 16", "# -- the tail end")
                      ^ SH.kernel_rev(CT.__file__, "    def _tail_heads_hs(", "# -- _tail_heads_hs end"))


def test_tail_helpers_are_cc_tails():
    """The fused kernels' tail helpers are imported from cc_tail itself, not copied."""
    assert SAT.CT is CT and CT.LP == 16 and CT.TP == 128 and CT.HS == 1
    for src in (SATWF, SATWFC):
        assert "from .cc_tail import LP, POISON_OUT, POISON_RX, TP, _dma, _tail_body, _tail_weights" in src


def test_tiles_are_permutations():
    """dsa_sat.tile_wuv / tile_o (the load-time copies the tail's DMAs read) move elements, never values."""
    g = torch.Generator().manual_seed(5)
    w_uv = torch.randn(2, 256, 512, generator=g).to(torch.bfloat16)
    o_dec = torch.randn(4096, 512, generator=g).to(torch.bfloat16)
    tw, to = SAT.tile_wuv(w_uv), SAT.tile_o(o_dec)
    assert torch.equal(tw.flatten().sort().values, w_uv.flatten().sort().values)
    assert torch.equal(to.flatten().sort().values, o_dec.flatten().sort().values)
    assert not torch.equal(to.flatten(), o_dec.flatten())  # control: the layout does move


# -- dsa_satwf: the section, the tail, the world sum and the FFN half ---------------------------------------------


def test_ffn_call_is_the_shared_body():
    """The kernel calls ffn_half_shared.ffn_half_body once, in the body's parameter order, Yp in Yp's place and the two
    renamed inputs (fscale, fwb) in scale's and wb's; it takes every other body parameter under the body's name."""
    k = _fn(SATWF, "dsa_satwf_kernel")
    body_params = [a.arg for a in _fn((KDIR / "ffn_half_shared.py").read_text(), "ffn_half_body").args.args]
    calls = [n for n in ast.walk(k) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "ffn_half_body"]
    assert len(calls) == 1
    assert [a.id for a in calls[0].args] == [RENAME.get(p, p) for p in body_params]
    kparams = {a.arg for a in k.args.args + k.args.kwonlyargs}
    assert {RENAME.get(p, p) for p in body_params if p != "Yp"} <= kparams
    assert "Yp" not in kparams  # built in the kernel


def test_mapping_matches_the_kernel():
    """satwf renames exactly scale / wb and pops Yp / rev of ffn_half.run's dict: the result is in the kernel's
    parameter set, and the section's parameters are the rest."""
    k = _fn(SATWF, "dsa_satwf_kernel")
    kparams = [a.arg for a in k.args.args]
    run_keys = [a.arg for a in _fn((KDIR / "ffn_half_pb.py").read_text(), "ffn_half_kernel").args.args]
    mapped = {RENAME.get(p, p) for p in run_keys if p not in ("Yp", "rev")}
    assert mapped <= set(kparams)
    src = ast.get_source_segment(SATWF, _fn(SATWF, "satwf"))
    for old, new in RENAME.items():
        assert f'fargs["{new}"]' in src and f'fargs.pop("{old}")' in src
    assert 'fargs.pop("Yp")' in src and 'fargs.pop("rev")' in src


# -- dsa_satwfc: dsa_satwf's kernel with the attention block's Sinkhorn inside -----------------------------------

O_OUT = "        o_out = nl.ndarray((nh, B, R), dtype=BF, buffer=nl.shared_hbm)"
AR_S2 = "        ncc.all_reduce(srcs=[s2src], dsts=[s2dst], replica_group=_cp_groups(world, A), op=nl.add)"
A2A = "                       collective_dim=0)"  # the all_to_all's last line
ISSUE = "        ar_world_issue(Y, S0, S1, Z1, world, pid, npg=npg, fence=fz)"
FFN = "        fh = ffn_half_body(S, post_p, comb_p, Yp,"
EDITS = [  # (anchor, lines inserted after it / the replacement, mode)
    (O_OUT, ["        comb_i = nl.ndarray((B, hc * hc), dtype=F32, buffer=nl.private_hbm)"], "after"),
    (AR_S2, ["        cpa = _sk_tile(comb_p, B, hc, eps, skf)",
             "        _sk_piece(cpa, comb_i, hc, iters, eps, skf, ckp, 0)"], "after"),
    (A2A, ["        _sk_piece(cpa, comb_i, hc, iters, eps, skf, ckp, 1)"], "after"),
    (ISSUE, ["        _sk_piece(cpa, comb_i, hc, iters, eps, skf, ckp, 2)"], "after"),
    (FFN, ["        fh = ffn_half_body(S, post_p, comb_i, Yp,"], "prefix"),
]


def _kbody(src: str, sig: str) -> list[str]:
    return _code(_span(src, sig, "        B = cand.shape[0]\n",
                       "        return (gv, gc, out, o, lse, srows, sbias, over, Yp, o_out) + tuple(fh)\n"))


def _pin_errors(satwf: str, satwfc: str) -> list:
    want, bad = _kbody(satwf, "    def dsa_satwf_kernel("), []
    for anchor, new, mode in EDITS:
        hits = [i for i, ln in enumerate(want) if (ln.startswith(anchor) if mode == "prefix" else ln == anchor)]
        if len(hits) != 1:
            bad.append(f"dsa_satwf: {anchor.strip()[:60]!r} {len(hits)} times")
            continue
        i = hits[0]
        if mode == "after":
            want[i + 1:i + 1] = new
        else:
            want[i] = new[0] + want[i][len(anchor):]
    got = _kbody(satwfc, "    def dsa_satwfc_kernel(")
    if got != want:
        bad.append(f"dsa_satwfc's kernel differs from dsa_satwf's beyond the named edits ({len(got)} vs {len(want)})")
    return bad


def test_kernel_is_satwfs_but_the_named_edits():
    assert _pin_errors(SATWF, SATWFC) == []
    assert len(_kbody(SATWFC, "    def dsa_satwfc_kernel(")) > 120  # the whole body, not an empty match


def test_arguments_are_satwfs_plus_ckp():
    a, b = _fn(SATWF, "dsa_satwf_kernel").args, _fn(SATWFC, "dsa_satwfc_kernel").args
    assert [x.arg for x in b.args] == [x.arg for x in a.args] + ["ckp"]
    assert [ast.unparse(d) for d in b.defaults] == [ast.unparse(d) for d in a.defaults] + ["3"]
    assert ([ast.unparse(x.annotation) for x in b.args if x.annotation is not None]
            == [ast.unparse(x.annotation) for x in a.args if x.annotation is not None] + ["int"])


@pytest.mark.parametrize("plant", ["satwf-drift", "satwfc-drift", "edit-dropped"])
def test_a_planted_change_breaks_the_pin(plant):
    if plant == "satwf-drift":  # the served kernel changes; the copy does not follow
        old = "        _rows_i32(out, B, keep, RS)\n"
        assert SATWF.count(old) == 1
        assert _pin_errors(SATWF.replace(old, ""), SATWFC)
    elif plant == "satwfc-drift":  # the copy drifts beyond the named edits
        old = "        split = npg == 2 and B >= 2\n"
        assert SATWFC.count(old) == 1
        assert _pin_errors(SATWF, SATWFC.replace(old, "        split = npg == 2 and B >= 3\n"))
    else:  # mix_in reads comb_p again: the Sinkhorn's result unused
        new = "        fh = ffn_half_body(S, post_p, comb_i, Yp,"
        assert SATWFC.count(new) == 1
        assert _pin_errors(SATWF, SATWFC.replace(new, FFN))


def _sk_piece():
    """dsa_satwfc._sk_piece, compiled from its source with a stand-in sinkhorn_body that records (lo, hi)."""
    calls = []
    f = _fn(SATWFC, "_sk_piece")

    def body(cp, comb, hc, iters, eps, skf, lo, hi):
        calls.append((lo, hi))

    g = {"MS": type("MS", (), {"sinkhorn_body": staticmethod(body)})}
    exec(compile(ast.Module(body=[f], type_ignores=[]), "dsa_satwfc", "exec"), g)
    return g["_sk_piece"], calls


@pytest.mark.parametrize("ckp", [1, 2, 3])
@pytest.mark.parametrize("iters", [1, 2, 20])
def test_pieces_partition_the_half_rounds(ckp, iters):
    """_sk_piece's pieces partition the 2 iters - 1 half-rounds for every ckp, and exactly one reaches the end (writes
    comb)."""
    piece, calls = _sk_piece()
    for j in range(3):  # the kernel's three issue points
        piece(None, None, 4, iters, 1e-6, 1, ckp, j)
    n = 2 * iters - 1
    assert calls[0][0] == 0 and calls[-1][1] == n and len(calls) <= ckp
    assert all(hi > lo for lo, hi in calls) and all(calls[i][1] == calls[i + 1][0] for i in range(len(calls) - 1))
    assert [hi for _, hi in calls].count(n) == 1
    if ckp <= n:  # (chunk_bounds refuses more pieces than half-rounds; the kernel's extra pieces are empty)
        assert calls == MS.chunk_bounds(iters, ckp)


def test_front_then_sinkhorn_is_mhc_pre_on_the_host():
    """The front's c through the host Sinkhorn is mhc_pre's comb, and its x and post are mhc_pre's, bit for bit."""
    g = torch.Generator().manual_seed(0)
    T, hc, H = 3, 4, 256
    S = torch.randn(T, hc, H, generator=g).bfloat16()
    fn = (0.05 * torch.randn(2 * hc + hc * hc, hc * H, generator=g)).bfloat16()
    base, scale = 0.1 * torch.randn(2 * hc + hc * hc, generator=g), 1 + 0.1 * torch.randn(3, generator=g)
    x, post, c = MF.emulate(S, fn, base, scale, 20, 1e-6, 1e-6)
    xr, postr, combr = MP.emulate(S, fn, base, scale, 20, 1e-6, 1e-6)
    assert torch.equal(x, xr) and torch.equal(post, postr) and torch.equal(MF.sinkhorn(c, 20, 1e-6), combr)
    assert not torch.equal(c, combr)  # c is the Sinkhorn's input, not comb


def test_front_wraps_c():
    S = torch.randn(2, 4, 128).bfloat16()
    fn = (0.05 * torch.randn(24, 512)).bfloat16()
    x, post, c = MF.front(S, fn, torch.zeros(24), torch.ones(3), 20, 1e-6, 1e-6)
    assert isinstance(c, MF.PendingComb) and tuple(c.c.shape) == (2, 4, 4)
    with pytest.raises(TypeError):
        c.c + c  # a PendingComb is no tensor


def test_satwf_dispatches_by_comb_type(monkeypatch):
    """dsa_satwf.satwf hands a PendingComb (the served attention block's comb) to dsa_satwfc's kernel, and anything else
    to the plain fused call."""
    monkeypatch.setattr(SATWFC_M, "satwf", lambda *a, comb_a, **k: "fused")
    monkeypatch.setattr(SATWF_M, "_satwf_served", lambda *a, comb_a, **k: "served")
    assert SATWF_M.satwf(comb_a=MF.PendingComb(torch.zeros(1))) == "fused"
    assert SATWF_M.satwf(comb_a=torch.zeros(1)) == "served"


def test_comb_done_recomputes_only_a_pending_comb(monkeypatch):
    """Where the fused call does not take, hybrid._comb_done replaces a PendingComb by mhc_pre's comb; other combs
    pass."""
    from trnrt.models import hybrid as HY

    monkeypatch.setattr(HY, "_mhc", lambda model, layer, b, S, decode=False: (None, None, ("served", b, decode)))
    t = torch.zeros(1)
    assert HY._comb_done(None, None, t, True, MF.PendingComb(t)) == ("served", "attn", True)
    assert HY._comb_done(None, None, t, True, t) is t


# -- cc_tail: the heads stacked on partitions ------------------------------------------------------------------------


if CT.nki is not None:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def _heads_kernel(rs, w_uv_p, ident, T: int, A: int, nh: int, r: int, dv: int, so: int, wu: int, nf: int, hs: int):
        """_tail_heads on rs's rows (hs keys the trace: CT.HS is set by the caller), loading W_UV and the identity as
        _tail_body's callers do; returns uvT, rowok and o_out."""
        f32, bf16 = nl.float32, nl.bfloat16
        RT, FT = r // CT.TP, nh * dv // CT.TP
        wuv_all = nl.ndarray((CT.TP, nh * RT, dv), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wuv_all, src=w_uv_p)
        idt = nl.ndarray((T, T), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=idt, src=ident)
        uvT = nl.ndarray((CT.TP, FT, T), dtype=bf16, buffer=nl.sbuf)
        rowok = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=rowok, value=1.0)
        o_out = nl.ndarray((nh, T, r), dtype=bf16, buffer=nl.shared_hbm)
        CT._tail_heads(rs, wuv_all, uvT, rowok, idt, o_out, T, A, nh, r, dv, so, wu, nf, 1)
        uv_o = nl.ndarray((CT.TP, FT, T), dtype=bf16, buffer=nl.shared_hbm)
        rk_o = nl.ndarray((T, 1), dtype=f32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=uv_o, src=uvT)
        nisa.dma_copy(dst=rk_o, src=rowok)
        return uv_o, rk_o, o_out


def _tail_case(T: int, A: int, nh: int, r: int, dv: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    Cn = r + CT.LP
    rs = torch.zeros(A * nh * T, Cn)
    rs[:, :r] = torch.randn(A * nh * T, r, generator=g)
    rs[:, r] = torch.randn(A * nh * T, generator=g) * 3.0  # the lse column
    rs[(A - 1) * nh * T + T + 1, 5] = float("nan")  # head 1, row 1, last source: rowok 0 at row 1 (nf's screen)
    w = (torch.randn(CT.TP, nh * (r // CT.TP), dv, generator=g) * 0.05).to(torch.bfloat16)
    return rs.float().contiguous(), w.contiguous(), torch.eye(T, dtype=torch.bfloat16)


def _bits(a, b) -> bool:
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    if a.dtype == torch.bfloat16:
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    return torch.equal(a.float().view(torch.int32), b.float().view(torch.int32))


@needs_nki
@pytest.mark.parametrize("T", [4, 8, 16])
@pytest.mark.parametrize("so,wu", [(1, 1), (1, 0), (0, 0)])
def test_heads_stacked_bitwise(monkeypatch, T, so, wu):
    """_tail_heads_hs (the served form) == the per-head form on uvT, rowok and o_out, bit for bit (A 8 sources, 2 heads,
    r 512, dv 256)."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    A, nh, r, dv = 8, 2, 512, 256
    rs, w, ident = _tail_case(T, A, nh, r, dv, seed=31 + T)
    kw = dict(T=T, A=A, nh=nh, r=r, dv=dv, so=so, wu=wu, nf=1)
    monkeypatch.setattr(CT, "HS", 0)
    a = nki.simulate(_heads_kernel)(rs, w, ident, hs=0, **kw)
    monkeypatch.setattr(CT, "HS", 1)
    b = nki.simulate(_heads_kernel)(rs, w, ident, hs=1, **kw)
    for x, y in zip(a, b):
        assert _bits(x, y)
    assert float(torch.as_tensor(a[1])[1, 0]) == 0.0 and float(torch.as_tensor(a[1])[0, 0]) == 1.0  # nf's screen bit


def _no_heads(*args, **kwargs):
    """A stand-in that writes nothing (a named function: the tracer reads called functions' source)."""
    return None


@needs_nki
def test_heads_stacked_dispatch_control(monkeypatch):
    """The positive control: with _tail_heads_hs replaced by a function that writes nothing, HS 1's o_out differs from
    HS 0's (so the bitwise test above did run the stacked form)."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    T, A, nh, r, dv = 8, 8, 2, 512, 256
    rs, w, ident = _tail_case(T, A, nh, r, dv, seed=5)
    kw = dict(T=T, A=A, nh=nh, r=r, dv=dv, so=1, wu=1, nf=1)
    monkeypatch.setattr(CT, "HS", 0)
    a = nki.simulate(_heads_kernel)(rs, w, ident, hs=0, **kw)
    monkeypatch.setattr(CT, "HS", 1)
    monkeypatch.setattr(CT, "_tail_heads_hs", _no_heads)
    b = nki.simulate(_heads_kernel)(rs, w, ident, hs=2, **kw)
    assert not _bits(a[2], b[2]), "HS 1 did not reach _tail_heads_hs"
