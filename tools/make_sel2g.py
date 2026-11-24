"""Generate trnrt/kernels/dsa_sel2g.py: dsa_shell's sel2 (dsa_shell_sel2_kernel's body, _compact2, _compact2s) with its
compaction generalised to a score tile of any width NT = NG PPP <= 512, a multiple of 8, so a 3,072-page decode bucket
(NT 192) builds. sel2's compaction takes NT // 128 chunks of 128 columns, so at NT 192 it would drop columns 128-191
(dsa_sa._sel2's assert refuses it); here it takes ceil(NT / 128) chunks, the last one's transposes and prefix matmuls on
its own width (rows 0 .. w - 1). At NT a multiple of 128 every chunk is 128 wide: the same ops as sel2's, so the same bits.
A copy, not an edit: dsa_shell's REV2 (every decode graph's key) stays where it is. tests/test_dsa_sel2g_generated.py
checks the file is this transform of the current source.
    python tools/make_sel2g.py [--check]"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "trnrt", "kernels", "dsa_shell.py")
DST = os.path.join(HERE, "trnrt", "kernels", "dsa_sel2g.py")


def _slice(src: str, begin: str, end: str) -> str:
    a = src.index(begin)
    return src[a : src.index(end, a)]


def _once(s: str, a: str, b: str) -> str:
    if s.count(a) != 1:
        raise ValueError(f"expected once: {a[:80]!r} ({s.count(a)})")
    return s.replace(a, b)


def transform(src: str) -> str:
    c2 = _slice(src, "    def _compact2(X, ii, bb):\n", "    _C2S_G = 4")
    c2s = _slice(src, "    def _compact2s(X, ii, bb):\n", "    @nki.jit\n    def dsa_shell_sel2_kernel(")
    body = _slice(src, "    def dsa_shell_sel2_kernel(", "\nelse:\n    dsa_shell_sel2_kernel = None")
    # _compact2: the transposes and the prefix matmuls on each chunk's own width
    c2 = _once(c2, "    def _compact2(X, ii, bb):\n", "    def _compact2g(X, ii, bb):\n")
    c2 = _once(
        c2,
        "nisa.nc_matmul(dst=ptr[r][0:cw, j * 128:(j + 1) * 128], stationary=rs[r][:, t, j * cw:(j + 1) * cw],",
        "nisa.nc_matmul(dst=ptr[r][0:_cwj(NT, j), j * 128:(j + 1) * 128],\n"
        "                                   stationary=rs[r][:, t, j * 128:j * 128 + _cwj(NT, j)],",
    )
    c2 = _once(
        c2,
        "nisa.nc_matmul(dst=prc[r], stationary=rts[r][0:cw, t, j * 128:(j + 1) * 128],\n"
        '                                       moving=X["UT"][0:cw, j * NT:(j + 1) * NT], accumulate=(j > 0))',
        "nisa.nc_matmul(dst=prc[r], stationary=rts[r][0:_cwj(NT, j), t, j * 128:(j + 1) * 128],\n"
        '                                       moving=X["UT"][0:_cwj(NT, j), j * NT:(j + 1) * NT], accumulate=(j > 0))',
    )
    # _compact2s: the same two
    c2s = _once(c2s, "    def _compact2s(X, ii, bb):\n", "    def _compact2sg(X, ii, bb):\n")
    c2s = _once(
        c2s,
        "nisa.nc_matmul(dst=ptr[i][0:cw, u, c * 128:(c + 1) * 128],\n"
        '                                       stationary=rs[q][:, c0 + u, c * cw:(c + 1) * cw], moving=X["IB"],',
        "nisa.nc_matmul(dst=ptr[i][0:_cwj(NT, c), u, c * 128:(c + 1) * 128],\n"
        "                                       stationary=rs[q][:, c0 + u, c * 128:c * 128 + _cwj(NT, c)],\n"
        '                                       moving=X["IB"],',
    )
    c2s = _once(
        c2s,
        "nisa.nc_matmul(dst=p_, stationary=rts[q][0:cw, j * KT + t, c * 128:(c + 1) * 128],\n"
        '                                           moving=X["UT"][0:cw, c * NT:(c + 1) * NT], accumulate=(c > 0))',
        "nisa.nc_matmul(dst=p_, stationary=rts[q][0:_cwj(NT, c), j * KT + t, c * 128:(c + 1) * 128],\n"
        '                                           moving=X["UT"][0:_cwj(NT, c), c * NT:(c + 1) * NT],\n'
        "                                           accumulate=(c > 0))",
    )
    # the body: a plain function (the kernel below calls it), the width assert relaxed, ceil chunks, the plant
    body = _once(
        body,
        "    def dsa_shell_sel2_kernel(qT, wb, pkc, ppg, cand, identb, rank, keep: int, A: int, nbits: int,"
        " rbits: int,\n                                   rev: int, dge: int = 0, nr: int = 4, qr: int = 4, pf:"
        " int = 2, cs: int = 0):",
        "    def _sel2g(qT, wb, pkc, ppg, cand, identb, rank, keep: int, A: int, nbits: int, rbits: int, rev: int,\n"
        "               dge: int = 0, nr: int = 4, qr: int = 4, pf: int = 2, cs: int = 0, dbg: int = 0):",
    )
    body = _once(
        body,
        "assert keep < P and NT <= 512 and (NT <= 128 or NT % 128 == 0) and rbits > 1",
        "assert keep < P and NT <= 512 and NT % 8 == 0 and rbits > 1",
    )
    body = _once(
        body,
        "            nch = NT // 128 if NT >= 128 else 1\n",
        "            nch = -(-NT // 128) if NT >= 128 else 1  # ceil: the last chunk _cwj(NT, nch - 1) wide\n"
        "            if dbg == 1:  # the planted fault (tests only): sel2's chunk count, the partial chunk dropped\n"
        "                nch = NT // 128\n",
    )
    body = _once(body, "                _compact2s(XC, ii, rows)", "                _compact2sg(XC, ii, rows)")
    body = _once(
        body,
        "                _compact2(XC, ii, rows)  # the program's rows interleaved",
        "                _compact2g(XC, ii, rows)  # the program's rows interleaved",
    )
    head = '''"""GENERATED by tools/make_sel2g.py from dsa_shell.py: sel2 (dsa_shell_sel2_kernel's body, _compact2, _compact2s) with
the compaction generalised to any score-tile width NT <= 512 (a multiple of 8): ceil(NT / 128) chunks, the last on its own
width, so a 3,072-page bucket (NT 192) builds. At NT a multiple of 128, sel2's ops. Do not edit; regenerate."""

from __future__ import annotations

import zlib

import torch

from . import dsa_shell as dsh
from . import shell as SH

try:  # the Neuron venv
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

if nki is not None:  # sel2g's kernel: its own module and revision (REVG), so no other call's key moves
    from .dsa_shell import _C2S_G, BF16, D, F32, I32, LPP, NEG_INF, PPP, VE, _c2s_waves, _ix_gather, _ix_row_setup
    from .dsa_shell import _ix_stage_a, _ix_stage_b, _sb, _select2s, _split3, _topk_consts, dma_static

    def _cwj(NT: int, j: int) -> int:
        """The width of the compaction's chunk j of a [128, NT] tile: 128, the last one NT - 128 j."""
        if NT - 128 * j < 128:
            return NT - 128 * j
        return 128

'''
    tail = '''
    @nki.jit
    def dsa_sel2g_kernel(qT, wb, pkc, ppg, cand, identb, rank, keep: int, A: int, nbits: int, rbits: int, rev: int,
                              dge: int = 0, nr: int = 4, qr: int = 4, pf: int = 2, cs: int = 0, dbg: int = 0):
        """sel2's arguments (rev: REVG; dbg 1 the planted fault, tests only). Returns fp32 [3, B, keep] (sel2's)."""
        return _sel2g(qT, wb, pkc, ppg, cand, identb, rank, keep, A, nbits, rbits, rev, dge, nr, qr, pf, cs, dbg)
else:
    dsa_sel2g_kernel = None


REVG = (SH.kernel_rev(__file__, "if nki is not None:  # sel2g's kernel", "    dsa_sel2g_kernel = None") ^ dsh.REV2
        ^ zlib.crc32(b"sel2g"))


def takes(table: torch.Tensor) -> bool:
    """Whether a decode call over `table` [rows, NPg] needs sel2g (dsa_satwf's SEL2G dispatch): whole 128-page
    groups whose score tile (NT = 8 NG) is past one chunk and not whole 128-column chunks (P3072: NT 192), within 512."""
    npg = table.shape[-1]
    nt = (npg // 128) * dsh.PPP
    return npg % 128 == 0 and 128 < nt <= 512 and nt % 128 != 0


def select2g(q: torch.Tensor, w: torch.Tensor, pkc: torch.Tensor, table: torch.Tensor, nloc: torch.Tensor, keep: int,
             A: int, rank: torch.Tensor, *, rbits: int = 2, dge: int = 0, simulate: bool = False, grid: int | None = None,
             nr: int | None = None, pf: int = 2, cs: int = 0, dbg: int = 0):
    """dsa_shell.select2's call and outputs (lval, cpool, rows) at any page bucket of whole 128-page groups (NT = 8 NG <=
    512): dsa_sel2g_kernel on a device or in the simulator, emulate() on the host (the same bits)."""
    from . import dsa_index

    if q.device.type == "cpu" and not simulate and not SH.HOST_SIM:
        lval, cpool, rows, _ = dsh.emulate(q, w, pkc, table, nloc, keep, A, int(rank.reshape(())))
        return lval, cpool, rows
    if dsa_sel2g_kernel is None:
        raise RuntimeError("the NKI DSA selection needs the nki package (the Neuron venv)")
    if cs not in (0, 1) or dbg not in (0, 1):
        raise ValueError(f"select2g: cs={cs}, dbg={dbg}: 0 or 1")
    if grid is None:
        from .. import platform

        grid = 1 if simulate else platform.nki_grid()
    B, Hi, _ = q.shape
    NPg = table.shape[1]
    P = NPg * dsh.PPP
    cand = torch.where(torch.arange(P, device=q.device).view(1, P) < nloc.view(-1, 1), 0.0, dsh.NEG_INF)
    args = dict(qT=q.to(torch.bfloat16).reshape(B * Hi, dsh.D).t().contiguous(),
                wb=w.float().repeat(1, 4).unsqueeze(1).expand(B, 128, 4 * Hi).contiguous(),
                pkc=pkc.to(torch.bfloat16), ppg=dsa_index.page_groups(table), cand=cand.float(),
                identb=torch.eye(128, device=q.device).to(torch.bfloat16), rank=rank.to(torch.float32).reshape(1, 1),
                keep=int(keep), A=int(A), nbits=int(P.bit_length()), rbits=int(rbits), rev=REVG, dge=int(dge),
                nr=dsa_index.NR if nr is None else int(nr), qr=dsa_index.QR, pf=int(pf), cs=int(cs), dbg=int(dbg))
    k = dsa_sel2g_kernel
    if simulate:
        import os

        import ml_dtypes
        import numpy as np

        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"  # an LNC=2 (trn2) kernel

        def to_np(t):
            if not isinstance(t, torch.Tensor):
                return t
            return t.float().numpy().astype(ml_dtypes.bfloat16) if t.dtype == torch.bfloat16 else t.numpy()

        o = nki.simulate(k[grid] if grid > 1 else k)(**{n: to_np(t) for n, t in args.items()})
        o = torch.from_numpy(np.asarray(o, dtype=np.float32))
        return o[0], o[1], o[2].to(torch.int32)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    o = wrap_nki(k)[grid](**args)  # one output: wrap_nki returns a bare tensor
    return o[0], o[1], o[2].to(torch.int32)
'''
    return head + c2 + c2s + body + "\n" + tail


if __name__ == "__main__":
    with open(SRC) as f:
        want = transform(f.read())
    if "--check" in sys.argv:
        ok = False
        if os.path.exists(DST):
            with open(DST) as f:
                ok = f.read() == want
        print("dsa_sel2g", "matches" if ok else "STALE")
        sys.exit(0 if ok else 1)
    with open(DST, "w") as f:
        f.write(want)
    print("wrote", DST)
