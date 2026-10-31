"""kernels/dsa_slots_n.py's attention (the row count read at run time) with each device-loop iteration's rows split over
the two physical cores of a logical core (trn2 at LNC=2, grid 2).

Why: dsa_slots_n never reads program_id, so at grid 2 both physical cores run every row and write the same output. It
is the context-parallel prefill's attention (models/mla.py _cp_attend_classes): in a served 4096-token prefill call on
trn2, 9 calls in one 12-layer piece took 51.6 of its 465 ms, ~186 ms over the call's four pieces.

The split: each program attends a contiguous half of an iteration's RPI rows (dsa_slots._row, unchanged) and writes their rows of o and lse; every per-row buffer is SBUF or PSUM, so the halves share nothing but
the outputs, whose rows are disjoint. A core barrier on o (and lse) holds both programs until every row is written.
The loop runs the same run-time iteration count on both programs. Grid 2 therefore gives grid 1's bits; one program
per iteration row (RPI 1) or grid 1 runs every row in one program, as dsa_slots_n does.

This kernel's REV is its own source and dsa_slots_n.REV.
"""

from __future__ import annotations

import zlib

from . import dsa_slots as ds
from . import dsa_slots_n as dsn

if dsn.dsa_slots_n_kernel is not None:
    from .dsa_slots import KTR, _pair, _psum, _row, _sb  # the kernel tracer resolves names, not module attributes

    nki, nisa, nl = ds.nki, ds.nisa, ds.nl
    F32, BF16, I32 = nl.float32, nl.bfloat16, nl.int32
    KP = ds.KP

    @nki.jit
    def dsa_slots_n_s_kernel(q_lat, kc, rows_t, bias, identb, n_iter, scale: float, fp8: int, want_lse: int,
                                  rev: int, qkw: int = 1):
        """dsa_slots_n_kernel's arguments and outputs; at grid 2 each program runs half of every iteration's
        rows."""
        NI, rpi, H, R = q_lat.shape
        NCH = rows_t.shape[3]
        LC = R // 128
        T = NCH * KP * 128
        o = nl.ndarray((NI, rpi, H, R), dtype=F32, buffer=nl.shared_hbm)
        lse = None
        if want_lse:
            lse = nl.ndarray((NI, rpi, H, 1), dtype=F32, buffer=nl.shared_hbm)
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        split = npg == 2 and rpi >= 2
        k_lo, k_hi = (0, rpi) if not split else ((0, (rpi + 1) // 2) if pid == 0 else ((rpi + 1) // 2, rpi))
        KD = kc.dtype
        if fp8:
            KD = nl.float8_e4m3
        X = dict(H=H, R=R, LC=LC, NCH=NCH, T=T, rpi=rpi, q_lat=q_lat, kc=kc, rows_t=rows_t, bias=bias, o=o, lse=lse,
                 want_lse=want_lse, qkw=qkw)
        IB = _sb((128, 128), BF16)
        nisa.dma_copy(dst=IB, src=identb)
        X["IB"] = IB
        X["scale"] = scale
        X["K"] = []
        for _ in range(2):
            kv = []
            for _ in range(NCH):
                kv.append(_sb((128, KP * R), KD))
            X["K"].append(kv)
        X["KT"] = []
        if qkw:
            X["KTA"] = _pair((128, LC, T), BF16)
            X["BR"] = _pair((1, T), F32)
            X["RSC"] = _sb((1, 128))
            nisa.memset(dst=X["RSC"], value=1.0 / scale)
            if bias.dtype == BF16:  # a bf16 bias (dsa_slots_n.BIAS_BF16): the K = 1 bias matmul at the bf16 rate  # BB
                X["BR"] = _pair((1, T), BF16)  # BB
                X["RSC"] = _sb((1, 128), BF16)  # BB
                nisa.memset(dst=X["RSC"], value=1.0 / scale)  # BB
        else:
            for _ in range(KTR):
                X["KT"].append(_sb((128, LC, 128), BF16))
        X["RS"] = _pair((128, NCH), I32)
        X["QR"] = _pair((H, R), BF16)
        X["QT"] = _pair((128, LC, H), BF16)
        X["S"] = _pair((H, T), F32)
        X["P"] = _pair((H, T), BF16)
        X["OB"] = _pair((H, R), F32)
        X["MX"] = _pair((H, 1), F32)
        X["NMX"] = _pair((H, 1), F32)
        X["DEN"] = _pair((H, 1), F32)
        X["RDEN"] = _pair((H, 1), F32)
        X["LS"] = _pair((H, 1), F32)
        X["PTb"] = _pair((128, H), BF16)
        cnt = _sb((1, 1), I32)
        nisa.dma_copy(dst=cnt, src=n_iter)  # the run-time iteration count, the same on both programs
        rt = nisa.register_alloc()
        nisa.register_load(rt, cnt)

        def body(it):
            X["psum"] = _psum(H, LC)
            for k in range(k_lo, k_hi):
                _row(X, it, k)

        nl.fori_loop(0, rt, body)
        if split:  # every row written, both halves, before either program returns
            nisa.core_barrier(data=o, cores=(0, 1))
            if want_lse:
                nisa.core_barrier(data=lse, cores=(0, 1))
        if want_lse:
            return o, lse
        return o
else:
    dsa_slots_n_s_kernel = None


BB_TAG = "  # BB"


def _kernel_rev(bb: bool = False) -> int:
    """CRC-32 of this module's kernel source, XOR dsa_slots_n.REV (the row code and layout it shares): the static
    argument `rev`. The lines ending in BB_TAG (the bf16-bias branch, dsa_slots_n.BIAS_BF16) count only in REV_BB, the
    rev of a bf16-bias call, so every fp32-bias call (decode's fallbacks; any call with the knob unset) keeps its key."""
    src = open(__file__).read()
    a = src.index("if dsn.dsa_slots_n_kernel is not None:")
    b = src.index("    dsa_slots_n_s_kernel = None", a)
    body = src[a:b]
    if not bb:
        body = "".join(ln for ln in body.splitlines(True) if not ln.rstrip("\n").endswith(BB_TAG))
    return zlib.crc32(body.encode()) ^ dsn.REV


REV = _kernel_rev()
REV_BB = _kernel_rev(True)
