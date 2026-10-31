"""The shared FFN (MoE) half: the served FFN half's body as one plain NKI function, for the megakernels that run an FFN
half inside a bigger kernel (the fused DSA layer's). They call this, so they run the served code.

ffn_half_body is kernels/ffn_half_pb.py ffn_half_kernel's body statement for statement, with the kernel's arguments
less `rev`; tests/test_ffn_half_shared.py pins it to the served source, AST for AST, and plants changes that must break
the pin. It allocates its outputs (shared HBM) as the kernel does, so a caller gets the kernel's outputs as its own
intermediates. The pieces it calls (SH.mix_in_h, MS.front_body, _norm_xc, half_body_pb) are the served module's
objects, imported, not copied, so ffn_half_pb's forms act here exactly as there.

REV: a caller folds this module's REV into its own static `rev` (the compile-cache key hashes no NKI source). It covers
this module's body source and ffn_half_pb.REV, which covers the served kernel's source and its helpers' REVs.
"""

from __future__ import annotations

import zlib

from . import ffn_half_pb as FB
from . import mhc_split as MS
from . import shell as SH

P = 128

try:  # the Neuron venv
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


# -- the body source (REV covers from here to ffn_half_body = None) --------------------------------------------------

if nki is not None:
    _norm_xc, half_body_pb = FB._norm_xc, FB.half_body_pb  # the served helpers, by the names the body calls

    def ffn_half_body(S, post_p, comb_p, Yp, fnb, base, scale, pnorm, bb, bf, wb, hc: int, iters: int, eps: float,
                      rms_eps: float, sig: int, skf: int, Is: int, K: int, rsc: float, group: int, ring: int,
                      limit: float, rsp: int, o_rt: int, o_gu: int, o_dn: int, o_rb: int, dbg: int, fin: int):
        """ffn_half_kernel's body, verbatim (tests/test_ffn_half_shared.py): the attention output's mix, mHC's
        FFN-side front, the norm and row isolation, the MoE body with the Sinkhorn's pieces at its hooks. Returns
        (S_out, post_out, comb_out, rp, sp) (+ dbg's outputs), the kernel's tuple."""
        bf16, f32, u16, i32 = nl.bfloat16, nl.float32, nl.uint16, nl.int32
        VE = nisa.vector_engine
        T, F = S.shape
        H = F // hc
        W = H // P
        npg = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
        # outputs first, the same in both programs' traces
        S_out = nl.ndarray((T, F), dtype=bf16, buffer=nl.shared_hbm)
        if dbg:
            xn_out = nl.ndarray((T, H), dtype=bf16, buffer=nl.shared_hbm)
            ti_out = nl.ndarray((T, K), dtype=i32, buffer=nl.shared_hbm)
            tv_out = nl.ndarray((T, K), dtype=bf16, buffer=nl.shared_hbm)
        if dbg >= 2:
            ch_out = nl.ndarray((T, wb.shape[0]), dtype=f32, buffer=nl.shared_hbm)

        # the attention output's mix (mhc_mix's arithmetic), then this program's copy for the front's k-layout read
        sh = nl.ndarray((P, hc, T, W), dtype=bf16, buffer=nl.sbuf)
        SH.mix_in_h(sh, S, post_p, comb_p, Yp, S_out, hc, T, H, npg)
        s_pv = nl.ndarray((T, F), dtype=bf16, buffer=nl.private_hbm)
        for n in range(hc):
            SH.dma_static(dst=s_pv.ap(pattern=[[W, P], [F, T], [1, W]], offset=n * H), src=sh[:, n, :, :])

        # mHC's front (mhc_split): post written, comb allocated, cp the Sinkhorn's tile; xb its x in h-layout
        _, post_out, comb_out, cp, xb = MS.front_body(s_pv, fnb, base, scale, hc, iters, eps, rms_eps, sig, skf)
        # rms_norm(post_norm) and the row isolation in xb's h-layout, then the body's k-major x (no x_pv)
        xn, xc = _norm_xc(xb, pnorm, rms_eps, fin, T, H)

        # the MoE body with the Sinkhorn's pieces at its hooks: the last piece writes comb_out
        rp, sp, i8f, pvb, ch = half_body_pb(None, bb, bf, wb, Is, K, rsc, group, ring, limit, rsp, o_rt, o_gu, o_dn,
                                            o_rb, cp, comb_out, hc, iters, eps, skf, xc_in=xc)
        if dbg:  # both programs write the same values
            SH.dma_static(dst=xn_out.ap(pattern=[[W, P], [H, T], [1, W]], offset=0), src=xn)  # xn: h-layout
            ti = nl.ndarray((T, K), dtype=i32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ti, src=i8f, engine=VE)
            SH.dma_static(dst=ti_out, src=ti)
            SH.dma_static(dst=tv_out, src=pvb)
            if dbg >= 2:  # the router's choice as its top 8 read it (rsp 0: the same on both programs)
                SH.dma_static(dst=ch_out, src=ch)
                return S_out, post_out, comb_out, rp, sp, xn_out, ti_out, tv_out, ch_out
            return S_out, post_out, comb_out, rp, sp, xn_out, ti_out, tv_out
        return S_out, post_out, comb_out, rp, sp
else:
    ffn_half_body = None


REV = zlib.crc32(repr((SH.kernel_rev(__file__, "# -- the body source", "    ffn_half_body = None"), FB.REV)).encode())
