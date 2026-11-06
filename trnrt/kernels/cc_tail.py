"""The DSA decode tail after the attend, as kernel helpers: the context-parallel combine, W_UV and o_proj into this
rank's partial of the attention output. The fused DSA layer (kernels/dsa_satwf.py, kernels/dsa_satwfc.py) calls
_tail_weights at its entry and _tail_body after its all_to_all.

The served lines it replaces (the XLA form of models/mla.py attention_cp's decode branch):
    lse_all = gather(lse_r)                                   # a zero-padded group all-reduce
    LSE = logsumexp(lse_all, 0); part = o_r * exp(lse_r - LSE)[..., None]
    o = reduce_scatter(part over the group)                   # this rank's heads
    out = einsum("thr,hvr->thv", o.bf16, W_UV); y = F.linear(out, o_dec)
Here:
1. the group sum: every rank sends rank b the partials and lse of b's heads in one all_to_all, [A nh T, r + 16] fp32
   (rows (b, j, t); column r the lse, the rest zero); program 0 reads its own received rows;
2. the combine per head j, program 0: LSE = m + log(sum exp(L - m)), w = exp(L - LSE) (logsumexp's form), o = sum
   over sources s = 0 .. A-1 of w_s o_s in fp32, summed in XLA's reduce-scatter tree order, then bf16;
3. W_UV per head from a pre-tiled copy w_uv_p [128, nh RT, dv] (o^T made on the PE by an identity matmul: exact), bf16
   out, in XLA's own form (two PSUM groups of K tiles, summed in fp32);
4. o_proj from a pre-tiled copy o_p [128, NCH FT, 512], all H columns, into the caller's world-sum send buffer.
The pre-tiled copies (models/mla.py, dsa_sat.tile_wuv / tile_o, a repack once at load) are the SBUF tiles' own layout,
partition-major, so each weight DMA reads 2-4 KB contiguous a partition (K-major copies read 0.5-1 KB runs, ~131 GB/s,
~42 us a call exposed with fresh weights).
Only program 0 consumes the collective's result and writes the outputs: on the device, program 1's reads of a
collective's result, or of a shared buffer program 0 wrote, are not reliable. Every write is a static-descriptor DMA."""

from __future__ import annotations

import os
import zlib

from . import shell as SH

# The heads stacked on partitions (_tail_heads_hs below): every per-row op once for all heads, each (head, row)'s
# arithmetic unchanged.
HS = 1
LP = 16  # the lse column and its padding: rows of r + 16 fp32
TP = 128
POISON_RX, POISON_OUT = -7777.0, -5555.0

try:  # the Neuron venv
    import nki
    import nki.collectives as ncc
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    def _dma(dst, src):
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.none)

    def _transpose(dst, src, ident, T: int, n: int, off: int = 0):
        """dst[:, off + i, :] ([128, T] bf16) = src [T, n 128]'s tile i transposed on the PE (x * 1 plus zeros: exact)."""
        for i in range(n):
            pt = nl.ndarray((TP, T), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=pt, stationary=src[:, i * TP:(i + 1) * TP], moving=ident, accumulate=False)
            nisa.tensor_copy(dst=dst[:, off + i, :], src=pt, engine=nisa.vector_engine)

    def _tail_weights(w_uv_p, o_p, wuv_all, wo_all, nh: int, RT: int, dv: int, NCH: int, FT: int):
        """The tail's two pre-tiled weight loads into persistent SBUF tiles (one DMA a head, one a column chunk,
        contiguous a partition). The caller issues them where it wants, on program 0 (the fused layer: at its entry)."""
        for j in range(nh):
            pat = [[nh * RT * dv, TP], [1, RT * dv]]
            _dma(dst=wuv_all.ap(pattern=pat, offset=j * RT * dv), src=w_uv_p.ap(pattern=pat, offset=j * RT * dv))
        for c in range(NCH):
            pat = [[NCH * FT * 512, TP], [1, FT * 512]]
            _dma(dst=wo_all.ap(pattern=pat, offset=c * FT * 512), src=o_p.ap(pattern=pat, offset=c * FT * 512))

    def _tail_heads(rs, wuv_all, uvT, rowok, idt, o_out, T: int, A: int, nh: int, r: int, dv: int, so: int, wu: int,
                    nf: int, oo: int):
        """Program 0: the combine of every head j from the all_to_all's receive buffer rs [(s, j, t) rows, r + LP]
        (sources s by rank within the CP group; columns 0 .. r only are read, the padding never), in sum order so, then
        W_UV (form wu) into out^T uvT [128, nh dv / 128, T]; rowok [T, 1] := min over a row's finite flags (nf); o_out
        [nh, T, r] bf16 gets the combine's o when oo."""
        if HS:
            return _tail_heads_hs(rs, wuv_all, uvT, rowok, idt, o_out, T, A, nh, r, dv, so, wu, nf, oo)
        Cn = r + LP
        RT = r // TP
        f32, bf16 = nl.float32, nl.bfloat16
        for j in range(nh):
            L = nl.ndarray((T, A), dtype=f32, buffer=nl.sbuf)
            # head j's received rows (s, j, t) straight from the receive buffer into [t, s, :], one DMA a head
            cb = nl.ndarray((T, A, Cn), dtype=f32, buffer=nl.sbuf)
            _dma(dst=cb, src=rs.ap(pattern=[[Cn, T], [nh * T * Cn, A], [1, Cn]], offset=j * T * Cn))
            nisa.tensor_copy(dst=L, src=cb.ap(pattern=[[A * Cn, T], [Cn, A]], offset=r), engine=nisa.vector_engine)
            m = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=m, op=nl.maximum, data=L, axis=1)
            nm = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=nm, data=m, op0=nl.multiply, operand0=-1.0, engine=nisa.vector_engine)
            e = nl.ndarray((T, A), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=e, op=nl.exp, data=L, bias=nm, scale=1.0)
            S = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=S, op=nl.add, data=e, axis=1)
            lnS = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=lnS, op=nl.log, data=S)
            nL = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)  # -(m + log S)
            nisa.tensor_tensor(dst=nL, data1=nm, data2=lnS, op=nl.subtract, engine=nisa.vector_engine)
            w = nl.ndarray((T, A), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=w, op=nl.exp, data=L, bias=nL, scale=1.0)
            acc = nl.ndarray((T, r), dtype=f32, buffer=nl.sbuf)
            if so:  # XLA's reduce-scatter order: the products, then a pairwise tree in place (literal strides)
                pt = nl.ndarray((T, A, r), dtype=f32, buffer=nl.sbuf)
                for s_ in range(A):
                    nisa.tensor_scalar(dst=pt[:, s_, :], data=cb[:, s_, 0:r], op0=nl.multiply,
                                       operand0=w[:, s_:s_ + 1], engine=nisa.vector_engine)
                for i in range(A // 2):  # (0+1), (2+3), ..
                    nisa.tensor_tensor(dst=pt[:, 2 * i, :], data1=pt[:, 2 * i, :], data2=pt[:, 2 * i + 1, :],
                                       op=nl.add, engine=nisa.vector_engine)
                if A >= 4:  # ((0+1)+(2+3)), ..
                    for i in range(A // 4):
                        nisa.tensor_tensor(dst=pt[:, 4 * i, :], data1=pt[:, 4 * i, :],
                                           data2=pt[:, 4 * i + 2, :], op=nl.add, engine=nisa.vector_engine)
                if A >= 8:
                    nisa.tensor_tensor(dst=pt[:, 0, :], data1=pt[:, 0, :], data2=pt[:, 4, :], op=nl.add,
                                       engine=nisa.vector_engine)
                nisa.tensor_copy(dst=acc, src=pt[:, 0, :], engine=nisa.vector_engine)
            else:
                tmp = nl.ndarray((T, r), dtype=f32, buffer=nl.sbuf)
                for s_ in range(A):  # sources in rank order
                    nisa.tensor_scalar(dst=acc if s_ == 0 else tmp, data=cb[:, s_, 0:r], op0=nl.multiply,
                                       operand0=w[:, s_:s_ + 1], engine=nisa.vector_engine)
                    if s_:
                        nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add, engine=nisa.vector_engine)
            ob = nl.ndarray((T, r), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ob, src=acc, engine=nisa.vector_engine)
            if oo:
                _dma(dst=o_out.ap(pattern=[[r, T], [1, r]], offset=j * T * r), src=ob)
            src_t = ob
            if nf:
                dz = nl.ndarray((T, r), dtype=f32, buffer=nl.sbuf)  # 0 where finite, NaN where NaN or inf
                nisa.tensor_tensor(dst=dz, data1=ob, data2=ob, op=nl.subtract, engine=nisa.vector_engine)
                okm = nl.ndarray((T, r), dtype=f32, buffer=nl.sbuf)  # 1.0 / 0.0
                nisa.tensor_scalar(dst=okm, data=dz, op0=nl.equal, operand0=0.0, engine=nisa.vector_engine)
                obc = nl.ndarray((T, r), dtype=bf16, buffer=nl.sbuf)
                nisa.memset(dst=obc, value=0.0)
                nisa.tensor_copy_predicated(dst=obc, src=ob, predicate=okm.view(nl.uint32))
                rk = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=rk, op=nl.minimum, data=okm, axis=1)
                nisa.tensor_tensor(dst=rowok, data1=rowok, data2=rk, op=nl.minimum, engine=nisa.vector_engine)
                src_t = obc
            oT = nl.ndarray((TP, RT, T), dtype=bf16, buffer=nl.sbuf)
            _transpose(oT, src_t, idt, T, RT)
            if wu:  # XLA's W_UV form: two PSUM groups of K tiles (their own tiles, not slices of one), fp32 sum
                for vt in range(dv // TP):
                    pa = nl.ndarray((TP, T), dtype=f32, buffer=nl.psum)
                    pb = nl.ndarray((TP, T), dtype=f32, buffer=nl.psum)
                    for rt in range(RT // 2):
                        nisa.nc_matmul(dst=pa, stationary=wuv_all[:, j * RT + rt, vt * TP:(vt + 1) * TP],
                                       moving=oT[:, rt, :], accumulate=(rt > 0))
                    for rt in range(RT // 2, RT):
                        nisa.nc_matmul(dst=pb, stationary=wuv_all[:, j * RT + rt, vt * TP:(vt + 1) * TP],
                                       moving=oT[:, rt, :], accumulate=(rt > RT // 2))
                    sa = nl.ndarray((TP, T), dtype=f32, buffer=nl.sbuf)
                    sb = nl.ndarray((TP, T), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=sa, src=pa, engine=nisa.vector_engine)
                    nisa.tensor_copy(dst=sb, src=pb, engine=nisa.vector_engine)
                    nisa.tensor_tensor(dst=sa, data1=sa, data2=sb, op=nl.add, engine=nisa.vector_engine)
                    nisa.tensor_copy(dst=uvT[:, j * (dv // TP) + vt, :], src=sa, engine=nisa.vector_engine)
            else:
                pu = nl.ndarray((T, dv), dtype=f32, buffer=nl.psum)
                for rt in range(RT):
                    nisa.nc_matmul(dst=pu, stationary=oT[:, rt, :], moving=wuv_all[:, j * RT + rt, :],
                                   accumulate=(rt > 0))
                ub = nl.ndarray((T, dv), dtype=bf16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ub, src=pu, engine=nisa.vector_engine)
                _transpose(uvT, ub, idt, T, dv // TP, off=j * (dv // TP))

    def _tail_oproj(uvT, wo_all, rowok, sp, T: int, H: int, NCH: int, FT: int, nf: int, ev: int):
        """Program 0: o_proj from out^T into sp [T, H] bf16 (this rank's partial P), the rows rowok flags made NaN (nf),
        the evacuation alternating between Vector and Scalar a chunk (ev)."""
        f32, bf16 = nl.float32, nl.bfloat16
        if nf:  # o_proj's evacuation scale per row, 1.0 on healthy rows and NaN on the flagged ones
            bad = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=bad, data=rowok, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0,
                               engine=nisa.vector_engine)
            sc = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
            nisa.memset(dst=sc, value=1.0)
            nan1 = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
            nisa.memset(dst=nan1, value=float("nan"))
            nisa.tensor_copy_predicated(dst=sc, src=nan1, predicate=bad.view(nl.uint32))
        for c in range(NCH):  # all of o_proj's columns, into program 0's world-sum send buffer
            col = c * 512
            pm = nl.ndarray((T, 512), dtype=f32, buffer=nl.psum)
            for ft in range(FT):
                nisa.nc_matmul(dst=pm, stationary=uvT[:, ft, :], moving=wo_all[:, c * FT + ft, :],
                               accumulate=(ft > 0))
            # one pass a chunk (rather than the cast, then a NaN predicated copy), the chunks alternating between
            # Vector and Scalar when ev; x * 1.0 is exact (-0 kept), x * NaN = NaN
            eng = nisa.scalar_engine if ev and c % 2 else nisa.vector_engine
            yb = nl.ndarray((T, 512), dtype=bf16, buffer=nl.sbuf)
            if nf:
                nisa.tensor_scalar(dst=yb, data=pm, op0=nl.multiply, operand0=sc[:, 0:1], engine=eng)
            else:
                nisa.tensor_copy(dst=yb, src=pm, engine=eng)
            _dma(dst=sp.ap(pattern=[[H, T], [1, 512]], offset=col), src=yb)

    def _tail_body(rs, wuv_all, wo_all, idt, sp, o_out, T: int, A: int, nh: int, r: int, dv: int, H: int, so: int = 1,
                   wu: int = 1, nf: int = 1, ev: int = 1, oo: int = 1):
        """Program 0's whole tail after the all_to_all (the fused DSA layer's join calls this verbatim): the combine
        (so), W_UV (wu) and o_proj from the receive buffer rs into sp [T, H] bf16; the weights already in wuv_all /
        wo_all (_tail_weights); idt the [T, T] bf16 identity in SBUF."""
        FT, NCH = nh * dv // TP, H // 512
        uvT = nl.ndarray((TP, FT, T), dtype=nl.bfloat16, buffer=nl.sbuf)  # out^T: feature f = j dv + v on partitions
        rowok = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=rowok, value=1.0)
        _tail_heads(rs, wuv_all, uvT, rowok, idt, o_out, T, A, nh, r, dv, so, wu, nf, oo)
        _tail_oproj(uvT, wo_all, rowok, sp, T, H, NCH, FT, nf, ev)
# -- the tail end


if nki is not None:  # the heads-stacked form (HS)
    def _tail_heads_hs(rs, wuv_all, uvT, rowok, idt, o_out, T: int, A: int, nh: int, r: int, dv: int, so: int,
                       wu: int, nf: int, oo: int):
        """_tail_heads with the heads stacked on partitions, p = j T + t (nh T <= 128): one DMA brings every head's
        rows, every per-row op (the weights' softmax, the products and their pairwise tree, the cast, the finite
        screen) runs once for all heads with each partition's arithmetic unchanged, one transpose set covers every
        head (an nh T identity made here), and W_UV stays per head on the PE. rowok's min over the heads takes head j's
        flags to partitions 0 .. T - 1 by a static SBUF DMA first (min is exact)."""
        Cn = r + LP
        RT = r // TP
        NT = nh * T
        f32, bf16 = nl.float32, nl.bfloat16
        L = nl.ndarray((NT, A), dtype=f32, buffer=nl.sbuf)
        cb = nl.ndarray((NT, A, Cn), dtype=f32, buffer=nl.sbuf)
        _dma(dst=cb, src=rs.ap(pattern=[[Cn, NT], [nh * T * Cn, A], [1, Cn]], offset=0))
        nisa.tensor_copy(dst=L, src=cb.ap(pattern=[[A * Cn, NT], [Cn, A]], offset=r), engine=nisa.vector_engine)
        m = nl.ndarray((NT, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=m, op=nl.maximum, data=L, axis=1)
        nm = nl.ndarray((NT, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nm, data=m, op0=nl.multiply, operand0=-1.0, engine=nisa.vector_engine)
        e = nl.ndarray((NT, A), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=e, op=nl.exp, data=L, bias=nm, scale=1.0)
        S = nl.ndarray((NT, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=S, op=nl.add, data=e, axis=1)
        lnS = nl.ndarray((NT, 1), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=lnS, op=nl.log, data=S)
        nL = nl.ndarray((NT, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=nL, data1=nm, data2=lnS, op=nl.subtract, engine=nisa.vector_engine)
        w = nl.ndarray((NT, A), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=w, op=nl.exp, data=L, bias=nL, scale=1.0)
        acc = nl.ndarray((NT, r), dtype=f32, buffer=nl.sbuf)
        if so:  # XLA's reduce-scatter order: the products, then a pairwise tree in place (literal strides)
            pt = nl.ndarray((NT, A, r), dtype=f32, buffer=nl.sbuf)
            for s_ in range(A):
                nisa.tensor_scalar(dst=pt[:, s_, :], data=cb[:, s_, 0:r], op0=nl.multiply, operand0=w[:, s_:s_ + 1],
                                   engine=nisa.vector_engine)
            for i in range(A // 2):
                nisa.tensor_tensor(dst=pt[:, 2 * i, :], data1=pt[:, 2 * i, :], data2=pt[:, 2 * i + 1, :], op=nl.add,
                                   engine=nisa.vector_engine)
            if A >= 4:
                for i in range(A // 4):
                    nisa.tensor_tensor(dst=pt[:, 4 * i, :], data1=pt[:, 4 * i, :], data2=pt[:, 4 * i + 2, :],
                                       op=nl.add, engine=nisa.vector_engine)
            if A >= 8:
                nisa.tensor_tensor(dst=pt[:, 0, :], data1=pt[:, 0, :], data2=pt[:, 4, :], op=nl.add,
                                   engine=nisa.vector_engine)
            nisa.tensor_copy(dst=acc, src=pt[:, 0, :], engine=nisa.vector_engine)
        else:
            tmp = nl.ndarray((NT, r), dtype=f32, buffer=nl.sbuf)
            for s_ in range(A):  # sources in rank order
                nisa.tensor_scalar(dst=acc if s_ == 0 else tmp, data=cb[:, s_, 0:r], op0=nl.multiply,
                                   operand0=w[:, s_:s_ + 1], engine=nisa.vector_engine)
                if s_:
                    nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add, engine=nisa.vector_engine)
        ob = nl.ndarray((NT, r), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ob, src=acc, engine=nisa.vector_engine)
        if oo:
            _dma(dst=o_out.ap(pattern=[[r, NT], [1, r]], offset=0), src=ob)
        src_t = ob
        if nf:
            dz = nl.ndarray((NT, r), dtype=f32, buffer=nl.sbuf)  # 0 where finite, NaN where NaN or inf
            nisa.tensor_tensor(dst=dz, data1=ob, data2=ob, op=nl.subtract, engine=nisa.vector_engine)
            okm = nl.ndarray((NT, r), dtype=f32, buffer=nl.sbuf)  # 1.0 / 0.0
            nisa.tensor_scalar(dst=okm, data=dz, op0=nl.equal, operand0=0.0, engine=nisa.vector_engine)
            obc = nl.ndarray((NT, r), dtype=bf16, buffer=nl.sbuf)
            nisa.memset(dst=obc, value=0.0)
            nisa.tensor_copy_predicated(dst=obc, src=ob, predicate=okm.view(nl.uint32))
            rk = nl.ndarray((NT, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=rk, op=nl.minimum, data=okm, axis=1)
            for j in range(nh):  # head j's flags onto partitions 0 .. T - 1, then the same min as _tail_heads'
                if j == 0:
                    nisa.tensor_tensor(dst=rowok, data1=rowok, data2=rk[0:T, :], op=nl.minimum, engine=nisa.vector_engine)
                else:
                    rkj = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
                    _dma(dst=rkj, src=rk[j * T:(j + 1) * T, :])
                    nisa.tensor_tensor(dst=rowok, data1=rowok, data2=rkj, op=nl.minimum, engine=nisa.vector_engine)
            src_t = obc
        io = nl.ndarray((NT, NT), dtype=nl.int32, buffer=nl.sbuf)  # the nh T identity: col - row == 0
        nisa.iota(dst=io, pattern=[[1, NT]], offset=0, channel_multiplier=-1)
        iof = nl.ndarray((NT, NT), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=iof, src=io, engine=nisa.vector_engine)
        idn = nl.ndarray((NT, NT), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=idn, data=iof, op0=nl.equal, operand0=0.0, engine=nisa.vector_engine)
        oT = nl.ndarray((TP, RT, NT), dtype=bf16, buffer=nl.sbuf)
        _transpose(oT, src_t, idn, NT, RT)
        for j in range(nh):
            if wu:  # XLA's W_UV form: two PSUM groups of K tiles (their own tiles, not slices of one), fp32 sum
                for vt in range(dv // TP):
                    pa = nl.ndarray((TP, T), dtype=f32, buffer=nl.psum)
                    pb = nl.ndarray((TP, T), dtype=f32, buffer=nl.psum)
                    for rt in range(RT // 2):
                        nisa.nc_matmul(dst=pa, stationary=wuv_all[:, j * RT + rt, vt * TP:(vt + 1) * TP],
                                       moving=oT[:, rt, j * T:(j + 1) * T], accumulate=(rt > 0))
                    for rt in range(RT // 2, RT):
                        nisa.nc_matmul(dst=pb, stationary=wuv_all[:, j * RT + rt, vt * TP:(vt + 1) * TP],
                                       moving=oT[:, rt, j * T:(j + 1) * T], accumulate=(rt > RT // 2))
                    sa = nl.ndarray((TP, T), dtype=f32, buffer=nl.sbuf)
                    sb = nl.ndarray((TP, T), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=sa, src=pa, engine=nisa.vector_engine)
                    nisa.tensor_copy(dst=sb, src=pb, engine=nisa.vector_engine)
                    nisa.tensor_tensor(dst=sa, data1=sa, data2=sb, op=nl.add, engine=nisa.vector_engine)
                    nisa.tensor_copy(dst=uvT[:, j * (dv // TP) + vt, :], src=sa, engine=nisa.vector_engine)
            else:
                pu = nl.ndarray((T, dv), dtype=f32, buffer=nl.psum)
                for rt in range(RT):
                    nisa.nc_matmul(dst=pu, stationary=oT[:, rt, j * T:(j + 1) * T], moving=wuv_all[:, j * RT + rt, :],
                                   accumulate=(rt > 0))
                ub = nl.ndarray((T, dv), dtype=bf16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ub, src=pu, engine=nisa.vector_engine)
                _transpose(uvT, ub, idt, T, dv // TP, off=j * (dv // TP))
# -- _tail_heads_hs end


REV = (SH.kernel_rev(__file__, "LP = 16", "# -- the tail end")
       ^ SH.kernel_rev(__file__, "    def _tail_heads_hs(", "# -- _tail_heads_hs end"))

# -- the hardware-DGE span (TRNRT_DMA_HW): with TRNRT_DMA_HW=1 this module's static DMA helper (_dma) issues its DMAs in
# hardware DGE (dge_mode hwdge) instead of static descriptors, which cost a descriptor-ring entry per contiguous run in
# every loaded NEFF (the decode spill rings). Every MTP preset sets it (it is in every build behind the published MTP
# numbers); the default presets do not. Read at import, before any kernel module binds _dma by name (dsa_sat and the
# dsa_satwf family's `from .cc_tail import _dma`) or folds REV, so the process environment has to carry it
# (trnrt/__main__.py sets it for a preset that needs it). Below the hashed spans; REV folds this span's source when it
# is set (_dma_hw_rev), so every kernel folding REV re-keys, and unset no source, REV or key moves.
if os.environ.get("TRNRT_DMA_HW", "0") == "1" and nki is not None:
    def _dma(dst, src):  # noqa: F811
        """_dma under TRNRT_DMA_HW=1: hardware DGE."""
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge)
# -- the hardware-DGE span end


def _dma_hw_rev() -> int:
    """CRC-32 of the hardware-DGE span above (the TRNRT_DMA_HW variant's source): REV's term when it is set."""
    src = open(os.path.abspath(__file__)).read()
    a = src.index("# -- the hardware-DGE span (TRNRT_DMA_HW)")
    return zlib.crc32(src[a:src.index("# -- the hardware-DGE span end", a)].encode())


if os.environ.get("TRNRT_DMA_HW", "0") == "1":
    REV ^= _dma_hw_rev()
