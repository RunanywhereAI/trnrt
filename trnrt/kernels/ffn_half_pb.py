"""GLM-5.3-Flash's FFN half of a decode layer after its attention all-reduce, as ONE NKI kernel (the served form;
kernels/ffn_half.py has its host side):

    mix_in:   S_n = the attention output's mix of S (kernels/mhc_mix.py's arithmetic, shell.mix_in_h)
    front:    x, post of S_n and comb's Sinkhorn input (mhc_split.front_body, mhc_pre's statements)
    rms_norm: xn = post_norm * (x r) in mHC's h-layout (_norm_xc), then the row isolation (rowiso)
    MoE:      the router, the shared expert and the routed pairs on xn (half_body_pb; its weights moe_half's)

with mHC's Sinkhorn issued in pieces INSIDE the MoE body, at static hooks between its stages. Issued as one block
after the body, the scheduler kept it there: a median 0.20 / 0.16 of the body's vector instructions inside the
chain's span, against 0.03 / 0.04 ahead of it. The chain stays near its source position, so here it is cut into
pieces placed between the body's own stages:
- H0 after the router's logits matmuls, H1 after the shared expert's gate / up matmuls, H2 after the a^T matmul (the
  body's vector work before the routed loop), and L1 .. Ln after routed group it's down matmuls, it = 1 .. n;
- n = NQ // 2, the smaller program's group count, so both programs issue every piece at the same static point; pieces
  of ceil(39 / (3 + n)) half-rounds in hook order, none past the end; the last non-empty one writes comb.

half_body_pb is the MoE half's body (the router, the shared expert, moe_dedupe.moe_tiles_pairs_v1's routed pairs; see
kernels/moe_half.py) plus the hook calls. sinkhorn_body's pieces join exactly (mhc_split's chunk test). The kernel's rev
is this module's source with its bodies' (mhc_split's REV, moe_half's for the layout it reads).

The norm and the row isolation run in mHC's h-layout on 128 lanes (_norm_xc), and the body takes the k-major x from it
(half_body_pb's xc_in path, the pin's one named divergence). That removes a private-HBM round trip of x, the 4-lane
norm and isolation passes, and the body's own transposes. The norm follows XLA's device arithmetic for the served
rms_norm (see _norm_xc), so xn matches the XLA path's bits."""

from __future__ import annotations

import zlib

from . import mhc_split as MS
from . import moe_half as MH
from . import rowiso as RI
from . import shell
from . import shell as SH
from .moe_half import KTOP

P = 128
# The routed experts' ring loads: by the compiler's choice for wb.select (GpSimd's software DGE). GATHER_HW (off): by
# hardware DGE, each expert's rows at a run-time offset (kda_half's state-load form).
GATHER_HW = False
# The router's chosen-expert scores tv: KTOP x (compare, multiply, reduce) on [T, ER]. TV_WIDE (off): one broadcast
# compare [T, KTOP, ER], one multiply and one reduce, the same per-element arithmetic.
TV_WIDE = False
# _norm_xc's rms_norm(post_norm), in XLA's device arithmetic for the served rms_norm (decoder.rms_norm on the XLA path):
# NORM_1R one rounding, bf16(w (x r)) from fp32 (not the torch order's two); NORM_NN no Newton step on r; RSQ_POOL r is
# GpSimd's rsqrt, XLA's instruction for that rms_norm (its decode graphs load no sqrt ACT set, and the backend runs
# rsqrt on GpSimd); SS_XLA the row sum of squares in XLA's device order (element i + 128 c on partition i, one in-order
# fp32 chain over c at T >= 2, two at T 1, c < 16 and c >= 16, added in fp32; then the ones-matmul over the
# partitions), measured per bucket on 1024-1032 rows. An earlier form in the torch order put 26% of xn's elements one
# bf16 ulp off XLA's rms_norm.
NORM_1R = True
NORM_NN = True
SS_XLA = True
RSQ_POOL = True

try:  # the Neuron venv
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


# -- the kernel source (REV covers from here to ffn_half_kernel = None) -----------------------------------------

def pb_hooks(NQ: int, n_half: int) -> tuple[int, int]:
    """(n, step): the loop hooks L1 .. Ln (n = NQ // 2, the smaller program's group count) and the piece length,
    ceil(n_half / (3 + n)) half-rounds, for the 3 + n hooks H0, H1, H2, L1 .. Ln. Integer arithmetic only (called in
    the kernel body)."""
    n = NQ // 2
    return n, (n_half + 3 + n - 1) // (3 + n)


if nki is not None:
    def _piece(cp, comb_out, hc: int, iters: int, eps: float, skf: int, p: int, step: int):
        """Hook p's piece of the Sinkhorn: half-rounds [p step, min((p + 1) step, 2 iters - 1)); nothing past the end.
        Module level: the NKI frontend refuses calls to inner functions."""
        n_half = 2 * iters - 1
        lo = p * step
        if lo < n_half:
            hi = lo + step
            if hi > n_half:
                hi = n_half
            MS.sinkhorn_body(cp, comb_out, hc, iters, eps, skf, lo, hi)

    def _sel_kmajor(P: int, R: int, C: int):
        """(mb bf16 [P, R] = [p % R = r], selc bf16 [P, C] = [p // R = c]): _expand_kmajor's masks (R = P / W)."""
        bf16, f32, i32 = nl.bfloat16, nl.float32, nl.int32
        VE = nisa.vector_engine
        d = nl.ndarray((P, R, C), dtype=i32, buffer=nl.sbuf)  # p - r - R c
        nisa.iota(dst=d, pattern=[[-1, R], [-R, C]], offset=0, channel_multiplier=1)
        df = nl.ndarray((P, R, C), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=df, src=d, engine=VE)
        nisa.tensor_scalar(dst=df, data=df, op0=nl.equal, operand0=0.0, engine=VE)
        mask = nl.ndarray((P, R), dtype=f32, buffer=nl.sbuf)  # [p % R = r]: at most one c matches
        nisa.tensor_reduce(dst=mask, op=nl.add, data=df, axis=2)
        mb = nl.ndarray((P, R), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mb, src=mask, engine=VE)
        g = nl.ndarray((P, C), dtype=i32, buffer=nl.sbuf)  # p - R c: in [0, R) exactly when p // R = c
        nisa.iota(dst=g, pattern=[[-R, C]], offset=0, channel_multiplier=1)
        gf = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gf, src=g, engine=VE)
        lo = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lo, data=gf, op0=nl.greater_equal, operand0=0.0, engine=VE)
        nisa.tensor_scalar(dst=gf, data=gf, op0=nl.greater_equal, operand0=float(R), engine=VE)
        nisa.tensor_tensor(dst=lo, data1=lo, data2=gf, op=nl.subtract, engine=VE)
        selc = nl.ndarray((P, C), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=selc, src=lo, engine=VE)
        return mb, selc

    def _expand_kmajor(xs, sel, T: int, R: int, C: int, W: int):
        """ps fp32 [P, T, C] (PSUM), ps[i, t, c] = x[t, c P + i] for xs bf16 [P, T, W] in h-layout (xs[p, t, w] =
        x[t, p W + w]): the masked expansion, then per row one start-only matmul by selc. Every output is one bf16 x 1
        product, so exact."""
        bf16, f32 = nl.bfloat16, nl.float32
        mb, selc = sel
        xe = nl.ndarray((P, T, R, W), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=xe, data1=xs.ap(pattern=[[T * W, P], [W, T], [0, R], [1, W]], offset=0),
                           data2=mb.ap(pattern=[[R, P], [0, T], [1, R], [0, W]], offset=0), op=nl.multiply,
                           engine=nisa.vector_engine)
        ps = nl.ndarray((P, T, C), dtype=f32, buffer=nl.psum)
        for t in range(T):
            nisa.nc_matmul(dst=ps[:, t, :], stationary=xe[:, t, :, :].flatten_dims(1, 2), moving=selc,
                           accumulate=False)
        return ps

    def _norm_xc(xb, pnorm, rms_eps: float, fin: int, T: int, H: int):
        """rms_norm(post_norm) and the row isolation on mHC's
        h-layout x (xb bf16 [P, T, W], xb[p, t, w] = x[t, p W + w]) on all 128 lanes, then the MoE body's k-major
        input xc bf16 [P, C, T] (xc[i, c, t] = xn[t, c P + i]), with no private-HBM round trip and no 4-lane pass.
        Every op is the served kernel's element for element except the row sum of squares: in order over w per
        partition, then across the partitions by a ones-matmul (fp64 rounded once on the device; the simulator sums in
        order). SS_XLA (served): that sum in XLA's device order instead (x relaid k-major by the xc expansion, chained
        over c, the same ones-matmul), measured bit-exact against XLA's rms_norm in every bucket.
        xc: xe[p, t, r W + w] = xs[p, t, w] [p % R = r] (R = P / W), then per row one start-only matmul by
        selc[p, c] = [p // R = c]: no partition offset, no accumulation group, every output one bf16 x 1 product.
        Returns (xn bf16 [P, T, W], xc bf16 [P, C, T])."""
        bf16, f32, u16, i32 = nl.bfloat16, nl.float32, nl.uint16, nl.int32
        VE = nisa.vector_engine
        W = H // P
        C = H // P
        R = P // W
        # the sum of squares of each row, on every partition: per partition over w, then a ones-matmul (fp32 both)
        xf = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xf, src=xb, engine=VE)
        sel = _sel_kmajor(P, R, C) if SS_XLA else None
        if SS_XLA:  # XLA's order: x[t, c P + i] on partition i (the xc expansion on x, exact), chained over c in order
            xk = nl.ndarray((P, T, C), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xk, src=_expand_kmajor(xb, sel, T, R, C, W), engine=VE)
            sqk = nl.ndarray((P, T, C), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sqk, data1=xk, data2=xk, op=nl.multiply, engine=VE)
            ssp = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
            if T == 1:  # two chains, added in fp32 (the compiler's reduce2x16x128: one row split in halves)
                sh2 = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=ssp, op=nl.add, data=sqk[:, :, 0:C // 2], axis=2)
                nisa.tensor_reduce(dst=sh2, op=nl.add, data=sqk[:, :, C // 2:C], axis=2)
                nisa.tensor_tensor(dst=ssp, data1=ssp, data2=sh2, op=nl.add, engine=VE)
            else:  # one chain over all of c (reduce32xTx128: the rows split over the pair)
                nisa.tensor_reduce(dst=ssp, op=nl.add, data=sqk, axis=2)
        else:
            sq = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=xf, data2=xf, op=nl.multiply, engine=VE)
            ssp = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=ssp, op=nl.add, data=sq, axis=2)
        ones = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=1.0)
        pss = nl.ndarray((P, T), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pss, stationary=ones, moving=ssp, accumulate=False)
        ss = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ss, src=pss, engine=VE)
        # decoder.rms_norm's order as the served kernel's: fp32 x rsqrt(mean(x^2) + eps) with one Newton step, bf16,
        # times the weight, bf16; here r is per (p, t), the same value on every partition
        a = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=a, data=ss, op0=nl.multiply, operand0=1.0 / H, op1=nl.add, operand1=rms_eps, engine=VE)
        r0 = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
        if RSQ_POOL:  # GpSimd's rsqrt (XLA's), no Newton step below
            nisa.tensor_scalar(dst=r0, data=a, op0=nl.rsqrt, operand0=None, engine=nisa.gpsimd_engine)
        else:
            nisa.activation(dst=r0, op=nl.rsqrt, data=a)
        if NORM_NN:  # no Newton step: r as computed above
            r = r0
        else:
            nt = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=nt, data1=r0, data2=r0, op=nl.multiply, engine=VE)
            nisa.tensor_tensor(dst=nt, data1=nt, data2=a, op=nl.multiply, engine=VE)
            nisa.tensor_scalar(dst=nt, data=nt, op0=nl.multiply, operand0=-0.5, op1=nl.add, operand1=1.5, engine=VE)
            r = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r, data1=r0, data2=nt, op=nl.multiply, engine=VE)
        nisa.tensor_tensor(dst=xf, data1=xf, data2=r.ap(pattern=[[T, P], [1, T], [0, W]], offset=0), op=nl.multiply,
                           engine=VE)
        if NORM_1R:  # one rounding, the product below from fp32 x r
            xq = xf
        else:
            xq = nl.ndarray((P, T, W), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xq, src=xf, engine=VE)
        wn = nl.ndarray((P, W), dtype=bf16, buffer=nl.sbuf)  # post_norm in h-layout: wn[p, w] = pnorm[p W + w]
        SH.dma_static(dst=wn, src=pnorm.ap(pattern=[[W, P], [1, W]], offset=0))
        xn = nl.ndarray((P, T, W), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=xn, data1=wn.ap(pattern=[[W, P], [0, T], [1, W]], offset=0), data2=xq, op=nl.multiply,
                           engine=VE)
        # row isolation (elementwise, so layout-free: the served result on the same xn)
        xs = xn
        if fin:
            fz = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
            x16 = nl.ndarray((P, T, W), dtype=u16, buffer=nl.sbuf)
            if fin == 2:
                RI.finite_or_zero3(xn.view(u16), x16, fz)
            else:
                SH.finite_or_zero(xn.view(u16), x16, fz)
            xs = x16.view(bf16)
        # xc by the masked expansion and one start-only matmul per row
        ps = _expand_kmajor(xs, sel if SS_XLA else _sel_kmajor(P, R, C), T, R, C, W)
        xc = nl.ndarray((P, C, T), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc.ap(pattern=[[C * T, P], [1, T], [T, C]], offset=0), src=ps, engine=VE)
        return xn, xc

    def half_body_pb(x, bb, bf, wb, Is: int, K: int, scale: float, group: int, ring: int, limit: float, rsp: int,
                     o_rt: int, o_gu: int, o_dn: int, o_rb: int, cp, comb_out, hc: int, iters: int, eps: float,
                     skf: int, xt_in=None, fin: int = 0, xc_in=None):
        """The MoE half's body (kernels/moe_half.py), plus the Sinkhorn's pieces at its hooks: H0 after the router's
        logits matmuls, H1 after the shared expert's gate / up matmuls, H2 after the a^T matmul, and L1 .. Ln after
        routed group k's down matmuls (it = k), n from pb_hooks. cp / comb_out / hc / iters / eps / skf:
        mhc_split.sinkhorn_body's; the last non-empty piece writes comb_out. Returns (rp, sp, i8f, pvb, ch): the
        partials, the final ids, the weights and the router's choice. xc_in: the k-major input xc [P, C, T] built by the caller (_norm_xc), in place
        of the rows' load, isolation and transposes; each `if xc_in is None:` block is the served body's verbatim."""
        f32, bf16, i32, u32 = nl.float32, nl.bfloat16, nl.int32, nl.uint32
        if xc_in is None:
            T, H = x.shape if xt_in is None else xt_in.shape
        else:  # xc_in is [P, C, T]
            T, H = xc_in.shape[2], xc_in.shape[1] * P
        E = wb.shape[0]
        C = H // P
        DW = H // 2
        C2 = DW // P
        F = wb.shape[2]
        SB = (F - H - DW) // (2 * C)  # bytes per tile scale: 2 (bf16) or 4 (fp32)
        o_dw, o_sg, o_sd = H, H + DW, H + DW + C * SB
        sdt = nl.bfloat16 if SB == 2 else nl.float32
        Q = group
        N = T * K
        NQ = -(-N // Q)
        n_hk, step = pb_hooks(NQ, 2 * iters - 1)  # the loop hooks L1 .. Ln and the piece length
        NP = NQ * Q  # pairs rounded to whole groups (the extra ones load expert 0, weight 0)
        NR = ring
        PD = NR - 2
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        assert K == KTOP and T <= P and E <= 512 and (npg == 1 or NQ >= 2)
        ih = Is // npg
        J = 2 * ih
        Eh = E // npg
        rsp_on = rsp != 0 and npg == 2
        ER = Eh if rsp_on else E  # the experts this program scores
        e0 = pid * Eh if rsp_on else 0
        o_gup = o_gu + pid * P * C * J  # this program's shared expert rows and columns
        o_dnp = o_dn + pid * ih * H
        # outputs first, the same in both programs' traces
        rp = nl.ndarray((npg, P, T, C), dtype=f32, buffer=nl.shared_hbm)
        sp = nl.ndarray((npg, T, H), dtype=f32, buffer=nl.shared_hbm)
        if rsp_on:  # allocated first, the same in both programs' traces: the peer's sendrecv lands here
            rcv = nl.ndarray((T, 3 * KTOP), dtype=f32, buffer=nl.sbuf)
        # this program's contiguous half of the pair groups (moe_tiles_pairs_v1's split)
        split = npg == 2
        q_lo, q_hi = (0, NQ) if not split else ((0, (NQ + 1) // 2) if pid == 0 else ((NQ + 1) // 2, NQ))
        NL = q_hi - q_lo

        # --- loads: the input, the router's [k, e] copy, this program's shared expert rows and columns, the bias
        if xc_in is None:
            if xt_in is None:
                xt = nl.ndarray((T, H), dtype=bf16, buffer=nl.sbuf)
                shell.dma_static(dst=xt, src=x)
            else:  # the normalised input already in SBUF, rows on the partitions
                xt = xt_in
            xs = xt  # what the transposes read: x, or (fin) x with each row's NaN / inf zeroed
            if fin:  # x's rows are the contraction of the identity matmuls below (xc, a^T, the pair sums)
                xz = nl.ndarray((T, H), dtype=f32, buffer=nl.sbuf)
                x16 = nl.ndarray((T, H), dtype=nl.uint16, buffer=nl.sbuf)
                shell.finite_or_zero(xt.view(nl.uint16), x16, xz)
                xs = x16.view(bf16)
        r_sb = nl.ndarray((P, C, ER), dtype=bf16, buffer=nl.sbuf)
        if rsp_on:
            shell.dma_static(dst=r_sb,
                             src=bb.ap(pattern=[[C * Eh, P], [Eh, C], [1, Eh]], offset=o_rt + pid * P * C * Eh))
        else:
            for q in range(npg):
                shell.dma_static(dst=r_sb[:, :, q * Eh:(q + 1) * Eh],
                                 src=bb.ap(pattern=[[C * Eh, P], [Eh, C], [1, Eh]], offset=o_rt + q * P * C * Eh))
        g_sb = nl.ndarray((P, C, J), dtype=bf16, buffer=nl.sbuf)
        shell.dma_static(dst=g_sb, src=bb.ap(pattern=[[C * J, P], [J, C], [1, J]], offset=o_gup))
        d_sb = nl.ndarray((ih, H), dtype=bf16, buffer=nl.sbuf)
        shell.dma_static(dst=d_sb, src=bb.ap(pattern=[[H, ih], [1, H]], offset=o_dnp))
        rb = nl.ndarray((T, ER), dtype=f32, buffer=nl.sbuf)
        shell.dma_static(dst=rb, src=bf.ap(pattern=[[0, T], [1, ER]], offset=o_rb + e0))

        # --- constants: the partition index, the identity [T, T] (rows 0 .. T - 1 of eye), expert ids along e
        ipi = nl.ndarray((P, 1), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ipi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        ip = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ip, src=ipi)
        jti = nl.ndarray((P, T), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=jti, pattern=[[1, T]], offset=0, channel_multiplier=0)
        eye = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eye, src=jti)
        nisa.tensor_scalar(dst=eye, data=eye, op0=nl.equal, operand0=ip)
        eyeb = nl.ndarray((P, T), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eyeb, src=eye)
        ioei = nl.ndarray((P, ER), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ioei, pattern=[[1, ER]], offset=0, channel_multiplier=0)
        ioe = nl.ndarray((P, ER), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ioe, src=ioei)

        # --- x in the k-major layout: xc[i, c, t] = x[t, c 128 + i], by matmuls with the identity (exact)
        if xc_in is None:
            xps = nl.ndarray((P, C, T), dtype=f32, buffer=nl.psum)
            for c in range(C):
                nisa.nc_matmul(dst=xps[:, c, :], stationary=xs[0:T, c * P:(c + 1) * P], moving=eyeb[0:T, 0:T],
                               accumulate=False)
            xc = nl.ndarray((P, C, T), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xc, src=xps, engine=nisa.vector_engine)
        else:  # built by the caller from the h-layout norm
            xc = xc_in

        # --- the router: logits [T, ER], sigmoid, + bias, top 8 (RSPLIT: of this program's ER, then merged), weights
        pl = nl.ndarray((T, ER), dtype=f32, buffer=nl.psum)
        for c in range(C):
            nisa.nc_matmul(dst=pl, stationary=xc[:, c, :], moving=r_sb[:, c, :], accumulate=(c > 0))
        _piece(cp, comb_out, hc, iters, eps, skf, 0, step)  # H0
        ex = nl.ndarray((T, ER), dtype=f32, buffer=nl.sbuf)
        sc = nl.ndarray((T, ER), dtype=f32, buffer=nl.sbuf)
        shell.sigmoid_exp(dst=sc, src=pl, e=ex)
        ch = nl.ndarray((T, ER), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ch, data1=sc, data2=rb, op=nl.add, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=ch, data=ch, op0=nl.maximum, operand0=-3.0e38, engine=nisa.vector_engine)
        v8 = nl.ndarray((T, KTOP), dtype=f32, buffer=nl.sbuf)
        nisa.max8(dst=v8, src=ch)
        i8 = nl.ndarray((T, KTOP), dtype=u32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=i8, data=ch, vals=v8)
        i8f = nl.ndarray((T, KTOP), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=i8f, src=i8)
        nisa.tensor_scalar(dst=i8f, data=i8f, op0=nl.maximum, operand0=0.0, op1=nl.minimum, operand1=float(ER - 1),
                           engine=nisa.vector_engine)
        tv = nl.ndarray((T, KTOP), dtype=f32, buffer=nl.sbuf)  # the chosen experts' scores: sum_e [e == id] sc[e]
        if not TV_WIDE:
            mk = nl.ndarray((T, ER), dtype=f32, buffer=nl.sbuf)
            for j in range(KTOP):
                nisa.tensor_scalar(dst=mk, data=ioe[0:T, :], op0=nl.equal, operand0=i8f[:, j:j + 1],
                                   engine=nisa.vector_engine)
                nisa.tensor_tensor(dst=mk, data1=mk, data2=sc, op=nl.multiply, engine=nisa.vector_engine)
                nisa.tensor_reduce(dst=tv[:, j:j + 1], op=nl.add, data=mk, axis=1)
        else:  # TV_WIDE: all KTOP slots at once
            mk8 = nl.ndarray((T, KTOP, ER), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=mk8, data1=ioe[0:T, :].expand_dim(1).broadcast(1, KTOP),
                               data2=i8f.expand_dim(2).broadcast(2, ER), op=nl.equal, engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=mk8, data1=mk8, data2=sc.expand_dim(1).broadcast(1, KTOP), op=nl.multiply,
                               engine=nisa.vector_engine)
            nisa.tensor_reduce(dst=tv, op=nl.add, data=mk8, axis=2)
        if rsp_on:  # this program's 8 candidates (choice, global id, score) swapped with the peer's; the top 8 of 16
            cand = nl.ndarray((T, 3 * KTOP), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cand[:, 0:KTOP], src=v8, engine=nisa.vector_engine)
            nisa.tensor_scalar(dst=cand[:, KTOP:2 * KTOP], data=i8f, op0=nl.add, operand0=float(e0),
                               engine=nisa.vector_engine)
            nisa.tensor_copy(dst=cand[:, 2 * KTOP:3 * KTOP], src=tv, engine=nisa.vector_engine)
            if rsp == 2:
                nisa.sendrecv(src=cand, dst=rcv, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                              dma_engine=nisa.dma_engine.gpsimd_dma)
            else:
                nisa.sendrecv(src=cand, dst=rcv, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0)
            mg = nl.ndarray((T, 3, 2 * KTOP), dtype=f32, buffer=nl.sbuf)  # [choice | id | score] x program 0's, 1's
            for k3 in range(3):
                if pid == 0:
                    nisa.tensor_copy(dst=mg[:, k3, 0:KTOP], src=cand[:, k3 * KTOP:(k3 + 1) * KTOP])
                    nisa.tensor_copy(dst=mg[:, k3, KTOP:2 * KTOP], src=rcv[:, k3 * KTOP:(k3 + 1) * KTOP])
                else:
                    nisa.tensor_copy(dst=mg[:, k3, 0:KTOP], src=rcv[:, k3 * KTOP:(k3 + 1) * KTOP])
                    nisa.tensor_copy(dst=mg[:, k3, KTOP:2 * KTOP], src=cand[:, k3 * KTOP:(k3 + 1) * KTOP])
            nisa.max8(dst=v8, src=mg[:, 0, :])
            p8 = nl.ndarray((T, KTOP), dtype=u32, buffer=nl.sbuf)
            nisa.nc_find_index8(dst=p8, data=mg[:, 0, :], vals=v8)
            p8f = nl.ndarray((T, KTOP), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=p8f, src=p8)
            io16i = nl.ndarray((P, 2 * KTOP), dtype=i32, buffer=nl.sbuf)
            nisa.iota(dst=io16i, pattern=[[1, 2 * KTOP]], offset=0, channel_multiplier=0)
            io16 = nl.ndarray((P, 2 * KTOP), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=io16, src=io16i)
            m16 = nl.ndarray((T, 2 * KTOP), dtype=f32, buffer=nl.sbuf)
            g16 = nl.ndarray((T, 2 * KTOP), dtype=f32, buffer=nl.sbuf)
            for j in range(KTOP):
                nisa.tensor_scalar(dst=m16, data=io16[0:T, :], op0=nl.equal, operand0=p8f[:, j:j + 1],
                                   engine=nisa.vector_engine)
                nisa.tensor_tensor(dst=g16, data1=m16, data2=mg[:, 1, :], op=nl.multiply, engine=nisa.vector_engine)
                nisa.tensor_reduce(dst=i8f[:, j:j + 1], op=nl.add, data=g16, axis=1)
                nisa.tensor_tensor(dst=g16, data1=m16, data2=mg[:, 2, :], op=nl.multiply, engine=nisa.vector_engine)
                nisa.tensor_reduce(dst=tv[:, j:j + 1], op=nl.add, data=g16, axis=1)
        ssum = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ssum, op=nl.add, data=tv, axis=1)
        nisa.tensor_scalar(dst=ssum, data=ssum, op0=nl.add, operand0=1e-20, engine=nisa.vector_engine)
        rs = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs, data=ssum)
        pv = nl.ndarray((T, KTOP), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pv, data=tv, op0=nl.multiply, operand0=rs, op1=nl.multiply, operand1=scale,
                           engine=nisa.vector_engine)
        pvb = nl.ndarray((T, KTOP), dtype=bf16, buffer=nl.sbuf)  # the graph's topv is bf16
        nisa.tensor_copy(dst=pvb, src=pv)
        # the pairs u = t K + k: ids on partition 0 (e_sb), weights on every partition (wb_), by matmuls that sum
        # M[t', t, k] = value[t', k] [t' == t] over t'
        mi = nl.ndarray((T, T, KTOP), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=mi, data1=i8f.expand_dim(1).broadcast(1, T),
                           data2=eye[0:T].expand_dim(2).broadcast(2, KTOP), op=nl.multiply)
        mv = nl.ndarray((T, T, KTOP), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=mv, data1=pvb.expand_dim(1).broadcast(1, T),
                           data2=eyeb[0:T].expand_dim(2).broadcast(2, KTOP), op=nl.multiply)
        ones_c = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ones_c, value=1.0)
        pe = nl.ndarray((1, N), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pe, stationary=ones_c[0:T, :], moving=mi.flatten_dims(1, 2), accumulate=False)
        e_sb = nl.ndarray((1, NP), dtype=i32, buffer=nl.sbuf)
        nisa.memset(dst=e_sb, value=0)
        nisa.tensor_copy(dst=e_sb[:, 0:N], src=pe)
        ones_b = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=ones_b, value=1.0)
        pwb = nl.ndarray((P, N), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pwb, stationary=ones_b[0:T, :], moving=mv.flatten_dims(1, 2), accumulate=False)
        wv = nl.ndarray((P, NP), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=wv, value=0.0)
        nisa.tensor_copy(dst=wv[:, 0:N], src=pwb)

        # --- the shared expert: this program's Is / npg intermediate rows (hybrid._swiglu, bf16 where torch rounds)
        gps = nl.ndarray((T, J), dtype=f32, buffer=nl.psum)
        for c in range(C):
            nisa.nc_matmul(dst=gps, stationary=xc[:, c, :], moving=g_sb[:, c, :], accumulate=(c > 0))
        _piece(cp, comb_out, hc, iters, eps, skf, 1, step)  # H1
        gub = nl.ndarray((T, J), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gub, src=gps, engine=nisa.vector_engine)
        gcl = nl.ndarray((T, ih), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gcl, data=gub[:, 0:ih], op0=nl.minimum, operand0=limit, engine=nisa.vector_engine)
        ucl = nl.ndarray((T, ih), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ucl, data=gub[:, ih:J], op0=nl.maximum, operand0=-limit, op1=nl.minimum,
                           operand1=limit, engine=nisa.vector_engine)
        slf = nl.ndarray((T, ih), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=slf, op=nl.silu, data=gcl)
        slb = nl.ndarray((T, ih), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=slb, src=slf, engine=nisa.vector_engine)
        ab = nl.ndarray((T, ih), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ab, data1=slb, data2=ucl, op=nl.multiply, engine=nisa.vector_engine)
        aps = nl.ndarray((ih, T), dtype=f32, buffer=nl.psum)  # a^T: the down projection's stationary
        nisa.nc_matmul(dst=aps, stationary=ab, moving=eyeb[0:T, 0:T], accumulate=False)
        _piece(cp, comb_out, hc, iters, eps, skf, 2, step)  # H2
        at = nl.ndarray((ih, T), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=at, src=aps, engine=nisa.vector_engine)
        ssb = nl.ndarray((T, H), dtype=f32, buffer=nl.sbuf)
        dps = nl.ndarray((T, 512), dtype=f32, buffer=nl.psum)
        for h0 in range(0, H, 512):
            nisa.nc_matmul(dst=dps, stationary=at, moving=d_sb[:, h0:h0 + 512], accumulate=False)
            nisa.tensor_copy(dst=ssb[:, h0:h0 + 512], src=dps, engine=nisa.vector_engine)
        shell.dma_static(dst=sp.ap(pattern=[[H, T], [1, H]], offset=pid * T * H), src=ssb)

        # --- the routed experts: moe_tiles_pairs_v1's pairs on this program's groups
        dd_i = nl.ndarray((P, P), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=dd_i, pattern=[[1, P]], offset=0, channel_multiplier=0)
        dd = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dd, src=dd_i)
        nisa.tensor_scalar(dst=dd, data=dd, op0=nl.subtract, operand0=ip)
        fd = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf)
        f1 = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf)
        f2 = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=fd, data=dd, op0=nl.equal, operand0=0.0)
        nisa.tensor_scalar(dst=f1, data=dd, op0=nl.equal, operand0=64.0)
        nisa.tensor_scalar(dst=f2, data=dd, op0=nl.equal, operand0=-64.0)
        nisa.tensor_tensor(dst=fd, data1=fd, data2=f1, op=nl.add)
        nisa.tensor_tensor(dst=fd, data1=fd, data2=f2, op=nl.add)
        mh = nl.ndarray((P, 2), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mh[:, 0:1], data=ip, op0=nl.less, operand0=64.0)
        nisa.tensor_scalar(dst=mh[:, 1:2], data=ip, op0=nl.greater_equal, operand0=64.0)
        acc = nl.ndarray((P, T, C), dtype=f32, buffer=nl.sbuf)  # [p, t, g]: g = 2 c' + half
        nisa.memset(dst=acc, value=0.0)
        bufs = (nl.ndarray((P, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((P, Q, F), dtype=nl.uint8, buffer=nl.sbuf),
                nl.ndarray((P, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((P, Q, F), dtype=nl.uint8, buffer=nl.sbuf),
                nl.ndarray((P, Q, F), dtype=nl.uint8, buffer=nl.sbuf))[:NR]
        sg_r = (nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf))
        sd_r = (nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf))
        q_r = (nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf),
               nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf))
        g_r = (nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf),
               nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf))
        mm_r = (nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf), nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf),
                nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf))
        sl_r = (nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf))
        au_r = (nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf))
        a2_r = (nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf), nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf),
                nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf))
        y_r = (nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.sbuf),
               nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.sbuf))
        pg_r = (nl.ndarray((P, Q, C), dtype=f32, buffer=nl.psum), nl.ndarray((P, Q, C), dtype=f32, buffer=nl.psum),
                nl.ndarray((P, Q, C), dtype=f32, buffer=nl.psum))
        pf_r = (nl.ndarray((P, Q, 2), dtype=f32, buffer=nl.psum), nl.ndarray((P, Q, 2), dtype=f32, buffer=nl.psum),
                nl.ndarray((P, Q, 2), dtype=f32, buffer=nl.psum))
        pd_r = (nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.psum), nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.psum),
                nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.psum))
        # step i: the loads of group i + PD, stage A of group i, stages B and C of group i - 1 (the pairs kernel's)
        for it in range(-PD, NL + 1):
            gl = it + PD
            if 0 <= gl < NL:
                bq = bufs[gl % NR]
                for s in range(Q):
                    e = e_sb.ap(pattern=[[NP, 1], [1, 1]], offset=(q_lo + gl) * Q + s)
                    if not GATHER_HW:
                        nisa.dma_copy(dst=bq[:, s, :], src=wb.select(0, e))
                    else:  # GATHER_HW: expert e's [128, F] rows, e a run-time offset on wb's axis 0
                        nisa.dma_copy(dst=bq[:, s, :], src=wb.ap(pattern=[[F, P], [1, F]], offset=0, scalar_offset=e,
                                                                indirect_dim=0), dge_mode=nisa.dge_mode.hwdge)
            if 0 <= it < NL:  # stage A: gate_up, scales
                gq = it
                bq = bufs[gq % NR]
                pg = pg_r[gq % 3]
                for s in range(Q):
                    t = min(T - 1, ((q_lo + gq) * Q + s) // K)
                    wg = bq[:, s, 0:H].view(nl.float8_e4m3)
                    for c in range(C):
                        nisa.nc_matmul(dst=pg[:, s, c:c + 1], stationary=wg[:, c * P:(c + 1) * P],
                                       moving=xc[:, c, t:t + 1], accumulate=False)
                nisa.activation(dst=sg_r[gq % 3], op=nl.copy, data=bq[:, :, o_sg:o_sd].view(sdt))
                nisa.activation(dst=sd_r[gq % 3], op=nl.copy, data=bq[:, :, o_sd:F].view(sdt))
                q = q_r[gq % 3]
                nisa.tensor_tensor(dst=q, data1=pg, data2=sg_r[gq % 3], op=nl.multiply)
                g = g_r[gq % 3]
                nisa.tensor_reduce(dst=g, op=nl.add, data=q, axis=2)
                nisa.tensor_tensor(dst=mm_r[gq % 3], data1=g.expand_dim(2).broadcast(2, 2),
                                   data2=mh.expand_dim(1).broadcast(1, Q), op=nl.multiply)
            if 1 <= it <= NL:  # stages B and C of group it - 1
                gq = it - 1
                pf = pf_r[gq % 3]
                nisa.nc_matmul(dst=pf.flatten_dims(1, 2), stationary=fd, moving=mm_r[gq % 3].flatten_dims(1, 2),
                               accumulate=False)
                sl = sl_r[gq % 3]
                au = au_r[gq % 3]
                gc = nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf)  # the clamped SwiGLU (ACTS silu_clamp)
                nisa.tensor_scalar(dst=gc, data=pf[:, :, 0], op0=nl.minimum, operand0=limit)
                uc = nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=uc, data=pf[:, :, 1], op0=nl.maximum, operand0=-limit, op1=nl.minimum,
                                   operand1=limit)
                nisa.activation(dst=sl, op=nl.silu, data=gc)
                nisa.tensor_tensor(dst=au, data1=sl, data2=uc, op=nl.multiply)
                nisa.tensor_tensor(dst=a2_r[gq % 3], data1=au.expand_dim(2).broadcast(2, 2),
                                   data2=mh.expand_dim(1).broadcast(1, Q), op=nl.multiply)
                bq = bufs[gq % NR]
                pd = pd_r[gq % 3]
                for s in range(Q):
                    wd = bq[:, s, o_dw:o_sg].view(nl.float8_e4m3)
                    for c in range(C2):
                        nisa.nc_matmul(dst=pd[:, s, c, :], stationary=wd[:, c * P:(c + 1) * P],
                                       moving=a2_r[gq % 3][:, s, :], accumulate=False)
                if it <= n_hk:  # L1 .. Ln: both programs have these groups (n_hk <= either program's NL)
                    _piece(cp, comb_out, hc, iters, eps, skf, 2 + it, step)
                y = y_r[gq % 3]
                nisa.tensor_tensor(dst=y.flatten_dims(2, 3), data1=pd.flatten_dims(2, 3), data2=sd_r[gq % 3],
                                   op=nl.multiply)
                for s in range(Q):
                    u = (q_lo + gq) * Q + s
                    if u < N:
                        t = u // K
                        nisa.scalar_tensor_tensor(dst=acc[:, t, :], data=y[:, s].flatten_dims(1, 2), op0=nl.multiply,
                                                  operand0=wv[:, u:u + 1], op1=nl.add, operand1=acc[:, t, :])
        shell.dma_static(dst=rp.ap(pattern=[[T * C, P], [C, T], [1, C]], offset=pid * P * T * C), src=acc)
        if npg > 1:  # both programs' partials written before either ends
            nisa.core_barrier(data=rp, cores=(0, 1))
            nisa.core_barrier(data=sp, cores=(0, 1))
        return rp, sp, i8f, pvb, ch

    @nki.jit
    def ffn_half_kernel(S, post_p, comb_p, Yp, fnb, base, scale, pnorm, bb, bf, wb, hc: int, iters: int,
                             eps: float, rms_eps: float, sig: int, skf: int, Is: int, K: int, rsc: float, group: int,
                             ring: int, limit: float, rsp: int, o_rt: int, o_gu: int, o_dn: int, o_rb: int, dbg: int,
                             fin: int, rev: int):
        """The FFN half (module docstring), its Sinkhorn in pieces inside the MoE body (half_body_pb). S bf16 [T, hc H]
        (the streams before the attention output's mix); post_p fp32 [T, hc], comb_p fp32 [T, hc hc] and Yp bf16 [npg,
        T, H] (that mix: shell.mix_in_h's convention); fnb / base / scale / pnorm: the FFN hyper-connection's fn (fn_j's
        j-major copy), base, scale and post_norm; bb / bf / wb: moe_half's packed blobs and the layer's w_blob; hc,
        iters, eps, rms_eps, sig, skf: mhc_pre's; Is, K, rsc (the routed scaling factor), group, ring, limit, rsp, o_*:
        moe_half's; dbg: 1 also returns (xn, ids, weights), 2 those and the router's choice (rsp 0); fin: the row
        isolation's form (rowiso); rev: this kernel's source revision with its bodies'. Returns (S_n, post, comb, rp,
        sp) [+ (xn, ids, weights) with dbg] [+ choice with dbg=2]."""
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
    ffn_half_kernel = None


# its source, its helpers' and its bodies' revisions (MH: the body half_body_pb copies), and the constant it reads, as one
# CRC of them all: as an XOR, shell's term cancelled, since MH.REV folds SH.REV too
REV = zlib.crc32(repr((SH.kernel_rev(__file__, "# -- the kernel source", "    ffn_half_kernel = None"),
                       SH.REV, MS.REV, MH.REV, RI.REV, P)).encode()) ^ SH.MIX_REV
