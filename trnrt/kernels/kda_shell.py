"""The KDA attention half of a GLM-5.3-Flash decode layer as a decode shell (all-reduce to all-reduce): ONE NKI kernel
from the residual streams to the two programs' partials of the half's output projection, with the hyper-connection
around it inside the call.

    call (S, [post_p, comb_p, Y_p]):
        mix_in: y = round_bf16(Y_p[0] + Y_p[1]);  S = mix(post_p, comb_p, y, S)      the previous block's mhc_mix
        x, post, comb = mhc_pre(S)                                                    this block's (kernels/mhc_pre.py)
        P = the KDA half on rms_norm(x, in_norm), each program's bf16 partial         (kernels/kda_half.py's body)
        returns [S (mix_in)], post, comb, P [npg, T, H], conv_pool, rec_pool
    the caller: Y = all_reduce(P) as one collective; the next block mixes it in (or the caller does, hybrid.mix_out)

What it replaces per KDA attention half: the mhc_pre kernel call, the XLA rms_norm, kda_half's call (or the ~25 XLA
ops and two kernel calls of the served path) and the glue between them; with mix_in, also the previous block's mhc_mix
call.

Layouts. Everything stays in h-layout (h = p W + j, W = H / 128), the layout kda_half reads x in: the streams S as
[128, hc, T, W]. mhc_pre's projection contracts over k = n H + h, so it runs in h-layout too, from fn laid out at load
as fnh [128, hc, W, O] (fnh[p, n, j, o] = fn[o, n H + p W + j]): hc W matmuls of [128, T] by [128, O] into one PSUM
[T, O], the same count as mhc_pre's NJ k-tiles. No layout change, no exchange: both programs compute the mix and
mhc_pre on all of S (unsplit), each its own KDA head.

Numerics against the emulation (emulate(), the served torch forms): the projections' products are exact and only their
fp32 summation order differs; sigmoids are 1 / (1 + exp(-z)) (the exp table; the mhc_pre kernel takes the scalar
engine's sigmoid); rsqrt takes one Newton step; the Sinkhorn is mhc_pre's eps form. Each program's partial is rounded
to bf16 before the all-reduce.

Weights: one bf16 blob and one fp32 blob per layer (shell.register_packed, prefix "ks_"), the offsets static int
arguments of the call (shell.packed_layout of shapes(), computed on the host).

Forms. The kernel body keeps some forms the served build does not take, behind module constants: SKL (the comb's
Sinkhorn issued late, split or serialized: scheduling probes, off), OPB (the out projection's PSUM copies over several
banks, off), and kda_oob's RCLAMP (off). Their lines end in "# SKL", "# OPB" and "# RCLAMP". The served form batches
xn's normalisation over the rows ("# XNB") except at XNB_SKIP_T rows, where the per-row form is faster.
"""

from __future__ import annotations

import os
import zlib

import torch

from . import kda_half as KH
from . import kda_oob as KO
from . import mhc_pre_split as MPS
from . import rowiso as RI
from . import shell as SH

P = 128
PREFIX = "ks_"
# the recurrence's form (a static kernel argument, so each form has its own graph key): 0 kda_half's (row, head) pair
# pipeline; 1 row-batched, stage-major (served from BATCH_MIN_T rows). recb 1 was WRONG on the device at T >= 2 until no
# vector op read two PSUM tiles: since then its device state error is recb 0's at every T (rec summed rel 1.4e-5 / 3.9e-5
# / 1.6e-5 at T = 4 / 6 / 8, both forms).
RECB = 1
# the short conv's state, read and written: 0 a row at a time (an indirect DMA and 3 hp PE transposes per row: part of
# the ~8.5-10.5 us per row per half that made the shell lose to the XLA half above four rows); 1 (served from
# BATCH_MIN_T rows) every row at once, (row, k) on B (K - 1) partitions by one indirect DMA with a vector offset each
# way (cidx: slot (K - 1) + k, computed in the graph) and 3 hp transposes in all
CBAT = 1
# the rows from which RECB / CBAT apply; below it a call keeps recb 0 / cbat 0
BATCH_MIN_T = 3
# XNB: xn = in_norm * bf16(x rx) as two vector ops over [128, W, B] (rx broadcast along W, in_norm along B) instead of
# two per row: the same fp32 multiplies and bf16 roundings, 2 B - 2 fewer instructions a call
XNB = True
# the one row count T at which the kernel keeps the per-row form (XNB cost +90.6 us over the 34 KDA layers at T 6, 2.7
# us a layer, while it gained at T 1 / 2 / 4 / 8). An int: the NKI tracer reads a frozenset global as None, and `in`
# then refuses to trace
XNB_SKIP_T = 6
# SKL (off): mHC pre's Sinkhorn (comb's ~117 vector ops a call, which nothing in this kernel reads) issued 1 "late":
# after the KDA projection's matmuls; 2 "split": the split helpers in the default order; 3 "serial": the default order
# and xn's sum of squares + an exact +0 from the Sinkhorn's last tile, so the projection waits for the whole Sinkhorn
SKL = 0
# OPB (off): the out projection's PSUM -> bf16 copies over 2 / 4 PSUM banks an ACT instruction instead of one (8 a call
# -> 4 / 2), each bank's chunk still its own lone-start matmul; only where a chunk is one matmul (hp 1)
OPB = 0


def forms(T: int, recb: int | None = None, cbat: int | None = None) -> tuple[int, int]:
    """(recb, cbat) of a call at T rows: given values win, else the knobs from BATCH_MIN_T rows on, else 0."""
    on = T >= BATCH_MIN_T
    return (int(recb) if recb is not None else (RECB if on else 0),
            int(cbat) if cbat is not None else (CBAT if on else 0))


def shapes(H: int, nv: int, K: int, npg: int, hc: int, dtype=torch.bfloat16) -> dict:
    """{name: (shape, dtype)}: a layer's packed tensors, bf16 then fp32 (insertion order is the layout)."""
    O = 2 * hc + hc * hc
    W = H // P
    s = dict(KH.arranged_shapes(H, nv, K, npg, dtype))
    s["fnh"] = ((P, hc, W, O), dtype)
    s["in_norm"] = ((H,), dtype)
    s["A_log"] = ((nv,), torch.float32)
    s["o_norm"] = ((P,), torch.float32)
    s["base"] = ((O,), torch.float32)
    s["scale"] = ((3,), torch.float32)
    s["ident"] = ((P, P), torch.float32)
    return s


def offsets(H: int, nv: int, K: int, npg: int, hc: int) -> dict:
    """{"o_<name>": element offset in its dtype's blob}: the kernel's static arguments."""
    out = {}
    for _, (_, idx) in SH.packed_layout(shapes(H, nv, K, npg, hc)).items():
        for name, (off, _) in idx.items():
            out["o_" + name] = int(off)
    return out


def arrange(wcat, conv_w, f_b, g_b, dt_bias, out_w, fn, in_norm, A_log, o_norm, base, scale, nv: int, npg: int,
            hc: int) -> dict:
    """Every packed tensor of a layer, by name (shapes()' order)."""
    t = KH.arrange(wcat, conv_w, f_b, g_b, dt_bias, out_w, nv, npg)
    t["fnh"] = SH.arrange_fn(fn, hc)
    t["in_norm"] = in_norm
    t["A_log"] = A_log.float()
    t["o_norm"] = o_norm.float()
    t["base"] = base.float()
    t["scale"] = scale.float()
    t["ident"] = torch.eye(P, dtype=torch.float32)
    return t


def pack(t: dict, H: int, nv: int, K: int, npg: int, hc: int, dtype=torch.bfloat16) -> dict:
    """{dtype: flat blob} of arrange()'s tensors (host; the layer's blobs are filled the same way, packed_views)."""
    blobs = {}
    for dt, (n, idx) in SH.packed_layout(shapes(H, nv, K, npg, hc, dtype)).items():
        b = torch.zeros(n, dtype=dt)
        for name, (off, shape) in idx.items():
            v = t[name].to(dt).reshape(-1)
            b[off:off + v.numel()] = v
        blobs[dt] = b
    return blobs


def emulate(S, fn, base, scale, iters: int, hc_eps: float, rms_eps: float, in_norm, wcat, conv_w, conv_pool, f_b,
            g_b, dt_bias, A_log, o_norm, out_w, rec_pool, slots, lb: float, npg: int, pend=None):
    """The call's arithmetic in torch, as the served forms compute it (hybrid._mhc / mix_out's elementwise forms,
    linear_attn's decode form): (S_n, post [T, hc], comb [T, hc, hc], P [npg, T, H] bf16, conv_pool', rec_pool').
    S bf16 [T, hc, H] (with pend, the streams before the previous block's mix); pend = (post_p, comb_p,
    Y_p [npg, T, H])."""
    from . import mhc_mix as MM
    from . import mhc_pre as MP

    T, hc, H = S.shape
    if pend is not None:
        post_p, comb_p, Yp = pend
        y = Yp.float().sum(0).to(torch.bfloat16)
        S = MM.emulate(post_p, comb_p, y, S).view(T, hc, H)
    x, post, comb = MP.emulate(S, fn, base, scale, iters, hc_eps, rms_eps)
    Pp, conv_o, rec_o = emulate_half(x, in_norm, wcat, conv_w, conv_pool, f_b, g_b, dt_bias, A_log, o_norm, out_w,
                                     rec_pool, slots, lb, rms_eps, npg)
    return S, post, comb, Pp, conv_o, rec_o


def emulate_half(x, in_norm, wcat, conv_w, conv_pool, f_b, g_b, dt_bias, A_log, o_norm, out_w, rec_pool, slots,
                 lb: float, rms_eps: float, npg: int, qkv=None):
    """The KDA half of emulate() from the block input x [T, H] (before its rms_norm): (P [npg, T, H] bf16, each
    program's heads' partial; the conv pool and the rec pool after the step)."""
    nv, D = A_log.shape[0], o_norm.shape[0]
    hp = nv // npg
    parts, conv_o, rec_o = [], None, None
    for p in range(npg):
        keep = torch.zeros(nv * D, dtype=out_w.dtype)
        keep[p * hp * D:(p + 1) * hp * D] = 1
        y, c, r = KH.emulate(x, in_norm, wcat, conv_w, conv_pool, f_b, g_b, dt_bias, A_log, o_norm, out_w * keep,
                             rec_pool, slots, lb, rms_eps, norm=True, qkv=qkv)
        parts.append(y)
        if p == 0:
            conv_o, rec_o = c, r
    return torch.stack(parts), conv_o, rec_o


try:  # the Neuron venv
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    F32, BF16, I32, U16 = nl.float32, nl.bfloat16, nl.int32, nl.uint16
    _sb, _ps, _dma = KH._sb, KH._ps, KH._dma

    @nki.jit
    def kda_shell_kernel(S, post_p, comb_p, Yp, bb, bf, conv_pool, rec_pool, slots, cidx, H: int, nv: int,
                              hc: int, iters: int, hc_eps: float, rms_eps: float, lb: float, mix_in: int, rev: int,
                              recb: int, cbat: int,
                              o_wp: int, o_convp: int, o_fbT: int, o_gbT: int, o_outT: int, o_fnh: int, o_in_norm: int,
                              o_dtb: int, o_A_log: int, o_o_norm: int, o_base: int, o_scale: int, o_ident: int,
                              fin: int = 1):
        """S bf16 [T, hc H]; post_p fp32 [T, hc], comb_p fp32 [T, hc hc], Yp bf16 [npg, T, H] (read only with mix_in);
        bb / bf the layer's bf16 / fp32 blobs; conv_pool bf16 [R, K - 1, 3 nv 128], rec_pool fp32 [R, nv, 128, 128]
        (updated in place, returned); slots int32 [2, T]; cidx int32 [2, T (K - 1)] the conv rows slot (K - 1) + k
        (read, write) of cbat 1, any [2, 1] for 0. fin (default 1): each row's NaN / inf zeroed before the PE
        transposes that contract over rows, so it cannot reach other rows; 0 none; 2 (served) rowiso's: xn zeroed
        (not pr) and only this program's conv history columns, by finite_or_zero3. Returns
        ([S_n bf16 [T, hc H]] with mix_in, post fp32 [T, hc], comb fp32 [T, hc hc], P bf16 [npg, T, H], conv_pool,
        rec_pool)."""
        T = S.shape[0]
        F = hc * H
        W = H // P
        O = 2 * hc + hc * hc
        R, _, D, _ = rec_pool.shape
        KC, CD = conv_pool.shape[1], conv_pool.shape[2]
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        hp = nv // npg
        h0 = pid * hp
        NT = 3 * hp
        NF = NT + 2
        HB = hp * T
        NCp = 3 * hp * P + 2 * P + hp
        jq = W // KH.JQ
        B = T
        # outputs and scratch, before anything that depends on the program
        S_out = nl.ndarray((T, F), dtype=BF16, buffer=nl.shared_hbm)
        post_out = nl.ndarray((T, hc), dtype=F32, buffer=nl.shared_hbm)
        comb_out = nl.ndarray((T, hc * hc), dtype=F32, buffer=nl.shared_hbm)
        P_out = nl.ndarray((npg, T, H), dtype=BF16, buffer=nl.shared_hbm)

        # The KDA body's weights first (the largest stream), then the hyper-connection's
        Wq = []
        for q in range(KH.JQ):
            wt = _sb((P, jq, NCp), BF16)
            _dma(dst=wt, src=bb.ap(pattern=[[W * NCp, P], [NCp, jq], [1, NCp]],
                                   offset=o_wp + pid * H * NCp + q * jq * NCp))
            Wq.append(wt)
        fw = _sb((P, hc, W, O), BF16)
        _dma(dst=fw, src=bb.ap(pattern=[[hc * W * O, P], [W * O, hc], [O, W], [1, O]], offset=o_fnh))
        ow = _sb((P, hp, H), BF16)
        _dma(dst=ow, src=bb.ap(pattern=[[nv * H, P], [H, hp], [1, H]], offset=o_outT + h0 * H))

        # The streams in h-layout [128, hc, T, W]: as given, or after the previous block's mix
        sh = _sb((P, hc, T, W), BF16)
        if mix_in == 0:
            SH.streams_h(S, sh, hc, T, H)
        else:
            SH.mix_in_h(sh, S, post_p, comb_p, Yp, S_out, hc, T, H, npg)

        # mhc_pre on them: x in h-layout [128, T, W] bf16 (kda_half's), post and comb out
        if SKL:  # SKL
            mix_c, stt_c, bt_c, pb_c = MPS.pre_a(sh, fw, bf, o_base, o_scale, post_out, hc, T, H, iters,  # SKL
                                                 hc_eps, rms_eps)  # SKL
            if SKL != 1:  # SKL
                cp_c = MPS.pre_comb(mix_c, stt_c, bt_c, comb_out, hc, T, iters, hc_eps)  # SKL
            xh = MPS.pre_b(sh, pb_c, hc, T, H)  # SKL
        else:  # SKL
            xh = SH.mhc_pre_h(sh, fw, bf, o_base, o_scale, post_out, comb_out, hc, T, H, iters, hc_eps, rms_eps)

        # ---- the KDA half (kernels/kda_half.py's body, norm 1, its tensors from the blobs) ----
        inw = _sb((P, W), BF16)
        _dma(dst=inw, src=bb.ap(pattern=[[W, P], [1, W]], offset=o_in_norm))
        sl = _sb((1, 2 * B), I32)
        _dma(dst=sl, src=slots.reshape((1, 2 * B)))
        sl, nk = KO.clamp_rows(sl, slots, B, R, P) if KO.RCLAMP else (sl, None)  # RCLAMP
        I = _sb((P, P))
        _dma(dst=I, src=bf.ap(pattern=[[P, P], [1, P]], offset=o_ident))
        IB = _sb((P, P), BF16)
        nisa.tensor_copy(dst=IB, src=I, engine=nisa.vector_engine)
        ones_c = _sb((P, 1))
        nisa.memset(dst=ones_c, value=1.0, engine=nisa.gpsimd_engine)
        ones_r = _sb((1, P))
        nisa.memset(dst=ones_r, value=1.0, engine=nisa.gpsimd_engine)
        one = _sb((1, 1))
        nisa.memset(dst=one, value=1.0, engine=nisa.gpsimd_engine)
        cw = _sb((P, NT, 4), BF16)
        _dma(dst=cw, src=bb.ap(pattern=[[NT * 4, P], [4, NT], [1, 4]], offset=o_convp + pid * P * NT * 4))
        fb = _sb((P, hp * D), BF16)
        _dma(dst=fb, src=bb.ap(pattern=[[hp * D, P], [1, hp * D]], offset=o_fbT + pid * P * hp * D))
        gb = _sb((P, hp * D), BF16)
        _dma(dst=gb, src=bb.ap(pattern=[[hp * D, P], [1, hp * D]], offset=o_gbT + pid * P * hp * D))
        dt = _sb((P, hp))
        _dma(dst=dt, src=bf.ap(pattern=[[hp, P], [1, hp]], offset=o_dtb + pid * P * hp))
        al = _sb((P, hp))
        _dma(dst=al, src=bf.ap(pattern=[[0, P], [1, hp]], offset=o_A_log + h0))
        on = _sb((P, 1))
        _dma(dst=on, src=bf.ap(pattern=[[1, P], [1, 1]], offset=o_o_norm))
        X = {"ones_c": ones_c, "ones_r": ones_r}

        # xn = in_norm * bf16(x rsqrt(mean(x^2) + eps)), j-major [128, W, B]
        xn = _sb((P, W, B), BF16)
        xf = _sb((P, W, B))
        nisa.tensor_copy(dst=xf, src=xh.ap(pattern=[[B * W, P], [1, W], [W, B]], offset=0), engine=nisa.vector_engine)
        sq = _sb((P, W, B))
        nisa.tensor_tensor(dst=sq, data1=xf, data2=xf, op=nl.multiply, engine=nisa.vector_engine)
        part2 = _sb((P, B))
        nisa.tensor_reduce(dst=part2, op=nl.add, data=sq.ap(pattern=[[W * B, P], [1, B], [B, W]], offset=0), axis=2)
        if SKL == 3:  # SKL
            MPS.wait_on(part2, cp_c, B, T, hc)  # SKL
        rx = KH._rsqrt_bcast(X, part2, B, 1.0 / H, rms_eps)
        if XNB and B != XNB_SKIP_T:  # XNB
            t1w = _sb((P, W, B), BF16)  # XNB
            nisa.tensor_tensor(dst=t1w, data1=xf, data2=rx.ap(pattern=[[B, P], [0, W], [1, B]], offset=0),  # XNB
                               op=nl.multiply, engine=nisa.vector_engine)  # XNB
            nisa.tensor_tensor(dst=xn, data1=t1w, data2=inw.ap(pattern=[[W, P], [1, W], [0, B]], offset=0),  # XNB
                               op=nl.multiply, engine=nisa.vector_engine)  # XNB
        for b in range(0 if XNB and B != XNB_SKIP_T else B):
            t1 = _sb((P, W), BF16)
            nisa.tensor_scalar(dst=t1, data=xf[:, :, b], op0=nl.multiply, operand0=rx[:, b:b + 1],
                               engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=xn[:, :, b], data1=t1, data2=inw, op=nl.multiply, engine=nisa.vector_engine)
        xs = xn  # what the projection reads: xn, or (fin 2) xn with each NaN / inf zeroed, on all 128 lanes:
        if fin == 2:  # the projection contracts over h, not the rows, so a finite xn row gives a finite pr row
            xzf = _sb((P, W, B))
            xz = _sb((P, W, B), U16)
            RI.finite_or_zero3(xn.view(U16), xz, xzf)
            xs = xz.view(BF16)

        # the fused projection, weights stationary (~1.7x the tokens-stationary rate at decode widths, the same
        # accumulation order): features on the partitions straight from the PE, ft [128, NF, B] and beta's
        # pre-activations [1, hp B] on one partition, each rounded to bf16 as the row form rounded them, with no
        # transposes. j-major, so the matmuls follow the Wq chunks as they land; one PSUM tile per accumulation group
        # (groups in slices of one tile are wrong on the device)
        pt = []
        for t in range(NF):
            pt.append(_ps((P, B)))
        pb1 = []
        for h in range(hp):
            pb1.append(_ps((1, B)))
        for j in range(W):
            wt = Wq[j // jq]
            for t in range(NF):
                nisa.nc_matmul(dst=pt[t], stationary=wt[:, j % jq, t * P:(t + 1) * P], moving=xs[:, j, :],
                               accumulate=(j > 0))
            for h in range(hp):
                nisa.nc_matmul(dst=pb1[h], stationary=wt[:, j % jq, NF * P + h:NF * P + h + 1], moving=xs[:, j, :],
                               accumulate=(j > 0))
        if SKL == 1:  # SKL
            MPS.pre_comb(mix_c, stt_c, bt_c, comb_out, hc, T, iters, hc_eps)  # SKL
        ft = _sb((P, NF, B), BF16)
        for t in range(NF):
            nisa.tensor_copy(dst=ft[:, t, :], src=pt[t], engine=nisa.vector_engine)
        bp = _sb((1, HB), BF16)
        for h in range(hp):
            nisa.tensor_copy(dst=bp[:, h * B:(h + 1) * B], src=pb1[h], engine=nisa.vector_engine)
        if fin == 1:  # each NaN / inf zeroed, as the row form zeroed them: the same values reach the rest
            fz = _sb((P, NF, B))
            fu = _sb((P, NF, B), U16)
            SH.finite_or_zero(ft.view(U16), fu, fz)
            ft = fu.view(BF16)
            bz = _sb((1, HB))
            bu = _sb((1, HB), U16)
            SH.finite_or_zero(bp.view(U16), bu, bz)
            bp = bu.view(BF16)
        bs = _sb((1, HB), BF16)
        eb = _sb((1, HB))
        KH._sigmoid(bs, bp, eb)
        NBR = _sb((1, HB))
        nisa.tensor_scalar(dst=NBR, data=bs, op0=nl.multiply, operand0=-1.0, engine=nisa.vector_engine)

        # exp(A_log) after xn's rsqrt in the scalar engine's in-order queue: A_log's DMA lands behind the Wq stream (~40
        # us), and when the compiler queued the exp first, xn's rsqrt and so the projection waited with it (the
        # projection at 41.8 / 43.2 us, against 38.7 in this order). z is +0 whatever rx holds (zeros
        # copied onto zeros, rx's bits only the predicate: a NaN row reaches nothing), and al + z differs from al only in
        # -0's sign, which exp does not see.
        z = _sb((P, 1))
        nisa.memset(dst=z, value=0.0, engine=nisa.gpsimd_engine)
        zs = _sb((P, 1))
        nisa.memset(dst=zs, value=0.0, engine=nisa.gpsimd_engine)
        nisa.tensor_copy_predicated(dst=z, src=zs, predicate=rx.view(nl.uint32)[:, 0:1])
        al2 = _sb((P, hp))
        nisa.tensor_scalar(dst=al2, data=al, op0=nl.add, operand0=z[:, 0:1], engine=nisa.vector_engine)
        rate = _sb((P, hp))
        nisa.activation(dst=rate, op=nl.exp, data=al2)
        pf = _ps((P, HB))
        pz = _ps((P, HB))
        for j in range(hp):
            nisa.nc_matmul(dst=pf[:, j * B:(j + 1) * B], stationary=fb[:, j * D:(j + 1) * D], moving=ft[:, NT, :],
                           accumulate=False)
            nisa.nc_matmul(dst=pz[:, j * B:(j + 1) * B], stationary=gb[:, j * D:(j + 1) * D], moving=ft[:, NT + 1, :],
                           accumulate=False)
        fbf = _sb((P, HB), BF16)
        nisa.tensor_copy(dst=fbf, src=pf, engine=nisa.vector_engine)
        ga = _sb((P, HB))
        for j in range(hp):
            nisa.tensor_scalar(dst=ga[:, j * B:(j + 1) * B], data=fbf[:, j * B:(j + 1) * B], op0=nl.add,
                               operand0=dt[:, j:j + 1], op1=nl.multiply, operand1=rate[:, j:j + 1],
                               engine=nisa.vector_engine)
        gs = _sb((P, HB))
        eg = _sb((P, HB))
        KH._sigmoid(gs, ga, eg)
        G = _sb((P, HB))
        nisa.tensor_scalar(dst=G, data=gs, op0=nl.multiply, operand0=lb, engine=nisa.vector_engine)
        E = _sb((P, HB))
        nisa.activation(dst=E, op=nl.exp, data=G)
        zb = _sb((P, HB), BF16)
        nisa.tensor_copy(dst=zb, src=pz, engine=nisa.vector_engine)
        sg = _sb((P, HB))
        ezz = _sb((P, HB))
        KH._sigmoid(sg, zb, ezz)
        # the short conv: each row's previous K - 1 inputs, read before any write
        prv = _sb((P, NT, B, KC), BF16)
        BK = B * KC
        if cbat == 1:  # every row at once: (b, k) on partition b KC + k, then 3 hp transposes for all of them
            ci = _sb((BK, 2), I32)
            _dma(dst=ci, src=cidx.ap(pattern=[[1, BK], [BK, 2]], offset=0))
            ci, nkc = KO.clamp_conv(ci, BK, R * KC) if KO.RCLAMP else (ci, None)  # RCLAMP
            stB = _sb((BK, CD), BF16)
            nisa.memset(dst=stB, value=0.0, engine=nisa.gpsimd_engine)
            # flat conv rows [R (K - 1), CD], so the vector offset's row stride is CD however the index is applied
            cv2 = conv_pool.reshape((R * KC, CD))
            # a vector-offset DMA is software DGE only (nki 0.6: "dma_copy_indirect only supports SWDGE"), as dsa_decode's
            nisa.dma_copy(dst=stB, src=cv2.ap(pattern=[[CD, BK], [1, CD]], offset=0,
                                              vector_offset=ci.ap(pattern=[[2, BK], [1, 1]], offset=0),
                                              indirect_dim=0), oob_mode=nisa.oob_mode.skip)
            if KO.RCLAMP == 1:  # RCLAMP
                KO.zero(stB, nkc, BK, CD, 0, 1)  # RCLAMP
            stS = stB  # (b, k) on the transposes' contraction: with fin, a row's non-finite conv history zeroed
            if fin == 1:
                cz = _sb((BK, CD))
                sz = _sb((BK, CD), U16)
                SH.finite_or_zero(stB.view(U16), sz, cz)
                stS = sz.view(BF16)
            elif fin == 2:  # only the 3 hp head blocks this program's transposes read: [BK, 3, hp 128], i-major
                cz = _sb((BK, 3, hp * P))
                sz = _sb((BK, 3, hp * P), U16)
                RI.finite_or_zero3(stB.view(U16).ap(pattern=[[CD, BK], [nv * P, 3], [1, hp * P]], offset=h0 * P), sz,
                                   cz)
                stS = sz.view(BF16)
            pcB = _ps((P, NT, BK))
            for i in range(3):
                for j in range(hp):
                    cb = (i * nv + h0 + j) * P
                    st = stS[0:BK, i, j * P:(j + 1) * P] if fin == 2 else stS[0:BK, cb:cb + P]
                    nisa.nc_matmul(dst=pcB[:, i * hp + j, :], stationary=st, moving=IB[0:BK, 0:BK], accumulate=False)
            nisa.tensor_copy(dst=prv.ap(pattern=[[NT * BK, P], [BK, NT], [1, BK]], offset=0), src=pcB,
                             engine=nisa.vector_engine)
        for b in range(B if cbat == 0 else 0):
            stc = _sb((KC, CD), BF16)
            nisa.memset(dst=stc, value=0.0, engine=nisa.gpsimd_engine)
            ridx = sl.ap(pattern=[[2 * B, 1], [1, 1]], offset=b)
            nisa.dma_copy(dst=stc, src=conv_pool.ap(pattern=[[CD, KC], [1, CD]], offset=0, scalar_offset=ridx,
                                                    indirect_dim=0), oob_mode=nisa.oob_mode.skip,
                          dge_mode=KO.HW)
            if KO.RCLAMP == 1:  # RCLAMP
                KO.zero(stc, nk, KC, CD, b, B)  # RCLAMP
            pc = _ps((P, NT, KC))
            for i in range(3):
                for j in range(hp):
                    cb = (i * nv + h0 + j) * P
                    nisa.nc_matmul(dst=pc[:, i * hp + j, :], stationary=stc[0:KC, cb:cb + P], moving=IB[0:KC, 0:KC],
                                   accumulate=False)
            nisa.tensor_copy(dst=prv[:, :, b, :], src=pc, engine=nisa.vector_engine)
        cwf = _sb((P, NT, 4))
        nisa.tensor_copy(dst=cwf, src=cw, engine=nisa.vector_engine)
        yca = _sb((P, NT, B))
        ycb2 = _sb((P, NT, B))
        for t in range(NT):
            nisa.tensor_scalar(dst=yca[:, t, :], data=prv[:, t, :, 0], op0=nl.multiply, operand0=cwf[:, t, 0:1],
                               engine=nisa.vector_engine)
            nisa.scalar_tensor_tensor(dst=ycb2[:, t, :], data=prv[:, t, :, 1], op0=nl.multiply,
                                      operand0=cwf[:, t, 1:2], op1=nl.add, operand1=yca[:, t, :])
            nisa.scalar_tensor_tensor(dst=yca[:, t, :], data=prv[:, t, :, 2], op0=nl.multiply,
                                      operand0=cwf[:, t, 2:3], op1=nl.add, operand1=ycb2[:, t, :])
            nisa.scalar_tensor_tensor(dst=ycb2[:, t, :], data=ft[:, t, :], op0=nl.multiply, operand0=cwf[:, t, 3:4],
                                      op1=nl.add, operand1=yca[:, t, :])
        ycb = _sb((P, NT, B), BF16)
        nisa.tensor_copy(dst=ycb, src=ycb2, engine=nisa.vector_engine)
        sy = _sb((P, NT, B))
        ey = _sb((P, NT, B))
        KH._sigmoid(sy, ycb, ey)
        qkv = _sb((P, NT, B), BF16)
        nisa.tensor_tensor(dst=qkv, data1=ycb, data2=sy, op=nl.multiply, engine=nisa.vector_engine)
        nw = _sb((P, NT, B, KC), BF16)
        nisa.tensor_copy(dst=nw[:, :, :, 0:KC - 1], src=prv[:, :, :, 1:KC], engine=nisa.vector_engine)
        nisa.tensor_copy(dst=nw[:, :, :, KC - 1], src=ft[:, 0:NT, :], engine=nisa.vector_engine)
        if cbat == 1:  # every row at once: [P, B KC] transposed to (b, k) on the partitions, one indirect write
            wsB = _sb((BK, 3, hp * P), BF16)
            for i in range(3):
                pw = _ps((BK, hp * P))
                for j in range(hp):
                    nisa.nc_matmul(dst=pw[:, j * P:(j + 1) * P],
                                   stationary=nw.ap(pattern=[[NT * BK, P], [1, BK]], offset=(i * hp + j) * BK),
                                   moving=IB, accumulate=False)
                nisa.tensor_copy(dst=wsB[:, i, :], src=pw, engine=nisa.vector_engine)
            nisa.dma_copy(dst=cv2.ap(pattern=[[CD, BK], [nv * P, 3], [1, hp * P]], offset=h0 * P,
                                     vector_offset=ci.ap(pattern=[[2, BK], [1, 1]], offset=1), indirect_dim=0),
                          src=wsB, oob_mode=nisa.oob_mode.skip)
        for b in range(B if cbat == 0 else 0):
            ws = _sb((KC, 3, hp * P), BF16)
            for i in range(3):
                pw = _ps((KC, hp * P))
                for j in range(hp):
                    nisa.nc_matmul(dst=pw[:, j * P:(j + 1) * P], stationary=nw[:, i * hp + j, b, :], moving=IB,
                                   accumulate=False)
                nisa.tensor_copy(dst=ws[:, i, :], src=pw, engine=nisa.vector_engine)
            widx = sl.ap(pattern=[[2 * B, 1], [1, 1]], offset=B + b)
            nisa.dma_copy(dst=conv_pool.ap(pattern=[[CD, KC], [nv * P, 3], [1, hp * P]], offset=h0 * P,
                                           scalar_offset=widx, indirect_dim=0), src=ws, oob_mode=nisa.oob_mode.skip,
                          dge_mode=KO.HW)

        qf = _sb((P, HB))
        nisa.tensor_copy(dst=qf, src=qkv.ap(pattern=[[NT * B, P], [1, HB]], offset=0), engine=nisa.vector_engine)
        kf = _sb((P, HB))
        nisa.tensor_copy(dst=kf, src=qkv.ap(pattern=[[NT * B, P], [1, HB]], offset=HB), engine=nisa.vector_engine)
        NVT = _sb((P, HB))
        nisa.tensor_scalar(dst=NVT, data=qkv.ap(pattern=[[NT * B, P], [1, HB]], offset=2 * HB), op0=nl.multiply,
                           operand0=-1.0, engine=nisa.vector_engine)
        sq2 = _sb((P, 2 * HB))
        nisa.tensor_tensor(dst=sq2[:, 0:HB], data1=qf, data2=qf, op=nl.multiply, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=sq2[:, HB:2 * HB], data1=kf, data2=kf, op=nl.multiply, engine=nisa.vector_engine)
        rqk = KH._rsqrt_bcast(X, sq2, 2 * HB, 1.0, 1e-6)
        qn = _sb((P, HB))
        nisa.tensor_tensor(dst=qn, data1=qf, data2=rqk[:, 0:HB], op=nl.multiply, engine=nisa.vector_engine)
        QS = _sb((P, HB))
        nisa.tensor_scalar(dst=QS, data=qn, op0=nl.multiply, operand0=float(D) ** -0.5, engine=nisa.vector_engine)
        KT = _sb((P, HB))
        nisa.tensor_tensor(dst=KT, data1=kf, data2=rqk[:, HB:2 * HB], op=nl.multiply, engine=nisa.vector_engine)
        QE = _sb((P, HB))
        nisa.tensor_tensor(dst=QE, data1=QS, data2=E, op=nl.multiply, engine=nisa.vector_engine)
        KE = _sb((P, HB))
        nisa.tensor_tensor(dst=KE, data1=KT, data2=E, op=nl.multiply, engine=nisa.vector_engine)
        qkp = _sb((P, HB))
        nisa.tensor_tensor(dst=qkp, data1=QS, data2=KT, op=nl.multiply, engine=nisa.vector_engine)
        pq = _ps((1, HB))
        nisa.nc_matmul(dst=pq, stationary=ones_c, moving=qkp, accumulate=False)
        QK = _sb((1, HB))
        nisa.tensor_copy(dst=QK, src=pq, engine=nisa.vector_engine)

        POT = _ps((P, HB))  # o as columns, head-major (j B + b): both forms write it
        if recb == 0:  # kernels/kda_half.py's pipeline: (row, head) pairs one after another, two stages skewed
            Sr, Sn, Or = [], [], []
            ring = KH.ring_rows(hp)
            for _ in range(ring):
                Sr.append(_sb((P, hp, D)))
                Sn.append(_sb((P, hp, D)))
                Or.append(_sb((1, hp, D)))
            Pr, Po, Dl, Kr = [], [], [], []
            for _ in range(KH.HR):
                Pr.append(_ps((1, 3, D)))
                Po.append(_ps((P, D)))
                Dl.append(_sb((1, D)))
                Kr.append(_sb((1, D)))
            X = {"pool": rec_pool, "sl": sl, "Sr": Sr, "Sn": Sn, "Or": Or, "Pr": Pr, "Po": Po, "Dl": Dl, "Kr": Kr,
                 "QE": QE, "KE": KE, "KT": KT, "NVT": NVT, "E": E, "I": I, "NBR": NBR, "QK": QK, "POT": POT, "one": one,
                 "B": B, "hp": hp, "D": D, "h0": h0, "ring": ring, "ones_c": ones_c, "ones_r": ones_r}
            X["nk"] = nk  # RCLAMP
            N = B * hp
            for b in range(min(KH.PD, B)):
                KH._load(X, b)
            for c in range(N + KH.SKEW):
                if c < N and c % hp == 0 and c // hp + KH.PD < B:
                    KH._load(X, c // hp + KH.PD)
                if c < N:
                    KH._stage_a(X, c)
                if c >= KH.SKEW:
                    KH._stage_b(X, c - KH.SKEW)
        else:
            # Row-batched, stage-major (the pair pipeline grows ~5 us a row): pairs in head-major
            # columns (col = j B + b), in groups of G; per group the PE's r0 / r1 / -v / k rows for every pair back to
            # back, delta, k and o for the group in one vector op each (r1 - v in one more), the outer products back to
            # back, and the state update diag(e) S + k delta^T in two vector ops. The same fp32 operations per element
            # as the pipeline (activation copies and fused multiply-adds as separate vector ops), so the two forms
            # agree bit for bit.
            # Every PE write into a slice of a group's shared PSUM tile is a lone start (accumulate=False): G
            # accumulation groups in slices of one tile ("form C", r1 then += -v per slice) are exact in the
            # simulator and wrong on the device (a forms probe: rel 0.79 at G = 2;
            # recb 1's state 3.2e-3 off at T = 2). r1 and -v go to two tiles and the vector engine adds them.
            G = 4
            N = B * hp
            Sall = _sb((P, N, D))
            nisa.memset(dst=Sall, value=0.0, engine=nisa.gpsimd_engine)
            for col in range(N):
                ridx = sl.ap(pattern=[[2 * B, 1], [1, 1]], offset=col % B)
                nisa.dma_copy(dst=Sall[:, col, :], src=rec_pool.ap(pattern=[[D, P], [1, D]],
                                                                   offset=(h0 + col // B) * D * D, scalar_offset=ridx,
                                                                   indirect_dim=0),
                              oob_mode=nisa.oob_mode.skip, dge_mode=KO.HW)
                if KO.RCLAMP == 1:  # RCLAMP
                    KO.zero(Sall[:, col, :], nk, P, D, col % B, B)  # RCLAMP
            Snew = _sb((P, N, D))
            Orow = _sb((1, N, D))
            for g0 in range(0, N, G):
                gn = min(G, N - g0)
                pa = _ps((1, G, D))
                pb = _ps((1, G, D))
                pv = _ps((1, G, D))
                pk = _ps((1, G, D))
                for i in range(gn):
                    col = g0 + i
                    nisa.nc_matmul(dst=pa[:, i, :], stationary=QE[:, col:col + 1], moving=Sall[:, col, :],
                                   accumulate=False)
                    nisa.nc_matmul(dst=pb[:, i, :], stationary=KE[:, col:col + 1], moving=Sall[:, col, :],
                                   accumulate=False)
                    nisa.nc_matmul(dst=pv[:, i, :], stationary=NVT[:, col:col + 1], moving=I, accumulate=False)
                    nisa.nc_matmul(dst=pk[:, i, :], stationary=KT[:, col:col + 1], moving=I, accumulate=False)
                # A vector op may not read two PSUM operands ("data1 and data2 cannot be in PSUM simultaneously"),
                # so -v goes to SBUF first: an fp32 -> fp32 copy, exact.
                vs = _sb((1, G, D))
                nisa.tensor_copy(dst=vs[:, 0:gn, :], src=pv[:, 0:gn, :], engine=nisa.vector_engine)
                rv = _sb((1, G, D))  # r1 - v
                nisa.tensor_tensor(dst=rv[:, 0:gn, :], data1=pb[:, 0:gn, :], data2=vs[:, 0:gn, :], op=nl.add,
                                   engine=nisa.vector_engine)
                dl = _sb((1, G, D))  # delta = -beta (r1 - v)
                nisa.tensor_tensor(dst=dl[:, 0:gn, :], data1=rv[:, 0:gn, :],
                                   data2=NBR.ap(pattern=[[HB, 1], [1, gn], [0, D]], offset=g0), op=nl.multiply,
                                   engine=nisa.vector_engine)
                kr = _sb((1, G, D))
                nisa.tensor_copy(dst=kr[:, 0:gn, :], src=pk[:, 0:gn, :], engine=nisa.vector_engine)
                tq = _sb((1, G, D))  # o = delta (q . k) + r0
                nisa.tensor_tensor(dst=tq[:, 0:gn, :], data1=dl[:, 0:gn, :],
                                   data2=QK.ap(pattern=[[HB, 1], [1, gn], [0, D]], offset=g0), op=nl.multiply,
                                   engine=nisa.vector_engine)
                nisa.tensor_tensor(dst=Orow[:, g0:g0 + gn, :], data1=tq[:, 0:gn, :], data2=pa[:, 0:gn, :], op=nl.add,
                                   engine=nisa.vector_engine)
                po = _ps((P, G, D))  # k delta^T per pair (K = 1)
                for i in range(gn):
                    nisa.nc_matmul(dst=po[:, i, :], stationary=kr[:, i, :], moving=dl[:, i, :], accumulate=False)
                se = _sb((P, G, D))  # S' = diag(e) S + k delta^T
                nisa.tensor_tensor(dst=se[:, 0:gn, :], data1=Sall[:, g0:g0 + gn, :],
                                   data2=E.ap(pattern=[[HB, P], [1, gn], [0, D]], offset=g0), op=nl.multiply,
                                   engine=nisa.vector_engine)
                nisa.tensor_tensor(dst=Snew[:, g0:g0 + gn, :], data1=se[:, 0:gn, :], data2=po[:, 0:gn, :],
                                   op=nl.add, engine=nisa.vector_engine)
                for i in range(gn):
                    col = g0 + i
                    nisa.nc_matmul(dst=POT[:, col:col + 1], stationary=Orow[:, col, :], moving=one, accumulate=False)
                    widx = sl.ap(pattern=[[2 * B, 1], [1, 1]], offset=B + col % B)
                    nisa.dma_copy(dst=rec_pool.ap(pattern=[[D, P], [1, D]], offset=(h0 + col // B) * D * D,
                                                  scalar_offset=widx, indirect_dim=0), src=Snew[:, col, :],
                                  oob_mode=nisa.oob_mode.skip, dge_mode=KO.HW)

        obf = _sb((P, HB), BF16)
        nisa.tensor_copy(dst=obf, src=POT, engine=nisa.vector_engine)
        of = _sb((P, HB))
        nisa.tensor_copy(dst=of, src=obf, engine=nisa.vector_engine)
        sqo = _sb((P, HB))
        nisa.tensor_tensor(dst=sqo, data1=of, data2=of, op=nl.multiply, engine=nisa.vector_engine)
        ro = KH._rsqrt_bcast(X, sqo, HB, 1.0 / D, rms_eps)
        y1 = _sb((P, HB))
        nisa.tensor_tensor(dst=y1, data1=of, data2=ro, op=nl.multiply, engine=nisa.vector_engine)
        y2 = _sb((P, HB))
        nisa.tensor_scalar(dst=y2, data=y1, op0=nl.multiply, operand0=on, engine=nisa.vector_engine)
        ogT = _sb((P, HB), BF16)
        nisa.tensor_tensor(dst=ogT, data1=y2, data2=sg, op=nl.multiply, engine=nisa.vector_engine)

        # the output projection over this program's heads, every column; its partial rounded to bf16 into P_out[pid]
        ysb = _sb((B, H), BF16)
        opb = OPB if hp == 1 and OPB > 1 and (H // 512) % OPB == 0 else 1  # OPB
        if opb > 1:  # OPB
            for n0 in range(0, H // 512, opb):  # OPB
                pyw = _ps((P, opb * 512))  # OPB
                for n in range(n0, n0 + opb):  # OPB
                    nisa.nc_matmul(dst=pyw[0:B, (n - n0) * 512:(n - n0 + 1) * 512], stationary=ogT[:, 0:B],  # OPB
                                   moving=ow[:, 0, n * 512:(n + 1) * 512], accumulate=False)  # OPB
                nisa.activation(dst=ysb[:, n0 * 512:(n0 + opb) * 512], op=nl.copy, data=pyw[0:B, :])  # OPB
        for n in range(0 if opb > 1 else H // 512):
            py = _ps((P, 512))
            for j in range(hp):
                nisa.nc_matmul(dst=py[0:B, :], stationary=ogT[:, j * B:(j + 1) * B],
                               moving=ow[:, j, n * 512:(n + 1) * 512], accumulate=(j > 0))
            # the 4-lane PSUM -> bf16 copy on the scalar engine (the vector engine carries the critical path, ACT idles)
            nisa.activation(dst=ysb[:, n * 512:(n + 1) * 512], op=nl.copy, data=py[0:B, :])
        _dma(dst=P_out.ap(pattern=[[H, B], [1, H]], offset=pid * B * H), src=ysb)
        if npg > 1:  # every shared output written by both programs before either ends
            nisa.core_barrier(data=P_out, cores=(0, 1))
            nisa.core_barrier(data=conv_pool, cores=(0, 1))
            nisa.core_barrier(data=rec_pool, cores=(0, 1))
            nisa.core_barrier(data=post_out, cores=(0, 1))
            nisa.core_barrier(data=comb_out, cores=(0, 1))
            if mix_in:
                nisa.core_barrier(data=S_out, cores=(0, 1))
        if mix_in:
            return S_out, post_out, comb_out, P_out, conv_pool, rec_pool
        return post_out, comb_out, P_out, conv_pool, rec_pool
else:
    kda_shell_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of the kernel's source span (the compile-cache key does not hash NKI kernel source)."""
    src = open(os.path.abspath(__file__)).read()
    a = src.index("try:  # the Neuron venv")
    return zlib.crc32(src[a:src.index("    kda_shell_kernel = None", a)].encode())


# the kernel's source, its helpers' revisions (kda_half's, shell's, rowiso's) and the constant it reads, as one CRC (an
# XOR cancels a term that two parts both fold: kernels/ffn_half.py); mix_in_h's served form (shell.MIX_REV) XORed in
REV = zlib.crc32(repr((_kernel_rev(), KH.REV, SH.REV, RI.REV, P)).encode()) ^ SH.MIX_REV


def run(S, blobs: dict, conv_pool, rec_pool, slots, *, H: int, nv: int, hc: int, iters: int, hc_eps: float,
        rms_eps: float, lb: float, pend=None, simulate: bool = False, grid: int | None = None, recb: int | None = None,
        cbat: int | None = None, fin: int | None = None, build_only: bool = False):
    """The kernel in the caller's graph on the device (or in the NKI simulator): (S_n or None, post [T, hc],
    comb [T, hc, hc], P [npg, T, H] bf16, conv_pool, rec_pool). S bf16 [T, hc, H]; blobs {dtype: flat blob} packed for
    this grid's program count; pend = (post_p, comb_p, Y_p) to mix the previous block's output in first. fin None:
    the served form (rowiso.FIN). build_only: the kernel's arguments, nothing run."""
    if fin is None:
        fin = RI.FIN
    T = S.shape[0]
    rev = REV
    if kda_shell_kernel is None:
        raise RuntimeError("the NKI KDA shell needs the nki package (the Neuron venv)")
    if grid is None:
        from .. import platform

        grid = 1 if simulate else platform.nki_grid()
    K = conv_pool.shape[1] + 1
    dev = S.device
    if pend is None:
        pend_t = (torch.zeros(T, hc, dtype=torch.float32, device=dev), torch.zeros(T, hc * hc, dtype=torch.float32,
                  device=dev), torch.zeros(grid, T, H, dtype=torch.bfloat16, device=dev))
    else:
        pend_t = (pend[0].float(), pend[1].float().reshape(T, hc * hc), pend[2].to(torch.bfloat16))
    recb, cbat = forms(T, recb, cbat)
    if cbat:  # the conv rows slot (K - 1) + k of every row, read and write: [2, T (K - 1)]
        cidx = (slots.to(torch.int32).reshape(2, T, 1) * (K - 1)
                + torch.arange(K - 1, dtype=torch.int32, device=slots.device)).reshape(2, T * (K - 1))
    else:
        cidx = torch.zeros(2, 1, dtype=torch.int32, device=slots.device)
    args = dict(S=S.to(torch.bfloat16).reshape(T, hc * H), post_p=pend_t[0], comb_p=pend_t[1], Yp=pend_t[2],
                bb=blobs[torch.bfloat16], bf=blobs[torch.float32], conv_pool=conv_pool, rec_pool=rec_pool, slots=slots,
                cidx=cidx, H=int(H), nv=int(nv), hc=int(hc), iters=int(iters), hc_eps=float(hc_eps),
                rms_eps=float(rms_eps), lb=float(lb), mix_in=int(pend is not None), rev=rev,
                recb=recb, cbat=cbat, fin=int(fin), **offsets(H, nv, K, grid, hc))
    k = kda_shell_kernel
    if build_only:
        return args
    if simulate:
        import ml_dtypes
        import numpy as np

        def to_np(t):
            if not isinstance(t, torch.Tensor):
                return t
            return t.float().numpy().astype(ml_dtypes.bfloat16) if t.dtype == torch.bfloat16 else t.numpy()

        outs = nki.simulate(k[grid] if grid > 1 else k)(**{n: to_np(t) for n, t in args.items()})

        def back(o, bf16):
            return torch.from_numpy(np.asarray(o).astype(np.float32)).to(torch.bfloat16 if bf16 else torch.float32)

        cbf = conv_pool.dtype == torch.bfloat16
        if pend is not None:
            Sn, post, comb, Pp, cpo, rpo = outs
            Sn = back(Sn, True).view(T, hc, H)
        else:
            (post, comb, Pp, cpo, rpo), Sn = outs, None
        return (Sn, back(post, False), back(comb, False).view(T, hc, hc), back(Pp, True), back(cpo, cbf),
                back(rpo, False))
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    outs = wrap_nki(k)[grid](**args)
    if pend is not None:
        Sn, post, comb, Pp, cpo, rpo = outs
        return Sn.view(T, hc, H), post, comb.view(T, hc, hc), Pp, cpo, rpo
    post, comb, Pp, cpo, rpo = outs
    return None, post, comb.view(T, hc, hc), Pp, cpo, rpo


# -- in the model's decode graph (models/hybrid.py) ------------------------------------------------------------------

def register(layer, cfg, spec, npg: int) -> bool:
    """A KDA layer's two packed decode-only blobs (ks_blob_bf16 / ks_blob_f32, shell.register_packed), when the kernel
    takes its spec (kda_half.takes_spec); layer.ks_dims keeps (H, nv, K, npg, hc) for fill() and decode()."""
    H, hc = cfg.hidden_size, cfg.hybrid.hc
    if not KH.takes_spec(spec, H, layer.nk, layer.nv, npg):
        return False
    layer.ks_dims = (H, layer.nv, spec.conv_kernel, npg, hc)
    SH.register_packed(layer, PREFIX, shapes(*layer.ks_dims, dtype=layer.conv_w.dtype))
    return True


def _weight(layer, n: str) -> torch.Tensor:
    """The layer's KDA weight n dequantized as model._w reads it (GLM-5.3-Flash keeps them bf16)."""
    from ..models.quant import dequant

    sc = getattr(layer, n + "_scale", None)
    return dequant(getattr(layer, n).data, None if sc is None else sc.data, layer.conv_w.dtype)


def fill(layer) -> None:
    """The blobs from the layer's loaded weights (hybrid.load_layer, after the mixer's and the hyper-connection's):
    arrange() of the KDA half's (dequantized as model._w reads them) and the attention block's hc_attn_*."""
    dt = layer.conv_w.dtype

    def w(n):
        return _weight(layer, n)

    H, nv, K, npg, hc = layer.ks_dims
    wcat = torch.cat([w("in_qkv"), layer.f_a.data, layer.g_a.data, w("in_b")])
    t = arrange(wcat, layer.conv_w.data, layer.f_b.data, layer.g_b.data, layer.dt_bias.data, w("out"),
                layer.hc_attn_fn.data, layer.in_norm.data, layer.A_log.data, layer.o_norm.data, layer.hc_attn_base.data,
                layer.hc_attn_scale.data, nv, npg, hc)
    for name, v in SH.packed_views(layer, PREFIX, shapes(*layer.ks_dims, dtype=dt)).items():
        v.copy_(t[name].to(v.dtype).reshape(v.shape))


def _host(model, layer, S, slots, pend):
    """decode()'s call on host tensors (models/hybrid.py SHELL_HOST, tests): emulate() from the layer's own tensors,
    with the state pools updated in place as the kernel updates them. (S_n, post, comb, P [npg, T, H])."""
    hy = model.cfg.hybrid
    wcat = torch.cat([_weight(layer, "in_qkv"), layer.f_a.data, layer.g_a.data, _weight(layer, "in_b")])
    Sn, post, comb, Pp, conv_o, rec_o = emulate(S, layer.hc_attn_fn, layer.hc_attn_base, layer.hc_attn_scale,
                                                hy.sinkhorn_iters, hy.hc_eps, model.cfg.rms_norm_eps, layer.in_norm,
                                                wcat, layer.conv_w, layer.conv_state, layer.f_b, layer.g_b,
                                                layer.dt_bias, layer.A_log, layer.o_norm, _weight(layer, "out"),
                                                layer.rec_state, slots, layer.spec.lower_bound, layer.ks_dims[3], pend)
    layer.conv_state.copy_(conv_o)
    layer.rec_state.copy_(rec_o)
    return Sn, post, comb, Pp


def decode(model, layer, S, positions, state_slot, pend=None):
    """The attention block of a KDA layer's decode call as one shell call: (the streams this block reads [T, hc, H],
    (post, comb, Y)), Y [npg, T, H] the partials all-reduced as one collective. The state pools are updated in place; a
    row at position 0 starts from zero states (read row R). pend = (post_p, comb_p, Y_p [npg, T, H]): the previous
    block's output, mixed into S first; the streams returned are then the mixed ones."""
    hy = model.cfg.hybrid
    H, nv, K, npg, hc = layer.ks_dims
    R = layer.rec_state.shape[0]
    slots = SH.slots(state_slot, positions, SH.start_row(R))
    from ..models import hybrid

    if hybrid.SHELL_HOST and S.device.type == "cpu":  # tests: the call's arithmetic on the host
        Sn, post, comb, Pp = _host(model, layer, S, slots, pend)
        return Sn, (post, comb, model._attn_all_reduce(Pp))
    blobs = {torch.bfloat16: getattr(layer, PREFIX + "blob_bf16"), torch.float32: getattr(layer, PREFIX + "blob_f32")}
    Sn, post, comb, Pp, _, _ = run(S, blobs, layer.conv_state, layer.rec_state, slots, H=H, nv=nv, hc=hc,
                                   iters=hy.sinkhorn_iters, hc_eps=hy.hc_eps, rms_eps=model.cfg.rms_norm_eps,
                                   lb=layer.spec.lower_bound, pend=pend, grid=npg, fin=RI.FIN)
    return (S if Sn is None else Sn), (post, comb, model._attn_all_reduce(Pp))
