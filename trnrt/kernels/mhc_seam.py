"""One mHC seam as one NKI kernel, in decode calls: the previous block's output mix (sublayer k's post-mix) fused with
this layer's attention-block pre (sublayer k + 1's norm and pre), for a layer whose attention block runs without the
KDA shell (models/hybrid.py MHC_SEAM, EngineConfig.mhc_seam):

    S' = mix_out(post, comb, y, S)                    (kernels/mix_wide.py's wide-cast form, op for op: S' is its bits)
    x', post', comb' = mhc_pre(S', fn, base, scale)   (mhc_split.front_body + sinkhorn_body at the served switches:
                                                       the scalar engine's sigmoid, the eps Sinkhorn, pre broadcast on
                                                       chip)

What changes against the separate calls (mhc_mix, then mhc_pre):
- S is loaded once. S' never leaves SBUF between the mix and the pre: the separate pre loads S twice (k-layout for the
  projection, h-layout for the pre-mix), so a seam read S three times.
- The rows split over the LNC pair (program p takes rows [p T / 2, (p + 1) T / 2)): the separate mHC kernels run
  unsplit, both physical cores computing every row, so each core's element work halves.
- The projection contracts S' in h-layout (h = p W + j), so fn is re-laid at load to fn_w's [P, hc W, O]
  ([p, n W + j, o] = fn[o, n H + p W + j]) instead of k-layout; 128 matmuls either way.
- The coefficient path (sum of squares, rsqrt with one Newton step, sigmoid / softmax, the padded Sinkhorn) keeps a row
  per partition, as the separate front does.

Not bitwise against the separate pre: the sum of squares and the projection add in another order (h-layout partials),
so r, the mix logits and everything after them differ in fp32 ulps; S' is bitwise mix_wide's. `plant` (0 served) is a
test's must-fail control: 1 one Sinkhorn iteration short; 2 comb transposed in the post-mix; 3 silu for pre's sigmoid;
4 the last stream dropped from the post-mix.
"""

from __future__ import annotations

import torch

from . import mhc_pre as MP
from . import shell as SH

P = 128


def takes(device: torch.device, T: int, H: int, hc: int) -> bool:
    """The kernel's shapes: a NeuronCore, T even (each program its half of the rows) up to 128, H a multiple of 128."""
    return device.type != "cpu" and mhc_seam_kernel is not None and T % 2 == 0 and 2 <= T <= 128 and H % P == 0


def seam(post: torch.Tensor, comb: torch.Tensor, y: torch.Tensor, S: torch.Tensor, fnw: torch.Tensor, fn: torch.Tensor,
         base: torch.Tensor, scale: torch.Tensor, iters: int, eps: float, rms_eps: float):
    """(S' [T, hc H] bf16, x [T, H] bf16, post' [T, hc] fp32, comb' [T, hc, hc] fp32) for streams S [T, hc, H], the
    previous block's (post [T, hc], comb [T, hc, hc], y [T, H]) and this block's pre weights (fnw = fn_w(fn)): the kernel
    on the device, emulate() on the host."""
    T, hc, H = S.shape
    if S.device.type == "cpu":
        S2, x, p2, c2 = emulate(S.reshape(T, hc * H).to(torch.bfloat16), y.to(torch.bfloat16), post.float(),
                                comb.float().reshape(T, hc * hc), fn, base, scale, iters, eps, rms_eps)
        return S2, x, p2, c2.view(T, hc, hc)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    S2, x, p2, c2 = wrap_nki(mhc_seam_kernel)[platform.nki_grid()](
        S=S.to(torch.bfloat16).reshape(T, hc * H).contiguous(), y=y.to(torch.bfloat16).contiguous(),
        post=post.float().contiguous(), comb=comb.float().reshape(T, hc * hc).contiguous(),
        fnw=fnw.to(torch.bfloat16).contiguous(), base=base.float().contiguous(), scale=scale.float().contiguous(),
        hc=int(hc), iters=int(iters), eps=float(eps), rms_eps=float(rms_eps), plant=0, rev=REV)
    return S2, x, p2, c2.view(T, hc, hc)


def fn_w(fn: torch.Tensor, hc: int) -> torch.Tensor:
    """The load-time copy: fn [O, hc H] -> [P, hc W, O], [p, n W + j, o] = fn[o, n H + p W + j] (W = H / 128)."""
    O, F = fn.shape
    H = F // hc
    return fn.reshape(O, hc, P, H // P).permute(2, 1, 3, 0).reshape(P, F // P, O).contiguous()


def emulate(S: torch.Tensor, y: torch.Tensor, post: torch.Tensor, comb: torch.Tensor, fn: torch.Tensor,
            base: torch.Tensor, scale: torch.Tensor, iters: int, eps: float, rms_eps: float):
    """The seam on the host: mix_wide's wide-cast mix (fp32, m in order, one RNE cast), then mhc_pre.emulate on S'.
    S [T, hc H] bf16, y [T, H] bf16, post [T, hc], comb [T, hc hc] -> (S' [T, hc H] bf16, x [T, H] bf16, post' [T, hc],
    comb' [T, hc hc])."""
    T, hc = post.shape
    H = y.shape[1]
    Sf = S.view(T, hc, H).float()
    yf = y.float()
    cm = comb.view(T, hc, hc).float()
    outs = []
    for n in range(hc):
        o = post[:, n:n + 1].float() * yf
        for m in range(hc):
            o = o + cm[:, m, n:n + 1] * Sf[:, m]
        outs.append(o.to(torch.bfloat16))
    S2 = torch.stack(outs, 1)
    x, post2, comb2 = MP.emulate(S2, fn, base, scale, iters, eps, rms_eps)
    return S2.reshape(T, hc * H), x, post2, comb2.reshape(T, hc * hc)


try:  # the Neuron venv
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def mhc_seam_kernel(S, y, post, comb, fnw, base, scale, hc: int, iters: int, eps: float, rms_eps: float,
                        plant: int, rev: int):
        """S bf16 [T, hc H], y bf16 [T, H], post fp32 [T, hc], comb fp32 [T, hc hc] (the previous pre's), fnw bf16
        [P, hc W, O] (fn_w), base fp32 [O], scale fp32 [3]. Returns (S' bf16 [T, hc H], x' bf16 [T, H], post' fp32
        [T, hc], comb' fp32 [T, hc hc]). T even at grid 2 (each program its half of the rows). plant (0 served): a
        planted error a test must see (module docstring)."""
        f32, bf16 = nl.float32, nl.bfloat16
        VE = nisa.vector_engine
        TT, F = S.shape
        H = F // hc
        W = H // P
        KW = hc * W
        O = fnw.shape[2]
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        T, r0 = TT, 0
        if npg == 2:
            assert TT % 2 == 0
            T = TT // 2
            r0 = pid * T
        S_out = nl.ndarray((TT, F), dtype=bf16, buffer=nl.shared_hbm)
        x_out = nl.ndarray((TT, H), dtype=bf16, buffer=nl.shared_hbm)
        post_out = nl.ndarray((TT, hc), dtype=f32, buffer=nl.shared_hbm)
        comb_out = nl.ndarray((TT, hc * hc), dtype=f32, buffer=nl.shared_hbm)
        # --- loads: this program's rows, h-layout (h = p W + j); post / comb on every partition (stride 0)
        pb0 = nl.ndarray((P, T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=pb0, src=post.ap(pattern=[[0, P], [hc, T], [1, hc]], offset=r0 * hc))
        cb0 = nl.ndarray((P, T, hc * hc), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cb0, src=comb.ap(pattern=[[0, P], [hc * hc, T], [1, hc * hc]], offset=r0 * hc * hc))
        ybf = nl.ndarray((P, T, W), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=ybf, src=y.ap(pattern=[[W, P], [H, T], [1, W]], offset=r0 * H))
        st0 = nl.ndarray((P, hc, T, W), dtype=bf16, buffer=nl.sbuf)
        for n in range(hc):
            nisa.dma_copy(dst=st0[:, n, :, :], src=S.ap(pattern=[[W, P], [F, T], [1, W]], offset=r0 * F + n * H))
        wt = nl.ndarray((P, KW, O), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=fnw)
        # --- the post-mix (mix_wide._mix, cast 1): o[p, n, t, j] = y post_n, then + S_m comb[m, n], m in order; one cast
        o = nl.ndarray((P, hc, T, W), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o, data1=ybf.ap(pattern=[[T * W, P], [0, hc], [W, T], [1, W]], offset=0),
                           data2=pb0.ap(pattern=[[T * hc, P], [1, hc], [hc, T], [0, W]], offset=0),
                           op=nl.multiply, engine=VE)
        for m in range(hc):
            if plant == 4 and m == hc - 1:
                continue
            if plant == 2:  # comb[n, m] read for comb[m, n]
                cbm = cb0.ap(pattern=[[T * hc * hc, P], [hc, hc], [hc * hc, T], [0, W]], offset=m)
            else:
                cbm = cb0.ap(pattern=[[T * hc * hc, P], [1, hc], [hc * hc, T], [0, W]], offset=m * hc)
            pr = nl.ndarray((P, hc, T, W), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pr, data1=st0.ap(pattern=[[hc * T * W, P], [0, hc], [W, T], [1, W]], offset=m * T * W),
                               data2=cbm, op=nl.multiply, engine=VE)
            nisa.tensor_tensor(dst=o, data1=o, data2=pr, op=nl.add, engine=VE)
        sb = nl.ndarray((P, hc, T, W), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sb, src=o, engine=VE)
        for n in range(hc):
            nisa.dma_copy(dst=S_out.ap(pattern=[[W, P], [F, T], [1, W]], offset=r0 * F + n * H), src=sb[:, n, :, :])
        # --- the projection on S' from SBUF: a (stream, j)-major copy, so each matmul's stationary is a contiguous [P, T]
        sbj = nl.ndarray((P, KW, T), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sbj, src=sb.ap(pattern=[[hc * T * W, P], [T * W, hc], [1, W], [W, T]], offset=0),
                         engine=VE)
        pm = nl.ndarray((T, O), dtype=f32, buffer=nl.psum)
        for k in range(KW):
            nisa.nc_matmul(dst=pm, stationary=sbj[:, k, :], moving=wt[:, k, :], accumulate=(k > 0))
        # --- the sum of squares: per partition and row, then over the partitions by a ones column
        sv = sbj.ap(pattern=[[KW * T, P], [1, T], [T, KW]], offset=0)
        sq = nl.ndarray((P, T, KW), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=sv, data2=sv, op=nl.multiply, engine=VE)
        part = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=part, op=nl.add, data=sq, axis=2)
        ones = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=1.0)
        pss = nl.ndarray((T, 1), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pss, stationary=part, moving=ones, accumulate=False)
        # --- a row per partition from here (front_body's statements at the served switches)
        raw = nl.ndarray((T, O), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=raw, src=pm, engine=VE)
        ss = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ss, src=pss, engine=VE)
        a = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=a, data=ss, op0=nl.multiply, operand0=1.0 / F, op1=nl.add, operand1=rms_eps, engine=VE)
        r0t = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=r0t, op=nl.rsqrt, data=a)
        nt = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=nt, data1=r0t, data2=r0t, op=nl.multiply, engine=VE)
        nisa.tensor_tensor(dst=nt, data1=nt, data2=a, op=nl.multiply, engine=VE)
        nisa.tensor_scalar(dst=nt, data=nt, op0=nl.multiply, operand0=-0.5, op1=nl.add, operand1=1.5, engine=VE)
        r = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r, data1=r0t, data2=nt, op=nl.multiply, engine=VE)
        mix = nl.ndarray((T, O), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mix, data=raw, op0=nl.multiply, operand0=r, engine=VE)
        bt = nl.ndarray((T, O), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bt, src=base.ap(pattern=[[0, T], [1, O]], offset=0))
        st = nl.ndarray((T, 3), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=st, src=scale.ap(pattern=[[0, T], [1, 3]], offset=0))
        z = nl.ndarray((T, 2 * hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=z[:, 0:hc], data=mix[:, 0:hc], op0=nl.multiply, operand0=st[:, 0:1], engine=VE)
        nisa.tensor_scalar(dst=z[:, hc:2 * hc], data=mix[:, hc:2 * hc], op0=nl.multiply, operand0=st[:, 1:2], engine=VE)
        nisa.tensor_tensor(dst=z, data1=z, data2=bt[:, 0:2 * hc], op=nl.add, engine=VE)
        sg = nl.ndarray((T, 2 * hc), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=sg, op=nl.sigmoid, data=z)
        pre = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
        if plant == 3:  # silu: z sigmoid(z)
            zs = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=zs, data1=sg[:, 0:hc], data2=z[:, 0:hc], op=nl.multiply, engine=VE)
            nisa.tensor_scalar(dst=pre, data=zs, op0=nl.add, operand0=eps, engine=VE)
        else:
            nisa.tensor_scalar(dst=pre, data=sg[:, 0:hc], op0=nl.add, operand0=eps, engine=VE)
        D = nl.ndarray((T, T * hc), dtype=f32, buffer=nl.sbuf)
        nisa.affine_select(dst=D, pattern=[[1, T], [0, hc]], channel_multiplier=-1, offset=0,
                           on_true_tile=pre.ap(pattern=[[hc, T], [0, T], [1, hc]], offset=0), on_false_value=0.0,
                           cmp_op=nl.equal)
        ones_t = nl.ndarray((T, P), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ones_t, value=1.0)
        pbp = nl.ndarray((P, T * hc), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pbp, stationary=ones_t, moving=D, accumulate=False)
        pb = nl.ndarray((P, T * hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pb, src=pbp, engine=VE)
        post2 = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=post2, data=sg[:, hc:2 * hc], op0=nl.multiply, operand0=2.0, engine=VE)
        nisa.dma_copy(dst=post_out.ap(pattern=[[hc, T], [1, hc]], offset=r0 * hc), src=post2)
        lg = nl.ndarray((T, hc, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lg, data=mix.ap(pattern=[[O, T], [hc, hc], [1, hc]], offset=2 * hc), op0=nl.multiply,
                           operand0=st[:, 2:3], engine=VE)
        nisa.tensor_tensor(dst=lg, data1=lg, data2=bt.ap(pattern=[[O, T], [hc, hc], [1, hc]], offset=2 * hc),
                           op=nl.add, engine=VE)
        mx = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=mx, op=nl.maximum, data=lg, axis=2)
        c = nl.ndarray((T, hc, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c, data1=lg, data2=mx.ap(pattern=[[hc, T], [1, hc], [0, hc]], offset=0),
                           op=nl.subtract, engine=VE)
        nisa.activation(dst=c, op=nl.exp, data=c)
        s = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=s, op=nl.add, data=c, axis=2)
        nisa.reciprocal(dst=s, data=s)
        nisa.tensor_tensor(dst=c, data1=c, data2=s.ap(pattern=[[hc, T], [1, hc], [0, hc]], offset=0),
                           op=nl.multiply, engine=VE)
        E = hc + 1
        cp = nl.ndarray((T, E, E), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=cp, value=eps)
        nisa.tensor_scalar(dst=cp[:, 0:hc, 0:hc], data=c, op0=nl.add, operand0=eps, engine=VE)
        # --- the pre-mix from S' in SBUF (front_body's statements with S's h-layout reload replaced by sb)
        x = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=x, data1=pb.ap(pattern=[[T * hc, P], [hc, T], [0, W]], offset=0), data2=sb[:, 0, :, :],
                           op=nl.multiply, engine=VE)
        for n in range(1, hc):
            pr2 = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pr2, data1=pb.ap(pattern=[[T * hc, P], [hc, T], [0, W]], offset=n),
                               data2=sb[:, n, :, :], op=nl.multiply, engine=VE)
            nisa.tensor_tensor(dst=x, data1=x, data2=pr2, op=nl.add, engine=VE)
        cx = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=cx, data=x, op0=nl.multiply, operand0=65537.0, engine=VE)
        dx = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=dx, data1=cx, data2=x, op=nl.subtract, engine=VE)
        nisa.tensor_tensor(dst=cx, data1=cx, data2=dx, op=nl.subtract, engine=VE)
        xb = nl.ndarray((P, T, W), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xb, src=cx, engine=VE)
        nisa.dma_copy(dst=x_out.ap(pattern=[[W, P], [H, T], [1, W]], offset=r0 * H), src=xb)
        # --- the Sinkhorn (sinkhorn_body, skf 1: the padded tile), 2 iters - 1 half-rounds, columns first
        n_half = 2 * (iters - 1 if plant == 1 else iters) - 1
        for k in range(n_half):
            d = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
            if k % 2 == 0:
                nisa.tensor_reduce(dst=d, op=nl.add, data=cp.ap(pattern=[[E * E, T], [1, hc], [E, E]], offset=0), axis=2)
                db = d.ap(pattern=[[hc, T], [0, hc], [1, hc]], offset=0)
            else:
                nisa.tensor_reduce(dst=d, op=nl.add, data=cp.ap(pattern=[[E * E, T], [E, hc], [1, E]], offset=0), axis=2)
                db = d.ap(pattern=[[hc, T], [1, hc], [0, hc]], offset=0)
            nisa.reciprocal(dst=d, data=d)
            nisa.tensor_tensor(dst=cp[:, 0:hc, 0:hc], data1=cp[:, 0:hc, 0:hc], data2=db, op=nl.multiply, engine=VE)
        nisa.dma_copy(dst=comb_out.ap(pattern=[[hc * hc, T], [hc, hc], [1, hc]], offset=r0 * hc * hc),
                      src=cp[:, 0:hc, 0:hc])
        return S_out, x_out, post_out, comb_out
else:
    mhc_seam_kernel = None


REV = SH.kernel_rev(__file__, "try:  # the Neuron venv", "    mhc_seam_kernel = None")
