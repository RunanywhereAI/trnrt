"""kda_shell's SKL forms, off in the served build (the kernel body references them): shell.mhc_pre_h in three
in-kernel helpers, split at its comb block (the Sinkhorn: ~117 vector ops a call; comb is only written out, nothing in
kda_shell reads it), so kda_shell can issue the Sinkhorn after the KDA projection instead of before xa / xn. pre_a:
mhc_pre_h up to post (returns what the other two read: mix, stt, bt, pb); pre_comb: the comb block (returns cp); pre_b:
xa to xh. Their lines are mhc_pre_h's verbatim, in its order, the lines marked SPLIT aside, so a, comb, b is
mhc_pre_h's program, and any order of them the same instructions on the same inputs. wait_on (the positive control's
dependency): part += +0 from cp's last tile."""

from __future__ import annotations

import zlib

from .shell import P, kernel_rev

# -- in-kernel (NKI) ---------------------------------------------------------------------------------------------------

try:  # the Neuron venv
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nisa = None

if nisa is not None:
    from .shell import dma_static, sigmoid_exp

    def pre_a(sh, fw, bf, o_base: int, o_scale: int, post_out, hc: int, T: int, H: int, iters: int, hc_eps: float,
              rms_eps: float):
        """mhc_pre_h through post (post_out written): (mix, stt, bt, pb)."""
        f32, bf16 = nl.float32, nl.bfloat16
        W, F = H // P, hc * H
        O = 2 * hc + hc * hc
        shj = nl.ndarray((P, hc, W, T), dtype=bf16, buffer=nl.sbuf)
        for n in range(hc):
            nisa.tensor_copy(dst=shj[:, n, :, :],
                             src=sh.ap(pattern=[[hc * T * W, P], [1, W], [W, T]], offset=n * T * W),
                             engine=nisa.vector_engine)
        pm = nl.ndarray((T, O), dtype=f32, buffer=nl.psum)
        for n in range(hc):
            for j in range(W):
                nisa.nc_matmul(dst=pm, stationary=shj[:, n, j, :], moving=fw[:, n, j, :], accumulate=(n + j > 0))
        ones = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=1.0)
        ones_t = nl.ndarray((T, P), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ones_t, value=1.0)
        part = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
        for n in range(hc):
            sqn = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sqn, data1=sh[:, n, :, :], data2=sh[:, n, :, :], op=nl.multiply,
                               engine=nisa.vector_engine)
            if n == 0:
                nisa.tensor_reduce(dst=part, op=nl.add, data=sqn, axis=2)
            else:
                pn = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=pn, op=nl.add, data=sqn, axis=2)
                nisa.tensor_tensor(dst=part, data1=part, data2=pn, op=nl.add, engine=nisa.vector_engine)
        pss = nl.ndarray((T, 1), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pss, stationary=part, moving=ones, accumulate=False)
        raw = nl.ndarray((T, O), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=raw, src=pm, engine=nisa.vector_engine)
        ss = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ss, src=pss, engine=nisa.vector_engine)
        a = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=a, data=ss, op0=nl.multiply, operand0=1.0 / F, op1=nl.add, operand1=rms_eps,
                           engine=nisa.vector_engine)
        r0 = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=r0, op=nl.rsqrt, data=a)
        nt = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=nt, data1=r0, data2=r0, op=nl.multiply, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=nt, data1=nt, data2=a, op=nl.multiply, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=nt, data=nt, op0=nl.multiply, operand0=-0.5, op1=nl.add, operand1=1.5,
                           engine=nisa.vector_engine)
        r = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r, data1=r0, data2=nt, op=nl.multiply, engine=nisa.vector_engine)
        mix = nl.ndarray((T, O), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mix, data=raw, op0=nl.multiply, operand0=r, engine=nisa.vector_engine)
        bt = nl.ndarray((T, O), dtype=f32, buffer=nl.sbuf)
        dma_static(dst=bt, src=bf.ap(pattern=[[0, T], [1, O]], offset=o_base))
        stt = nl.ndarray((T, 3), dtype=f32, buffer=nl.sbuf)
        dma_static(dst=stt, src=bf.ap(pattern=[[0, T], [1, 3]], offset=o_scale))
        z = nl.ndarray((T, 2 * hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=z[:, 0:hc], data=mix[:, 0:hc], op0=nl.multiply, operand0=stt[:, 0:1],
                           engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=z[:, hc:2 * hc], data=mix[:, hc:2 * hc], op0=nl.multiply, operand0=stt[:, 1:2],
                           engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=z, data1=z, data2=bt[:, 0:2 * hc], op=nl.add, engine=nisa.vector_engine)
        sgz = nl.ndarray((T, 2 * hc), dtype=f32, buffer=nl.sbuf)
        ez = nl.ndarray((T, 2 * hc), dtype=f32, buffer=nl.sbuf)
        sigmoid_exp(sgz, z, ez)
        pre = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pre, data=sgz[:, 0:hc], op0=nl.add, operand0=hc_eps, engine=nisa.vector_engine)
        D = nl.ndarray((T, T * hc), dtype=f32, buffer=nl.sbuf)
        nisa.affine_select(dst=D, pattern=[[1, T], [0, hc]], channel_multiplier=-1, offset=0,
                           on_true_tile=pre.ap(pattern=[[hc, T], [0, T], [1, hc]], offset=0), on_false_value=0.0,
                           cmp_op=nl.equal)
        pbp = nl.ndarray((P, T * hc), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pbp, stationary=ones_t, moving=D, accumulate=False)
        pb = nl.ndarray((P, T * hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pb, src=pbp, engine=nisa.vector_engine)
        post = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=post, data=sgz[:, hc:2 * hc], op0=nl.multiply, operand0=2.0, engine=nisa.vector_engine)
        dma_static(dst=post_out, src=post)
        return mix, stt, bt, pb  # SPLIT

    def pre_comb(mix, stt, bt, comb_out, hc: int, T: int, iters: int, hc_eps: float):
        """mhc_pre_h's comb block (the Sinkhorn), comb_out written: cp [T, hc + 1, hc + 1]."""
        f32 = nl.float32  # SPLIT
        O = 2 * hc + hc * hc  # SPLIT
        lg = nl.ndarray((T, hc, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lg, data=mix.ap(pattern=[[O, T], [hc, hc], [1, hc]], offset=2 * hc), op0=nl.multiply,
                           operand0=stt[:, 2:3], engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=lg, data1=lg, data2=bt.ap(pattern=[[O, T], [hc, hc], [1, hc]], offset=2 * hc),
                           op=nl.add, engine=nisa.vector_engine)
        mx = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=mx, op=nl.maximum, data=lg, axis=2)
        cm = nl.ndarray((T, hc, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=cm, data1=lg, data2=mx.ap(pattern=[[hc, T], [1, hc], [0, hc]], offset=0),
                           op=nl.subtract, engine=nisa.vector_engine)
        nisa.activation(dst=cm, op=nl.exp, data=cm)
        sm = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sm, op=nl.add, data=cm, axis=2)
        nisa.reciprocal(dst=sm, data=sm)
        nisa.tensor_tensor(dst=cm, data1=cm, data2=sm.ap(pattern=[[hc, T], [1, hc], [0, hc]], offset=0),
                           op=nl.multiply, engine=nisa.vector_engine)
        E1 = hc + 1
        cp = nl.ndarray((T, E1, E1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=cp, value=hc_eps)
        nisa.tensor_scalar(dst=cp[:, 0:hc, 0:hc], data=cm, op0=nl.add, operand0=hc_eps, engine=nisa.vector_engine)
        for k in range(2 * iters - 1):
            dk = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
            if k % 2 == 0:  # columns: sum over i = 0..hc (the eps row last), per j
                nisa.tensor_reduce(dst=dk, op=nl.add, data=cp.ap(pattern=[[E1 * E1, T], [1, hc], [E1, E1]], offset=0),
                                   axis=2)
                db = dk.ap(pattern=[[hc, T], [0, hc], [1, hc]], offset=0)
            else:  # rows: sum over j = 0..hc (the eps column last), per i
                nisa.tensor_reduce(dst=dk, op=nl.add, data=cp.ap(pattern=[[E1 * E1, T], [E1, hc], [1, E1]], offset=0),
                                   axis=2)
                db = dk.ap(pattern=[[hc, T], [1, hc], [0, hc]], offset=0)
            nisa.reciprocal(dst=dk, data=dk)
            nisa.tensor_tensor(dst=cp[:, 0:hc, 0:hc], data1=cp[:, 0:hc, 0:hc], data2=db, op=nl.multiply,
                               engine=nisa.vector_engine)
        dma_static(dst=comb_out.ap(pattern=[[hc * hc, T], [hc, hc], [1, hc]], offset=0), src=cp[:, 0:hc, 0:hc])
        return cp  # SPLIT

    def pre_b(sh, pb, hc: int, T: int, H: int):
        """mhc_pre_h from xa: x [128, T, W] bf16 (h-layout)."""
        f32, bf16 = nl.float32, nl.bfloat16  # SPLIT
        W = H // P  # SPLIT
        xa = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=xa, data1=pb.ap(pattern=[[T * hc, P], [hc, T], [0, W]], offset=0), data2=sh[:, 0, :, :],
                           op=nl.multiply, engine=nisa.vector_engine)
        for n in range(1, hc):
            prn = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prn, data1=pb.ap(pattern=[[T * hc, P], [hc, T], [0, W]], offset=n),
                               data2=sh[:, n, :, :], op=nl.multiply, engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=xa, data1=xa, data2=prn, op=nl.add, engine=nisa.vector_engine)
        cx = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=cx, data=xa, op0=nl.multiply, operand0=65537.0, engine=nisa.vector_engine)
        dx = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=dx, data1=cx, data2=xa, op=nl.subtract, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=cx, data1=cx, data2=dx, op=nl.subtract, engine=nisa.vector_engine)
        xh = nl.ndarray((P, T, W), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xh, src=cx, engine=nisa.vector_engine)
        return xh

    def wait_on(part, cp, B: int, T: int, hc: int):
        """part [128, B] fp32 += +0 made from cp's last tile (cp > 0 and finite: x 0.0 is +0, and the sum of squares part
        is >= +0, so its bits stay): every lane of part, and so xn and the projection after it, waits for the whole
        Sinkhorn (SKL 3, the positive control)."""
        E1 = hc + 1
        z0 = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=z0, data=cp.ap(pattern=[[E1 * E1, T], [1, 1]], offset=0), op0=nl.multiply, operand0=0.0,
                           engine=nisa.vector_engine)
        ot = nl.ndarray((T, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ot, value=1.0)
        zp = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=zp, stationary=ot, moving=z0, accumulate=False)
        zs = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=zs, src=zp, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=part, data=part, op0=nl.add, operand0=zs[:, 0:1], engine=nisa.vector_engine)

# -- end of the in-kernel helpers
REV = kernel_rev(__file__, "# -- in-kernel (NKI)", "# -- end of the in-kernel helpers") ^ zlib.crc32(repr((P,)).encode())
