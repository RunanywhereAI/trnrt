"""mhc_split.front_body with the static fn weights taken already j-major (the served form). front_body loads fn [O, F]
in k-layout (k = p NJ + j) and re-lays it out to [P, NJ, O] on the vector engine on every call (786 KB, ~2.7 us);
front_body_fnj takes fn_j's copy of it, made once at load (models/hybrid.py: hc_ffn_fnj), and DMAs it straight into that
operand (each partition's NJ O elements one contiguous HBM run). The same values in the same SBUF tile, so the same
bits. front_body_fnj is front_body statement for statement but that one named divergence (tests/test_mhc_fnj.py pins
it). mhc_split rebinds its front_body to this one and folds REV."""

from __future__ import annotations

import zlib

import torch

from . import mhc_pre as MP

P = MP.P


def fn_j(fn: torch.Tensor) -> torch.Tensor:
    """The load-time copy: fn [O, F] (F = P NJ) -> [P, NJ, O], [p, j, o] = fn[o, p NJ + j]: the operand front_body's
    k-layout load and re-layout build on every call."""
    O, F = fn.shape
    return fn.reshape(O, P, F // P).permute(1, 2, 0).contiguous()


try:  # the Neuron venv
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nisa = None


if nisa is not None:
    def front_body_fnj(S, fn, base, scale, hc: int, iters: int, eps: float, rms_eps: float, sig: int, skf: int):
        """mhc_pre.pre_body up to the Sinkhorn's input, then its pre-mix. Returns (x_out, post_out, comb_out, cp, xb):
        x_out and post_out written, comb_out allocated for sinkhorn_body, cp the Sinkhorn's tile, xb x in h-layout
        [128, T, W] bf16 in SBUF (h = p W + j)."""
        f32, bf16 = nl.float32, nl.bfloat16
        T, F = S.shape
        O = fn.shape[2]  # 2 hc + hc^2 (fn: fn_j's [P, NJ, O])
        H = F // hc
        NJ = F // P
        W = H // P
        x_out = nl.ndarray((T, H), dtype=bf16, buffer=nl.shared_hbm)
        post_out = nl.ndarray((T, hc), dtype=f32, buffer=nl.shared_hbm)
        comb_out = nl.ndarray((T, hc * hc), dtype=f32, buffer=nl.shared_hbm)
        if not MP.PBC:
            pscr = nl.ndarray((T, hc), dtype=f32, buffer=nl.private_hbm)
        # k-layout (k = p NJ + j): the streams, NJ consecutive k per partition
        sk = nl.ndarray((P, T, NJ), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=sk, src=S.ap(pattern=[[NJ, P], [F, T], [1, NJ]], offset=0))
        wkj = nl.ndarray((P, NJ, O), dtype=bf16, buffer=nl.sbuf)  # fn_j's copy: each partition's NJ O one run
        nisa.dma_copy(dst=wkj, src=fn)
        # the streams' j-major copy, so that each matmul operand below is a contiguous [P, T] slice
        skj = nl.ndarray((P, NJ, T), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=skj, src=sk.ap(pattern=[[T * NJ, P], [1, NJ], [NJ, T]], offset=0),
                         engine=nisa.vector_engine)
        # the projection: bf16 x bf16 products (exact in fp32), summed over the partitions and the NJ matmuls in PSUM
        pm = nl.ndarray((T, O), dtype=f32, buffer=nl.psum)
        for j in range(NJ):
            nisa.nc_matmul(dst=pm, stationary=skj[:, j, :], moving=wkj[:, j, :], accumulate=(j > 0))
        # the sum of squares: per partition on the vector engine, then over the partitions by a ones column
        sq = nl.ndarray((P, T, NJ), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=sk, data2=sk, op=nl.multiply, engine=nisa.vector_engine)
        part = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=part, op=nl.add, data=sq, axis=2)
        ones = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=1.0)
        pss = nl.ndarray((T, 1), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pss, stationary=part, moving=ones, accumulate=False)
        # a token per partition from here: r = rsqrt(ss / F + rms_eps), mix = sums r (one Newton step, as mhc_pre)
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
        nisa.dma_copy(dst=bt, src=base.ap(pattern=[[0, T], [1, O]], offset=0))
        st = nl.ndarray((T, 3), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=st, src=scale.ap(pattern=[[0, T], [1, 3]], offset=0))
        # pre = sigmoid(mix s0 + b) + eps; post = 2 sigmoid(mix s1 + b)
        z = nl.ndarray((T, 2 * hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=z[:, 0:hc], data=mix[:, 0:hc], op0=nl.multiply, operand0=st[:, 0:1],
                           engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=z[:, hc:2 * hc], data=mix[:, hc:2 * hc], op0=nl.multiply, operand0=st[:, 1:2],
                           engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=z, data1=z, data2=bt[:, 0:2 * hc], op=nl.add, engine=nisa.vector_engine)
        sg = nl.ndarray((T, 2 * hc), dtype=f32, buffer=nl.sbuf)
        if sig == 0:  # the scalar engine's sigmoid
            nisa.activation(dst=sg, op=nl.sigmoid, data=z)
        else:  # 1 / (1 + exp(-z))
            nisa.activation(dst=sg, op=nl.exp, data=z, scale=-1.0)
            nisa.tensor_scalar(dst=sg, data=sg, op0=nl.add, operand0=1.0, engine=nisa.vector_engine)
            nisa.reciprocal(dst=sg, data=sg)
        pre = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pre, data=sg[:, 0:hc], op0=nl.add, operand0=eps, engine=nisa.vector_engine)
        if MP.PBC:  # pre on every partition on chip (shell.mhc_pre_h's form), no private HBM round trip
            D = nl.ndarray((T, T * hc), dtype=f32, buffer=nl.sbuf)
            nisa.affine_select(dst=D, pattern=[[1, T], [0, hc]], channel_multiplier=-1, offset=0,
                               on_true_tile=pre.ap(pattern=[[hc, T], [0, T], [1, hc]], offset=0), on_false_value=0.0,
                               cmp_op=nl.equal)
            ones_t = nl.ndarray((T, P), dtype=f32, buffer=nl.sbuf)
            nisa.memset(dst=ones_t, value=1.0)
            pbp = nl.ndarray((P, T * hc), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pbp, stationary=ones_t, moving=D, accumulate=False)
            pb = nl.ndarray((P, T * hc), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pb, src=pbp, engine=nisa.vector_engine)
        else:
            nisa.dma_copy(dst=pscr, src=pre)
        post = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=post, data=sg[:, hc:2 * hc], op0=nl.multiply, operand0=2.0, engine=nisa.vector_engine)
        nisa.dma_copy(dst=post_out, src=post)
        # comb: logits mix s2 + b, the row softmax, + eps: the Sinkhorn's input (its rounds are sinkhorn_body's)
        lg = nl.ndarray((T, hc, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lg, data=mix.ap(pattern=[[O, T], [hc, hc], [1, hc]], offset=2 * hc), op0=nl.multiply,
                           operand0=st[:, 2:3], engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=lg, data1=lg, data2=bt.ap(pattern=[[O, T], [hc, hc], [1, hc]], offset=2 * hc),
                           op=nl.add, engine=nisa.vector_engine)
        m = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=m, op=nl.maximum, data=lg, axis=2)
        c = nl.ndarray((T, hc, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c, data1=lg, data2=m.ap(pattern=[[hc, T], [1, hc], [0, hc]], offset=0),
                           op=nl.subtract, engine=nisa.vector_engine)
        nisa.activation(dst=c, op=nl.exp, data=c)
        s = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=s, op=nl.add, data=c, axis=2)
        nisa.reciprocal(dst=s, data=s)
        nisa.tensor_tensor(dst=c, data1=c, data2=s.ap(pattern=[[hc, T], [1, hc], [0, hc]], offset=0),
                           op=nl.multiply, engine=nisa.vector_engine)
        if skf == 0:  # (not `sk`: that name is the k-layout streams tile above)
            nisa.tensor_scalar(dst=c, data=c, op0=nl.add, operand0=eps, engine=nisa.vector_engine)
            cp = c
        else:
            # skf 1 (the served eps form): c inside a padded tile [T, hc + 1, hc + 1] whose last row and last column
            # hold eps, so one reduce over a row or a column of it is the hc entries' sum and then + eps
            E = hc + 1
            cp = nl.ndarray((T, E, E), dtype=f32, buffer=nl.sbuf)
            nisa.memset(dst=cp, value=eps)
            nisa.tensor_scalar(dst=cp[:, 0:hc, 0:hc], data=c, op0=nl.add, operand0=eps, engine=nisa.vector_engine)
        # the pre-mix in h-layout (h = p W + j): pre on every partition by a stride-0 read of its HBM copy, then
        # x = pre_0 S_0 + pre_1 S_1 + ... in fp32 and _round's barrier (c = 65537 x; hi = c - (c - x)), bf16 out
        if not MP.PBC:
            pb = nl.ndarray((P, T, hc), dtype=f32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pb, src=pscr.ap(pattern=[[0, P], [hc, T], [1, hc]], offset=0))
        sh = nl.ndarray((P, hc, T, W), dtype=bf16, buffer=nl.sbuf)
        for n in range(hc):
            nisa.dma_copy(dst=sh[:, n, :, :], src=S.ap(pattern=[[W, P], [F, T], [1, W]], offset=n * H))
        x = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=x, data1=pb.ap(pattern=[[T * hc, P], [hc, T], [0, W]], offset=0), data2=sh[:, 0, :, :],
                           op=nl.multiply, engine=nisa.vector_engine)
        for n in range(1, hc):
            pr = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pr, data1=pb.ap(pattern=[[T * hc, P], [hc, T], [0, W]], offset=n),
                               data2=sh[:, n, :, :], op=nl.multiply, engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=x, data1=x, data2=pr, op=nl.add, engine=nisa.vector_engine)
        cx = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=cx, data=x, op0=nl.multiply, operand0=65537.0, engine=nisa.vector_engine)
        dx = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=dx, data1=cx, data2=x, op=nl.subtract, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=cx, data1=cx, data2=dx, op=nl.subtract, engine=nisa.vector_engine)
        xb = nl.ndarray((P, T, W), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xb, src=cx, engine=nisa.vector_engine)
        nisa.dma_copy(dst=x_out.ap(pattern=[[W, P], [H, T], [1, W]], offset=0), src=xb)
        return x_out, post_out, comb_out, cp, xb
else:
    front_body_fnj = None


def _kernel_rev() -> int:
    """CRC-32 of this module's in-kernel source (the cache key does not hash NKI source)."""
    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    front_body_fnj = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev() ^ zlib.crc32(repr((P,)).encode())
