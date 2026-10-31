"""GLM-5.3-Flash's hyper-connection before a block (models/hybrid.py _mhc, its "elementwise" form) as one NKI kernel,
in decode calls: from the streams S [T, hc, H]
(bf16) and the layer's hc_*_fn [2 hc + hc^2, hc H] (bf16, the model dtype), hc_*_base and hc_*_scale (fp32),
    r = rsqrt(mean(S^2) + rms_eps);  mix = (S fn^T) r                                         (fp32, per token)
    pre = sigmoid(mix[:hc] s0 + b[:hc]) + eps;  post = 2 sigmoid(mix[hc:2hc] s1 + b[hc:2hc])
    comb = Sinkhorn(softmax(mix[2hc:] s2 + b[2hc:]) + eps), exactly `iters` rounds (_mhc's)
    x = round_bf16(pre_0 S_0 + pre_1 S_1 + ...)                                      (fp32, in order, _round's barrier)
returning (x [T, H] bf16, post [T, hc] fp32, comb [T, hc * hc] fp32).

Why: XLA's _mhc measured 0.0985 ms per hyper-connection at one decode row and 0.1566 at four (dependent chains, on
trn1), 90 hyper-connections per step: an fp32 norm over hc H values, a [2 hc + hc^2,
hc H] projection, a dozen [T, 2 hc + hc^2] ops and hc [T, H] multiply-adds, each with a token per partition.

Layouts. k-layout, k = p NJ + j over hc H (NJ = hc H / 128): a DMA of NJ-element runs puts every token's row and every
weight row across all 128 partitions; one vector copy each makes them j-major, so the projection is NJ matmuls of
[128, T] (stationary) by [128, 2 hc + hc^2] (moving) accumulating [T, 2 hc + hc^2] in one PSUM bank, and the sum of
squares is a per-partition reduce then one matmul by a ones column. From there a token per partition.
h-layout, h = p W + j (W = H / 128), for the pre-mix, as in kernels/mhc_mix.py: pre reaches
every partition through a private HBM copy read with partition stride 0.

Numerics against _mhc on the host: the projection's products are exact (bf16 x bf16 in fp32) and only their fp32
summation order differs; the sum of squares likewise; rsqrt, sigmoid and exp are the scalar engine's; every division is
a reciprocal times a multiply (trn1's parser has no divide); the pre-mix and the barrier are _mhc's own fp32 ops in its
order. So x, post and comb match the host to a few fp32 ulps of mix, not bit for bit. (On trn1, with no --auto-cast
passed, XLA's fp32 matmul runs in bf16; on the served bf16 fn and bf16 streams that cast is lossless, so XLA's
projection should also differ from the host only in summation order. [unverified on the device])
"""

from __future__ import annotations

import zlib

import torch

from . import shell as SH

# pre / post's sigmoid: 0 the scalar engine's (served); 1 1 / (1 + exp(-z)), the scalar engine's exp (median 1.5e-6
# relative off) and the vector engine's reciprocal.
SIGMOID = 0
# the Sinkhorn's half-round (2 iters - 1 of them, each a dependent chain of tiny vector instructions: ~77% of the
# kernel's vector time in a trn2 decode step): 0 reduce, + eps, reciprocal, multiply (4); 1 (served) the + eps inside
# the reduce (an eps row and column, 3; the same fp32 values if the reduce adds in order). A divide for the reciprocal
# and multiply is not an option: the NKI frontend (0.6.0) has no divide operator ("invalid kwarg 'op': unsupported
# operator 'divide'", trn2 and trn1 targets alike).
SK = 1
# PBC (served): pre on every partition on chip, as kernels/shell.py's mhc_pre_h does (D [T, T hc] holds pre[k, c] at
# column (k, c) of row k and an exact 0 elsewhere, a select, so a NaN row reaches no other row's column; then ones [T,
# 128]^T D on the PE, exact in fp32: one nonzero term per sum), in place of the private HBM round trip (pscr) that put
# ~5.5 us between pre and x in the FFN half. The same fp32 values: bit-identical by construction.
# kernels/mhc_split.py's front_body mirrors it.
PBC = True
PBC_REV = zlib.crc32(b"mhc_pre: pre broadcast on chip")
P = 128
MAX_T = 128  # the projection's stationary [128, T]


def emulate(S: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, iters: int, eps: float,
            rms_eps: float):
    """_mhc's elementwise form in host fp32: (x [T, H] bf16, post [T, hc], comb [T, hc, hc])."""
    T, hc, H = S.shape
    flat = S.reshape(T, -1).float()
    r = torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + rms_eps)
    mix = torch.nn.functional.linear(flat, fn.float()) * r
    pre = torch.sigmoid(mix[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mix[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    c = torch.softmax(mix[:, 2 * hc :].view(T, hc, hc) * scale[2] + base[2 * hc :].view(hc, hc), dim=-1) + eps
    c = c / (c.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        c = c / (c.sum(dim=-1, keepdim=True) + eps)
        c = c / (c.sum(dim=-2, keepdim=True) + eps)
    x = pre[:, 0:1] * S[:, 0].float()
    for n in range(1, hc):
        x = x + pre[:, n : n + 1] * S[:, n].float()
    k = x * 65537.0
    return (k - (k - x)).to(torch.bfloat16), post, c


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    def pre_body(S, fn, base, scale, hc: int, iters: int, eps: float, rms_eps: float, sig: int, skf: int):
        """mhc_pre_kernel's body, verbatim, as a function another kernel can call (kernels/ffn_half.py):
        the same instructions in the same order. Returns (x_out, post_out, comb_out, xb): the kernel's three outputs
        and xb, x in h-layout [128, T, W] bf16 in SBUF (h = p W + j)."""
        f32, bf16 = nl.float32, nl.bfloat16
        T, F = S.shape
        O = fn.shape[0]  # 2 hc + hc^2
        H = F // hc
        NJ = F // P
        W = H // P
        x_out = nl.ndarray((T, H), dtype=bf16, buffer=nl.shared_hbm)
        post_out = nl.ndarray((T, hc), dtype=f32, buffer=nl.shared_hbm)
        comb_out = nl.ndarray((T, hc * hc), dtype=f32, buffer=nl.shared_hbm)
        if not PBC:
            pscr = nl.ndarray((T, hc), dtype=f32, buffer=nl.private_hbm)
        # k-layout (k = p NJ + j): the streams and the weights, NJ consecutive k per partition
        sk = nl.ndarray((P, T, NJ), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=sk, src=S.ap(pattern=[[NJ, P], [F, T], [1, NJ]], offset=0))
        wk = nl.ndarray((P, O, NJ), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wk, src=fn.ap(pattern=[[NJ, P], [F, O], [1, NJ]], offset=0))
        # j-major copies, so that each matmul operand below is a contiguous [P, T] / [P, O] slice
        skj = nl.ndarray((P, NJ, T), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=skj, src=sk.ap(pattern=[[T * NJ, P], [1, NJ], [NJ, T]], offset=0),
                         engine=nisa.vector_engine)
        wkj = nl.ndarray((P, NJ, O), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wkj, src=wk.ap(pattern=[[O * NJ, P], [1, NJ], [NJ, O]], offset=0),
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
        # a token per partition from here: r = rsqrt(ss / F + rms_eps), mix = sums r
        raw = nl.ndarray((T, O), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=raw, src=pm, engine=nisa.vector_engine)
        ss = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ss, src=pss, engine=nisa.vector_engine)
        # a = ss / F + rms_eps (the vector engine's fp32 ops, F a power of two in GLM-5.3-Flash); r0 = the scalar
        # engine's rsqrt(a), then one Newton step r = r0 (1.5 - 0.5 a r0^2) on the vector engine: the scalar engine's
        # functions are approximations (its exp is up to 1.5e-5 relative off), and without the step post and comb came
        # out ~1e-5 off the host's, 10x XLA's (on trn1)
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
        if PBC:  # pre on every partition on chip (shell.mhc_pre_h's form), no private HBM round trip
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
        # comb: logits mix s2 + b, the row softmax, + eps, then the Sinkhorn rounds (_mhc's)
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
            for k in range(2 * iters - 1):
                d = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
                if k % 2 == 0:  # columns: sum over i (the transposed view's last axis), per j
                    nisa.tensor_reduce(dst=d, op=nl.add,
                                       data=c.ap(pattern=[[hc * hc, T], [1, hc], [hc, hc]], offset=0), axis=2)
                    nisa.tensor_scalar(dst=d, data=d, op0=nl.add, operand0=eps, engine=nisa.vector_engine)
                    nisa.reciprocal(dst=d, data=d)
                    nisa.tensor_tensor(dst=c, data1=c, data2=d.ap(pattern=[[hc, T], [0, hc], [1, hc]], offset=0),
                                       op=nl.multiply, engine=nisa.vector_engine)
                else:  # rows: sum over j, per i
                    nisa.tensor_reduce(dst=d, op=nl.add, data=c, axis=2)
                    nisa.tensor_scalar(dst=d, data=d, op0=nl.add, operand0=eps, engine=nisa.vector_engine)
                    nisa.reciprocal(dst=d, data=d)
                    nisa.tensor_tensor(dst=c, data1=c, data2=d.ap(pattern=[[hc, T], [1, hc], [0, hc]], offset=0),
                                       op=nl.multiply, engine=nisa.vector_engine)
            nisa.dma_copy(dst=comb_out.ap(pattern=[[hc * hc, T], [hc, hc], [1, hc]], offset=0), src=c)
        else:
            # skf 1 (the served eps form): c inside a padded tile [T, hc + 1, hc + 1] whose last row and last column
            # hold eps, so one reduce over a row or a column of it is the hc entries' sum and then + eps, the eps last
            # as skf 0's separate add adds it (the same fp32 sum if the reduce adds in order): 3 per half-round.
            E = hc + 1
            cp = nl.ndarray((T, E, E), dtype=f32, buffer=nl.sbuf)
            nisa.memset(dst=cp, value=eps)
            nisa.tensor_scalar(dst=cp[:, 0:hc, 0:hc], data=c, op0=nl.add, operand0=eps, engine=nisa.vector_engine)
            for k in range(2 * iters - 1):
                d = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
                if k % 2 == 0:  # columns: sum over i = 0..hc (the eps row last), per j
                    nisa.tensor_reduce(dst=d, op=nl.add, data=cp.ap(pattern=[[E * E, T], [1, hc], [E, E]], offset=0),
                                       axis=2)
                    db = d.ap(pattern=[[hc, T], [0, hc], [1, hc]], offset=0)
                else:  # rows: sum over j = 0..hc (the eps column last), per i
                    nisa.tensor_reduce(dst=d, op=nl.add, data=cp.ap(pattern=[[E * E, T], [E, hc], [1, E]], offset=0),
                                       axis=2)
                    db = d.ap(pattern=[[hc, T], [1, hc], [0, hc]], offset=0)
                nisa.reciprocal(dst=d, data=d)
                nisa.tensor_tensor(dst=cp[:, 0:hc, 0:hc], data1=cp[:, 0:hc, 0:hc], data2=db, op=nl.multiply,
                                   engine=nisa.vector_engine)
            nisa.dma_copy(dst=comb_out.ap(pattern=[[hc * hc, T], [hc, hc], [1, hc]], offset=0), src=cp[:, 0:hc, 0:hc])
        # the pre-mix in h-layout (h = p W + j): pre on every partition by a stride-0 read of its HBM copy, then
        # x = pre_0 S_0 + pre_1 S_1 + ... in fp32 and _round's barrier (c = 65537 x; hi = c - (c - x)), bf16 out
        if not PBC:
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
        return x_out, post_out, comb_out, xb

    @nki.jit
    def mhc_pre_kernel(S, fn, base, scale, hc: int, iters: int, eps: float, rms_eps: float, sig: int, rev: int,
                            skf: int):
        """See the module docstring. S bf16 [T, hc H] (T <= MAX_T), fn bf16 [2 hc + hc^2, hc H], base fp32 [2 hc +
        hc^2], scale fp32 [3]; rev this module's source revision. Returns (x bf16 [T, H], post fp32 [T, hc], comb fp32
        [T, hc * hc])."""
        x_out, post_out, comb_out, _ = pre_body(S, fn, base, scale, hc, iters, eps, rms_eps, sig, skf)
        return x_out, post_out, comb_out
else:
    mhc_pre_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source, passed as the static argument `rev` (LNL's compile-cache key does not
    include NKI kernel source: kernels/delta_rule.py _kernel_rev)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    mhc_pre_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev() ^ zlib.crc32(repr((P,)).encode()) ^ PBC_REV  # and the constants its kernel reads


def takes(device: torch.device, T: int, H: int, hc: int, fn_dtype: torch.dtype) -> bool:
    """Whether _mhc uses the kernel: on a device, T <= MAX_T, 128 | H, bf16 fn. At LNC=2 (trn2) it runs at the
    runtime's grid with no split: both physical cores compute every row and write the same outputs."""
    return ((device.type != "cpu" or SH.HOST_SIM) and T <= MAX_T and H % P == 0
            and fn_dtype == torch.bfloat16)


def pre(S: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, iters: int, eps: float,
        rms_eps: float):
    """(x [T, H] bf16, post [T, hc] fp32, comb [T, hc, hc] fp32): the kernel in the caller's graph on the device,
    emulate() on the host."""
    T, hc, H = S.shape
    if S.device.type == "cpu" and not SH.HOST_SIM:
        return emulate(S, fn, base, scale, iters, eps, rms_eps)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if mhc_pre_kernel is None:
        raise RuntimeError("the NKI mHC pre kernel needs the nki package (the Neuron venv)")
    x, post, comb = wrap_nki(mhc_pre_kernel)[platform.nki_grid()](
        S=S.to(torch.bfloat16).reshape(T, hc * H).contiguous(), fn=fn.to(torch.bfloat16).contiguous(),
        base=base.float().contiguous(), scale=scale.float().contiguous(), hc=int(hc), iters=int(iters),
        eps=float(eps), rms_eps=float(rms_eps), sig=SIGMOID, rev=REV, skf=SK)
    return x, post, comb.view(T, hc, hc)
