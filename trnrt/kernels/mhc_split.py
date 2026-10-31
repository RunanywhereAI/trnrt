"""mHC's hyper-connection before a block (kernels/mhc_pre.py), split in two: the front, which the block needs at once,
and the Sinkhorn, whose comb only the expand after the block reads, so a fused kernel can issue the Sinkhorn's
half-rounds under its own waits. Code beside mhc_pre, which keeps its own REV.

    front_body(S, fn, base, scale, ...) -> (x_out, post_out, comb_out, cp, xb)
        mhc_pre.pre_body's instructions up to the Sinkhorn's input, then its pre-mix: rms (one Newton step), the
        projection, mix, pre / post, the comb logits' row softmax + eps, and x. post_out and x_out are written;
        comb_out is allocated, not written. cp is the Sinkhorn's tile: skf 1 the padded [T, hc + 1, hc + 1] with its
        eps row and column, skf 0 [T, hc, hc].
    sinkhorn_body(cp, comb_out, hc, iters, eps, skf, lo, hi)
        half-rounds lo .. hi - 1 of pre_body's 2 iters - 1 (k even: the columns, k odd: the rows); the call whose hi is
        2 iters - 1 writes comb_out. A host kernel issues the 39 half-rounds in pieces between its own stages.

The instructions are pre_body's, copied (tests/test_mhc_split.py pins it against mhc_pre.py's source), with one move:
the pre-mix (x) comes before the Sinkhorn instead of after it. It reads pre, not comb, so the values are the same;
The kernel below (mhc_split_kernel, a test entry: no served path calls it) is checked against mhc_pre_kernel bit for bit
in the simulator.
"""

from __future__ import annotations

import zlib

import torch

from . import mhc_pre as MP

P = MP.P


def chunk_bounds(iters: int, chunks: int) -> list[tuple[int, int]]:
    """The Sinkhorn's 2 iters - 1 half-rounds in `chunks` consecutive pieces [(lo, hi), ...], each of ceil(n / chunks)
    but the last, as the kernel below issues them. Host side and tests only: never call it in a kernel body (the NKI
    parser refuses its comprehension and its raise)."""
    n = 2 * iters - 1
    if not 1 <= chunks <= n:
        raise ValueError(f"chunks {chunks} not in [1, {n}]")
    step = (n + chunks - 1) // chunks
    return [(lo, min(lo + step, n)) for lo in range(0, n, step)]


def emulate(S: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, iters: int, eps: float,
            rms_eps: float):
    """The same function as mhc_pre's: (x [T, H] bf16, post [T, hc], comb [T, hc, hc])."""
    return MP.emulate(S, fn, base, scale, iters, eps, rms_eps)


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    def front_body(S, fn, base, scale, hc: int, iters: int, eps: float, rms_eps: float, sig: int, skf: int):
        """mhc_pre.pre_body up to the Sinkhorn's input, then its pre-mix. Returns (x_out, post_out, comb_out, cp, xb):
        x_out and post_out written, comb_out allocated for sinkhorn_body, cp the Sinkhorn's tile, xb x in h-layout
        [128, T, W] bf16 in SBUF (h = p W + j)."""
        f32, bf16 = nl.float32, nl.bfloat16
        T, F = S.shape
        O = fn.shape[0]  # 2 hc + hc^2
        H = F // hc
        NJ = F // P
        W = H // P
        x_out = nl.ndarray((T, H), dtype=bf16, buffer=nl.shared_hbm)
        post_out = nl.ndarray((T, hc), dtype=f32, buffer=nl.shared_hbm)
        comb_out = nl.ndarray((T, hc * hc), dtype=f32, buffer=nl.shared_hbm)
        if not MP.PBC:
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

    def sinkhorn_body(cp, comb_out, hc: int, iters: int, eps: float, skf: int, lo: int = 0, hi: int = -1):
        """Half-rounds lo .. hi - 1 (hi -1: to the end) of pre_body's 2 iters - 1, on front_body's cp, in place; the
        call that reaches the end writes comb_out. k even sums the columns, k odd the rows, as pre_body."""
        f32 = nl.float32
        n_half = 2 * iters - 1
        if hi < 0:
            hi = n_half
        assert lo >= 0
        assert lo <= hi
        assert hi <= n_half
        T = cp.shape[0]
        if skf == 0:
            c = cp
            for k in range(lo, hi):
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
            if hi == n_half:
                nisa.dma_copy(dst=comb_out.ap(pattern=[[hc * hc, T], [hc, hc], [1, hc]], offset=0), src=c)
        else:
            E = hc + 1
            for k in range(lo, hi):
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
            if hi == n_half:
                nisa.dma_copy(dst=comb_out.ap(pattern=[[hc * hc, T], [hc, hc], [1, hc]], offset=0),
                              src=cp[:, 0:hc, 0:hc])

    @nki.jit
    def mhc_split_kernel(S, fn, base, scale, hc: int, iters: int, eps: float, rms_eps: float, sig: int, rev: int,
                              skf: int, chunks: int):
        """mhc_pre_kernel's function as front_body then the Sinkhorn in `chunks` back-to-back pieces (1: one
        call; the chunk test checks that the pieces join exactly). Same arguments and outputs as mhc_pre's kernel, plus
        `chunks`; rev this module's source revision."""
        x_out, post_out, comb_out, cp, _ = front_body(S, fn, base, scale, hc, iters, eps, rms_eps, sig, skf)
        n_half = 2 * iters - 1
        step = (n_half + chunks - 1) // chunks  # chunk_bounds' pieces, in a form the parser takes
        for lo in range(0, n_half, step):
            hi = lo + step
            if hi > n_half:
                hi = n_half
            sinkhorn_body(cp, comb_out, hc, iters, eps, skf, lo, hi)
        return x_out, post_out, comb_out
else:
    front_body = sinkhorn_body = None
    mhc_split_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source, the static `rev` (LNL's compile-cache key ignores NKI kernel source)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    mhc_split_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev() ^ zlib.crc32(repr((P,)).encode()) ^ MP.PBC_REV  # and the constants its kernel reads
# The k-layout front (fn [O, hc H], an attention-side fn: kernels/mhc_front.py) and its REV, before front_body is
# rebound below (that re-lays only the FFN-side fn)
front_body_k, REV_K = front_body, REV

# The FFN side (kernels/mhc_fnj.py): front_body takes fn already j-major (hc_ffn_fnj, fn_j's copy made at load) and
# drops the per-call re-layout of the static weights. Rebound here, outside the hashed span; REV folds mhc_fnj's REV.
FNJ = True
from . import mhc_fnj as _FNJ  # noqa: E402

if nki is not None:
    front_body = _FNJ.front_body_fnj
REV ^= _FNJ.REV


def fn_arg(layer):
    """The FFN-side mHC fn an FFN-half kernel takes (its "fn" blob): hc_ffn_fnj, fn_j's copy."""
    return layer.hc_ffn_fnj


def split(S: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, iters: int, eps: float,
          rms_eps: float, chunks: int = 1):
    """(x [T, H] bf16, post [T, hc] fp32, comb [T, hc, hc] fp32): the split kernel on the device, at mhc_pre's served
    switches (mhc_pre.SIGMOID, mhc_pre.SK); emulate() on the host. A test entry only: no served path calls it."""
    T, hc, H = S.shape
    if S.device.type == "cpu":
        return emulate(S, fn, base, scale, iters, eps, rms_eps)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if mhc_split_kernel is None:
        raise RuntimeError("the NKI mHC split kernel needs the nki package (the Neuron venv)")
    x, post, comb = wrap_nki(mhc_split_kernel)[platform.nki_grid()](
        S=S.to(torch.bfloat16).reshape(T, hc * H).contiguous(),
        fn=_FNJ.fn_j(fn.to(torch.bfloat16)),
        base=base.float().contiguous(), scale=scale.float().contiguous(), hc=int(hc), iters=int(iters),
        eps=float(eps), rms_eps=float(rms_eps), sig=MP.SIGMOID, rev=REV, skf=MP.SK, chunks=int(chunks))
    return x, post, comb.view(T, hc, hc)
