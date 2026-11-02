"""The fused KDA layer of GLM-5.3-Flash's decode, in its served split form: kda_shell's kernel (then the model's
attention all-reduce), then ONE kernel with ffn_half_pb's body and the glue after it (_b5), then the model's output
all-reduce. The FFN half's kernel is traced as a function (its undecorated body), so its source and its REV are
ffn_half_pb's own.

On the device the split form is bit-equal to the chain it replaces (kda_shell.decode, ffn_half.decode and XLA's
glue). It keeps XLA's two all-reduces between the kernels; the forms with the sums inside the kernel are not served.

Above 8 rows at grid 2 a program's rows no longer fit _b5's one chunk of 128 partitions, and the second kernel is
mk_ffn_b5c_kernel instead, the same glue in chunks of rows (_b5c, its own REVC, at the end of this file). Decode steps
of 8 rows or fewer keep mk_ffn_b5_kernel and REV."""

from __future__ import annotations

import zlib

from . import ffn_half_pb as FB
from . import kda_shell as KS
from . import mhc_split as MS
from . import shell as SH

MODE = "split"

try:  # the Neuron venv
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


def layer_takes(model, layer, T: int) -> bool:
    """The split form applies to this decode call: T a multiple of the programs (the glue splits the rows between
    them), and H / 128 chunks <= 128. A program's rows x H / 128 chunks <= 128 run _b5's one chunk, more run _b5c's
    chunks of rows (decode_layer_split picks the kernel). The programs are the ones the layer's shell blobs were packed
    for (layer.ks_dims, as decode_layer_split reads them; the runtime's NKI grid at serve), so the rule holds on a host
    whose runtime was never configured too."""
    npg = layer.ks_dims[3]
    return T % npg == 0 and model.cfg.hidden_size // 128 <= 128


def decode_layer_split(model, layer, S, positions, state_slot, pend=None):
    """One KDA + MoE layer's decode call: kda_shell.decode (its kernel, then model._attn_all_reduce), then
    mk_ffn_b5_kernel (ffn_half_pb's body and _b5, the arguments ffn_half.decode's run builds; past 8 rows at grid 2
    mk_ffn_b5c_kernel with the chunked _b5c), then model._out_reduce.
    pend: the previous layer's (post, comb, y [T, H]) as it returned them. Returns (S_n [T, hc, H], post fp32 [T, hc],
    comb fp32 [T, hc, hc], y [T, H] all-reduced), as kda_shell.decode then ffn_half.decode return them."""
    import torch

    from ..models.hybrid import _shell_pend
    from . import ffn_half as FH
    from . import moe_dedupe as mdd
    from . import moe_half as MH

    hy, cfg = model.cfg.hybrid, model.cfg
    H, nv, K, npg, hc = layer.ks_dims
    T = S.shape[0]
    S1, (post_a, comb_a, Y_a) = KS.decode(model, layer, S, positions, state_slot,
                                          pend=None if pend is None else _shell_pend(layer, pend))
    fb = {torch.bfloat16: layer.mh_blob_bf16, torch.float32: layer.mh_blob_f32, "fn": MS.fn_arg(layer),
          "post_norm": layer.post_norm, "base": layer.hc_ffn_base, "scale": layer.hc_ffn_scale}
    fa = FH.run(S1, (post_a, comb_a, Y_a), fb, layer.w_blob, Is=int(layer.shared_gate_up.shape[0] // 2),
                K=int(cfg.num_experts_per_tok), scale=float(cfg.routed_scaling_factor), group=mdd.group_size(1),
                ring=FH.RING, limit=float(hy.swiglu_limit), rsp=MH.RSPLIT, hc=hc, iters=hy.sinkhorn_iters,
                hc_eps=hy.hc_eps, rms_eps=cfg.rms_norm_eps, grid=npg, build_only=True)
    # past one chunk of this program's rows (T > 8 at grid 2) the chunked glue's kernel; at T <= 8 the served one
    chunked = (T // npg) * (H // 128) > 128
    args = dict(fa, rev=REVC if chunked else REV, frev=fa["rev"])
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    outs = wrap_nki(mk_ffn_b5c_kernel if chunked else mk_ffn_b5_kernel)[npg](**args)
    y = model._out_reduce(outs[3].view(T, H))
    return outs[0].view(T, hc, H), outs[1], outs[2].view(T, hc, hc), y


# -- the kernel source (REV covers from here to mk_ffn_b5_kernel = None) ---------------------------------------------

if nki is not None:
    _ffn_body = FB.ffn_half_kernel.func  # the FFN half (ffn_half_pb), traced inline

    def _b5(rp, sp):
        """XLA's glue after the FFN half (ffn_half.decode) as the DEVICE computes it, bit for bit, for this program's
        half of the rows: y = bf16((r0 + r1) + (s0 + s1)), fp32 throughout and ONE rounding. The source reads
        bf16(sum rp) + bf16(sum sp), and host torch (so the simulator harness) rounds twice, but neuronx-cc keeps the
        fused glue in fp32 and rounds once: XLA's y read equal to the single rounding on every element at T 4 and 8,
        and to the double rounding (this function's first form) on 63%. rp
        fp32 [npg, P, T, C] (the pairs layout: element (p, t, c = 2 j + k) is h = k W/2 + j P + p), sp fp32 [npg, T, W].
        Its rows go through the chunk layout [n C, P]: partition m = t C + k C/2 + j holds row t's h = m' P ..
        m' P + P - 1 (m' = k C/2 + j, as k W/2 + j P = (k C/2 + j) P), so its HBM side is one affine pattern (row
        offset + m P, P contiguous) for sp's loads and y's store. rp's fp32 sum goes into a stationary whose columns run
        (t, k, j) (2 n copies), and ONE fp32 identity matmul puts it in the chunk layout exactly (each output one fp32 x
        1; PE fp32 matmuls are exact); every add and the cast are on n C lanes. Returns the shared
        [1, T, W] holding this program's rows."""
        G, _, T, C = rp.shape
        W = sp.shape[2]
        npg = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
        pid = nl.program_id(axis=0) if npg > 1 else 0
        n = T // npg
        f32, bf16 = nl.float32, nl.bfloat16
        VE = nisa.vector_engine
        P = 128
        J = C // 2
        M = n * C  # chunk rows: this program's rows x C chunks of P
        assert W == C * P and C % 2 == 0 and M <= P, "_b5: W = C P, C even, n C <= 128 (T <= 8 at grid 2)"
        yh = nl.ndarray((1, T, W), dtype=bf16, buffer=nl.shared_hbm)
        r0 = nl.ndarray((P, n, J, 2), dtype=f32, buffer=nl.sbuf)  # [.., j, k]: c = 2 j + k, C's own order
        r1 = nl.ndarray((P, n, J, 2), dtype=f32, buffer=nl.sbuf)
        SH.dma_static(dst=r0, src=rp.ap(pattern=[[T * C, P], [C, n], [2, J], [1, 2]], offset=pid * n * C))
        SH.dma_static(dst=r1, src=rp.ap(pattern=[[T * C, P], [C, n], [2, J], [1, 2]],
                                        offset=P * T * C + pid * n * C))
        nisa.tensor_tensor(dst=r0, data1=r0, data2=r1, op=nl.add, engine=VE)  # program 0's + program 1's (sum(0))
        rbp = nl.ndarray((P, M), dtype=f32, buffer=nl.sbuf)  # columns (t, k, j), fp32 (no rounding here)
        for t in range(n):
            for k in range(2):
                nisa.tensor_copy(dst=rbp[:, t * C + k * J:t * C + (k + 1) * J], src=r0[:, t, :, k], engine=VE)
        ipi = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)  # the identity [P, P] in fp32
        nisa.iota(dst=ipi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        ipf = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ipf, src=ipi)
        jti = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=jti, pattern=[[1, P]], offset=0, channel_multiplier=0)
        eye = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eye, src=jti)
        nisa.tensor_scalar(dst=eye, data=eye, op0=nl.equal, operand0=ipf)
        pt = nl.ndarray((M, P), dtype=f32, buffer=nl.psum)  # one accumulation group, its own tile
        nisa.nc_matmul(dst=pt, stationary=rbp, moving=eye, accumulate=False)
        yr = nl.ndarray((M, P), dtype=f32, buffer=nl.sbuf)  # routed r0 + r1, chunk layout, exact
        nisa.tensor_copy(dst=yr, src=pt, engine=VE)
        s0 = nl.ndarray((M, P), dtype=f32, buffer=nl.sbuf)
        s1 = nl.ndarray((M, P), dtype=f32, buffer=nl.sbuf)
        SH.dma_static(dst=s0, src=sp.ap(pattern=[[P, M], [1, P]], offset=pid * n * W))
        SH.dma_static(dst=s1, src=sp.ap(pattern=[[P, M], [1, P]], offset=T * W + pid * n * W))
        nisa.tensor_tensor(dst=s0, data1=s0, data2=s1, op=nl.add, engine=VE)  # s0 + s1, fp32
        y = nl.ndarray((M, P), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y, data1=yr, data2=s0, op=nl.add, engine=VE)  # (r0 + r1) + (s0 + s1), one RNE cast
        SH.dma_static(dst=yh.ap(pattern=[[P, M], [1, P]], offset=pid * n * W), src=y)
        return yh

    @nki.jit
    def mk_ffn_b5_kernel(S, post_p, comb_p, Yp, fnb, base, scale, pnorm, bb, bf, wb, hc: int, iters: int,
                              eps: float, rms_eps: float, sig: int, skf: int, Is: int, K: int, rsc: float, group: int,
                              ring: int, limit: float, rsp: int, o_rt: int, o_gu: int, o_dn: int, o_rb: int, dbg: int,
                              fin: int, rev: int, frev: int = 0):
        """The split form's second half (decode_layer_split): ffn_half_pb's body (its own arguments; frev its REV) and
        the glue after it (_b5), no sum: (S_n, post_f, comb_f, y [1, T, H] this rank's partial). dbg is not read."""
        fo = _ffn_body(S, post_p, comb_p, Yp, fnb, base, scale, pnorm, bb, bf, wb, hc, iters, eps, rms_eps, sig, skf,
                       Is, K, rsc, group, ring, limit, rsp, o_rt, o_gu, o_dn, o_rb, 0, fin, frev)
        return fo[0], fo[1], fo[2], _b5(fo[3], fo[4])
else:
    mk_ffn_b5_kernel = None


# its source with the body it traces (ffn_half_pb's REV) and its helper's (shell's), one CRC
REV = zlib.crc32(repr((SH.kernel_rev(__file__, "# -- the kernel source", "    mk_ffn_b5_kernel = None"), FB.REV,
                       SH.REV)).encode())


# -- the chunked glue (_b5c, mk_ffn_b5c_kernel: the split layer above 8 rows at grid 2; REVC covers to its end) ------
# _b5 takes a program's rows in ONE chunk of n C <= 128 partitions (T <= 8 at grid 2). _b5c is the same glue in chunks
# of u = 128 // C rows, each its own stationary, identity matmul and n_q C lanes, so decode steps of 12 or 16 rows run
# the split layer. Its own kernel and REVC, after the served span: decode_layer_split calls it only when a program's
# rows pass one chunk, so every graph of 8 rows or fewer keeps mk_ffn_b5_kernel and REV.

if nki is not None and mk_ffn_b5_kernel is not None:
    def _b5c(rp, sp):
        """_b5 for any row count: the same y = bf16((r0 + r1) + (s0 + s1)), fp32 throughout and one rounding, per chunk
        of u = 128 // C of this program's rows (its rows' rp sum once, then each chunk's stationary columns (t, k, j),
        one fp32 identity matmul into the chunk layout, sp's two partials and the cast on n_q C lanes). Returns the
        shared [1, T, W] holding this program's rows."""
        G, _, T, C = rp.shape
        W = sp.shape[2]
        npg = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
        pid = nl.program_id(axis=0) if npg > 1 else 0
        n = T // npg
        f32, bf16 = nl.float32, nl.bfloat16
        VE = nisa.vector_engine
        P = 128
        J = C // 2
        u = min(n, P // C)  # rows per chunk: a chunk's rows x C chunks of P on <= 128 partitions
        assert W == C * P and C % 2 == 0 and u >= 1, "_b5c: W = C P, C even, C <= 128"
        yh = nl.ndarray((1, T, W), dtype=bf16, buffer=nl.shared_hbm)
        r0 = nl.ndarray((P, n, J, 2), dtype=f32, buffer=nl.sbuf)  # [.., j, k]: c = 2 j + k, C's own order
        r1 = nl.ndarray((P, n, J, 2), dtype=f32, buffer=nl.sbuf)
        SH.dma_static(dst=r0, src=rp.ap(pattern=[[T * C, P], [C, n], [2, J], [1, 2]], offset=pid * n * C))
        SH.dma_static(dst=r1, src=rp.ap(pattern=[[T * C, P], [C, n], [2, J], [1, 2]],
                                        offset=P * T * C + pid * n * C))
        nisa.tensor_tensor(dst=r0, data1=r0, data2=r1, op=nl.add, engine=VE)  # program 0's + program 1's (sum(0))
        ipi = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)  # the identity [P, P] in fp32
        nisa.iota(dst=ipi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        ipf = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ipf, src=ipi)
        jti = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=jti, pattern=[[1, P]], offset=0, channel_multiplier=0)
        eye = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eye, src=jti)
        nisa.tensor_scalar(dst=eye, data=eye, op0=nl.equal, operand0=ipf)
        for t0 in range(0, n, u):  # chunk q: rows t0 .. t0 + nq - 1 of this program's
            nq = min(u, n - t0)
            M = nq * C
            o = pid * n * W + t0 * W  # the chunk's first row in sp / yh
            rbp = nl.ndarray((P, M), dtype=f32, buffer=nl.sbuf)  # columns (t, k, j), fp32 (no rounding here)
            for t in range(nq):
                for k in range(2):
                    nisa.tensor_copy(dst=rbp[:, t * C + k * J:t * C + (k + 1) * J], src=r0[:, t0 + t, :, k], engine=VE)
            pt = nl.ndarray((M, P), dtype=f32, buffer=nl.psum)  # one accumulation group, its own tile
            nisa.nc_matmul(dst=pt, stationary=rbp, moving=eye, accumulate=False)
            yr = nl.ndarray((M, P), dtype=f32, buffer=nl.sbuf)  # routed r0 + r1, chunk layout, exact
            nisa.tensor_copy(dst=yr, src=pt, engine=VE)
            s0 = nl.ndarray((M, P), dtype=f32, buffer=nl.sbuf)
            s1 = nl.ndarray((M, P), dtype=f32, buffer=nl.sbuf)
            SH.dma_static(dst=s0, src=sp.ap(pattern=[[P, M], [1, P]], offset=o))
            SH.dma_static(dst=s1, src=sp.ap(pattern=[[P, M], [1, P]], offset=T * W + o))
            nisa.tensor_tensor(dst=s0, data1=s0, data2=s1, op=nl.add, engine=VE)  # s0 + s1, fp32
            y = nl.ndarray((M, P), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=y, data1=yr, data2=s0, op=nl.add, engine=VE)  # (r0 + r1) + (s0 + s1), one RNE cast
            SH.dma_static(dst=yh.ap(pattern=[[P, M], [1, P]], offset=o), src=y)
        return yh

    @nki.jit
    def mk_ffn_b5c_kernel(S, post_p, comb_p, Yp, fnb, base, scale, pnorm, bb, bf, wb, hc: int, iters: int,
                          eps: float, rms_eps: float, sig: int, skf: int, Is: int, K: int, rsc: float, group: int,
                          ring: int, limit: float, rsp: int, o_rt: int, o_gu: int, o_dn: int, o_rb: int, dbg: int,
                          fin: int, rev: int, frev: int = 0):
        """mk_ffn_b5_kernel with the chunked glue (_b5c): ffn_half_pb's body, then _b5c, no sum."""
        fo = _ffn_body(S, post_p, comb_p, Yp, fnb, base, scale, pnorm, bb, bf, wb, hc, iters, eps, rms_eps, sig, skf,
                       Is, K, rsc, group, ring, limit, rsp, o_rt, o_gu, o_dn, o_rb, 0, fin, frev)
        return fo[0], fo[1], fo[2], _b5c(fo[3], fo[4])
else:
    mk_ffn_b5c_kernel = None
# -- the chunked glue end

# its own span, and REV (which covers the served span its kernel binds _ffn_body in, and ffn_half_pb's and shell's REVs)
REVC = zlib.crc32(repr((SH.kernel_rev(__file__, "# -- the chunked glue (_b5c", "# -- the chunked glue end"),
                        REV)).encode())
