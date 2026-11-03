"""A dense KDA layer's decode FFN block (GLM-5.3-Flash L0-L2), the attention output's mix included, as ONE kernel:
kernels/ffn_half_pb.py's FFN half with a dense SwiGLU body in place of the MoE body.

In XLA a dense layer's FFN block was two NKI calls (mhc_mix, mhc_pre) around the attention output's slot sum, rms_norm,
the in-graph dequant of 4.7 M FP8 weights, two dots and their layout transposes: 188-191 us of XLA glue a step in
L0-L2. Here:
- the front is ffn_half_pb's kernel's, statement for statement (tests/test_ffn_half.py pins it): shell.mix_in_h,
  mhc_split.front_body (the j-major fn through mhc_split.fn_arg), ffn_half_pb._norm_xc, the served objects imported,
  not copied, so ffn_half_pb's forms act here as there;
- the body is moe_half's shared-expert section (half_body_pb's lines: bf16 where torch rounds, the clamped SwiGLU) at
  this program's ih = Ir / npg intermediate rows, tiled on the partitions in 128-row tiles (ih 192 at tp 32, npg 2:
  128 + 64; one PSUM tile per accumulation group), with mHC's Sinkhorn in three pieces between its stages
  (ffn_half_pb._piece: after gate_up, after a^T, after down), the last writing comb.
Weights: the layer's gate_up / down dequantized once at load (quant.dequant, model._w's arithmetic) into a decode-only
packed blob in moe_half's shared-expert layout, so no prefill graph sees it: +9.4 MB a dense layer a rank.

Numerics: S_n, post and comb are the XLA path's bit for bit (the front is ffn_half_pb's; the Sinkhorn's pieces join
exactly); xn is the FFN halves' norm, as on the MoE layers; y is close, not bit-equal (the PE's K grouping is not XLA's
dots', and the programs' fp32 partials are summed and rounded once).

Wiring: models/hybrid.py's ffn_half_takes takes a dense layer and kernels/ffn_half.decode hands it to decode() here;
decoder.py registers the blob (register: decode-only) and loader.py fills it (fill)."""

from __future__ import annotations

import zlib

import torch

from . import ffn_half_pb as FB
from . import mhc_split as MS
from . import shell as SH

P = 128
BF16, F32 = torch.bfloat16, torch.float32
PREFIX = "dh_"
DENSE_HALF = True
# decode row counts at which the dense half is off, so those buckets' decode graphs take the XLA path (the predicate is
# read at trace time): at two rows the kernel was 54.76 us slower than XLA, while it saved 459.21 us at one
SKIP_T = frozenset((2,))

try:  # the Neuron venv
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


def shapes(H: int, Ir: int, npg: int) -> dict:
    """The packed decode-only tensors of one dense layer, {name: (shape, dtype)}: moe_half's shared-expert layout at
    the layer's Ir rows a rank: each program's gate_up rows [i, c, j] = Wgu_p[j, c 128 + i] (its Ir / npg gate rows,
    then its Ir / npg up rows) and its down columns [j, h] = Wd[h, p Ir / npg + j]."""
    C, ih = H // P, Ir // npg
    return {"gu": ((npg, P, C, 2 * ih), BF16), "dn": ((npg, ih, H), BF16)}


def offsets(H: int, Ir: int, npg: int) -> dict:
    """The kernel's o_gu / o_dn: each packed tensor's offset in the bf16 blob (shell.packed_layout)."""
    lay = SH.packed_layout(shapes(H, Ir, npg))
    return {"o_" + n: lay[BF16][1][n][0] for n in ("gu", "dn")}


def takes(layer, cfg) -> bool:
    """A dense layer whose decode FFN block this kernel computes: GLM-5.3-Flash's dense KDA layers (glm5_next, a clamped
    SwiGLU MLP with FP8 block-scaled gate_up / down)."""
    hy = getattr(cfg, "hybrid", None)
    return (DENSE_HALF and hy is not None and hy.family == "glm5_next" and not getattr(layer, "moe", True)
            and hy.swiglu_limit is not None and getattr(layer, "gate_up_scale", None) is not None
            and getattr(layer, "down_scale", None) is not None)


def register(layer, cfg, Ir: int) -> None:
    """The layer's packed decode-only blob (shell.register_packed), when takes(): listed in _decode_only, so no prefill or
    piecewise graph sees it. Ir: the layer's intermediate rows a rank."""
    if not takes(layer, cfg):
        return
    SH.register_packed(layer, PREFIX, shapes(cfg.hidden_size, Ir, SH.programs()))


def fill(layer, dtype: torch.dtype) -> None:
    """The blob from the loaded FP8 gate_up / down and their scales (after the MLP's load), model._w's dequant."""
    H, Ir = layer.down.shape
    npg = SH.programs()
    fill_views(SH.packed_views(layer, PREFIX, shapes(H, Ir, npg)), layer.gate_up.data, layer.gate_up_scale.data,
               layer.down.data, layer.down_scale.data, dtype, npg)


def fill_views(v: dict, gate_up: torch.Tensor, gate_up_scale, down: torch.Tensor, down_scale, dtype: torch.dtype,
               npg: int) -> None:
    """Write the packed views v (shell.packed_views) from the layer's FP8 gate_up [2 Ir, H] / down [H, Ir] and their
    block scales, dequantized as model._w reads them (quant.dequant), then bf16: moe_half.fill's shared-expert loop."""
    from ..models.quant import dequant

    w = dequant(gate_up, gate_up_scale, dtype).to(BF16)  # [2 Ir, H]
    d = dequant(down, down_scale, dtype).to(BF16)  # [H, Ir]
    Ir, H = w.shape[0] // 2, w.shape[1]
    ih, C = Ir // npg, H // P
    for p in range(npg):
        rows = torch.cat([w[p * ih:(p + 1) * ih], w[Ir + p * ih:Ir + (p + 1) * ih]])  # [2 ih, H]
        v["gu"][p].copy_(rows.view(2 * ih, C, P).permute(2, 1, 0))
        v["dn"][p].copy_(d[:, p * ih:(p + 1) * ih].t())


def pack(gate_up, gate_up_scale, down, down_scale, dtype: torch.dtype, npg: int) -> torch.Tensor:
    """The flat bf16 blob for npg programs (tools and tests; the loader's copy is register_packed + fill_views)."""
    H, Ir = down.shape
    lay = SH.packed_layout(shapes(H, Ir, npg))
    blob = torch.zeros(lay[BF16][0], dtype=BF16)
    v = {}
    for name, (off, shp) in lay[BF16][1].items():
        n = 1
        for d in shp:
            n *= d
        v[name] = blob[off:off + n].view(shp)
    fill_views(v, gate_up, gate_up_scale, down, down_scale, dtype, npg)
    return blob


def emulate(S, pend_a, fn, base, scale, post_norm, gate_up, gate_up_scale, down, down_scale, iters: int, eps: float,
            rms_eps: float, limit: float):
    """The XLA form on host tensors, from the served code: the slot sum and mhc_mix's arithmetic, mhc_split's emulation
    of mhc_pre, decoder.rms_norm, hybrid._swiglu's math on model._w's dequant. (S_n [T, hc, H], post, comb, xn, y [T, H]
    bf16, this rank's partial before AR2)."""
    import torch.nn.functional as F

    from ..models.decoder import rms_norm
    from ..models.quant import dequant
    from . import mhc_mix as MM

    T, hc, H = S.shape
    post_a, comb_a, Y_a = pend_a
    Sn = MM.emulate(post_a, comb_a, Y_a.float().sum(0).to(S.dtype), S).view(T, hc, H)
    x, post, comb = MS.emulate(Sn, fn, base, scale, iters, eps, rms_eps)
    xn = rms_norm(x, post_norm, rms_eps)
    gu = F.linear(xn, dequant(gate_up, gate_up_scale, xn.dtype))
    n = gu.shape[-1] // 2
    g, u = gu[..., :n].clamp(max=limit), gu[..., n:].clamp(min=-limit, max=limit)
    y = F.linear(F.silu(g) * u, dequant(down, down_scale, xn.dtype))
    return Sn, post, comb, xn, y


# -- the kernel source (REV covers from here to dense_half_kernel = None) -----------------------------------------

if nki is not None:
    _norm_xc, _piece = FB._norm_xc, FB._piece  # the served helpers, by the names the kernel calls

    def dense_body(xc, bb, Ir: int, limit: float, o_gu: int, o_dn: int, cp, comb_out, hc: int, iters: int,
                   eps: float, skf: int):
        """This program's ih = Ir / npg rows of the clamped SwiGLU on the k-major input xc bf16 [P, C, T]: moe_half's
        shared-expert section (half_body_pb's lines) with ih in two partition tiles (n0 = min(ih, 128) rows, then the
        rest), the down matmuls accumulating both tiles in one PSUM tile; the Sinkhorn's pieces after gate_up (H0),
        after a^T (H1) and after the down matmuls (H2), the last writing comb_out. Returns sp fp32 [npg, T, H], this
        program's partial written (both programs' before either ends)."""
        f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
        VE = nisa.vector_engine
        C, T = xc.shape[1], xc.shape[2]
        H = C * P
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        ih = Ir // npg
        J = 2 * ih
        n0 = ih if ih < P else P
        n1 = ih - n0
        assert T <= P and J <= 512 and n1 <= P and Ir == ih * npg
        step = (2 * iters - 1 + 2) // 3  # three pieces of the 2 iters - 1 half-rounds
        o_gup = o_gu + pid * P * C * J  # this program's gate_up rows and down columns
        o_dnp = o_dn + pid * ih * H
        # the output first, the same in both programs' traces
        sp = nl.ndarray((npg, T, H), dtype=f32, buffer=nl.shared_hbm)

        # --- loads: this program's gate_up [i, c, j] and its down rows in two partition tiles
        g_sb = nl.ndarray((P, C, J), dtype=bf16, buffer=nl.sbuf)
        SH.dma_static(dst=g_sb, src=bb.ap(pattern=[[C * J, P], [J, C], [1, J]], offset=o_gup))
        d0 = nl.ndarray((n0, H), dtype=bf16, buffer=nl.sbuf)
        SH.dma_static(dst=d0, src=bb.ap(pattern=[[H, n0], [1, H]], offset=o_dnp))
        if n1:
            d1 = nl.ndarray((n1, H), dtype=bf16, buffer=nl.sbuf)
            SH.dma_static(dst=d1, src=bb.ap(pattern=[[H, n1], [1, H]], offset=o_dnp + n0 * H))

        # --- the identity [T, T] (rows 0 .. T - 1 of eye), for a^T
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

        # --- gate_up, the clamped SwiGLU (hybrid._swiglu, bf16 where torch rounds)
        gps = nl.ndarray((T, J), dtype=f32, buffer=nl.psum)
        for c in range(C):
            nisa.nc_matmul(dst=gps, stationary=xc[:, c, :], moving=g_sb[:, c, :], accumulate=(c > 0))
        _piece(cp, comb_out, hc, iters, eps, skf, 0, step)  # H0
        gub = nl.ndarray((T, J), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gub, src=gps, engine=VE)
        gcl = nl.ndarray((T, ih), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gcl, data=gub[:, 0:ih], op0=nl.minimum, operand0=limit, engine=VE)
        ucl = nl.ndarray((T, ih), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ucl, data=gub[:, ih:J], op0=nl.maximum, operand0=-limit, op1=nl.minimum,
                           operand1=limit, engine=VE)
        slf = nl.ndarray((T, ih), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=slf, op=nl.silu, data=gcl)
        slb = nl.ndarray((T, ih), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=slb, src=slf, engine=VE)
        ab = nl.ndarray((T, ih), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ab, data1=slb, data2=ucl, op=nl.multiply, engine=VE)

        # --- a^T per partition tile: the down projection's stationaries
        a0p = nl.ndarray((n0, T), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=a0p, stationary=ab[:, 0:n0], moving=eyeb[0:T, 0:T], accumulate=False)
        a0 = nl.ndarray((n0, T), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=a0, src=a0p, engine=VE)
        if n1:
            a1p = nl.ndarray((n1, T), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=a1p, stationary=ab[:, n0:ih], moving=eyeb[0:T, 0:T], accumulate=False)
            a1 = nl.ndarray((n1, T), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=a1, src=a1p, engine=VE)
        _piece(cp, comb_out, hc, iters, eps, skf, 1, step)  # H1

        # --- down: both tiles accumulated in one PSUM tile per HB columns (512, or all of a narrower H), fp32 out
        HB = H if H < 512 else 512
        ssb = nl.ndarray((T, H), dtype=f32, buffer=nl.sbuf)
        dps = nl.ndarray((T, HB), dtype=f32, buffer=nl.psum)
        for h0 in range(0, H, HB):
            nisa.nc_matmul(dst=dps, stationary=a0, moving=d0[:, h0:h0 + HB], accumulate=False)
            if n1:
                nisa.nc_matmul(dst=dps, stationary=a1, moving=d1[:, h0:h0 + HB], accumulate=True)
            nisa.tensor_copy(dst=ssb[:, h0:h0 + HB], src=dps, engine=VE)
        _piece(cp, comb_out, hc, iters, eps, skf, 2, step)  # H2
        SH.dma_static(dst=sp.ap(pattern=[[H, T], [1, H]], offset=pid * T * H), src=ssb)
        if npg > 1:  # both programs' partials written before either ends
            nisa.core_barrier(data=sp, cores=(0, 1))
        return sp

    @nki.jit
    def dense_half_kernel(S, post_p, comb_p, Yp, fnb, base, scale, pnorm, bb, hc: int, iters: int, eps: float,
                               rms_eps: float, sig: int, skf: int, Ir: int, limit: float, o_gu: int, o_dn: int,
                               fin: int, dbg: int, rev: int):
        """The dense FFN half: (S_out bf16 [T, hc H], post fp32 [T, hc], comb fp32 [T, hc hc], sp fp32 [npg, T, H])
        [+ xn bf16 [T, H] with dbg]. The front is ffn_half_pb.ffn_half_kernel's (tests pin it)."""
        bf16 = nl.bfloat16
        T, F = S.shape
        H = F // hc
        W = H // P
        npg = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
        # outputs first, the same in both programs' traces
        S_out = nl.ndarray((T, F), dtype=bf16, buffer=nl.shared_hbm)
        if dbg:
            xn_out = nl.ndarray((T, H), dtype=bf16, buffer=nl.shared_hbm)

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

        # the dense SwiGLU with the Sinkhorn's pieces at its hooks: the last piece writes comb_out
        sp = dense_body(xc, bb, Ir, limit, o_gu, o_dn, cp, comb_out, hc, iters, eps, skf)
        if dbg:  # both programs write the same values
            SH.dma_static(dst=xn_out.ap(pattern=[[W, P], [H, T], [1, W]], offset=0), src=xn)  # xn: h-layout
            return S_out, post_out, comb_out, sp, xn_out
        return S_out, post_out, comb_out, sp
else:
    dense_body = dense_half_kernel = None


# its source and ffn_half_pb's REV (which covers the front's objects: shell, mhc_split, the norm), as one CRC
REV = zlib.crc32(repr((SH.kernel_rev(__file__, "# -- the kernel source", "    dense_body = dense_half_kernel = None"),
                       FB.REV, P)).encode())


def run(S, pend_a, fn, base, scale, post_norm, bb, *, Ir: int, limit: float, hc: int, iters: int, hc_eps: float,
        rms_eps: float, simulate: bool = False, grid: int | None = None, dbg: int = 0, fin: int | None = None):
    """The kernel in the caller's graph on the device (or in the NKI simulator). S bf16 [T, hc, H]: the streams before
    the attention output's mix; pend_a = (post_a fp32 [T, hc], comb_a fp32 [T, hc, hc], Y_a bf16 [npg, T, H]);
    fn: mhc_split.fn_arg's (the j-major copy); bb: pack()'s bf16 blob. Returns (S_n bf16
    [T, hc, H], post fp32 [T, hc], comb fp32 [T, hc, hc], sp fp32 [npg, T, H]) [+ xn bf16 [T, H] with dbg]. The caller:
    y = shell.add_partials(sp, bf16), then AR2."""
    from . import mhc_pre as MP
    from . import rowiso as RI

    if fin is None:  # the served form (rowiso.FIN)
        fin = RI.FIN
    if dense_half_kernel is None:
        raise RuntimeError("the NKI dense FFN half needs the nki package (the Neuron venv)")
    if grid is None:
        from .. import platform

        grid = 1 if simulate else platform.nki_grid()
    T, _, H = S.shape
    post_a, comb_a, Y_a = pend_a
    args = dict(S=S.to(BF16).reshape(T, hc * H), post_p=post_a.float(), comb_p=comb_a.float().reshape(T, hc * hc),
                Yp=Y_a.to(BF16), fnb=fn.to(BF16).contiguous(), base=base.float().contiguous(),
                scale=scale.float().contiguous(), pnorm=post_norm.to(BF16).contiguous(), bb=bb, hc=int(hc),
                iters=int(iters), eps=float(hc_eps), rms_eps=float(rms_eps), sig=MP.SIGMOID, skf=MP.SK, Ir=int(Ir),
                limit=float(limit), fin=int(fin), dbg=int(dbg), rev=REV, **offsets(H, Ir, grid))
    k = dense_half_kernel
    if simulate:
        import ml_dtypes
        import numpy as np

        def to_np(t):
            if not isinstance(t, torch.Tensor):
                return t
            return t.float().numpy().astype(ml_dtypes.bfloat16) if t.dtype == BF16 else t.numpy()

        outs = nki.simulate(k[grid] if grid > 1 else k)(**{n: to_np(t) for n, t in args.items()})
        dts = (BF16, F32, F32, F32, BF16)
        outs = [torch.from_numpy(np.asarray(o).astype(np.float32)).to(dt) for o, dt in zip(outs, dts)]
    else:
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        outs = list(wrap_nki(k)[grid](**args))
    Sn, post, comb, sp = outs[:4]
    res = (Sn.view(T, hc, H), post, comb.view(T, hc, hc), sp)
    return res + tuple(outs[4:]) if dbg else res


# -- in the model's decode graph (models/hybrid.py layer) ------------------------------------------------------------

def decode(model, layer, S, pend_a):
    """A dense layer's decode FFN block, the attention output's mix included, as one call: (S_n [T, hc, H], post fp32
    [T, hc], comb fp32 [T, hc, hc], y [T, H]), y the FFN output all-reduced (AR2) as _mlp returns it (the programs' fp32
    partials summed, one bf16 rounding). S [T, hc, H]: the streams before the attention output's mix; pend_a = (post_a,
    comb_a, Y_a [n, T, H]): the KDA shell call's slots, padded with zero slots to the kernel's programs (mix_in sums
    them once in fp32: the same y). On host tensors under hybrid.SHELL_HOST (tests): the CPU forms."""
    from ..models import hybrid

    hy, cfg = model.cfg.hybrid, model.cfg
    T, hc, H = S.shape
    if hybrid.SHELL_HOST and S.device.type == "cpu":
        from . import mhc_mix as MM
        from ..models.decoder import rms_norm

        post_a, comb_a, Y_a = pend_a
        Sn = MM.emulate(post_a, comb_a, Y_a.float().sum(0).to(S.dtype), S).view(T, hc, H)
        x, post, comb = hybrid._mhc(model, layer, "ffn", Sn, True)
        xn = rms_norm(x, layer.post_norm, cfg.rms_norm_eps)
        xs = torch.where(torch.isfinite(xn), xn, torch.zeros_like(xn))  # the kernel's row isolation (fin)
        return Sn, post, comb, hybrid._ffn(model, layer, False)(xs)
    from .. import platform

    npg = platform.nki_grid()
    post_a, comb_a, Y_a = pend_a
    if Y_a.shape[0] < npg:
        Y_a = torch.cat([Y_a, Y_a.new_zeros(npg - Y_a.shape[0], T, H)])
    Sn, post, comb, sp = run(S, (post_a, comb_a, Y_a), MS.fn_arg(layer), layer.hc_ffn_base, layer.hc_ffn_scale,
                             layer.post_norm, layer.dh_blob_bf16, Ir=int(layer.down.shape[1]),
                             limit=float(hy.swiglu_limit), hc=hc, iters=hy.sinkhorn_iters, hc_eps=hy.hc_eps,
                             rms_eps=cfg.rms_norm_eps, grid=npg)
    return Sn, post, comb, model._out_reduce(SH.add_partials(sp, BF16).to(S.dtype))
