"""GLM-5.3-Flash's FFN half of a decode layer after its attention all-reduce, as ONE NKI kernel: the host side. The
kernel is kernels/ffn_half_pb.py's:

    mix_in:   S_n = the attention output's mix of S (kernels/mhc_mix.py's arithmetic, shell.mix_in_h)
    mhc_pre:  x, post, comb of S_n (mhc_pre's statements: mhc_split.front_body, the Sinkhorn in pieces)
    rms_norm: xn = post_norm * bf16(x r), in XLA's device arithmetic for the served rms_norm
    isolate:  each row's NaN / inf in xn zeroed (rowiso; fin) before the identity matmuls over the rows
    MoE:      the router, the shared expert and the routed pairs on xn (ffn_half_pb.half_body_pb, kernels/moe_half.py)

It replaces, per MoE layer of a decode call, the mhc_mix, mhc_pre and moe_half calls and the XLA rms_norm between them.
The caller still adds the programs' partials, rounds them where the graph rounds and all-reduces; the FFN block's own
output mix is the next layer's (the KDA shell's mix_in, or the fused DSA layer's) or the caller's.

Per program (grid 2): the mix, mhc_pre and the norm run unsplit on both programs; the MoE body splits the routed pair
groups and the shared expert's rows between them (kernels/moe_half.py). S_n is written to its HBM output and to this
program's private HBM copy, which mHC's front reads back in its own k-layout. Private copies, so no program reads what
the other wrote.

Outputs: S_n bf16 [T, hc H], post fp32 [T, hc], comb fp32 [T, hc hc], rp fp32 [npg, 128, T, C] (the pairs layout;
run() permutes each program's to [T, H], exact), sp fp32 [npg, T, H]; with dbg, also xn bf16 [T, H], the ids int32
[T, 8] and the weights bf16 [T, 8]; with dbg=2 also the router's choice fp32 [T, E], the exact scores its top 8 read
(sigmoid + bias, raised to -3e38): the ids are its top 8 by value, equal values at ascending ids (max8 /
nc_find_index8). dbg=2 needs rsp 0 (each program then holds every expert's choice). dbg=1's outputs are unchanged.
"""

from __future__ import annotations

import torch

from . import ffn_half_pb as FB
from . import mhc_pre as MP
from . import mhc_split as MS
from . import moe_dedupe as mdd
from . import moe_half as MH
from . import rowiso as RI
from . import shell as SH

P = 128
BF16, F32 = torch.bfloat16, torch.float32


def pack(router, router_bias, shared_gate_up, shared_gate_up_scale, shared_down, shared_down_scale, fn, post_norm,
         base, scale, *, npg: int) -> dict:
    """The kernel's weights for npg programs, from the loaded tensors: {torch.bfloat16: bb, torch.float32: bf} (the
    MoE half's packed blobs, built by moe_half.fill itself, so its bits) plus the FFN hyper-connection as the kernel
    reads it: fn bf16 [2 hc + hc^2, hc H] (mhc_pre's raw layout), post_norm bf16 [H], base fp32 [2 hc + hc^2],
    scale fp32 [3]. The shared expert is FP8 with its block scales, the router bf16, its bias fp32."""
    E, H = router.shape
    Is = shared_gate_up.shape[0] // 2
    layer = torch.nn.Module()
    layer.router, layer.router_bias = router, router_bias
    layer.shared_gate_up, layer.shared_gate_up_scale = shared_gate_up, shared_gate_up_scale
    layer.shared_down, layer.shared_down_scale = shared_down, shared_down_scale
    SH.register_packed(layer, MH.PREFIX, MH.shapes(E, H, Is, npg))
    MH.fill(layer, BF16, npg=npg)
    return {BF16: layer.mh_blob_bf16.data, F32: layer.mh_blob_f32.data, "fn": fn.to(BF16).contiguous(),
            "post_norm": post_norm.to(BF16).contiguous(), "base": base.float().contiguous(),
            "scale": scale.float().contiguous()}


def to_rows(rp: torch.Tensor) -> torch.Tensor:
    """rp fp32 [npg, 128, T, C] (each program's routed partial in the pairs layout) -> [npg, T, H]: moe_dedupe
    .from_pairs' permutation per program, before the programs are added. A permutation commutes with the add and the
    rounding, so add_partials of this is the served from_pairs(add_partials(rp)) bit for bit."""
    npg, _, T, C = rp.shape
    return rp.view(npg, P, T, C // 2, 2).permute(0, 2, 4, 3, 1).reshape(npg, T, C * P)


# the kernel's revision (ffn_half_pb's: its source and its bodies')
REV = FB.REV
# the routed experts' load ring the FFN half's MoE body runs with (its argument `ring`: NR buffers, prefetch distance
# NR - 2): moe_dedupe's RING
RING = mdd.RING


def run(S, pend_a, blobs: dict, w_blob, *, Is: int, K: int, scale: float, group: int, ring: int, limit: float,
        rsp: int, hc: int, iters: int, hc_eps: float, rms_eps: float, simulate: bool = False, grid: int | None = None,
        dbg: int = 0, fin: int | None = None, build_only: bool = False):
    """The kernel in the caller's graph on the device (or in the NKI simulator). S bf16 [T, hc, H]: the streams before
    the attention output's mix; pend_a = (post_a fp32 [T, hc], comb_a fp32 [T, hc, hc], Y_a bf16 [npg, T, H]);
    blobs: pack()'s; w_blob: the layer's routed experts (moe_dedupe's tile layout). Returns (S_n bf16 [T, hc, H],
    post fp32 [T, hc], comb fp32 [T, hc, hc], rp fp32 [npg, T, H], sp fp32 [npg, T, H]) [+ (xn bf16 [T, H], ids int32
    [T, 8], weights bf16 [T, 8]) with dbg] [+ choice fp32 [T, E] with dbg=2]. The caller: y = bf16(bf16(sum rp) +
    bf16(sum sp)) (shell.add_partials of each), then AR2."""
    if fin is None:  # the served form (rowiso.FIN): decode() names none
        fin = RI.FIN
    if FB.ffn_half_kernel is None:
        raise RuntimeError("the NKI FFN half needs the nki package (the Neuron venv)")
    if dbg >= 2 and rsp:
        raise ValueError("ffn_half dbg=2 (the router's choice) needs rsp 0: at rsp 1 / 2 a program holds half the experts")
    if grid is None:
        from .. import platform

        grid = 1 if simulate else platform.nki_grid()
    T, _, H = S.shape
    E = w_blob.shape[0]
    post_a, comb_a, Y_a = pend_a
    args = dict(S=S.to(BF16).reshape(T, hc * H), post_p=post_a.float(), comb_p=comb_a.float().reshape(T, hc * hc),
                Yp=Y_a.to(BF16), fnb=blobs["fn"], base=blobs["base"], scale=blobs["scale"], pnorm=blobs["post_norm"],
                bb=blobs[BF16], bf=blobs[F32], wb=w_blob, hc=int(hc), iters=int(iters), eps=float(hc_eps),
                rms_eps=float(rms_eps), sig=MP.SIGMOID, skf=MP.SK, Is=int(Is), K=int(K), rsc=float(scale),
                group=int(group), ring=int(ring), limit=float(limit), rsp=int(rsp), dbg=int(dbg), fin=int(fin),
                rev=REV, **MH.offsets(E, H, Is, grid))
    if build_only:  # the kernel's arguments, nothing run (the fused kernels compose them with their own)
        return args
    k = FB.ffn_half_kernel
    if simulate:
        import ml_dtypes
        import numpy as np

        nki = FB.nki

        def to_np(t):
            if not isinstance(t, torch.Tensor):
                return t
            return t.float().numpy().astype(ml_dtypes.bfloat16) if t.dtype == BF16 else t.numpy()

        outs = nki.simulate(k[grid] if grid > 1 else k)(**{n: to_np(t) for n, t in args.items()})

        def back(o, dt):
            a = np.asarray(o)
            return torch.from_numpy(a.astype(np.float32)).to(dt) if dt != torch.int32 else torch.from_numpy(
                a.astype(np.int32))

        dts = (BF16, F32, F32, F32, F32, BF16, torch.int32, BF16, F32)
        outs = [back(o, dt) for o, dt in zip(outs, dts)]
    else:
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        outs = list(wrap_nki(k)[grid](**args))
    Sn, post, comb, rp, sp = outs[:5]
    res = (Sn.view(T, hc, H), post, comb.view(T, hc, hc), to_rows(rp), sp)
    return res + tuple(outs[5:]) if dbg else res


# -- in the model's decode graph (models/hybrid.py layer) ------------------------------------------------------------

def decode(model, layer, S, pend_a):
    """A MoE layer's decode FFN block, the attention output's mix included, as one call: (S_n [T, hc, H], post fp32
    [T, hc], comb fp32 [T, hc, hc], y [T, H]), y the FFN output all-reduced (AR2) as _mlp returns it (the
    routed partials permuted and added, the shared ones added, routed + shared in bf16).
    S [T, hc, H]: the streams before the attention output's mix; pend_a = (post_a, comb_a, Y_a [n, T, H]): that mix,
    Y_a the attention output's n all-reduced partial slots (a KDA shell call's npg; 1 otherwise), padded here with
    zero slots to the kernel's programs (mix_in sums them once in fp32: the same y). On host tensors under
    hybrid.SHELL_HOST (tests): the served CPU forms of the same steps."""
    from ..models import hybrid

    if hybrid.SHELL_HOST and S.device.type == "cpu":
        return _host(model, layer, S, pend_a)
    from .. import platform

    T, hc, H = S.shape
    npg = platform.nki_grid()
    post_a, comb_a, Y_a = pend_a
    if Y_a.shape[0] < npg:
        Y_a = torch.cat([Y_a, Y_a.new_zeros(npg - Y_a.shape[0], T, H)])
    blobs, kw = decode_args(model, layer, hc)
    Sn, post, comb, rp, sp = run(S, (post_a, comb_a, Y_a), blobs, layer.w_blob, grid=npg, **kw)
    return Sn, post, comb, decode_out(model, rp, sp, S.dtype)


def decode_args(model, layer, hc: int) -> tuple[dict, dict]:
    """decode()'s blobs and run keywords for a layer (also the fused DSA layer's: the same FFN half inside its kernel)."""
    hy, cfg = model.cfg.hybrid, model.cfg
    blobs = {BF16: layer.mh_blob_bf16, F32: layer.mh_blob_f32, "fn": MS.fn_arg(layer), "post_norm": layer.post_norm,
             "base": layer.hc_ffn_base, "scale": layer.hc_ffn_scale}
    kw = dict(Is=int(layer.shared_gate_up.shape[0] // 2), K=int(cfg.num_experts_per_tok),
              scale=float(cfg.routed_scaling_factor), group=mdd.group_size(1), ring=RING, limit=float(hy.swiglu_limit),
              rsp=MH.RSPLIT, hc=hc, iters=hy.sinkhorn_iters, hc_eps=hy.hc_eps, rms_eps=cfg.rms_norm_eps)
    return blobs, kw


def decode_out(model, rp, sp, dtype):
    """decode()'s y from run's partials: the routed partials (to_rows) and the shared ones added in bf16, then AR2 (model._out_reduce), as _mlp returns it."""
    y = SH.add_partials(rp, BF16) + SH.add_partials(sp, BF16)
    return model._out_reduce(y.to(dtype))


def _host(model, layer, S, pend_a):
    """decode()'s steps on host tensors (hybrid.SHELL_HOST): the served CPU forms (mhc_mix's arithmetic, _mhc,
    decoder.rms_norm, _mlp), with finite_or_zero's emulation between the norm and the MoE half, so a host run with the
    knob matches one without it bit for bit on finite inputs."""
    from ..models import hybrid
    from ..models.decoder import rms_norm
    from . import mhc_mix as MM

    T, hc, H = S.shape
    post_a, comb_a, Y_a = pend_a
    Sn = MM.emulate(post_a, comb_a, Y_a.float().sum(0).to(S.dtype), S).view(T, hc, H)
    x, post, comb = hybrid._mhc(model, layer, "ffn", Sn, True)
    xn = rms_norm(x, layer.post_norm, model.cfg.rms_norm_eps)
    xs = torch.where(torch.isfinite(xn), xn, torch.zeros_like(xn))
    return Sn, post, comb, hybrid._ffn(model, layer, False)(xs)


# A dense layer's decode FFN block goes to kernels/dense_half.py (models/hybrid.py's ffn_half_takes takes it).
from . import dense_half as _DH  # noqa: E402

_decode_moe = decode


def decode(model, layer, S, pend_a):
    """A dense layer's to dense_half.decode; an MoE layer's to the MoE form above."""
    if not getattr(layer, "moe", True):
        return _DH.decode(model, layer, S, pend_a)
    return _decode_moe(model, layer, S, pend_a)
