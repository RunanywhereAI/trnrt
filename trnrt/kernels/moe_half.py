"""GLM-5.3-Flash's MoE FFN half in a decode call, as the FFN half's kernel computes it (kernels/ffn_half_pb.py,
half_body_pb): from the normalised input xn to per-program partials of the routed and the shared experts' outputs. This
module holds its packed decode-only weights, their layout and the torch emulation; the caller adds the partials, rounds
them where the graph rounds, adds the two and all-reduces.

Why: a MoE half cost ~198 us at one decode row in XLA, 42 of them 45% of the step. Of that, ~67 us is XLA (the fp32
router, its top-k, gather and normalisation, the shared expert with its in-graph FP8 dequant, the add), ~14 us the
permutes into and out of the pairs kernel's layout, ~14 us the pairs kernel. Here they are one call: no XLA op and no
layout change between them.

Per program (trn2 at LNC=2: grid 2, two programs; grid 1: one program does all of it):
- x: xn bf16 [T, H] -> xc[i, c, t] = x[t, c 128 + i] by C identity matmuls (instead of the XLA permute).
- router, on both programs (its weight is replicated per rank, and splitting it would need an exchange):
  logits [T, E] = x W_r^T from the router's [k, e] copy (bf16 x bf16 products, exact in fp32; summed in PSUM),
  scores = sigmoid(logits) through the exp table (shell.sigmoid_exp), choice = scores + bias, its -inf / NaN raised
  to finfo.min first; top K = 8 by max8 / nc_find_index8 (descending, equal values at ascending indices), the ids
  clamped to [0, E - 1] (a device top-k over a NaN row can return ids out of range); weights = scores at
  the ids / (sum + 1e-20) * scale, rounded to bf16 (decoder._route, noaux_tc with one group).
- routed experts: moe_tiles_pairs_v1's pairs (kernels/moe_dedupe.py), its instructions, on the same contiguous
  half of the pair groups per program. Each program's fp32 accumulator is its partial, and the caller's p0 + p1 is
  the pairs kernel's own exchange-and-add (each program adds the peer's columns to its own), so with the same
  routing the routed output is the pairs kernel's bits.
- shared expert: hybrid._swiglu's, split by its intermediate rows (Is = 64 per rank at tp 32, Is / npg per program),
  from bf16 copies dequantized at load as model._w reads the FP8 weights: gate_up rounded to bf16 (F.linear's
  output), the gate clamped from above and up into [-limit, limit], silu(gate) rounded to bf16 (F.silu's output),
  times up rounded to bf16, then this program's fp32 partial of the down projection.
Outputs: rp fp32 [npg, 128, T, C] (the pairs kernel's [p, t, g] layout, moe_dedupe.from_pairs) and sp fp32
[npg, T, H]; the caller: y = bf16(rp summed) as [T, H] + bf16(sp summed), in bf16, as _mlp's routed + shared.

Weights: the routed experts are read where they are, layer.w_blob (moe_dedupe's tile layout: no copy). The router's
[k, e] copy and the shared expert's per-program copies are one packed bf16 blob per layer, the router bias one fp32
blob (shell.register_packed, prefix "mh_"): +3.9 MB per MoE layer per rank, ~165 MB at 42 layers.
"""

from __future__ import annotations

import zlib

import torch

from . import moe_dedupe as mdd
from . import rowiso
from . import shell

P = 128
KTOP = 8  # max8: 8 values per partition (GLM-5.3-Flash's num_experts_per_tok)
PREFIX = "mh_"
BF16, F32 = torch.bfloat16, torch.float32
MOE_HALF = True
# the router at grid 2: 0 (served) both programs score every expert (2.36 MB each at GLM's shapes). 1 / 2: each program
# scores its E / 2 experts and takes its own top 8, the two swap those candidates (choice, id, score) by nisa.sendrecv on
# the default DMA engine (1) or the GpSimd's (2), and both take the top 8 of the 16: the same routing (the global top 8
# is within the union of the halves' top 8s; equal choices at ascending ids, program 0's candidates first), half the
# router's bytes per program.
RSPLIT = 0


def shapes(E: int, H: int, Is: int, npg: int) -> dict:
    """The packed decode-only tensors of one layer, {name: (shape, dtype)} (shell.packed_layout's input): the router
    as [i, c, e] = W_r[e, c 128 + i]; each program's shared gate_up rows [i, c, j] = Wgu_p[j, c 128 + i] (its Is / npg
    gate rows, then its Is / npg up rows) and its down columns [j, h] = Wd[h, p Is / npg + j]; the router bias. The
    router's copy is in npg blocks of E / npg experts, so that with the router split (RSPLIT) each program reads its
    own block, and without it both programs read every block."""
    C, ih = H // P, Is // npg
    return {"rt": ((npg, P, C, E // npg), BF16), "gu": ((npg, P, C, 2 * ih), BF16), "dn": ((npg, ih, H), BF16),
            "rb": ((1, E), F32)}


def takes(layer, cfg) -> bool:
    """A layer whose decode MoE half this kernel computes: GLM-5.3-Flash's (sigmoid noaux_tc routing in one group,
    top 8, a clamped SwiGLU shared expert) with the experts in moe_dedupe's tile layout on the device."""
    hy = getattr(cfg, "hybrid", None)
    return (MOE_HALF and getattr(layer, "moe", False) and getattr(layer, "pack_tiles", False)
            and hy is not None and hy.family == "glm5_next"
            and cfg.router_scoring == "sigmoid" and cfg.n_group <= 1 and cfg.num_experts_per_tok == KTOP
            and cfg.norm_topk_prob and getattr(layer, "shared_gate_up", None) is not None
            and getattr(layer, "router_bias", None) is not None and hy.swiglu_limit is not None)


def register(layer, cfg) -> None:
    """Create the layer's packed decode-only blobs (shell.register_packed), when takes(): listed in _decode_only, so
    no prefill or piecewise graph sees them."""
    if not takes(layer, cfg):
        return
    E, H = layer.router.shape
    Is = layer.shared_gate_up.shape[0] // 2
    shell.register_packed(layer, PREFIX, shapes(E, H, Is, shell.programs()))


def fill(layer, dtype: torch.dtype, npg: int | None = None) -> None:
    """The packed blobs from the loaded router, router bias and shared expert (after loader._load_experts and the
    shared expert's load): the shared weights dequantized exactly as model._w reads them (quant.dequant). npg: the
    programs they are laid out for (default shell.programs(); kernels/ffn_half.py pack() passes its own)."""
    from ..models.quant import dequant

    E, H = layer.router.shape
    Is = layer.shared_gate_up.shape[0] // 2
    npg = shell.programs() if npg is None else npg
    ih, C = Is // npg, H // P
    v = shell.packed_views(layer, PREFIX, shapes(E, H, Is, npg))
    rt, eh = router_t(layer.router.data), E // npg  # [P, C, E]
    for p in range(npg):
        v["rt"][p].copy_(rt[:, :, p * eh:(p + 1) * eh])
    w = dequant(layer.shared_gate_up.data, layer.shared_gate_up_scale.data, dtype).to(BF16)  # [2 Is, H]
    d = dequant(layer.shared_down.data, layer.shared_down_scale.data, dtype).to(BF16)  # [H, Is]
    for p in range(npg):
        rows = torch.cat([w[p * ih:(p + 1) * ih], w[Is + p * ih:Is + (p + 1) * ih]])  # [2 ih, H]
        v["gu"][p].copy_(rows.view(2 * ih, C, P).permute(2, 1, 0))
        v["dn"][p].copy_(d[:, p * ih:(p + 1) * ih].t())
    v["rb"].copy_(layer.router_bias.data.float().view(1, E))


def router_t(w_router: torch.Tensor) -> torch.Tensor:
    """The router weight [E, H] in the kernel's layout, rT[i, c, e] = W_r[e, c 128 + i] (bf16), made once at load."""
    E, H = w_router.shape
    return w_router.to(torch.bfloat16).view(E, H // P, P).permute(2, 1, 0).contiguous()


def offsets(E: int, H: int, Is: int, npg: int) -> dict:
    """The kernel's o_rt / o_gu / o_dn / o_rb: each packed tensor's offset in its blob (shell.packed_layout)."""
    lay = shell.packed_layout(shapes(E, H, Is, npg))
    return {"o_" + n: lay[dt][1][n][0] for n, dt in (("rt", BF16), ("gu", BF16), ("dn", BF16), ("rb", F32))}


def emulate(model, layer, x: torch.Tensor) -> torch.Tensor:
    """The served form in torch, from the model's own code: decoder._route, the pairs kernel's arithmetic
    (moe_dedupe.emulate_pairs) and hybrid._swiglu's shared expert; y = routed + shared in bf16 [T, H]."""
    from ..models.hybrid import _swiglu

    lim = model.cfg.hybrid.swiglu_limit
    topv, topi = model._route(layer, x)
    routed = mdd.emulate_pairs(x, topv, topi, layer.w_blob, act=mdd.ACTS["silu_clamp"], limit=lim).to(x.dtype)
    return routed + _swiglu(model, layer, x, "shared_gate_up", "shared_down", lim)


# what the FFN half's kernel folds into its REV for this module's layout and the helpers its MoE body reads
# (shell.dma_static, shell.sigmoid_exp, rowiso's), with the constants
REV = zlib.crc32(repr((shell.REV, rowiso.REV, KTOP, P)).encode())
