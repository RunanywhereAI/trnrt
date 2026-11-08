"""GLM-5.3-Flash's manifold-constrained hyper-connections (models/glm5_next.py).

The model keeps hc = 4 residual STREAMS per token instead of one hidden state. Around every block
(token mixer, then MLP) the streams are collapsed into the block's input and the block's output
is mixed back into them (Glm5NextTextHyperConnection): from the RMS-normalised concatenated
streams one linear map gives pre [hc], post [hc] and comb [hc, hc] logits; pre = sigmoid(...) + eps
collapses the streams (sum_n pre_n S_n), the block runs on that after its usual RMSNorm, and the
new streams are post_n y + sum_m comb_mn S_m, comb made doubly stochastic by hc_sinkhorn_iters
Sinkhorn-Knopp normalisations. The final hidden state is the plain mean of the streams, then the
model norm.

Numerics: transformers v5.18.0 models/glm5_next/modeling_glm5_next.py (URL in models/glm5_next.py).
The stream arithmetic is replicated on every tensor-parallel rank in a decode call; only the
blocks inside are sharded. Prefill chunks keep sequence-parallel streams (below).

The hidden state between graphs is [T, hc * H] (the streams of a token side by side).
DecoderForCausalLM hands every layer to `layer` below and its final normalisation to `final`.

Decode forms. A decode step is one graph, and on the device most of a layer runs as a few NKI
calls (docs/MEGAKERNELS.md):
- the KDA shell (kernels/kda_shell.py): a KDA layer's attention block, mhc_pre, the block's
  rms_norm and the KDA half in one call, the two programs' bf16 partials all-reduced as one
  collective; its weights are two packed decode-only blobs per layer;
- the FFN half (kernels/ffn_half.py): a layer's FFN block, the attention output's mix included,
  as one call (the MoE half of kernels/moe_half.py for an MoE layer, kernels/dense_half.py for a
  dense one);
- the mix-in: the previous layer's FFN mix moves into the next KDA layer's shell call (its pend
  argument), so one mhc_mix call per seam goes. The FFN block returns its streams unmixed with
  (post, comb, y), DecoderForCausalLM._run_layers carries them, and the next shell mixes them
  first;
- the fused KDA layer (kernels/mk_kda_layer.py): the shell call and the FFN half of an MoE KDA
  layer as two kernels with the model's own sums between them;
- the fused DSA layer (models/mla.py): a DSA layer's attention block and its FFN half as one call,
  with the attention-block Sinkhorn left to that call (kernels/mhc_front.py), which runs it under
  its own collective waits.
Every one of them is decode-only: their weights are listed in layer._decode_only, so no prefill
graph sees them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..kernels import dense_half as _DH
from ..kernels import shell as SH
from .config import LinearSpec
from .decoder import rms_norm

ARCHITECTURES = ("Glm5NextForConditionalGeneration",)


@dataclass(frozen=True)
class HybridSpec:
    family: str  # "glm5_next"
    hc: int  # residual streams (hc_mult)
    sinkhorn_iters: int = 0  # hc_sinkhorn_iters
    hc_eps: float = 0.0  # hc_eps
    swiglu_limit: float | None = None  # gate clamped from above, up to [-limit, limit]
    block_norms: bool = True  # input_layernorm / post_attention_layernorm / model.norm exist


# -- parameters --------------------------------------------------------------------------------


def _f32(*shape):
    return nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False)


def init_layer(layer, cfg, spec, index: int, tp: int, p) -> None:
    """The hyper-connection parameters of one layer (both blocks), on the DecoderLayer (p: a parameter in the model
    dtype)."""
    hy, H = cfg.hybrid, cfg.hidden_size
    hc = hy.hc
    n = (2 + hc) * hc  # pre, post, comb logits
    for b in ("attn", "ffn"):
        setattr(layer, f"hc_{b}_fn", p(n, hc * H))
        setattr(layer, f"hc_{b}_base", _f32(n))
        setattr(layer, f"hc_{b}_scale", _f32(3))
    # The FFN half's fn, j-major (kernels/mhc_fnj.py fn_j), filled at load. Decode-only, so no prefill graph's tensor
    # list (decoder._layer_split) shifts.
    layer.hc_ffn_fnj = p(128, hc * H // 128, n)
    layer._decode_only = tuple(getattr(layer, "_decode_only", ())) + ("hc_ffn_fnj",)
    if isinstance(spec, LinearSpec):  # the KDA shell's packed blobs
        from ..kernels import kda_half, kda_shell

        kda_shell.register(layer, cfg, spec, kda_half.half_programs())


def load_layer(layer, ck, pre: str, cfg, r: int, n: int, dtype) -> None:
    """Checkpoint names: zai-org/GLM-5.3-Flash model.safetensors.index.json (hc_attn_fn / _base / _scale,
    hc_ffn_*)."""
    for b in ("attn", "ffn"):
        getattr(layer, f"hc_{b}_fn").data.copy_(ck.get(f"{pre}hc_{b}_fn").to(dtype))
        getattr(layer, f"hc_{b}_base").data.copy_(ck.get(f"{pre}hc_{b}_base").float())
        getattr(layer, f"hc_{b}_scale").data.copy_(ck.get(f"{pre}hc_{b}_scale").float())
    from ..kernels import mhc_fnj

    layer.hc_ffn_fnj.data.copy_(mhc_fnj.fn_j(layer.hc_ffn_fn.data))  # the FFN half's copy of the FFN block's fn
    if getattr(layer, "ks_blob_bf16", None) is not None:  # the KDA shell's packed blobs
        from ..kernels import kda_shell

        kda_shell.fill(layer)


# -- forward -----------------------------------------------------------------------------------


def hidden_in(model, input_ids: torch.Tensor, sp: bool = False) -> torch.Tensor:
    """The initial hidden state [T, hc * H]: every stream starts as the token embedding (Glm5NextTextModel: expand).
    sp: this rank's rows only (sequence-parallel prefill streams, models/decoder.py)."""
    e = model._embed(input_ids)
    if sp:
        e = model._sp_rows(e)
    return e.repeat(1, model.cfg.hybrid.hc)


def final(model, h: torch.Tensor) -> torch.Tensor:
    """The final normalised hidden state [T, H] the lm_head reads: Glm5NextTextHyperHead (an unweighted mean), then
    the model norm."""
    cfg = model.cfg
    hy, H = cfg.hybrid, cfg.hidden_size
    T = h.shape[0]
    streams = h[:, : hy.hc * H]
    return rms_norm(streams.view(T, hy.hc, H).mean(dim=1), model.norm, cfg.rms_norm_eps)


# How the hyper-connection stream arithmetic is written: vLLM v0.30.0's numerics
# (vllm/model_executor/kernels/mhc/torch.py, mhc_pre_torch / mhc_post_torch) with no [T, hc * H] fp32 temporary and no
# per-token batched matmul. The RMS scale multiplies the [T, 2 hc + hc^2] logits after the fp32 projection (one scalar
# per token, (x r) W^T = (x W^T) r), the collapse is hc per-token multiply-adds of [T, H] rows in fp32, and each output
# stream post_n y + sum_m comb[m, n] S_m is accumulated in fp32 from [T, H] rows. The collapse and the streams are
# rounded to the model dtype through _round: a plain .to(bf16) between fp32 ops is folded away inside a device graph
# (XLA allows excess precision), so the streams would stay fp32 across a whole layer group and the device would score
# differently from the CPU. transformers' own spelling (a broadcast product summed over the streams, and the output as
# a matmul of comb with the streams in the model dtype) made neuronx-cc spill the prefill layer graphs on trn1 (layer 0
# at C=512, 8 heads per rank: 10.7 ms, about 80 ms once any NKI call sits in the mixer, against 3.8 ms).

# The KDA shell takes decode steps of at most this many rows; the XLA half runs above. Its per-row recurrence costs
# ~8.5-10.5 us per row x 34 layers, which a batched recurrence could lift. The engine's --kda-shell-max-rows
# (EngineConfig.kda_shell_max_rows, set on every rank by engine.build_shard before it builds the shard; 8 here and
# under --preset batch8, 12 served, so every decode step up to 12 rows takes the shell and the split layer).
KDA_SHELL_MAXB = 8


def kda_shell_rows(rows: int) -> bool:
    """Whether a decode step of `rows` rows takes the KDA shell (else the XLA half)."""
    return rows <= KDA_SHELL_MAXB


# SHELL_HOST (tests only, never read from the environment): the shells' branches on host tensors too, their calls
# computed by the kernels' emulations (kda_shell.emulate, ffn_half / dense_half's host forms).
SHELL_HOST = False


def kda_shell_takes(model, layer, streams, state_slot, dec: bool, sp: bool) -> bool:
    """Whether a layer's attention block runs as the KDA shell's call: a decode call (dec) of a KDA layer with its packed
    blobs, replicated streams (not sp), a state row per row, at most KDA_SHELL_MAXB rows, on the device (or the host,
    SHELL_HOST / shell.HOST_SIM). streams: any tensor with the call's rows and device. layer() and _run_layers (through
    defers) both ask this."""
    return (dec and not sp and getattr(layer, "ks_blob_bf16", None) is not None
            and (streams.device.type != "cpu" or SHELL_HOST or SH.HOST_SIM) and state_slot is not None
            and state_slot.dim() == 1
            and state_slot.shape[0] == streams.shape[0] and kda_shell_rows(streams.shape[0]))


def call_kind(model, h, positions, table) -> tuple:
    """(sp, dec) of a layer call over h, as layer() reads them: sequence-parallel prefill streams and a decode call (a
    block table per row)."""
    sp = (table is not None and h.shape[0] * model.tp_size == positions.shape[0]
          and table.dim() == 1 and model._sp_on())
    dec = table is not None and table.dim() == 2
    return sp, dec


def defers(model, layer, nxt, h, positions, table, bias, state_slot) -> bool:
    """Whether `layer` leaves its FFN mix to `nxt`, the next layer, in this call (h the call's hidden state, table /
    bias nxt's; bias is not read): a decode call whose next attention block is a KDA shell call (the mix-in)."""
    sp, dec = call_kind(model, h, positions, table)
    return kda_shell_takes(model, nxt, h, state_slot, dec, sp)


# Decode calls of fewer rows than this take the FFN half (kernels/ffn_half.py): moe_dedupe's pairs kernel's bound. The
# engine's --ffn-half-below (EngineConfig.ffn_half_below, set by engine.build_shard; 16 here and under --preset
# batch8, 17 served).
MOE_HALF_BELOW = 16
# A decode MoE block that is not inside the FFN half's call (_mlp: the plain MTP layer's) runs as moe_half's kernel below
# MOE_HALF_BELOW rows.
MOE_HALF = True


def ffn_half_takes(model, layer, streams, dec: bool, sp: bool) -> bool:
    """Whether a layer's decode FFN block, with the attention output's mix, runs as one FFN-half call: a decode call,
    replicated streams (not sp), fewer than MOE_HALF_BELOW rows, and either an MoE layer (with moe_half's packed blobs)
    or a dense KDA layer at a row count the dense half takes (dense_half.SKIP_T: at 2 rows the XLA FFN is faster), with
    the dense half's blob. On the host only with SHELL_HOST (tests: the emulations need no blobs). streams: any tensor
    with the call's rows and device."""
    if not (dec and not sp and streams.shape[0] < MOE_HALF_BELOW):
        return False
    if getattr(layer, "moe", False):
        blob = "mh_blob_bf16"
    elif (isinstance(layer.spec, LinearSpec) and model.cfg.hybrid.swiglu_limit is not None
          and streams.shape[0] not in _DH.SKIP_T):
        blob = "dh_blob_bf16"
    else:
        return False
    if streams.device.type == "cpu" and not SH.HOST_SIM:
        return SHELL_HOST
    return getattr(layer, blob, None) is not None


def _attn_unmixed(model, layer, streams, fn, dec: bool):
    """_block's attention block without its final mix_out: (S [T, hc, H], (post, comb, y [1, T, H])), the mix left to
    the FFN half's call (its mix_in, from one partial slot)."""
    hy, H = model.cfg.hybrid, model.cfg.hidden_size
    S = streams.view(streams.shape[0], hy.hc, H)
    x, post, comb = _mhc(model, layer, "attn", S, dec)
    y = fn(rms_norm(x, layer.in_norm, model.cfg.rms_norm_eps))
    return S, (post, comb, y.unsqueeze(0))


def _shell_pend(layer, pend):
    """A deferred (post, comb, y) as kda_shell's mix_in takes it: y in the first of the call's npg partial slots and
    zeros in the rest (mix_in adds them in fp32 and rounds once, so y exactly)."""
    if pend is None:
        return None
    post, comb, y = pend
    npg = layer.ks_dims[3]
    Yp = y.unsqueeze(0) if npg == 1 else torch.cat([y.unsqueeze(0), y.new_zeros(npg - 1, *y.shape)])
    return post, comb, Yp


def _mk4_rows_on(rows: int) -> bool:
    """The fused DSA layer's row cap (models/mla.py _mk_rows_on): layer()'s fused-DSA branch only where the layer call
    can take, so a step past the cap keeps the plain path's code."""
    from . import mla as _mla

    return _mla._mk_rows_on(rows)


def _round(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """x.to(dtype); with a bf16 dtype, x is first rounded to bf16's 8 significant bits in fp32 arithmetic (Veltkamp's
    split: c = (2^16 + 1) x, hi = c - (c - x)), so the value is already bf16-representable whether or not the compiler
    keeps the convert (a convert pair f32 -> bf16 -> f32 between elementwise ops is foldable, and a bitcast view to int16
    is rejected inside an LNL graph: "Expected all tensors ... to be XLA tensors. Got: XLABFloat16Type")."""
    if dtype == torch.bfloat16 and x.dtype == torch.float32:
        c = x * 65537.0
        x = c - (c - x)
    return x.to(dtype)


def _mhc(model, layer, b: str, S: torch.Tensor, decode: bool = False):
    """Glm5NextTextHyperConnection.forward on streams S [T, hc, H]: (block input [T, H], post [T, hc] fp32, comb
    [T, hc, hc] fp32). decode: a decode call's (layer(): a block table per row), the only kind that takes the whole of
    it as one kernel (kernels/mhc_pre.py), so no prefill graph moves."""
    hy = model.cfg.hybrid
    hc, eps = hy.hc, hy.hc_eps
    T = S.shape[0]
    from ..kernels import mhc_pre

    fn = getattr(layer, f"hc_{b}_fn")
    if decode and mhc_pre.takes(S.device, T, S.shape[-1], hc, fn.dtype):
        return mhc_pre.pre(S, fn, getattr(layer, f"hc_{b}_base"), getattr(layer, f"hc_{b}_scale"),
                           hy.sinkhorn_iters, eps, model.cfg.rms_norm_eps)
    flat = S.reshape(T, -1)
    r = torch.rsqrt(flat.float().pow(2).mean(-1, keepdim=True) + model.cfg.rms_norm_eps)
    mix = F.linear(flat.float(), getattr(layer, f"hc_{b}_fn").float()) * r  # (x r) W^T = (x W^T) r
    base, scale = getattr(layer, f"hc_{b}_base"), getattr(layer, f"hc_{b}_scale")
    pre = torch.sigmoid(mix[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mix[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = torch.softmax(mix[:, 2 * hc :].view(T, hc, hc) * scale[2] + base[2 * hc :].view(hc, hc), dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)  # Sinkhorn-Knopp: columns, then rows and columns
    for _ in range(hy.sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    x = pre[:, 0:1] * S[:, 0].float()
    for n in range(1, hc):
        x = x + pre[:, n : n + 1] * S[:, n].float()
    return _round(x, S.dtype), post, comb


def mix_out(post: torch.Tensor, comb: torch.Tensor, y: torch.Tensor, S: torch.Tensor,
            decode: bool = False) -> torch.Tensor:
    """The new streams [T, hc * H] from the block output y [T, H], the old streams S [T, hc, H], post [T, hc] and comb
    [T, hc, hc] (fp32): stream n = post_n y + sum_m comb[m, n] S_m. A decode call's runs as one kernel on the device
    (kernels/mhc_mix.py)."""
    T, hc, H = S.shape
    from ..kernels import mhc_mix

    if decode and mhc_mix.takes(S.device, H, T):
        return mhc_mix.mix(post, comb, y, S)
    yf, outs = y.float(), []
    for n in range(hc):  # stream n: post_n y + sum_m comb[m, n] S_m, in fp32
        o = post[:, n : n + 1] * yf
        for m in range(hc):
            o = o + comb[:, m, n : n + 1] * S[:, m].float()
        outs.append(_round(o, S.dtype))
    return torch.cat(outs, dim=-1)


# With sequence-parallel prefill streams, an MoE layer's router runs on each rank's own R / tp rows and the [R, 2k]
# routing is gathered like the rows themselves (_sp_route), instead of every rank routing all R gathered rows (1.13 ms of
# an MoE block at R = 4096 on trn1). The per-row arithmetic is the same; only the router matmul's row count differs.
# A block's output reduction (the MLP's all-reduce, the token mixer's) is a world reduce-scatter that leaves each rank
# its own rows, instead of the all-reduce of every row and DecoderForCausalLM._sp_rows: the same sums, the collective's
# own summation order may differ in the last bf16 bit.


def _sp_route(model, layer, x: torch.Tensor):
    """(topv, topi) of every rank's rows, in rank order, from this rank's rows x [r, H]: routed here and gathered as one
    fp32 [R, 2k] tensor (DecoderForCausalLM._sp_gather: one rank's value plus zeros per element, so the weights and
    the expert indices, integers below 2^24, come back exactly)."""
    topv, topi = model._route(layer, x)
    k = topi.shape[-1]
    both = model._sp_gather(torch.cat([topv.float(), topi.float()], dim=-1))
    return both[:, :k].to(topv.dtype), both[:, k:].to(topi.dtype)


def _sp_out(model, fn, rs: bool):
    """With sequence-parallel streams, a block fn over every rank's rows y [R, ...] -> this rank's rows of its output:
    its output reduction a reduce-scatter (rs) or the all-reduce and _sp_rows."""
    if rs:
        return lambda y, *a: model._sp_rs_call(fn, y, *a)
    return lambda y, *a: model._sp_rows(fn(y, *a))


def _attn_rows(model, fn, sp: bool):
    """The token mixer's block fn for _block: fn, or with sequence-parallel streams (sp) on this rank's rows: every row
    gathered and the output reduce-scattered (attention TP = TP) or all-reduced and this rank's taken."""
    if not sp:
        return fn
    out = _sp_out(model, fn, model._sp_rs_attn())
    return lambda x: out(model._sp_gather(x))


def _ffn(model, layer, sp: bool):
    """The FFN block's fn for _block: _mlp, or with sequence-parallel streams (sp) on this rank's rows, gathering them,
    keeping its own rows of the output, and (an MoE layer) routing them here."""
    if not sp:
        return lambda x: _mlp(model, layer, x)
    out = _sp_out(model, lambda y, r=None: _mlp(model, layer, y, r), True)
    if layer.moe and model.cfg.hybrid.swiglu_limit is not None:
        return lambda x: out(model._sp_gather(x), _sp_route(model, layer, x))
    return lambda x: out(model._sp_gather(x))


def _block(model, layer, b: str, streams: torch.Tensor, norm_w, fn, decode: bool = False) -> torch.Tensor:
    """streams [T, hc * H] after one block fn(x) -> [T, H] wrapped in its hyper-connection (decode: _mhc's)."""
    hy, H = model.cfg.hybrid, model.cfg.hidden_size
    T = streams.shape[0]
    S = streams.view(T, hy.hc, H)
    x, post, comb = _mhc(model, layer, b, S, decode)
    y = fn(rms_norm(x, norm_w, model.cfg.rms_norm_eps))
    return mix_out(post, comb, y, S, decode)


def _mix(model, layer, x, positions, slot_mapping, table, bias, state_slot, seq, ffn=None):
    """The token mixer on its input x [T, H], after its attention-group all-reduce. ffn (the fused DSA layer, layer()):
    (S, post, comb) of the attention block, for a fused DSA layer that runs the FFN half too: its (S_n, post, comb, y)
    tuple is returned as is (its world sum is inside the call); any other result is all-reduced as always."""
    from . import linear_attn
    from . import mla as _mla

    sp = layer.spec
    if isinstance(sp, LinearSpec):
        return linear_attn.mix(model, layer, x, positions, slot_mapping, state_slot)
    if seq is not None:
        return model._attn_all_reduce(_mla.reference(model, layer, x, positions, seq))
    if ffn is not None:  # the fused DSA layer may take the FFN half too
        r = _mla.attention(model, layer, x, positions, slot_mapping, table, bias, ffn=ffn)
        return r if isinstance(r, tuple) else model._attn_all_reduce(r)
    return model._attn_all_reduce(_mla.attention(model, layer, x, positions, slot_mapping, table, bias))


def _swiglu(model, layer, x, gate_up: str, down: str, limit: float | None) -> torch.Tensor:
    gu = F.linear(x, model._w(layer, gate_up))
    n = gu.shape[-1] // 2
    g, u = gu[..., :n], gu[..., n:]
    if limit is not None:  # Glm5NextTextMLP / Glm5NextTextExperts._apply_gate
        g, u = g.clamp(max=limit), u.clamp(min=-limit, max=limit)
    return F.linear(F.silu(g) * u, model._w(layer, down))


def _moe_clamped(model, layer, x: torch.Tensor, limit: float, route=None) -> torch.Tensor:
    """The routed experts with Glm5NextTextExperts' clamped SwiGLU. route: (topv, topi) of x's rows made elsewhere
    (_sp_route), instead of routing x here.

    On the device a packed layer always runs an NKI kernel: the XLA paths cannot read the blob there (LNL rejects the
    in-graph uint8 -> float8 view the unpack needs), so they serve packed layers on the CPU only. From
    MOE_PREFILL_MIN_TOKENS tokens on that is kernels/moe_prefill_tsb.py (a grouped GEMM over the whole chunk), below it
    kernels/moe_dedupe.py."""
    cfg = model.cfg
    topv, topi = model._route(layer, x) if route is None else route
    T, H = x.shape
    k, Im = cfg.num_experts_per_tok, model.moe_inter

    if getattr(layer, "moe_blob", False) and x.device.type != "cpu":  # the NKI kernels, clamped
        from ..kernels.moe_dedupe import ACTS, moe_dedupe
        from .decoder import MOE_PREFILL_MIN_TOKENS, moe_prefill_down

        if T >= MOE_PREFILL_MIN_TOKENS:  # the fused stage-C drain with the token scatter built in SBUF
            from ..kernels.moe_prefill_tsb import moe_prefill_tsb

            return moe_prefill_tsb(x, topv, topi, layer.w_blob, act=ACTS["silu_clamp"], limit=limit,
                                   down=moe_prefill_down(layer))
        return moe_dedupe(x, topv, topi, layer.w_blob, act=ACTS["silu_clamp"], limit=limit)

    def act(gu):
        return F.silu(gu[..., :Im].clamp(max=limit)) * gu[..., Im:].clamp(min=-limit, max=limit)

    if T * k <= model.MOE_GATHER_MAX_PAIRS:
        flat = topi.reshape(T * k)
        xs = x.unsqueeze(1).expand(T, k, H).reshape(T * k, H, 1)
        a = act(torch.bmm(model._experts(layer, "w_gu", flat), xs).squeeze(-1))
        y = torch.bmm(a.unsqueeze(1), model._experts(layer, "w_down", flat)).squeeze(1)
        return (y.view(T, k, H) * topv.unsqueeze(-1)).sum(dim=1)
    a = act(torch.einsum("th,eih->eti", x, model._experts(layer, "w_gu")))
    ye = torch.einsum("eti,eih->eth", a, model._experts(layer, "w_down"))
    w = torch.zeros(T, cfg.num_experts, dtype=model.dtype, device=x.device).scatter(1, topi, topv)
    return torch.einsum("eth,te->th", ye, w)


def _mlp(model, layer, x: torch.Tensor, route=None) -> torch.Tensor:
    """The MLP block on its input x [T, H], after its all-reduce; route: _moe_clamped's."""
    lim = model.cfg.hybrid.swiglu_limit
    if not layer.moe:
        return model._out_reduce(_swiglu(model, layer, x, "gate_up", "down", lim))
    if (MOE_HALF and route is None and getattr(layer, "mh_blob_bf16", None) is not None
            and (x.device.type != "cpu" or SH.HOST_SIM) and x.shape[0] < MOE_HALF_BELOW):
        # A decode MoE block outside the FFN half's call (the plain MTP layer's: the hyper-connection layers' decode
        # MoE runs inside ffn_half_takes' call) as moe_half's one kernel call.
        from ..kernels import moe_half

        return model._out_reduce(moe_half.decode_half(model, layer, x))
    y = _moe_clamped(model, layer, x, lim, route)
    if layer.shared_gate_up is not None:  # mlp.shared_experts, clamped like the experts
        y = y + _swiglu(model, layer, x, "shared_gate_up", "shared_down", lim)
    return model._out_reduce(y)


def layer(model, layer, h, positions, slot_mapping, table, bias, state_slot=None, seq: dict | None = None, *,
          pend=None, defer: bool = False):
    """One decoder layer over the hidden state h [T, hc * H], every batch form of DecoderForCausalLM._layer; seq (a
    dict, MLA's top-k carrier) for the whole-sequence reference form without a cache.
    pend / defer (the mix-in, decode calls; DecoderForCausalLM._run_layers): pend = (post, comb, y), the previous
    layer's FFN block left unmixed, which this layer's KDA shell call mixes in first (or this layer mixes before its
    attention block, if it takes no shell call); defer: return (h with the streams unmixed, (post, comb, y)) of this
    layer's FFN block instead of mixing it."""
    hy, H = model.cfg.hybrid, model.cfg.hidden_size
    W = hy.hc * H
    streams = h[:, :W]
    # Sequence-parallel prefill streams (models/decoder.py): h holds this rank's R / tp of a chunk's R rows. The
    # hyper-connections run on those rows; each block gathers every rank's normalised input, runs over all R rows exactly
    # as with replicated streams (its own reduction included) and keeps this rank's rows of the output.
    # sp and dec inline, as call_kind computes them: through a helper call dynamo realizes its tensor arguments in the
    # call's order (positions before table), which reorders a prefill graph's inputs.
    sp = (seq is None and table is not None and h.shape[0] * model.tp_size == positions.shape[0]
          and table.dim() == 1 and model._sp_on())

    def rows(fn):
        return _attn_rows(model, fn, sp)

    dec = seq is None and table is not None and table.dim() == 2  # a decode call (_mhc's kernel)

    # The KDA shell's guard keeps its head inline, before the helper: dynamo makes a tensor argument of a call
    # (state_slot) a graph input at the call, so asking kda_shell_takes first would reorder (decode) or add (prefill)
    # the graphs' inputs.
    fh = ffn_half_takes(model, layer, streams, dec, sp)
    mk = False  # the fused KDA layer (kernels/mk_kda_layer.py): the shell call and the FFN half as two kernels
    if (dec and not sp and getattr(layer, "ks_blob_bf16", None) is not None
            and kda_shell_takes(model, layer, streams, state_slot, dec, sp)):
        from ..kernels import kda_shell

        T = streams.shape[0]
        if fh:  # asked only with the FFN half taking (its guard reads ints only: no tensor enters a helper first)
            from ..kernels import mk_kda_layer as MK

            mk = MK.layer_takes(model, layer, T) and layer.moe
        if mk:  # both halves, with XLA's sums between them, as ffn_half.decode
            S, post, comb, y = MK.decode_layer_split(model, layer, streams.view(T, hy.hc, H), positions, state_slot,
                                                     pend=pend)  # (post, comb, y) as is: the kernel builds y's slots
        else:
            S, (post, comb, Y) = kda_shell.decode(model, layer, streams.view(T, hy.hc, H), positions, state_slot,
                                                  pend=_shell_pend(layer, pend))
            if fh:  # this mix moves into the FFN half's call
                pend_a = (post, comb, Y)
            else:
                # the next block is not a shell: its mix in here, y the two programs' all-reduced partials summed
                streams = mix_out(post, comb, Y.float().sum(0).to(S.dtype), S, True)
    else:
        if pend is not None:  # the previous layer's deferred FFN mix
            streams = mix_out(*pend, streams.view(streams.shape[0], hy.hc, H), True)
        if (fh and dec and not sp and getattr(layer.spec, "mla", None) is not None
                and _mk4_rows_on(streams.shape[0])):
            # The fused DSA layer: _attn_unmixed's steps, the attention block's (S, post, comb) passed down, its
            # Sinkhorn left to the fused call (mhc_front); where that call takes, it ran the FFN half too (mk: no
            # ffn_half call below).
            S = streams.view(streams.shape[0], hy.hc, H)
            xm, post_a, comb_a = _mhc_csk(model, layer, "attn", S, dec)
            r4 = _mix(model, layer, rms_norm(xm, layer.in_norm, model.cfg.rms_norm_eps), positions, slot_mapping,
                      table, bias, state_slot, seq, ffn=(S, post_a, comb_a))
            if isinstance(r4, tuple):
                S, post, comb, y = r4
                mk = True
            else:
                pend_a = (post_a, _comb_done(model, layer, S, dec, comb_a), r4.unsqueeze(0))
        elif fh:  # the attention block without its mix, which moves into the FFN half's call
            S, pend_a = _attn_unmixed(model, layer, streams,
                                      rows(lambda x: _mix(model, layer, x, positions, slot_mapping, table, bias,
                                                          state_slot, seq)), dec)
        else:
            streams = _block(model, layer, "attn", streams, layer.in_norm,
                             rows(lambda x: _mix(model, layer, x, positions, slot_mapping, table, bias, state_slot,
                                                 seq)), dec)
    if fh:  # the attention output's mix, mhc_pre, rms_norm and the FFN half as one call
        from ..kernels import ffn_half

        if not mk:  # (the fused KDA or DSA layer has run the FFN half already)
            S, post, comb, y = ffn_half.decode(model, layer, S, pend_a)
        if defer:  # the FFN output's mix left to the next layer's KDA shell call
            return S.reshape(S.shape[0], W), (post, comb, y)
        return mix_out(post, comb, y, S, True)
    if defer:  # the FFN block's mix left to the next layer's KDA shell call
        if not dec:
            raise ValueError("hybrid.layer(defer=True) is a decode call's (the mix-in)")
        S = streams.view(streams.shape[0], hy.hc, H)
        x, post, comb = _mhc(model, layer, "ffn", S, dec)
        y = _ffn(model, layer, sp)(rms_norm(x, layer.post_norm, model.cfg.rms_norm_eps))
        return streams, (post, comb, y)
    return _block(model, layer, "ffn", streams, layer.post_norm, _ffn(model, layer, sp), dec)


def forward_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Logits of a whole sequence without a cache (tests): every layer's sequence form."""
    T = input_ids.shape[0]
    positions = torch.arange(T, device=input_ids.device)
    h = hidden_in(model, input_ids)
    seq: dict = {}
    for l in model.layers:
        h = layer(model, l, h, positions, None, None, None, None, seq)
    return model._head(final(model, h))


# The fused DSA layer's attention-block mHC (kernels/mhc_front.py, kernels/dsa_satwfc.py) leaves its Sinkhorn to the
# fused call, which runs it under its own collective waits.


def _mhc_csk(model, layer, b: str, S: torch.Tensor, decode: bool = False):
    """_mhc for the fused DSA layer: (x, post, mhc_front.PendingComb(c)) where mhc_pre's kernel would take, _mhc's own
    elsewhere."""
    from ..kernels import mhc_front, mhc_pre

    hy = model.cfg.hybrid
    fn = getattr(layer, f"hc_{b}_fn")
    if decode and mhc_pre.takes(S.device, S.shape[0], S.shape[-1], hy.hc, fn.dtype):
        return mhc_front.front(S, fn, getattr(layer, f"hc_{b}_base"), getattr(layer, f"hc_{b}_scale"),
                               hy.sinkhorn_iters, hy.hc_eps, model.cfg.rms_norm_eps)
    return _mhc(model, layer, b, S, decode)


def _comb_done(model, layer, S: torch.Tensor, decode: bool, comb):
    """The attention block's comb where the fused DSA layer's call did not take: a PendingComb is replaced by
    mhc_pre's comb, recomputed (the front's x and post are mhc_pre's to the bit); other combs pass."""
    from ..kernels import mhc_front

    if isinstance(comb, mhc_front.PendingComb):
        return _mhc(model, layer, "attn", S, decode)[2]
    return comb
