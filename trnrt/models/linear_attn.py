"""GLM-5.3-Flash's linear-attention token mixer: Kimi Delta Attention (KDA).

A linear-attention layer replaces softmax attention over a KV cache with a fixed-size state
per sequence: the last K - 1 inputs of a short causal depthwise conv, and one [Dk, Dv]
matrix per head updated by the gated delta rule

    S_t = exp(g_t) * S_{t-1} + k_t (beta_t (v_t - (exp(g_t) * S_{t-1})^T k_t))^T,   o_t = S_t^T q_t

with q, k L2-normalised and q scaled by Dk^-0.5. KDA decays each key channel separately (g_t is
a vector per head). The layer's output is RMSNorm(o) * sigmoid(gate) followed by the output
projection.

Numerics follow the Hugging Face transformers ports, which are the CPU references:
transformers models/kimi_linear/modeling_kimi_linear.py (v5.18.0, KimiLinearDeltaAttention,
KimiLinearForgetGate, chunk_kimi_delta_attention) and models/glm5_next/modeling_glm5_next.py
(v5.18.0, Glm5NextTextLinearAttention, whose forget gate adds the "safe gate" lower bound
lower_bound * sigmoid(exp(A_log) * g)).

Static-shape formulation (the same code runs on CPU and on a NeuronCore):
- decode (B sequences x 1 token): the recurrent step above as batched einsums. On the device a
  decode step's KDA layers run as the KDA shell instead (models/hybrid.py, kernels/kda_shell.py), and a step above
  the shell's row cap runs the recurrence here as kernels/kda_decode.py.
- prefill chunk (one sequence x C tokens): the chunked form (fla / the transformers chunk
  kernels) over fixed sub-chunks of CHUNK tokens, unrolled at trace time; on the device the NKI
  chunked delta rule (kernels/delta_rule.py). The per-chunk triangular solve (I + A)^-1, which
  the references do by forward substitution (CHUNK - 1 sequential row updates), is computed with
  plain matmuls (_unit_lower_inverse). Padded tokens get beta = 0 and g = 0, which leaves the
  state untouched.
- Every per-sequence state lives in a device pool (engine/state_pool.py) indexed by a state
  slot per sequence; a call reads rows with a gather and writes them back with index_put_.
  A sequence whose first position is 0 starts from zero state (torch.where, so a stale row,
  even one holding NaN, is never read), which also resets a reused slot.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import LinearSpec

NEG = -1e30
# Tokens per sub-chunk of the CPU's chunked delta rule (the references use 64). KDA's per-channel decay materialises
# [heads, L, L, Dk] per sub-chunk. One layer, prefill C=128 on trn1.2xlarge at 32 heads: 3.38 / 3.29 / 4.10 ms for
# L = 16 / 32 / 64.
CHUNK = 32


# -- parameters ----------------------------------------------------------------------------


def init_mixer(layer: nn.Module, cfg, spec: LinearSpec, tp: int, p, lin) -> None:
    """Parameters of one KDA layer on ONE attention rank (tp is the attention TP, DecoderForCausalLM), set on the
    DecoderLayer: heads are split across ranks, the output projection on its input dim (an all-reduce follows). p / lin
    are DecoderLayer's parameter makers (lin: a [out, in] weight, FP8 + scale when the checkpoint quantizes it)."""
    if spec.num_k_heads % tp or spec.num_v_heads % tp:
        raise ValueError(f"tp={tp} does not divide {spec.num_k_heads} k / {spec.num_v_heads} v heads")
    if (spec.kind != "kda" or spec.num_k_heads != spec.num_v_heads or not spec.gate_rank
            or spec.gate_act != "sigmoid"):
        raise NotImplementedError(f"linear attention {spec}: GLM-5.3-Flash's KDA has equal k / v heads and a "
                                  "low-rank sigmoid output gate")
    H, dk, dv = cfg.hidden_size, spec.head_k_dim, spec.head_v_dim
    nk, nv = spec.num_k_heads // tp, spec.num_v_heads // tp
    layer.nk, layer.nv = nk, nv
    layer.conv_dim = 2 * nk * dk + nv * dv  # conv channels: q, then k, then v

    def f32(*shape):
        return nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False)

    m = "self_attn."
    lin("in_qkv", m + "q_proj", layer.conv_dim, H)
    layer.conv_w = p(layer.conv_dim, spec.conv_kernel)
    layer.A_log = f32(nv)
    layer.f_a = p(dk, H)  # forget gate, low rank (head_dim), replicated
    layer.f_b = p(nv * dk, dk)
    layer.dt_bias = f32(nv * dk)
    lin("in_b", m + "b_proj", nv, H)
    layer.g_a, layer.g_b = p(spec.gate_rank, H), p(nv * dv, spec.gate_rank)
    layer.o_norm = f32(dv)  # KimiLinearRMSNormGated keeps the norm and its weight in fp32
    lin("out", m + "o_proj", H, nv * dv)
    layer.conv_state = None  # [slots, K - 1, conv_dim], bound by the runner (engine/state_pool.py)
    layer.rec_state = None  # [slots, nv, dk, dv] fp32


def state_shapes(layer) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Per-sequence (conv state, recurrent state) shapes of one layer on this rank."""
    sp = layer.spec
    return (sp.conv_kernel - 1, layer.conv_dim), (layer.nv, sp.head_k_dim, sp.head_v_dim)


def load_mixer(layer, ck, pre: str, cfg, r: int, n: int, dtype: torch.dtype, blk, fp8_max: float) -> None:
    """Fill one KDA layer from checkpoint names under `pre` (model.layers.{i}.), as attention rank r of n
    (DecoderForCausalLM.attn_rank / attn_tp): self_attn.{q,k,v}_proj, {q,k,v}_conv1d, f_a_proj, f_b_proj, dt_bias,
    A_log [H], b_proj, g_a_proj / g_b_proj, o_norm, o_proj (zai-org/GLM-5.3-Flash model.safetensors.index.json).
    transformers' own layout (conv1d stacked, forget_gate.* and A_log [1, 1, H, 1], conversion_mapping.py
    "kimi_linear") is accepted as well."""
    from .loader import _assign, _concat, _part

    sp = layer.spec
    dk = sp.head_k_dim
    kd_all = sp.num_k_heads * dk
    qs = _part(kd_all, r, n)
    heads = _part(sp.num_v_heads, r, n)

    def name(*cands):
        for c in cands:
            if c in ck:
                return c
        raise KeyError(cands[0])

    def put(param, nm, rows=None):
        param.data.copy_(ck.get(nm, rows).to(param.dtype))

    def dense(param, nm, rows=None):  # a weight kept in the model dtype, dequantized if stored FP8
        from .quant import dequant

        w, sc = ck.linear(nm[: -len(".weight")], rows, block=blk)
        param.data.copy_(dequant(w, sc, torch.float32).to(param.dtype) if sc is not None else w.to(param.dtype))

    def proj(attr, base, rows=None, cols=None):
        w, sc = ck.linear(base, rows, cols, block=blk)
        _assign(getattr(layer, attr), getattr(layer, attr + "_scale"), w, sc, dtype, blk, fp8_max)

    shift = lambda sl, by: slice(sl.start + by, sl.stop + by)  # noqa: E731
    a = pre + "self_attn."
    w, sc = _concat([ck.linear(a + f"{x}_proj", qs, block=blk) for x in "qkv"])
    _assign(layer.in_qkv, layer.in_qkv_scale, w, sc, dtype, blk, fp8_max)
    if a + "conv1d.weight" in ck:  # transformers' stacked [q | k | v] conv
        conv = ck.get(a + "conv1d.weight")
        parts = [conv[shift(qs, i * kd_all)] for i in range(3)]
    else:
        parts = [ck.get(a + f"{x}_conv1d.weight", qs) for x in "qkv"]
    layer.conv_w.data.copy_(torch.cat(parts).reshape(layer.conv_dim, -1).to(dtype))
    fg = lambda x: name(a + x, a + "forget_gate." + x)  # noqa: E731
    dense(layer.f_a, fg("f_a_proj.weight"))
    dense(layer.f_b, fg("f_b_proj.weight"), qs)
    put(layer.dt_bias, fg("dt_bias"), qs)
    layer.A_log.data.copy_(ck.get(fg("A_log")).float().reshape(-1)[heads])
    proj("in_b", a + "b_proj", heads)
    dense(layer.g_a, a + "g_a_proj.weight")
    dense(layer.g_b, a + "g_b_proj.weight", qs)
    put(layer.o_norm, a + "o_norm.weight")
    proj("out", a + "o_proj", cols=qs)


# -- the mixer ---------------------------------------------------------------------------------


def _softplus(x: torch.Tensor) -> torch.Tensor:
    """log(1 + exp(x)) without a comparison against a float literal (neuronx-cc can lower those through f64); equal to
    F.softplus within fp32 rounding."""
    return F.relu(x) + torch.log1p(torch.exp(-x.abs()))


def _causal_conv(xe: torch.Tensor, w: torch.Tensor, T: int) -> torch.Tensor:
    """Depthwise causal conv: xe [..., K - 1 + T, D] holds the K - 1 inputs before the T new
    ones, w [D, K]; out[t] = sum_j w[:, j] * xe[t + j], as F.conv1d with K - 1 left padding
    (the references' causal_conv1d_fn / causal_conv1d_update). Accumulated in fp32. Chunk and sequence forms on the
    device run kernels/short_conv.py (the same products in the same order)."""
    from ..kernels import short_conv

    if short_conv.takes(xe, w, T):
        return short_conv.conv(xe, w)
    K = w.shape[1]
    wf = w.float()
    y = xe[..., 0:T, :].float() * wf[:, 0]
    for j in range(1, K):
        y = y + xe[..., j : j + T, :].float() * wf[:, j]
    return y


INVERSE_BASE = 8  # diagonal blocks of _unit_lower_inverse inverted by repeated squaring


def _unit_lower_inverse(N: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
    """(I - N)^-1 for strictly lower-triangular N [..., L, L], with plain matmuls only.

    Blocks of up to INVERSE_BASE rows are inverted as I + N + ... + N^(b-1) = (I + N)(I + N^2)
    (I + N^4)...; larger ones by merging halves, [[A, 0], [C, D]]^-1 = [[A^-1, 0], [-D^-1 C A^-1,
    D^-1]] (fla's solve_tril: 16 x 16 diagonal blocks merged into 32 and 64, vLLM v0.30.0
    third_party/flash_linear_attention/ops/solve_tril.py). Repeated squaring over a whole sub-chunk
    is exact algebra but not exact arithmetic: the powers of N grow like binomial coefficients
    before they cancel, and with real weights (63 tokens, fp32) one sub-chunk of 64 came out wrong
    by 9.5e7 (32: 0.21, 16: 1.8e-6) against the token-by-token recurrence; the blocked form is
    exact to rounding at every length.

    When L is b times a power of two every level works on full-size matrices: X holds the inverses
    of the diagonal blocks of size s (zero elsewhere), first for s = b by squaring N restricted to
    those blocks, then each level is X + X (N on the lower-left halves of the 2s blocks) X, which
    places D^-1 N21 A^-1 under every pair: two matmuls per level, as many as a squaring step. The
    block masks are built on the host at trace time: as in-graph arange / expand / compare ops one
    prefill layer (trn1, C = 128, sub-chunks of 64) took 3.23 ms, as constants 0.51 ms, the time of
    the unstable squaring."""
    L = N.shape[-1]
    b = INVERSE_BASE
    if L <= b:
        t, pw, span = eye + N, N, 2
        while span < L:
            pw = pw @ pw
            t = t + t @ pw
            span *= 2
        return t
    if L % b == 0 and (L // b) & (L // b - 1) == 0:
        r = np.arange(L)
        block = torch.tensor((r[:, None] // b == r[None, :] // b).astype(np.float32), device=N.device)
        X, pw, span = eye + N * block, N * block, 2
        while span < b:
            pw = pw @ pw
            X = X + X @ pw
            span *= 2
        s = b
        while s < L:
            low = (r[:, None] // (2 * s) == r[None, :] // (2 * s)) & (r[:, None] // s % 2 == 1) & (r[None, :] // s % 2 == 0)
            X = X + X @ (N * torch.tensor(low.astype(np.float32), device=N.device)) @ X
            s *= 2
        return X
    h = 1 << ((L - 1).bit_length() - 1)  # any other L: by halves, the largest power of two first
    a = _unit_lower_inverse(N[..., :h, :h], eye[:h, :h])
    d = _unit_lower_inverse(N[..., h:, h:], eye[h:, h:])
    c = d @ N[..., h:, :h] @ a
    top = torch.cat([a, torch.zeros_like(N[..., :h, h:])], dim=-1)
    return torch.cat([top, torch.cat([c, d], dim=-1)], dim=-2)


def chunk_scan(q, k, v, g, beta, S, chunk: int = 64):
    """The gated delta rule over one sequence in sub-chunks of `chunk` tokens (fla's chunked
    form; transformers chunk_kimi_delta_attention).

    q, k [T, H, Dk] (q already scaled), v [T, H, Dv], g [T, H, Dk] log decays, beta [T, H],
    S [H, Dk, Dv]; fp32. T is padded up to a multiple of `chunk` with beta = g = 0 tokens, which
    do not change S. Returns o [T, H, Dv] and the final S."""
    T, H, Dk = k.shape
    L = min(chunk, T)
    pad = -T % L
    if pad:
        q, k, v, g, beta = (torch.cat([x, x.new_zeros((pad, *x.shape[1:]))]) for x in (q, k, v, g, beta))
    i = torch.arange(L, device=q.device)
    lower = i.view(-1, 1) >= i.view(1, -1)  # j <= i, the diagonal included
    strict = i.view(-1, 1) > i.view(1, -1)
    tri = lower.to(torch.float32)
    eye = (i.view(-1, 1) == i.view(1, -1)).to(torch.float32)
    outs = []
    for c in range((T + pad) // L):
        sl = slice(c * L, (c + 1) * L)
        qc, kc, vc = (x[sl].transpose(0, 1) for x in (q, k, v))  # [H, L, D]
        bc = beta[sl].transpose(0, 1).unsqueeze(-1)  # [H, L, 1]
        kb, vb = kc * bc, vc * bc
        G = torch.einsum("ij,jhd->hid", tri, g[sl])  # cumulative log decay [H, L, Dk]
        D = G.unsqueeze(2) - G.unsqueeze(1)  # [H, L(i), L(j), Dk]: G_i - G_j
        decay = torch.exp(torch.where(lower.unsqueeze(-1), D, NEG))
        A = (kb.unsqueeze(2) * kc.unsqueeze(1) * decay).sum(-1)
        qk = (qc.unsqueeze(2) * kc.unsqueeze(1) * decay).sum(-1)
        eG, gl = torch.exp(G), G[:, -1]  # gl [H, Dk]
        tail = torch.exp(gl.unsqueeze(1) - G)  # [H, L, Dk]
        s_decay = torch.exp(gl).unsqueeze(-1)  # [H, Dk, 1]
        Tm = _unit_lower_inverse(-torch.where(strict, A, torch.zeros_like(A)), eye)
        u = Tm @ vb  # [H, L, Dv]
        w = Tm @ (kb * eG)  # [H, L, Dk]
        vn = u - w @ S
        outs.append(((qc * eG) @ S + qk @ vn).transpose(0, 1))
        S = S * s_decay + (kc * tail).transpose(-1, -2) @ vn
    return torch.cat(outs)[:T], S


def recurrent_step(q, k, v, g, beta, S):
    """One token per row: q, k [B, H, Dk], v [B, H, Dv], g [B, H, Dk], beta [B, H], S [B, H, Dk, Dv]; fp32
    (transformers recurrent_kimi_delta_attention).

    The references decay S, read k^T S, update S, then read q^T S. With E = diag(exp(g)) the
    same values come from ONE read of the old state, [q E; k E] S, and one update:
      delta = beta (v - (k E)^T S),   o = (q E)^T S + (q . k) delta,   S' = E S + k delta^T,
    which on trn1 avoids the tensor-engine transposes of three passes over S (the einsum form
    measured 1.05 ms per layer at B=8, 2.0 ms of it transposes)."""
    e = torch.exp(g)
    r = torch.stack([q * e, k * e], dim=2) @ S  # [B, H, 2, Dv]
    delta = (v - r[:, :, 1]) * beta.unsqueeze(-1)
    o = r[:, :, 0] + (q * k).sum(-1, keepdim=True) * delta
    S = S * e.unsqueeze(-1) + k.unsqueeze(-1) * delta.unsqueeze(-2)
    return o, S


def kernel_takes(spec: LinearSpec) -> bool:
    """Whether the NKI delta-rule kernel (kernels/delta_rule.py) runs this layer's prefill on the device: head dims of
    128 and a gate lower bound (unbounded decays fall outside the kernel's sub-chunk range). The CPU always runs
    chunk_scan."""
    return spec.head_k_dim == 128 and spec.head_v_dim == 128 and spec.lower_bound is not None


def _nki_chunk(q, k, v, g, beta, S, keep_pad: bool = False, ug: int | None = None):
    """chunk_scan's contract through the NKI kernel; keep_pad: o with the kernel's padding rows, ug: its unit group
    (delta_rule.chunk)."""
    from ..kernels import delta_rule

    return delta_rule.chunk(q, k, v, g, beta, S, keep_pad=keep_pad, ug=ug)


def verify_scan(q, k, v, g, beta, S):
    """B sequences x Q tokens each, from their own state (speculative verify): q, k [B, Q, H, Dk] (q scaled),
    v [B, Q, H, Dv], g [B, Q, H, Dk], beta [B, Q, H], S [B, H, Dk, Dv]; fp32. recurrent_step once per position, the
    decode step's own arithmetic. Returns o [B, Q, H, Dv] and the state after EVERY position, [B, Q, H, Dk, Dv]: a
    rejected draft is rolled back by continuing from the state after the last accepted position (vLLM v0.30.0
    v1/attention/backends/gdn_attn.py spec_state_indices_tensor: num_spec + 1 state slots per request, the next step
    reading the one at num_accepted_tokens - 1; SGLang v0.5.21 mem_cache/memory_pool.py intermediate_ssm_state_cache,
    one state per draft token)."""
    outs, states = [], []
    for t in range(k.shape[1]):
        o, S = recurrent_step(q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], S)
        outs.append(o)
        states.append(S)
    return torch.stack(outs, 1), torch.stack(states, 1)


def _write_rows(pool: torch.Tensor, rows: torch.Tensor, values: torch.Tensor) -> None:
    """pool[rows] = values. A ONE-row index_put_ is written with the scratch row added (it may
    hold anything): on trn1 (neuronx-cc 2.27, SDK 2.32) a one-row index_put_ of a value computed
    in the graph measured 8.7 ms for a 1 MB state row against 0.14 ms for the same write as
    two rows."""
    if rows.shape[0] == 1:
        rows = torch.cat([rows, torch.zeros_like(rows)])
        values = torch.cat([values, values])
    pool.index_put_((rows,), values)


# A decode call of B >= 2 rows reads and writes its B sequences' conv windows in conv_state [slots, K - 1, D] as
# one-hot matmuls: read as onehot(state_slot) @ view, written as where(written, onehot^T @ windows, view). As
# conv_state[state_slot] and index_put_, neuronx-cc lowers each as indirect DMAs in 256-B elements of the 4608-B
# window (on trn2 at tp 32, 306 + 306 DMA_INDIRECT per core per step, ~1.3 ms of DMA instruction time); the one-hot
# form has no indirect DMA. Exact: each output element is one product with 1.0 plus products with 0.0. Rows sharing a
# slot (padding on the scratch row) sum there, where index_put_ keeps one of them; the scratch row holds anything. At
# B = 1 the index form is one direct copy already.


def _conv_onehot(state_slot: torch.Tensor, slots: int, dtype: torch.dtype) -> torch.Tensor:
    """[B, slots]: 1 at each row's slot."""
    return (state_slot.view(-1, 1) == torch.arange(slots, device=state_slot.device).view(1, slots)).to(dtype)


def _conv_read(conv_state: torch.Tensor, state_slot: torch.Tensor) -> torch.Tensor:
    """conv_state[state_slot] [B, K - 1, D], as a one-hot matmul."""
    S, B = conv_state.shape[0], state_slot.shape[0]
    view = conv_state.view(S, -1)
    return (_conv_onehot(state_slot, S, view.dtype) @ view).view(B, *conv_state.shape[1:])


def _conv_write(conv_state: torch.Tensor, state_slot: torch.Tensor, win: torch.Tensor) -> None:
    """conv_state[state_slot] = win [B, K - 1, D], as a one-hot matmul and a select."""
    S, B = conv_state.shape[0], state_slot.shape[0]
    view = conv_state.view(S, -1)
    oh = _conv_onehot(state_slot, S, view.dtype)
    view.copy_(torch.where((oh.sum(0) > 0).view(S, 1), oh.t() @ win.reshape(B, -1), view))


def _l2norm_kda(x):  # Kimi's l2norm, in fp32: x / sqrt(sum + eps)
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def mix(model, layer, x: torch.Tensor, positions: torch.Tensor, slot_mapping: torch.Tensor,
        state_slot: torch.Tensor | None) -> torch.Tensor:
    """The block on an already normalised input x (no residual): the output projection after its all-reduce over the
    attention group (DecoderForCausalLM: attention TP); models/hybrid.py mixes it into the residual streams. Batch
    forms, told apart by state_slot:
      decode   x [B, H], state_slot [B]    (B sequences, one token each)
      chunk    x [C, H], state_slot [1]    (one sequence; padded rows have slot_mapping in
                                            the null page and are left out of the state)
      chunk    x [C, H], state_slot [3]    (with an in-chunk checkpoint: see _mix_rows)
      sequence x [T, H], state_slot None   (forward_logits: a whole sequence from zero state)
    """
    # Not stored in a local first: dynamo renames a graph node after the first local a value is stored in
    # (torch/_dynamo STORE_FAST -> TensorVariable.set_name_hint), and the node names are part of the compile cache key.
    return model._attn_all_reduce(_mix_rows(model, layer, x, positions, slot_mapping, state_slot))


def _mix_rows(model, layer, x: torch.Tensor, positions: torch.Tensor, slot_mapping: torch.Tensor,
              state_slot: torch.Tensor | None) -> torch.Tensor:
    """mix on one batch form's rows: the output projection before its all-reduce."""
    cfg, sp = model.cfg, layer.spec
    dt = model.dtype
    T = x.shape[0]
    K = sp.conv_kernel
    qkv = F.linear(x, model._w(layer, "in_qkv"))  # [T, conv_dim]
    mid_row = mid_k = None
    if state_slot is None:
        form = "sequence"
    elif state_slot.dim() == 2:  # a speculative verify (decoder.forward_extend): B sequences x Q positions
        form = "verify"
        B, Q = state_slot.shape[0], state_slot.shape[1] - 1
        read, write = state_slot[:, 0], state_slot[:, 1:].reshape(B * Q)
    elif state_slot.dim() != 1:
        raise ValueError(f"state_slot {tuple(state_slot.shape)} matches no batch form of {T} tokens")
    elif state_slot.shape[0] == T:
        form = "decode"
    elif state_slot.shape[0] == 1:
        form = "chunk"
    elif state_slot.shape[0] == 3 and T > 3:
        # A chunk with an in-chunk checkpoint: [the sequence's row, the checkpoint's row, k]: the state and conv window
        # after the chunk's first k rows go to the checkpoint's row, so a returning turn resumes there without the
        # prefill being split at it (k = T: the scratch row gets the chunk's end, unused).
        form = "chunk"
        mid_row, mid_k = state_slot[1:2], state_slot[2]
        state_slot = state_slot[:1]
    else:
        raise ValueError(f"state_slot {tuple(state_slot.shape)} matches no batch form of {T} tokens")

    # Short conv over q, k, v, with the K - 1 previous inputs from the state.
    if form == "verify":  # each sequence's window read once, then one window written per position
        keep = (positions.view(B, Q)[:, 0] > 0).view(B, 1, 1)
        prev = layer.conv_state[read]
        prev = torch.where(keep, prev, torch.zeros_like(prev)).to(qkv.dtype)
        xe = torch.cat([prev, qkv.view(B, Q, -1)], dim=1)  # [B, K - 1 + Q, D]
        y = _causal_conv(xe, layer.conv_w, Q).reshape(T, -1)
        win = torch.stack([xe[:, t + 1 : t + K] for t in range(Q)], dim=1)  # the last K - 1 inputs after each position
        _write_rows(layer.conv_state, write, win.reshape(B * Q, K - 1, -1).to(layer.conv_state.dtype))
    elif form == "decode":
        keep = (positions > 0).view(T, 1, 1)
        if T == 1:  # one row is a direct copy already
            prev = layer.conv_state[state_slot]
        else:
            prev = _conv_read(layer.conv_state, state_slot)
        prev = torch.where(keep, prev, torch.zeros_like(prev)).to(qkv.dtype)
        xe = torch.cat([prev, qkv.unsqueeze(1)], dim=1)  # [B, K, D]
        y = _causal_conv(xe, layer.conv_w, 1)[:, 0]
        if T == 1:
            _write_rows(layer.conv_state, state_slot, xe[:, 1:].to(layer.conv_state.dtype))
        else:
            _conv_write(layer.conv_state, state_slot, xe[:, 1:].to(layer.conv_state.dtype))
    else:
        if form == "chunk":
            keep = (positions[:1] > 0).view(1, 1)
            prev = layer.conv_state[state_slot][0]
            prev = torch.where(keep, prev, torch.zeros_like(prev)).to(qkv.dtype)
            valid = slot_mapping >= model.page_size  # real tokens; padding writes the null page
        else:
            prev = qkv.new_zeros(K - 1, qkv.shape[1])
        xe = torch.cat([prev, qkv])  # [K - 1 + T, D]
        y = _causal_conv(xe, layer.conv_w, T)
        if form == "chunk":
            # The last K - 1 inputs of the sequence: rows n .. n + K - 2 of xe, n real tokens.
            last = valid.sum() + torch.arange(K - 1, device=xe.device)
            if mid_k is None:
                _write_rows(layer.conv_state, state_slot, xe.index_select(0, last).unsqueeze(0).to(layer.conv_state.dtype))
            else:  # and the K - 1 inputs before row k: rows k .. k + K - 2 of xe
                win = xe.index_select(0, last).unsqueeze(0).to(layer.conv_state.dtype)
                at = mid_k + torch.arange(K - 1, device=xe.device)
                _write_rows(layer.conv_state, torch.cat([state_slot, mid_row]),
                            torch.cat([win, xe.index_select(0, at).unsqueeze(0).to(layer.conv_state.dtype)]))
    y = F.silu(y.to(dt))
    nk, nv, dk, dv = layer.nk, layer.nv, sp.head_k_dim, sp.head_v_dim
    kd = nk * dk
    q = y[:, :kd].reshape(T, nk, dk)  # explicit slices, not torch.split
    k = y[:, kd : 2 * kd].reshape(T, nk, dk)
    v = y[:, 2 * kd :].reshape(T, nv, dv)

    f = F.linear(F.linear(x, layer.f_a), layer.f_b).float() + layer.dt_bias
    f = f.reshape(T, nv, dk)
    rate = torch.exp(layer.A_log).view(1, nv, 1)
    g = sp.lower_bound * torch.sigmoid(rate * f) if sp.lower_bound is not None else -rate * _softplus(f)
    beta = torch.sigmoid(F.linear(x, model._w(layer, "in_b"))).float()
    q, k = _l2norm_kda(q.float()), _l2norm_kda(k.float())
    z = F.linear(F.linear(x, layer.g_a), layer.g_b).reshape(T, nv, dv)
    q = q * dk ** -0.5
    v = v.float()
    nki_prefill = form in ("chunk", "sequence") and x.device.type != "cpu" and kernel_takes(sp)
    # The gated norm below as one kernel on the delta rule's output at LNC = 2 (kernels/gated_norm.py): ints only.
    from ..kernels import gated_norm

    norm2 = nki_prefill and dt == torch.bfloat16 and gated_norm.takes2(dv, T)
    if nki_prefill:  # the delta rule's unit group of 6
        scan = lambda *a: _nki_chunk(*a, keep_pad=False, ug=6)  # noqa: E731
    else:
        scan = lambda *a: chunk_scan(*a, CHUNK)  # noqa: E731

    if form == "verify":  # from each sequence's state; the state after every position written to its own row
        S = layer.rec_state[read]
        S = torch.where(keep.view(B, 1, 1, 1), S, torch.zeros_like(S))
        o, states = verify_scan(q.view(B, Q, nv, dk), k.view(B, Q, nv, dk), v.view(B, Q, nv, dv),
                                g.view(B, Q, *g.shape[1:]), beta.view(B, Q, nv), S)
        _write_rows(layer.rec_state, write, states.reshape(B * Q, nv, dk, dv))
        o = o.reshape(T, nv, dv)
    elif form == "decode":
        # Above the KDA shell's row cap (hybrid.kda_shell_rows), on the device: the recurrence as one kernel, which
        # reads, updates and writes the pool rows in place (kernels/kda_decode.py); the einsums below elsewhere.
        from ..kernels import kda_decode

        if kda_decode.takes(x, dk, dv, sp.kind == "kda" and g.dim() == 3):
            o = kda_decode.decode_step(layer.rec_state, state_slot, keep.view(T), q, k, v, g, beta)
        else:
            keep4 = keep.view(T, 1, 1, 1)
            S = layer.rec_state[state_slot]
            S = torch.where(keep4, S, torch.zeros_like(S))
            o, S = recurrent_step(q, k, v, g, beta, S)
            _write_rows(layer.rec_state, state_slot, S)
    elif form == "chunk":
        S = layer.rec_state[state_slot][0]
        S = torch.where(keep.view(1, 1, 1), S, torch.zeros_like(S))
        vf = valid.view(T, 1)
        beta = torch.where(vf, beta, torch.zeros_like(beta))
        g = torch.where(vf.view(T, 1, 1), g, torch.zeros_like(g))
        S0 = S
        o, S = scan(q, k, v, g, beta, S)
        if mid_k is not None:
            # The state after the first k rows: the same scan with the rows from k on masked as padding is (beta and
            # the gate's log-decay zero leave the state untouched), so the checkpoint costs one more pass, not a call.
            mf = vf & (torch.arange(T, device=vf.device).view(T, 1) < mid_k)
            beta_m = torch.where(mf, beta, torch.zeros_like(beta))
            g_m = torch.where(mf.view(T, 1, 1), g, torch.zeros_like(g))
            _, S_m = scan(q, k, v, g_m, beta_m, S0)
            _write_rows(layer.rec_state, torch.cat([state_slot, mid_row]), torch.stack([S, S_m]))
        else:
            _write_rows(layer.rec_state, state_slot, S.unsqueeze(0))
    else:
        o, _ = scan(q, k, v, g, beta, q.new_zeros(nv, dk, dv))

    # Gated RMSNorm per head (KimiLinearRMSNormGated: fp32 throughout), then the output projection.
    if norm2:  # the 128-row tiles split between the LNC programs
        return F.linear(gated_norm.apply2(o, z, layer.o_norm, cfg.rms_norm_eps), model._w(layer, "out"))
    of = o.to(dt).float()
    of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    of = layer.o_norm * of
    o = (of * torch.sigmoid(z.float())).to(dt)
    return F.linear(o.reshape(T, nv * dv), model._w(layer, "out"))
