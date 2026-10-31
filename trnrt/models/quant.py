"""Quantized weights: FP8 storage with fp32 block scales along the input dim.

The one in-memory format is `w` float8_e4m3fn [N, K] plus `scale` fp32 [N, K / bk], and a weight is
`w * scale` with each scale covering bk consecutive input columns of one row. The checkpoint's FP8
with 128 x 128 block scales (`weight_scale_inv` [N/128, K/128], dequantised as w_fp8 * scale_inv)
converts to it EXACTLY: repeating each block's scale over its 128 rows gives per-row scales, which
also lets tensor parallelism slice rows anywhere.
"""

from __future__ import annotations

import torch

FP8 = torch.float8_e4m3fn


def dequant(w: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
    """w [..., N, K] (fp8 or already dtype), scale [..., N, K / bk] or None -> dtype [..., N, K]."""
    if scale is None:
        return w
    K = w.shape[-1]
    bk = -(-K // scale.shape[-1])
    s = scale.repeat_interleave(bk, dim=-1)[..., :K]
    return (w.to(torch.float32) * s).to(dtype)


def dequant_t(w: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
    """dequant for a weight stored transposed: w [..., K, N], scale [..., K / bk, N]."""
    if scale is None:
        return w
    K, nb = w.shape[-2], scale.shape[-2]
    if nb > 1:
        scale = scale.repeat_interleave(-(-K // nb), dim=-2)[..., :K, :]
    return (w.to(torch.float32) * scale.to(torch.float32)).to(dtype)


def quantize_fp8_rows(w: torch.Tensor, bk: int = 128):
    """The quantiser for weights a checkpoint leaves unquantized (and for tests): per (row, bk columns)
    absmax scale onto e4m3 max 240, the largest finite value on trn2."""
    N, K = w.shape
    pad = (-K) % bk
    wf = torch.nn.functional.pad(w.float(), (0, pad)).view(N, -1, bk)
    scale = wf.abs().amax(dim=-1).clamp(min=1e-12) / 240.0
    q = (wf / scale.unsqueeze(-1)).clamp(-240, 240).view(N, -1)[:, :K]
    return q.to(FP8), scale


def fit_e4m3_max(w: torch.Tensor, scale: torch.Tensor, max_finite: float = 240.0, row_group: int | None = None):
    """Checkpoint FP8 is e4m3fn (max 448); trn2 FP8 is e4m3 with inf (max 240), which
    neuronx-cc only accepts e4m3fn graphs as under --experimental-unsafe-fp8e4m3fn-as-fp8e4m3
    ("unsafe" because the larger fn values do not fit; AWS's vllm-neuron clamps to 240 before
    it for that reason). Halve the values of any (row, block) whose max exceeds max_finite and
    double its scale: exact for every normal e4m3 value (448 / 2 = 224).

    row_group g: decide the halving per (g consecutive rows, block) instead, where the scales are
    constant over each such group (a 128 x 128 block-scaled checkpoint's shard; ignored otherwise),
    so the scales stay constant over the group: the rows of a group that did not exceed max_finite
    are halved too. Halving is exact except for codes below 2^-5, which it makes subnormal: those
    with an odd last bit round to even, an error of 2^-10 times the doubled scale (4.4e-6 of the
    block's range); the per-row fit rounds the same codes of the rows it halves."""
    N, K = w.shape[-2], w.shape[-1]
    nb = scale.shape[-1]
    bk = -(-K // nb)
    pad = nb * bk - K
    wf = torch.nn.functional.pad(w.float(), (0, pad)).view(*w.shape[:-1], nb, bk)
    over = wf.abs().amax(dim=-1) > max_finite  # [..., N, nb]
    if not bool(over.any()):
        return w, scale
    g = row_group
    if g and g > 1 and N % g == 0 and scale.shape[-2] == N:
        sg = scale.reshape(*scale.shape[:-2], N // g, g, nb)
        if bool(torch.equal(sg, sg[..., :1, :].expand_as(sg))):  # scales constant over each row group
            og = over.reshape(*over.shape[:-2], N // g, g, nb).any(dim=-2, keepdim=True)
            over = og.expand(*over.shape[:-2], N // g, g, nb).reshape(over.shape)
    wf = torch.where(over.unsqueeze(-1), wf / 2, wf)
    return wf.reshape(*w.shape[:-1], nb * bk)[..., :K].to(FP8), torch.where(over, scale * 2, scale)
