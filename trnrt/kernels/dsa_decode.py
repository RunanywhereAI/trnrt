"""The attention of a pooled DSA layer's decode rows (GLM-5.3-Flash, models/glm5_next.py) over the SELECTED pools only:
the slot layout the DSA attention kernels share (NCH chunks of 128 pool slots, KP tokens a pool) and its arithmetic in
torch (emulate), the host form of kernels/dsa_slots.py's and kernels/dsa_slots_c.py's kernels.

The selection keeps index_topk / index_kpool = 512 pools of 4 tokens (2048 tokens) plus the query's own incomplete
pool, so the attention reads ~1/4 of the bucket's bytes and never materialises a bf16 copy of the latent cache.

What it computes, per sequence row b and head h (absorbed MLA, NoPE: the latent is both key and value):
s[t] = scale * q_lat[b, h] . K[t] + bias[b, t] over the gathered tokens t, p = softmax(s), o[b, h] = sum_t p[t] K[t].
The gathered tokens are NS = NCH * 128 pool slots of KP consecutive cache rows each (a pool never crosses a page:
page_size is a multiple of KP), slot j of row b starting at cache row rows[b, j]; bias is 0 for a token to attend
and NEG_INF for padding (unselected slots, the tail pool's invisible tokens). With NEG_INF on every non-selected key
the mask form's softmax is this one (exp underflows to 0), so the two differ only in rounding: the scores stay fp32,
p is bf16 and the products accumulate in fp32.
"""

from __future__ import annotations

import torch

NEG_INF = -1e30  # models/decoder.NEG_INF
NCH = 5  # 128-slot chunks per row: 512 selected pools + the tail pool, padded to 640 slots
KP = 4  # tokens per pool (GLM-5.3-Flash index_kpool)


def emulate(q_lat: torch.Tensor, kc: torch.Tensor, rows: torch.Tensor, bias: torch.Tensor, scale: float) -> torch.Tensor:
    """The kernel's arithmetic in torch (CPU): q_lat [B, H, R], kc [N, R] (the latent cache's token rows), rows [B, NS]
    int (each slot's pool row: cache rows KP rows .. KP rows + KP - 1), bias [B, NS, KP] fp32 -> o [B, H, R] fp32."""
    B, H, R = q_lat.shape
    NS = rows.shape[1]
    tok = (rows.long().unsqueeze(-1) * KP + torch.arange(KP, device=rows.device)).reshape(B, NS * KP)
    # the kernel's bf16 operands; an fp32 cache (CPU tests) stays fp32 throughout, the mask form's arithmetic there
    rnd = (lambda t: t.to(torch.bfloat16).float()) if kc.dtype != torch.float32 else (lambda t: t.float())
    K = rnd(kc[tok])  # [B, T, R]
    s = torch.einsum("bhr,btr->bht", rnd(q_lat), K) * scale + bias.reshape(B, 1, NS * KP)
    p = rnd(torch.softmax(s, dim=-1))
    return torch.einsum("bht,btr->bhr", p, K)


def attend(q_lat: torch.Tensor, kc: torch.Tensor, rows: torch.Tensor, bias: torch.Tensor, scale: float) -> torch.Tensor:
    """o [B, H, R] fp32: emulate() of q_lat [B, H, R] over kc the latent cache [N, 1, R] or [N, R], rows int [B, NS]
    each slot's pool row (glm5_next.decode_slots), bias fp32 [B, NS, KP]."""
    return emulate(q_lat, kc.reshape(kc.shape[0], -1), rows, bias, scale)
