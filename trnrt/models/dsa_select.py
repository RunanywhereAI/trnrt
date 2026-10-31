"""Device-safe top-k indices (the MoE router's), and the DSA selection's tie rule.

The DSA layers' selections themselves (models/glm5_next.py, models/dsa_long.py) never run torch.topk:
they go through kernels/dsa_topk.py, an exact radix search over the threshold's bit pattern and a
binary search over the tied positions in one NKI kernel (its torch emulation on the host). torch.topk
over the context dominated a DSA layer on trn1 (3.2 ms of a 6.0 ms indexer layer at 4K keys, decode
B=8), and turning its indices into an attention mask with a scatter cost 6.7 ms more in a 128-token
prefill chunk.

Tie rule (the definition the tests check, reference_mask): exactly `keep` positions are selected
(every position when a row has fewer); every selected score is >= every unselected one; among the
scores equal to the keep-th largest, the lowest indices are taken. torch.topk returns the same set
whenever the keep-th and (keep + 1)-th largest scores differ, and an arbitrary choice of the tied ones
otherwise (CPU torch.topk is not lowest-index-first among ties). -0.0 and +0.0 are equal, as in every
float comparison. Scores must not be NaN.
"""

from __future__ import annotations

import torch


def topk_ids(scores: torch.Tensor, k: int) -> torch.Tensor:
    """torch.topk's indices along the last axis, safe to gather or scatter at on a Neuron device. A device top-k over
    a row holding NaN returns index 0xFFFFFFFF for every k, and the gather at it faults (trn2, SDK 2.32: "scatter/gather
    (indirect memory copy via vector DGE) out-of-bound access"), so NaN and -inf rank as finfo.min and the indices are
    clamped into range. A row without NaN or -inf selects exactly as before."""
    lo = torch.full_like(scores, torch.finfo(scores.dtype).min)
    s = torch.where(torch.isnan(scores) | torch.isneginf(scores), lo, scores)
    return torch.topk(s, k, dim=-1).indices.clamp(0, scores.shape[-1] - 1)


def reference_mask(scores: torch.Tensor, keep: int) -> torch.Tensor:
    """The tie rule spelled out with a stable sort (tests): descending score, then ascending index."""
    if keep >= scores.shape[-1]:
        return torch.ones_like(scores, dtype=torch.bool)
    order = torch.sort(scores.float(), dim=-1, descending=True, stable=True).indices[..., :keep]
    return torch.zeros_like(scores, dtype=torch.bool).scatter(-1, order, True)
