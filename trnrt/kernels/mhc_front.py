"""A DSA layer's attention-block mHC without its Sinkhorn. mhc_pre's comb chain ran alone ~15.5 us a call at four decode
rows, and only the fused DSA layer's mix_in reads comb, so the chain moves into that call (kernels/dsa_satwfc.py), which
steps it under its own collective waits.

    front(S, fn, base, scale, iters, eps, rms_eps) -> (x [T, H] bf16, post [T, hc] fp32, PendingComb(c [T, hc, hc]))

The kernel is mhc_split.front_body's k-layout form (front_body_k: an attention-side fn, which the j-major copy never
re-lays), with the Sinkhorn's input c = softmax(logits) + eps written where mhc_pre writes comb. front_body is
pre_body's instructions (tests/test_mhc_split.py), so x and post are mhc_pre's to the bit, and
mhc_split.sinkhorn_body on c's tile (dsa_satwfc) is mhc_pre's comb. c is wrapped in PendingComb so no consumer can
read it as comb: the fused call unwraps it; models/hybrid.py's fallback recomputes mhc_pre's comb."""

from __future__ import annotations

import zlib

import torch

from . import mhc_pre as MP
from . import mhc_split as MS
from . import shell as SH

P = MP.P


class PendingComb:
    """The Sinkhorn's input c [T, hc, hc] fp32 (softmax + eps), not yet comb: only kernels/dsa_satwfc.py consumes it."""

    __slots__ = ("c",)

    def __init__(self, c: torch.Tensor):
        self.c = c


def emulate(S: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, iters: int, eps: float,
            rms_eps: float):
    """mhc_pre.emulate's steps up to the Sinkhorn: (x [T, H] bf16, post [T, hc], c [T, hc, hc])."""
    T, hc, H = S.shape
    flat = S.reshape(T, -1).float()
    r = torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + rms_eps)
    mix = torch.nn.functional.linear(flat, fn.float()) * r
    pre = torch.sigmoid(mix[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mix[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    c = torch.softmax(mix[:, 2 * hc :].view(T, hc, hc) * scale[2] + base[2 * hc :].view(hc, hc), dim=-1) + eps
    x = pre[:, 0:1] * S[:, 0].float()
    for n in range(1, hc):
        x = x + pre[:, n : n + 1] * S[:, n].float()
    k = x * 65537.0
    return (k - (k - x)).to(torch.bfloat16), post, c


def sinkhorn(c: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """mhc_pre.emulate's Sinkhorn rounds on c (host fp32): emulate()'s c through this is mhc_pre.emulate's comb."""
    c = c / (c.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        c = c / (c.sum(dim=-1, keepdim=True) + eps)
        c = c / (c.sum(dim=-2, keepdim=True) + eps)
    return c


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    @nki.jit
    def mhc_front_kernel(S, fn, base, scale, hc: int, iters: int, eps: float, rms_eps: float, sig: int, rev: int,
                              skf: int):
        """mhc_pre's kernel less the Sinkhorn: the same arguments; returns (x bf16 [T, H], post fp32 [T, hc], c fp32
        [T, hc * hc]), c the Sinkhorn's input (softmax + eps) in comb's place; rev this module's source revision."""
        T = S.shape[0]
        x_out, post_out, c_out, cp, _ = MS.front_body_k(S, fn, base, scale, hc, iters, eps, rms_eps, sig, skf)
        if skf == 0:
            nisa.dma_copy(dst=c_out.ap(pattern=[[hc * hc, T], [hc, hc], [1, hc]], offset=0), src=cp)
        else:  # the padded tile's inner hc x hc (its eps row and column are the consumer's memset)
            nisa.dma_copy(dst=c_out.ap(pattern=[[hc * hc, T], [hc, hc], [1, hc]], offset=0), src=cp[:, 0:hc, 0:hc])
        return x_out, post_out, c_out
else:
    mhc_front_kernel = None


REV = (SH.kernel_rev(__file__, "try:  # the Neuron venv", "    mhc_front_kernel = None") ^ MS.REV_K
       ^ zlib.crc32(repr((P,)).encode()))


def front(S: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, iters: int, eps: float,
          rms_eps: float):
    """(x [T, H] bf16, post [T, hc] fp32, PendingComb(c [T, hc, hc] fp32)): the kernel on the device at mhc_pre's
    switches (mhc_pre.SIGMOID, SK, PBC); emulate() on the host."""
    T, hc, H = S.shape
    if S.device.type == "cpu" and not SH.HOST_SIM:
        x, post, c = emulate(S, fn, base, scale, iters, eps, rms_eps)
        return x, post, PendingComb(c)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if mhc_front_kernel is None:
        raise RuntimeError("the NKI mHC front kernel needs the nki package (the Neuron venv)")
    x, post, c = wrap_nki(mhc_front_kernel)[platform.nki_grid()](
        S=S.to(torch.bfloat16).reshape(T, hc * H).contiguous(), fn=fn.to(torch.bfloat16).contiguous(),
        base=base.float().contiguous(), scale=scale.float().contiguous(), hc=int(hc), iters=int(iters),
        eps=float(eps), rms_eps=float(rms_eps), sig=MP.SIGMOID, rev=REV, skf=MP.SK)
    return x, post, PendingComb(c.view(T, hc, hc))
