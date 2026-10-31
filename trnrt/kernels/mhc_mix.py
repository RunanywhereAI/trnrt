"""GLM-5.3-Flash's hyper-connection output streams (models/hybrid.py mix_out, its "elementwise" form) as one NKI
kernel, in decode calls:
    out[t, n] = round_bf16(post[t, n] y[t] + sum_m comb[t, m, n] S[t, m])     (fp32, m in order, n = 0 .. hc - 1)
with y the block's output [T, H] and S the streams [T, hc, H], both bf16.

Why: at a decode call's 1-4 rows XLA runs these as ~20 [T, H] multiply-adds with a token per partition, each 4096
elements on one partition. Here H is spread over the partitions (h = p W + j, W = H / 128): a token's row of a stream
is a [128, W] slice, every op works on [128, T W], and the per-token scalars post / comb reach every partition by a
stride-0 DMA.

Arithmetic, mix_out's order: o = post_n * y; then o = o + comb[m, n] * S_m for m = 0 .. hc - 1, each product and sum
an fp32 op; then _round's barrier (Veltkamp: c = 65537 o, hi = c - (c - o)) and the bf16 copy, which is exact. So the
kernel is bit-identical to the host's mix_out where the vector engine's fp32 multiply and add are IEEE.
"""

from __future__ import annotations

import zlib

import torch

from . import shell as SH

P = 128


def emulate(post: torch.Tensor, comb: torch.Tensor, y: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """mix_out("elementwise")'s arithmetic: [T, hc * H] bf16 from post [T, hc], comb [T, hc, hc] fp32, y [T, H], S
    [T, hc, H] bf16."""
    T, hc, H = S.shape
    yf, outs = y.float(), []
    for n in range(hc):
        o = post[:, n : n + 1] * yf
        for m in range(hc):
            o = o + comb[:, m, n : n + 1] * S[:, m].float()
        c = o * 65537.0
        outs.append((c - (c - o)).to(torch.bfloat16))
    return torch.cat(outs, dim=-1)


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    @nki.jit
    def mhc_mix_kernel(post, comb, y, S, hc: int, rev: int):
        """See the module docstring. post fp32 [T, hc], comb fp32 [T, hc * hc] (row-major [m, n]), y bf16 [T, H], S bf16
        [T, hc * H]; T <= 128 (a decode call's rows); rev this module's source revision. Returns bf16 [T, hc * H]."""
        f32, bf16 = nl.float32, nl.bfloat16
        T, H = y.shape
        W = H // P
        out = nl.ndarray((T, hc * H), dtype=bf16, buffer=nl.shared_hbm)
        # the per-token scalars on every partition (stride-0 partition reads)
        pb = nl.ndarray((P, T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=pb, src=post.ap(pattern=[[0, P], [hc, T], [1, hc]], offset=0))
        cb = nl.ndarray((P, T, hc * hc), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cb, src=comb.ap(pattern=[[0, P], [hc * hc, T], [1, hc * hc]], offset=0))
        # y and the streams with h = p W + j: partition p holds W consecutive elements of every token's row
        yt = nl.ndarray((P, T, W), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=yt, src=y.ap(pattern=[[W, P], [H, T], [1, W]], offset=0))
        st = nl.ndarray((P, hc, T, W), dtype=bf16, buffer=nl.sbuf)
        for m in range(hc):
            nisa.dma_copy(dst=st[:, m, :, :], src=S.ap(pattern=[[W, P], [hc * H, T], [1, W]], offset=m * H))
        for n in range(hc):
            o = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=o, data1=yt, data2=pb.ap(pattern=[[T * hc, P], [hc, T], [0, W]], offset=n),
                               op=nl.multiply, engine=nisa.vector_engine)
            for m in range(hc):
                pr = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=pr, data1=st[:, m, :, :],
                                   data2=cb.ap(pattern=[[T * hc * hc, P], [hc * hc, T], [0, W]], offset=m * hc + n),
                                   op=nl.multiply, engine=nisa.vector_engine)
                nisa.tensor_tensor(dst=o, data1=o, data2=pr, op=nl.add, engine=nisa.vector_engine)
            # _round's barrier: c = 65537 o; hi = c - (c - o); then the exact bf16 copy
            c = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=c, data=o, op0=nl.multiply, operand0=65537.0, engine=nisa.vector_engine)
            d = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=d, data1=c, data2=o, op=nl.subtract, engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=c, data1=c, data2=d, op=nl.subtract, engine=nisa.vector_engine)
            r = nl.ndarray((P, T, W), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=r, src=c, engine=nisa.vector_engine)
            nisa.dma_copy(dst=out.ap(pattern=[[W, P], [hc * H, T], [1, W]], offset=n * H), src=r)
        return out
else:
    mhc_mix_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source, passed as the static argument `rev` (LNL's compile-cache key does not
    include NKI kernel source: kernels/delta_rule.py _kernel_rev)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    mhc_mix_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev() ^ zlib.crc32(repr((P,)).encode())  # and the constants its kernel reads


def takes(device: torch.device, H: int, T: int) -> bool:
    """Whether mix_out uses the kernel: on a device, 128 | H, T <= 128. At LNC=2 (trn2) it runs at the runtime's grid
    with no split: both physical cores compute every row and write the same output."""
    return (device.type != "cpu" or SH.HOST_SIM) and H % P == 0 and T <= P


def mix(post: torch.Tensor, comb: torch.Tensor, y: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """[T, hc * H] bf16: the kernel in the caller's graph on the device, emulate() on the host."""
    T, hc, H = S.shape
    if S.device.type == "cpu" and not SH.HOST_SIM:
        return emulate(post, comb, y, S)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if mhc_mix_kernel is None:
        raise RuntimeError("the NKI mHC mix kernel needs the nki package (the Neuron venv)")
    return wrap_nki(mhc_mix_kernel)[platform.nki_grid()](
        post=post.float().contiguous(), comb=comb.float().reshape(T, hc * hc).contiguous(),
        y=y.to(torch.bfloat16).contiguous(), S=S.to(torch.bfloat16).reshape(T, hc * H).contiguous(), hc=int(hc),
        rev=REV)
