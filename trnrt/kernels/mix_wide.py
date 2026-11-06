"""shell.mix_in_h's mix with the hc output streams batched into each vector instruction: the served form, which
kernels/shell.py binds as mix_in_h outside its hashed in-kernel span. Every caller of mix_in_h folds shell.MIX_REV into
its REV.

Why: the per-stream form (shell.py's mix_in_h, "fin") runs every op once per output stream at free T W (128 at four
rows), so about half of each vector instruction is its fixed ~153 ns. At four rows that was 1108 us per step over 75
calls, 219% of the vector engine's fp32 floor.

The form (wide-cast): every op runs on [128, hc, T, W] (hc T W elements per partition) instead of hc ops on [128, T,
W]; the operand that does not depend on the stream n is read through a stride-0 access pattern. Each element sees fin's
fp32 ops in fin's order (y post_n, then + comb[m, n] S_m for m = 0 .. hc - 1), and fin's barrier and cast to bf16 are
one fp32 -> bf16 cast (round to nearest even). That equals fin's bits on every normal value up to fp32_max / 65537
(5.19e33), and on +-0 (exhaustive over the fp32 patterns, numpy). On subnormals the barrier double-rounds (it flushes,
or lands one bf16 ulp off) and the cast is the RNE result; the device's vector-engine cast is RNE with subnormals
kept. Above 5.19e33 the barrier is non-finite and the cast is not. cast 0 in _mix is the barrier form, bit for bit
fin's by construction.
"""

from __future__ import annotations

import os
import zlib

P = 128

try:  # the Neuron venv
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

if nki is not None:
    def _dma(dst, src):
        """A DMA with every address static (shell.dma_static's form)."""
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.none)

    def _mix(sh, S, post_p, comb_p, Yp, S_out, hc: int, T: int, H: int, npg: int, cast: int):
        """shell.mix_in_h's contract, wide: sh [128, hc, T, W] bf16 and S_out (HBM [T, hc H]) = stream n = post_n y +
        sum_m comb[m, n] S_m in fp32, m in order, rounded to bf16 (cast 0: _round's barrier, then the cast; 1: the
        cast alone). y = round_bf16(sum of the npg partials Yp [npg, T, H])."""
        f32, bf16 = nl.float32, nl.bfloat16
        VE = nisa.vector_engine
        W, F = H // P, hc * H
        pb0 = nl.ndarray((P, T, hc), dtype=f32, buffer=nl.sbuf)
        _dma(pb0, post_p.ap(pattern=[[0, P], [hc, T], [1, hc]], offset=0))
        cb0 = nl.ndarray((P, T, hc * hc), dtype=f32, buffer=nl.sbuf)
        _dma(cb0, comb_p.ap(pattern=[[0, P], [hc * hc, T], [1, hc * hc]], offset=0))
        st0 = nl.ndarray((P, hc, T, W), dtype=bf16, buffer=nl.sbuf)
        for n in range(hc):
            _dma(st0[:, n, :, :], S.ap(pattern=[[W, P], [F, T], [1, W]], offset=n * H))
        ya = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
        for g in range(npg):
            yg = nl.ndarray((P, T, W), dtype=bf16, buffer=nl.sbuf)
            _dma(yg, Yp.ap(pattern=[[W, P], [H, T], [1, W]], offset=g * T * H))
            if g == 0:
                nisa.tensor_copy(dst=ya, src=yg, engine=VE)
            else:
                nisa.tensor_tensor(dst=ya, data1=ya, data2=yg, op=nl.add, engine=VE)
        ybf = nl.ndarray((P, T, W), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ybf, src=ya, engine=VE)
        # o[p, n, t, j] = y[p, t, j] post[t, n], then + S_m[p, t, j] comb[t, m, n] for m in order: one op each
        o = nl.ndarray((P, hc, T, W), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o, data1=ybf.ap(pattern=[[T * W, P], [0, hc], [W, T], [1, W]], offset=0),
                           data2=pb0.ap(pattern=[[T * hc, P], [1, hc], [hc, T], [0, W]], offset=0),
                           op=nl.multiply, engine=VE)
        for m in range(hc):
            pr = nl.ndarray((P, hc, T, W), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pr, data1=st0.ap(pattern=[[hc * T * W, P], [0, hc], [W, T], [1, W]], offset=m * T * W),
                               data2=cb0.ap(pattern=[[T * hc * hc, P], [1, hc], [hc * hc, T], [0, W]], offset=m * hc),
                               op=nl.multiply, engine=VE)
            nisa.tensor_tensor(dst=o, data1=o, data2=pr, op=nl.add, engine=VE)
        if cast:
            nisa.tensor_copy(dst=sh, src=o, engine=VE)
        else:
            c = nl.ndarray((P, hc, T, W), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=c, data=o, op0=nl.multiply, operand0=65537.0, engine=VE)
            d = nl.ndarray((P, hc, T, W), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=d, data1=c, data2=o, op=nl.subtract, engine=VE)
            nisa.tensor_tensor(dst=c, data1=c, data2=d, op=nl.subtract, engine=VE)
            nisa.tensor_copy(dst=sh, src=c, engine=VE)
        for n in range(hc):
            _dma(S_out.ap(pattern=[[W, P], [F, T], [1, W]], offset=n * H), sh[:, n, :, :])

    def mix_in_h_wide_cast(sh, S, post_p, comb_p, Yp, S_out, hc: int, T: int, H: int, npg: int):
        """shell.mix_in_h's served form: the streams batched, one cast for the barrier and the cast."""
        _mix(sh, S, post_p, comb_p, Yp, S_out, hc, T, H, npg, 1)
else:
    mix_in_h_wide_cast = None


def _kernel_rev() -> int:
    """CRC-32 of the in-kernel source span and the constant it reads."""
    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    mix_in_h_wide_cast = None", a)
    return zlib.crc32(f"{src[a:b]}|{P}".encode())


REV = _kernel_rev()

# -- the hardware-DGE span (TRNRT_DMA_HW): with TRNRT_DMA_HW=1 this module's static DMA helper (_dma) issues its DMAs in
# hardware DGE (dge_mode hwdge) instead of static descriptors, which cost a descriptor-ring entry per contiguous run in
# every loaded NEFF (the decode spill rings). Every MTP preset sets it (it is in every build behind the published MTP
# numbers); the default presets do not. Read at import, before any kernel module binds _dma by name or folds REV, so the
# process environment has to carry it (trnrt/__main__.py sets it for a preset that needs it). Below the hashed span;
# REV folds this span's source when it is set (_dma_hw_rev), so every kernel folding REV re-keys, and unset no source,
# REV or key moves.
if os.environ.get("TRNRT_DMA_HW", "0") == "1" and nki is not None:
    def _dma(dst, src):  # noqa: F811
        """_dma under TRNRT_DMA_HW=1: hardware DGE."""
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge)
# -- the hardware-DGE span end


def _dma_hw_rev() -> int:
    """CRC-32 of the hardware-DGE span above (the TRNRT_DMA_HW variant's source): REV's term when it is set."""
    src = open(os.path.abspath(__file__)).read()
    a = src.index("# -- the hardware-DGE span (TRNRT_DMA_HW)")
    return zlib.crc32(src[a:src.index("# -- the hardware-DGE span end", a)].encode())


if os.environ.get("TRNRT_DMA_HW", "0") == "1":
    REV ^= _dma_hw_rev()

# What a mix_in_h caller folds into its REV: a CRC of this module's REV and the form.
MIX_REV = zlib.crc32(repr((REV, "wide-cast")).encode())
