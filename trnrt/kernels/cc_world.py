"""The in-kernel world all-reduce of a fused decode layer (the fused DSA layer, kernels/dsa_satwf.py): one form,
bit-equal to XLA's all-reduce at 32 ranks on trn2 at LNC 2:
- both programs issue the all-reduce (NCC_ILLC059 otherwise); program 0 sends its rank's whole partial, program 1 zeros;
- program 0's sum lands straight in Y (a collective takes one buffer type on both sides, so the send buffers are shared
  too, one per program, and program 1 sums into its own Z1, never read);
- program 1 never reads a collective's own result (on the device it never lands there); it reads Y back after a
  core_barrier (ar_world_back).
The overlap is the caller's: any work traced between ar_world_issue and the first read of Y that touches neither S0
nor Y runs under the collective (~83% of the sum covered at world 4). The caller forms the rank's partial exactly as
XLA does (e.g. an LNC pair's halves summed the way XLA's LNCShuffle does), or the sum isn't XLA's.

In a kernel (every program runs every line; R <= 128 rows, W columns):
    Y, S0, S1, Z1 = CW.ar_world_alloc(R, W, nl.bfloat16)   # before any program-id branch (same address in both)
    CW.ar_world_zero1(S1, R, W, pid)                        # early, off program 0's critical path
    ...                                                    # program 0 writes its partial into S0
    CW.ar_world_issue(Y, S0, S1, Z1, world, pid)           # the collective
    ...                                                    # independent work (the overlap)
    CW.ar_world_back(Y, Y1, R, W, pid, npg)                # program 1's (and 0's) rows of Y, read back
A world of 1 (cc 0: the simulator refuses collectives) copies S0 into Y. Fold REV into the caller's kernel REV."""

from __future__ import annotations

import os
import zlib

try:  # the Neuron venv
    import nki
    import nki.collectives as ncc
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = ncc = nisa = nl = None

if nisa is not None:
    def _dma_static(dst, src):
        """Every address static (kernels/shell.py dma_static's form)."""
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.none)

    def world_ranks(world: int):
        """The world's ranks as a list (the tracer takes list literals and append, not list(range()))."""
        ranks = []
        for i in range(world):
            ranks.append(i)
        return ranks

    def ar_world_alloc(R: int, W: int, dtype):
        """(Y, S0, S1, Z1), shared HBM [R, W]: the sum, program 0's send buffer, program 1's (zeros) and its receive
        buffer (never read). Allocate before any program-id branch."""
        Y = nl.ndarray((R, W), dtype=dtype, buffer=nl.shared_hbm)
        S0 = nl.ndarray((R, W), dtype=dtype, buffer=nl.shared_hbm)
        S1 = nl.ndarray((R, W), dtype=dtype, buffer=nl.shared_hbm)
        Z1 = nl.ndarray((R, W), dtype=dtype, buffer=nl.shared_hbm)
        return Y, S0, S1, Z1

    def ar_world_zero1(S1, R: int, W: int, pid):
        """Program 1 fills its send buffer with zeros, from one [R, 512] tile a column chunk (whole-buffer memsets were
        3.5 us each of the R-partition Vector queue)."""
        if pid != 0:
            Wc = 512 if W % 512 == 0 else W
            zt = nl.ndarray((R, Wc), dtype=S1.dtype, buffer=nl.sbuf)
            nisa.memset(dst=zt, value=0.0)
            for c in range(W // Wc):
                _dma_static(dst=S1.ap(pattern=[[W, R], [1, Wc]], offset=c * Wc), src=zt)

    def ar_world_issue(Y, S0, S1, Z1, world: int, pid, cc: int = 1, npg: int = 2, fence: int = 1):
        """The world all-reduce, issued by both programs: program 0 sums S0 into Y, program 1 its zeros into Z1. cc 0:
        program 0 copies S0 into Y (world 1).
        fence 1: a core_barrier on each send buffer first. Nothing orders a DMA write into shared HBM before a later
        collective's read of it, and program 1's send feeds the sum: without the fence (its zero-fill issued just
        before the collective) the sum read wrong, nondeterministic 512-column blocks on the device; write ->
        core_barrier -> read is exact. fence 0 keeps that unfenced path, a negative control."""
        src = S0 if pid == 0 else S1
        dst = Y if pid == 0 else Z1
        if fence and npg > 1:
            nisa.core_barrier(data=S0, cores=(0, 1))
            nisa.core_barrier(data=S1, cores=(0, 1))
        if cc:
            ncc.all_reduce(srcs=[src], dsts=[dst], replica_group=ncc.ReplicaGroup([world_ranks(world)]), op=nl.add)
        else:
            if pid == 0:
                _dma_static(dst=Y, src=S0)

    def ar_world_back(Y, Y1, R: int, W: int, pid, npg):
        """After a core_barrier on Y (program 0's sum landed), each program reads its R / npg rows of Y back
        into Y1 (shared HBM [R, W]) through SBUF, then a barrier on Y1. R a multiple of npg."""
        n = R // npg
        if npg > 1:
            nisa.core_barrier(data=Y, cores=(0, 1))
        back = nl.ndarray((n, W), dtype=Y.dtype, buffer=nl.sbuf)
        _dma_static(dst=back, src=Y.ap(pattern=[[W, n], [1, W]], offset=pid * n * W))
        _dma_static(dst=Y1.ap(pattern=[[W, n], [1, W]], offset=pid * n * W), src=back)
        if npg > 1:
            nisa.core_barrier(data=Y1, cores=(0, 1))

    def ar_world_close(Y, npg):
        """Without ar_world_back: program 0's sum lands before either program ends."""
        if npg > 1:
            nisa.core_barrier(data=Y, cores=(0, 1))
else:
    world_ranks = ar_world_alloc = ar_world_zero1 = ar_world_issue = ar_world_back = ar_world_close = None


def _rev() -> int:
    """CRC-32 of this module's kernel source (LNL's compile-cache key does not hash NKI source)."""
    src = open(os.path.abspath(__file__)).read()
    a = src.index("if nisa is not None:")
    return zlib.crc32(src[a:src.index("def _rev() -> int:", a)].encode())


REV = _rev()
