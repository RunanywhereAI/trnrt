"""The sequence-parallel world row gather (models/decoder.py _sp_gather) as an all_gather issued from an NKI kernel
(nki.collectives): every rank's rows [r, W] -> [n r, W] in rank order.

Why: the default form is a zero-padded all-reduce (x in its block of a zero [n, r, W], summed over the ranks), because a
graph holding an XLA all-gather breaks when a later process loads its NEFF from the compile cache (a replica group
signature mismatch). A ring all-reduce of the padded buffer moves ~2 (n - 1) / n of it per rank, an all-gather (n - 1) / n.
Measured on trn2.48xlarge at LNC=2 (32 ranks, bf16 [128, 4096] per rank): 437 us for a 32 MiB gather in a graph, against
950 us for the zero-padded all-reduce, with the cached NEFF reloading in a later process.

nki.collectives constraints (neuronx-cc 2.27): a collective cannot read the kernel's IO tensors, and writing the kernel's
output directly fails to compile too, so the rows are copied into private HBM scratch, gathered there and copied to the
output. At LNC=2 both programs of the logical core issue the whole all_gather (a local collective needs both cores' memory
locations), fenced by core barriers on its source, its destination and the output; the collective delivers into program
0's buffer, which program 0 copies to the output.

Exactness: a gather moves bytes, so the output equals the zero-padded all-reduce's bit for bit (one rank's value plus
zeros per element).

The engine sets MODE ("xla", the default, or "nki") and TP_EXPERTS on every rank before the shard is built
(EngineConfig.sp_gather / sp_gather_tp_experts), as it does the decode kernel routes.
"""

from __future__ import annotations

import torch

# "xla": the zero-padded all-reduce; "nki": this kernel for the world gather.
MODE = "xla"
# Rows per rank from which the kernel replaces the zero-padded all-reduce: a prefill chunk's 128 (4096 rows at tp 32), not a
# decode call's few, so the decode graphs keep the all-reduce.
MIN_ROWS = 16
# Row width from which it does: the hidden rows (4096), not the MoE routing's [r, 2k] fp32 (16 wide), whose gathers are
# latency-bound and kept as the zero-padded all-reduce.
MIN_WIDTH = 256
# Whether the kernel gathers with tensor-parallel experts too (models/decoder.py _sp_gather_kernel_ok).
TP_EXPERTS = False

try:  # the Neuron venv
    import nki
    import nki.collectives as ncc
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:

    def _groups(world: int, gsize: int):
        """[[0 .. gsize - 1], [gsize .. 2 gsize - 1], ..] as list literals (the NKI tracer takes no list(range()))."""
        out = []
        for g in range(world // gsize):
            ranks = []
            for i in range(gsize):
                ranks.append(g * gsize + i)
            out.append(ranks)
        return ncc.ReplicaGroup(out)

    @nki.jit
    def sp_gather_dupf_kernel(x, world: int, gsize: int, rev: int):
        """x [r, W] this rank's rows -> [gsize r, W] (gsize = world): xs <- x, a core barrier on xs, the all_gather into
        private gs (both programs), a core barrier on gs, program 0 copies gs to the output, a core barrier on the output.
        rev: this module's kernel source revision."""
        r, W = x.shape
        out = nl.ndarray((gsize * r, W), dtype=x.dtype, buffer=nl.shared_hbm)
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        xs = nl.ndarray((r, W), dtype=x.dtype, buffer=nl.private_hbm)
        gs = nl.ndarray((gsize * r, W), dtype=x.dtype, buffer=nl.private_hbm)
        nisa.dma_copy(dst=xs, src=x)
        if npg > 1:
            nisa.core_barrier(data=xs, cores=(0, 1))
        ncc.all_gather(srcs=[xs], dsts=[gs], replica_group=_groups(world, gsize), collective_dim=0)
        if npg > 1:
            nisa.core_barrier(data=gs, cores=(0, 1))
        if pid == 0:
            nisa.dma_copy(dst=out, src=gs)
        if npg > 1:
            nisa.core_barrier(data=out, cores=(0, 1))
        return out

else:
    sp_gather_dupf_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source, passed as the static argument `rev` (LNL's compile-cache key does not include
    NKI kernel source: kernels/delta_rule.py _kernel_rev)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    sp_gather_dupf_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()


def enabled(x: torch.Tensor) -> bool:
    """Whether the kernel gathers x (the world gather)."""
    return (MODE == "nki" and x.device.type != "cpu" and x.dim() == 2 and x.shape[0] >= MIN_ROWS
            and x.shape[1] >= MIN_WIDTH and sp_gather_dupf_kernel is not None)


def gather(x: torch.Tensor, world: int, gsize: int) -> torch.Tensor:
    """Every rank's x [r, W], [world r, W], by the kernel (enabled() must hold). The world only: no in-kernel group
    collective is proven at LNC=2."""
    if gsize != world:  # (before any device import: the host tests reach it)
        raise ValueError(f"sp_gather gathers the world only ({gsize} of {world} asked)")
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    grid = platform.nki_grid()
    if grid < 2:
        raise ValueError(f"sp_gather's kernel is the LNC=2 form (an NKI grid of 2), not grid {grid}")
    return wrap_nki(sp_gather_dupf_kernel)[grid](x=x.contiguous(), world=world, gsize=gsize, rev=REV)
