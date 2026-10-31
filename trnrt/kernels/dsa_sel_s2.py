"""S2, the context-parallel sum of sel2's local lists inside the fused DSA decode layer's kernels
(kernels/dsa_sa.py's section).

After sel2 writes lval / cpool / rows to `out` [3, B, keep] and closes with its core barrier, both programs read every
row's lval and cpool back and build models/mla.py _cp_merge_gather's zero-padded payload in SBUF: slot a is
[lval | cpool] * onehot[a] in fp32, the rank's values in its own slot and x * 0.0 in the others (XLA's signed zeros,
-1e30 * 0.0 = -0.0). The caller stages it to HBM and all-reduces it over the rank's CP group, bit-equal to XLA's group
sum at 32 ranks.

Its own module, because the NKI frontend resolves nki.collectives only from the kernel's own module. REV carries
dsa_shell's sel2 revision (REV2)."""

from __future__ import annotations

from . import dsa_shell as DS
from . import shell as SH

try:  # the Neuron venv
    import nki
    import nki.collectives as ncc
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

# -- the kernel source
if nki is not None:
    from .dsa_shell import I32, VE, _sb, dma_static

    def _s2_payload(out, rank, B: int, keep: int, A: int, src, pq: int = 0):
        """S2's payload: every row's [lval | cpool] from out [3, B, keep] (both programs' rows, after sel2's barrier),
        times this rank's one-hot over the A slots, into src [A B, 2 keep] (row a B + b: slot a, row b). rank fp32
        [1, 1] is the rank's index in its CP group (models/mla.py long_grp_index), as sel2 takes it.
        pq 1 (served): the same payload in one pass, row a B + b on partition a B + b: [lval | cpool]
        replicated over the A slots by 2 A independent DMAs from out, the one-hot as a per-partition scalar
        ((p - rank B) in [0, B)), one multiply (x * 1.0 / x * 0.0 per element, pq 0's arithmetic, so its bits), one DMA
        into src. pq 0's A multiplies of [B, 2 keep] through one reused tile wait on each other's DMAs (~20 us a DSA layer
        from sel2's end to S2's issue at 4 rows on trn2)."""
        W2 = 2 * keep
        if pq:
            AB = A * B
            XR = _sb((AB, W2))
            for a in range(A):
                dma_static(dst=XR[a * B:(a + 1) * B, 0:keep], src=out.ap(pattern=[[keep, B], [1, keep]], offset=0))
                dma_static(dst=XR[a * B:(a + 1) * B, keep:W2], src=out.ap(pattern=[[keep, B], [1, keep]],
                                                                         offset=B * keep))
            RKP = _sb((AB, 1))
            dma_static(dst=RKP, src=rank.ap(pattern=[[0, AB], [1, 1]], offset=0))
            lo = _sb((AB, 1))
            nisa.tensor_scalar(dst=lo, data=RKP, op0=nl.multiply, operand0=float(B), engine=VE)  # rank B, exact
            pi = _sb((AB, 1), I32)
            nisa.iota(dst=pi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
            pfp = _sb((AB, 1))
            nisa.tensor_copy(dst=pfp, src=pi, engine=VE)
            dd = _sb((AB, 1))
            nisa.tensor_scalar(dst=dd, data=pfp, op0=nl.subtract, operand0=lo, engine=VE)  # p - rank B
            g0 = _sb((AB, 1))
            nisa.tensor_scalar(dst=g0, data=dd, op0=nl.greater_equal, operand0=0.0, engine=VE)
            g1 = _sb((AB, 1))
            nisa.tensor_scalar(dst=g1, data=dd, op0=nl.less, operand0=float(B), engine=VE)
            OHP = _sb((AB, 1))  # [a B + b] = (a == rank): 1.0 / 0.0
            nisa.tensor_scalar(dst=OHP, data=g0, op0=nl.multiply, operand0=g1, engine=VE)
            PR = _sb((AB, W2))
            nisa.tensor_scalar(dst=PR, data=XR, op0=nl.multiply, operand0=OHP, engine=VE)
            dma_static(dst=src.ap(pattern=[[W2, AB], [1, W2]], offset=0), src=PR)
            return
        XL = _sb((B, W2))
        dma_static(dst=XL[:, 0:keep], src=out.ap(pattern=[[keep, B], [1, keep]], offset=0))
        dma_static(dst=XL[:, keep:W2], src=out.ap(pattern=[[keep, B], [1, keep]], offset=B * keep))
        RKB = _sb((B, 1))
        dma_static(dst=RKB, src=rank.ap(pattern=[[0, B], [1, 1]], offset=0))
        ai = _sb((B, A), I32)
        nisa.iota(dst=ai, pattern=[[1, A]], offset=0, channel_multiplier=0)
        af = _sb((B, A))
        nisa.tensor_copy(dst=af, src=ai, engine=VE)
        OH = _sb((B, A))  # [b, a] = (a == rank): 1.0 / 0.0, models/mla.py long_grp_onehot
        nisa.tensor_scalar(dst=OH, data=af, op0=nl.equal, operand0=RKB, engine=VE)
        for a in range(A):
            PA = _sb((B, W2))
            nisa.tensor_scalar(dst=PA, data=XL, op0=nl.multiply, operand0=OH[:, a:a + 1], engine=VE)
            dma_static(dst=src.ap(pattern=[[W2, B], [1, W2]], offset=a * B * W2), src=PA)
else:
    _s2_payload = None
# -- the kernel source end


REV = (SH.kernel_rev(__file__, "# -- the kernel source", "# -- the kernel source end") ^ DS.REV2)
