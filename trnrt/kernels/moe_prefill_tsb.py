"""TSB, the prefill MoE's token scatter built in SBUF: the served prefill MoE kernel (models/hybrid.py _moe_clamped).
kernels/moe_prefill_mdr.py's kernel (MDR: the fused stage-C drain), copied as its own kernel function with its own REV,
with stage 1 replaced. MDR writes tos (each slot's token) into HBM with NT x K indirect DMAs of 128 int32 (256 a call per
program at C 4,096, software DGE on the GpSimd queue) and reads it back as ts [lane, tile]; dropping those scatters made
a 4,096-row call ~0.7 ms faster. TSB builds ts directly in PSUM: for each group (token tile T, choice k) a 0/1 matmul of
the lane one-hot [p, lane(slot)] (stationary) by the tile one-hot [p, tile(slot)] times C - token (moving, as two
bf16-exact parts 64 hi + lo), accumulated over the groups in fp32. Every slot holds one pair or none, so each sum is one
exact integer (C - token, or 0 where the slot is empty), and ts = C - sum is the scatter's ts bit for bit: C (out of
range, skipped) where empty. No HBM tos, no indirect DMA; the rest of the kernel is MDR's, so the output is MDR's bit for
bit."""

from __future__ import annotations

import zlib

import torch

from . import moe_prefill_mdr as MDR
from .moe_prefill import ACT_ORDER, DRT, DVD, P, RK  # the module constants the copied kernel reads (folded into REV)


# --- kernel ---------------------------------------------------------------------------------------

try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from nki.isa.constants import oob_mode
except ImportError:
    nki = None

# Defined at import, not inside the traced code.
if nki is not None:
    @nki.jit
    def moe_prefill_tsb_kernel(x, topi, wts, blob, dsc, dfr, fold_g, fold_u, B: int, NB: int, act: int,
                                lim: float, dq: int, pc: int, asp: int, skp: int, ord_: int, nyb: int, rev: int,
                                spl: int = 1, inb: int = 0, nsx: int = 0, bng: int = 0, dge: int = 0):
        """x bf16 [C, H]; topi int32 [C, K] the experts of each token and wts bf16 [C, K] their routing
        weights; blob uint8 [E, 128, F] (moe_dedupe tiles layout, fp32 or bf16 tile scales); dsc fp32
        [E, CT] and dfr bf16 [E, H]: the down scales as s[h] = dsc[chunk] * dfr[h] (down_factors; dfr
        powers of two), read when pc = 1; fold_g / fold_u
        bf16 [128, 128] = [o == q % 64] / [o == 64 + q % 64]; B lanes per block (64 or 128) and NB =
        n_blocks(C K, E, B); act 0 silu, 1 silu with gate <= lim and |up| <= lim (moe_dedupe.ACTS);
        dq 1: the blob's gate rows share each tile scale and so do its up rows (check_blob), so the
        gate_up tiles are dequantized to bf16 in SBUF and accumulated in PSUM; dq 0: each tile's
        partial is scaled per row on the vector engine; pc 1: down scales per output column (dsc, dfr:
        the drain applies dsc per 128-column chunk, then the bf16 product is multiplied by dfr, exact for
        a power of two), 0: one per 128-column chunk, read from the blob; asp: dq only, how many of
        the eight half-tile dequantizations of each block and round run on the scalar engine instead
        of the vector engine (ACT_ORDER; the same fp32 product rounded once to bf16); rev: this
        function's source revision (REV, unused here: see _kernel_rev); skp: 0 runs every static lane
        tile, n > 0 runs the tiles past the first n in chunks of n that are skipped once the routing
        has no block left in them (device loops of trip count 0 or 1); ord_: 1 issues each pipeline round's
        stage B (gate_up matmuls and their scaling) before its stage C down matmuls (_stage_b), 0 after;
        nyb: stage C's down outputs in a ring of nyb buffers (0: one allocation per step). Neither
        changes an operation or its order on any accumulator: the output is the same bit for bit.
        pc 2 (MDR; pc 1's path otherwise): stage C's drain fused, dsc then holding sel3_matrix(B) (bf16 [3 G, 128]) and
        dfr the fused_drain_table (bf16 [E, 3 H]): the selector matmul sums each block's three pieces into
        T = s f exactly in fp32 PSUM, the scalar engine copies T to SBUF, and the vector engine writes the
        chunk as bf16(py T) straight from PSUM (two PSUM operands are refused), equal bit for bit to fdr 0;
        pc 3 the positive control (tests): py drained to bf16 first, then times T (rounded twice).
        Returns bf16 [C, H]."""
        C, H = x.shape
        K = topi.shape[1]
        E = blob.shape[0]
        G = 128 // B
        NTL = NB // G
        F = blob.shape[2]
        DW = H // 2
        CT = H // 128
        SB = (F - H - DW) // (2 * CT)
        o_sg = H + DW
        o_sd = o_sg + CT * SB
        MX = 1 if SB == 2 else 0  # bf16 tile scales (re-based MXFP4): per-row gate_up scales in bf16
        PC = pc  # down scales per output column: dsc per chunk times dfr per column (always for MX)
        fdr = pc - 1 if pc >= 2 else 0  # FDR: pc 2 is pc 1 with stage C's drain fused (MDR), pc 3 its rounded-twice control
        NT = C // 128
        fp8 = nl.float8_e4m3
        f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
        # LNC (trn2 at LNC=2, grid 2): the kernel is traced once per program (the two physical cores of
        # the logical core; nki/_backends/mlir_tracer program_id: "kernel is traced LNC times with
        # different program_id_value"), so npg / pid are Python ints. Each program runs half of the lane
        # tiles and half of the combine's token tiles; the plan is computed by both. With grid 1 (trn1)
        # npg = 1 and every step is the single-core kernel's.
        # spl 0 (no LNC split): both programs do all of the work, as before the split.
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if spl and nl.program_ndim() != 0 else (1, 0)
        # dge 1 (an experiment, off): every dynamic DMA (vector_offset gathers / scatters,
        # scalar_offset expert loads) on software DGE, where neuronx-cc otherwise picks the mode; trn2 has hardware DGE
        # on the sync and scalar engines ("Hardware based DGE is only supported for NeuronCore-v3 or newer",
        # nki/isa dma_copy), and the split's out-of-bound notification came from an SP-engine instruction
        DG = nisa.dge_mode.swdge if dge else nisa.dge_mode.unknown
        if npg > 1 and inb:  # inb 1 (an experiment, off): both programs at the kernel's start before either
            # reads its inputs (an input written by XLA ops on the other physical core)
            nisa.core_barrier(data=topi, cores=(0, 1))
            nisa.core_barrier(data=x, cores=(0, 1))
            nisa.core_barrier(data=wts, cores=(0, 1))

        fg = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=fg, src=fold_g)
        fu = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=fu, src=fold_u)
        ident = nl.shared_identity_matrix(n=128, dtype=bf16)

        # 0. The plan (plan() is its CPU reference), all integers exact in fp32: a pair's rank among
        # the earlier pairs of its expert in the order (token tile, k, token), then its block, lane,
        # slot and tos position, and each block's expert. Constants first.
        io = nl.ndarray((128, E), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=io, pattern=[[1, E]], offset=0, channel_multiplier=0)
        iof = nl.ndarray((128, E), dtype=f32, buffer=nl.sbuf)  # e on every partition
        nisa.tensor_copy(dst=iof, src=io, engine=nisa.vector_engine)
        uu = nl.ndarray((128, 128), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=uu, pattern=[[1, 128]], offset=0, channel_multiplier=-1)  # q - p
        uuf = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=uuf, src=uu, engine=nisa.vector_engine)
        lt = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)  # [p < q]: earlier tokens of a group
        nisa.tensor_scalar(dst=lt, data=uuf, op0=nl.greater, operand0=0.0, engine=nisa.vector_engine)
        r64 = nl.ndarray((1, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=r64, value=64.0)
        r1 = nl.ndarray((1, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=r1, value=1.0)
        on64 = nl.ndarray((128, 64), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=on64, value=1.0)
        tk = nl.ndarray((128, NT, K), dtype=i32, buffer=nl.sbuf)  # token T * 128 + p
        nisa.dma_copy(dst=tk, src=topi.ap(pattern=[[K, 128], [128 * K, NT], [1, K]], offset=0))
        tkf = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tkf, src=tk, engine=nisa.vector_engine)
        # Earlier groups' one-hots, summed per partition (token), in ranges of 256 groups so that every
        # entry is an integer <= 256 (bf16-exact); a ones matmul sums them over the partitions into
        # each expert's count of earlier pairs, on every partition at once.
        NG = NT * K
        on128 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=on128, value=1.0)
        srs = []
        for r in range(-(-NG // 256)):
            s0 = nl.ndarray((128, E), dtype=bf16, buffer=nl.sbuf)
            nisa.memset(dst=s0, value=0.0)
            srs.append(s0)
        rk = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)
        for T in range(NT):
            for k in range(K):
                gi = T * K + k
                oh = nl.ndarray((128, E), dtype=bf16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=oh, data=iof, op0=nl.equal, operand0=tkf[:, T, k:k + 1],
                                   engine=nisa.vector_engine)
                pp = nl.ndarray((128, E), dtype=f32, buffer=nl.psum)
                nisa.nc_matmul(dst=pp, stationary=lt, moving=oh, accumulate=False)
                for r in range(-(-gi // 256)):  # ranges holding earlier groups
                    nisa.nc_matmul(dst=pp, stationary=on128, moving=srs[r], accumulate=True)
                tmp = nl.ndarray((128, E), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=tmp, data1=oh, data2=pp, op=nl.multiply, engine=nisa.vector_engine)
                nisa.tensor_reduce(dst=rk[:, T, k:k + 1], op=nl.add, data=tmp, axis=1)
                nisa.tensor_tensor(dst=srs[gi // 256], data1=srs[gi // 256], data2=oh, op=nl.add,
                                   engine=nisa.vector_engine)
        pc = nl.ndarray((64, E), dtype=f32, buffer=nl.psum)  # pairs per expert (64 equal rows)
        for r in range(len(srs)):
            nisa.nc_matmul(dst=pc, stationary=on64, moving=srs[r], accumulate=(r > 0))
        cnt = nl.ndarray((1, E), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cnt, src=pc[0:1, :], engine=nisa.vector_engine)
        # Blocks per expert, ceil(cnt / B), and the first block of each: an exclusive prefix sum over
        # the experts on one partition, in log2(E) shifted adds.
        LB = 7 if B == 128 else 6  # B = 2 ** LB (the tracer takes no int methods)
        ci = nl.ndarray((1, E), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ci, data=cnt, op0=nl.add, operand0=float(B - 1), engine=nisa.vector_engine)
        nbi = nl.ndarray((1, E), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nbi, data=ci, op0=nl.right_shift, operand0=LB)
        nbk = nl.ndarray((1, E), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=nbk, src=nbi, engine=nisa.vector_engine)
        inc = nl.ndarray((1, E), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=inc, src=nbk, engine=nisa.vector_engine)
        sh = 1
        while sh < E:
            prev = nl.ndarray((1, E), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=prev, src=inc, engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=inc[:, sh:E], data1=prev[:, sh:E], data2=prev[:, 0:E - sh], op=nl.add,
                               engine=nisa.vector_engine)
            sh *= 2
        off = nl.ndarray((1, E), dtype=f32, buffer=nl.sbuf)  # exclusive prefix; < 2^12
        nisa.tensor_tensor(dst=off, data1=inc, data2=nbk, op=nl.subtract, engine=nisa.vector_engine)
        # off on every partition, exactly: 64 hi + lo with both parts bf16-exact
        oi = nl.ndarray((1, E), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=oi, src=off, engine=nisa.vector_engine)
        os_ = nl.ndarray((1, E), dtype=i32, buffer=nl.sbuf)
        olob = nl.ndarray((1, E), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=os_, data=oi, op0=nl.bitwise_and, operand0=63)
        nisa.tensor_copy(dst=olob, src=os_, engine=nisa.vector_engine)
        ohib = nl.ndarray((1, E), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=os_, data=oi, op0=nl.right_shift, operand0=6)
        nisa.tensor_copy(dst=ohib, src=os_, engine=nisa.vector_engine)
        pob = nl.ndarray((128, E), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pob, stationary=r64, moving=ohib, accumulate=False)
        nisa.nc_matmul(dst=pob, stationary=r1, moving=olob, accumulate=True)
        offb = nl.ndarray((128, E), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=offb, src=pob, engine=nisa.vector_engine)
        # Each pair's block offset off[e], gathered inside each partition (GpSimd); then slot, tile and
        # tos index, all at once.
        tku = nl.ndarray((128, NT, K), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tku, src=tk, engine=nisa.vector_engine)
        bo = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=bo, data=offb, indices=tku)
        sf = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)  # slot = off B + rank
        nisa.scalar_tensor_tensor(dst=sf, data=bo, op0=nl.multiply, operand0=float(B), op1=nl.add, operand1=rk)
        st = nl.ndarray((128, NT, K), dtype=i32, buffer=nl.sbuf)  # the combine's gather rows
        nisa.tensor_copy(dst=st, src=sf, engine=nisa.vector_engine)
        li = nl.ndarray((128, NT, K), dtype=i32, buffer=nl.sbuf)  # TSB: the slot's lane (slot & 127) and tile (slot >> 7)
        nisa.tensor_scalar(dst=li, data=st, op0=nl.bitwise_and, operand0=127)
        lf = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lf, src=li, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=li, data=st, op0=nl.right_shift, operand0=7)
        tf = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tf, src=li, engine=nisa.vector_engine)
        # Block b's expert: the number of experts whose blocks end at or before b (E past the
        # routing's blocks). incl = off + nblk goes onto the partitions (exactly, 64 hi + lo through
        # two bf16 transposes), each compared with b on the free axis and summed by a matmul.
        # The row is padded to EP = 64-multiple experts (a transpose takes 64 or more columns on trn1)
        # with 16383 = 64 * 255 + 63 (bf16-exact parts, past every block), which counts for no block.
        EP = -(-E // 64) * 64
        incl = nl.ndarray((1, EP), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=incl, value=16383.0)
        nisa.tensor_tensor(dst=incl[:, 0:E], data1=off, data2=nbk, op=nl.add, engine=nisa.vector_engine)
        ii = nl.ndarray((1, EP), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ii, src=incl, engine=nisa.vector_engine)
        is_ = nl.ndarray((1, EP), dtype=i32, buffer=nl.sbuf)
        ilob = nl.ndarray((1, EP), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=is_, data=ii, op0=nl.bitwise_and, operand0=63)
        nisa.tensor_copy(dst=ilob, src=is_, engine=nisa.vector_engine)
        ihib = nl.ndarray((1, EP), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=is_, data=ii, op0=nl.right_shift, operand0=6)
        nisa.tensor_copy(dst=ihib, src=is_, engine=nisa.vector_engine)
        one1 = nl.ndarray((1, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=one1, value=1.0)
        ib = nl.ndarray((128, NB), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ib, pattern=[[1, NB]], offset=0, channel_multiplier=0)  # b on every partition
        ibf = nl.ndarray((128, NB), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ibf, src=ib, engine=nisa.vector_engine)
        cmps = []
        for e0 in range(0, EP, 128):  # chunks of 128 experts on the partitions, the last 64 or more
            ec = min(128, EP - e0)
            pt = nl.ndarray((ec, 2), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pt[:, 0:1], stationary=ihib[:, e0:e0 + ec], moving=one1)  # incl hi, on partitions
            nisa.nc_matmul(dst=pt[:, 1:2], stationary=ilob[:, e0:e0 + ec], moving=one1)
            pts = nl.ndarray((ec, 2), dtype=f32, buffer=nl.sbuf)  # (the vector engine reads one PSUM operand)
            nisa.tensor_copy(dst=pts, src=pt, engine=nisa.vector_engine)
            icl = nl.ndarray((ec, 1), dtype=f32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=icl, data=pts[:, 0:1], op0=nl.multiply, operand0=64.0, op1=nl.add,
                                      operand1=pts[:, 1:2])
            cm = nl.ndarray((ec, NB), dtype=bf16, buffer=nl.sbuf)  # [incl_e <= b]
            nisa.tensor_scalar(dst=cm, data=ibf[0:ec, :], op0=nl.greater_equal, operand0=icl,
                               engine=nisa.vector_engine)
            cmps.append((ec, cm))
        bef = nl.ndarray((1, NB), dtype=f32, buffer=nl.sbuf)
        for b0 in range(0, NB, 512):
            nbc = min(512, NB - b0)
            pbx = nl.ndarray((64, nbc), dtype=f32, buffer=nl.psum)
            for ci in range(len(cmps)):
                ec, cm = cmps[ci]
                nisa.nc_matmul(dst=pbx, stationary=on64[0:ec, :], moving=cm[:, b0:b0 + nbc], accumulate=(ci > 0))
            nisa.tensor_copy(dst=bef[:, b0:b0 + nbc], src=pbx[0:1, :], engine=nisa.vector_engine)
        be = nl.ndarray((1, NB), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=be, src=bef, engine=nisa.vector_engine)

        # 1. ts[lane, tile] = the token in slot tile 128 + lane, C (out of range: skipped) where empty (nsx, an
        # experiment, off: token 0 there; empty lanes' Y rows are never read). TSB: built in PSUM
        # with no HBM tos and no indirect DMA. Group (T, k): v = C - token (1 .. C) as 64 hi + lo (both < 65, bf16-exact);
        # the lane one-hot oa[p, q] = [lane(slot of p) == q] (stationary) times the tile one-hot [p, t] = [tile == t]
        # times hi (or lo) (moving) accumulates into ph (pl) [lane, tile]. Each slot holds one pair or none, so every
        # entry is one exact product or 0, and ts = C - (64 ph + pl). One PSUM tile per accumulation group: NTL in
        # chunks of 512 columns, hi and lo apart.
        tid = nl.ndarray((128, NT), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=tid, pattern=[[128, NT]], offset=0, channel_multiplier=1)  # token T 128 + p
        vtf = nl.ndarray((128, NT), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=vtf, src=tid, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=vtf, data=vtf, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=float(C),
                           engine=nisa.vector_engine)  # C - token, exact
        vti = nl.ndarray((128, NT), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=vti, src=vtf, engine=nisa.vector_engine)
        vhi = nl.ndarray((128, NT), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=vhi, data=vti, op0=nl.right_shift, operand0=6)
        hif = nl.ndarray((128, NT), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=hif, src=vhi, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=vhi, data=vti, op0=nl.bitwise_and, operand0=63)
        lof = nl.ndarray((128, NT), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lof, src=vhi, engine=nisa.vector_engine)
        iq = nl.ndarray((128, 128), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=iq, pattern=[[1, 128]], offset=0, channel_multiplier=0)  # q on every partition
        iqf = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=iqf, src=iq, engine=nisa.vector_engine)
        NCK = -(-NTL // 512)
        phs, pls = [], []
        for ck in range(NCK):
            phs.append(nl.ndarray((128, min(512, NTL - ck * 512)), dtype=f32, buffer=nl.psum))
            pls.append(nl.ndarray((128, min(512, NTL - ck * 512)), dtype=f32, buffer=nl.psum))
        for T in range(NT):
            for k in range(K):
                gi = T * K + k
                oa = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=oa, data=iqf, op0=nl.equal, operand0=lf[:, T, k:k + 1], engine=nisa.vector_engine)
                bh = nl.ndarray((128, NTL), dtype=bf16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=bh, data=ibf[:, 0:NTL], op0=nl.equal, operand0=tf[:, T, k:k + 1], op1=nl.multiply,
                                   operand1=hif[:, T:T + 1], engine=nisa.vector_engine)
                bl = nl.ndarray((128, NTL), dtype=bf16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=bl, data=ibf[:, 0:NTL], op0=nl.equal, operand0=tf[:, T, k:k + 1], op1=nl.multiply,
                                   operand1=lof[:, T:T + 1], engine=nisa.vector_engine)
                for ck in range(NCK):
                    c0, c1 = ck * 512, min(NTL, (ck + 1) * 512)
                    nisa.nc_matmul(dst=phs[ck], stationary=oa, moving=bh[:, c0:c1], accumulate=(gi > 0))
                    nisa.nc_matmul(dst=pls[ck], stationary=oa, moving=bl[:, c0:c1], accumulate=(gi > 0))
        tsf = nl.ndarray((128, NTL), dtype=f32, buffer=nl.sbuf)
        for ck in range(NCK):
            c0, c1 = ck * 512, min(NTL, (ck + 1) * 512)
            hs = nl.ndarray((128, c1 - c0), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=hs, op=nl.copy, data=phs[ck])  # (the vector engine reads one PSUM operand)
            nisa.scalar_tensor_tensor(dst=tsf[:, c0:c1], data=hs, op0=nl.multiply, operand0=64.0, op1=nl.add,
                                      operand1=pls[ck])
        if nsx:  # empty (sum 0) -> token 0: (C - v) [v != 0]
            nz = nl.ndarray((128, NTL), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=nz, data=tsf, op0=nl.not_equal, operand0=0.0, engine=nisa.vector_engine)
            nisa.tensor_scalar(dst=tsf, data=tsf, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=float(C),
                               engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=tsf, data1=tsf, data2=nz, op=nl.multiply, engine=nisa.vector_engine)
        else:
            nisa.tensor_scalar(dst=tsf, data=tsf, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=float(C),
                               engine=nisa.vector_engine)
        ts = nl.ndarray((128, NTL), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ts, src=tsf, engine=nisa.vector_engine)

        # 2. Lane tiles, software-pipelined in four stages so that every engine has independent work
        # from several tiles in its in-order instruction stream, and every hand-off between engines
        # has work of other tiles between producer and consumer: step i issues the loads of tile
        # i + PF (expert blobs, x rows); then, interleaved in eight rounds, one 512-column chunk of
        # the down matmuls and scaled drains of tile i - 3 (stage C), four x transposes and their
        # gate_up dequantization of tile i (stage A), four gate_up matmuls of tile i - 1 (stage B),
        # and spread over rounds 0, 1, 2, 3 and 5 the fold and activation of tile i - 2 (stage F:
        # vector copy of its gate_up sums, fold matmuls, clamp, SiLU, product); then the store of
        # tile i - 3. Rings hold each stage's inputs until the last stage that reads them: expert
        # buffers PF + 4 tiles (i + PF .. i - 3), x rows PF + 1, x^T, dequantized tiles and activations 2. The x
        # rows ring is zeroed once, so a skipped (empty) lane holds finite values; the activation
        # stationaries are zero outside their block's own columns.
        # Y in slot order. With two programs each writes the rows of its own lane tiles and the combine
        # reads any row, so Y is in HBM both cores share, and a core barrier (nisa.core_barrier: "two
        # NeuronCores both need to write to disjoint portions of a shared HBM tensor ... and they both
        # need to consume the tensor after both cores have finished", nki/isa/_lnc.py) separates the
        # stores of stage 2 from the gathers of stage 3.
        Y = nl.ndarray((NTL * 128, H), dtype=bf16, buffer=nl.private_hbm if npg == 1 else nl.shared_hbm)
        PF = 2 if G == 1 else 1  # loads run PF tiles ahead (SBUF: one block of 128 lanes per tile, or two)
        NW = PF + 4
        NX = PF + 1
        wr, xr, xtr, ar, tbr, wqr = [], [], [], [], [], []
        for r in range(NW):
            row = []
            for g in range(G):
                row.append(nl.ndarray((128, F), dtype=nl.uint8, buffer=nl.sbuf))
            wr.append(row)
        for r in range(NX):
            row2 = []
            for g in range(G):
                row2.append(nl.ndarray((128, 2, CT * 4), dtype=nl.uint8, buffer=nl.sbuf))
            tbr.append(row2)  # dq: the gate and the up rows' tile scales, on every partition
            xb0 = nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf)
            nisa.memset(dst=xb0, value=0.0)
            xr.append(xb0)
        for r in range(2):
            xtr.append(nl.ndarray((128, CT, 128), dtype=bf16, buffer=nl.sbuf))
            row2 = []
            for g in range(G if dq else 0):  # the per-row path never reads them: no SBUF for it
                row2.append(nl.ndarray((128, CT, 128), dtype=bf16, buffer=nl.sbuf))
            wqr.append(row2)  # dq: dequantized gate_up tiles [h, c, o]
            row = []
            for g in range(G):
                a0 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
                nisa.memset(dst=a0, value=0.0)
                row.append(a0)
            ar.append(row)

        scr, frr, tsr = [], [], []  # PC: each block's chunk scales on its lanes' partitions, its column factors
        # as row g; stage C. selg [G, 128]: row g ones over block g's lanes (broadcasts row g there). fdr: no
        # chunk scales; frr rows 3 g + p hold piece p of block g's T = s f, selg is sel3 [3 G, 128] (from dsc),
        # and tsr is a ring of T in SBUF (fp32), one chunk each.
        if PC and fdr:
            for r in range(2):
                frr.append(nl.ndarray((3 * G, H), dtype=bf16, buffer=nl.sbuf))
                tsr.append(nl.ndarray((128, 512), dtype=f32, buffer=nl.sbuf))
            selg = nl.ndarray((3 * G, 128), dtype=bf16, buffer=nl.sbuf)
            nisa.dma_copy(dst=selg, src=dsc)
        elif PC:
            for r in range(2):
                scr.append(nl.ndarray((128, CT), dtype=f32, buffer=nl.sbuf))
                frr.append(nl.ndarray((G, H), dtype=bf16, buffer=nl.sbuf))
            sgi = nl.ndarray((G, 128), dtype=i32, buffer=nl.sbuf)
            nisa.iota(dst=sgi, pattern=[[1, 128]], offset=0, channel_multiplier=-B)  # lane - B g
            sgf = nl.ndarray((G, 128), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=sgf, src=sgi, engine=nisa.vector_engine)
            sga = nl.ndarray((G, 128), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=sga, data=sgf, op0=nl.greater_equal, operand0=0.0, engine=nisa.vector_engine)
            sgb = nl.ndarray((G, 128), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=sgb, data=sgf, op0=nl.less, operand0=float(B), engine=nisa.vector_engine)
            selg = nl.ndarray((G, 128), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=selg, data1=sga, data2=sgb, op=nl.multiply, engine=nisa.vector_engine)
        ybr = []  # nyb > 0: a ring of stage C's bf16 down outputs (measured: 3 is faster than one allocation
        for r in range(nyb):  # per step, 2 slower; see NYB)
            ybr.append(nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf))
        accr = []  # each block's gate_up sums of tile t at accr[t % 2] (PSUM for dq), stage B to F
        for r in range(2):
            row = []
            for g in range(G):
                if dq:
                    row.append(nl.ndarray((128, B), dtype=f32, buffer=nl.psum))
                else:
                    row.append(nl.ndarray((128, B), dtype=f32, buffer=nl.sbuf))
            accr.append(row)
        if RK:  # RK
            accr = []  # RK
            for r in range(2):  # RK
                row = []  # RK
                for g in range(G):  # RK
                    row.append(nl.ndarray((128, B), dtype=f32, buffer=nl.psum))  # RK
                accr.append(row)  # RK
        NCH = 2 * (DW // 512)  # 512-column chunks of the down output
        NR = max(CT // 4, NCH)
        r1, r2, r3, r5 = min(1, NR - 1), min(2, NR - 1), min(3, NR - 1), min(5, NR - 1)  # stage F's rounds
        S = dict(NTL=NTL, PF=PF, G=G, NB=NB, F=F, NW=NW, NX=NX, be=be, blob=blob, wr=wr, tbr=tbr, o_sg=o_sg,
                 o_sd=o_sd, CT=CT, x=x, H=H, xr=xr, ts=ts, PC=PC, scr=scr, frr=frr, dsc=dsc, dfr=dfr, B=B, f32=f32,
                 bf16=bf16, fp8=fp8, MX=MX, NR=NR, NCH=NCH, DW=DW, r1=r1, r2=r2, r3=r3, r5=r5, ar=ar,
                 xtr=xtr, wqr=wqr, fg=fg, fu=fu, act=act, lim=lim, selg=selg if PC else None, asp=asp, dq=dq, Y=Y,
                 bfirst=ord_ == 1, ybr=ybr, DG=DG, fdr=fdr, tsr=tsr)
        T1 = NTL - skp * ((NTL - (NTL + 1) // 2) // skp) if skp > 0 else NTL  # tiles that always run
        # This program's share of the tiles that always run: all of them (one program), else the first or
        # the second half (every tile costs the same, empty ones included).
        t_lo, t_hi = (0, T1) if npg == 1 else ((0, (T1 + 1) // 2) if pid == 0 else ((T1 + 1) // 2, T1))
        for i in range(t_lo - PF, t_hi + 3):
            _tile_step(S, accr, i, t_lo, t_hi)
        if T1 < NTL:
            # The tiles past T1 in segments of skp, each its own pipeline (prologue to epilogue) in a device
            # loop of trip count f[s] = [first tile of segment s < U / G], U the routing's block count
            # (the last entry of the inclusive prefix sum `inc`; blocks are laid out by expert, the rest
            # have expert E): segments wholly past the routing's blocks do not run. Device loops with
            # static addresses run on trn1 ("MoE that reads each selected expert once per call"); a PSUM
            # tile referenced in two device-loop regions fails to compile ([NCC_IBIR092] Live-in/Live-out
            # MemoryLocation ... not allocated to MemoryType: DRAM), so each region gets its
            # own gate_up accumulators.
            KS = (NTL - T1) // skp
            # With two programs every program runs every segment's device loop on its own half of the
            # segment's tiles (s_off .. s_off + s_n of it) with the SAME trip count, read at the segment's
            # first tile: the LNC=2 compile requires both cores' programs to have the same basic blocks
            # ("[NCC_IXGM002] Expected function ... to have X basic blocks, but on core 1 it has Y",
            # measured when the segments alternated between the programs), and trip counts that
            # differ between the cores (each read at its own half) hung the device: execution timeout,
            # GpSimd waiting (C=4096, skp 20, on trn2).
            s_off, s_n = (0, skp) if npg == 1 else ((0, (skp + 1) // 2) if pid == 0 else ((skp + 1) // 2, skp // 2))
            kio = nl.ndarray((1, KS), dtype=i32, buffer=nl.sbuf)
            nisa.iota(dst=kio, pattern=[[skp * G, KS]], offset=T1 * G, channel_multiplier=0)
            kif = nl.ndarray((1, KS), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=kif, src=kio, engine=nisa.vector_engine)
            fk = nl.ndarray((1, KS), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=fk, data=kif, op0=nl.less, operand0=inc[:, E - 1:E], engine=nisa.vector_engine)
            fli = nl.ndarray((1, KS), dtype=i32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=fli, src=fk, engine=nisa.vector_engine)
            for k in range(KS):
                lo = T1 + k * skp + s_off
                rk = nisa.register_alloc()
                nisa.register_load(rk, fli.ap(pattern=[[KS, 1], [1, 1]], offset=k))

                def segment(it, lo=lo):
                    _segment(S, lo, lo + s_n, it)

                nl.fori_loop(0, rk, segment)

        # 3. Combine: out[t] = sum_k w[t, k] Y[slot[t, k]], fp32 on the tensor engine; with two programs
        # each takes half of the token tiles, after both have stored their Y rows.
        out = nl.ndarray((C, H), dtype=x.dtype, buffer=nl.shared_hbm)
        # bng (an experiment): the engine the barriers run on, 0 GpSimd (the default), 1 sync, 2 vector
        beng = (nisa.engine.gpsimd, nisa.engine.sync, nisa.engine.vector)[bng]
        if npg > 1:
            nisa.core_barrier(data=Y, cores=(0, 1), engine=beng)
        c_lo, c_hi = (0, NT) if npg == 1 else ((0, (NT + 1) // 2) if pid == 0 else ((NT + 1) // 2, NT))
        for t in range(c_lo, c_hi):
            wv = nl.ndarray((128, K), dtype=f32, buffer=nl.sbuf)
            nisa.dma_copy(dst=wv, src=wts[t * 128:(t + 1) * 128, :])
            yk, dg = [], []
            for j in range(K):
                yj = nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf)
                nisa.dma_copy(dst=yj, src=Y.ap(pattern=[[H, 128], [1, H]], offset=0,
                                               vector_offset=st.ap(pattern=[[NT * K, 128], [1, 1]], offset=t * K + j),
                                               indirect_dim=0), dge_mode=DG)
                yk.append(yj)
                dj = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=dj, data=ident, op0=nl.multiply, operand0=wv[:, j:j + 1],
                                   engine=nisa.vector_engine)
                dg.append(dj)
            ob = nl.ndarray((128, H), dtype=x.dtype, buffer=nl.sbuf)
            for ch in range(H // 512):
                po = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
                for j in range(K):
                    nisa.nc_matmul(dst=po, stationary=dg[j], moving=yk[j][:, ch * 512:(ch + 1) * 512],
                                   accumulate=(j > 0))
                nisa.activation(dst=ob[:, ch * 512:(ch + 1) * 512], op=nl.copy, data=po)
            nisa.dma_copy(dst=out[t * 128:(t + 1) * 128, :], src=ob)
        if npg > 1:
            # LNC split: every program has written its share of the output before either program's kernel ends,
            # so that whatever runs next on either physical core reads the whole output (nisa.core_barrier on the
            # shared HBM output, nki/isa/_lnc.py).
            nisa.core_barrier(data=out, cores=(0, 1), engine=beng)
        return out

    def _segment(S, lo, hi, it):
        """Lane tiles lo .. hi - 1 as a pipeline of their own (_tile_step), with gate_up accumulators of
        its own: the body of a device loop (it: its induction register, unused)."""
        accs = []
        for r in range(2):
            row = []
            for g in range(S["G"]):
                row.append(nl.ndarray((128, S["B"]), dtype=S["f32"], buffer=nl.psum if S["dq"] else nl.sbuf))
            accs.append(row)
        if RK:  # RK
            accs = []  # RK
            for r in range(2):  # RK
                row = []  # RK
                for g in range(S["G"]):  # RK
                    row.append(nl.ndarray((128, S["B"]), dtype=S["f32"], buffer=nl.psum))  # RK
                accs.append(row)  # RK
        for i in range(lo - S["PF"], hi + 3):
            _tile_step(S, accs, i, lo, hi)

    def _stage_b(S, acc, sgs, tb, c4):
        """Stage B of tile tb, round c4: four gate_up tiles of each block, accumulated in PSUM (dq) or
        each tile's partial scaled per row and summed in fp32 on the vector engine. With ord_ 1 the
        rounds issue it before stage C's down matmuls, so that the tensor engine's in-order stream does
        not hold the partials the vector engine waits for behind matmuls that wait on the vector engine
        (stage C reads stage F's activations): C=4096 9.725 -> 9.551 ms on GLM-5.3-Flash's real experts
        (trn1), the output the same bit for bit."""
        G = S["G"]
        B = S["B"]
        f32 = S["f32"]
        fp8 = S["fp8"]
        NW = S["NW"]
        wr = S["wr"]
        wqr = S["wqr"]
        xtr = S["xtr"]
        if RK:  # one scale per row over K (the RK form, off): the fp8 tiles accumulated in PSUM  # RK
            for g in range(G):  # RK
                wf = wr[tb % NW][g].view(fp8)  # RK
                for j in range(4):  # RK
                    c = c4 * 4 + j  # RK
                    nisa.nc_matmul(dst=acc[g], stationary=wf[:, c * 128:(c + 1) * 128],  # RK
                                   moving=xtr[tb % 2][:, c, g * B:(g + 1) * B], accumulate=(c > 0))  # RK
            return  # RK
        if S["dq"]:  # four dequantized tiles
            for g in range(G):
                for j in range(4):
                    c = c4 * 4 + j
                    nisa.nc_matmul(dst=acc[g], stationary=wqr[tb % 2][g][:, c, :],
                                   moving=xtr[tb % 2][:, c, g * B:(g + 1) * B], accumulate=(c > 0))
            return
        for g in range(G):  # four tiles per block, scaled
            wf = wr[tb % NW][g].view(fp8)
            sg = sgs[g]  # [128 o, CT] tile scales
            pg = nl.ndarray((128, 4, B), dtype=f32, buffer=nl.psum)
            for j in range(4):  # accumulate=False: the default (None) accumulates onto a reused tile
                c = c4 * 4 + j
                nisa.nc_matmul(dst=pg[:, j, :], stationary=wf[:, c * 128:(c + 1) * 128],
                               moving=xtr[tb % 2][:, c, g * B:(g + 1) * B], accumulate=False)
            for j in range(4):
                c = c4 * 4 + j
                if c == 0:
                    nisa.tensor_scalar(dst=acc[g], data=pg[:, j, :], op0=nl.multiply,
                                       operand0=sg[:, 0:1], engine=nisa.vector_engine)
                else:
                    nisa.scalar_tensor_tensor(dst=acc[g], data=pg[:, j, :], op0=nl.multiply,
                                              operand0=sg[:, c:c + 1], op1=nl.add, operand1=acc[g])

    def _tile_step(S, accr, i, lo, hi):
        """Step i of the lane-tile pipeline over tiles lo .. hi - 1 (moe_prefill_tsb_kernel, part 2):
        the loads of tile i + PF, then the stages of tiles i (A), i - 1 (B), i - 2 (F) and i - 3 (C),
        and the store of tile i - 3, each only for a tile in the range. S: the kernel's tensors and
        constants by name; accr: the gate_up accumulators' ring. A module-level helper, so that the
        same step can be issued inside and outside a device loop."""
        NTL = S["NTL"]
        PF = S["PF"]
        G = S["G"]
        NB = S["NB"]
        F = S["F"]
        NW = S["NW"]
        NX = S["NX"]
        be = S["be"]
        blob = S["blob"]
        wr = S["wr"]
        tbr = S["tbr"]
        o_sg = S["o_sg"]
        o_sd = S["o_sd"]
        CT = S["CT"]
        x = S["x"]
        H = S["H"]
        xr = S["xr"]
        ts = S["ts"]
        PC = S["PC"]
        scr = S["scr"]
        frr = S["frr"]
        dsc = S["dsc"]
        dfr = S["dfr"]
        B = S["B"]
        f32 = S["f32"]
        bf16 = S["bf16"]
        fp8 = S["fp8"]
        MX = S["MX"]
        NR = S["NR"]
        NCH = S["NCH"]
        DW = S["DW"]
        r1 = S["r1"]
        r2 = S["r2"]
        r3 = S["r3"]
        r5 = S["r5"]
        ar = S["ar"]
        xtr = S["xtr"]
        wqr = S["wqr"]
        fg = S["fg"]
        fu = S["fu"]
        act = S["act"]
        lim = S["lim"]
        selg = S["selg"]
        asp = S["asp"]
        dq = S["dq"]
        Y = S["Y"]
        DG = S["DG"]
        if lo <= i + PF < hi:  # loads of tile i + PF
            tl = i + PF
            for g in range(G):
                e = be.ap(pattern=[[NB, 1], [1, 1]], offset=tl * G + g)
                nisa.dma_copy(dst=wr[tl % NW][g], src=blob.ap(pattern=[[F, 128], [1, F]], offset=0,
                                                               scalar_offset=e, indirect_dim=0),
                              oob_mode=oob_mode.skip, dge_mode=DG)
                if dq:  # rows 0 (gate) and 64 (up) of the tile scales, broadcast to every partition
                    nisa.dma_copy(dst=tbr[tl % NX][g], src=blob.ap(pattern=[[0, 128], [64 * F, 2], [1, CT * 4]],
                                                                   offset=o_sg, scalar_offset=e, indirect_dim=0),
                                  oob_mode=oob_mode.skip, dge_mode=DG)
            nisa.dma_copy(dst=xr[tl % NX], src=x.ap(pattern=[[H, 128], [1, H]], offset=0,
                                                   vector_offset=ts.ap(pattern=[[NTL, 128], [1, 1]], offset=tl),
                                                   indirect_dim=0), oob_mode=oob_mode.skip, dge_mode=DG)
        if i < lo:
            return
        fdr = S["fdr"]
        if PC and fdr and lo <= i - 2 < hi:  # fdr: tile i - 2's three pieces of T per block, rows 3 g + p
            for g in range(G):
                e = be.ap(pattern=[[NB, 1], [1, 1]], offset=(i - 2) * G + g)
                nisa.dma_copy(dst=frr[(i - 2) % 2][3 * g:3 * g + 3, :],
                              src=dfr.ap(pattern=[[H, 3], [1, H]], offset=0, scalar_offset=e, indirect_dim=0),
                              oob_mode=oob_mode.skip, dge_mode=DG)
        elif PC and lo <= i - 2 < hi:  # tile i - 2's down scales (stage C next step)
            for g in range(G):
                e = be.ap(pattern=[[NB, 1], [1, 1]], offset=(i - 2) * G + g)
                nisa.dma_copy(dst=scr[(i - 2) % 2][g * B:(g + 1) * B, :],
                              src=dsc.ap(pattern=[[0, B], [1, CT]], offset=0, scalar_offset=e, indirect_dim=0),
                              oob_mode=oob_mode.skip, dge_mode=DG)
                nisa.dma_copy(dst=frr[(i - 2) % 2][g:g + 1, :],
                              src=dfr.ap(pattern=[[H, 1], [1, H]], offset=0, scalar_offset=e, indirect_dim=0),
                              oob_mode=oob_mode.skip, dge_mode=DG)
        ta, tb, tf, tc = i, i - 1, i - 2, i - 3
        doA, doB, doF, doC = i < hi, lo <= tb < hi, lo <= tf < hi, lo <= tc < hi
        yb = None
        if doC:  # stage C, tile i - 3: every partition of a blob holds each chunk's down scale
            if fdr:
                sdr = None  # T carries the chunk scales
            elif PC:
                sdr = scr[tc % 2]
            else:
                sdr = nl.ndarray((128, CT), dtype=f32, buffer=nl.sbuf)
                for g in range(G):
                    nisa.tensor_copy(dst=sdr[g * B:(g + 1) * B, :],
                                     src=wr[tc % NW][g][g * B:(g + 1) * B, o_sd:F].view(f32),
                                     engine=nisa.vector_engine)
            if S["ybr"]:
                yb = S["ybr"][tc % len(S["ybr"])]
            else:
                yb = nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf)
        gbs, pfs, gcs, sls = [], [], [], []
        if doF:  # stage F, tile i - 2: its gate_up sums rounded to bf16 (vector, first in this step)
            for g in range(G):
                gb = nl.ndarray((128, B), dtype=bf16, buffer=nl.sbuf)
                if RK:  # RK
                    nisa.tensor_scalar(dst=accr[tf % 2][g], data=accr[tf % 2][g], op0=nl.multiply,  # RK
                                       operand0=wr[tf % NW][g][:, o_sg:o_sg + 4].view(f32), engine=nisa.vector_engine)  # RK
                nisa.tensor_copy(dst=gb, src=accr[tf % 2][g], engine=nisa.vector_engine)
                gbs.append(gb)
        acc = accr[tb % 2]
        sgs = []  # per-row tile scales of tile i - 1's blocks, fp32 [128 o, CT] (stage B, not dq)
        if doB and not dq:
            for g in range(G):
                if MX:
                    sgf = nl.ndarray((128, CT), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=sgf, src=wr[tb % NW][g][:, o_sg:o_sd].view(bf16), engine=nisa.vector_engine)
                    sgs.append(sgf)
                else:
                    sgs.append(wr[tb % NW][g][:, o_sg:o_sd].view(f32))
        doCd, doC = doC, doC and not DRT  # DRT: the two halves' down chunks paired below, in row tiles  # DRT
        for c4 in range(NR):
            py = None
            if doB and c4 < CT // 4 and S["bfirst"]:  # ord_ 1: stage B first in the round (see _stage_b)
                _stage_b(S, acc, sgs, tb, c4)
            if doC and c4 < NCH:  # stage C, tile i - 3: one chunk's down matmuls
                hf, c5 = c4 // (DW // 512), c4 % (DW // 512)
                py = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
                for g in range(G):
                    nisa.nc_matmul(dst=py, stationary=ar[tc % 2][g][hf * 64:(hf + 1) * 64, :],
                                   moving=wr[tc % NW][g][hf * 64:(hf + 1) * 64,
                                                         H + c5 * 512:H + (c5 + 1) * 512].view(fp8),
                                   accumulate=(g > 0))
                if PC:  # each block's column factors broadcast onto its lanes (0/1 selector: exact; fdr:
                    # the sum of the block's three pieces, exact in fp32 in any order: split3)
                    pfb = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
                    nisa.nc_matmul(dst=pfb, stationary=selg, moving=frr[tc % 2][:, hf * DW + c5 * 512:
                                                                                 hf * DW + (c5 + 1) * 512])
            if doA and c4 < CT // 4:  # stage A, tile i: four transposes, one drain, four tiles dequantized
                # A tensor-engine transpose writes fp32 PSUM on gen2 (trn1) but must write its input's
                # dtype from gen3 on: "gen3: transpose dst must match input dtype" (nki/isa/
                # _validation.py, matmul dst checks; trn2 refused fp32 here). The values are x's
                # bf16 either way, copied into the bf16 xtr below.
                px = nl.ndarray((128, 4, 128), dtype=f32 if nisa.get_nc_version() == nisa.nc_version.gen2
                                else nl.bfloat16, buffer=nl.psum)
                for j in range(4):
                    c = c4 * 4 + j
                    nisa.nc_transpose(dst=px[:, j, :], data=xr[ta % NX][:, c * 128:(c + 1) * 128],
                                      engine=nisa.tensor_engine)
                nisa.activation(dst=xtr[ta % 2][:, c4 * 4:(c4 + 1) * 4, :], op=nl.copy, data=px)
                if dq:  # [h, c, (gate | up) o]: each 64-column half times its scale, a per-partition
                    # scalar (the scale on every partition; the vector engine runs a tensor-scalar op
                    # at twice the rate of a tensor-tensor op on two SBUF operands, measured)
                    # asp of them on the scalar engine, which otherwise only drains PSUM tiles.
                    for g in range(G):
                        for j in range(4):
                            c = c4 * 4 + j
                            for hh in range(2):
                                wdst = wqr[ta % 2][g][:, c, hh * 64:(hh + 1) * 64]
                                wsrc = wr[ta % NW][g][:, c * 128 + hh * 64:c * 128 + (hh + 1) * 64].view(fp8)
                                wsc = tbr[ta % NX][g].ap(pattern=[[2 * CT, 128], [1, 1]], offset=hh * CT + c, dtype=f32)
                                on_act = False
                                for q in range(asp):
                                    if ACT_ORDER[q] == j * 2 + hh:
                                        on_act = True
                                if on_act:
                                    nisa.activation(dst=wdst, op=nl.copy, data=wsrc, scale=wsc)
                                else:
                                    nisa.tensor_scalar(dst=wdst, data=wsrc, op0=nl.multiply, operand0=wsc,
                                                       engine=nisa.vector_engine)
            if doF and c4 == r1:  # stage F: the up rows folded onto the gate rows (both partition halves)
                for g in range(G):
                    pf = nl.ndarray((128, 2, B), dtype=f32, buffer=nl.psum)
                    nisa.nc_matmul(dst=pf[:, 0, :], stationary=fg, moving=gbs[g])
                    nisa.nc_matmul(dst=pf[:, 1, :], stationary=fu, moving=gbs[g])
                    pfs.append(pf)
            if doB and c4 < CT // 4 and not S["bfirst"]:
                _stage_b(S, acc, sgs, tb, c4)
            if doF and c4 == r2:  # stage F: gate (clamped) to SBUF, SiLU on the scalar engine
                for g in range(G):
                    gc = nl.ndarray((128, B), dtype=f32, buffer=nl.sbuf)
                    if act == 1:
                        nisa.tensor_scalar(dst=gc, data=pfs[g][:, 0, :], op0=nl.minimum, operand0=lim,
                                           engine=nisa.vector_engine)
                    else:
                        nisa.tensor_copy(dst=gc, src=pfs[g][:, 0, :], engine=nisa.vector_engine)
                    uc = nl.ndarray((128, B), dtype=f32, buffer=nl.sbuf)
                    if act == 1:
                        nisa.tensor_scalar(dst=uc, data=pfs[g][:, 1, :], op0=nl.minimum, operand0=lim, op1=nl.maximum,
                                           operand1=-lim, engine=nisa.vector_engine)
                    else:
                        nisa.tensor_copy(dst=uc, src=pfs[g][:, 1, :], engine=nisa.vector_engine)
                    gcs.append((gc, uc))
            if doCd and DRT and c4 < NCH // 2:  # stage C, tile i - 3: chunk c4 of both halves (64-row contractions)  # DRT
                pys = [nl.ndarray((128, 512), dtype=f32, buffer=nl.psum), nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)]  # DRT
                for g in range(G):  # the halves alternate, so consecutive matmuls sit in different row tiles  # DRT
                    for hf in range(2):  # DRT
                        nisa.nc_matmul(dst=pys[hf], stationary=ar[tc % 2][g][hf * 64:(hf + 1) * 64, :],  # DRT
                                       moving=wr[tc % NW][g][hf * 64:(hf + 1) * 64, H + c4 * 512:H + (c4 + 1) * 512].view(fp8),  # DRT
                                       accumulate=(g > 0), tile_position=(hf * 64, 0), tile_size=(64, 128))  # DRT
                for hf in range(2):  # DVD 8's drains  # DRT
                    col = hf * DW + c4 * 512  # DRT
                    nisa.tensor_tensor(dst=yb[:, col:col + 512].reshape((128, 4, 128)), data1=pys[hf].reshape((128, 4, 128)),  # DRT
                                       data2=sdr.ap(pattern=[[CT, 128], [2, 4], [0, 128]], offset=8 * c4 + hf),  # DRT
                                       op=nl.multiply, engine=nisa.vector_engine)  # DRT
            if py is not None and DVD and (c4 * DVD) % NCH < DVD:  # DVD
                col = hf * DW + c5 * 512  # DVD
                nisa.tensor_tensor(dst=yb[:, col:col + 512].reshape((128, 4, 128)), data1=py.reshape((128, 4, 128)),  # DVD
                                   data2=sdr.ap(pattern=[[CT, 128], [2, 4], [0, 128]], offset=8 * c5 + hf),  # DVD
                                   op=nl.multiply, engine=nisa.vector_engine)  # DVD
                py = None  # DVD
            if py is not None and fdr:  # FDR: stage C fused (MDR): T = s f broadcast onto the lanes (pfb), then one rounding
                col = hf * DW + c5 * 512  # FDR
                tsb = S["tsr"][c4 % 2]  # FDR
                nisa.activation(dst=tsb, op=nl.copy, data=pfb)  # FDR
                if fdr == 2:  # FDR: the positive control (tests only): py rounded to bf16 first, then times T
                    nisa.activation(dst=yb[:, col:col + 512], op=nl.copy, data=py)  # FDR
                    nisa.tensor_tensor(dst=yb[:, col:col + 512], data1=yb[:, col:col + 512], data2=tsb,  # FDR
                                       op=nl.multiply, engine=nisa.vector_engine)  # FDR
                else:  # FDR
                    nisa.tensor_tensor(dst=yb[:, col:col + 512], data1=py, data2=tsb, op=nl.multiply,  # FDR
                                       engine=nisa.vector_engine)  # FDR
                py = None  # FDR
            if py is not None:  # stage C: that chunk's drains, scaled per 128 columns
                for j in range(4):
                    cc = c5 * 4 + j  # chunk c' of this half: output columns hf * DW + cc * 128
                    col = hf * DW + cc * 128
                    nisa.activation(dst=yb[:, col:col + 128], op=nl.copy, data=py[:, j * 128:(j + 1) * 128],
                                    scale=sdr[:, 2 * cc + hf:2 * cc + hf + 1])
                if PC:  # times the column factors: a power of two, so bf16(y s_c) f = bf16(y s_c f)
                    col = hf * DW + c5 * 512
                    nisa.tensor_tensor(dst=yb[:, col:col + 512], data1=yb[:, col:col + 512], data2=pfb,
                                       op=nl.multiply, engine=nisa.vector_engine)
            if doF and c4 == r3:
                for g in range(G):
                    sl = nl.ndarray((128, B), dtype=f32, buffer=nl.sbuf)
                    nisa.activation(dst=sl, op=nl.silu, data=gcs[g][0])
                    sls.append(sl)
            if doF and c4 == r5:  # stage F: a = silu(gate) up, bf16, into the block's own columns
                for g in range(G):
                    nisa.tensor_tensor(dst=ar[tf % 2][g][:, g * B:(g + 1) * B], data1=sls[g], data2=gcs[g][1],
                                       op=nl.multiply, engine=nisa.vector_engine)
        if yb is not None:  # Y rows in slot order: a static DMA (every lane, empty or not)
            nisa.dma_copy(dst=Y[tc * 128:(tc + 1) * 128, :], src=yb)

else:
    moe_prefill_tsb_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source and the moe_prefill constants it reads (LNL's cache key ignores NKI source)."""
    src = open(__file__).read()
    a = src.index("    @nki.jit\n    def moe_prefill_tsb_kernel")
    b = src.index("else:\n    moe_prefill_tsb_kernel = None", a)
    return zlib.crc32(repr((ACT_ORDER, DVD, RK, DRT, P)).encode(), zlib.crc32(src[a:b].encode()))


REV = _kernel_rev()  # at import: the traced caller reads a constant


def kernel():
    """The NKI kernel (raises where the NKI package is missing)."""
    if moe_prefill_tsb_kernel is None:
        raise RuntimeError("the tos-in-SBUF prefill MoE kernel needs the nki package (the Neuron venv)")
    return moe_prefill_tsb_kernel


def kernel_inputs(x, topv, topi, blob, act: int = 0, limit: float = 0.0, B: int | None = None, down=None,
                  ctl: int = 0) -> dict:
    """moe_prefill_mdr.kernel_inputs (the fused drain's arguments), with this module's REV."""
    a = MDR.kernel_inputs(x, topv, topi, blob, act, limit, B=B, down=down, ctl=ctl)
    a.update(rev=REV)
    return a


def moe_prefill_tsb(x, topv, topi, blob, act: int = 0, limit: float = 0.0, down=None) -> torch.Tensor:
    """moe_prefill_mdr() with tos built in SBUF: the same chunk padding, one kernel call on the layer's tiles blob."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    T = x.shape[0]
    pad = -T % P
    if pad:
        x = torch.cat([x, x.new_zeros(pad, x.shape[1])])
        topi = torch.cat([topi, topi.new_zeros(pad, topi.shape[1])])
        topv = torch.cat([topv, topv.new_zeros(pad, topv.shape[1])])
    out = wrap_nki(kernel())[platform.nki_grid()](**kernel_inputs(x, topv, topi, blob, act, limit, down=down))
    return out[:T] if pad else out
