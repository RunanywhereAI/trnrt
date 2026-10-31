"""kernels/dsa_long_pipe.py's pipelined long-context DSA selection with its query tiles split over the two physical cores
of a logical core (trn2 at LNC=2, grid 2).

Why: at grid 2 dsa_long_pipe runs on program 0 alone while program 1 waits at a core barrier (dsa_long_select._lnc2_note:
the two programs used to run the whole kernel on one shared HBM score scratch, and the 4096-page graphs answered wrong
and faulted). In a served 4096-token prefill call on trn2 (one 12-layer piece on rank 0) it was 44.9 of 465 ms, with one
physical core idle throughout.

Here program p runs a contiguous part of the tiles (the first ceil(NT / 2) on program 0) through the same three-stage
pipeline, on a score scratch of its OWN: one shared-HBM tensor holds four regions, (program, tile parity), and each
program's score writes and gathers address its two alone (the gathers' row offsets carry the region's first row). So
the programs share nothing they write but the outputs, whose tiles are disjoint; a core barrier on out and outv holds
both until every tile is written. Each tile runs dsa_long_pipe's instructions on the same data (scores, level 1, the
sub-block sort and gathers, level 2), in a buffer of its program's parity, so grid 2 gives grid 1's bits. Each program's
pipeline is half as long, so its fill and drain are paid on both cores.

This kernel's REV is its own source and dsa_long_pipe.REV_LNC2 (the helpers it calls).
"""

from __future__ import annotations

import zlib

from . import dsa_long_pipe as dlp
from . import dsa_long_select as dls

CH = dls.CH

if dlp.dsa_long_pipe_kernel is not None:
    from .dsa_long_pipe import _dma, _level1_round, _level2, _prologue2, _scores_begin  # the tracer resolves names
    from .dsa_long_pipe import _keys, _round
    from .dsa_long_select import _const, _psum, _sb

    nki, nisa, nl = dlp.nki, dlp.nisa, dlp.nl
    F32, BF16, I32 = nl.float32, nl.bfloat16, nl.int32
    VE = nisa.vector_engine

    def _scores_chunk_at(X, scr, base: int, bm, c: int):
        """dsa_long_pipe._scores_chunk with the score row written into scr's region at element offset base."""
        Hi, NC, sub, Pp = X["Hi"], X["NC"], X["sub"], X["Pp"]
        qs, ws = X["qs"], X["ws"]
        TP, SP = X["psum"]
        x = c % 2
        if c + 1 < NC:
            _prologue2(X, TP, X["thr"], c + 1)
        kt = X["KTall"][:, c * CH:(c + 1) * CH] if X["ktall"] else X["KT"][x]
        acc = X["ACC"][x]
        acc1 = X["ACC1"][x]
        for h in range(Hi):
            sp = SP[(c * Hi + h) % len(SP)]
            nisa.nc_matmul(dst=sp, stationary=qs[:, h, :], moving=kt, accumulate=False)
            nisa.activation(dst=sp, op=nl.relu, data=sp, scale=X["sclt"], bias=X["zero"])
            a = acc1 if h % 2 else acc
            nisa.scalar_tensor_tensor(dst=a, data=sp, op0=nl.multiply, operand0=ws[:, h:h + 1], op1=nl.add,
                                      operand1=a)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=acc1, op=nl.add, engine=VE)
        nisa.tensor_reduce(dst=bm[:, c * (CH // sub):(c + 1) * (CH // sub)], op=nl.maximum,
                           data=acc.reshape((128, CH // sub, sub)), axis=2)
        _dma(scr.ap(pattern=[[Pp, 128], [1, CH]], offset=base + c * CH), acc)

    def _sort_gather_at(X, scr, base: int, qnb, cand, par: int, dep_from):
        """dsa_long_pipe._sort_gather reading scr's region at element offset base: the one-level copy from that offset,
        the two-level gathers at the rows qnb + sub-block (qnb [128, 1]: q NB plus the region's first row)."""
        keep, sub, Pp = X["keep"], X["sub"], X["Pp"]
        V, I = X["V"], X["I"]
        rounds = keep // 8
        if not X["two"]:
            nisa.dma_copy(dst=cand[:, 0:Pp], src=scr.ap(pattern=[[Pp, 128], [1, Pp]], offset=base))
            return
        key = X["key"]
        nisa.tensor_copy(dst=key, src=I.view(I32), engine=VE)  # uint32 -> fp32, exact (< 2^24)
        nisa.tensor_scalar(dst=key, data=key, op0=nl.multiply, operand0=-1.0, engine=VE)
        blk = X["blk2"][par]
        for r in range(rounds):  # dsa_long_select._sort_asc
            _round(V, None, key, r, rounds)
        nisa.tensor_scalar(dst=blk, data=V, op0=nl.multiply, operand0=-1.0, engine=VE)
        rf = X["rowf"]
        nisa.tensor_scalar(dst=rf, data=blk, op0=nl.add, operand0=qnb, engine=VE)
        dep = X["dep"]
        nisa.tensor_scalar(dst=dep, data=dep_from, op0=nl.multiply, operand0=0.0, engine=VE)
        nisa.tensor_scalar(dst=rf, data=rf, op0=nl.add, operand0=dep, engine=VE)
        ro = X["roff"]
        nisa.tensor_copy(dst=ro, src=rf, engine=VE)
        for j in range(keep):
            nisa.dma_copy(dst=cand[:, j * sub:(j + 1) * sub],
                          src=scr.ap(pattern=[[sub, 128], [1, sub]], offset=0,
                                     vector_offset=ro.ap(pattern=[[keep, 128], [1, 1]], offset=j),
                                     indirect_dim=0))

    @nki.jit
    def dsa_long_pipe_s_kernel(qT, w, npool, pk, identb, keep: int, scale: float, sub: int, lsub: int, one: int,
                                    rev: int, per_chunk: int, ktall: int, vorder: int = 0):
        """dsa_long_pipe_kernel's arguments and outputs; at grid 2 each program pipelines its part of the tiles on
        its own score scratch."""
        NT, D, Hi, Q = qT.shape
        Pp = pk.shape[0]
        NC = Pp // CH
        NB = Pp // sub
        out = nl.ndarray((NT, 128, keep), dtype=I32, buffer=nl.shared_hbm)
        outv = nl.ndarray((NT, 128, keep), dtype=F32, buffer=nl.shared_hbm)
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        # the scores, q-major: 128 NB rows of sub per region, regions (program, tile parity)
        scr = nl.ndarray((npg * 2 * 128 * NB, sub), dtype=F32, buffer=nl.shared_hbm)
        half = (NT + 1) // 2
        t_lo, t_hi = (0, NT) if npg == 1 else ((0, half) if pid == 0 else (half, NT))
        NTl = t_hi - t_lo
        r0 = 2 * pid if npg == 2 else 0  # this program's first region
        bases = (r0 * 128 * Pp, (r0 + 1) * 128 * Pp)  # element offsets of its parity 0 / 1 regions
        X = dict(Hi=Hi, D=D, Pp=Pp, NC=NC, NB=NB, two=1 - one, keep=keep, sub=sub, scale=scale, qT=qT, w=w,
                 npool=npool, pk=pk, out=out, outv=outv, lsub=lsub, dyn=0, vorder=vorder, ktall=ktall)
        IB = _sb((128, 128), BF16)
        _dma(IB, identb)
        X["IB"] = IB
        io_i = _sb((128, CH), I32)
        nisa.iota(dst=io_i, pattern=[[1, CH]], offset=0, channel_multiplier=0)
        iota = _sb((128, CH))
        nisa.tensor_copy(dst=iota, src=io_i, engine=VE)
        X["iota"] = iota
        qi0 = _sb((128, 1), I32)
        nisa.iota(dst=qi0, pattern=[[0, 1]], offset=r0 * 128 * NB, channel_multiplier=NB)
        qnb0 = _sb((128, 1))
        nisa.tensor_copy(dst=qnb0, src=qi0, engine=VE)
        qi1 = _sb((128, 1), I32)
        nisa.iota(dst=qi1, pattern=[[0, 1]], offset=(r0 + 1) * 128 * NB, channel_multiplier=NB)
        qnb1 = _sb((128, 1))
        nisa.tensor_copy(dst=qnb1, src=qi1, engine=VE)
        qnbs = (qnb0, qnb1)
        X["qs"] = _sb((D, Hi, 128), BF16)
        X["ws"] = _sb((128, Hi))
        X["npf"] = _sb((128, 1))
        X["thr"] = _sb((128, NC))
        ci_i = _sb((128, NC), I32)
        nisa.iota(dst=ci_i, pattern=[[CH, NC]], offset=0, channel_multiplier=0)
        X["ciota"] = _sb((128, NC))
        nisa.tensor_copy(dst=X["ciota"], src=ci_i, engine=VE)
        X["sclt"] = _const(scale)
        X["zero"] = _const(0.0)
        X["PKr"] = [_sb((128, 4, D), BF16), _sb((128, 4, D), BF16)]
        X["KT"] = [_sb((128, CH), BF16), _sb((128, CH), BF16)]
        X["ACC"] = [_sb((128, CH)), _sb((128, CH))]
        X["ACC1"] = [_sb((128, CH)), _sb((128, CH))]
        bms = [_sb((128, NB)), _sb((128, NB))]
        W2 = keep * sub if not one else Pp
        cands = [_sb((128, W2)), _sb((128, W2))]
        for name in ("V", "V2", "key", "rowf", "lof", "g", "pool", "vm", "srt", "vals"):
            X[name] = _sb((128, keep))
        X["blk2"] = [_sb((128, keep)), _sb((128, keep))]
        X["I"] = _sb((128, keep), nl.uint32)
        for name in ("roff", "hi", "lo", "oi"):
            X[name] = _sb((128, keep), I32)
        X["dep"] = _sb((128, 1))
        X["psum"] = _psum()
        if ktall:
            X["KTall"] = _sb((128, Pp), BF16)
            for c in range(NC):
                _keys(X, X["psum"][0], c, X["KTall"][:, c * CH:(c + 1) * CH])
        rounds = keep // 8
        for u in range(NTl + 2 if NTl else 0):  # u: this program's tile count so far, t = t_lo + u the tile
            t = t_lo + u
            if u < NTl:  # scores(t), with level 1 of tile t - 1 in its chunks
                bm = bms[u % 2]
                _scores_begin(X, t)
                for c in range(NC):
                    _scores_chunk_at(X, scr, bases[u % 2], bm, c)
                    if 1 <= u:
                        for k in range(per_chunk):
                            rr = c * per_chunk + k
                            if rr < rounds:
                                _level1_round(X, bms[(u - 1) % 2], rr)
                if 1 <= u:
                    for rr in range(NC * per_chunk, rounds):
                        _level1_round(X, bms[(u - 1) % 2], rr)
            elif u == NTl:  # the last tile's level 1
                for rr in range(rounds):
                    _level1_round(X, bms[(u - 1) % 2], rr)
            if 1 <= u <= NTl:  # gathers(t - 1), after scores(t)
                src = bms[u % 2] if u < NTl else bms[(u - 1) % 2]
                p = (u - 1) % 2
                _sort_gather_at(X, scr, bases[p], qnbs[p], cands[p], p, src[:, NB - 1:NB])
            if 2 <= u:  # level2(t - 2) while the gathers are in flight
                _level2(X, cands[(u - 2) % 2], (u - 2) % 2, t - 2)
        if npg == 2:  # every tile written, both programs', before either returns
            nisa.core_barrier(data=out, cores=(0, 1))
            nisa.core_barrier(data=outv, cores=(0, 1))
        return out, outv
else:
    dsa_long_pipe_s_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source with dsa_long_pipe.REV_LNC2 (the helpers it calls) and CH: the static
    argument `rev`."""
    src = open(__file__).read()
    a = src.index("if dlp.dsa_long_pipe_kernel is not None:")
    b = src.index("    dsa_long_pipe_s_kernel = None", a)
    # one CRC of the parts, CH (dsa_long_select's, re-bound here and read by this kernel) among them
    return zlib.crc32(repr((src[a:b], dlp.REV_LNC2, CH)).encode())


REV = _kernel_rev()
