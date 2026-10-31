"""The fused DSA decode layer (kernels/dsa_satwf.py) with the attention block's Sinkhorn inside it: the served form.
mhc_pre's comb chain runs alone ~15.5 us a call at 4 rows on trn2; its only reader is this call's mix_in, and this call
waits on three collectives (S2, the a2a, the world sum) with little to do meanwhile.

dsa_satwfc_kernel is dsa_satwf_kernel statement for statement but:
- comb_p is kernels/mhc_front.py's c (the Sinkhorn's input, softmax + eps), not comb;
- comb_i, this program's comb (private HBM, both programs: mix_in reads every row in each), allocated with the buffers;
- after S2's all_reduce, the Sinkhorn's tile from c (_sk_tile: front_body's eps row and column, c by DMA, so
  front_body's cp to the bit), then the half-rounds in ckp pieces (_sk_piece, mhc_split.sinkhorn_body's), issued after
  S2's all_reduce, the all_to_all and ar_world_issue; the piece reaching the end writes comb_i;
- the FFN half's mix_in reads comb_i.
The tile and pieces read no collective's result (no false edge), so the scheduler may place them anywhere before
mix_in; the issue points are where they can hide, not a constraint."""

from __future__ import annotations

import torch

from . import dsa_sat as SAT
from . import dsa_satwf as SATWF
from . import ffn_half as FH
from . import mhc_front as MF
from . import mhc_split as MS
from . import shell as SH

# the Sinkhorn's pieces (1 .. 3: one after each collective's issue)
CKP = 3

try:  # the Neuron venv
    import nki
    import nki.collectives as ncc
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

# -- the kernel source
if nki is not None:
    from .cc_tail import LP, POISON_OUT, POISON_RX, TP, _dma, _tail_body, _tail_weights

    from .dsa_sa import _attend, _rows_i32, _sel2, _snd_write
    from .dsa_sat import _snd_write_hbm
    from .dsa_sel_s2 import _s2_payload
    from .dsa_shell import F32, I32, _sb
    from .cc_world import _dma_static, ar_world_alloc, ar_world_back, ar_world_close, ar_world_issue, ar_world_zero1
    from .dsa_satwf import _cp_groups
    from .ffn_half_shared import ffn_half_body

    def _sk_tile(comb_p, B: int, hc: int, eps: float, skf: int):
        """The attention block's Sinkhorn tile from mhc_front's c [B, hc hc]: skf 1 front_body's padded [B, hc + 1,
        hc + 1], its eps row and column a memset as there, c (fp32) by DMA into the inner hc x hc; skf 0 [B, hc, hc]."""
        if skf:
            cpa = _sb((B, hc + 1, hc + 1))
            nisa.memset(dst=cpa, value=eps)
            _dma_static(dst=cpa[:, 0:hc, 0:hc], src=comb_p.ap(pattern=[[hc * hc, B], [hc, hc], [1, hc]], offset=0))
        else:
            cpa = _sb((B, hc, hc))
            _dma_static(dst=cpa, src=comb_p.ap(pattern=[[hc * hc, B], [hc, hc], [1, hc]], offset=0))
        return cpa

    def _sk_piece(cpa, comb_i, hc: int, iters: int, eps: float, skf: int, ckp: int, j: int):
        """Piece j of ckp of the 2 iters - 1 half-rounds (mhc_split.chunk_bounds' split, in a form the parser takes):
        sinkhorn_body's, on cpa in place; the piece reaching the end writes comb_i. Pieces past the last are empty."""
        n_half = 2 * iters - 1
        step = (n_half + ckp - 1) // ckp
        lo = j * step
        hi = lo + step
        if hi > n_half:
            hi = n_half
        if lo < n_half:
            MS.sinkhorn_body(cpa, comb_i, hc, iters, eps, skf, lo, hi)

    @nki.jit
    def dsa_satwfc_kernel(qT, wb, pkc, ppg, cand, identb, rank, table, pos, q_lat, kc, w_uv_p, o_p, ident, S,
                               post_p, comb_p, fnb, base, fscale, pnorm, bb, bf, fwb, keep: int, A: int, nbits2: int,
                               rbits2: int, world: int, nbits: int, rbits: int, kp: int, ns: int, lkp: int, la: int,
                               scale: float, fp8: int, rev: int, hc: int, iters: int, eps: float, rms_eps: float,
                               sig: int, skf: int, Is: int, K: int, rsc: float, group: int, ring: int, limit: float,
                               rsp: int, o_rt: int, o_gu: int, o_dn: int, o_rb: int, fin: int, dge: int = 0,
                               nr: int = 4, qr: int = 4, pf: int = 2, ts: int = 0, il: int = 1, sb: int = 1,
                               st: int = 1, ss: int = 1, tl: int = 1, fo: int = 1, fq: int = 1, so: int = 1,
                               wu: int = 1, nf: int = 1, ev: int = 1, pn: int = 0, dbg: int = 0, fz: int = 0,
                               wl: int = 0, po: int = 1, ob: int = 0, sh: int = 0, rp: int = 1, pw: int = 0,
                               yb: int = 0, pq: int = 0, cs: int = 0, ckp: int = 3):
        """dsa_satwf_kernel's arguments, comb_p mhc_front's c [T, hc hc] in comb's place, and ckp the Sinkhorn's
        pieces (1 .. 3). Its outputs."""
        B = cand.shape[0]
        _, _, H, R = q_lat.shape
        nh = H // A
        dv = w_uv_p.shape[2]
        Fd = nh * dv
        HID = o_p.shape[1] // (Fd // TP) * 512
        RT, FT, NCH = R // TP, Fd // TP, HID // 512
        Cn = R + LP
        BF = nl.bfloat16
        npg = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
        pid = nl.program_id(axis=0) if npg > 1 else 0
        # every buffer allocated in both programs before anything depends on the program id (so it sits at the same
        # address in both traces)
        gv = nl.ndarray((A, B, keep), dtype=F32, buffer=nl.shared_hbm)
        gc = nl.ndarray((A, B, keep), dtype=F32, buffer=nl.shared_hbm)
        if fz:  # cc_world's fenced shape, a shared send / receive buffer a program (a barrier on private HBM is
            # untested in any form)
            s2s0 = nl.ndarray((A * B, 2 * keep), dtype=F32, buffer=nl.shared_hbm)
            s2s1 = nl.ndarray((A * B, 2 * keep), dtype=F32, buffer=nl.shared_hbm)
            s2d0 = nl.ndarray((A * B, 2 * keep), dtype=F32, buffer=nl.shared_hbm)
            s2d1 = nl.ndarray((A * B, 2 * keep), dtype=F32, buffer=nl.shared_hbm)
            s2src, s2dst = (s2s0, s2d0) if pid == 0 else (s2s1, s2d1)
        else:
            s2src = nl.ndarray((A * B, 2 * keep), dtype=F32, buffer=nl.private_hbm)
            s2dst = nl.ndarray((A * B, 2 * keep), dtype=F32, buffer=nl.private_hbm)
        SND = nl.ndarray((H * B, Cn), dtype=F32, buffer=nl.shared_hbm)  # internal: the all_to_all's send buffer
        RCV0 = nl.ndarray((H * B, Cn), dtype=F32, buffer=nl.shared_hbm)  # internal: each program's receive buffer
        RCV1 = nl.ndarray((H * B, Cn), dtype=F32, buffer=nl.shared_hbm)
        RS = nl.ndarray((B, keep), dtype=I32, buffer=nl.private_hbm)
        Y, S0, S1, Z1 = ar_world_alloc(B, HID, BF)  # the world sum's buffers (cc_world's), internal
        Yp = nl.ndarray((npg, B, HID), dtype=BF, buffer=nl.shared_hbm)  # the FFN half's input: [Y, zeros ..]
        o_out = nl.ndarray((nh, B, R), dtype=BF, buffer=nl.shared_hbm)  # the combine's o
        comb_i = nl.ndarray((B, hc * hc), dtype=F32, buffer=nl.private_hbm)  # this program's attention-block comb
        if not wl:  # wl 0: the tail's weights and identity at entry, held through the section
            wuv_all = nl.ndarray((TP, nh * RT, dv), dtype=BF, buffer=nl.sbuf)
            wo_all = nl.ndarray((TP, NCH * FT, 512), dtype=BF, buffer=nl.sbuf)
            if pid == 0:  # program 0 consumes; dge none, off sel2's queues
                _tail_weights(w_uv_p, o_p, wuv_all, wo_all, nh, RT, dv, NCH, FT)
        ar_world_zero1(S1, B, HID, pid)  # program 1's zeros, early, off program 0's critical path
        if pid != 0:  # Yp's zero slots, program 1, early (ffn_half.decode's padding of a DSA layer's one slot)
            yz = _sb((B, 512), BF)
            nisa.memset(dst=yz, value=0.0)
            for g in range(1, npg):
                for c in range(NCH):
                    _dma_static(dst=Yp.ap(pattern=[[HID, B], [1, 512]], offset=g * B * HID + c * 512), src=yz)
        if not wl:
            idt = nl.ndarray((B, B), dtype=BF, buffer=nl.sbuf)
            _dma(dst=idt, src=ident)
        # sel2
        out = _sel2(qT, wb, pkc, ppg, cand, identb, rank, keep, A, nbits2, rbits2, rev, dge, nr, qr, pf, cs)
        # S2, the context-parallel sum of the lists: program 0 writes gv / gc
        _s2_payload(out, rank, B, keep, A, s2src, pq)
        if fz and npg > 1:  # both programs' payload writes land before the collective reads them
            nisa.core_barrier(data=s2s0, cores=(0, 1))
            nisa.core_barrier(data=s2s1, cores=(0, 1))
        ncc.all_reduce(srcs=[s2src], dsts=[s2dst], replica_group=_cp_groups(world, A), op=nl.add)
        cpa = _sk_tile(comb_p, B, hc, eps, skf)  # the attention block's Sinkhorn: its tile, then piece 0 under S2
        _sk_piece(cpa, comb_i, hc, iters, eps, skf, ckp, 0)
        if fz and npg > 1:  # program 0's sum lands before its DMA reads it
            nisa.core_barrier(data=s2d0, cores=(0, 1))
            nisa.core_barrier(data=s2d1, cores=(0, 1))  # program 1's sum (never read) lands too: one in flight
        if pid == 0:
            nisa.dma_copy(dst=gv.ap(pattern=[[keep, A * B], [1, keep]], offset=0),
                          src=s2dst.ap(pattern=[[2 * keep, A * B], [1, keep]], offset=0))
            nisa.dma_copy(dst=gc.ap(pattern=[[keep, A * B], [1, keep]], offset=0),
                          src=s2dst.ap(pattern=[[2 * keep, A * B], [1, keep]], offset=keep))
        _rows_i32(out, B, keep, RS)
        if npg > 1:
            nisa.core_barrier(data=gv, cores=(0, 1))
            nisa.core_barrier(data=gc, cores=(0, 1))
        # P3 / P3f: the attend with the fold
        if wl:  # wl 1: the tail's weights and identity after the attend, no tile held across the section (an early
            # device run read entry-held tiles wrong after it [inference])
            wuv_all = nl.ndarray((TP, nh * RT, dv), dtype=BF, buffer=nl.sbuf)
            wo_all = nl.ndarray((TP, NCH * FT, 512), dtype=BF, buffer=nl.sbuf)
            if pid == 0:
                _tail_weights(w_uv_p, o_p, wuv_all, wo_all, nh, RT, dv, NCH, FT)
            idt = nl.ndarray((B, B), dtype=BF, buffer=nl.sbuf)
            _dma(dst=idt, src=ident)
        o, lse, srows, sbias, over = _attend(gv, gc, RS, table, pos, rank, q_lat, kc, keep, A, kp, ns, nbits, rbits,
                                             lkp, la, scale, fp8, rev, ts, il, sb, st, ss, tl, fo, fq)
        if ob and npg > 1:  # ob 1: the attend's HBM outputs o / lse land before _snd_write's DMA reads them back
            nisa.core_barrier(data=o, cores=(0, 1))  # (a same-tensor DMA write then read: rows t >= B / 2 read wrong)
            nisa.core_barrier(data=lse, cores=(0, 1))
        # the receive buffers poisoned (each program its own, cc_tail's form), then the send rows
        if rp:  # rp 0: no poison of the receive buffers (an SBUF-sourced write the a2a's write may overtake)
            pzr = _sb((H * B, Cn))
            nisa.memset(dst=pzr, value=POISON_RX)
            _dma(dst=RCV0 if pid == 0 else RCV1, src=pzr)
            if fz and npg > 1:  # the poison lands before the all_to_all writes the receive buffers
                nisa.core_barrier(data=RCV0, cores=(0, 1))
                nisa.core_barrier(data=RCV1, cores=(0, 1))
        split = npg == 2 and B >= 2
        k_lo, k_hi = (0, B) if not split else ((0, (B + 1) // 2) if pid == 0 else ((B + 1) // 2, B))
        rows = []
        if pw:  # pw 1 (served): program 0 writes every row, program 1 none
            if pid == 0:
                for t in range(B):
                    rows.append(t)
        elif split or pid == 0:
            for t in range(k_lo, k_hi):
                rows.append(t)
        if sh or pw:  # sh 1: each program's rows HBM -> HBM into SND (no SBUF-sourced write for the other to consume)
            _snd_write_hbm(o, lse, SND, B, H, R, rows)
        else:
            _snd_write(o, lse, SND, B, H, R, rows)
        if pn == 1:  # a check: NaN in these rows' padding columns r+1 .., which no receiver reads
            pnn = _sb((H, LP - 1))
            nisa.memset(dst=pnn, value=float("nan"))
            for t in rows:
                _dma(dst=SND.ap(pattern=[[B * Cn, H], [1, LP - 1]], offset=t * Cn + R + 1), src=pnn)
        if pn == 2:  # its positive control: NaN in row 0's lse column r (live), every head; row 0 must move, no other
            pnl = _sb((H, 1))
            nisa.memset(dst=pnl, value=float("nan"))
            for t in rows:
                if t == 0:
                    _dma(dst=SND.ap(pattern=[[B * Cn, H], [1, 1]], offset=R), src=pnl)
        if npg > 1:
            nisa.core_barrier(data=SND, cores=(0, 1))
        ncc.all_to_all(srcs=[SND], dsts=[RCV0 if pid == 0 else RCV1], replica_group=_cp_groups(world, A),
                       collective_dim=0)
        _sk_piece(cpa, comb_i, hc, iters, eps, skf, ckp, 1)  # piece 1 under the all_to_all
        if fz and npg > 1:  # the received rows land before the tail's DMA reads them
            nisa.core_barrier(data=RCV0, cores=(0, 1))
            nisa.core_barrier(data=RCV1, cores=(0, 1))
        if pid == 0:  # the tail, program 0 alone (only it consumes collective results), its o_proj into the world sum's send buffer S0
            if po:  # po 0: no poison writes (the tail writes every element; a same-program DMA write-after-write)
                pzy = _sb((B, 512), BF)
                nisa.memset(dst=pzy, value=POISON_OUT)
                for j in range(nh):
                    _dma(dst=o_out.ap(pattern=[[R, B], [1, R]], offset=j * B * R), src=pzy[:, 0:R])
            _tail_body(RCV0, wuv_all, wo_all, idt, S0, o_out, B, A, nh, R, dv, HID, so, wu, nf, ev, 1)
        ar_world_issue(Y, S0, S1, Z1, world, pid, npg=npg, fence=fz)  # cc_world's fence under fz
        _sk_piece(cpa, comb_i, hc, iters, eps, skf, ckp, 2)  # piece 2 under the world sum
        if (npg == 1 or B % npg == 0) and not yb:  # yb 1: program 0 copies Y (program 1 never reads Y)
            ar_world_back(Y, Yp, B, HID, pid, npg)  # into slot 0 (its rows 0 .. B - 1), then a barrier on Yp
        else:  # a row count the programs can't split (T 1 at grid 2): program 0 copies Y after the sum lands
            ar_world_close(Y, npg)
            if pid == 0:
                yb = _sb((B, HID), BF)
                _dma(dst=yb, src=Y)
                _dma(dst=Yp.ap(pattern=[[HID, B], [1, HID]], offset=0), src=yb)
            if npg > 1:
                nisa.core_barrier(data=Yp, cores=(0, 1))
        if fz and npg > 1:  # program 1's sum (Z1), never read, also lands before anything else issues
            nisa.core_barrier(data=Z1, cores=(0, 1))
        if npg > 1:
            nisa.core_barrier(data=o_out, cores=(0, 1))
        # the FFN half (ffn_half_shared's body, the decode FFN half kernel's statements), both programs
        fh = ffn_half_body(S, post_p, comb_i, Yp, fnb, base, fscale, pnorm, bb, bf, fwb, hc, iters, eps, rms_eps, sig,
                           skf, Is, K, rsc, group, ring, limit, rsp, o_rt, o_gu, o_dn, o_rb, dbg, fin)
        return (gv, gc, out, o, lse, srows, sbias, over, Yp, o_out) + tuple(fh)
else:
    dsa_satwfc_kernel = None


REV = (SH.kernel_rev(__file__, "# -- the kernel source", "    dsa_satwfc_kernel = None") ^ SATWF.REV ^ MS.REV_K)


def satwf(*args, S, post_a, comb_a, blobs: dict, w_blob, ffn: dict, grid: int | None = None, ckp: int | None = None,
          **kw):
    """dsa_satwf.satwf's call with the attention block's comb_a a mhc_front.PendingComb (c, the Sinkhorn's input): the
    same arguments and returns."""
    if not isinstance(comb_a, MF.PendingComb):
        raise TypeError("dsa_satwfc.satwf takes the attention block's comb as mhc_front.PendingComb")
    ckp = CKP if ckp is None else int(ckp)
    if not 1 <= ckp <= 3:
        raise ValueError(f"ckp {ckp}: 1, 2 or 3 (one piece after each collective's issue)")
    sargs, kern, grid, T, H, R = SAT.sat_args(*args, grid=grid, _kernel=dsa_satwfc_kernel, _rev=REV, **kw)
    _, hc, Hm = S.shape
    fargs = FH.run(S, (post_a, comb_a.c, S.new_zeros(1)), blobs, w_blob, grid=grid, build_only=True, **ffn)
    fargs.pop("Yp")  # built in the kernel
    fargs.pop("rev")  # REV carries FS.REV (through SATWF.REV)
    fargs["fscale"], fargs["fwb"] = fargs.pop("scale"), fargs.pop("wb")
    clash = set(fargs) & set(sargs)
    if clash:
        raise ValueError(f"satwfc: the FFN half's arguments {sorted(clash)} are the section's names")
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    outs = wrap_nki(kern)[grid](**sargs, **fargs, ckp=ckp)
    gv, gc, o3, o, ls, _, _, ov, Yp, o_out = outs[:10]
    Sn, post, comb, rp, sp = outs[10:15]
    return ((gv, gc, o3[0], o3[1], o3[2].to(torch.int32), o.view(T, H, R), ls.view(T, H), ov.amax(dim=1), Yp, o_out,
             Sn.view(T, hc, Hm), post, comb.view(T, hc, hc), FH.to_rows(rp), sp) + tuple(outs[15:]))


def dispatch(served):
    """dsa_satwf.satwf's dispatch: a PendingComb comb_a takes this module's kernel, else `served`."""
    def satwf_(*args, comb_a, **kw):
        if isinstance(comb_a, MF.PendingComb):
            return satwf(*args, comb_a=comb_a, **kw)
        return served(*args, comb_a=comb_a, **kw)

    return satwf_
