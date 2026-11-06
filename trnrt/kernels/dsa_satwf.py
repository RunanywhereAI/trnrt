"""The fused DSA decode layer (the served form at decode steps of up to 8 rows): the selection-attend section, the
tail, the world sum of the attention output and then the layer's FFN half (kernels/ffn_half_shared.py, the decode FFN
half's kernel body statement for statement), in one kernel: one call a DSA layer from sel2's input to the FFN half's
partials.

- The section (kernels/dsa_sa.py: sel2, the context-parallel sum of its lists, the attend with the over fallback folded
  in, the send rows), the all_to_all and the tail (kernels/cc_tail.py: the combine, W_UV, o_proj), then the world sum
  of the partial P through cc_world's ar_world_issue, with the FFN half's inputs added (renamed where a name is the
  section's: the hc scale `fscale`, the routed experts `fwb`).
- The attention output enters the FFN half as ffn_half.decode builds it for a DSA layer: Yp [npg, T, H] bf16, slot 0
  the world-summed attention output, slots 1 .. zeros (mix_in sums the slots in fp32 and rounds once: y exactly).
  Program 1 writes the zeros early; ar_world_back reads Y back into slot 0 (its last barrier on Yp), or at a row count
  the programs can't split program 0 copies Y in after ar_world_close, then a barrier on Yp.
- Both programs then run FS.ffn_half_body (the mix, mHC's FFN front, the norm, the MoE body); the routed / shared
  partials' sums, the FFN all-reduce and the next mix stay XLA's.
- Returns the section's outputs, Yp, the combine's o, then the FFN half's (S_out, post_out, comb_out, rp, sp) (+ dbg's).

The served attention block's comb is mhc_front's PendingComb, so satwf dispatches to kernels/dsa_satwfc.py, this kernel
with the block's Sinkhorn inside the call (below). REV carries the section's, the tail's, the world sum's and
ffn_half_shared's."""

from __future__ import annotations

import torch

from . import cc_world as CW
from . import dsa_sat as SAT
from . import ffn_half as FH
from . import ffn_half_shared as FS
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
    from .cc_tail import LP, POISON_OUT, POISON_RX, TP, _dma, _tail_body, _tail_weights

    from .dsa_sa import _attend, _rows_i32, _sel2, _snd_write
    from .dsa_sat import _snd_write_hbm
    from .dsa_sel_s2 import _s2_payload
    from .dsa_shell import F32, I32, _sb
    from .cc_world import _dma_static, ar_world_alloc, ar_world_back, ar_world_close, ar_world_issue, ar_world_zero1
    from .ffn_half_shared import ffn_half_body

    def _cp_groups(world: int, a: int):
        """[[0 .. a - 1], [a .. 2 a - 1], ...] as list literals (the tracer takes no list(range()))."""
        groups = []
        for g in range(world // a):
            ranks = []
            for i in range(a):
                ranks.append(g * a + i)
            groups.append(ranks)
        return ncc.ReplicaGroup(groups)

    @nki.jit
    def dsa_satwf_kernel(qT, wb, pkc, ppg, cand, identb, rank, table, pos, q_lat, kc, w_uv_p, o_p, ident, S,
                              post_p, comb_p, fnb, base, fscale, pnorm, bb, bf, fwb, keep: int, A: int, nbits2: int,
                              rbits2: int, world: int, nbits: int, rbits: int, kp: int, ns: int, lkp: int, la: int,
                              scale: float, fp8: int, rev: int, hc: int, iters: int, eps: float, rms_eps: float, sig: int,
                              skf: int, Is: int, K: int, rsc: float, group: int, ring: int, limit: float, rsp: int,
                              o_rt: int, o_gu: int, o_dn: int, o_rb: int, fin: int, dge: int = 0, nr: int = 4,
                              qr: int = 4, pf: int = 2, ts: int = 0, il: int = 1, sb: int = 1, st: int = 1, ss: int = 1,
                              tl: int = 1, fo: int = 1, fq: int = 1, so: int = 1, wu: int = 1, nf: int = 1, ev: int = 1,
                              pn: int = 0, dbg: int = 0, fz: int = 0, wl: int = 0,
                              po: int = 1, ob: int = 0,
                              sh: int = 0, rp: int = 1, pw: int = 0,
                              yb: int = 0, pq: int = 0, cs: int = 0):
        """The section's and the tail's arguments (dsa_sat.sat_args), then the FFN half's (ffn_half_pb's kernel's, less
        Yp and rev; its scale `fscale`, its wb `fwb`): S bf16 [T, hc H] the streams before the attention output's mix,
        post_p fp32 [T, hc], comb_p fp32 [T, hc hc], the layer's blobs and routed experts, the scalars as ffn_half.run
        builds them. fz 1: the fenced form (core barriers on the collectives' buffers on both sides, cc_world's fence);
        fz 0 the unfenced one."""
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
        # the attend with the over fallback folded in
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
        fh = ffn_half_body(S, post_p, comb_p, Yp, fnb, base, fscale, pnorm, bb, bf, fwb, hc, iters, eps, rms_eps, sig,
                           skf, Is, K, rsc, group, ring, limit, rsp, o_rt, o_gu, o_dn, o_rb, dbg, fin)
        return (gv, gc, out, o, lse, srows, sbias, over, Yp, o_out) + tuple(fh)
else:
    dsa_satwf_kernel = None


REV = (SH.kernel_rev(__file__, "# -- the kernel source", "    dsa_satwf_kernel = None") ^ SAT.REV ^ CW.REV ^ FS.REV)


def satwf(*args, S, post_a, comb_a, blobs: dict, w_blob, ffn: dict, grid: int | None = None, **kw):
    """The section, the tail, the world sum and the FFN half in one kernel. args / kw: dsa_sat.sat_args'; S bf16
    [T, hc, H] the streams before the attention output's mix, (post_a fp32 [T, hc], comb_a fp32 [T, hc, hc]) that mix,
    blobs / w_blob as ffn_half.decode passes them, ffn: ffn_half.run's keywords (Is, K, scale, group, ring, limit, rsp,
    hc, iters, hc_eps, rms_eps[, fin, dbg]). Returns (gv, gc, lval, cpool, rows, o [T, H, R], lse [T, H], over [1], Yp
    bf16 [npg, T, H], o_out [nh, T, R] bf16), then ffn_half.run's (S_n [T, hc, H], post, comb [T, hc, hc], rp [npg, T,
    H] (to_rows), sp) (+ dbg's)."""
    sargs, kern, grid, T, H, R = SAT.sat_args(*args, grid=grid, _kernel=dsa_satwf_kernel, _rev=REV, **kw)
    _, hc, Hm = S.shape
    fargs = FH.run(S, (post_a, comb_a, S.new_zeros(1)), blobs, w_blob, grid=grid, build_only=True, **ffn)
    fargs.pop("Yp")  # built in the kernel
    fargs.pop("rev")  # REV carries FS.REV
    fargs["fscale"], fargs["fwb"] = fargs.pop("scale"), fargs.pop("wb")
    clash = set(fargs) & set(sargs)
    if clash:
        raise ValueError(f"satwf: the FFN half's arguments {sorted(clash)} are the section's names")
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    outs = wrap_nki(kern)[grid](**sargs, **fargs)
    gv, gc, o3, o, ls, _, _, ov, Yp, o_out = outs[:10]
    Sn, post, comb, rp, sp = outs[10:15]
    return ((gv, gc, o3[0], o3[1], o3[2].to(torch.int32), o.view(T, H, R), ls.view(T, H), ov.amax(dim=1), Yp, o_out,
             Sn.view(T, hc, Hm), post, comb.view(T, hc, hc), FH.to_rows(rp), sp) + tuple(outs[15:]))


# The served attention block's comb is mhc_front's PendingComb: it takes dsa_satwfc's kernel (the block's Sinkhorn
# inside this call). Here, below the kernel, its REV and satwf, so nothing above moves. dsa_satwfc is imported at the
# call (it imports this module).
_satwf_served = satwf


def satwf(*args, comb_a, **kw):  # noqa: F811
    from . import dsa_satwfc

    return dsa_satwfc.dispatch(_satwf_served)(*args, comb_a=comb_a, **kw)


# The P3072 decode page bucket (SEL2G; the engine sets it on every rank from EngineConfig.dsa_sel2g before the shard is
# built): a decode call whose score tile is not whole 128-column chunks (dsa_sel2g.takes: P3072's NT 192) takes
# dsa_satwfg's kernel, whose selection is dsa_sel2g's (sel2 with its compaction at any tile width). Off, satwf is the
# function above, so no other call's graph moves. Read at call time; tools/make_satwf_sel2g.py cuts this block from the
# copies it generates.
SEL2G = False
_satwf_pre_sel2g = satwf


def satwf(*args, **kw):  # noqa: F811
    if SEL2G:  # the block table is sat_args' fourth argument (models/mla.py passes it positionally)
        from . import dsa_satwfg, dsa_sel2g

        if dsa_sel2g.takes(args[3] if len(args) > 3 else kw["table"]):
            return dsa_satwfg.satwf(*args, **kw)
    return _satwf_pre_sel2g(*args, **kw)
