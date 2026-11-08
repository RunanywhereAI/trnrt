"""What the fused DSA decode layer's kernels (kernels/dsa_satwf.py, kernels/dsa_satwfc.py) share: sat_args builds
their arguments for the selection-attend section (kernels/dsa_sa.py: sel2, the context-parallel sum of its lists, the
attend with the over fallback folded in, the send rows) and the tail (kernels/cc_tail.py: the all_to_all, the combine,
W_UV, o_proj); _snd_write_hbm writes the send rows; tile_wuv / tile_o repack W_UV and o_proj into the tail's pre-tiled
layout once at load. REV carries the section's (dsa_sa.REV) and the tail's (cc_tail.REV)."""

from __future__ import annotations

import torch

from . import cc_tail as CT
from . import dsa_sa as SA
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
    from .cc_tail import _dma

    def _snd_write_hbm(o, lse, SND, T: int, H: int, R: int, rows):
        """_snd_write's rows copied HBM -> HBM, o / lse straight into SND with no SBUF staging: row h T + t gets
        o[0, t, h, :] in columns 0 .. R - 1 and lse[0, t, h, 0] in column R (sh 1 / pw 1: on the device, program 1's
        SBUF-sourced rows were invisible to program 0 after a core barrier; HBM -> HBM writes are not)."""
        Cn = R + 16
        for t in rows:
            _dma(dst=SND.ap(pattern=[[T * Cn, H], [1, R]], offset=t * Cn),
                 src=o.ap(pattern=[[R, H], [1, R]], offset=t * H * R))
            _dma(dst=SND.ap(pattern=[[T * Cn, H], [1, 1]], offset=t * Cn + R),
                 src=lse.ap(pattern=[[1, H], [1, 1]], offset=t * H))
# -- the kernel source end


REV = (SH.kernel_rev(__file__, "# -- the kernel source", "# -- the kernel source end") ^ SA.REV ^ CT.REV)


def sat_args(q, w, pkc, table, nloc, positions, q_all, kc, w_uv_p, o_p, ident, keep: int, A: int, rank, world: int,
        scale: float, *, kp: int = 4, ns: int = 640, rbits2: int = 2, rbits: int = 4, nr: int | None = None, pf: int = 2,
        il: int = 1, st: int = 1, ss: int = 1, tl: int = 1, fo: int = 1, so: int = 1, wu: int = 1, nf: int = 1,
        ev: int = 1, pn: int = 0, fz: int = 0, wl: int = 0, po: int = 1, dw: int = 0, ob: int = 0,
        sh: int = 0, rp: int = 1, pw: int = 0, yb: int = 0, pq: int = 0, cs: int = 0, grid: int | None = None,
        _kernel=None,
        _rev: int | None = None):
    """The fused layer kernel _kernel's arguments for the section and the tail (dsa_satwf / dsa_satwfc add the FFN
    half's): (args, kern, grid, T, H, R). _rev: the calling kernel's REV."""
    from . import dsa_index, dsa_slots

    kern = _kernel
    if kern is None:
        raise RuntimeError("the fused DSA kernel needs the nki package (the Neuron venv)")
    if grid is None:
        from .. import platform

        grid = platform.nki_grid()
    T, Hi, _ = q.shape
    H, R = q_all.shape[1], q_all.shape[2]
    NPg = table.shape[1]
    P = NPg * DS.PPP
    cand = torch.where(torch.arange(P, device=q.device).view(1, P) < nloc.view(-1, 1), 0.0, DS.NEG_INF)
    kc2 = kc.reshape(kc.shape[0], -1)
    fp8 = kc2.dtype == torch.float8_e4m3fn
    kcp = kc2.reshape(kc2.shape[0] // dsa_slots.KP, dsa_slots.KP * R)
    args = dict(qT=q.to(torch.bfloat16).reshape(T * Hi, DS.D).t().contiguous(),
                wb=w.float().repeat(1, 4).unsqueeze(1).expand(T, 128, 4 * Hi).contiguous(),
                pkc=pkc.to(torch.bfloat16), ppg=dsa_index.page_groups(table), cand=cand.float(),
                identb=torch.eye(128, device=q.device).to(torch.bfloat16), rank=rank.to(torch.float32).reshape(1, 1),
                table=table.to(torch.int32).contiguous(), pos=positions.to(torch.float32).reshape(1, T),
                q_lat=q_all.to(torch.bfloat16).reshape(1, T, H, R), kc=kcp, w_uv_p=w_uv_p, o_p=o_p, ident=ident,
                keep=int(keep), A=int(A), nbits2=int(P.bit_length()), rbits2=int(rbits2), world=int(world),
                nbits=int((NPg * DS.PPP * A - 1).bit_length()), rbits=int(rbits), kp=int(kp), ns=int(ns),
                lkp=int(kp).bit_length() - 1, la=int(A).bit_length() - 1, scale=float(scale), fp8=int(fp8),
                rev=int(_rev), dge=0, nr=dsa_index.NR if nr is None else int(nr), qr=dsa_index.QR,
                pf=int(pf), ts=0, il=int(il), sb=1, st=int(st), ss=int(ss), tl=int(tl), fo=int(fo), fq=int(dsa_slots.QKW),
                so=int(so), wu=int(wu), nf=int(nf), ev=int(ev), pn=int(pn), fz=int(fz),
                wl=int(wl), po=int(po), ob=int(ob), sh=int(sh), rp=int(rp), pw=int(pw),
                **({"dw": int(dw)} if dw else {}), **({"yb": int(yb)} if yb else {}),
                **({"pq": int(pq)} if pq else {}),  # pq: S2's payload in one pass
                **({"cs": int(cs)} if cs else {}))  # cs 1: sel2's compaction with a program's rows stacked
    return args, kern, grid, T, H, R


def tile_wuv(w_uv: torch.Tensor) -> torch.Tensor:
    """W_UV [nh, dv, r] -> the tail's pre-tiled [128, nh RT, dv] (row j RT + rt of partition p is head j's
    K-major row rt 128 + p; a repack once at load)."""
    nh, dv, r = w_uv.shape
    return w_uv.permute(0, 2, 1).reshape(nh, r // 128, 128, dv).permute(2, 0, 1, 3).reshape(128, nh * r // 128, dv) \
        .contiguous()


def tile_o(o_dec: torch.Tensor) -> torch.Tensor:
    """o_proj [H, nh dv] -> the tail's pre-tiled [128, NCH FT, 512] (row c FT + ft of partition p is K-major row ft
    128 + p, columns c 512 .. c 512 + 511)."""
    H, Fd = o_dec.shape
    return o_dec.t().reshape(Fd // 128, 128, H // 512, 512).permute(1, 2, 0, 3).reshape(128, H // 512 * Fd // 128, 512) \
        .contiguous()


# -- the layer kernel without the FFN half: sat(), the fused DSA layer call of a layer whose FFN is not in the call (the
# plain MTP layer's decode). Below REV's section, so the kernels that fold REV keep their compile-cache keys; REV_K is
# this section's.
if nki is not None:
    from .cc_tail import LP, POISON_OUT, POISON_RX, TP, _tail_body, _tail_weights
    from .dsa_sa import _attend, _rows_i32, _sel2, _snd_write
    from .dsa_sel_s2 import _s2_payload
    from .dsa_shell import F32, I32, _sb

    def _cp_groups(world: int, a: int):
        """[[0 .. a - 1], [a .. 2 a - 1], ...] as list literals."""
        groups = []
        for g in range(world // a):
            ranks = []
            for i in range(a):
                ranks.append(g * a + i)
            groups.append(ranks)
        return ncc.ReplicaGroup(groups)

    @nki.jit
    def dsa_sat_kernel(qT, wb, pkc, ppg, cand, identb, rank, table, pos, q_lat, kc, w_uv_p, o_p, ident, keep: int,
                       A: int, nbits2: int, rbits2: int, world: int, nbits: int, rbits: int, kp: int, ns: int,
                       lkp: int, la: int, scale: float, fp8: int, rev: int, dge: int = 0, nr: int = 4, qr: int = 4,
                       pf: int = 2, ts: int = 0, il: int = 1, sb: int = 1, st: int = 1, ss: int = 1, tl: int = 1,
                       fo: int = 1, fq: int = 1, so: int = 1, wu: int = 1, nf: int = 1, ev: int = 1, pn: int = 0,
                       fz: int = 0, wl: int = 0, po: int = 1, dw: int = 0, ob: int = 0,
                       sh: int = 0, rp: int = 1, pw: int = 0, cs: int = 0):
        """dsa_sa's arguments plus the tail's: w_uv_p bf16 [128, nh RT, dv], o_p bf16 [128, NCH FT, 512] (tile_wuv /
        tile_o), ident bf16 [T, T]; so / wu / nf / ev as cc_tail's; pn 1 the padding plant, pn 2 its positive control
        (row 0's lse column). fz 1: the fenced form: C1 S2 in cc_world's fenced shape, C2 core barriers on the receive
        buffers after the poison and after the all_to_all; fz 0 the unfenced control."""
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
        # every buffer allocated in both programs before anything depends on the program id
        gv = nl.ndarray((A, B, keep), dtype=F32, buffer=nl.shared_hbm)
        gc = nl.ndarray((A, B, keep), dtype=F32, buffer=nl.shared_hbm)
        if fz:  # (fz, C1) cc_world's fenced shape, a shared send / receive buffer a program (a barrier on private HBM
            # is untested in any form)
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
        y = nl.ndarray((B, HID), dtype=BF, buffer=nl.shared_hbm)  # P: this rank's o_proj partial (wa 0)
        o_out = nl.ndarray((nh, B, R), dtype=BF, buffer=nl.shared_hbm)  # the combine's o
        if dw == 1:  # dw 1: the held tiles as the tail reads them (a debugging dump)
            wdmp = nl.ndarray((TP, nh * RT, dv), dtype=BF, buffer=nl.shared_hbm)
            odmp = nl.ndarray((TP, NCH * FT, 512), dtype=BF, buffer=nl.shared_hbm)
            idmp = nl.ndarray((B, B), dtype=BF, buffer=nl.shared_hbm)
        if dw == 2:  # dw 2: SND as the a2a sends it and RCV0 as the tail reads it
            sdmp = nl.ndarray((H * B, Cn), dtype=F32, buffer=nl.shared_hbm)
            rdmp = nl.ndarray((H * B, Cn), dtype=F32, buffer=nl.shared_hbm)
        if not wl:  # wl 0: the tail's weights and identity at entry, held through the section
            wuv_all = nl.ndarray((TP, nh * RT, dv), dtype=BF, buffer=nl.sbuf)
            wo_all = nl.ndarray((TP, NCH * FT, 512), dtype=BF, buffer=nl.sbuf)
            if pid == 0:  # program 0 consumes; dge none, off sel2's queues
                _tail_weights(w_uv_p, o_p, wuv_all, wo_all, nh, RT, dv, NCH, FT)
        if not wl:
            idt = nl.ndarray((B, B), dtype=BF, buffer=nl.sbuf)
            _dma(dst=idt, src=ident)
        # P1: sel2
        out = _sel2(qT, wb, pkc, ppg, cand, identb, rank, keep, A, nbits2, rbits2, rev, dge, nr, qr, pf, cs)
        # P2: S2 (step 1's), program 0 writes gv / gc
        _s2_payload(out, rank, B, keep, A, s2src)
        if fz and npg > 1:  # (fz, C1) both programs' payload writes land before the collective reads them
            nisa.core_barrier(data=s2s0, cores=(0, 1))
            nisa.core_barrier(data=s2s1, cores=(0, 1))
        ncc.all_reduce(srcs=[s2src], dsts=[s2dst], replica_group=_cp_groups(world, A), op=nl.add)
        if fz and npg > 1:  # (fz, C1) program 0's sum lands before its DMA reads it
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
        if wl:  # wl 1: the tail's weights and identity after the attend, no tile held across the section
            wuv_all = nl.ndarray((TP, nh * RT, dv), dtype=BF, buffer=nl.sbuf)
            wo_all = nl.ndarray((TP, NCH * FT, 512), dtype=BF, buffer=nl.sbuf)
            if pid == 0:
                _tail_weights(w_uv_p, o_p, wuv_all, wo_all, nh, RT, dv, NCH, FT)
            idt = nl.ndarray((B, B), dtype=BF, buffer=nl.sbuf)
            _dma(dst=idt, src=ident)
        o, lse, srows, sbias, over = _attend(gv, gc, RS, table, pos, rank, q_lat, kc, keep, A, kp, ns, nbits, rbits,
                                             lkp, la, scale, fp8, rev, ts, il, sb, st, ss, tl, fo, fq)
        if ob and npg > 1:  # ob 1: the attend's HBM outputs o / lse land before _snd_write's DMA reads them back
            nisa.core_barrier(data=o, cores=(0, 1))  # (a same-tensor DMA write then read)
            nisa.core_barrier(data=lse, cores=(0, 1))
        # the receive buffers poisoned (each program its own, cc_tail's form), then the send rows
        if rp:  # rp 0: no poison of the receive buffers (an SBUF-sourced write the a2a's write may overtake)
            pzr = _sb((H * B, Cn))
            nisa.memset(dst=pzr, value=POISON_RX)
            _dma(dst=RCV0 if pid == 0 else RCV1, src=pzr)
            if fz and npg > 1:  # (fz, C2) the poison lands before the all_to_all writes the receive buffers
                nisa.core_barrier(data=RCV0, cores=(0, 1))
                nisa.core_barrier(data=RCV1, cores=(0, 1))
        split = npg == 2 and B >= 2
        k_lo, k_hi = (0, B) if not split else ((0, (B + 1) // 2) if pid == 0 else ((B + 1) // 2, B))
        rows = []
        if pw:  # pw 1 (informative, expected to fail): program 0 writes every row, program 1 none
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
        if pn == 1:  # the padding plant: NaN in these rows' padding columns r+1 .., which no receiver reads
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
        if dw == 2 and pid == 0:  # SND after its barrier (both programs' rows)
            _dma(dst=sdmp, src=SND)
        ncc.all_to_all(srcs=[SND], dsts=[RCV0 if pid == 0 else RCV1], replica_group=_cp_groups(world, A),
                       collective_dim=0)
        if fz and npg > 1:  # (fz, C2) the received rows land before the tail's DMA reads them
            nisa.core_barrier(data=RCV0, cores=(0, 1))
            nisa.core_barrier(data=RCV1, cores=(0, 1))
        if pid == 0:  # the tail, program 0 alone: y and o_out poisoned first (their only writer)
            if po:  # po 0: no poison writes (the tail writes every element; a same-program DMA write-after-write)
                pzy = _sb((B, 512), BF)
                nisa.memset(dst=pzy, value=POISON_OUT)
                for c in range(NCH):
                    _dma(dst=y.ap(pattern=[[HID, B], [1, 512]], offset=c * 512), src=pzy)
                for j in range(nh):
                    _dma(dst=o_out.ap(pattern=[[R, B], [1, R]], offset=j * B * R), src=pzy[:, 0:R])
            if dw == 1:  # just before the tail reads them
                _dma(dst=wdmp, src=wuv_all)
                _dma(dst=odmp, src=wo_all)
                _dma(dst=idmp, src=idt)
            if dw == 2:  # RCV0 just before the tail reads it
                _dma(dst=rdmp, src=RCV0)
            _tail_body(RCV0, wuv_all, wo_all, idt, y, o_out, B, A, nh, R, dv, HID, so, wu, nf, ev, 1)
        if npg > 1:
            nisa.core_barrier(data=y, cores=(0, 1))
            nisa.core_barrier(data=o_out, cores=(0, 1))
        if dw == 1:
            return gv, gc, out, o, lse, srows, sbias, over, y, o_out, wdmp, odmp, idmp
        if dw == 2:
            return gv, gc, out, o, lse, srows, sbias, over, y, o_out, sdmp, rdmp
        return gv, gc, out, o, lse, srows, sbias, over, y, o_out
else:
    dsa_sat_kernel = None

# -- the layer kernel end


REV_K = (SH.kernel_rev(__file__, "# -- the layer kernel without", "# -- the layer kernel end") ^ SA.REV ^ CT.REV)


def sat(q, w, pkc, table, nloc, positions, q_all, kc, w_uv_p, o_p, ident, keep: int, A: int, rank, world: int,
        scale: float, *, kp: int = 4, ns: int = 640, rbits2: int = 2, rbits: int = 4, nr: int | None = None, pf: int = 2,
        il: int = 1, st: int = 1, ss: int = 1, tl: int = 1, fo: int = 1, so: int = 1, wu: int = 1, nf: int = 1,
        ev: int = 1, pn: int = 0, fz: int = 0, wl: int = 0, po: int = 1, dw: int = 0, ob: int = 0,
        sh: int = 0, rp: int = 1, pw: int = 0, yb: int = 0, cs: int = 0, grid: int | None = None):
    """The joined layer on a Neuron device (dsa_sat_kernel): dsa_sa.sa's arguments plus the tail's pre-tiled weights and
    identity. Returns (gv, gc, lval, cpool, rows, o [T, H, R], lse [T, H], over [1], y [T, HID] bf16 (this rank's
    o_proj partial P, which the caller all-reduces over the world), o_out [nh, T, R] bf16)."""
    args, kern, grid, T, H, R = sat_args(q, w, pkc, table, nloc, positions, q_all, kc, w_uv_p, o_p, ident, keep, A, rank,
                                         world, scale, kp=kp, ns=ns, rbits2=rbits2, rbits=rbits, nr=nr, pf=pf, il=il,
                                         st=st, ss=ss, tl=tl, fo=fo, so=so, wu=wu, nf=nf, ev=ev, pn=pn, fz=fz,
                                         wl=wl, po=po, dw=dw, ob=ob, sh=sh, rp=rp, pw=pw, yb=yb, cs=cs, grid=grid,
                                         _kernel=dsa_sat_kernel, _rev=REV_K)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    outs = wrap_nki(kern)[grid](**args)
    gv, gc, o3, o, ls, _srows, _sbias, ov, y, o_out = outs[:10]
    return (gv, gc, o3[0], o3[1], o3[2].to(torch.int32), o.view(T, H, R), ls.view(T, H), ov.amax(dim=1), y,
            o_out) + tuple(outs[10:])  # + dw's dumps
