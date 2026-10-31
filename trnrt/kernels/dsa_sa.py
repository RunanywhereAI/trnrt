"""The fused DSA decode layer's selection-attend section, as kernel helpers the fused layer kernels
(kernels/dsa_satwf.py, kernels/dsa_satwfc.py) call: sel2 (dsa_shell's sel2 kernel, its body verbatim), S2 in-kernel
(kernels/dsa_sel_s2.py's payload and context-parallel sum), the attend with the over fallback folded in (dsa_shell's
attend kernel, its body verbatim, reading the gathered lists in place), then the attend -> tail hand-off's send rows
SND.

- sel2 writes out [3, B, keep] (lval, cpool, rows) and closes with its barrier.
- S2: both programs build _cp_merge_gather's zero-padded payload from out, all-reduce it over the CP group; program 0
  writes the sums to gv and gc [A, B, keep] (the attend's ev / ec), and a core barrier on each follows.
- Each program converts every row's pool rows to int32 into its own private copy (the attend's rows_sel), so no barrier.
- The attend body runs on gv / gc / rows_sel (st / ss / tl / fo as the caller passes, all 1 served); its o, lse,
  srows, sbias and over are what the plain decode path's attend call returns.
- SND [H B, r + 16] fp32 (shared HBM), H = A nh the attend's heads: row h B + t holds o[t, h, :] in columns 0 .. r - 1
  and lse[t, h] in column r; columns r + 1 .. are never written (the receiver never reads them). SND and the receive
  buffers are internal (a collective may not touch the kernel's IO: NCC_IBTN132).

Its own module: nki.collectives resolves from the kernel's own module. REV carries dsa_shell's sel2 and attend
revisions (REV2, REVA) and dsa_sel_s2's."""

from __future__ import annotations

from . import dsa_sel_s2 as S2
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
    from .dsa_sel_s2 import _s2_payload
    from .dsa_shell import BF16, D, F32, I32, NEG_INF, PPP, VE, _compact2, _compact2s, _compact_sb, _ix_gather
    from .dsa_shell import _ix_row_setup
    from .dsa_shell import _ix_stage_a, _ix_stage_b, _loop0, _merge_init, _merge_rows, _merge_rows_st, _sb
    from .dsa_shell import _SC_C, _sc_attend, _sc_compact, _select2s, _SL_KP, _SL_KTR, _sl_pair, _sl_psum
    from .dsa_shell import _slot_live_sb, _slot_rows_sb, _sn_row, _topk_consts, dma_static

    def _sel2(qT, wb, pkc, ppg, cand, identb, rank, keep: int, A: int, nbits: int, rbits: int, rev: int, dge: int,
              nr: int, qr: int, pf: int, cs: int = 0):
        """dsa_shell.dsa_shell_sel2_kernel's body, verbatim: out [3, B, keep]. cs: its compaction form (0: _compact2;
        1, served: _compact2s, the same bits)."""
        Dk, BH = qT.shape
        B = cand.shape[0]
        Hi = BH // B
        NG = ppg.shape[2]
        PPP_ = pkc.shape[1] // Dk
        NT = NG * PPP_
        P = NT * 128
        KT = keep // 128
        assert Dk == D and PPP_ == PPP and wb.shape[2] == 4 * Hi and qr * 4 * Hi <= 512 and keep % 128 == 0
        assert keep < P and NT <= 512 and (NT <= 128 or NT % 128 == 0) and rbits > 1  # _select2s: w-bit rounds
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        out = nl.ndarray((3, B, keep), dtype=F32, buffer=nl.shared_hbm)  # lval, cpool, rows (exact integers < 2^24)
        scout = nl.ndarray((B, P), dtype=F32, buffer=nl.private_hbm)  # dsa_index's score DMA (unread)
        IB = _sb((128, 128), BF16)
        dma_static(dst=IB, src=identb)
        QT = _sb((D, BH), BF16)
        dma_static(dst=QT, src=qT)
        Kr, KTr, PTr, PSr, Rr, OFr, WBr, Sr, Cr = [], [], [], [], [], [], [], [], []
        for _ in range(qr):
            PTr.append(nl.ndarray((D, 512), dtype=F32, buffer=nl.psum))
        PSb = nl.ndarray((128, 512), dtype=F32, buffer=nl.psum)
        for i in range(qr):
            PSr.append(PSb[:, i * 4 * Hi:(i + 1) * 4 * Hi])
        for _ in range(nr):
            Kr.append(_sb((128, PPP * D), BF16))
        for _ in range(qr):
            KTr.append(_sb((D, 512), BF16))
            Rr.append(_sb((128, 4 * Hi)))
        for _ in range(3):
            OFr.append(_sb((128, NG), I32))
            WBr.append(_sb((128, 4 * Hi)))
            Sr.append(_sb((128, NT)))
            Cr.append(_sb((128, NT)))
        rows = []
        for b in range(B):
            if b % npg == pid:
                rows.append(b)
        SALL = []
        for _ in range(len(rows)):
            SALL.append(_sb((128, NT)))
        SS = _sb((128, max(1, len(rows)), NT))  # the rows' score tiles side by side, for the stacked select
        dmode = nisa.dge_mode.hwdge if dge == 2 else (nisa.dge_mode.swdge if dge == 1 else nisa.dge_mode.unknown)
        G = []
        for i in range(len(rows)):
            for g in range(NG):
                G.append((i, rows[i], g))
        QS = []
        for k in range(len(G)):
            for h4 in range(PPP // 4):
                QS.append((k, G[k][0], G[k][1], G[k][2], h4))
        PRE = min(nr - 1, pf, NG)  # page groups gathered ahead (pf; dsa_index's pipeline takes 2)
        LAG = min(qr - 1, 2, PPP // 4)
        X = dict(B=B, NG=NG, Hi=Hi, NT=NT, P=P, PPP=PPP, nr=nr, qr=qr, dbg=0, dmode=dmode, ppg=ppg, wb=wb, cand=cand,
                 pkc=pkc, out=scout, IB=IB, QT=QT, OF=OFr, WB=WBr, S=Sr, C=Cr, K=Kr, KT=KTr, PT=PTr, PS=PSr, R=Rr, G=G,
                 QS=QS)
        if len(rows) > 0:
            _ix_row_setup(X, 0, rows[0])
            for k in range(min(PRE, len(G))):
                if G[k][2] == 0 and G[k][0] > 0:
                    _ix_row_setup(X, G[k][0], G[k][1])
                _ix_gather(X, k)
        for n in range(len(QS) + LAG):
            if n < len(QS):
                k = QS[n][0]
                if QS[n][4] == 0 and k + PRE < len(G):
                    if G[k + PRE][2] == 0:
                        _ix_row_setup(X, G[k + PRE][0], G[k + PRE][1])
                    _ix_gather(X, k + PRE)
                _ix_stage_a(X, n)
            if n >= LAG:
                m = n - LAG
                _ix_stage_b(X, m)
                if QS[m][3] == NG - 1 and QS[m][4] == PPP // 4 - 1:  # the row's scores are complete: keep them
                    nisa.tensor_copy(dst=SALL[QS[m][1]], src=Sr[QS[m][1] % 3], engine=VE)
                    nisa.tensor_copy(dst=SS[:, QS[m][1], :], src=Sr[QS[m][1] % 3], engine=VE)
        if len(rows) > 0:
            # constants: the selection's, the identities and triangles, iotas (exact integers in fp32)
            wio, po, TG, bits = _topk_consts(NT, 128, 7)
            po0 = _sb((128, 1))
            nisa.memset(dst=po0, value=0.0)
            posi = _sb((128, NT), I32)  # the pool of each tile entry: p NT + c, the tie order
            nisa.iota(dst=posi, pattern=[[1, NT]], offset=0, channel_multiplier=NT)
            POS = _sb((128, NT))
            nisa.tensor_copy(dst=POS, src=posi, engine=VE)
            O = _select2s(SS, len(rows), NT, TG, POS, keep, nbits, rbits)
            ji_ = _sb((128, 512), I32)
            nisa.iota(dst=ji_, pattern=[[1, 512]], offset=0, channel_multiplier=0)
            io512 = _sb((128, 512))
            nisa.tensor_copy(dst=io512, src=ji_, engine=VE)
            pi = _sb((128, 1), I32)
            nisa.iota(dst=pi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
            pf = _sb((128, 1))
            nisa.tensor_copy(dst=pf, src=pi, engine=VE)
            dd = _sb((128, 128))  # c - p
            nisa.tensor_scalar(dst=dd, data=io512[:, 0:128], op0=nl.subtract, operand0=pf, engine=VE)
            IF = _sb((128, 128))
            nisa.tensor_scalar(dst=IF, data=dd, op0=nl.equal, operand0=0.0, engine=VE)
            UPI = _sb((128, 128))  # [p' <= p] at [p', p]
            nisa.tensor_scalar(dst=UPI, data=dd, op0=nl.greater_equal, operand0=0.0, engine=VE)
            nch = NT // 128 if NT >= 128 else 1
            cw = 128 if NT >= 128 else NT
            UT = _sb((128, nch * NT), BF16)  # block j: [c', c] = (j cw + c' <= c), c' < cw, c < NT
            for j in range(nch):
                dj = _sb((128, NT))
                nisa.tensor_scalar(dst=dj, data=io512[:, 0:NT], op0=nl.subtract, operand0=pf, op1=nl.subtract,
                                   operand1=float(j * cw), engine=VE)
                nisa.tensor_scalar(dst=UT[:, j * NT:(j + 1) * NT], data=dj, op0=nl.greater_equal, operand0=0.0,
                                   engine=VE)
            ONES1 = _sb((1, 128))
            nisa.memset(dst=ONES1, value=1.0)
            kvi = _sb((128, KT), I32)
            nisa.iota(dst=kvi, pattern=[[128, KT]], offset=0, channel_multiplier=1)
            KV = _sb((128, KT))
            nisa.tensor_copy(dst=KV, src=kvi, engine=VE)
            RK = _sb((128, 1))
            dma_static(dst=RK, src=rank.ap(pattern=[[0, 128], [1, 1]], offset=0))
            XC = dict(NT=NT, NG=NG, B=B, keep=keep, KT=KT, nch=nch, cw=cw, A=A, O=O, SALL=SALL, ppg=ppg,
                      IO512=io512,
                      UPI=UPI, IF=IF, ONES1=ONES1, IB=IB, IOTA=io512[:, 0:128], IONT=io512[:, 0:NT],
                      IONG=io512[:, 0:NG], UT=UT, KV=KV, RK=RK, out=out)
            ii = []
            for i in range(len(rows)):
                ii.append(i)
            if cs == 1 and len(rows) >= 4:  # cs 1: the program's rows stacked, up to _C2S_G a tile (the same bits), at
                # >= 4 rows a program only (measured on trn2: 2 rows +39.6 us a call, 3 rows -0.1)
                _compact2s(XC, ii, rows)
            else:
                _compact2(XC, ii, rows)  # the program's rows interleaved
        if npg > 1:
            nisa.core_barrier(data=out, cores=(0, 1))  # one output, one barrier (three cost ~10 us, 0a46c746's view)
        return out

    def _attend(ev, ec, rows_sel, table, pos, rank, q_lat, kc, keep: int, A: int, kp: int, ns: int, nbits: int,
                rbits: int, lkp: int, la: int, scale: float, fp8: int, rev: int, ts: int, il: int, sb: int, st: int,
                ss: int, tl: int, fo: int, fq: int):
        """dsa_shell.dsa_shell_attend_kernel's body, verbatim: (o, lse, srows, sbias, over)."""
        AA, T, K = ev.shape
        _, rpi, H, R = q_lat.shape
        W = AA * K // 128
        NCH = ns // 128
        LC = R // 128
        TC = _SL_KP * _SC_C
        assert rpi == T and ns % 128 == 0 and kp == _SL_KP
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        split = npg == 2 and T >= 2
        k_lo, k_hi = (0, T) if not split else ((0, (T + 1) // 2) if pid == 0 else ((T + 1) // 2, T))
        srows = nl.ndarray((T, ns), dtype=I32, buffer=nl.shared_hbm)
        sbias = nl.ndarray((T, ns * kp), dtype=F32, buffer=nl.shared_hbm)
        o = nl.ndarray((1, T, H, R), dtype=F32, buffer=nl.shared_hbm)
        lse = nl.ndarray((1, T, H, 1), dtype=F32, buffer=nl.shared_hbm)
        over = nl.ndarray((1, npg), dtype=F32, buffer=nl.shared_hbm)
        msel = nl.ndarray((T * 128, W), dtype=F32, buffer=nl.private_hbm)
        assert (st or not ss) and (ss or not tl)
        cpx = None if ss else nl.ndarray((1, 128, T, NCH, 4), dtype=BF16, buffer=nl.private_hbm)
        XM = _merge_init(ev, ec, rows_sel, table, pos, rank, srows, sbias, msel, keep, A, kp, ns, nbits, rbits, lkp, la,
                         ts, il, sb)
        mine = []
        for b in range(k_lo, k_hi):
            mine.append(b)
        if ss:  # the slot table's pool rows first: they need nothing of the selection
            cpt = _sb((128, T, NCH, 4), BF16)
            nisa.memset(dst=cpt, value=0.0)
            cnt = _sb((128, T))
            nisa.memset(dst=cnt, value=0.0)
            _slot_rows_sb(XM, mine, cpt)
        if st:
            assert il and sb and not ts
            XM["pos_in"] = pos
            XM["rank_in"] = rank
            MS = _merge_rows_st(XM, mine, tl)
        else:
            _merge_rows(XM, mine)
        if ts:  # every program the same number of device loops (_loop0)
            need = 1 if il else ((T + 1) // 2 if split else T)
            have = len(mine)
            if il:
                have = min(1, len(mine))
            for _ in range(have, need):
                _loop0()
        if ss:
            _slot_live_sb(XM, MS, mine, cpt, cnt)
        else:
            # the slot table of this program's rows (the others' entries zero: no live slot), from the rows and bias
            # just written: digits hi, mid, lo of the pool row (>> 16, >> 8 & 255, & 255: exact below 2^24) and the live
            # tokens
            cpt = _sb((128, T, NCH, 4), BF16)
            nisa.memset(dst=cpt, value=0.0)
            cnt = _sb((128, T))
            nisa.memset(dst=cnt, value=0.0)
            for b in mine:
                ri = _sb((128, NCH), I32)
                dma_static(dst=ri, src=srows.ap(pattern=[[1, 128], [128, NCH]], offset=b * ns))
                bb = _sb((128, NCH, kp))
                dma_static(dst=bb, src=sbias.ap(pattern=[[kp, 128], [128 * kp, NCH], [1, kp]], offset=b * ns * kp))
                dg = _sb((128, NCH), I32)
                df = _sb((128, NCH))
                nisa.tensor_scalar(dst=dg, data=ri, op0=nl.right_shift, operand0=16, engine=VE)
                nisa.tensor_copy(dst=df, src=dg, engine=VE)
                nisa.tensor_copy(dst=cpt[:, b, :, 0], src=df, engine=VE)
                nisa.tensor_scalar(dst=dg, data=ri, op0=nl.right_shift, operand0=8, op1=nl.bitwise_and, operand1=255,
                                   engine=VE)
                nisa.tensor_copy(dst=df, src=dg, engine=VE)
                nisa.tensor_copy(dst=cpt[:, b, :, 1], src=df, engine=VE)
                nisa.tensor_scalar(dst=dg, data=ri, op0=nl.bitwise_and, operand0=255, engine=VE)
                nisa.tensor_copy(dst=df, src=dg, engine=VE)
                nisa.tensor_copy(dst=cpt[:, b, :, 2], src=df, engine=VE)
                lv = _sb((128, NCH, kp))
                nisa.tensor_scalar(dst=lv, data=bb, op0=nl.greater, operand0=NEG_INF / 2, engine=VE)
                nv = _sb((128, NCH))
                nisa.tensor_reduce(dst=nv, op=nl.add, data=lv, axis=2)
                nisa.tensor_copy(dst=cpt[:, b, :, 3], src=nv, engine=VE)
                m = _sb((128, NCH))
                nisa.tensor_scalar_reduce(dst=m, data=nv, op0=nl.greater, operand0=0.0, reduce_op=nl.add,
                                          reduce_res=cnt[:, b:b + 1])
            dma_static(dst=cpx.ap(pattern=[[T * NCH * 4, 128], [1, T * NCH * 4]], offset=0),
                       src=cpt.reshape((128, T * NCH * 4)))
        # over: some row of this program with more than C live slots (its count summed over the partitions)
        ones = _sb((128, 128))
        nisa.memset(dst=ones, value=1.0)
        pct = nl.ndarray((128, T), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=pct, stationary=ones, moving=cnt, accumulate=False)
        ovr = _sb((1, T))
        nisa.tensor_scalar(dst=ovr, data=pct[0:1, :], op0=nl.greater, operand0=float(_SC_C), engine=VE)
        ov = _sb((1, 1))
        nisa.tensor_reduce(dst=ov, op=nl.maximum, data=ovr, axis=1)
        dma_static(dst=over.ap(pattern=[[npg, 1], [1, 1]], offset=pid), src=ov)
        if fo and split:  # every program's flag (write, barrier, read back): one path for every row
            nisa.core_barrier(data=over, cores=(0, 1))
            ova = _sb((1, npg))
            dma_static(dst=ova, src=over)
            ov = _sb((1, 1))
            nisa.tensor_reduce(dst=ov, op=nl.maximum, data=ova, axis=1)
        ncf = _sb((1, 1))  # this program's iterations: 1 unless one of its rows is over (fo: one of any program's)
        nisa.tensor_scalar(dst=ncf, data=ov, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0, engine=VE)
        nci = _sb((1, 1), I32)
        nisa.tensor_copy(dst=nci, src=ncf, engine=VE)
        # dsa_slots_c's tiles and constants (its kernel's own, for its helpers)
        KD = kc.dtype
        if fp8:
            KD = nl.float8_e4m3
        X = dict(H=H, R=R, LC=LC, NCH=NCH, rpi=rpi, q_lat=q_lat, kc=kc, o=o, lse=lse, want_lse=1, cparts=cpx,
                 k_lo=k_lo, k_hi=k_hi)
        ji = _sb((128, 128), I32)
        nisa.iota(dst=ji, pattern=[[1, 128]], offset=0, channel_multiplier=0)
        io = _sb((128, 128))  # io[p, c] = c
        nisa.tensor_copy(dst=io, src=ji)
        pi = _sb((128, 1), I32)
        nisa.iota(dst=pi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        pfc = _sb((128, 1))
        nisa.tensor_copy(dst=pfc, src=pi)
        dd = _sb((128, 128))  # c - p
        nisa.tensor_scalar(dst=dd, data=io, op0=nl.subtract, operand0=pfc)
        idf = _sb((128, 128))  # [p == c]
        nisa.tensor_scalar(dst=idf, data=dd, op0=nl.equal, operand0=0.0)
        IB = _sb((128, 128), BF16)
        nisa.tensor_copy(dst=IB, src=idf)
        X["IB"] = IB
        X["scale"] = scale
        X["KC"] = []
        for _ in range(rpi):
            X["KC"].append(_sb((128, _SL_KP * R), KD))
        X["KTAC"] = _sl_pair((128, LC, TC), BF16)
        X["SC"] = _sl_pair((H, TC), F32)
        X["PC"] = _sl_pair((H, TC), BF16)
        X["QR"] = _sl_pair((H, R), BF16)
        X["QT"] = _sl_pair((128, LC, H), BF16)
        X["OB"] = _sl_pair((H, R), F32)
        X["MX"] = _sl_pair((H, 1), F32)
        X["NMX"] = _sl_pair((H, 1), F32)
        X["DEN"] = _sl_pair((H, 1), F32)
        X["RDEN"] = _sl_pair((H, 1), F32)
        X["LS"] = _sl_pair((H, 1), F32)
        X["PTb"] = _sl_pair((128, H), BF16)
        X["BRS"] = _sb((rpi, TC), BF16)
        X["CPT"] = cpt if ss else _sb((128, rpi, NCH, 4), BF16)
        X["M"] = _sb((128, rpi, NCH))
        X["INCL"] = _sb((128, rpi, NCH))
        X["CNB"] = _sb((128, rpi), BF16)
        X["PXS"] = _sb((128, rpi))
        X["PL"] = _sb((128, rpi, NCH))
        X["OH"] = _sb((128, 4, 128), BF16)
        X["CM"] = _sb((128, rpi, NCH, 4))
        X["CS"] = _sb((128, rpi, 4))
        X["CRF"] = _sb((128, rpi))
        X["RSI"] = _sb((128, rpi), I32)
        X["NVB"] = _sb((128, rpi), BF16)
        X["IOTA"] = io
        up = _sb((128, 128), BF16)  # [p < c]: the exclusive prefix over the partitions
        nisa.tensor_scalar(dst=up, data=dd, op0=nl.greater, operand0=0.0)
        X["UP"] = up
        oneh = _sb((rpi, H))
        nisa.memset(dst=oneh, value=1.0)
        eb = _sb((rpi, rpi, H), BF16)  # eb[p, k, h] = [p == k] 1e30 / scale
        for k in range(rpi):
            nisa.tensor_scalar(dst=eb[:, k, :], data=oneh, op0=nl.multiply, operand0=idf[0:rpi, k:k + 1],
                               op1=nl.multiply, operand1=-NEG_INF / scale)
        X["EB"] = eb
        rc = nisa.register_alloc()
        nisa.register_load(rc, nci)

        def body_c(it):
            X["psum"] = _sl_psum(H, LC)
            X["psum_c"] = (nl.ndarray((128, rpi), dtype=F32, buffer=nl.psum),
                           nl.ndarray((128, rpi, NCH, 4), dtype=F32, buffer=nl.psum),
                           nl.ndarray((rpi, 128), dtype=F32, buffer=nl.psum))
            if ss:
                _compact_sb(X, it)
            else:
                _sc_compact(X, it)
            for k in range(X["k_lo"], X["k_hi"]):
                _sc_attend(X, it, k)

        nl.fori_loop(0, rc, body_c)
        if fo:  # the fallback's rows, 1 trip when over: dsa_slots_n_s2._row_nat's tiles, each pair one tile
            TN = NCH * _SL_KP * 128
            XN = dict(H=H, R=R, LC=LC, NCH=NCH, T=TN, rpi=T, q_lat=q_lat, kc=kc, rows_t=srows.reshape((1, T, NCH, 128)),
                      bias=sbias.reshape((1, T, NCH * 128 * _SL_KP)), o=o, lse=lse, want_lse=1, qkw=fq, IB=IB,
                      scale=scale)
            kvn = []
            for _ in range(NCH):
                kvn.append(_sb((128, _SL_KP * R), KD))
            XN["K"] = [kvn, kvn]
            XN["KT"] = []
            if fq:
                kta = _sb((128, LC, TN), BF16)
                XN["KTA"] = [kta, kta]
                br = _sb((1, TN))
                XN["BR"] = [br, br]
                XN["RSC"] = _sb((1, 128))
                nisa.memset(dst=XN["RSC"], value=1.0 / scale)
            else:
                for _ in range(_SL_KTR):
                    XN["KT"].append(_sb((128, LC, 128), BF16))
            # (the device tracer takes only simple loop variables: no tuple targets)
            for spec in (("RS", (128, NCH), I32), ("QR", (H, R), BF16), ("QT", (128, LC, H), BF16),
                         ("S", (H, TN), F32), ("P", (H, TN), BF16), ("OB", (H, R), F32), ("MX", (H, 1), F32),
                         ("NMX", (H, 1), F32), ("DEN", (H, 1), F32), ("RDEN", (H, 1), F32), ("LS", (H, 1), F32),
                         ("PTb", (128, H), BF16)):
                t1 = _sb(spec[1], spec[2])
                XN[spec[0]] = [t1, t1]
            nfi = _sb((1, 1), I32)
            nisa.tensor_copy(dst=nfi, src=ov, engine=VE)
            rf = nisa.register_alloc()
            nisa.register_load(rf, nfi)

            def body_n(it):
                XN["psum"] = _sl_psum(H, LC)
                for k in range(k_lo, k_hi):
                    _sn_row(XN, it, k)

            nl.fori_loop(0, rf, body_n)
        if npg > 1:
            nisa.core_barrier(data=srows, cores=(0, 1))
            nisa.core_barrier(data=sbias, cores=(0, 1))
            nisa.core_barrier(data=o, cores=(0, 1))
            nisa.core_barrier(data=lse, cores=(0, 1))
            nisa.core_barrier(data=over, cores=(0, 1))
        return o, lse, srows, sbias, over

    def _rows_i32(out, B: int, keep: int, RS):
        """Every row's pool rows (out[2], exact integers in fp32) as int32 into RS [B, keep] (this program's own copy)."""
        RF = _sb((B, keep))
        dma_static(dst=RF, src=out.ap(pattern=[[keep, B], [1, keep]], offset=2 * B * keep))
        RI = _sb((B, keep), I32)
        nisa.tensor_copy(dst=RI, src=RF, engine=VE)
        dma_static(dst=RS, src=RI)

    def _snd_write(o, lse, SND, T: int, H: int, R: int, rows):
        """The rows' send rows: row h T + t of SND [H T, R + 16] gets o[0, t, h, :] in columns 0 .. R - 1 and
        lse[0, t, h, 0] in column R, through SBUF (the padding columns untouched)."""
        Cn = R + 16
        for t in rows:
            OBt = _sb((H, R))
            dma_static(dst=OBt, src=o.ap(pattern=[[R, H], [1, R]], offset=t * H * R))
            LSt = _sb((H, 1))
            dma_static(dst=LSt, src=lse.ap(pattern=[[1, H], [1, 1]], offset=t * H))
            dma_static(dst=SND.ap(pattern=[[T * Cn, H], [1, R]], offset=t * Cn), src=OBt)
            dma_static(dst=SND.ap(pattern=[[T * Cn, H], [1, 1]], offset=t * Cn + R), src=LSt)
# -- the kernel source end


REV = (SH.kernel_rev(__file__, "# -- the kernel source", "# -- the kernel source end") ^ DS.REV2 ^ DS.REVA ^ S2.REV)
