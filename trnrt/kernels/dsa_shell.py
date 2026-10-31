"""The DSA decode shell (GLM-5.3-Flash under context parallelism, models/mla.py attention_cp's decode branch): the
decode attention's selection and attend as NKI calls between the context-parallel collectives. The served decode layer
runs both bodies inside one fused kernel (kernels/dsa_sa.py, kernels/dsa_satwf.py); the standalone calls here are their
host-side (emulate) and device forms.

The calls, each from one collective to the next:

    D1  prologue (previous mix, mhc_pre), the projections, this rank's indexer-query heads   -> C1 (indexer q gather)
    D2  the local pools' scores, the local top-keep, its compaction (sel2)                    -> C2, C3 (values, pools)
    D3  the CP merge, the slot rows and bias, the partial attention (attend)                  -> C5, C6 (lse, partials)
    D4  the combine, w_uv, o_proj: the partials P [npg, T, H]                                 -> C7 (the all-reduce)

D2 replaces kernels/dsa_index.py's scores (the whole page bucket, one core at B = 1), kernels/dsa_topk.py's local
selection (one core), and the XLA glue around them (cand, wb, the page groups, the compaction, lval, cpool, pool_row).

What D2 computes, per decode row b (attention_cp's arithmetic, bit for bit):
- scores of the rank's local pools m = 0 .. P - 1 (P = 8 per page of the row's table): dsa_index's per-pool arithmetic
  (bf16 keys transposed exactly on the PE, scored against the row's Hi queries, relu times the head weights, the head
  sum), plus cand (0 for m < nloc, NEG_INF after), so the same fp32 values as dsa_index's kernel;
- the keep best of them (dsa_topk's rule: the threshold, then the lowest pool index among the tied; vis_only), the set
  dsa_long.select_device selects;
- the local list in ascending pool order, as compact() gives it: lval [B, keep] fp32 (the score, NEG_INF past the
  count), cpool [B, keep] fp32 (lp A + rank, the context pool; lp = 0 past the count) and rows [B, keep] int32 (the
  pool's cache row, pool_row(lp) = table[lp // 8] 8 + lp % 8).

The selection is dsa_topk's _select with the true pool index of each tile entry as its position (the tie order), so
the tile's layout does not change the set. Every count is an fp32 integer below 2^24 (exact), every gather in range
(clamped).
"""

from __future__ import annotations

import zlib

import torch

from . import shell as SH

NEG_INF = -1e30  # models/decoder.NEG_INF
D = 128  # index_head_dim (GLM-5.3-Flash)
PPP = 8  # local pools per page row at page 256, kpool 4, CP 8 (models/mla.py attention_cp: ppk)
UNIT = 128  # pages per unit (one per partition)
RF = 18  # a page record: exclusive count, 8 selection bits, 8 scores, page id
LPP = 3  # log2(PPP): the kernels take shift amounts as static ints (the NKI tracer resolves no int methods)
BIG_T = float(2**20)  # dsa_topk.BIG: an untied score's position offset in the tie search
VISIBLE_T = -5e29  # dsa_topk.VISIBLE


def emulate(q: torch.Tensor, w: torch.Tensor, pkc: torch.Tensor, table: torch.Tensor, nloc: torch.Tensor, keep: int,
            A: int, rank: int):
    """D2 in torch, as attention_cp's decode branch computes it (dsa_index's scores over the page rows): q [B, Hi, D]
    the gathered indexer query, w [B, Hi] fp32 the head weights times Di ** -0.5, pkc [Npages, PPP D] the pool-key cache
    as page rows, table [B, NPg] int, nloc [B] the visible local pools, keep, A (CP degree), rank. Returns (lval [B, keep]
    fp32, cpool [B, keep] fp32, rows [B, keep] int32, scores [B, P] fp32)."""
    from ..models import dsa_long
    from . import dsa_index

    B, NPg = table.shape
    P = NPg * PPP
    cand = torch.where(torch.arange(P).view(1, P) < nloc.view(-1, 1), 0.0, NEG_INF)
    sc = dsa_index.scores(q, w.float(), pkc, dsa_index.page_groups(table), cand)
    lp, lc = dsa_long.select_device(sc, keep)
    k = torch.arange(keep).view(1, keep)
    lval = torch.where(k < lc.view(B, 1), sc.gather(1, lp), torch.full_like(sc[:, :keep], NEG_INF))
    cpool = (lp * A + rank).to(torch.float32)
    pg = torch.floor(lp.to(torch.float32) * (1.0 / PPP)).to(torch.int64)
    rows = table.long().gather(1, pg.clamp(max=NPg - 1)) * PPP + (lp - pg * PPP)
    return lval, cpool, rows.to(torch.int32), sc


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    from .dsa_topk import _consts as _topk_consts
    from .dsa_topk import _select as _topk_select
    from .shell import dma_static, exchange, exchange_buffer

    F32, BF16, I32 = nl.float32, nl.bfloat16, nl.int32
    VE = nisa.vector_engine

    def _sb(shape, dtype=None):
        return nl.ndarray(shape, dtype=dtype or F32, buffer=nl.sbuf)



if nki is not None:  # the selection with the tie-search skip (both calls; its own revision, REVT)
    from .dsa_topk import _rlo as _topk_rlo
    from .dsa_topk import _rwidth as _topk_rwidth
    from .dsa_topk import _total as _topk_total
    from .dsa_topk import _wconsts as _topk_wconsts

    def _tie_rounds(X):
        """_select_ts's tie search (the rounds of LIM), at module level: the device tracer takes an inner function only as
        a device loop's body, so the inline form (tskip 0) calls this directly and the loop's body calls it too."""
        for i in range(X["nt"]):
            nisa.memset(dst=X["LIM"][i], value=0.0)
        for r in range((X["nbits"] - 1 + X["rbits"]) // X["rbits"]):
            lo = _topk_rlo(X["nbits"] - 1, X["rbits"], r)
            Jw = (1 << _topk_rwidth(X["nbits"] - 1, X["rbits"], r)) - 1
            for i in range(X["nt"]):
                n = X["NS"][i]
                thr = _sb((n, Jw))
                nisa.tensor_tensor(dst=thr, data1=X["LIM"][i].ap(pattern=[[1, n], [0, Jw]], offset=0),
                                   data2=X["TC"][r][0:n, 0:Jw], op=nl.add, engine=VE)
                lt = _sb((n, Jw, X["W"]))
                nisa.tensor_tensor(dst=lt, data1=X["TJ"][i].ap(pattern=[[X["W"], n], [0, Jw], [1, X["W"]]], offset=0),
                                   data2=thr.ap(pattern=[[Jw, n], [1, Jw], [0, X["W"]]], offset=0), op=nl.less,
                                   engine=VE)
                cp = _sb((n, Jw))
                nisa.tensor_reduce(dst=cp, op=nl.add, data=lt, axis=2)
                tot = _topk_total(cp, n, X["g"], X["G"], Jw)
                okj = _sb((n, Jw))
                dg = _sb((n, 1))
                nisa.tensor_scalar_reduce(dst=okj, data=tot, op0=nl.less, operand0=X["ROOM"][i], reduce_op=nl.add,
                                          reduce_res=dg)
                nisa.scalar_tensor_tensor(dst=X["LIM"][i], data=dg, op0=nl.multiply, operand0=float(1 << lo),
                                          op1=nl.add, operand1=X["LIM"][i])

    def _select_ts(S, R0, NS, W: int, g: int, G, po, wios, keep: int, nbits: int, out, rbits: int, tskip: int = 1):
        """dsa_topk._select (vis_only, one score per entry, rbits > 1) whose tie search runs only when some row of the
        group has more scores tied at its threshold than room: its rounds sit in a device loop of 0 or 1 trips. Without
        it every tied score is selected (LIM = 2^nbits - 1 takes every tied position, below 2^nbits; no untied one, at
        BIG + position >= 2^20 >= 2^nbits), which is the search's answer when the tied fit the room. The same set as
        _select for every input. wios: each row tile's positions (the tie order; one tile per entry of S, so rows with
        different positions interleave their rounds in one call). tskip 0: the tie search always runs, inline (no
        device loop), as _select's."""
        assert rbits > 1 and nbits <= 20
        P = g * W
        nt = len(NS)
        S2, SG, K2, PRE = [], [], [], []
        for i in range(nt):
            n = NS[i]
            scr = _sb((n, W))
            cp = _sb((n, 1))
            nisa.tensor_scalar_reduce(dst=scr, data=S[i], op0=nl.greater_equal, operand0=0.0, reduce_op=nl.add,
                                      reduce_res=cp)
            tot = _topk_total(cp, n, g, G)
            f = _sb((n, 1))
            nisa.tensor_scalar(dst=f, data=tot, op0=nl.less, operand0=float(keep), engine=VE)
            sg = _sb((n, 1))
            nisa.tensor_scalar(dst=sg, data=f, op0=nl.multiply, operand0=-2.0, op1=nl.add, operand1=1.0, engine=VE)
            SG.append(sg)
            k2 = _sb((n, 1))
            nisa.tensor_scalar(dst=k2, data=f, op0=nl.multiply, operand0=float(P + 1 - 2 * keep), op1=nl.add,
                               operand1=float(keep), engine=VE)
            K2.append(k2)
            s2 = _sb((n, W))
            nisa.tensor_scalar(dst=s2, data=S[i], op0=nl.multiply, operand0=sg, engine=VE)
            S2.append(s2)
            pre = _sb((n, 1), I32)
            nisa.memset(dst=pre, value=0)
            PRE.append(pre)
        RC, TC = _topk_wconsts(po, nbits, rbits)
        for r in range((30 + rbits) // rbits):
            lo = _topk_rlo(30, rbits, r)
            Jw = (1 << _topk_rwidth(30, rbits, r)) - 1
            for i in range(nt):
                n = NS[i]
                cand = _sb((n, Jw), I32)
                nisa.tensor_tensor(dst=cand, data1=PRE[i].ap(pattern=[[1, n], [0, Jw]], offset=0),
                                   data2=RC[r][0:n, 0:Jw], op=nl.bitwise_or, engine=VE)
                ge = _sb((n, Jw, W))
                nisa.tensor_tensor(dst=ge, data1=S2[i].ap(pattern=[[W, n], [0, Jw], [1, W]], offset=0),
                                   data2=cand.view(F32).ap(pattern=[[Jw, n], [1, Jw], [0, W]], offset=0),
                                   op=nl.greater_equal, engine=VE)
                cp = _sb((n, Jw))
                nisa.tensor_reduce(dst=cp, op=nl.add, data=ge, axis=2)
                tot = _topk_total(cp, n, g, G, Jw)
                okj = _sb((n, Jw))
                dg = _sb((n, 1))
                nisa.tensor_scalar_reduce(dst=okj, data=tot, op0=nl.greater_equal, operand0=K2[i], reduce_op=nl.add,
                                          reduce_res=dg)
                inc = _sb((n, 1), I32)
                nisa.tensor_scalar(dst=inc, data=dg, op0=nl.multiply, operand0=float(1 << lo), engine=VE)
                nisa.tensor_tensor(dst=PRE[i], data1=PRE[i], data2=inc, op=nl.bitwise_or, engine=VE)
        AB, ROOM, TJ, LIM = [], [], [], []
        need = _sb((1, 1))  # 1 when some row's tied scores exceed its room
        nisa.memset(dst=need, value=0.0)
        for i in range(nt):
            n = NS[i]
            th = _sb((n, 1))
            nisa.tensor_scalar(dst=th, data=PRE[i].view(F32), op0=nl.multiply, operand0=SG[i], engine=VE)
            ab = _sb((n, W))
            abp = _sb((n, 1))
            nisa.tensor_scalar_reduce(dst=ab, data=S[i], op0=nl.greater, operand0=th, reduce_op=nl.add,
                                      reduce_res=abp)
            AB.append(ab)
            abt = _topk_total(abp, n, g, G)
            room = _sb((n, 1))
            nisa.tensor_scalar(dst=room, data=abt, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=float(keep),
                               engine=VE)
            ROOM.append(room)
            tied = _sb((n, W))
            if tskip:
                tp = _sb((n, 1))
                nisa.tensor_scalar_reduce(dst=tied, data=S[i], op0=nl.equal, operand0=th, reduce_op=nl.add,
                                          reduce_res=tp)
                tt = _topk_total(tp, n, g, G)
                ov = _sb((n, 1))
                nisa.tensor_tensor(dst=ov, data1=tt, data2=room, op=nl.greater, engine=VE)
                nisa.tensor_tensor(dst=need, data1=need, data2=ov[0:1, 0:1], op=nl.maximum, engine=VE)
            else:
                nisa.tensor_scalar(dst=tied, data=S[i], op0=nl.equal, operand0=th, engine=VE)
            tj = _sb((n, W))
            nisa.scalar_tensor_tensor(dst=tj, data=tied, op0=nl.multiply, operand0=-BIG_T, op1=nl.add,
                                      operand1=wios[i][0:n])
            nisa.tensor_scalar(dst=tj, data=tj, op0=nl.add, operand0=BIG_T, engine=VE)
            TJ.append(tj)
            lim = _sb((n, 1))
            nisa.memset(dst=lim, value=float((1 << nbits) - 1))
            LIM.append(lim)
        X = {"NS": NS, "TJ": TJ, "ROOM": ROOM, "LIM": LIM, "TC": TC, "W": W, "g": g, "G": G, "nt": nt,
             "nbits": nbits, "rbits": rbits}

        def body(it):
            _tie_rounds(X)

        if tskip:
            needi = _sb((1, 1), I32)
            nisa.tensor_copy(dst=needi, src=need, engine=VE)
            rt = nisa.register_alloc()
            nisa.register_load(rt, needi)
            nl.fori_loop(0, rt, body)
        else:
            _tie_rounds(X)
        sels = []  # out None: each entry's 0 / 1 selection tile, left in SBUF for the caller
        for i in range(nt):
            n = NS[i]
            lp = _sb((n, 1))
            nisa.tensor_tensor(dst=lp, data1=LIM[i], data2=po[0:n], op=nl.subtract, engine=VE)
            sel = _sb((n, W))
            nisa.scalar_tensor_tensor(dst=sel, data=TJ[i], op0=nl.less_equal, operand0=lp, op1=nl.add, operand1=AB[i])
            vm = _sb((n, W))
            nisa.tensor_scalar(dst=vm, data=S[i], op0=nl.greater, operand0=VISIBLE_T, engine=VE)
            nisa.tensor_tensor(dst=sel, data1=sel, data2=vm, op=nl.multiply, engine=VE)
            if out is None:
                sels.append(sel)
            else:
                o = _sb((n, W))
                nisa.tensor_scalar(dst=o, data=sel, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=-NEG_INF,
                                   engine=VE)
                nisa.dma_copy(dst=out.ap(pattern=[[W, n], [1, W]], offset=R0[i] * W), src=o)
        return sels
    def _loop0():
        """A device loop of 0 trips. At grid 2 the two programs' modules must hold the same number of functions
        (neuronx-cc GroupModules: "modZeroFunc.size() == modNFunc->size()"); _select_ts's loop exists only
        on a program that selects rows, so a program with fewer selections pads with these."""
        z = _sb((1, 1), I32)
        nisa.memset(dst=z, value=0)
        rz = nisa.register_alloc()
        nisa.register_load(rz, z)
        pad = _sb((1, 1))

        def body(it):
            nisa.memset(dst=pad, value=0.0)

        nl.fori_loop(0, rz, body)
else:
    _select_ts = None


def _revt() -> int:
    return SH.kernel_rev(__file__, "if nki is not None:  # the selection with the tie-search skip", "    _select_ts = None")


REVT = _revt()


# -- D3a: the CP merge and the slot table ------------------------------------------------------------------------------

def emulate_merge(ev: torch.Tensor, ec: torch.Tensor, rows_sel: torch.Tensor, table: torch.Tensor,
                  positions: torch.Tensor, keep: int, A: int, rank: int, kp: int = 4, ns: int = 640):
    """D3a in torch, written as attention_cp's decode branch serves it after the gathers C2 and C3: ev / ec [A, T, keep]
    every rank's local values and context pools (the gathers, rank order), rows_sel [T, keep] this rank's slot rows,
    table [T, NPg], positions [T]. The merge (dsa_long.cp_merge: the global top keep, the tied by smallest context
    pool), this rank's entries of it, the tail pool, and the slot rows / bias dsa_slots_c attends: (srows [T, ns]
    int32, sbias [T, ns, kp] fp32)."""
    from ..models import dsa_long

    _, T, K = ev.shape
    NPg = table.shape[1]
    Pl = NPg * PPP
    sel = dsa_long.cp_merge(ev.permute(1, 0, 2), ec.permute(1, 0, 2), keep, None, Pl * A, two_k=T >= 2)
    mine = sel[:, rank, :]  # [T, keep]
    npool = dsa_long.npools(positions, kp)
    tail_m = torch.floor(npool.to(torch.float32) * (1.0 / A)).to(torch.int64)
    tail_own = (npool - tail_m * A) == rank
    mm = tail_m.clamp(max=Pl - 1)
    pg = torch.floor(mm.to(torch.float32) * (1.0 / PPP)).to(torch.int64)
    rows_tail = table.long().gather(1, pg.clamp(max=NPg - 1).view(T, 1)).view(T) * PPP + (mm - pg * PPP)
    sl = torch.arange(ns).view(1, ns)
    cl = sl.clamp(max=keep - 1).expand(T, ns)
    srows = torch.where(sl < keep, torch.gather(rows_sel.long(), 1, cl),
                        torch.where(sl == keep, rows_tail.view(T, 1), torch.zeros_like(rows_tail.view(T, 1))))
    t4 = torch.arange(kp).view(1, 1, kp)
    sel_ok = ((sl < keep) & torch.gather(mine, 1, cl)).unsqueeze(-1)
    tail_ok = ((sl == keep) & tail_own.view(T, 1)).unsqueeze(-1) & (npool.view(T, 1, 1) * kp + t4
                                                                     <= positions.view(T, 1, 1))
    sbias = torch.where(sel_ok | tail_ok, 0.0, NEG_INF).to(torch.float32)
    return srows.to(torch.int32), sbias


if nki is not None:  # D3a's kernel helpers (the merge rows the attend runs): their own block and revision (REVM)

    def _merge_init(ev, ec, rows_sel, table, pos, rank, srows, sbias, msel, keep: int, A: int, kp: int, ns: int,
                    nbits: int, rbits: int, lkp: int, la: int, ts: int, il: int = 0, sb: int = 0):
        """The merge's constants and tensors for _merge_rows (D3a and D3b). sb (with il): ohr [128, PR] = [p == q + rank
        PR], the one-hot whose matmul takes this rank's PR partitions of a row's selection straight from SBUF."""
        AA, T, K = ev.shape
        NPg = table.shape[1]
        W = AA * K // 128
        PR = 128 // AA  # partitions per rank
        NT = ns - K  # the tail slot and the padding
        assert AA == A and K == keep and (A * K) % 128 == 0 and K == PR * W and NT >= 1
        rk = _sb((1, 1))
        dma_static(dst=rk, src=rank)
        r16f = _sb((1, 1))
        nisa.tensor_scalar(dst=r16f, data=rk, op0=nl.multiply, operand0=float(PR), engine=VE)
        r16 = _sb((1, 1), I32)
        nisa.tensor_copy(dst=r16, src=r16f, engine=VE)
        pf = _sb((1, T))
        dma_static(dst=pf, src=pos)
        tio = _sb((1, kp), I32)
        nisa.iota(dst=tio, pattern=[[1, kp]], offset=0, channel_multiplier=0)
        tf = _sb((1, kp))
        nisa.tensor_copy(dst=tf, src=tio, engine=VE)
        ohr = None
        if sb and il:
            rkc = _sb((128, 1))
            nisa.dma_copy(dst=rkc, src=rank.ap(pattern=[[0, 128], [1, 1]], offset=0))  # the rank on every partition
            nr = _sb((128, 1))  # -(this rank's first partition)
            nisa.tensor_scalar(dst=nr, data=rkc, op0=nl.multiply, operand0=-float(PR), engine=VE)
            qi = _sb((128, PR), I32)
            nisa.iota(dst=qi, pattern=[[1, PR]], offset=0, channel_multiplier=0)
            qf = _sb((128, PR))  # qf[p, q] = q
            nisa.tensor_copy(dst=qf, src=qi)
            pi = _sb((128, 1), I32)
            nisa.iota(dst=pi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
            pfc = _sb((128, 1))  # p
            nisa.tensor_copy(dst=pfc, src=pi)
            dq = _sb((128, PR))  # q - p
            nisa.tensor_scalar(dst=dq, data=qf, op0=nl.subtract, operand0=pfc)
            ohr = _sb((128, PR))  # [q - p == -rank PR]
            nisa.tensor_scalar(dst=ohr, data=dq, op0=nl.equal, operand0=nr)
        return dict(ev=ev, ec=ec, rows_sel=rows_sel, table=table, srows=srows, sbias=sbias, msel=msel, T=T, K=K,
                    NPg=NPg, W=W, PR=PR, NT=NT, keep=keep, A=A, kp=kp, ns=ns, nbits=nbits, rbits=rbits, lkp=lkp, la=la,
                    lpp=LPP, ts=ts, il=il, rk=rk, r16=r16, pf=pf, tf=tf, ohr=ohr)

    def _merge_rows(X, rows):
        """D3a's work for these rows (module docstring, D3a): the merge of each row's A keep candidates, this rank's
        selection bits, the slot rows and bias into srows / sbias."""
        ev, ec, rows_sel, table, srows, sbias, msel = (X["ev"], X["ec"], X["rows_sel"], X["table"], X["srows"],
                                                       X["sbias"], X["msel"])
        T, K, NPg, W, PR, NT, keep, A, kp, ns = (X["T"], X["K"], X["NPg"], X["W"], X["PR"], X["NT"], X["keep"],
                                                 X["A"], X["kp"], X["ns"])
        nbits, rbits, lkp, la, lpp, ts = X["nbits"], X["rbits"], X["lkp"], X["la"], X["lpp"], X["ts"]
        rk, r16, pf, tf = X["rk"], X["r16"], X["pf"], X["tf"]
        if len(rows) > 0:
            wio, po, G, bits = _topk_consts(W, 128, 7)
            po0 = _sb((128, 1))
            nisa.memset(dst=po0, value=0.0)
            SL, PL, R0, NS = [], [], [], []
            for b in rows:
                s = _sb((128, W))
                c = _sb((128, W))
                if T == 1:  # rank blocks are contiguous: one DMA each
                    dma_static(dst=s, src=ev.ap(pattern=[[W, 128], [1, W]], offset=0))
                    dma_static(dst=c, src=ec.ap(pattern=[[W, 128], [1, W]], offset=0))
                else:
                    for a in range(A):
                        dma_static(dst=s[a * PR:(a + 1) * PR, :], src=ev.ap(pattern=[[W, PR], [1, W]],
                                                                           offset=(a * T + b) * K))
                        dma_static(dst=c[a * PR:(a + 1) * PR, :], src=ec.ap(pattern=[[W, PR], [1, W]],
                                                                           offset=(a * T + b) * K))
                SL.append(s)
                PL.append(c)
                R0.append(b * 128)
                NS.append(128)
            sels = None
            if X["il"]:  # every row in one selection, their rounds interleaved (a position tile per row)
                if X["ohr"] is not None:  # sb: the selections stay in SBUF (no HBM round trip)
                    sels = _select_ts(SL, R0, NS, W, 128, G, po0, PL, keep, nbits, None, rbits, ts)
                else:
                    _select_ts(SL, R0, NS, W, 128, G, po0, PL, keep, nbits, msel, rbits, ts)
            else:
                for i in range(len(SL)):  # one row at a time: _select takes one position tile for its rows
                    if ts:
                        _select_ts([SL[i]], [R0[i]], [NS[i]], W, 128, G, po0, [PL[i]], keep, nbits, msel, rbits)
                    else:
                        _topk_select([SL[i]], [R0[i]], [NS[i]], W, 128, G, po0, PL[i], bits, keep, nbits, 1, 1, 0,
                                     msel, rbits)
            for j in range(len(rows)):
                b = rows[j]
                # this rank's selection bits, slots k = p W + c, and their bias on every token
                bi = _sb((PR, W))  # 0 selected, NEG_INF not
                if sels is not None:  # sb: partitions rank PR .. of row j's 0 / 1 selection, one exact matmul
                    pm = nl.ndarray((PR, W), dtype=F32, buffer=nl.psum)
                    nisa.nc_matmul(dst=pm, stationary=X["ohr"], moving=sels[j], accumulate=False)
                    nisa.tensor_scalar(dst=bi, data=pm, op0=nl.subtract, operand0=1.0, op1=nl.multiply,
                                       operand1=-NEG_INF, engine=VE)
                else:  # the selection written to msel above, read back at this rank's partitions
                    mn = _sb((PR, W))
                    nisa.dma_copy(dst=mn, src=msel.ap(pattern=[[W, PR], [1, W]], offset=b * 128 * W, scalar_offset=r16,
                                                      indirect_dim=0), dge_mode=nisa.dge_mode.hwdge)
                    nisa.tensor_scalar(dst=bi, data=mn, op0=nl.equal, operand0=0.0, op1=nl.subtract, operand1=1.0,
                                       engine=VE)
                    nisa.tensor_scalar(dst=bi, data=bi, op0=nl.multiply, operand0=-NEG_INF, engine=VE)
                b4 = _sb((PR, W, kp))
                for t in range(kp):
                    nisa.tensor_copy(dst=b4[:, :, t], src=bi, engine=VE)
                dma_static(dst=sbias.ap(pattern=[[W * kp, PR], [1, W * kp]], offset=b * ns * kp),
                           src=b4.reshape((PR, W * kp)))
                rs = _sb((1, K), I32)
                dma_static(dst=rs, src=rows_sel.ap(pattern=[[K, 1], [1, K]], offset=b * K))
                dma_static(dst=srows.ap(pattern=[[ns, 1], [1, K]], offset=b * ns), src=rs)
                # the tail pool: npool = (pos + 1) >> lkp, tail_m = npool >> la, own = npool & (A - 1) == rank
                p1 = _sb((1, 1))
                nisa.tensor_scalar(dst=p1, data=pf[:, b:b + 1], op0=nl.add, operand0=1.0, engine=VE)
                p1i = _sb((1, 1), I32)
                nisa.tensor_copy(dst=p1i, src=p1, engine=VE)
                npi = _sb((1, 1), I32)
                nisa.tensor_scalar(dst=npi, data=p1i, op0=nl.right_shift, operand0=lkp, engine=VE)
                tmi = _sb((1, 1), I32)
                nisa.tensor_scalar(dst=tmi, data=npi, op0=nl.right_shift, operand0=la, engine=VE)
                owi = _sb((1, 1), I32)
                nisa.tensor_scalar(dst=owi, data=npi, op0=nl.bitwise_and, operand0=A - 1, engine=VE)
                ow = _sb((1, 1))
                nisa.tensor_copy(dst=ow, src=owi, engine=VE)
                nisa.tensor_tensor(dst=ow, data1=ow, data2=rk, op=nl.equal, engine=VE)
                tmf = _sb((1, 1))
                nisa.tensor_copy(dst=tmf, src=tmi, engine=VE)
                nisa.tensor_scalar(dst=tmf, data=tmf, op0=nl.minimum, operand0=float(NPg * PPP - 1), engine=VE)
                mmi = _sb((1, 1), I32)
                nisa.tensor_copy(dst=mmi, src=tmf, engine=VE)
                pgi = _sb((1, 1), I32)
                nisa.tensor_scalar(dst=pgi, data=mmi, op0=nl.right_shift, operand0=lpp, engine=VE)
                wii = _sb((1, 1), I32)
                nisa.tensor_scalar(dst=wii, data=mmi, op0=nl.bitwise_and, operand0=PPP - 1, engine=VE)
                tv = _sb((1, 1), I32)
                nisa.dma_copy(dst=tv, src=table.reshape((T * NPg, 1)).ap(pattern=[[1, 1], [1, 1]], offset=b * NPg,
                                                                         scalar_offset=pgi, indirect_dim=0),
                              dge_mode=nisa.dge_mode.hwdge)
                tvf = _sb((1, 1))
                nisa.tensor_copy(dst=tvf, src=tv, engine=VE)
                wif = _sb((1, 1))
                nisa.tensor_copy(dst=wif, src=wii, engine=VE)
                rt = _sb((1, NT))
                nisa.memset(dst=rt, value=0.0)
                nisa.scalar_tensor_tensor(dst=rt[:, 0:1], data=tvf, op0=nl.multiply, operand0=float(PPP), op1=nl.add,
                                          operand1=wif)
                rti = _sb((1, NT), I32)
                nisa.tensor_copy(dst=rti, src=rt, engine=VE)
                dma_static(dst=srows.ap(pattern=[[ns, 1], [1, NT]], offset=b * ns + K), src=rti)
                # the tail's tokens npool kp + t visible (<= pos) on its owner, every other slot past keep masked
                npf = _sb((1, 1))
                nisa.tensor_copy(dst=npf, src=npi, engine=VE)
                npk = _sb((1, 1))
                nisa.tensor_scalar(dst=npk, data=npf, op0=nl.multiply, operand0=float(kp), engine=VE)
                tk = _sb((1, kp))  # the tail's tokens npool kp + t
                nisa.tensor_scalar(dst=tk, data=tf, op0=nl.add, operand0=npk, engine=VE)
                ok = _sb((1, kp))
                nisa.tensor_scalar(dst=ok, data=tk, op0=nl.less_equal, operand0=pf[:, b:b + 1], op1=nl.multiply,
                                   operand1=ow, engine=VE)
                bt = _sb((1, NT * kp))
                nisa.memset(dst=bt, value=NEG_INF)
                nisa.tensor_scalar(dst=bt[:, 0:kp], data=ok, op0=nl.subtract, operand0=1.0, op1=nl.multiply,
                                   operand1=-NEG_INF, engine=VE)
                dma_static(dst=sbias.ap(pattern=[[ns * kp, 1], [1, NT * kp]], offset=(b * ns + K) * kp), src=bt)
# -- D3a end


def _revm() -> int:
    from . import dsa_topk

    return (SH.kernel_rev(__file__, "if nki is not None:  # D3a's kernel", "# -- D3a end")
            ^ dsa_topk.REV ^ SH.REV ^ REVT)


REVM = (_revm() ^ zlib.crc32(repr((D, NEG_INF, PPP, RF, LPP, BIG_T, VISIBLE_T)).encode())  # the constants its
        ^ SH.kernel_rev(__file__, "    F32, BF16, I32 = nl.float32", "    def _sb("))  # kernels read, and the aliases


# -- D3b: D3a plus the decode attention --------------------------------------------------------------------------------

def emulate_attend(ev, ec, rows_sel, table, positions, q_all, kc, keep: int, A: int, rank: int, scale: float,
                   kp: int = 4, ns: int = 640):
    """D3b in torch, as attention_cp serves it: emulate_merge's slot rows and bias, then kernels/dsa_slots_c.attend
    (its compacted form when every row's live slots fit 128, else every slot). (o [T, H, R] fp32, lse [T, H] fp32,
    srows [T, ns] int32, sbias [T, ns, kp] fp32)."""
    from . import dsa_slots_c

    srows, sbias = emulate_merge(ev, ec, rows_sel, table, positions, keep, A, rank, kp, ns)
    o, lse = dsa_slots_c.attend(q_all, kc.reshape(kc.shape[0], -1), srows.long(), sbias, scale, lse=True)
    return o, lse, srows, sbias


if nki is not None:  # D3b's kernel: its own block and revision (REVA)
    from .dsa_slots import KP as _SL_KP
    from .dsa_slots import _pair as _sl_pair
    from .dsa_slots import _psum as _sl_psum
    from .dsa_slots_c import C as _SC_C
    from .dsa_slots_c import _attend_c as _sc_attend
    from .dsa_slots_c import _compact as _sc_compact
    from .dsa_slots_c import _gather as _sc_gather
    from .dsa_slots import KTR as _SL_KTR
    from .dsa_slots_n_s2 import _row_nat as _sn_row

    def _select_st(SS, PS, nt: int, W: int, G, keep: int, nbits: int, rbits: int):
        """_select_ts (tskip 0, g 128, piece offsets 0) over the program's nt rows at once, as sel2's _select2s: SS
        [128, nt, W] row i's merge candidates' scores at [:, i, :], PS [128, nt, W] their context pools (the tie order,
        each row its own). Each op runs once over every row where _select_ts issues it per row tile; the per-row scalars
        become [128, nt] tiles read through zero-stride access patterns. The element arithmetic is _select_ts's (integer
        counts, bit-pattern thresholds, x * +-1), so the selection is its bit for bit. The counts (each <= W a
        partition) go to the PE in bf16 when W <= 256, where every count is exact (their fp32 PSUM sums <= 128 W).
        Returns sel [128, nt, W] fp32 in SBUF: 1 selected, 0 not."""
        n = 128
        P = n * W
        J = (1 << rbits) - 1
        CD = BF16 if W <= 256 else F32
        Gc = _sb((n, n), CD)
        nisa.tensor_copy(dst=Gc, src=G[0:n, 0:n], engine=VE)
        po0 = _sb((n, 1))
        nisa.memset(dst=po0, value=0.0)
        # 1. sign
        ge0 = _sb((n, nt, W))
        nisa.tensor_scalar(dst=ge0, data=SS, op0=nl.greater_equal, operand0=0.0, engine=VE)
        cp = _sb((n, nt), CD)
        nisa.tensor_reduce(dst=cp, op=nl.add, data=ge0, axis=2)
        tot = nl.ndarray((n, nt), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=tot, stationary=Gc, moving=cp)
        f = _sb((n, nt))
        nisa.tensor_scalar(dst=f, data=tot, op0=nl.less, operand0=float(keep), engine=VE)
        sg = _sb((n, nt))
        nisa.tensor_scalar(dst=sg, data=f, op0=nl.multiply, operand0=-2.0, op1=nl.add, operand1=1.0, engine=VE)
        k2 = _sb((n, nt))
        nisa.tensor_scalar(dst=k2, data=f, op0=nl.multiply, operand0=float(P + 1 - 2 * keep), op1=nl.add,
                           operand1=float(keep), engine=VE)
        s2 = _sb((n, nt, W))
        nisa.tensor_tensor(dst=s2, data1=SS, data2=sg.ap(pattern=[[nt, n], [1, nt], [0, W]], offset=0),
                           op=nl.multiply, engine=VE)
        pre = _sb((n, nt), I32)
        nisa.memset(dst=pre, value=0)
        # 2. radix over s2's bit pattern, rbits a round
        RC, TC = _topk_wconsts(po0, nbits, rbits)
        for r in range((30 + rbits) // rbits):
            lo = _topk_rlo(30, rbits, r)
            Jw = (1 << _topk_rwidth(30, rbits, r)) - 1
            cand = _sb((n, nt, Jw), I32)
            nisa.tensor_tensor(dst=cand, data1=pre.ap(pattern=[[nt, n], [1, nt], [0, Jw]], offset=0),
                               data2=RC[r].ap(pattern=[[J, n], [0, nt], [1, Jw]], offset=0), op=nl.bitwise_or,
                               engine=VE)
            ge = _sb((n, nt, Jw, W))
            nisa.tensor_tensor(dst=ge, data1=s2.ap(pattern=[[nt * W, n], [W, nt], [0, Jw], [1, W]], offset=0),
                               data2=cand.view(F32).ap(pattern=[[nt * Jw, n], [Jw, nt], [1, Jw], [0, W]], offset=0),
                               op=nl.greater_equal, engine=VE)
            cpj = _sb((n, nt, Jw), CD)
            nisa.tensor_reduce(dst=cpj, op=nl.add, data=ge, axis=3)
            totj = nl.ndarray((n, nt * Jw), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(dst=totj, stationary=Gc, moving=cpj.reshape((n, nt * Jw)))
            okj = _sb((n, nt, Jw))
            nisa.tensor_tensor(dst=okj.reshape((n, nt * Jw)), data1=totj,
                               data2=k2.ap(pattern=[[nt, n], [1, nt], [0, Jw]], offset=0), op=nl.greater_equal,
                               engine=VE)
            dg = _sb((n, nt))
            nisa.tensor_reduce(dst=dg, op=nl.add, data=okj, axis=2)
            inc = _sb((n, nt), I32)
            nisa.tensor_scalar(dst=inc, data=dg, op0=nl.multiply, operand0=float(1 << lo), engine=VE)
            nisa.tensor_tensor(dst=pre, data1=pre, data2=inc, op=nl.bitwise_or, engine=VE)
        # 3. above, room, ties (the tie search always runs: _select_ts at tskip 0)
        th = _sb((n, nt))
        nisa.tensor_tensor(dst=th, data1=pre.view(F32), data2=sg, op=nl.multiply, engine=VE)
        thw = th.ap(pattern=[[nt, n], [1, nt], [0, W]], offset=0)
        ab = _sb((n, nt, W))
        nisa.tensor_tensor(dst=ab, data1=SS, data2=thw, op=nl.greater, engine=VE)
        abp = _sb((n, nt), CD)
        nisa.tensor_reduce(dst=abp, op=nl.add, data=ab, axis=2)
        abt = nl.ndarray((n, nt), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=abt, stationary=Gc, moving=abp)
        room = _sb((n, nt))
        nisa.tensor_scalar(dst=room, data=abt, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=float(keep),
                           engine=VE)
        tied = _sb((n, nt, W))
        nisa.tensor_tensor(dst=tied, data1=SS, data2=thw, op=nl.equal, engine=VE)
        tj = _sb((n, nt, W))  # the context pool where tied, + BIG elsewhere
        nisa.scalar_tensor_tensor(dst=tj, data=tied, op0=nl.multiply, operand0=-BIG_T, op1=nl.add, operand1=PS)
        nisa.tensor_scalar(dst=tj, data=tj, op0=nl.add, operand0=BIG_T, engine=VE)
        lim = _sb((n, nt))
        nisa.memset(dst=lim, value=0.0)
        for r in range((nbits - 1 + rbits) // rbits):
            lo = _topk_rlo(nbits - 1, rbits, r)
            Jw = (1 << _topk_rwidth(nbits - 1, rbits, r)) - 1
            thr = _sb((n, nt, Jw))
            nisa.tensor_tensor(dst=thr, data1=lim.ap(pattern=[[nt, n], [1, nt], [0, Jw]], offset=0),
                               data2=TC[r].ap(pattern=[[J, n], [0, nt], [1, Jw]], offset=0), op=nl.add, engine=VE)
            lt = _sb((n, nt, Jw, W))
            nisa.tensor_tensor(dst=lt, data1=tj.ap(pattern=[[nt * W, n], [W, nt], [0, Jw], [1, W]], offset=0),
                               data2=thr.ap(pattern=[[nt * Jw, n], [Jw, nt], [1, Jw], [0, W]], offset=0), op=nl.less,
                               engine=VE)
            cpj = _sb((n, nt, Jw), CD)
            nisa.tensor_reduce(dst=cpj, op=nl.add, data=lt, axis=3)
            totj = nl.ndarray((n, nt * Jw), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(dst=totj, stationary=Gc, moving=cpj.reshape((n, nt * Jw)))
            okj = _sb((n, nt, Jw))
            nisa.tensor_tensor(dst=okj.reshape((n, nt * Jw)), data1=totj,
                               data2=room.ap(pattern=[[nt, n], [1, nt], [0, Jw]], offset=0), op=nl.less, engine=VE)
            dg = _sb((n, nt))
            nisa.tensor_reduce(dst=dg, op=nl.add, data=okj, axis=2)
            nisa.scalar_tensor_tensor(dst=lim, data=dg, op0=nl.multiply, operand0=float(1 << lo), op1=nl.add,
                                      operand1=lim)
        # 4. sel (piece offsets 0: lp = lim)
        sel = _sb((n, nt, W))
        nisa.tensor_tensor(dst=sel, data1=tj, data2=lim.ap(pattern=[[nt, n], [1, nt], [0, W]], offset=0),
                           op=nl.less_equal, engine=VE)
        nisa.tensor_tensor(dst=sel, data1=sel, data2=ab, op=nl.add, engine=VE)
        vm = _sb((n, nt, W))
        nisa.tensor_scalar(dst=vm, data=SS, op0=nl.greater, operand0=VISIBLE_T, engine=VE)
        nisa.tensor_tensor(dst=sel, data1=sel, data2=vm, op=nl.multiply, engine=VE)
        return sel

    def _tail_rows_st(X, rows):
        """_merge_rows' per-row tail block for the program's rows at once, row b0 + j on partition j: the tail pool
        npool = (pos + 1) >> lkp, its page and slot by integer shifts, the table's page row by ONE indirect DMA (a
        vector offset a partition: b NPg + page), every row's tail slot row and bias by one DMA each. Issued before the
        selection (it needs none of it). Returns tv4 [1, nt, 4] fp32 on partition 0: each row's tail slot as the slot
        table holds it (its pool row's digits hi, mid, lo and its visible tokens), moved there by one exact masked
        matmul (one 0 / 1 product a sum)."""
        table, srows, sbias = X["table"], X["srows"], X["sbias"]
        T, K, NPg, NT, A, kp, ns = X["T"], X["K"], X["NPg"], X["NT"], X["A"], X["kp"], X["ns"]
        lkp, la, lpp = X["lkp"], X["la"], X["lpp"]
        nt = len(rows)
        b0 = rows[0]
        pc = _sb((nt, 1))  # the positions, one row a partition
        dma_static(dst=pc, src=X["pos_in"].ap(pattern=[[1, nt], [1, 1]], offset=b0))
        rkc = _sb((nt, 1))
        nisa.dma_copy(dst=rkc, src=X["rank_in"].ap(pattern=[[0, nt], [1, 1]], offset=0))  # the rank on every partition
        p1 = _sb((nt, 1))
        nisa.tensor_scalar(dst=p1, data=pc, op0=nl.add, operand0=1.0, engine=VE)
        p1i = _sb((nt, 1), I32)
        nisa.tensor_copy(dst=p1i, src=p1, engine=VE)
        npi = _sb((nt, 1), I32)
        nisa.tensor_scalar(dst=npi, data=p1i, op0=nl.right_shift, operand0=lkp, engine=VE)
        tmi = _sb((nt, 1), I32)
        nisa.tensor_scalar(dst=tmi, data=npi, op0=nl.right_shift, operand0=la, engine=VE)
        owi = _sb((nt, 1), I32)
        nisa.tensor_scalar(dst=owi, data=npi, op0=nl.bitwise_and, operand0=A - 1, engine=VE)
        ow = _sb((nt, 1))
        nisa.tensor_copy(dst=ow, src=owi, engine=VE)
        nisa.tensor_tensor(dst=ow, data1=ow, data2=rkc, op=nl.equal, engine=VE)
        tmf = _sb((nt, 1))
        nisa.tensor_copy(dst=tmf, src=tmi, engine=VE)
        nisa.tensor_scalar(dst=tmf, data=tmf, op0=nl.minimum, operand0=float(NPg * PPP - 1), engine=VE)
        mmi = _sb((nt, 1), I32)
        nisa.tensor_copy(dst=mmi, src=tmf, engine=VE)
        pgi = _sb((nt, 1), I32)
        nisa.tensor_scalar(dst=pgi, data=mmi, op0=nl.right_shift, operand0=lpp, engine=VE)
        wii = _sb((nt, 1), I32)
        nisa.tensor_scalar(dst=wii, data=mmi, op0=nl.bitwise_and, operand0=PPP - 1, engine=VE)
        rb = _sb((nt, 1), I32)  # row b's table base, b NPg
        nisa.iota(dst=rb, pattern=[[0, 1]], offset=b0 * NPg, channel_multiplier=NPg)
        idx = _sb((nt, 1), I32)
        nisa.tensor_tensor(dst=idx, data1=pgi, data2=rb, op=nl.add, engine=VE)
        tv = _sb((nt, 1), I32)
        nisa.dma_copy(dst=tv, src=table.reshape((T * NPg, 1)).ap(pattern=[[1, nt], [1, 1]], offset=0,
                                                                 vector_offset=idx.ap(pattern=[[1, nt], [1, 1]],
                                                                                      offset=0),
                                                                 indirect_dim=0))
        tvf = _sb((nt, 1))
        nisa.tensor_copy(dst=tvf, src=tv, engine=VE)
        wif = _sb((nt, 1))
        nisa.tensor_copy(dst=wif, src=wii, engine=VE)
        rt = _sb((nt, NT))
        nisa.memset(dst=rt, value=0.0)
        nisa.scalar_tensor_tensor(dst=rt[:, 0:1], data=tvf, op0=nl.multiply, operand0=float(PPP), op1=nl.add,
                                  operand1=wif)
        rti = _sb((nt, NT), I32)
        nisa.tensor_copy(dst=rti, src=rt, engine=VE)
        dma_static(dst=srows.ap(pattern=[[ns, nt], [1, NT]], offset=b0 * ns + K), src=rti)
        # the tail's tokens npool kp + t visible (<= pos) on its owner, every other slot past keep masked
        npf = _sb((nt, 1))
        nisa.tensor_copy(dst=npf, src=npi, engine=VE)
        npk = _sb((nt, 1))
        nisa.tensor_scalar(dst=npk, data=npf, op0=nl.multiply, operand0=float(kp), engine=VE)
        tki = _sb((nt, kp), I32)
        nisa.iota(dst=tki, pattern=[[1, kp]], offset=0, channel_multiplier=0)
        tk = _sb((nt, kp))
        nisa.tensor_copy(dst=tk, src=tki, engine=VE)
        nisa.tensor_scalar(dst=tk, data=tk, op0=nl.add, operand0=npk, engine=VE)
        ok = _sb((nt, kp))
        nisa.tensor_scalar(dst=ok, data=tk, op0=nl.less_equal, operand0=pc, op1=nl.multiply, operand1=ow, engine=VE)
        bt = _sb((nt, NT * kp))
        nisa.memset(dst=bt, value=NEG_INF)
        nisa.tensor_scalar(dst=bt[:, 0:kp], data=ok, op0=nl.subtract, operand0=1.0, op1=nl.multiply,
                           operand1=-NEG_INF, engine=VE)
        dma_static(dst=sbias.ap(pattern=[[ns * kp, nt], [1, NT * kp]], offset=(b0 * ns + K) * kp), src=bt)
        # the slot table's tail entries (digits of the pool row, visible tokens), moved to partition 0
        vals = _sb((nt, 4))
        dgi = _sb((nt, 1), I32)
        nisa.tensor_scalar(dst=dgi, data=rti[:, 0:1], op0=nl.right_shift, operand0=16, engine=VE)
        nisa.tensor_copy(dst=vals[:, 0:1], src=dgi, engine=VE)
        nisa.tensor_scalar(dst=dgi, data=rti[:, 0:1], op0=nl.right_shift, operand0=8, op1=nl.bitwise_and,
                           operand1=255, engine=VE)
        nisa.tensor_copy(dst=vals[:, 1:2], src=dgi, engine=VE)
        nisa.tensor_scalar(dst=dgi, data=rti[:, 0:1], op0=nl.bitwise_and, operand0=255, engine=VE)
        nisa.tensor_copy(dst=vals[:, 2:3], src=dgi, engine=VE)
        nisa.tensor_reduce(dst=vals[:, 3:4], op=nl.add, data=ok, axis=1)
        jj = _sb((nt, nt), I32)
        nisa.iota(dst=jj, pattern=[[1, nt]], offset=0, channel_multiplier=0)
        jjf = _sb((nt, nt))
        nisa.tensor_copy(dst=jjf, src=jj, engine=VE)
        pj = _sb((nt, 1), I32)
        nisa.iota(dst=pj, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        pjf = _sb((nt, 1))
        nisa.tensor_copy(dst=pjf, src=pj, engine=VE)
        idn = _sb((nt, nt))  # [j == j']
        nisa.tensor_scalar(dst=idn, data=jjf, op0=nl.equal, operand0=pjf, engine=VE)
        E = _sb((nt, nt, 4))  # [j, j', d] = [j == j'] vals[j, d]
        nisa.tensor_tensor(dst=E, data1=idn.ap(pattern=[[nt, nt], [1, nt], [0, 4]], offset=0),
                           data2=vals.ap(pattern=[[4, nt], [0, nt], [1, 4]], offset=0), op=nl.multiply, engine=VE)
        on = _sb((nt, 1))
        nisa.memset(dst=on, value=1.0)
        pv = nl.ndarray((1, nt * 4), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=pv, stationary=on, moving=E.reshape((nt, nt * 4)), accumulate=False)
        tv4 = _sb((1, nt, 4))
        nisa.tensor_copy(dst=tv4.reshape((1, nt * 4)), src=pv, engine=VE)
        return tv4

    def _merge_rows_st(X, rows, tl: int = 0):
        """_merge_rows (il 1, sb 1, ts 0) with the program's rows stacked: their candidates loaded as one [128, nt, W]
        tile (A DMAs each for the scores and the context pools, every row of a rank block in one: the rows are
        consecutive), selected by _select_st in one pass, this rank's partitions of every row's selection by one exact
        matmul, the selected slots' bias and rows by one DMA each. The tail pool's slot as _merge_rows writes it. The
        same srows and sbias bits. Returns its SBUF tiles for _slot_live_sb: sel [128, nt, W] (0 / 1) and each row's
        tail (rti [1, NT] int32: the tail slot's pool row at 0; ok [1, kp]: its visible tokens). tl 1: the tail block
        for every row at once, first (_tail_rows_st), and tv4 in place of the rows' tails."""
        ev, ec, rows_sel, table, srows, sbias = (X["ev"], X["ec"], X["rows_sel"], X["table"], X["srows"], X["sbias"])
        T, K, NPg, W, PR, NT, keep, A, kp, ns = (X["T"], X["K"], X["NPg"], X["W"], X["PR"], X["NT"], X["keep"],
                                                 X["A"], X["kp"], X["ns"])
        nbits, rbits, lkp, la, lpp = X["nbits"], X["rbits"], X["lkp"], X["la"], X["lpp"]
        rk, pf, tf = X["rk"], X["pf"], X["tf"]
        nt = len(rows)
        if nt == 0:
            return None
        b0 = rows[0]
        for j in range(nt):
            assert rows[j] == b0 + j
        tv4 = _tail_rows_st(X, rows) if tl else None
        wio, po, G, bits = _topk_consts(W, 128, 7)
        SS = _sb((128, nt, W))
        PS = _sb((128, nt, W))
        if T == 1:
            dma_static(dst=SS[:, 0, :], src=ev.ap(pattern=[[W, 128], [1, W]], offset=0))
            dma_static(dst=PS[:, 0, :], src=ec.ap(pattern=[[W, 128], [1, W]], offset=0))
        else:
            for a in range(A):
                dma_static(dst=SS[a * PR:(a + 1) * PR, :, :],
                           src=ev.ap(pattern=[[W, PR], [K, nt], [1, W]], offset=(a * T + b0) * K))
                dma_static(dst=PS[a * PR:(a + 1) * PR, :, :],
                           src=ec.ap(pattern=[[W, PR], [K, nt], [1, W]], offset=(a * T + b0) * K))
        sel = _select_st(SS, PS, nt, W, G, keep, nbits, rbits)
        # this rank's selection bits, slots k = p W + c, and their bias on every token: every row in one matmul
        pm = nl.ndarray((PR, nt * W), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=pm, stationary=X["ohr"], moving=sel.reshape((128, nt * W)), accumulate=False)
        bi = _sb((PR, nt * W))  # 0 selected, NEG_INF not
        nisa.tensor_scalar(dst=bi, data=pm, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=-NEG_INF,
                           engine=VE)
        b4 = _sb((PR, nt * W, kp))
        nisa.tensor_copy(dst=b4, src=bi.ap(pattern=[[nt * W, PR], [1, nt * W], [0, kp]], offset=0), engine=VE)
        dma_static(dst=sbias.ap(pattern=[[W * kp, PR], [ns * kp, nt], [1, W * kp]], offset=b0 * ns * kp),
                   src=b4.reshape((PR, nt, W * kp)))
        rs = _sb((1, nt, K), I32)
        dma_static(dst=rs, src=rows_sel.ap(pattern=[[nt * K, 1], [K, nt], [1, K]], offset=b0 * K))
        dma_static(dst=srows.ap(pattern=[[nt * ns, 1], [ns, nt], [1, K]], offset=b0 * ns), src=rs)
        tails = []
        if tl:
            return {"sel": sel, "tails": tails, "tv4": tv4}
        for b in rows:
            # the tail pool: npool = (pos + 1) >> lkp, tail_m = npool >> la, own = npool & (A - 1) == rank
            p1 = _sb((1, 1))
            nisa.tensor_scalar(dst=p1, data=pf[:, b:b + 1], op0=nl.add, operand0=1.0, engine=VE)
            p1i = _sb((1, 1), I32)
            nisa.tensor_copy(dst=p1i, src=p1, engine=VE)
            npi = _sb((1, 1), I32)
            nisa.tensor_scalar(dst=npi, data=p1i, op0=nl.right_shift, operand0=lkp, engine=VE)
            tmi = _sb((1, 1), I32)
            nisa.tensor_scalar(dst=tmi, data=npi, op0=nl.right_shift, operand0=la, engine=VE)
            owi = _sb((1, 1), I32)
            nisa.tensor_scalar(dst=owi, data=npi, op0=nl.bitwise_and, operand0=A - 1, engine=VE)
            ow = _sb((1, 1))
            nisa.tensor_copy(dst=ow, src=owi, engine=VE)
            nisa.tensor_tensor(dst=ow, data1=ow, data2=rk, op=nl.equal, engine=VE)
            tmf = _sb((1, 1))
            nisa.tensor_copy(dst=tmf, src=tmi, engine=VE)
            nisa.tensor_scalar(dst=tmf, data=tmf, op0=nl.minimum, operand0=float(NPg * PPP - 1), engine=VE)
            mmi = _sb((1, 1), I32)
            nisa.tensor_copy(dst=mmi, src=tmf, engine=VE)
            pgi = _sb((1, 1), I32)
            nisa.tensor_scalar(dst=pgi, data=mmi, op0=nl.right_shift, operand0=lpp, engine=VE)
            wii = _sb((1, 1), I32)
            nisa.tensor_scalar(dst=wii, data=mmi, op0=nl.bitwise_and, operand0=PPP - 1, engine=VE)
            tv = _sb((1, 1), I32)
            nisa.dma_copy(dst=tv, src=table.reshape((T * NPg, 1)).ap(pattern=[[1, 1], [1, 1]], offset=b * NPg,
                                                                     scalar_offset=pgi, indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge)
            tvf = _sb((1, 1))
            nisa.tensor_copy(dst=tvf, src=tv, engine=VE)
            wif = _sb((1, 1))
            nisa.tensor_copy(dst=wif, src=wii, engine=VE)
            rt = _sb((1, NT))
            nisa.memset(dst=rt, value=0.0)
            nisa.scalar_tensor_tensor(dst=rt[:, 0:1], data=tvf, op0=nl.multiply, operand0=float(PPP), op1=nl.add,
                                      operand1=wif)
            rti = _sb((1, NT), I32)
            nisa.tensor_copy(dst=rti, src=rt, engine=VE)
            dma_static(dst=srows.ap(pattern=[[ns, 1], [1, NT]], offset=b * ns + K), src=rti)
            # the tail's tokens npool kp + t visible (<= pos) on its owner, every other slot past keep masked
            npf = _sb((1, 1))
            nisa.tensor_copy(dst=npf, src=npi, engine=VE)
            npk = _sb((1, 1))
            nisa.tensor_scalar(dst=npk, data=npf, op0=nl.multiply, operand0=float(kp), engine=VE)
            tk = _sb((1, kp))
            nisa.tensor_scalar(dst=tk, data=tf, op0=nl.add, operand0=npk, engine=VE)
            ok = _sb((1, kp))
            nisa.tensor_scalar(dst=ok, data=tk, op0=nl.less_equal, operand0=pf[:, b:b + 1], op1=nl.multiply,
                               operand1=ow, engine=VE)
            bt = _sb((1, NT * kp))
            nisa.memset(dst=bt, value=NEG_INF)
            nisa.tensor_scalar(dst=bt[:, 0:kp], data=ok, op0=nl.subtract, operand0=1.0, op1=nl.multiply,
                               operand1=-NEG_INF, engine=VE)
            dma_static(dst=sbias.ap(pattern=[[ns * kp, 1], [1, NT * kp]], offset=(b * ns + K) * kp), src=bt)
            tails.append((rti, ok))
        return {"sel": sel, "tails": tails, "tv4": None}

    def _compact_sb(X, it):
        """dsa_slots_c._compact (steps 2-4, verbatim) over the slot table already in SBUF (X["CPT"], built by
        _slot_rows_sb / _slot_live_sb): no cparts round trip through HBM. it is unused (one iteration)."""
        NCH, rpi, R = X["NCH"], X["rpi"], X["R"]
        PX, PCM, PNV = X["psum_c"]
        cp = X["CPT"]  # (hi, mid, lo, nv) of row k's slot ch 128 + p at [p, k, ch, :]
        # 2. live slots, their inclusive prefix over a partition's chunks, the partitions' exclusive prefix, places
        m = X["M"]
        nisa.tensor_scalar(dst=m, data=cp[:, :, :, 3], op0=nl.greater, operand0=0.0)
        incl = X["INCL"]
        nisa.tensor_copy(dst=incl[:, :, 0], src=m[:, :, 0])
        for ch in range(1, NCH):
            nisa.tensor_tensor(dst=incl[:, :, ch], data1=incl[:, :, ch - 1], data2=m[:, :, ch], op=nl.add)
        cnb = X["CNB"]
        nisa.tensor_copy(dst=cnb, src=incl[:, :, NCH - 1])
        nisa.nc_matmul(dst=PX, stationary=X["UP"], moving=cnb, accumulate=False)  # every row's in one matmul
        pxs = X["PXS"]
        nisa.tensor_copy(dst=pxs, src=PX)
        pl_ = X["PL"]  # place = excl + incl - m for a live slot, excl + incl + 4096 - 4097 m in general
        nisa.scalar_tensor_tensor(dst=pl_, data=m, op0=nl.multiply, operand0=-4097.0, op1=nl.add, operand1=incl)
        for k in range(rpi):
            nisa.tensor_scalar(dst=pl_[:, k, :], data=pl_[:, k, :], op0=nl.add, operand0=pxs[:, k:k + 1], op1=nl.add,
                               operand1=4096.0)
        # 3. per (row, chunk) the one-hot and its partial of the compacted (hi, mid, lo, nv), each into its own PSUM
        # region with accumulate=False (no accumulation chains: several chains in one PSUM tensor lose partials on
        # trn1), then the chunks summed on the vector engine (one nonzero term per entry: exact)
        oh = X["OH"]
        for k in range(rpi):
            for ch in range(NCH):
                j = k * NCH + ch
                o_ = oh[:, j % 4, :]
                nisa.tensor_scalar(dst=o_, data=X["IOTA"], op0=nl.equal, operand0=pl_[:, k, ch:ch + 1])
                nisa.nc_matmul(dst=PCM[:, k, ch, :], stationary=o_, moving=cp[:, k, ch, :], accumulate=False)
        cm = X["CM"]
        nisa.tensor_copy(dst=cm, src=PCM)
        cs = X["CS"]
        nisa.tensor_tensor(dst=cs, data1=cm[:, :, 0, :], data2=cm[:, :, 1, :], op=nl.add)
        for ch in range(2, NCH):
            nisa.tensor_tensor(dst=cs, data1=cs, data2=cm[:, :, ch, :], op=nl.add)
        crf = X["CRF"]
        nisa.scalar_tensor_tensor(dst=crf, data=cs[:, :, 0], op0=nl.multiply, operand0=256.0, op1=nl.add,
                                  operand1=cs[:, :, 1])
        nisa.scalar_tensor_tensor(dst=crf, data=crf, op0=nl.multiply, operand0=256.0, op1=nl.add, operand1=cs[:, :, 2])
        rs = X["RSI"]
        nisa.tensor_copy(dst=rs, src=crf)
        for k in range(X["k_lo"], X["k_hi"]):  # every row's gather in flight before the first row attends
            _sc_gather(X, k)
        # 4. nv^T of every row by one PE transpose, the bias rows (0 visible, -1 masked)
        nvb = X["NVB"]
        nisa.tensor_copy(dst=nvb, src=cs[:, :, 3])
        nisa.nc_matmul(dst=PNV, stationary=nvb, moving=X["IB"], accumulate=False)
        brs = X["BRS"]
        for t in range(_SL_KP):
            nisa.tensor_scalar(dst=brs[:, t * 128:(t + 1) * 128], data=PNV, op0=nl.greater, operand0=float(t),
                               op1=nl.subtract, operand1=1.0)

    def _slot_rows_sb(X, rows, cpt):
        """The slot table's pool rows for these rows' K candidate slots, before the selection (they need none of it):
        slot ch 128 + p's pool row rows_sel[b, ch 128 + p] read from rows_sel (an input: no read after a write) in the
        table's layout, one DMA for every row, as its 8-bit digits hi, mid, lo (>> 16, >> 8 & 255, & 255: exact below
        2^24) into cpt[:, b, ch, 0..2], ch < K / 128."""
        rows_sel, K = X["rows_sel"], X["K"]
        nt = len(rows)
        if nt == 0:
            return
        b0 = rows[0]
        KC = K // 128
        assert K % 128 == 0
        ri = _sb((128, nt, KC), I32)
        dma_static(dst=ri, src=rows_sel.ap(pattern=[[1, 128], [K, nt], [128, KC]], offset=b0 * K))
        dg = _sb((128, nt, KC), I32)
        df = _sb((128, nt, KC))
        nisa.tensor_scalar(dst=dg, data=ri, op0=nl.right_shift, operand0=16, engine=VE)
        nisa.tensor_copy(dst=df, src=dg, engine=VE)
        nisa.tensor_copy(dst=cpt[:, b0:b0 + nt, 0:KC, 0], src=df, engine=VE)
        nisa.tensor_scalar(dst=dg, data=ri, op0=nl.right_shift, operand0=8, op1=nl.bitwise_and, operand1=255,
                           engine=VE)
        nisa.tensor_copy(dst=df, src=dg, engine=VE)
        nisa.tensor_copy(dst=cpt[:, b0:b0 + nt, 0:KC, 1], src=df, engine=VE)
        nisa.tensor_scalar(dst=dg, data=ri, op0=nl.bitwise_and, operand0=255, engine=VE)
        nisa.tensor_copy(dst=df, src=dg, engine=VE)
        nisa.tensor_copy(dst=cpt[:, b0:b0 + nt, 0:KC, 2], src=df, engine=VE)

    def _slot_live_sb(X, MS, rows, cpt, cnt):
        """The slot table's live tokens and the tail slot, after the selection, from the merge's SBUF tiles (MS:
        _merge_rows_st's sel and tail scalars), no srows / sbias read back: a K candidate slot q W + c (this rank's
        partition q, column c) holds kp live tokens when selected. Its bit is re-laid to the table's [p, ch] (p = (q %
        QW) W + c, ch = q // QW, QW = 128 / W) by one matmul a row (sel^T against the rank's one-hot ohr: the rank pick
        and the transpose, [c, q]) and QW accumulating placement matmuls (stationary [c, p] = [p == r W + c]), every
        product 0 / 1 and one nonzero term a sum: exact. The tail slot K (p 0, ch K / 128): its pool row's digits and
        its visible tokens from the tail scalars on partition 0. cnt[:, b]: the live slots on each partition, summed
        over the chunks. The same table as read back from srows / sbias."""
        W, PR, kp, K, ns = X["W"], X["PR"], X["kp"], X["K"], X["ns"]
        nt = len(rows)
        if nt == 0:
            return
        b0 = rows[0]
        NCH = ns // 128
        KC = K // 128
        QW = 128 // W
        assert 128 % W == 0 and PR == QW * KC and K < ns
        sel = MS["sel"]
        pmT = nl.ndarray((W, nt * PR), dtype=F32, buffer=nl.psum)  # [c, j PR + q] = this rank's bit of slot q W + c
        for j in range(nt):
            nisa.nc_matmul(dst=pmT[:, j * PR:(j + 1) * PR], stationary=sel[:, j, :], moving=X["ohr"],
                           accumulate=False)
        st_ = _sb((W, nt * PR))
        nisa.tensor_copy(dst=st_, src=pmT, engine=VE)
        ji = _sb((W, 128), I32)
        nisa.iota(dst=ji, pattern=[[1, 128]], offset=0, channel_multiplier=0)
        jf = _sb((W, 128))
        nisa.tensor_copy(dst=jf, src=ji, engine=VE)
        ci = _sb((W, 1), I32)
        nisa.iota(dst=ci, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        cf = _sb((W, 1))
        nisa.tensor_copy(dst=cf, src=ci, engine=VE)
        dpc = _sb((W, 128))  # p - c
        nisa.tensor_scalar(dst=dpc, data=jf, op0=nl.subtract, operand0=cf, engine=VE)
        PO = nl.ndarray((128, nt * KC), dtype=F32, buffer=nl.psum)  # [p, j KC + ch]: slot ch 128 + p's bit
        for r in range(QW):
            sh = _sb((W, 128))
            nisa.tensor_scalar(dst=sh, data=dpc, op0=nl.equal, operand0=float(r * W), engine=VE)
            tr = _sb((W, nt, KC))  # [c, j, ch] = the bit of q = QW ch + r
            nisa.tensor_copy(dst=tr, src=st_.ap(pattern=[[nt * PR, W], [PR, nt], [QW, KC]], offset=r), engine=VE)
            nisa.nc_matmul(dst=PO, stationary=sh, moving=tr.reshape((W, nt * KC)), accumulate=(r > 0))
        nvf = _sb((128, nt, KC))
        nisa.tensor_scalar(dst=nvf.reshape((128, nt * KC)), data=PO, op0=nl.multiply, operand0=float(kp), engine=VE)
        nisa.tensor_copy(dst=cpt[:, b0:b0 + nt, 0:KC, 3], src=nvf, engine=VE)
        if MS["tv4"] is not None:  # the tail slot K of every row: partition 0, chunk KC
            nisa.tensor_copy(dst=cpt[0:1, b0:b0 + nt, KC, :], src=MS["tv4"], engine=VE)
        else:
            for j in range(nt):  # the tail slot K: partition 0, chunk KC
                b = rows[j]
                rti, ok = MS["tails"][j]
                ti = _sb((1, 1), I32)  # its digits (the device tracer takes only simple loop variables: unrolled)
                tf_ = _sb((1, 1))
                nisa.tensor_scalar(dst=ti, data=rti[:, 0:1], op0=nl.right_shift, operand0=16, engine=VE)
                nisa.tensor_copy(dst=tf_, src=ti, engine=VE)
                nisa.tensor_copy(dst=cpt[0:1, b, KC, 0:1], src=tf_, engine=VE)
                nisa.tensor_scalar(dst=ti, data=rti[:, 0:1], op0=nl.right_shift, operand0=8, op1=nl.bitwise_and,
                                   operand1=255, engine=VE)
                nisa.tensor_copy(dst=tf_, src=ti, engine=VE)
                nisa.tensor_copy(dst=cpt[0:1, b, KC, 1:2], src=tf_, engine=VE)
                nisa.tensor_scalar(dst=ti, data=rti[:, 0:1], op0=nl.bitwise_and, operand0=255, engine=VE)
                nisa.tensor_copy(dst=tf_, src=ti, engine=VE)
                nisa.tensor_copy(dst=cpt[0:1, b, KC, 2:3], src=tf_, engine=VE)
                tn = _sb((1, 1))
                nisa.tensor_reduce(dst=tn, op=nl.add, data=ok, axis=1)
                nisa.tensor_copy(dst=cpt[0:1, b, KC, 3:4], src=tn, engine=VE)
        mm = _sb((128, nt, NCH))
        nisa.tensor_scalar(dst=mm, data=cpt[:, b0:b0 + nt, :, 3], op0=nl.greater, operand0=0.0, engine=VE)
        nisa.tensor_reduce(dst=cnt[:, b0:b0 + nt], op=nl.add, data=mm, axis=2)

    @nki.jit
    def dsa_shell_attend_kernel(ev, ec, rows_sel, table, pos, rank, q_lat, kc, keep: int, A: int, kp: int,
                                     ns: int, nbits: int, rbits: int, lkp: int, la: int, scale: float, fp8: int,
                                     rev: int, ts: int = 0, il: int = 0, sb: int = 0, st: int = 0, ss: int = 0,
                                     tl: int = 0, fo: int = 0, fq: int = 1):
        """D3b: D3a's slot rows and bias (_merge_rows; st 1: _merge_rows_st, the rows stacked), then
        kernels/dsa_slots_c.py's compaction and row over them, in one call (ss 1, with st 1: the slot table built in
        SBUF from the merge's tiles, _slot_rows_sb / _slot_live_sb, and compacted from there, _compact_sb; srows and
        sbias are still written, and nothing in the kernel reads them back; tl 1, with ss 1: the merge's tail block
        for every row at once, _tail_rows_st). D3a's arguments plus q_lat bf16 [1, T, H,
        R] (every head's latent query, gathered), kc the latent cache's pool rows [cache rows / KP, KP R] (fp8 = 1:
        e4m3), scale. Returns (o fp32 [1, T, H, R], lse fp32
        [1, T, H, 1], srows int32 [T, ns], sbias fp32 [T, ns kp], over fp32 [1, npg]).

        The slot table dsa_slots_c reads (each slot's pool row as 8-bit digits and its live tokens, [128 p, T, NCH, 4]
        for slot ch 128 + p) is built from the slot rows and bias this call just wrote, as dsa_slots_c.kernel_args builds
        it, and the compaction and row are dsa_slots_c's own helpers over it: the same bits. A program's rows (the
        halves as dsa_slots_c splits them; at one row both programs do it) whose live slots all fit 128 are attended;
        over[pid] is 1 when one does not, and the caller then takes kernels/dsa_slots_n.py over every slot for every
        row, as dsa_slots_c.attend does.

        fo 1 (served): that fallback in this call. The programs' over flags are exchanged (each writes its
        own, a core barrier, both read back), so every row takes the same path: when no program is over, the
        compaction and row as above and a second device loop of 0 trips; when one is, no compaction, and the loop runs
        dsa_slots_n_s2's _row_nat (the fallback's natural-layout row, fq its qkw) over this program's rows of the srows
        and sbias it wrote, single-buffered (each pair one tile), into o and lse: the fallback's bits, without its call,
        host permutes and select."""
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
else:
    dsa_shell_attend_kernel = None


def _reva() -> int:
    from . import dsa_slots_c, dsa_slots_n_s2

    return SH.kernel_rev(__file__, "if nki is not None:  # D3b's kernel", "    dsa_shell_attend_kernel = None") \
        ^ REVM ^ dsa_slots_c.REV ^ dsa_slots_n_s2.REV  # fo's row is dsa_slots_n_s2's


REVA = _reva()


def attend(ev, ec, rows_sel, table, positions, q_all, kc, keep: int, A: int, rank: torch.Tensor, scale: float, *,
           kp: int = 4, ns: int = 640, rbits: int = 4, ts: int = 0, il: int = 0, sb: int = 1, st: int = 0,
           ss: int = 0, tl: int = 0, fo: int = 0, simulate: bool = False, grid: int | None = None):
    """D3b on a Neuron device (or in the NKI simulator; emulate_attend() on the host): (o [T, H, R] fp32, lse [T, H]
    fp32, srows [T, ns] int32, sbias [T, ns, kp] fp32, over [1] fp32: 1 when a row has more than 128 live slots, the
    caller then attending every slot with kernels/dsa_slots_n.py as dsa_slots_c.attend does; 0 on the host, where
    emulate_attend already did). q_all [T, H, R]; kc the latent cache (its pool rows are read). st 1 (with il 1, sb 1,
    ts 0): the merge's rows stacked in one selection (_merge_rows_st), the same bits; ss 1 (with st 1): the slot table
    in SBUF (no srows / sbias / slot-table round trip through HBM), the same bits; tl 1 (with ss 1): the merge's tail
    block for every row at once, the same bits; fo 1: the over fallback inside the call (the kernel's docstring), so o
    and lse are final: the caller's dsa_slots_n call and select are not needed (over still reports it)."""
    if ev.device.type == "cpu" and not simulate and not SH.HOST_SIM:
        o, lse, srows, sbias = emulate_attend(ev, ec, rows_sel, table, positions, q_all, kc, keep, A,
                                              int(rank.reshape(())), scale, kp, ns)
        return o, lse, srows, sbias, torch.zeros(1)
    if dsa_shell_attend_kernel is None:
        raise RuntimeError("the NKI DSA shell needs the nki package (the Neuron venv)")
    if grid is None:
        from .. import platform

        grid = 1 if simulate else platform.nki_grid()
    from . import dsa_slots

    _, T, _ = ev.shape
    H, R = q_all.shape[1], q_all.shape[2]
    NPg = table.shape[1]
    kc2 = kc.reshape(kc.shape[0], -1)
    fp8 = kc2.dtype == torch.float8_e4m3fn
    kcp = kc2.reshape(kc2.shape[0] // dsa_slots.KP, dsa_slots.KP * R)
    args = dict(ev=ev.float().contiguous(), ec=ec.float().contiguous(), rows_sel=rows_sel.to(torch.int32).contiguous(),
                table=table.to(torch.int32).contiguous(), pos=positions.to(torch.float32).reshape(1, T),
                rank=rank.to(torch.float32).reshape(1, 1), q_lat=q_all.to(torch.bfloat16).reshape(1, T, H, R),
                kc=kcp, keep=int(keep), A=int(A), kp=int(kp), ns=int(ns), nbits=int((NPg * PPP * A - 1).bit_length()),
                rbits=int(rbits), lkp=int(kp).bit_length() - 1, la=int(A).bit_length() - 1, scale=float(scale),
                fp8=int(fp8), rev=REVA, ts=int(ts), il=int(il), sb=int(sb), st=int(st), ss=int(ss), tl=int(tl),
                fo=int(fo), fq=int(dsa_slots.QKW))
    k = dsa_shell_attend_kernel
    if simulate:
        import ml_dtypes
        import numpy as np

        def to_np(t):
            if not isinstance(t, torch.Tensor):
                return t
            if t.dtype == torch.bfloat16:
                return t.float().numpy().astype(ml_dtypes.bfloat16)
            if t.dtype == torch.float8_e4m3fn:
                return t.view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
            return t.numpy()

        o, ls, sr, sb, ov = nki.simulate(k[grid] if grid > 1 else k)(**{n: to_np(t) for n, t in args.items()})
        f = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float32))  # noqa: E731
        return (f(o).view(T, H, R), f(ls).view(T, H), torch.from_numpy(np.asarray(sr).astype(np.int32)),
                f(sb).view(T, ns, kp), f(ov).amax().view(1))
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    o, ls, sr, sb, ov = wrap_nki(k)[grid](**args)
    return o.view(T, H, R), ls.view(T, H), sr, sb.view(T, ns, kp), ov.amax(dim=1)


# -- D2 v2: dsa_index's static pipeline, one selection per program, the compaction on the PE ----------------------------

if nki is not None:  # D2v2's kernel: its own block and revision (REV2), so no other call's key moves
    from .dsa_index import _gather as _ix_gather
    from .dsa_index import _row_setup as _ix_row_setup
    from .dsa_index import _stage_a as _ix_stage_a
    from .dsa_index import _stage_b as _ix_stage_b
    from .dsa_topk import _rlo as _tk_rlo
    from .dsa_topk import _rwidth as _tk_rwidth
    from .dsa_topk import _wconsts as _tk_wconsts

    def _split3(S, n: int, W: int):
        """[n, W] fp32 -> (hi, mid, lo) bf16 [n, W] with hi + mid + lo == S exactly (each piece the next 8 significant
        bits: models/moe split3's arithmetic), so a 0 / 1 matmul of the three pieces accumulated in fp32 is exact."""
        hm = _sb((n, W), I32)
        nisa.tensor_scalar(dst=hm, data=S.view(I32), op0=nl.bitwise_and, operand0=-65536, engine=VE)
        r = _sb((n, W))
        nisa.tensor_tensor(dst=r, data1=S, data2=hm.view(F32), op=nl.subtract, engine=VE)
        mm = _sb((n, W), I32)
        nisa.tensor_scalar(dst=mm, data=r.view(I32), op0=nl.bitwise_and, operand0=-65536, engine=VE)
        lo32 = _sb((n, W))
        nisa.tensor_tensor(dst=lo32, data1=r, data2=mm.view(F32), op=nl.subtract, engine=VE)
        out = []
        for src in (hm.view(F32), mm.view(F32), lo32):
            t = _sb((n, W), BF16)
            nisa.tensor_copy(dst=t, src=src, engine=VE)
            out.append(t)
        return out

    def _select2s(SS, nt: int, W: int, G, POS, keep: int, nbits: int, rbits: int):
        """dsa_topk._select (vis_only, kp 1, no tail, rbits > 1) over the program's nt rows at once: SS [128, nt, W] fp32,
        row i's scores at [:, i, :] on all 128 partitions (g 128: G all ones), POS [128, W] each entry's position (the
        pool index, the served tie order), piece offsets 0. Each op runs once over every row where _select issues it per
        row tile; the per-row scalars become [128, nt] tiles read through zero-stride access patterns. The element
        arithmetic is _select's (integer counts, bit-pattern thresholds, x * +-1), so the selection is its bit for bit.
        Returns O [128, nt, W] fp32 in SBUF: 0 selected, NEG_INF not (no HBM round trip to the compaction). The counts
        (each <= W per partition) go to the PE in bf16 when W <= 256, where every count is exact: the all-ones G and the
        counts are integers <= 256, their fp32 PSUM sums <= 128 W (one PE pass, where fp32 takes four)."""
        n = 128
        P = n * W
        J = (1 << rbits) - 1
        CD = BF16 if W <= 256 else F32  # the counts' dtype to the PE: exact in bf16 up to 256
        Gc = _sb((n, n), CD)
        nisa.tensor_copy(dst=Gc, src=G[0:n, 0:n], engine=VE)
        po0 = _sb((n, 1))
        nisa.memset(dst=po0, value=0.0)
        # 1. sign
        ge0 = _sb((n, nt, W))
        nisa.tensor_scalar(dst=ge0, data=SS, op0=nl.greater_equal, operand0=0.0, engine=VE)
        cp = _sb((n, nt), CD)
        nisa.tensor_reduce(dst=cp, op=nl.add, data=ge0, axis=2)
        tot = nl.ndarray((n, nt), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=tot, stationary=Gc, moving=cp)
        f = _sb((n, nt))
        nisa.tensor_scalar(dst=f, data=tot, op0=nl.less, operand0=float(keep), engine=VE)
        sg = _sb((n, nt))
        nisa.tensor_scalar(dst=sg, data=f, op0=nl.multiply, operand0=-2.0, op1=nl.add, operand1=1.0, engine=VE)
        k2 = _sb((n, nt))
        nisa.tensor_scalar(dst=k2, data=f, op0=nl.multiply, operand0=float(P + 1 - 2 * keep), op1=nl.add,
                           operand1=float(keep), engine=VE)
        s2 = _sb((n, nt, W))
        nisa.tensor_tensor(dst=s2, data1=SS, data2=sg.ap(pattern=[[nt, n], [1, nt], [0, W]], offset=0),
                           op=nl.multiply, engine=VE)
        pre = _sb((n, nt), I32)
        nisa.memset(dst=pre, value=0)
        # 2. radix over t2's bit pattern, rbits a round
        RC, TC = _tk_wconsts(po0, nbits, rbits)
        for r in range((30 + rbits) // rbits):
            lo = _tk_rlo(30, rbits, r)
            Jw = (1 << _tk_rwidth(30, rbits, r)) - 1
            cand = _sb((n, nt, Jw), I32)  # PRE | j 2^lo: PRE's bits below the digit's are 0
            nisa.tensor_tensor(dst=cand, data1=pre.ap(pattern=[[nt, n], [1, nt], [0, Jw]], offset=0),
                               data2=RC[r].ap(pattern=[[J, n], [0, nt], [1, Jw]], offset=0), op=nl.bitwise_or,
                               engine=VE)
            ge = _sb((n, nt, Jw, W))
            nisa.tensor_tensor(dst=ge, data1=s2.ap(pattern=[[nt * W, n], [W, nt], [0, Jw], [1, W]], offset=0),
                               data2=cand.view(F32).ap(pattern=[[nt * Jw, n], [Jw, nt], [1, Jw], [0, W]], offset=0),
                               op=nl.greater_equal, engine=VE)
            cpj = _sb((n, nt, Jw), CD)
            nisa.tensor_reduce(dst=cpj, op=nl.add, data=ge, axis=3)
            totj = nl.ndarray((n, nt * Jw), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(dst=totj, stationary=Gc, moving=cpj.reshape((n, nt * Jw)))
            okj = _sb((n, nt, Jw))  # the digit: how many of the candidates keep k2 scores at or above them
            nisa.tensor_tensor(dst=okj.reshape((n, nt * Jw)), data1=totj,
                               data2=k2.ap(pattern=[[nt, n], [1, nt], [0, Jw]], offset=0), op=nl.greater_equal,
                               engine=VE)
            dg = _sb((n, nt))
            nisa.tensor_reduce(dst=dg, op=nl.add, data=okj, axis=2)
            inc = _sb((n, nt), I32)
            nisa.tensor_scalar(dst=inc, data=dg, op0=nl.multiply, operand0=float(1 << lo), engine=VE)
            nisa.tensor_tensor(dst=pre, data1=pre, data2=inc, op=nl.bitwise_or, engine=VE)
        # 3. above, room, ties
        th = _sb((n, nt))
        nisa.tensor_tensor(dst=th, data1=pre.view(F32), data2=sg, op=nl.multiply, engine=VE)
        thw = th.ap(pattern=[[nt, n], [1, nt], [0, W]], offset=0)
        ab = _sb((n, nt, W))
        nisa.tensor_tensor(dst=ab, data1=SS, data2=thw, op=nl.greater, engine=VE)
        abp = _sb((n, nt), CD)
        nisa.tensor_reduce(dst=abp, op=nl.add, data=ab, axis=2)
        abt = nl.ndarray((n, nt), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=abt, stationary=Gc, moving=abp)
        room = _sb((n, nt))
        nisa.tensor_scalar(dst=room, data=abt, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=float(keep),
                           engine=VE)
        tied = _sb((n, nt, W))
        nisa.tensor_tensor(dst=tied, data1=SS, data2=thw, op=nl.equal, engine=VE)
        tj = _sb((n, nt, W))  # w where tied, w + BIG elsewhere
        nisa.scalar_tensor_tensor(dst=tj, data=tied, op0=nl.multiply, operand0=-BIG_T, op1=nl.add,
                                  operand1=POS.ap(pattern=[[W, n], [0, nt], [1, W]], offset=0))
        nisa.tensor_scalar(dst=tj, data=tj, op0=nl.add, operand0=BIG_T, engine=VE)
        lim = _sb((n, nt))
        nisa.memset(dst=lim, value=0.0)
        for r in range((nbits - 1 + rbits) // rbits):
            lo = _tk_rlo(nbits - 1, rbits, r)
            Jw = (1 << _tk_rwidth(nbits - 1, rbits, r)) - 1
            thr = _sb((n, nt, Jw))  # lim + j 2^lo in this piece's positions
            nisa.tensor_tensor(dst=thr, data1=lim.ap(pattern=[[nt, n], [1, nt], [0, Jw]], offset=0),
                               data2=TC[r].ap(pattern=[[J, n], [0, nt], [1, Jw]], offset=0), op=nl.add, engine=VE)
            lt = _sb((n, nt, Jw, W))
            nisa.tensor_tensor(dst=lt, data1=tj.ap(pattern=[[nt * W, n], [W, nt], [0, Jw], [1, W]], offset=0),
                               data2=thr.ap(pattern=[[nt * Jw, n], [Jw, nt], [1, Jw], [0, W]], offset=0), op=nl.less,
                               engine=VE)
            cpj = _sb((n, nt, Jw), CD)
            nisa.tensor_reduce(dst=cpj, op=nl.add, data=lt, axis=3)
            totj = nl.ndarray((n, nt * Jw), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(dst=totj, stationary=Gc, moving=cpj.reshape((n, nt * Jw)))
            okj = _sb((n, nt, Jw))  # the digit: how many of the limits keep the tied count below room
            nisa.tensor_tensor(dst=okj.reshape((n, nt * Jw)), data1=totj,
                               data2=room.ap(pattern=[[nt, n], [1, nt], [0, Jw]], offset=0), op=nl.less, engine=VE)
            dg = _sb((n, nt))
            nisa.tensor_reduce(dst=dg, op=nl.add, data=okj, axis=2)
            nisa.scalar_tensor_tensor(dst=lim, data=dg, op0=nl.multiply, operand0=float(1 << lo), op1=nl.add,
                                      operand1=lim)
        # 4. out (piece offsets 0: lp = lim)
        sel = _sb((n, nt, W))
        nisa.tensor_tensor(dst=sel, data1=tj, data2=lim.ap(pattern=[[nt, n], [1, nt], [0, W]], offset=0),
                           op=nl.less_equal, engine=VE)
        nisa.tensor_tensor(dst=sel, data1=sel, data2=ab, op=nl.add, engine=VE)
        vm = _sb((n, nt, W))
        nisa.tensor_scalar(dst=vm, data=SS, op0=nl.greater, operand0=VISIBLE_T, engine=VE)
        nisa.tensor_tensor(dst=sel, data1=sel, data2=vm, op=nl.multiply, engine=VE)
        O = _sb((n, nt, W))
        nisa.tensor_scalar(dst=O, data=sel, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=-NEG_INF,
                           engine=VE)
        return O

    def _compact2(X, ii, bb):
        """The program's rows' local lists (row bb[r], the program's ii[r]-th) from their selections and score tiles, on
        the engines only: the k-th selected pool's partition from the partitions' inclusive counts, its partition row by
        a 0 / 1 matmul, its column by a prefix count of that row (a triangular matmul), its score by the same 0 / 1 matmul
        of the score tile's split3 pieces, its page from the page-group row's 8-bit digits. Every value exact. The rows'
        instructions are interleaved one op at a time (each engine runs its queue in order: one row's op fills the other's
        wait on the PE / PSUM), and each vector op runs once over the KT tiles of keep, with each element's arithmetic
        unchanged."""
        NT, NG, keep, KT = X["NT"], X["NG"], X["keep"], X["KT"]
        nch, cw = X["nch"], X["cw"]
        n = len(bb)
        s01, cnt, s01b, inc, incr, incb, pgi, t0i, t0f = [], [], [], [], [], [], [], [], []
        for r in range(n):
            s01.append(_sb((128, NT)))
            cnt.append(_sb((128, 1)))
            s01b.append(_sb((128, NT), BF16))
            inc.append(_sb((128, 1)))
            incr.append(_sb((1, 128)))
            incb.append(_sb((128, 128)))
            pgi.append(_sb((128, NG), I32))
            t0i.append(_sb((128, 1), I32))
            t0f.append(_sb((128, 1)))
        for r in range(n):  # the row's page groups and the page of pool 0 (pool_row of the entries past the count)
            dma_static(dst=pgi[r], src=X["ppg"].ap(pattern=[[X["B"] * NG, 128], [1, NG]], offset=bb[r] * NG))
            dma_static(dst=t0i[r], src=X["ppg"].ap(pattern=[[0, 128], [1, 1]], offset=bb[r] * NG))
        for r in range(n):
            nisa.tensor_scalar_reduce(dst=s01[r], data=X["O"][:, ii[r], :], op0=nl.equal, operand0=0.0,
                                      reduce_op=nl.add, reduce_res=cnt[r])
        pin = []
        for r in range(n):  # inclusive count at each partition's end
            pin.append(nl.ndarray((128, 1), dtype=F32, buffer=nl.psum))
            nisa.nc_matmul(dst=pin[r], stationary=X["UPI"], moving=cnt[r], accumulate=False)
        for r in range(n):
            nisa.tensor_copy(dst=s01b[r], src=s01[r], engine=VE)
        for r in range(n):
            nisa.tensor_copy(dst=inc[r], src=pin[r], engine=VE)
        pit = []
        for r in range(n):
            pit.append(nl.ndarray((1, 128), dtype=F32, buffer=nl.psum))
            nisa.nc_matmul(dst=pit[r], stationary=inc[r], moving=X["IF"], accumulate=False)
        for r in range(n):
            nisa.tensor_copy(dst=incr[r], src=pit[r], engine=VE)
        pib = []
        for r in range(n):  # every partition holds the row of counts
            pib.append(nl.ndarray((128, 128), dtype=F32, buffer=nl.psum))
            nisa.nc_matmul(dst=pib[r], stationary=X["ONES1"], moving=incr[r], accumulate=False)
        for r in range(n):
            nisa.tensor_copy(dst=incb[r], src=pib[r], engine=VE)
        hi, mid, lo = [], [], []
        for r in range(n):
            h, m, lw = _split3(X["SALL"][ii[r]], 128, NT)
            hi.append(h)
            mid.append(m)
            lo.append(lw)
        dgt, tmp, tmf = [], [], []
        for r in range(n):  # the row's page groups as 8-bit digits
            dgt.append(_sb((128, 3, NG), BF16))
            tmp.append(_sb((128, NG), I32))
            tmf.append(_sb((128, NG)))
        for r in range(n):
            nisa.tensor_scalar(dst=tmp[r], data=pgi[r], op0=nl.right_shift, operand0=16, engine=VE)
            nisa.tensor_copy(dst=tmf[r], src=tmp[r], engine=VE)
            nisa.tensor_copy(dst=dgt[r][:, 0, :], src=tmf[r], engine=VE)
            nisa.tensor_scalar(dst=tmp[r], data=pgi[r], op0=nl.right_shift, operand0=8, op1=nl.bitwise_and,
                               operand1=255, engine=VE)
            nisa.tensor_copy(dst=tmf[r], src=tmp[r], engine=VE)
            nisa.tensor_copy(dst=dgt[r][:, 1, :], src=tmf[r], engine=VE)
            nisa.tensor_scalar(dst=tmp[r], data=pgi[r], op0=nl.bitwise_and, operand0=255, engine=VE)
            nisa.tensor_copy(dst=tmf[r], src=tmp[r], engine=VE)
            nisa.tensor_copy(dst=dgt[r][:, 2, :], src=tmf[r], engine=VE)
            nisa.tensor_copy(dst=t0f[r], src=t0i[r], engine=VE)
            nisa.tensor_scalar(dst=t0f[r], data=t0f[r], op0=nl.multiply, operand0=float(PPP), engine=VE)
        LV, CP = [], []
        for r in range(n):
            LV.append(_sb((128, KT)))
            CP.append(_sb((128, KT)))
        # every per-k-tile op on the vector engine runs once over the KT tiles ([128, KT, .]: the tile index t is a free
        # axis, the k's partition-wise scalars become [128, KT] tiles read through zero-stride access patterns); the PE's
        # matmuls stay per tile, into slices of one PSUM tile per group of TP tiles (TP NT <= 512 fp32: one bank)
        TP = min(KT, max(1, 512 // NT))
        IO = X["IO512"]
        KV = X["KV"]
        pk, m1, ex, dd, ohk = [], [], [], [], []
        for r in range(n):
            pk.append(_sb((128, KT)))  # the partition of the k-th selected pool: partitions whose inclusive count <= k
            m1.append(_sb((128, KT, 128)))
            ex.append(_sb((128, KT)))  # its exclusive count: the largest inclusive count <= k (0 if none)
            dd.append(_sb((128, KT)))
            ohk.append(_sb((128, KT, 128), BF16))  # [k, t, p] = (p == partition(k))
        kvp = KV.ap(pattern=[[KT, 128], [1, KT], [0, 128]], offset=0)
        for r in range(n):
            nisa.tensor_tensor(dst=m1[r], data1=incb[r].ap(pattern=[[128, 128], [0, KT], [1, 128]], offset=0), data2=kvp,
                               op=nl.less_equal, engine=VE)
        for r in range(n):
            nisa.tensor_reduce(dst=pk[r], op=nl.add, data=m1[r], axis=2)
        for r in range(n):
            nisa.tensor_tensor(dst=m1[r], data1=m1[r], data2=incb[r].ap(pattern=[[128, 128], [0, KT], [1, 128]], offset=0),
                               op=nl.multiply, engine=VE)
        for r in range(n):
            nisa.tensor_reduce(dst=ex[r], op=nl.maximum, data=m1[r], axis=2)
        for r in range(n):
            nisa.tensor_scalar(dst=pk[r], data=pk[r], op0=nl.minimum, operand0=127.0, engine=VE)
        for r in range(n):
            nisa.tensor_tensor(dst=dd[r], data1=KV, data2=ex[r], op=nl.subtract, engine=VE)
        for r in range(n):
            nisa.tensor_tensor(dst=ohk[r], data1=IO.ap(pattern=[[512, 128], [0, KT], [1, 128]], offset=0),
                               data2=pk[r].ap(pattern=[[KT, 128], [1, KT], [0, 128]], offset=0), op=nl.equal, engine=VE)
        oht, rs = [], []
        for r in range(n):
            oht.append(_sb((128, KT, 128), BF16))  # [p, t, k]
            rs.append(_sb((128, KT, NT), BF16))  # the k-th pool's partition row of selections
        TQ = min(KT, 4)  # the transposes' group: 128 columns a tile
        for g0 in range(0, KT, TQ):
            tg = min(TQ, KT - g0)
            pt = []
            for r in range(n):
                pt.append(nl.ndarray((128, TQ, 128), dtype=F32, buffer=nl.psum))
                for u in range(tg):
                    nisa.nc_matmul(dst=pt[r][:, u, :], stationary=ohk[r][:, g0 + u, :], moving=X["IB"], accumulate=False)
            for r in range(n):
                nisa.tensor_copy(dst=oht[r][:, g0:g0 + tg, :], src=pt[r][:, 0:tg, :], engine=VE)
        for g0 in range(0, KT, TP):
            tg = min(TP, KT - g0)
            prs = []
            for r in range(n):
                prs.append(nl.ndarray((128, TP, NT), dtype=F32, buffer=nl.psum))
                for u in range(tg):
                    nisa.nc_matmul(dst=prs[r][:, u, :], stationary=oht[r][:, g0 + u, :], moving=s01b[r], accumulate=False)
            for r in range(n):
                nisa.tensor_copy(dst=rs[r][:, g0:g0 + tg, :], src=prs[r][:, 0:tg, :], engine=VE)
        rts = []
        for r in range(n):
            rts.append(_sb((128, KT, nch * 128), BF16))  # row t's selections transposed, chunk j at [0:cw, t, j 128:]
        for t in range(KT):
            ptr = []
            for r in range(n):
                ptr.append(nl.ndarray((128, nch * 128), dtype=F32, buffer=nl.psum))
                for j in range(nch):
                    nisa.nc_matmul(dst=ptr[r][0:cw, j * 128:(j + 1) * 128], stationary=rs[r][:, t, j * cw:(j + 1) * cw],
                                   moving=X["IB"], accumulate=False)
            for r in range(n):
                nisa.tensor_copy(dst=rts[r][0:cw, t, :], src=ptr[r][0:cw, :], engine=VE)
        col, sv, pds = [], [], []
        for r in range(n):
            col.append(_sb((128, KT)))
            sv.append(_sb((128, KT)))
            pds.append(_sb((128, KT, 3 * NG)))
        # An accumulation group (a start, then accumulates) gets a PSUM tile of its own: several groups in slices of one
        # tile pass nki.simulate and compute wrong on trn2 (memory nki-psum-slice-accum-device-wrong; this kernel's psc
        # in that form gave 1,536 of 2,048 lval wrong on device, cd4aa18a). Lone starts into slices are exact.
        if nch == 1:
            for g0 in range(0, KT, TP):
                tg = min(TP, KT - g0)
                prc = []
                for r in range(n):  # inclusive count along the row: one lone start a tile, into slices of one tile
                    prc.append(nl.ndarray((128, TP, NT), dtype=F32, buffer=nl.psum))
                    for u in range(tg):
                        nisa.nc_matmul(dst=prc[r][:, u, :], stationary=rts[r][0:cw, g0 + u, 0:128],
                                       moving=X["UT"][0:cw, 0:NT], accumulate=False)
                for r in range(n):
                    m3 = _sb((128, TP, NT))
                    nisa.tensor_tensor(dst=m3[:, 0:tg, :], data1=prc[r][:, 0:tg, :],
                                       data2=dd[r].ap(pattern=[[KT, 128], [1, tg], [0, NT]], offset=g0),
                                       op=nl.less_equal, engine=VE)
                    nisa.tensor_reduce(dst=col[r][:, g0:g0 + tg], op=nl.add, data=m3[:, 0:tg, :], axis=2)
        else:
            for t in range(KT):
                prc = []
                for r in range(n):  # inclusive count along the row: nch accumulated chunks, a PSUM tile a k tile
                    prc.append(nl.ndarray((128, NT), dtype=F32, buffer=nl.psum))
                    for j in range(nch):
                        nisa.nc_matmul(dst=prc[r], stationary=rts[r][0:cw, t, j * 128:(j + 1) * 128],
                                       moving=X["UT"][0:cw, j * NT:(j + 1) * NT], accumulate=(j > 0))
                for r in range(n):
                    m3 = _sb((128, NT))
                    nisa.tensor_scalar_reduce(dst=m3, data=prc[r], op0=nl.less_equal, operand0=dd[r][:, t:t + 1],
                                              reduce_op=nl.add, reduce_res=col[r][:, t:t + 1])
        for r in range(n):
            nisa.tensor_scalar(dst=col[r], data=col[r], op0=nl.minimum, operand0=float(NT - 1), engine=VE)
        ohc = []
        for r in range(n):
            ohc.append(_sb((128, KT, NT)))
            nisa.tensor_tensor(dst=ohc[r], data1=IO.ap(pattern=[[512, 128], [0, KT], [1, NT]], offset=0),
                               data2=col[r].ap(pattern=[[KT, 128], [1, KT], [0, NT]], offset=0), op=nl.equal, engine=VE)
        for t in range(KT):
            psc = []
            for r in range(n):  # the partition row of scores: split3's pieces accumulated, a PSUM tile a k tile
                psc.append(nl.ndarray((128, NT), dtype=F32, buffer=nl.psum))
                nisa.nc_matmul(dst=psc[r], stationary=oht[r][:, t, :], moving=hi[r], accumulate=False)
                nisa.nc_matmul(dst=psc[r], stationary=oht[r][:, t, :], moving=mid[r], accumulate=True)
                nisa.nc_matmul(dst=psc[r], stationary=oht[r][:, t, :], moving=lo[r], accumulate=True)
            for r in range(n):
                scr = _sb((128, NT))
                nisa.tensor_tensor(dst=scr, data1=ohc[r][:, t, :], data2=psc[r], op=nl.multiply, engine=VE)
                nisa.tensor_reduce(dst=sv[r][:, t:t + 1], op=nl.add, data=scr, axis=1)
        for g0 in range(0, KT, TP):
            tg = min(TP, KT - g0)
            ppd = []
            for r in range(n):  # the partition's page digits, to SBUF (a vector op may not read both operands from PSUM)
                ppd.append(nl.ndarray((128, TP, 3 * NG), dtype=F32, buffer=nl.psum))
                for u in range(tg):
                    nisa.nc_matmul(dst=ppd[r][:, u, :], stationary=oht[r][:, g0 + u, :],
                                   moving=dgt[r].reshape((128, 3 * NG)), accumulate=False)
            for r in range(n):
                nisa.tensor_copy(dst=pds[r][:, g0:g0 + tg, :], src=ppd[r][:, 0:tg, :], engine=VE)
        pgf, ci, gi, ji, gf, jf, ohg, page, valid, ninf, lp, row = [], [], [], [], [], [], [], [], [], [], [], []
        for r in range(n):
            pgf.append(_sb((128, KT, NG)))
            ci.append(_sb((128, KT), I32))
            gi.append(_sb((128, KT), I32))
            ji.append(_sb((128, KT), I32))
            gf.append(_sb((128, KT)))
            jf.append(_sb((128, KT)))
            ohg.append(_sb((128, KT, NG)))
            page.append(_sb((128, KT)))
            valid.append(_sb((128, KT)))
            ninf.append(_sb((128, KT)))
            lp.append(_sb((128, KT)))
            row.append(_sb((128, KT)))
        for r in range(n):
            nisa.scalar_tensor_tensor(dst=pgf[r], data=pds[r][:, :, 0:NG], op0=nl.multiply, operand0=256.0, op1=nl.add,
                                      operand1=pds[r][:, :, NG:2 * NG])
        for r in range(n):
            nisa.scalar_tensor_tensor(dst=pgf[r], data=pgf[r], op0=nl.multiply, operand0=256.0, op1=nl.add,
                                      operand1=pds[r][:, :, 2 * NG:3 * NG])
        for r in range(n):
            nisa.tensor_copy(dst=ci[r], src=col[r], engine=VE)
        for r in range(n):
            nisa.tensor_scalar(dst=gi[r], data=ci[r], op0=nl.right_shift, operand0=LPP, engine=VE)
        for r in range(n):
            nisa.tensor_scalar(dst=ji[r], data=ci[r], op0=nl.bitwise_and, operand0=PPP - 1, engine=VE)
        for r in range(n):
            nisa.tensor_copy(dst=gf[r], src=gi[r], engine=VE)
        for r in range(n):
            nisa.tensor_copy(dst=jf[r], src=ji[r], engine=VE)
        for r in range(n):
            nisa.tensor_tensor(dst=ohg[r], data1=IO.ap(pattern=[[512, 128], [0, KT], [1, NG]], offset=0),
                               data2=gf[r].ap(pattern=[[KT, 128], [1, KT], [0, NG]], offset=0), op=nl.equal, engine=VE)
        for r in range(n):
            nisa.tensor_tensor(dst=ohg[r], data1=ohg[r], data2=pgf[r], op=nl.multiply, engine=VE)
        for r in range(n):
            nisa.tensor_reduce(dst=page[r], op=nl.add, data=ohg[r], axis=2)
        for r in range(n):
            nisa.tensor_tensor(dst=valid[r], data1=KV, data2=incb[r].ap(pattern=[[128, 128], [0, KT]], offset=127),
                               op=nl.less, engine=VE)
        # lval = valid ? score : NEG_INF; lp = valid ? partition NT + col : 0; rows = valid ? page 8 + j : page0 8
        for r in range(n):
            nisa.tensor_scalar(dst=ninf[r], data=valid[r], op0=nl.subtract, operand0=1.0, op1=nl.multiply,
                               operand1=-NEG_INF, engine=VE)
        for r in range(n):
            nisa.tensor_tensor(dst=sv[r], data1=sv[r], data2=valid[r], op=nl.multiply, engine=VE)
        for r in range(n):
            nisa.tensor_tensor(dst=LV[r], data1=sv[r], data2=ninf[r], op=nl.add, engine=VE)
        for r in range(n):
            nisa.scalar_tensor_tensor(dst=lp[r], data=pk[r], op0=nl.multiply, operand0=float(NT), op1=nl.add,
                                      operand1=col[r])
        for r in range(n):
            nisa.tensor_tensor(dst=lp[r], data1=lp[r], data2=valid[r], op=nl.multiply, engine=VE)
        for r in range(n):
            nisa.tensor_scalar(dst=CP[r], data=lp[r], op0=nl.multiply, operand0=float(X["A"]), op1=nl.add,
                               operand1=X["RK"], engine=VE)
        for r in range(n):
            nisa.scalar_tensor_tensor(dst=row[r], data=page[r], op0=nl.multiply, operand0=float(PPP), op1=nl.add,
                                      operand1=jf[r])
        for r in range(n):
            nisa.tensor_tensor(dst=row[r], data1=row[r], data2=t0f[r].ap(pattern=[[1, 128], [0, KT]], offset=0),
                               op=nl.subtract, engine=VE)
        for r in range(n):
            nisa.tensor_tensor(dst=row[r], data1=row[r], data2=valid[r], op=nl.multiply, engine=VE)
        for r in range(n):
            nisa.tensor_tensor(dst=row[r], data1=row[r], data2=t0f[r].ap(pattern=[[1, 128], [0, KT]], offset=0),
                               op=nl.add, engine=VE)
        bk = X["B"] * keep
        for r in range(n):  # out [3, B, keep]: lval, cpool, rows (fp32, exact)
            dma_static(dst=X["out"].ap(pattern=[[1, 128], [128, KT]], offset=bb[r] * keep), src=LV[r])
            dma_static(dst=X["out"].ap(pattern=[[1, 128], [128, KT]], offset=bk + bb[r] * keep), src=CP[r])
            dma_static(dst=X["out"].ap(pattern=[[1, 128], [128, KT]], offset=2 * bk + bb[r] * keep), src=row[r])

    _C2S_G = 4  # _compact2s's rows a group: its pit [1, G, 128] / pib [128, G, 128] fp32 PSUM tiles are 2 KB (one bank)

    def _c2s_waves(GQ, KT: int, T: int):
        """_compact2s's PSUM chunks: group q's gs KT slots (row j's k tile t at s = j KT + t), T slots a PSUM tile, as
        waves (wave w: each group's w-th chunk as (q, its first slot, its slots)), so the groups take turns one chunk at
        a time, as _compact2's rows do. Python ints only (trace time)."""
        nw = 0
        for q in range(len(GQ)):
            nw = max(nw, (GQ[q][1] * KT + T - 1) // T)
        waves = []
        for w in range(nw):
            wv = []
            for q in range(len(GQ)):
                nsl = GQ[q][1] * KT
                if w * T < nsl:
                    wv.append((q, w * T, min(T, nsl - w * T)))
            waves.append(wv)
        return waves

    def _compact2s(X, ii, bb):
        """_compact2 (cs 0) with the program's rows stacked (cs 1, served): the rows in groups of up to
        _C2S_G (consecutive in ii), and each of _compact2's per-row tiles one tile a group with the row as a free axis
        ([128, gs, .]; a row's [128, KT, .] tiles as [128, gs KT, .], row j's k tile t at slot s = j KT + t), so each
        vector op runs once a group where _compact2 issues it once a row. Every element's arithmetic is _compact2's (the
        same op, operands and dtypes; a per-row operand read through an access pattern with the group's strides), so
        lval, cpool and rows are its bits. The matmuls stay one a row and k tile; lone starts (accumulate=False) go into
        slices of one PSUM tile a group and chunk (at most 512 fp32 a partition: one bank), drained by one vector op.
        The accumulation groups (prc's chunks at nch > 1, psc's split3 pieces) keep a PSUM tile each and the ops that
        read them stay one a row and k tile (memory nki-psum-slice-accum-device-wrong); psc's products go into ohc (read
        nowhere else) and are summed once a group (ohc is one-hot: one nonzero term a sum, exact in any order). The
        selection count is a tensor_scalar then a tensor_reduce over the column axis (tensor_scalar_reduce sums every
        free axis; 0 / 1 sums are exact). _split3 and the HBM DMAs stay per row."""
        NT, NG, keep, KT = X["NT"], X["NG"], X["keep"], X["KT"]
        nch, cw = X["nch"], X["cw"]
        IO = X["IO512"]
        KV = X["KV"]
        n = len(bb)
        GQ = []  # the groups: (first row, rows)
        for g0 in range(0, n, _C2S_G):
            gs = min(_C2S_G, n - g0)
            for j in range(gs):
                assert ii[g0 + j] == ii[g0] + j  # a group's selections are consecutive in O
            GQ.append((g0, gs))
        nq = len(GQ)
        s01, cnt, s01b, inc, incr, incb, pgi, t0i, t0f = [], [], [], [], [], [], [], [], []
        for q in range(nq):
            gs = GQ[q][1]
            s01.append(_sb((128, gs, NT)))
            cnt.append(_sb((128, gs)))
            s01b.append(_sb((128, gs, NT), BF16))
            inc.append(_sb((128, gs)))
            incr.append(_sb((1, gs, 128)))
            incb.append(_sb((128, gs, 128)))
            pgi.append(_sb((128, gs, NG), I32))
            t0i.append(_sb((128, gs), I32))
            t0f.append(_sb((128, gs)))
        for q in range(nq):  # per row (its own HBM addresses), into the row's slice
            g0, gs = GQ[q]
            for j in range(gs):
                dma_static(dst=pgi[q][:, j, :],
                           src=X["ppg"].ap(pattern=[[X["B"] * NG, 128], [1, NG]], offset=bb[g0 + j] * NG))
                dma_static(dst=t0i[q][:, j:j + 1], src=X["ppg"].ap(pattern=[[0, 128], [1, 1]], offset=bb[g0 + j] * NG))
        for q in range(nq):  # the selection's 0 / 1, then its count a partition, per row
            g0, gs = GQ[q]
            nisa.tensor_scalar(dst=s01[q], data=X["O"][:, ii[g0]:ii[g0] + gs, :], op0=nl.equal, operand0=0.0,
                               engine=VE)
        for q in range(nq):
            nisa.tensor_reduce(dst=cnt[q], op=nl.add, data=s01[q], axis=2)
        pin = []
        for q in range(nq):  # inclusive count at each partition's end: a lone start a row, into slices of one tile
            gs = GQ[q][1]
            pin.append(nl.ndarray((128, gs), dtype=F32, buffer=nl.psum))
            for j in range(gs):
                nisa.nc_matmul(dst=pin[q][:, j:j + 1], stationary=X["UPI"], moving=cnt[q][:, j:j + 1],
                               accumulate=False)
        for q in range(nq):
            nisa.tensor_copy(dst=s01b[q], src=s01[q], engine=VE)
        for q in range(nq):
            nisa.tensor_copy(dst=inc[q], src=pin[q], engine=VE)
        pit = []
        for q in range(nq):
            gs = GQ[q][1]
            pit.append(nl.ndarray((1, gs, 128), dtype=F32, buffer=nl.psum))
            for j in range(gs):
                nisa.nc_matmul(dst=pit[q][:, j, :], stationary=inc[q][:, j:j + 1], moving=X["IF"], accumulate=False)
        for q in range(nq):
            nisa.tensor_copy(dst=incr[q], src=pit[q], engine=VE)
        pib = []
        for q in range(nq):  # every partition holds the row of counts
            gs = GQ[q][1]
            pib.append(nl.ndarray((128, gs, 128), dtype=F32, buffer=nl.psum))
            for j in range(gs):
                nisa.nc_matmul(dst=pib[q][:, j, :], stationary=X["ONES1"], moving=incr[q][:, j, :], accumulate=False)
        for q in range(nq):
            nisa.tensor_copy(dst=incb[q], src=pib[q], engine=VE)
        hi, mid, lo = [], [], []
        for r in range(n):
            h, m, lw = _split3(X["SALL"][ii[r]], 128, NT)
            hi.append(h)
            mid.append(m)
            lo.append(lw)
        dgt, tmp, tmf = [], [], []
        for q in range(nq):  # the rows' page groups as 8-bit digits: row j's [d0 | d1 | d2] at [:, j, :]
            gs = GQ[q][1]
            dgt.append(_sb((128, gs, 3 * NG), BF16))
            tmp.append(_sb((128, gs, 3 * NG), I32))
            tmf.append(_sb((128, gs, 3 * NG)))
        for q in range(nq):
            nisa.tensor_scalar(dst=tmp[q][:, :, 0:NG], data=pgi[q], op0=nl.right_shift, operand0=16, engine=VE)
            nisa.tensor_scalar(dst=tmp[q][:, :, NG:2 * NG], data=pgi[q], op0=nl.right_shift, operand0=8,
                               op1=nl.bitwise_and, operand1=255, engine=VE)
            nisa.tensor_scalar(dst=tmp[q][:, :, 2 * NG:3 * NG], data=pgi[q], op0=nl.bitwise_and, operand0=255,
                               engine=VE)
            nisa.tensor_copy(dst=tmf[q], src=tmp[q], engine=VE)
            nisa.tensor_copy(dst=dgt[q], src=tmf[q], engine=VE)
            nisa.tensor_copy(dst=t0f[q], src=t0i[q], engine=VE)
            nisa.tensor_scalar(dst=t0f[q], data=t0f[q], op0=nl.multiply, operand0=float(PPP), engine=VE)
        LV, CP = [], []
        for q in range(nq):
            LV.append(_sb((128, GQ[q][1] * KT)))
            CP.append(_sb((128, GQ[q][1] * KT)))
        pk, m1, ex, dd, ohk = [], [], [], [], []
        for q in range(nq):
            gs = GQ[q][1]
            pk.append(_sb((128, gs * KT)))  # the partition of the k-th selected pool (slot s = j KT + t)
            m1.append(_sb((128, gs, KT, 128)))
            ex.append(_sb((128, gs * KT)))  # its exclusive count
            dd.append(_sb((128, gs * KT)))
            ohk.append(_sb((128, gs * KT, 128), BF16))  # [k, s, p] = (p == partition(k))
        for q in range(nq):  # [k, j, t, c] = incb[k, j, c] <= KV[k, t]
            gs = GQ[q][1]
            nisa.tensor_tensor(dst=m1[q],
                               data1=incb[q].ap(pattern=[[gs * 128, 128], [128, gs], [0, KT], [1, 128]], offset=0),
                               data2=KV.ap(pattern=[[KT, 128], [0, gs], [1, KT], [0, 128]], offset=0),
                               op=nl.less_equal, engine=VE)
        for q in range(nq):
            nisa.tensor_reduce(dst=pk[q], op=nl.add, data=m1[q].reshape((128, GQ[q][1] * KT, 128)), axis=2)
        for q in range(nq):
            gs = GQ[q][1]
            nisa.tensor_tensor(dst=m1[q], data1=m1[q],
                               data2=incb[q].ap(pattern=[[gs * 128, 128], [128, gs], [0, KT], [1, 128]], offset=0),
                               op=nl.multiply, engine=VE)
        for q in range(nq):
            nisa.tensor_reduce(dst=ex[q], op=nl.maximum, data=m1[q].reshape((128, GQ[q][1] * KT, 128)), axis=2)
        for q in range(nq):
            nisa.tensor_scalar(dst=pk[q], data=pk[q], op0=nl.minimum, operand0=127.0, engine=VE)
        for q in range(nq):  # [k, j, t] = KV[k, t] - ex[k, j, t]
            gs = GQ[q][1]
            nisa.tensor_tensor(dst=dd[q].reshape((128, gs, KT)),
                               data1=KV.ap(pattern=[[KT, 128], [0, gs], [1, KT]], offset=0),
                               data2=ex[q].reshape((128, gs, KT)), op=nl.subtract, engine=VE)
        for q in range(nq):
            nsl = GQ[q][1] * KT
            nisa.tensor_tensor(dst=ohk[q], data1=IO.ap(pattern=[[512, 128], [0, nsl], [1, 128]], offset=0),
                               data2=pk[q].ap(pattern=[[nsl, 128], [1, nsl], [0, 128]], offset=0), op=nl.equal,
                               engine=VE)
        oht, rs = [], []
        for q in range(nq):
            oht.append(_sb((128, GQ[q][1] * KT, 128), BF16))  # [p, s, k]
            rs.append(_sb((128, GQ[q][1] * KT, NT), BF16))  # the k-th pool's partition row of selections
        WT = _c2s_waves(GQ, KT, 4)  # the transposes: 4 [128, 128] fp32 slots a PSUM tile (2 KB)
        for w in range(len(WT)):
            wv = WT[w]
            pt = []
            for i in range(len(wv)):
                q, c0, sz = wv[i]
                pt.append(nl.ndarray((128, sz, 128), dtype=F32, buffer=nl.psum))
                for u in range(sz):
                    nisa.nc_matmul(dst=pt[i][:, u, :], stationary=ohk[q][:, c0 + u, :], moving=X["IB"],
                                   accumulate=False)
            for i in range(len(wv)):
                q, c0, sz = wv[i]
                nisa.tensor_copy(dst=oht[q][:, c0:c0 + sz, :], src=pt[i], engine=VE)
        WS = _c2s_waves(GQ, KT, max(1, 512 // NT))  # [128, NT] fp32 slots a PSUM tile (sz NT <= 512: one bank)
        for w in range(len(WS)):
            wv = WS[w]
            prs = []
            for i in range(len(wv)):
                q, c0, sz = wv[i]
                prs.append(nl.ndarray((128, sz, NT), dtype=F32, buffer=nl.psum))
                for u in range(sz):
                    nisa.nc_matmul(dst=prs[i][:, u, :], stationary=oht[q][:, c0 + u, :],
                                   moving=s01b[q][:, (c0 + u) // KT, :], accumulate=False)
            for i in range(len(wv)):
                q, c0, sz = wv[i]
                nisa.tensor_copy(dst=rs[q][:, c0:c0 + sz, :], src=prs[i], engine=VE)
        rts = []
        for q in range(nq):  # slot s's selections transposed, chunk c at [0:cw, s, c 128:]
            rts.append(_sb((128, GQ[q][1] * KT, nch * 128), BF16))
        WR = _c2s_waves(GQ, KT, max(1, 512 // (nch * 128)))  # [128, nch 128] fp32 slots a PSUM tile
        for w in range(len(WR)):
            wv = WR[w]
            ptr = []
            for i in range(len(wv)):
                q, c0, sz = wv[i]
                ptr.append(nl.ndarray((128, sz, nch * 128), dtype=F32, buffer=nl.psum))
                for u in range(sz):
                    for c in range(nch):
                        nisa.nc_matmul(dst=ptr[i][0:cw, u, c * 128:(c + 1) * 128],
                                       stationary=rs[q][:, c0 + u, c * cw:(c + 1) * cw], moving=X["IB"],
                                       accumulate=False)
            for i in range(len(wv)):
                q, c0, sz = wv[i]
                nisa.tensor_copy(dst=rts[q][0:cw, c0:c0 + sz, :], src=ptr[i][0:cw, :, :], engine=VE)
        col, sv, pds = [], [], []
        for q in range(nq):
            col.append(_sb((128, GQ[q][1] * KT)))
            sv.append(_sb((128, GQ[q][1] * KT)))
            pds.append(_sb((128, GQ[q][1] * KT, 3 * NG)))
        # _compact2's rule (its note above nch == 1): an accumulation group gets a PSUM tile of its own, and the ops
        # that read one stay per row and k tile; lone starts into slices are exact
        if nch == 1:
            for w in range(len(WS)):
                wv = WS[w]
                prc = []
                for i in range(len(wv)):  # inclusive count along the row: a lone start a slot, into slices of one tile
                    q, c0, sz = wv[i]
                    prc.append(nl.ndarray((128, sz, NT), dtype=F32, buffer=nl.psum))
                    for u in range(sz):
                        nisa.nc_matmul(dst=prc[i][:, u, :], stationary=rts[q][0:cw, c0 + u, 0:128],
                                       moving=X["UT"][0:cw, 0:NT], accumulate=False)
                for i in range(len(wv)):
                    q, c0, sz = wv[i]
                    m3 = _sb((128, sz, NT))
                    nisa.tensor_tensor(dst=m3, data1=prc[i],
                                       data2=dd[q].ap(pattern=[[GQ[q][1] * KT, 128], [1, sz], [0, NT]], offset=c0),
                                       op=nl.less_equal, engine=VE)
                    nisa.tensor_reduce(dst=col[q][:, c0:c0 + sz], op=nl.add, data=m3, axis=2)
        else:
            for t in range(KT):
                prc = []
                for q in range(nq):  # inclusive count along the row: nch accumulated chunks, a PSUM tile a row and tile
                    for j in range(GQ[q][1]):
                        p_ = nl.ndarray((128, NT), dtype=F32, buffer=nl.psum)
                        for c in range(nch):
                            nisa.nc_matmul(dst=p_, stationary=rts[q][0:cw, j * KT + t, c * 128:(c + 1) * 128],
                                           moving=X["UT"][0:cw, c * NT:(c + 1) * NT], accumulate=(c > 0))
                        prc.append(p_)
                for q in range(nq):
                    for j in range(GQ[q][1]):
                        s = j * KT + t
                        m3 = _sb((128, NT))
                        nisa.tensor_scalar_reduce(dst=m3, data=prc[GQ[q][0] + j], op0=nl.less_equal,
                                                  operand0=dd[q][:, s:s + 1], reduce_op=nl.add,
                                                  reduce_res=col[q][:, s:s + 1])
        for q in range(nq):
            nisa.tensor_scalar(dst=col[q], data=col[q], op0=nl.minimum, operand0=float(NT - 1), engine=VE)
        ohc = []
        for q in range(nq):
            nsl = GQ[q][1] * KT
            ohc.append(_sb((128, nsl, NT)))
            nisa.tensor_tensor(dst=ohc[q], data1=IO.ap(pattern=[[512, 128], [0, nsl], [1, NT]], offset=0),
                               data2=col[q].ap(pattern=[[nsl, 128], [1, nsl], [0, NT]], offset=0), op=nl.equal,
                               engine=VE)
        for t in range(KT):
            psc = []
            for q in range(nq):  # the partition row of scores: split3's pieces accumulated, a PSUM tile a row and tile
                g0, gs = GQ[q]
                for j in range(gs):
                    p_ = nl.ndarray((128, NT), dtype=F32, buffer=nl.psum)
                    nisa.nc_matmul(dst=p_, stationary=oht[q][:, j * KT + t, :], moving=hi[g0 + j], accumulate=False)
                    nisa.nc_matmul(dst=p_, stationary=oht[q][:, j * KT + t, :], moving=mid[g0 + j], accumulate=True)
                    nisa.nc_matmul(dst=p_, stationary=oht[q][:, j * KT + t, :], moving=lo[g0 + j], accumulate=True)
                    psc.append(p_)
            for q in range(nq):  # _compact2's scr, in place in ohc (read nowhere else)
                g0, gs = GQ[q]
                for j in range(gs):
                    s = j * KT + t
                    nisa.tensor_tensor(dst=ohc[q][:, s, :], data1=ohc[q][:, s, :], data2=psc[g0 + j], op=nl.multiply,
                                       engine=VE)
        for q in range(nq):  # the scores: ohc one-hot, so one nonzero term a sum (exact in any order)
            nisa.tensor_reduce(dst=sv[q], op=nl.add, data=ohc[q], axis=2)
        # the partition's page digits, to SBUF (a vector op may not read both operands from PSUM): [128, 3 NG] fp32
        # slots a PSUM tile
        WD = _c2s_waves(GQ, KT, max(1, 512 // (3 * NG)))
        for w in range(len(WD)):
            wv = WD[w]
            ppd = []
            for i in range(len(wv)):
                q, c0, sz = wv[i]
                ppd.append(nl.ndarray((128, sz, 3 * NG), dtype=F32, buffer=nl.psum))
                for u in range(sz):
                    nisa.nc_matmul(dst=ppd[i][:, u, :], stationary=oht[q][:, c0 + u, :],
                                   moving=dgt[q][:, (c0 + u) // KT, :], accumulate=False)
            for i in range(len(wv)):
                q, c0, sz = wv[i]
                nisa.tensor_copy(dst=pds[q][:, c0:c0 + sz, :], src=ppd[i], engine=VE)
        pgf, ci, gi, ji, gf, jf, ohg, page, valid, ninf, lp, row = [], [], [], [], [], [], [], [], [], [], [], []
        for q in range(nq):
            nsl = GQ[q][1] * KT
            pgf.append(_sb((128, nsl, NG)))
            ci.append(_sb((128, nsl), I32))
            gi.append(_sb((128, nsl), I32))
            ji.append(_sb((128, nsl), I32))
            gf.append(_sb((128, nsl)))
            jf.append(_sb((128, nsl)))
            ohg.append(_sb((128, nsl, NG)))
            page.append(_sb((128, nsl)))
            valid.append(_sb((128, nsl)))
            ninf.append(_sb((128, nsl)))
            lp.append(_sb((128, nsl)))
            row.append(_sb((128, nsl)))
        for q in range(nq):
            nisa.scalar_tensor_tensor(dst=pgf[q], data=pds[q][:, :, 0:NG], op0=nl.multiply, operand0=256.0, op1=nl.add,
                                      operand1=pds[q][:, :, NG:2 * NG])
        for q in range(nq):
            nisa.scalar_tensor_tensor(dst=pgf[q], data=pgf[q], op0=nl.multiply, operand0=256.0, op1=nl.add,
                                      operand1=pds[q][:, :, 2 * NG:3 * NG])
        for q in range(nq):
            nisa.tensor_copy(dst=ci[q], src=col[q], engine=VE)
        for q in range(nq):
            nisa.tensor_scalar(dst=gi[q], data=ci[q], op0=nl.right_shift, operand0=LPP, engine=VE)
        for q in range(nq):
            nisa.tensor_scalar(dst=ji[q], data=ci[q], op0=nl.bitwise_and, operand0=PPP - 1, engine=VE)
        for q in range(nq):
            nisa.tensor_copy(dst=gf[q], src=gi[q], engine=VE)
        for q in range(nq):
            nisa.tensor_copy(dst=jf[q], src=ji[q], engine=VE)
        for q in range(nq):
            nsl = GQ[q][1] * KT
            nisa.tensor_tensor(dst=ohg[q], data1=IO.ap(pattern=[[512, 128], [0, nsl], [1, NG]], offset=0),
                               data2=gf[q].ap(pattern=[[nsl, 128], [1, nsl], [0, NG]], offset=0), op=nl.equal,
                               engine=VE)
        for q in range(nq):
            nisa.tensor_tensor(dst=ohg[q], data1=ohg[q], data2=pgf[q], op=nl.multiply, engine=VE)
        for q in range(nq):
            nisa.tensor_reduce(dst=page[q], op=nl.add, data=ohg[q], axis=2)
        for q in range(nq):  # [k, j, t] = KV[k, t] < incb[k, j, 127] (row j's count)
            gs = GQ[q][1]
            nisa.tensor_tensor(dst=valid[q].reshape((128, gs, KT)),
                               data1=KV.ap(pattern=[[KT, 128], [0, gs], [1, KT]], offset=0),
                               data2=incb[q].ap(pattern=[[gs * 128, 128], [128, gs], [0, KT]], offset=127),
                               op=nl.less, engine=VE)
        # lval = valid ? score : NEG_INF; lp = valid ? partition NT + col : 0; rows = valid ? page 8 + j : page0 8
        for q in range(nq):
            nisa.tensor_scalar(dst=ninf[q], data=valid[q], op0=nl.subtract, operand0=1.0, op1=nl.multiply,
                               operand1=-NEG_INF, engine=VE)
        for q in range(nq):
            nisa.tensor_tensor(dst=sv[q], data1=sv[q], data2=valid[q], op=nl.multiply, engine=VE)
        for q in range(nq):
            nisa.tensor_tensor(dst=LV[q], data1=sv[q], data2=ninf[q], op=nl.add, engine=VE)
        for q in range(nq):
            nisa.scalar_tensor_tensor(dst=lp[q], data=pk[q], op0=nl.multiply, operand0=float(NT), op1=nl.add,
                                      operand1=col[q])
        for q in range(nq):
            nisa.tensor_tensor(dst=lp[q], data1=lp[q], data2=valid[q], op=nl.multiply, engine=VE)
        for q in range(nq):
            nisa.tensor_scalar(dst=CP[q], data=lp[q], op0=nl.multiply, operand0=float(X["A"]), op1=nl.add,
                               operand1=X["RK"], engine=VE)
        for q in range(nq):
            nisa.scalar_tensor_tensor(dst=row[q], data=page[q], op0=nl.multiply, operand0=float(PPP), op1=nl.add,
                                      operand1=jf[q])
        for q in range(nq):  # [k, j, t] -= t0f[k, j] (the row's page-0 row)
            gs = GQ[q][1]
            nisa.tensor_tensor(dst=row[q].reshape((128, gs, KT)), data1=row[q].reshape((128, gs, KT)),
                               data2=t0f[q].ap(pattern=[[gs, 128], [1, gs], [0, KT]], offset=0), op=nl.subtract,
                               engine=VE)
        for q in range(nq):
            nisa.tensor_tensor(dst=row[q], data1=row[q], data2=valid[q], op=nl.multiply, engine=VE)
        for q in range(nq):
            gs = GQ[q][1]
            nisa.tensor_tensor(dst=row[q].reshape((128, gs, KT)), data1=row[q].reshape((128, gs, KT)),
                               data2=t0f[q].ap(pattern=[[gs, 128], [1, gs], [0, KT]], offset=0), op=nl.add,
                               engine=VE)
        bk = X["B"] * keep
        for q in range(nq):  # out [3, B, keep]: lval, cpool, rows (fp32, exact), a DMA a row from its slice
            g0, gs = GQ[q]
            for j in range(gs):
                o0 = bb[g0 + j] * keep
                dma_static(dst=X["out"].ap(pattern=[[1, 128], [128, KT]], offset=o0), src=LV[q][:, j * KT:(j + 1) * KT])
                dma_static(dst=X["out"].ap(pattern=[[1, 128], [128, KT]], offset=bk + o0),
                           src=CP[q][:, j * KT:(j + 1) * KT])
                dma_static(dst=X["out"].ap(pattern=[[1, 128], [128, KT]], offset=2 * bk + o0),
                           src=row[q][:, j * KT:(j + 1) * KT])

    @nki.jit
    def dsa_shell_sel2_kernel(qT, wb, pkc, ppg, cand, identb, rank, keep: int, A: int, nbits: int, rbits: int,
                                   rev: int, dge: int = 0, nr: int = 4, qr: int = 4, pf: int = 2, cs: int = 0):
        """D2 v2. kernels/dsa_index.py's arguments (qT bf16 [D, B Hi], wb fp32 [B, 128, 4 Hi], pkc bf16 [Npages, PPP D],
        ppg int32 [128, B, NG], cand fp32 [B, P], identb bf16 [128, 128]) plus rank fp32 [1, 1], keep (a multiple of
        128, < P), A, nbits (2^nbits > P), rbits. Returns fp32 [3, B, keep]: lval, cpool and rows (pool rows, exact
        integers < 2^24), attend_cp's local list, the served form's bits (one output: one core barrier).

        The rows alternate between the programs (dsa_index's spl 1). Each program scores its rows with dsa_index's own
        static software pipeline (no device loop: the earlier D2's per-unit loops cost ~7-13 us each), keeps every row's
        score tile in SBUF (the pool p NT + c at [p, c]), selects all its rows in ONE dsa_topk _select (the rounds'
        latency paid once per program; the positions are the pools' indices, the served tie order) and compacts each
        row on the engines (_compact2: no HBM round trip). cs 1 (served): the compaction with the rows
        stacked, up to _C2S_G a tile (_compact2s: a vector op once a group of rows; the same bits)."""
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
else:
    dsa_shell_sel2_kernel = None


def _rev2() -> int:
    from . import dsa_index, dsa_topk

    return (SH.kernel_rev(__file__, "if nki is not None:  # D2v2's kernel", "    dsa_shell_sel2_kernel = None")
            ^ dsa_index.REV ^ dsa_topk.REV ^ SH.REV)


REV2 = (_rev2() ^ zlib.crc32(repr((D, NEG_INF, PPP, RF, LPP, BIG_T, VISIBLE_T)).encode())  # the constants its
        ^ SH.kernel_rev(__file__, "    F32, BF16, I32 = nl.float32", "    def _sb("))  # kernels read, and the aliases


def select2(q: torch.Tensor, w: torch.Tensor, pkc: torch.Tensor, table: torch.Tensor, nloc: torch.Tensor, keep: int,
            A: int, rank: torch.Tensor, *, rbits: int = 2, dge: int = 0, simulate: bool = False,
            grid: int | None = None, nr: int | None = None, pf: int = 2, cs: int = 0):
    """D2 v2 (dsa_shell_sel2_kernel) on a Neuron device, in the NKI simulator, or emulate() on the host: (lval
    [B, keep] fp32, cpool [B, keep] fp32, rows [B, keep] int32). Arguments as emulate()'s; rank a [1] or [] tensor
    (model.long_grp_index). Its inputs are dsa_index.scores' (qT, wb, the page groups, cand from nloc). cs 1: the
    compaction with the rows stacked (_compact2s), the same bits (emulate() has no compaction form: cs changes nothing
    there)."""
    from . import dsa_index

    if cs not in (0, 1):
        raise ValueError(f"select2: cs={cs}, 0 or 1")
    if q.device.type == "cpu" and not simulate and not SH.HOST_SIM:  # shell.HOST_SIM: the kernel for host tensors too
        lval, cpool, rows, _ = emulate(q, w, pkc, table, nloc, keep, A, int(rank.reshape(())))
        return lval, cpool, rows
    if dsa_shell_sel2_kernel is None:
        raise RuntimeError("the NKI DSA shell needs the nki package (the Neuron venv)")
    if grid is None:
        from .. import platform

        grid = 1 if simulate else platform.nki_grid()
    B, Hi, _ = q.shape
    NPg = table.shape[1]
    P = NPg * PPP
    cand = torch.where(torch.arange(P, device=q.device).view(1, P) < nloc.view(-1, 1), 0.0, NEG_INF)
    args = dict(qT=q.to(torch.bfloat16).reshape(B * Hi, D).t().contiguous(),
                wb=w.float().repeat(1, 4).unsqueeze(1).expand(B, 128, 4 * Hi).contiguous(),
                pkc=pkc.to(torch.bfloat16), ppg=dsa_index.page_groups(table), cand=cand.float(),
                identb=torch.eye(128, device=q.device).to(torch.bfloat16), rank=rank.to(torch.float32).reshape(1, 1),
                keep=int(keep), A=int(A), nbits=int(P.bit_length()), rbits=int(rbits), rev=REV2, dge=int(dge),
                nr=dsa_index.NR if nr is None else int(nr), qr=dsa_index.QR, pf=int(pf), cs=int(cs))
    k = dsa_shell_sel2_kernel
    if simulate:
        import ml_dtypes
        import numpy as np

        def to_np(t):
            if not isinstance(t, torch.Tensor):
                return t
            return t.float().numpy().astype(ml_dtypes.bfloat16) if t.dtype == torch.bfloat16 else t.numpy()

        o = nki.simulate(k[grid] if grid > 1 else k)(**{n: to_np(t) for n, t in args.items()})
        o = torch.from_numpy(np.asarray(o, dtype=np.float32))
        return o[0], o[1], o[2].to(torch.int32)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    o = wrap_nki(k)[grid](**args)  # one output: a bare tensor (memory wrap-nki-single-output-bare)
    return o[0], o[1], o[2].to(torch.int32)
