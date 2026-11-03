"""A prefill chunk's DSA selection glue as one NKI kernel. In models/mla.py attention_cp_rows' local-K branch
(CP_LOCAL_K), between the merge's threshold selection (kernels/dsa_topk.py) and the slot attention
(kernels/dsa_slots_n.py), it does what XLA does there for the main rows:

    sel    = cp_merge_rows' mask from the selection: t the keep-th largest, every candidate above it, of those tied at t
             the `room` of the smallest context pool (the exact 21-round binary search over the pool index)
    fail   = _local_k_fail(vk, nxt, sel, keep)      the certificate (a rank's (K + 1)-th reaches the merged keep-th, or
                                                    the merge selected fewer than keep)
    chosen = _cp_chosen(sel, rank)                  this rank's K entries
    rows, bias = _cp_attend_classes(single=True)'s buffer: pool_row(lp) (_pool_row_flat's page lookup) in the first K
                 slots, the tail's row in slot K, zero padding; bias 0 / NEG_INF per slot and token

Rows on partitions (128 a tile), the A K candidates of a row on the free axis; the tiles split between the programs
(row-local work: no exchange). Every step is a compare, a select, an exact fp32 count of 0 / 1 or an integer op on
values below 2^24, so the outputs are XLA's bit for bit (class E). One form differs from XLA's ops and not its values:
  - (the tie search runs on every tile, always. A form that skipped it in a device loop whose trip came from each LNC
    program's own rows deadlocked: the programs disagreed on real data. No data-dependent control flow remains.)
  - the page lookup reads the chunk's block table from SBUF (broadcast once to every partition) by nc_n_gather within
    the partition, instead of XLA's element gather from HBM (one DMA descriptor per element: ~0.6 ms a DSA layer).
Masks are selects (tensor_copy_predicated) or compares, never x * 0: a NaN score never enters an arithmetic step, and
a compare with NaN is false, as in XLA.
"""

from __future__ import annotations

import zlib

import torch

P = 128
NEG_INF = -1e30  # models/dsa_long.NEG_INF
VISIBLE = -5e29  # models/dsa_long.VISIBLE
BIG = 3.0e38  # cp_merge_rows' / _local_k_fail's empty-minimum value


def emulate(s, c, nx, sel, lp, table, rtail, town, npos, rank: int, keep: int, ppl: int, kp: int, C: int, bits: int):
    """The kernel's contract in torch (models/mla.py's arithmetic, in its order), row-major as the merge reads them:
    s [n, A K] fp32 (every rank's K local values: dsa_topk's input), c [n, A K] fp32 (their context pools), nx [n, A]
    (every rank's (K + 1)-th value), sel [n, A K] fp32 (dsa_topk.select of s: 0 selected, NEG_INF not), lp [n, K] int
    (this rank's local pools), table [P] int (the chunk's block table), rtail [n] (the tail's pool row), town [n] bool
    (this rank owns the tail pool), npos [n, 2] (npool, positions). Returns (rows [n, C] int32, bias [n, C, kp] fp32,
    fail [n] bool)."""
    n, A = nx.shape
    K = s.shape[1] // A
    s, c = s.float(), c.float()
    valid = s > torch.full_like(s, VISIBLE)
    sel0 = sel == 0
    t = torch.where(sel0, s, torch.full_like(s, BIG)).amin(-1, keepdim=True)
    above = s > t
    tied = (s == t) & valid
    room = keep - above.to(torch.float32).sum(-1, keepdim=True)
    lim = torch.zeros(n, 1, dtype=torch.float32)
    for b in range(bits - 1, -1, -1):
        cand = lim + float(2 ** b)
        ok = (tied & (c < cand)).to(torch.float32).sum(-1, keepdim=True) < room
        lim = torch.where(ok, cand, lim)
    m = above | (tied & (c <= lim))  # [n, A K]
    vk = s.view(n, A, K)
    mv = m.view(n, A, K)
    tk = torch.where(mv, vk, torch.full_like(vk, BIG)).amin(dim=-1).amin(dim=-1)
    short = mv.to(torch.float32).sum(dim=-1).sum(dim=-1) < torch.full_like(tk, float(keep))
    nxt = nx.float()  # [n, A]
    bad = (nxt > torch.full_like(nxt, VISIBLE)) & ((nxt >= tk.view(n, 1)) | short.view(n, 1))
    fail = bad.to(torch.float32).amax(dim=-1) > 0
    chosen = mv[:, rank, :]
    lpl = lp.long()
    pg = torch.floor(lpl.to(torch.float32) * (1.0 / ppl)).to(torch.int64)
    rows_sel = table.long().view(-1)[pg.clamp(max=table.numel() - 1)] * ppl + (lpl - pg * ppl)
    sl = torch.arange(C).view(1, C)
    rs = torch.cat([rows_sel, rows_sel.new_zeros(n, C - K)], dim=1)
    ms = torch.cat([chosen, chosen.new_zeros(n, C - K)], dim=1)
    rows = torch.where(sl < K, rs, torch.where(sl == K, rtail.long().view(n, 1), torch.zeros_like(rs)))
    npool, pos = npos[:, 0].long(), npos[:, 1].long()
    t4 = torch.arange(kp).view(1, 1, kp)
    tail_tok = town.view(n, 1, 1) & (npool.view(n, 1, 1) * kp + t4 <= pos.view(n, 1, 1))
    ok2 = ((sl < K) & ms).unsqueeze(-1) | ((sl == K).unsqueeze(-1) & tail_tok)
    bias = torch.where(ok2, 0.0, NEG_INF).to(torch.float32)
    return rows.to(torch.int32), bias, fail


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    @nki.jit
    def dsa_cp_glue_kernel(s, c, nx, sel, lp, table, rtail, town, npos, oh, keep: int, ppl: int, kp: int, C: int,
                                bits: int, rev: int):
        """See the module docstring. s, c, sel [n, A K] fp32, nx [n, A] fp32 (row-major: one contiguous run a row);
        lp [n, K] int32; table [1, Pt] int32; rtail, town [n, 1] fp32; npos [n, 2] fp32 (npool, positions); oh [1, A]
        fp32 (this rank one-hot). n a multiple of 128 programs. Returns rows [n, C] int32, bias [n, C kp] fp32, fail
        [n, 1] fp32 (0 / 1)."""
        f32, bf16, i32, u32 = nl.float32, nl.bfloat16, nl.int32, nl.uint32
        VE, GP = nisa.vector_engine, nisa.gpsimd_engine
        n, A = nx.shape
        F = s.shape[1]
        K = F // A
        Pt = table.shape[1]
        sh = 0
        while (1 << sh) < ppl:
            sh += 1
        rows_out = nl.ndarray((n, C), dtype=i32, buffer=nl.shared_hbm)
        bias_out = nl.ndarray((n, C * kp), dtype=f32, buffer=nl.shared_hbm)
        fail_out = nl.ndarray((n, 1), dtype=f32, buffer=nl.shared_hbm)
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        NT = n // P
        # once: the block table on every partition (fp32: page ids < 2^24), this rank's one-hot, token offsets 0 .. kp-1
        tbi = nl.ndarray((P, Pt), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=tbi, src=table.ap(pattern=[[0, P], [1, Pt]], offset=0))
        tbf = nl.ndarray((P, Pt), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tbf, src=tbi, engine=VE)
        ohb = nl.ndarray((P, A), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ohb, src=oh.ap(pattern=[[0, P], [1, A]], offset=0))
        ioi = nl.ndarray((P, kp), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ioi, pattern=[[1, kp]], offset=0, channel_multiplier=0)
        iof = nl.ndarray((P, kp), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=iof, src=ioi, engine=VE)
        # per-tile scratch, allocated once
        sv = nl.ndarray((P, F), dtype=f32, buffer=nl.sbuf)
        cv = nl.ndarray((P, F), dtype=f32, buffer=nl.sbuf)
        sl = nl.ndarray((P, F), dtype=f32, buffer=nl.sbuf)
        nxt = nl.ndarray((P, A), dtype=f32, buffer=nl.sbuf)
        m0 = nl.ndarray((P, F), dtype=f32, buffer=nl.sbuf)
        mu = nl.ndarray((P, F), dtype=u32, buffer=nl.sbuf)
        w = nl.ndarray((P, F), dtype=f32, buffer=nl.sbuf)
        va = nl.ndarray((P, F), dtype=f32, buffer=nl.sbuf)
        ab = nl.ndarray((P, F), dtype=f32, buffer=nl.sbuf)
        ti = nl.ndarray((P, F), dtype=f32, buffer=nl.sbuf)
        x = nl.ndarray((P, F), dtype=f32, buffer=nl.sbuf)
        mk = nl.ndarray((P, F), dtype=f32, buffer=nl.sbuf)
        tt = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        na = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nt_ = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        room = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        lim = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        cand = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        cnt = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        okb = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        t2 = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        ns = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        shr = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        g1 = nl.ndarray((P, A), dtype=f32, buffer=nl.sbuf)
        bd = nl.ndarray((P, A), dtype=f32, buffer=nl.sbuf)
        fl = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        ch = nl.ndarray((P, K), dtype=f32, buffer=nl.sbuf)
        lpi = nl.ndarray((P, K), dtype=i32, buffer=nl.sbuf)
        pgi = nl.ndarray((P, K), dtype=i32, buffer=nl.sbuf)
        rmi = nl.ndarray((P, K), dtype=i32, buffer=nl.sbuf)
        rmf = nl.ndarray((P, K), dtype=f32, buffer=nl.sbuf)
        gt = nl.ndarray((P, K), dtype=f32, buffer=nl.sbuf)
        rt = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        to = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        npp = nl.ndarray((P, 2), dtype=f32, buffer=nl.sbuf)
        dd = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        t4 = nl.ndarray((P, kp), dtype=f32, buffer=nl.sbuf)
        okj = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
        b3 = nl.ndarray((P, C, kp), dtype=f32, buffer=nl.sbuf)
        bo = nl.ndarray((P, C * kp), dtype=f32, buffer=nl.sbuf)
        rf = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
        ri = nl.ndarray((P, C), dtype=i32, buffer=nl.sbuf)
        for j in range(NT // npg):
            r0 = (j * npg + pid) * P
            # this tile's rows: values (the first K of K + 1) and the (K + 1)-th, pools, the selection, the local list
            nisa.dma_copy(dst=sv, src=s.ap(pattern=[[F, P], [1, F]], offset=r0 * F))
            nisa.dma_copy(dst=cv, src=c.ap(pattern=[[F, P], [1, F]], offset=r0 * F))
            nisa.dma_copy(dst=nxt, src=nx.ap(pattern=[[A, P], [1, A]], offset=r0 * A))
            nisa.dma_copy(dst=sl, src=sel.ap(pattern=[[F, P], [1, F]], offset=r0 * F))
            nisa.dma_copy(dst=lpi, src=lp.ap(pattern=[[K, P], [1, K]], offset=r0 * K))
            nisa.dma_copy(dst=rt, src=rtail.ap(pattern=[[1, P], [1, 1]], offset=r0))
            nisa.dma_copy(dst=to, src=town.ap(pattern=[[1, P], [1, 1]], offset=r0))
            nisa.dma_copy(dst=npp, src=npos.ap(pattern=[[2, P], [1, 2]], offset=r0 * 2))
            # t: the smallest selected value (a select from BIG: an unselected NaN never enters the minimum)
            nisa.tensor_scalar(dst=m0, data=sl, op0=nl.greater, operand0=-1.0, engine=VE)
            nisa.tensor_copy(dst=mu, src=m0, engine=VE)
            nisa.tensor_scalar(dst=w, data=sl, op0=nl.multiply, operand0=0.0, op1=nl.add, operand1=BIG, engine=VE)
            nisa.tensor_copy_predicated(dst=w, src=sv, predicate=mu)
            nisa.tensor_reduce(dst=tt, op=nl.minimum, data=w, axis=1)
            nisa.tensor_scalar(dst=va, data=sv, op0=nl.greater, operand0=VISIBLE, engine=VE)
            nisa.tensor_scalar(dst=ab, data=sv, op0=nl.greater, operand0=tt[:, 0:1], engine=VE)
            nisa.scalar_tensor_tensor(dst=ti, data=sv, op0=nl.equal, operand0=tt[:, 0:1], op1=nl.multiply, operand1=va)
            nisa.tensor_reduce(dst=na, op=nl.add, data=ab, axis=1)
            nisa.tensor_reduce(dst=nt_, op=nl.add, data=ti, axis=1)
            nisa.tensor_scalar(dst=room, data=na, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=float(keep),
                               engine=VE)
            # the exact tie search, always: no data-dependent loop. Under nl.fori_loop(0, trip) with trip from THIS
            # program's tiles, the two LNC programs' trips disagreed on real data and the call deadlocked at the next
            # collective. Every program emits the same code.
            nisa.memset(dst=lim, value=0.0, engine=GP)
            for b in range(bits - 1, -1, -1):
                nisa.tensor_scalar(dst=cand, data=lim, op0=nl.add, operand0=float(2 ** b), engine=VE)
                nisa.scalar_tensor_tensor(dst=x, data=cv, op0=nl.less, operand0=cand[:, 0:1], op1=nl.multiply,
                                          operand1=ti)
                nisa.tensor_reduce(dst=cnt, op=nl.add, data=x, axis=1)
                nisa.tensor_tensor(dst=okb, data1=cnt, data2=room, op=nl.less, engine=VE)
                nisa.scalar_tensor_tensor(dst=lim, data=okb, op0=nl.multiply, operand0=float(2 ** b), op1=nl.add,
                                          operand1=lim)
            # the merge's mask: above, or tied at a pool <= the bound (disjoint, so a sum of 0 / 1)
            nisa.scalar_tensor_tensor(dst=x, data=cv, op0=nl.less_equal, operand0=lim[:, 0:1], op1=nl.multiply,
                                      operand1=ti)
            nisa.tensor_tensor(dst=mk, data1=ab, data2=x, op=nl.add, engine=VE)
            # the certificate: the merged keep-th value (a select from BIG), short, then any rank's (K + 1)-th
            nisa.tensor_copy(dst=mu, src=mk, engine=VE)
            nisa.tensor_scalar(dst=w, data=mk, op0=nl.multiply, operand0=0.0, op1=nl.add, operand1=BIG, engine=VE)
            nisa.tensor_copy_predicated(dst=w, src=sv, predicate=mu)
            nisa.tensor_reduce(dst=t2, op=nl.minimum, data=w, axis=1)
            nisa.tensor_reduce(dst=ns, op=nl.add, data=mk, axis=1)
            nisa.tensor_scalar(dst=shr, data=ns, op0=nl.less, operand0=float(keep), engine=VE)
            nisa.tensor_scalar(dst=g1, data=nxt, op0=nl.greater_equal, operand0=t2[:, 0:1], op1=nl.maximum,
                               operand1=shr[:, 0:1], engine=VE)
            nisa.scalar_tensor_tensor(dst=bd, data=nxt, op0=nl.greater, operand0=VISIBLE, op1=nl.multiply, operand1=g1)
            nisa.tensor_reduce(dst=fl, op=nl.maximum, data=bd, axis=1)
            nisa.dma_copy(dst=fail_out.ap(pattern=[[1, P], [1, 1]], offset=r0), src=fl)
            # this rank's entries: the one-hot sum over the A slices (static offsets)
            for a in range(A):
                if a == 0:
                    nisa.tensor_scalar(dst=ch, data=mk[:, 0:K], op0=nl.multiply, operand0=ohb[:, 0:1], engine=VE)
                else:
                    nisa.scalar_tensor_tensor(dst=ch, data=mk[:, a * K:(a + 1) * K], op0=nl.multiply,
                                              operand0=ohb[:, a:a + 1], op1=nl.add, operand1=ch)
            # pool_row(lp): table[min(lp >> log2 ppl, Pt - 1)] ppl + (lp & (ppl - 1)), the table read in SBUF
            nisa.tensor_scalar(dst=pgi, data=lpi, op0=nl.right_shift, operand0=sh, engine=VE)
            nisa.tensor_scalar(dst=pgi, data=pgi, op0=nl.minimum, operand0=Pt - 1, engine=VE)
            nisa.tensor_scalar(dst=rmi, data=lpi, op0=nl.bitwise_and, operand0=ppl - 1, engine=VE)
            nisa.tensor_copy(dst=rmf, src=rmi, engine=VE)
            nisa.nc_n_gather(dst=gt, data=tbf, indices=pgi.view(u32))
            # rows: [pool rows (K), the tail's row, zeros]
            nisa.memset(dst=rf, value=0.0, engine=GP)
            nisa.scalar_tensor_tensor(dst=rf[:, 0:K], data=gt, op0=nl.multiply, operand0=float(ppl), op1=nl.add,
                                      operand1=rmf)
            nisa.tensor_copy(dst=rf[:, K:K + 1], src=rt, engine=VE)
            nisa.tensor_copy(dst=ri, src=rf, engine=VE)
            nisa.dma_copy(dst=rows_out.ap(pattern=[[C, P], [1, C]], offset=r0 * C), src=ri)
            # bias: 1 where a slot's token is attended (chosen entries; the tail's tokens at or before the position),
            # then 1 -> +0 and 0 -> NEG_INF by x 1e30 - 1e30 (exact both ways, +0 as XLA's select gives)
            nisa.memset(dst=okj, value=0.0, engine=GP)
            nisa.tensor_copy(dst=okj[:, 0:K], src=ch, engine=VE)
            nisa.tensor_copy(dst=b3, src=okj.ap(pattern=[[C, P], [1, C], [0, kp]], offset=0), engine=VE)
            nisa.tensor_scalar(dst=dd, data=npp[:, 0:1], op0=nl.multiply, operand0=float(-kp), op1=nl.add,
                               operand1=npp[:, 1:2], engine=VE)
            nisa.tensor_scalar(dst=t4, data=iof, op0=nl.less_equal, operand0=dd[:, 0:1], op1=nl.multiply,
                               operand1=to[:, 0:1], engine=VE)
            b2 = b3.reshape((P, C * kp))
            nisa.tensor_copy(dst=b2[:, K * kp:(K + 1) * kp], src=t4, engine=VE)
            nisa.tensor_scalar(dst=bo, data=b2, op0=nl.multiply, operand0=1e30, op1=nl.add, operand1=-1e30, engine=VE)
            nisa.dma_copy(dst=bias_out.ap(pattern=[[C * kp, P], [1, C * kp]], offset=r0 * C * kp), src=bo)
        return rows_out, bias_out, fail_out
else:
    dsa_cp_glue_kernel = None


def _rev() -> int:
    """CRC-32 of the kernel's source span: a kernel edit re-keys every graph that calls it (the cache key ignores NKI
    source)."""
    src = open(__file__).read()
    a = src.index("if nki is not None:\n    @nki.jit")
    b = src.index("    dsa_cp_glue_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _rev()


def kernel_args(s, c, nx, sel, lp, table, rtail, town, npos, oh, keep: int, ppl: int, kp: int, C: int, bits: int) -> dict:
    """The kernel's arguments, in its dtypes and shapes."""
    n = nx.shape[0]
    return dict(s=s.float().contiguous(), c=c.float().contiguous(), nx=nx.float().contiguous(), sel=sel.float().contiguous(),
                lp=lp.to(torch.int32).contiguous(), table=table.to(torch.int32).reshape(1, -1).contiguous(),
                rtail=rtail.to(torch.float32).reshape(n, 1).contiguous(),
                town=town.to(torch.float32).reshape(n, 1).contiguous(), npos=npos.to(torch.float32).contiguous(),
                oh=oh.to(torch.float32).reshape(1, -1).contiguous(), keep=int(keep), ppl=int(ppl), kp=int(kp), C=int(C),
                bits=int(bits), rev=REV)


def takes(n: int, A: int, K: int, C: int, ppl: int, grid: int) -> bool:
    """Whether the kernel holds a call's shapes: whole tiles split evenly between the programs, the buffer holding the K
    entries and the tail, ppl a power of two."""
    return (n % (P * grid) == 0 and n > 0 and C >= K + 1 and ppl > 0 and ppl & (ppl - 1) == 0
            and A * K <= 4096)


def glue(s, c, nx, sel, lp, table, rtail, town, npos, oh, keep: int, ppl: int, kp: int, C: int, bits: int):
    """The kernel in the caller's graph at the runtime's grid: (rows [n, C] int32, bias [n, C, kp] fp32, fail [n]
    bool)."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if dsa_cp_glue_kernel is None:
        raise RuntimeError("the NKI DSA glue kernel needs the nki package (the Neuron venv)")
    n = nx.shape[0]
    rows, bias, fail = wrap_nki(dsa_cp_glue_kernel)[platform.nki_grid()](
        **kernel_args(s, c, nx, sel, lp, table, rtail, town, npos, oh, keep, ppl, kp, C, bits))
    fl = fail.view(n)
    return rows, bias.view(n, C, kp), fl > torch.full_like(fl, 0.5)  # a tensor, not a float literal (f64, NCC_ESPP004)

