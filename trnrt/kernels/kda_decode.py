"""One decode step of KDA (Kimi delta attention) for B sequences as one NKI kernel: each row's recurrent state is
read from the state pool, updated and written back IN PLACE, and the step's output returned. models/linear_attn.py's
decode form takes it on the device. A build that runs the KDA shell (EngineConfig.kda_shell, models/hybrid.py) sends
every decode step up to the shell's row cap to the shell instead, so there only the steps above the cap reach this
kernel; a build without the shell runs every KDA layer's decode step here.

Why a kernel: left to the compiler, the updated state is laid out for its arithmetic and then rewritten into the
pool's layout with 1024 vector-engine STREAM_TRANSPOSEs per layer (32 x 32 fp32 blocks, ~0.24 ms per layer),
scheduled at the end of the graph with the pool scatters, and that tail grew faster than the rows: 0.68 / 2.75 /
7.72 / 13.76 ms for a graph's 9 KDA layers at 8 / 16 / 32 / 64 rows per DP group (a GLM-5.3-Flash decode layer-group
graph at tp 32, DP attention 4, 8 heads of [128, 128] fp32 per rank; neuron-explorer replays on 32 cores of a
trn1.32xlarge, Neuron SDK 2.32). On a trn2.48xlarge, the first form of this kernel together with the other decode
kernels and sequence-parallel decode streams took a decode-only step at 16 / 32 / 64 rows per DP group from 143.6 /
219.7 / 1438.7 ms to 110.7 / 154.8 / 264.2 ms (Neuron SDK 2.32).

Arithmetic (linear_attn.recurrent_step, fp32): e = exp(g); r0 = (q e)^T S, r1 = (k e)^T S;
delta = beta (v - r1); o = r0 + (q . k) delta; S' = diag(e) S + k delta^T.

No PSUM accumulation group. The first form of the kernel computed r1 - v as one: (k e)^T S started in PSUM, then
(-v)^T I accumulated onto it. On trn2 that group lost its started term in each LNC program's first row: head 2's
r1 read as exactly 0 there (delta = beta v), on nearly every call, while every other head and row was exact (a
standalone probe on one trn2 core; the state error was rank 1 along k). The NKI simulator does not model it. So
every matmul here is a lone start, and the two terms meet in a vector op: -beta is folded into r1's stationary
column (KEN = -beta k e, built in the prologue from a K = 1 broadcast matmul of -beta's row), v's and k's rows come
to SBUF by DMA (no PE transpose), and delta = beta v + (-beta r1) is ONE vector op with one PSUM operand. A form
that kept the PE rows and copied them through the scalar engine was exact too, but +33.6 us (+20%) a 24-row call.

Layout. q, k and g come in as row-major [B, H, 128] and are transposed once on the tensor engine into [128 d, B H]
column tiles (q e, -beta k e, k, e), with beta, -beta and q . k as rows. Per sequence row, its states are one DMA
from the pool row slots[0, b] into SBUF [128 dk, H, 128 dv] (the pool's row-major layout puts dk on the partitions,
no transpose); a read row >= R (padding, a sequence at position 0) skips the DMA and leaves the zeroed tile, i.e. the
state starts at zero as torch.where(keep, S, 0) does. The state is always the MOVING operand of the fp32 matmuls (an
fp32 stationary operand is rounded on trn2; the moving one is not).

Quadrant batching. Heads go in chunks of CH = 4, head h of a chunk at partition 32 h: its two matvec rows (r0 and
-beta r1, out [1, D]) are written to PSUM partition 32 h (a PSUM destination at a partition offset must sit on a
quadrant boundary: partitions 1-3 are a neuronx-cc internal error), its v / k / beta / q . k rows sit at partition
32 h, and delta and o are ONE vector op per chunk on [97, D] (every lane computes independently, so each head's
elements see the per-head form's ops in its order; the lanes between quadrants are zeroed and discarded). The K = 1
outer product k delta^T reads k and delta at partition 32 h. On trn2 every output and the state were bitwise the
per-head form's. Pipeline: state rows loaded PD rows ahead into a ring of RING; stage A of chunk g (8 matvecs, 2
vector ops, the o rows out) beside stage B of chunk g - 1 (4 outer products, 4 state updates, the rows' states back
to the pool). Every DMA is on hardware DGE (the runtime-row state loads and stores skip out-of-range rows): left to
the compiler they went to GpSimd's software DGE, the kernel's busiest engine (1.67 ms of a 24-row step).

The pool is returned as the kernel's output (NKI detect_must_alias: an input returned as an output).
"""

from __future__ import annotations

import torch

CH = 4  # heads a chunk: one per PSUM quadrant
PD = 4  # state rows loaded ahead of the chunk being computed
RING = 4 + PD  # state rows in flight: chunk g - 1 (stage B), chunk g (stage A) and the PD rows ahead (H = 2)


def emulate(pool: torch.Tensor, slots: torch.Tensor, q, k, v, g, beta):
    """The kernel's arithmetic in torch (CPU): (o [B, H, D], the pool after the step). slots [2, B]: read rows
    (>= R: start from zero) and write rows, written in row order (a later row wins a shared write row)."""
    from ..models.linear_attn import recurrent_step

    R = pool.shape[0]
    B = q.shape[0]
    S = torch.zeros(B, *pool.shape[1:], dtype=torch.float32)
    rd = slots[0].long()
    ok = rd < R
    S[ok] = pool[rd[ok]].float()
    o, S2 = recurrent_step(q.float(), k.float(), v.float(), g.float(), beta.float(), S)
    out = pool.clone()
    for b, w in enumerate(slots[1].long().tolist()):
        if w < R:
            out[w] = S2[b]
    return o, out


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    F32 = nl.float32
    I32 = nl.int32

    def _sb(shape, dt=None):
        return nl.ndarray(shape, dtype=dt or F32, buffer=nl.sbuf)

    def _ps(shape):
        return nl.ndarray(shape, dtype=F32, buffer=nl.psum)

    def _dq(dst, src, oob: bool = False):
        """A DMA on hardware DGE (out-of-range rows skipped when oob)."""
        if oob:
            nisa.dma_copy(dst=dst, src=src, oob_mode=nisa.oob_mode.skip, dge_mode=nisa.dge_mode.hwdge)
        else:
            nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge)

    def _load(X, b):
        """Row b's states into its ring entry; a read row >= R skips the DMA."""
        B, H, D = X["B"], X["H"], X["D"]
        S = X["Sr"][b % RING]
        nisa.memset(dst=S, value=0.0, engine=nisa.gpsimd_engine)
        ridx = X["sl"].ap(pattern=[[2 * B, 1], [1, 1]], offset=b)
        _dq(S, X["pool"].ap(pattern=[[D, 128], [D * D, H], [1, D]], offset=0, scalar_offset=ridx, indirect_dim=0), oob=True)

    def _stage_a(X, g):
        """Chunk g's heads: r0 / -beta r1 rows at PSUM partition 32 h, then delta and o as one vector op each on [n, D];
        the o rows out."""
        H, D, c_lo, c_hi = X["H"], X["D"], X["c_lo"], X["c_hi"]
        c0 = c_lo + g * CH
        nh = min(CH, c_hi - c0)
        n = 32 * (nh - 1) + 1
        p0, p1 = X["P0"][g % 2], X["P1"][g % 2]
        dl, orow = X["Dq"][g % 2], X["Oq"][g % 2]
        for h in range(nh):
            c, a = c0 + h, 32 * h
            b, j = c // H, c % H
            S = X["Sr"][b % RING]
            nisa.nc_matmul(dst=p0[a:a + 1, :], stationary=X["QE"][:, c:c + 1], moving=S[:, j, :], accumulate=False)
            nisa.nc_matmul(dst=p1[a:a + 1, :], stationary=X["KEN"][:, c:c + 1], moving=S[:, j, :], accumulate=False)
        gl = (c0 - c_lo) // CH
        nisa.scalar_tensor_tensor(dst=dl[0:n, :], data=X["Vq"][gl][0:n, :], op0=nl.multiply, operand0=X["Bq"][0:n, gl:gl + 1],
                                  op1=nl.add, operand1=p1[0:n, :])
        nisa.scalar_tensor_tensor(dst=orow[0:n, :], data=dl[0:n, :], op0=nl.multiply, operand0=X["Qq"][0:n, gl:gl + 1],
                                  op1=nl.add, operand1=p0[0:n, :])
        for h in range(nh):
            c, a = c0 + h, 32 * h
            _dq(X["of"][:, c * D:(c + 1) * D], orow[a:a + 1, :])

    def _stage_b(X, g):
        """Chunk g's heads: the K = 1 outer product from partition 32 h, S' = S e + k delta^T; a row's states back to
        the pool after its last head."""
        B, H, D, c_lo, c_hi = X["B"], X["H"], X["D"], X["c_lo"], X["c_hi"]
        c0 = c_lo + g * CH
        nh = min(CH, c_hi - c0)
        dl = X["Dq"][g % 2]
        gl = (c0 - c_lo) // CH
        for h in range(nh):
            c, a = c0 + h, 32 * h
            b, j = c // H, c % H
            S, Sx = X["Sr"][b % RING], X["Sn"][b % RING]
            po = X["Po"][c % 4]
            nisa.nc_matmul(dst=po, stationary=X["Kq"][gl][a:a + 1, :], moving=dl[a:a + 1, :], accumulate=False)
            nisa.scalar_tensor_tensor(dst=Sx[:, j, :], data=S[:, j, :], op0=nl.multiply, operand0=X["E"][:, c:c + 1],
                                      op1=nl.add, operand1=po)
            if j == H - 1:
                widx = X["sl"].ap(pattern=[[2 * B, 1], [1, 1]], offset=B + b)
                _dq(X["pool"].ap(pattern=[[D, 128], [D * D, H], [1, D]], offset=0, scalar_offset=widx, indirect_dim=0), Sx, oob=True)

    def _body(pool, slots, q, k, v, g, beta, ident, spl: int):
        """kda_decode_kernel's body: the prologue, the quadrant placement tiles, the chunk pipeline."""
        R, H, D, _ = pool.shape
        B = q.shape[0]
        N = B * H
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if spl == 1 and nl.program_ndim() != 0 else (1, 0)
        b_lo, b_hi = (0, B) if npg == 1 else ((0, (B + 1) // 2) if pid == 0 else ((B + 1) // 2, B))
        c_lo, c_hi = b_lo * H, b_hi * H
        o = nl.ndarray((B, H, D), dtype=F32, buffer=nl.shared_hbm)
        sl = _sb((1, 2 * B), I32)
        nisa.dma_copy(dst=sl, src=slots.reshape((1, 2 * B)))
        I = _sb((128, 128))
        nisa.dma_copy(dst=I, src=ident)
        ones = _sb((128, 1))
        nisa.memset(dst=ones, value=1.0, engine=nisa.gpsimd_engine)
        onesr = _sb((1, 128))
        nisa.memset(dst=onesr, value=1.0, engine=nisa.gpsimd_engine)
        # Column tiles [128 d, N], column c = b H + h: q e, k e, -beta k e (KEN), k, e; on partition 0: beta, -beta and
        # q . k.
        QE, KE, KEN, KT, E = _sb((128, N)), _sb((128, N)), _sb((128, N)), _sb((128, N)), _sb((128, N))
        BR = _sb((1, N))
        nisa.dma_copy(dst=BR, src=beta.reshape((1, N)))
        NBR = _sb((1, N))  # -beta
        nisa.tensor_scalar(dst=NBR, data=BR, op0=nl.multiply, operand0=-1.0, engine=nisa.vector_engine)
        QK = _sb((1, N))
        q2, k2, g2 = q.reshape((N, D)), k.reshape((N, D)), g.reshape((N, D))
        pq4 = _ps((128, N))  # q . k rows at partitions 0 / 32 / 64 / 96 (the same matmul, its output at each quadrant)
        for t in range(0, N, 128):
            n = min(128, N - t)
            rows = []
            for src in (q2, k2, g2):
                a = _sb((128, D))
                nisa.dma_copy(dst=a[0:n, :], src=src[t:t + n, :])
                rows.append(a)
            pt = _ps((128, 3, 128))
            for i in range(3):
                nisa.nc_matmul(dst=pt[:, i, 0:n], stationary=rows[i][0:n, :], moving=I[0:n, 0:n], accumulate=False)
            nisa.activation(dst=E[:, t:t + n], op=nl.exp, data=pt[:, 2, 0:n])
            nisa.tensor_tensor(dst=QE[:, t:t + n], data1=pt[:, 0, 0:n], data2=E[:, t:t + n], op=nl.multiply)
            nisa.tensor_tensor(dst=KE[:, t:t + n], data1=pt[:, 1, 0:n], data2=E[:, t:t + n], op=nl.multiply)
            nisa.activation(dst=KT[:, t:t + n], op=nl.copy, data=pt[:, 1, 0:n])
            pb = _ps((128, 128))  # -beta on every partition: a K = 1 matmul of a ones row against -beta's row
            nisa.nc_matmul(dst=pb[:, 0:n], stationary=onesr, moving=NBR[:, t:t + n], accumulate=False)
            nisa.tensor_tensor(dst=KEN[:, t:t + n], data1=pb[:, 0:n], data2=KE[:, t:t + n], op=nl.multiply)
            qkp = _sb((128, 128))
            nisa.tensor_tensor(dst=qkp[:, 0:n], data1=pt[:, 0, 0:n], data2=KT[:, t:t + n], op=nl.multiply)
            pq = _ps((1, 128))
            nisa.nc_matmul(dst=pq[:, 0:n], stationary=ones, moving=qkp[:, 0:n], accumulate=False)  # sum over d
            nisa.activation(dst=QK[:, t:t + n], op=nl.copy, data=pq[:, 0:n])
            for h in range(CH):
                nisa.nc_matmul(dst=pq4[32 * h:32 * h + 1, t:t + n], stationary=ones, moving=qkp[:, 0:n], accumulate=False)
        # The quadrant placements: v / k rows at partition 32 h by a DMA each; beta's row to each quadrant partition by
        # a DMA; then head h's column of each chunk copied onto partition 32 h (strided one-lane copies).
        NCH = -(-(c_hi - c_lo) // CH)
        v2 = v.reshape((N, D))
        Vq, Kq = [], []
        for gl in range(NCH):
            vt, kt = _sb((128, D)), _sb((128, D))
            nisa.memset(dst=vt, value=0.0, engine=nisa.gpsimd_engine)
            nisa.memset(dst=kt, value=0.0, engine=nisa.gpsimd_engine)
            for h in range(min(CH, c_hi - c_lo - gl * CH)):
                c, a = c_lo + gl * CH + h, 32 * h
                _dq(vt[a:a + 1, :], v2[c:c + 1, :])
                _dq(kt[a:a + 1, :], k2[c:c + 1, :])
            Vq.append(vt)
            Kq.append(kt)
        BRq = _sb((128, N))
        Bq, Qq = _sb((128, NCH)), _sb((128, NCH))
        nisa.memset(dst=Bq, value=0.0, engine=nisa.gpsimd_engine)
        nisa.memset(dst=Qq, value=0.0, engine=nisa.gpsimd_engine)
        for h in range(CH):
            a = 32 * h
            _dq(BRq[a:a + 1, :], beta.reshape((1, N)))
            m = max(0, (c_hi - c_lo - h + CH - 1) // CH)  # the chunks that hold a head h (the device tracer's len() takes no range)
            if m:
                nisa.tensor_copy(dst=Bq[a:a + 1, 0:m], src=BRq[a:a + 1, c_lo + h:c_hi:CH])
                nisa.tensor_copy(dst=Qq[a:a + 1, 0:m], src=pq4[a:a + 1, c_lo + h:c_hi:CH])
        of = o.reshape((1, N * D))
        # Rings: every entry a tensor of its own (the compiler tracks dependencies per tensor).
        Sr, Sn = [], []
        for _ in range(RING):
            Sr.append(_sb((128, H, D)))
            Sn.append(_sb((128, H, D)))
        P0 = [_ps((128, D)), _ps((128, D))]
        P1 = [_ps((128, D)), _ps((128, D))]
        for t_ in P0:  # the lanes between quadrants are read (and discarded) by the chunk ops: defined values
            nisa.memset(dst=t_, value=0.0)
        for t_ in P1:
            nisa.memset(dst=t_, value=0.0)
        Po = [_ps((128, D)), _ps((128, D)), _ps((128, D)), _ps((128, D))]
        Dq = [_sb((128, D)), _sb((128, D))]
        Oq = [_sb((128, D)), _sb((128, D))]
        X = {"pool": pool, "sl": sl, "Sr": Sr, "Sn": Sn, "P0": P0, "P1": P1, "Po": Po, "Dq": Dq, "Oq": Oq, "QE": QE, "KEN": KEN, "E": E,
             "Vq": Vq, "Kq": Kq, "Bq": Bq, "Qq": Qq, "of": of, "B": B, "H": H, "D": D, "c_lo": c_lo, "c_hi": c_hi}
        rows_a = CH // H  # rows a chunk (H 2: 2)
        assert H == 2, "the ring's sizing is for H 2"
        for b in range(b_lo, min(b_lo + rows_a + PD, b_hi)):  # chunk 0's rows and the PD rows ahead of it
            _load(X, b)
        for gg in range(NCH + 1):
            if gg < NCH:
                for b in range(b_lo + gg * rows_a + PD, min(b_lo + (gg + 1) * rows_a + PD, b_hi)):  # keep PD rows ahead of chunk gg
                    if gg >= 1:
                        _load(X, b)
                _stage_a(X, gg)
            if gg >= 1:
                _stage_b(X, gg - 1)
        if npg > 1:  # LNC: both programs' rows written before either ends (whatever runs next reads all of them)
            nisa.core_barrier(data=o, cores=(0, 1))
            nisa.core_barrier(data=pool, cores=(0, 1))
        return o, pool

    @nki.jit
    def kda_decode_kernel(pool, slots, q, k, v, g, beta, ident, rev: int, spl: int = 0):
        """pool fp32 [R, H, D, D] (rows slots[1] rewritten in place and returned), slots int32 [2, B] (read rows,
        R = none; write rows), q (scaled), k, v, g (log decay) fp32 [B, H, D], beta fp32 [B, H], ident fp32
        [128, 128] the identity (PE transposes); H = 2 heads (the ring's sizing). rev: this module's kernel source
        revision. Returns o fp32 [B, H, D] and pool.

        LNC split (spl 1, grid 2: trn2 at LNC=2; platform.LNC_SPLIT names kda_decode): each program (physical core) runs
        the chunk pipeline of its half of the rows, b_lo .. b_hi - 1, on the column tiles both build (rows are
        independent: disjoint o rows and state rows; padding rows share a scratch write row, whose content no one
        reads); both barrier on o and the pool before the kernel ends. spl 0 or one program: every row."""
        return _body(pool, slots, q, k, v, g, beta, ident, spl)
else:
    kda_decode_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source and the module constants it reads, passed as the static argument `rev`:
    the compile-cache key does not include NKI kernel source (kernels/delta_rule.py _kernel_rev)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kda_decode_kernel = None", a)
    return zlib.crc32(repr((CH, PD, RING)).encode(), zlib.crc32(src[a:b].encode()))


REV = _kernel_rev()


def takes(x: torch.Tensor, Dk: int, Dv: int, kda: bool) -> bool:
    """Whether a decode step of these shapes runs as the kernel: on the device (x not on the CPU), KDA (per-key
    decay), 128 x 128 heads."""
    return x.device.type != "cpu" and kda and Dk == 128 and Dv == 128


def decode_step(pool: torch.Tensor, state_slot: torch.Tensor, keep: torch.Tensor, q, k, v, g, beta) -> torch.Tensor:
    """One decode step on the device, in the caller's graph: pool fp32 [R, H, D, D] updated in place at rows
    state_slot [B]; keep [B] bool (False: the row starts from a zero state, as torch.where(keep, S, 0) does);
    q (scaled), k, v, g fp32 [B, H, D]; beta [B, H]. Returns o fp32 [B, H, D]."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if kda_decode_kernel is None:
        raise RuntimeError("the NKI KDA decode kernel needs the nki package (the Neuron venv)")
    R = pool.shape[0]
    rd = torch.where(keep, state_slot, torch.full_like(state_slot, R))
    slots = torch.stack([rd, state_slot]).to(torch.int32)
    ident = torch.eye(128, dtype=torch.float32, device=q.device)
    spl = {"spl": 1} if platform.nki_grid() == 2 and platform.lnc_split("kda_decode") else {}  # in the key only then
    o, _ = wrap_nki(kda_decode_kernel)[platform.nki_grid()](
        pool=pool, slots=slots, q=q.float().contiguous(), k=k.float().contiguous(), v=v.float().contiguous(),
        g=g.float().contiguous(), beta=beta.float().contiguous(), ident=ident, rev=REV, **spl)
    return o
