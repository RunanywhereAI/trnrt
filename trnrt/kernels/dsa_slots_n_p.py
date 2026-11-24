"""kernels/dsa_slots_n_s.py (the context-parallel prefill's slot attention, each iteration's rows split over the two
physical cores at LNC=2) with the rows software-pipelined: prefill calls only (models/mla.py _cp_slots_attend, under
EngineConfig.dsa_slots_pipe_rows).

Why: the split kernel emits each row whole (loads, q^T, K^T, scores, softmax, P^T / P K, the store) before the next, with
its loads double-buffered. The PE (the busiest engine: 8.0 of 18.8 us a program-row in-graph on trn2) then idles
through every row's softmax and copy phases, and a row's gather can only start once the row two back has finished its
P K.

What changes, and only this: the program order and the buffer slots. A row is cut into the same instructions as
dsa_slots._row, in four stages:
- load(k): the row's slot indices, its gather, q_lat and bias, into ring slot k % D (D = la + 2) of the load buffers;
- front(k): q^T, K^T, the scores (into PS[k % 2]) and the scaled copy into S;
- soft(k): the max, -max, and P = exp(S - max) with its sum;
- back(k): P^T and P K over the blocks in the same order, 1 / sum, o, lse.
They are emitted as load(k + la) front(k) back(k - 1) soft(k), so the PE runs row k's front while row k - 1's softmax
is in flight, and the gathers run la rows ahead. Every instruction has the operands, dtypes, flags and accumulation
groups of the split kernel's (each PSUM accumulation group alone in its tile), so every output element is computed by
the same arithmetic in the same order: o and lse are the split kernel's bit for bit (tests/test_dsa_slots_n_p.py, the
simulator). With sbat > 1 a program stores its rows' o / lse sbat rows a DMA (data movement only).

Measured on trn2 (a 1,024-row call, 128 slots, the bf16 bias): 8.93 ms a call for the split kernel; pipelined at 64 rows
an iteration with one store a program an iteration (sbat 32, the stores on the sync engine's hardware DGE), 4.21 ms.

A separate module: dsa_slots_n.py / dsa_slots_n_s.py / dsa_slots.py keep their source, REVs and graph keys.
"""

from __future__ import annotations

import zlib

from . import dsa_slots as ds
from . import dsa_slots_n as dsn

# Rows per device-loop iteration of _cp_slots_attend's calls (0: off, the split kernel's call). Each program stores its
# half of an iteration's rows in one DMA (sbat rpi / 2), on the sync engine's hardware DGE. The engine sets it on every
# rank (EngineConfig.dsa_slots_pipe_rows) before the shard is built; read when a prefill graph is traced.
PIPE_ROWS = 0
LA = 2  # rows the loads run ahead (load ring depth LA + 2: rows k - 1 .. k + LA live)

if dsn.dsa_slots_n_kernel is not None:
    from .dsa_slots import _pair, _sb  # the kernel tracer resolves names, not module attributes

    nki, nisa, nl = ds.nki, ds.nisa, ds.nl
    F32, BF16, I32 = nl.float32, nl.bfloat16, nl.int32
    VEP = nisa.vector_engine
    KPP = ds.KP

    def _ring(shape, dt, depth):  # no comprehensions in kernel code: the device tracer refuses them
        r = []
        for _ in range(depth):
            r.append(_sb(shape, dt))
        return r

    def _dma(dst, src, mode):
        """dma_copy in a descriptor-generation mode (see the kernel's docstring; the tracer takes no keyword expansion)."""
        if mode == 1:
            nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        elif mode == 2:
            nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
        elif mode == 3:
            nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge)
        else:
            nisa.dma_copy(dst=dst, src=src)

    def _load(X, it, k, j):
        """Row k's indices, gather, q_lat and bias into ring slot j % D."""
        R, NCH, T, H = X["R"], X["NCH"], X["T"], X["H"]
        L = j % X["D"]
        rs = X["RS"][L]
        LK = X["ldk"]
        _dma(rs, X["rows_t"].ap(pattern=[[NCH, 128], [1, NCH]], offset=k * 128 * NCH, scalar_offset=it, indirect_dim=0), LK)
        Kv = X["K"][L]
        for ch in range(NCH):
            nisa.dma_copy(dst=Kv[ch], src=X["kc"].ap(pattern=[[KPP * R, 128], [1, KPP * R]], offset=0,
                                                       vector_offset=rs.ap(pattern=[[NCH, 128], [1, 1]], offset=ch),
                                                       indirect_dim=0))
        _dma(X["QR"][L], X["q_lat"].ap(pattern=[[R, H], [1, R]], offset=k * H * R, scalar_offset=it, indirect_dim=0), LK)
        _dma(X["BR"][L], X["bias"].ap(pattern=[[T, 1], [1, T]], offset=k * T, scalar_offset=it, indirect_dim=0), LK)

    def _front(X, j):
        """q^T, every block's K^T, the scores (+ bias / scale, x scale) into S[j % 2]."""
        H, LC, NCH, T = X["H"], X["LC"], X["NCH"], X["T"]
        L, x = j % X["D"], j % 2
        IB = X["IB"]
        PT, PS, PQ = X["PT"], X["PS"], X["PQ"]
        Kv, qr, br = X["K"][L], X["QR"][L], X["BR"][L]
        for lc in range(LC):
            nisa.nc_matmul(dst=PQ[:, lc, :], stationary=qr[:, lc * 128:(lc + 1) * 128], moving=IB[0:H, 0:H],
                           accumulate=False)
        qt = X["QT"][x]
        nisa.tensor_copy(dst=qt, src=PQ, engine=VEP)
        kta, S = X["KTA"][x], X["S"][x]
        NG = T // 512
        for g in range(NG + 1):
            if g < NG:
                for b in range(4):
                    i = g * 4 + b
                    ch = i // KPP
                    t = i % KPP
                    pt = PT[i % X["npt"]]  # the copy engine stays the split kernel's (i % 2)
                    for lc in range(LC):
                        c0 = t * X["R"] + lc * 128
                        nisa.nc_matmul(dst=pt[:, lc, :], stationary=Kv[ch][:, c0:c0 + 128], moving=IB,
                                       accumulate=False)
                    if i % 2 == 0:
                        nisa.tensor_copy(dst=kta[:, :, i * 128:(i + 1) * 128], src=pt, engine=VEP)
                    else:
                        nisa.activation(dst=kta[:, :, i * 128:(i + 1) * 128], op=nl.copy, data=pt)
            if g >= 1:
                jj = (g - 1) * 512
                ps = PS[(x + g - 1) % 2]  # the split kernel's PS[(g - 1) % 2], offset by the row's parity
                for lc in range(LC):
                    nisa.nc_matmul(dst=ps, stationary=qt[:, lc, :], moving=kta[:, lc, jj:jj + 512],
                                   accumulate=True if lc > 0 else False)
                nisa.nc_matmul(dst=ps, stationary=X["RSC"][:, 0:H], moving=br[:, jj:jj + 512], accumulate=True)
                nisa.activation(dst=S[:, jj:jj + 512], op=nl.copy, data=ps, scale=X["scale"])

    def _soft(X, j):
        """P = exp(S - max) in bf16 and its fp32 sum in one ACT instruction (dsa_slots._row's softmax)."""
        x = j % 2
        mx, nmx, den = X["MX"][x], X["NMX"][x], X["DEN"][x]
        S = X["S"][x]
        nisa.tensor_reduce(dst=mx, op=nl.maximum, data=S, axis=1)
        nisa.tensor_scalar(dst=nmx, data=mx, op0=nl.multiply, operand0=-1.0, engine=VEP)
        nisa.activation(dst=X["P"][x], op=nl.exp, data=S, bias=nmx, scale=1.0, reduce_op=nl.add, reduce_res=den,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)

    def _back(X, it, k, j):
        """o = P K over the blocks (P^T against the gathered latent), / sum; o and lse to HBM."""
        H, R, NCH = X["H"], X["R"], X["NCH"]
        L, x = j % X["D"], j % 2
        IB, PP, PO = X["IB"], X["PP"], X["PO"]
        Kv, P = X["K"][L], X["P"][x]
        nb = NCH * KPP
        for i in range(nb):
            ch = i // KPP
            t = i % KPP
            jj = i * 128
            gg = 512 // H
            pp = PP[:, (i % gg) * H:(i % gg + 1) * H]
            nisa.nc_matmul(dst=pp, stationary=P[:, jj:jj + 128], moving=IB[0:H, 0:H], accumulate=False)
            pT = X["PTb"][i % 2]
            nisa.activation(dst=pT, op=nl.copy, data=pp)
            nisa.nc_matmul(dst=PO, stationary=pT, moving=Kv[ch][:, t * R:(t + 1) * R], accumulate=(i > 0))
        rden = X["RDEN"][x]
        nisa.reciprocal(dst=rden, data=X["DEN"][x])
        SK, sb = X["stk"], X["sbat"]
        if sb == 1:  # one store a row (the split kernel's form)
            ob = X["OB"][x]
            nisa.tensor_scalar(dst=ob, data=PO, op0=nl.multiply, operand0=rden, engine=VEP)
            _dma(X["o"].ap(pattern=[[R, H], [1, R]], offset=k * H * R, scalar_offset=it, indirect_dim=0), ob, SK)
            if X["want_lse"]:
                ls = X["LS"][x]
                nisa.activation(dst=ls, op=nl.log, data=X["DEN"][x])
                nisa.tensor_tensor(dst=ls, data1=ls, data2=X["MX"][x], op=nl.add, engine=VEP)
                _dma(X["lse"].ap(pattern=[[1, H], [1, 1]], offset=k * H, scalar_offset=it, indirect_dim=0), ls, SK)
            return
        # sb rows a store: row j's o and lse land in slot j % sb of the batch tiles (parity (j // sb) % nobb), the same
        # instructions into a slice; one DMA per batch writes the batch's consecutive rows (data movement only)
        c, xb = j % sb, (j // sb) % X["nobb"]
        obb = X["OBB"][xb]
        nisa.tensor_scalar(dst=obb[:, c, :], data=PO, op0=nl.multiply, operand0=rden, engine=VEP)
        if X["want_lse"]:
            lsb = X["LSB"][xb]
            nisa.activation(dst=lsb[:, c:c + 1], op=nl.log, data=X["DEN"][x])
            nisa.tensor_tensor(dst=lsb[:, c:c + 1], data1=lsb[:, c:c + 1], data2=X["MX"][x], op=nl.add, engine=VEP)
        if c == sb - 1 or j == X["m"] - 1:
            nr = c + 1
            k0 = k - c
            _dma(X["o"].ap(pattern=[[R, H], [H * R, nr], [1, R]], offset=k0 * H * R, scalar_offset=it,
                           indirect_dim=0), obb[:, 0:nr, :], SK)
            if X["want_lse"]:
                _dma(X["lse"].ap(pattern=[[1, H], [H, nr]], offset=k0 * H, scalar_offset=it, indirect_dim=0), lsb[:, 0:nr],
                     SK)

    @nki.jit
    def dsa_slots_n_p_kernel(q_lat, kc, rows_t, bias, identb, n_iter, scale: float, fp8: int, want_lse: int,
                                  rev: int, qkw: int = 1, plant: int = 0, la: int = LA, sfirst: int = 0, npt: int = 2,
                                  sbat: int = 1, sdge: int = 0, ldge: int = 0):
        """dsa_slots_n_s_kernel's arguments and outputs (qkw 1 only); at grid 2 each program runs half of every
        iteration's rows, pipelined. Schedule only (every form the same arithmetic): la, the rows the loads run ahead
        (ring depth la + 2); sfirst, a row's softmax emitted before the previous row's back (else after); npt, the K^T
        PSUM banks (2, or 3 with the eighth bank); sbat, rows of o / lse a store DMA (1: one a row); sdge / ldge, the
        stores' / the direct loads' descriptor generation (0 the compiler's choice, 1 hardware DGE on the sync engine,
        2 on the scalar engine, 3 hardware DGE, engine unnamed). plant must be 0 (the argument keeps the call's
        signature)."""
        assert plant == 0 and 0 <= sdge <= 3 and 0 <= ldge <= 3, "dsa_slots_n_p: plant 0, DGE modes 0 .. 3"
        NI, rpi, H, R = q_lat.shape
        NCH = rows_t.shape[3]
        LC = R // 128
        T = NCH * KPP * 128
        o = nl.ndarray((NI, rpi, H, R), dtype=F32, buffer=nl.shared_hbm)
        lse = None
        if want_lse:
            lse = nl.ndarray((NI, rpi, H, 1), dtype=F32, buffer=nl.shared_hbm)
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        split = npg == 2 and rpi >= 2
        k_lo, k_hi = (0, rpi) if not split else ((0, (rpi + 1) // 2) if pid == 0 else ((rpi + 1) // 2, rpi))
        KD = kc.dtype
        if fp8:
            KD = nl.float8_e4m3
        X = dict(H=H, R=R, LC=LC, NCH=NCH, T=T, rpi=rpi, q_lat=q_lat, kc=kc, rows_t=rows_t, bias=bias, o=o, lse=lse,
                 want_lse=want_lse, D=la + 2, npt=npt, sbat=sbat)
        X["stk"] = sdge
        X["ldk"] = ldge
        X["m"] = k_hi - k_lo
        IB = _sb((128, 128), BF16)
        nisa.dma_copy(dst=IB, src=identb)
        X["IB"] = IB
        X["scale"] = scale
        X["K"] = []
        for _ in range(la + 2):
            kv = []
            for _ in range(NCH):
                kv.append(_sb((128, KPP * R), KD))
            X["K"].append(kv)
        X["KTA"] = _pair((128, LC, T), BF16)
        BD = BF16 if bias.dtype == BF16 else F32  # a bf16 bias (dsa_slots_n.BIAS_BF16): the bias matmul in bf16
        X["BR"] = _ring((1, T), BD, la + 2)
        X["RSC"] = _sb((1, 128), BD)
        nisa.memset(dst=X["RSC"], value=1.0 / scale)
        X["RS"] = _ring((128, NCH), I32, la + 2)
        X["QR"] = _ring((H, R), BF16, la + 2)
        X["QT"] = _pair((128, LC, H), BF16)
        X["S"] = _pair((H, T), F32)
        X["P"] = _pair((H, T), BF16)
        X["nobb"] = 2 if sbat < k_hi - k_lo else 1  # one batch a program an iteration: one tile (SBUF: 64 KB at 32 rows)
        if sbat == 1:
            X["OB"] = _pair((H, R), F32)
        else:
            X["OBB"] = []
            X["LSB"] = []
            for _ in range(X["nobb"]):
                X["OBB"].append(_sb((H, sbat, R), F32))
                X["LSB"].append(_sb((H, sbat), F32))
        X["MX"] = _pair((H, 1), F32)
        X["NMX"] = _pair((H, 1), F32)
        X["DEN"] = _pair((H, 1), F32)
        X["RDEN"] = _pair((H, 1), F32)
        X["LS"] = _pair((H, 1), F32)
        X["PTb"] = _pair((128, H), BF16)
        cnt = _sb((1, 1), I32)
        nisa.dma_copy(dst=cnt, src=n_iter)  # the run-time iteration count, the same on both programs
        rt = nisa.register_alloc()
        nisa.register_load(rt, cnt)

        def body(it):
            # dsa_slots._psum's banks: the K^T banks, two score, the P^T bank, the P K accumulator, the q_lat^T bank
            X["PT"] = []
            X["PS"] = []
            for _ in range(npt):
                X["PT"].append(nl.ndarray((128, LC, 128), dtype=F32, buffer=nl.psum))
            for _ in range(2):
                X["PS"].append(nl.ndarray((H, 512), dtype=F32, buffer=nl.psum))
            X["PP"] = nl.ndarray((128, 512), dtype=F32, buffer=nl.psum)
            X["PO"] = nl.ndarray((H, 512), dtype=F32, buffer=nl.psum)
            X["PQ"] = nl.ndarray((128, LC, H), dtype=F32, buffer=nl.psum)
            m = k_hi - k_lo
            for j in range(min(la, m)):
                _load(X, it, k_lo + j, j)
            for j in range(m + 1):
                if j + la < m:
                    _load(X, it, k_lo + j + la, j + la)
                if j < m:
                    _front(X, j)
                if sfirst and j < m:
                    _soft(X, j)
                if j >= 1:
                    _back(X, it, k_lo + j - 1, j - 1)
                if not sfirst and j < m:
                    _soft(X, j)

        nl.fori_loop(0, rt, body)
        if split:  # every row written, both halves, before either program returns
            nisa.core_barrier(data=o, cores=(0, 1))
            if want_lse:
                nisa.core_barrier(data=lse, cores=(0, 1))
        if want_lse:
            return o, lse
        return o

else:
    dsa_slots_n_p_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source (the stages and the kernel) with dsa_slots.REV, the static argument `rev`."""
    import hashlib

    src = open(__file__).read()
    a = src.index("if dsn.dsa_slots_n_kernel is not None:")
    b = src.index("    dsa_slots_n_p_kernel = None", a)
    return zlib.crc32(hashlib.sha256(repr((src[a:b], ds.REV)).encode()).digest())


REV = _kernel_rev()


def takes() -> bool:
    """Whether _cp_slots_attend's call goes here: the route, and the split kernel's route (grid 2, platform.lnc_split
    naming dsa_slots_n, the qkw score form), whose bits this kernel reproduces."""
    return PIPE_ROWS > 0 and bool(ds.QKW) and dsn._split()


def attend(q_lat, kc, rows, bias, scale: float, n, lse: bool = False, rpi: int | None = None, bb: bool | None = None,
           la: int = LA, sfirst: int = 0, npt: int = 2, sbat: int | None = None, sdge: int | None = None, ldge: int = 0):
    """dsa_slots_n.attend's prefill call (fallback False, the split kernel) through the pipelined kernel: the same
    arguments, padding, layouts and outputs. rpi: PIPE_ROWS (with sbat rpi / 2 and the stores on the sync engine's
    hardware DGE) when set, else dsa_slots_n._rpi(N); bb: the bf16 bias (dsa_slots_n.BIAS_BF16's rule) unless given."""
    import torch
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if dsa_slots_n_p_kernel is None:
        raise RuntimeError("the pipelined NKI DSA slots kernel needs the nki package (the Neuron venv)")
    kc2 = kc.reshape(kc.shape[0], -1)
    N, H, R = q_lat.shape
    NS = rows.shape[1]
    if NS % 128 or not ds.QKW:
        raise ValueError(f"dsa_slots_n_p: {NS} slots per row (a multiple of 128) and the qkw score form only")
    NCH = NS // 128
    if rpi is None and PIPE_ROWS > 0:  # the route's form
        rpi = PIPE_ROWS
        sbat = max(PIPE_ROWS // 2, 1) if sbat is None else sbat
        sdge = 1 if sdge is None else sdge
    rpi = rpi or dsn._rpi(N)
    sbat = 1 if sbat is None else sbat
    sdge = 0 if sdge is None else sdge
    Np = -(-N // rpi) * rpi
    q = q_lat.to(torch.bfloat16)
    rw, bs = rows, bias.float()
    if bb is None:
        bb = dsn.BIAS_BF16 and N > ds.PREFILL_ROWS
    if bb:
        bs = bs.to(torch.bfloat16)
    if Np != N:  # padded rows read pool row 0, every slot masked
        q = torch.cat([q, q.new_zeros(Np - N, H, R)])
        rw = torch.cat([rw, rw.new_zeros(Np - N, NS)])
        bs = torch.cat([bs, bs.new_full((Np - N, NS, ds.KP), ds.NEG_INF)])
    fp8 = kc2.dtype == torch.float8_e4m3fn
    kcp = kc2.reshape(kc2.shape[0] // ds.KP, ds.KP * R)
    NI = Np // rpi
    rows_t = rw.to(torch.int32).reshape(NI, rpi, NCH, 128).permute(0, 1, 3, 2).contiguous()
    bias_t = bs.reshape(NI, rpi, NCH, 128, ds.KP).permute(0, 1, 2, 4, 3).reshape(NI, rpi, NCH * ds.KP * 128).contiguous()
    eye = torch.eye(128, device=q_lat.device).to(torch.bfloat16)
    it = torch.floor((n.reshape(1, 1).to(torch.float32) + (rpi - 1)) * (1.0 / rpi)).to(torch.int32)
    out = wrap_nki(dsa_slots_n_p_kernel)[platform.nki_grid()](
        q_lat=q.reshape(NI, rpi, H, R).contiguous(), kc=kcp, rows_t=rows_t, bias=bias_t, identb=eye, n_iter=it,
        scale=float(scale), fp8=int(fp8), want_lse=int(lse), rev=REV, qkw=1, plant=0, la=int(la),
        sfirst=int(sfirst), npt=int(npt), sbat=int(sbat), sdge=int(sdge), ldge=int(ldge))
    if lse:
        o, ls = out
        return o.reshape(Np, H, R)[:N], ls.reshape(Np, H)[:N]
    return out.reshape(Np, H, R)[:N]
