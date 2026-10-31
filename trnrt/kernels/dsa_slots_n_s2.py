"""kernels/dsa_slots_n_s.py's kernel reading its rows and bias in their NATURAL layout, for the DECODE FALLBACK only (dsa_slots_n.attend with
fallback=True, from dsa_slots_c.attend and the DSA shell; the context-parallel prefill keeps dsa_slots_n_s and its host
permutes).

Why: dsa_slots_n.attend's host permutes of the fallback's rows ([.., NCH, 128] -> [.., 128, NCH]) and bias run in XLA
every decode step, though the fallback runs 0 iterations unless a row has over C live slots. In the 8-row decode graph
the rows permute is 11 sets of 128 small SBUF -> DRAM DMAs per program, ~124 us summed each on the device. Here the
reads are strided access patterns inside the kernel's run-time loop, so the layout work happens only when the fallback
runs. The DSA shell's attend kernel runs _row_nat as its folded fallback's row.

Numerics: the same elements land in the same SBUF tiles in the same order (the DMAs' source and destination patterns
are the permuted layout's, rewritten); every instruction after them is dsa_slots._row's: o and lse are dsa_slots_n_s's
bit for bit (in the simulator). REV: this module's kernel source, its _row_nat included, and dsa_slots_n_s.REV."""

from __future__ import annotations

import zlib

from . import dsa_slots as ds
from . import dsa_slots_n as dsn
from . import dsa_slots_n_s as dns

# The decode fallbacks that take the natural layouts: dsa_slots_c's ("c"); the DSA shell's ("shell") keeps the permuted
# form.
NAT = True
SITES = ("c",)

if dsn.dsa_slots_n_kernel is not None:
    from .dsa_slots import KTR, _pair, _psum, _sb  # the kernel tracer resolves names, not module attributes

    nki, nisa, nl = ds.nki, ds.nisa, ds.nl
    F32, BF16, I32 = nl.float32, nl.bfloat16, nl.int32
    KP = ds.KP
    VE = nisa.vector_engine

    def _row_nat(X, it, k):
        """dsa_slots._row with rows_t and bias read in their NATURAL layout (rows [.., NCH, 128], bias
        [.., NCH 128 KP] as slot-major, kp-minor), strided into the tiles the served layout fills."""
        H, R, LC, NCH, T, rpi = X["H"], X["R"], X["LC"], X["NCH"], X["T"], X["rpi"]
        x = k % 2
        IB = X["IB"]
        PT, PS, PP, PO, PQ = X["psum"]
        # the row's slot indices, q_lat, bias (broadcast over its H head partitions)
        rs = X["RS"][x]
        nisa.dma_copy(dst=rs, src=X["rows_t"].ap(pattern=[[1, 128], [128, NCH]], offset=k * 128 * NCH,
                                                 scalar_offset=it, indirect_dim=0))
        Kv = X["K"][x]
        for ch in range(NCH):
            nisa.dma_copy(dst=Kv[ch], src=X["kc"].ap(pattern=[[KP * R, 128], [1, KP * R]], offset=0,
                                                       vector_offset=rs.ap(pattern=[[NCH, 128], [1, 1]], offset=ch),
                                                       indirect_dim=0))
        qr = X["QR"][x]
        nisa.dma_copy(dst=qr, src=X["q_lat"].ap(pattern=[[R, H], [1, R]], offset=k * H * R, scalar_offset=it,
                                                indirect_dim=0))
        S = X["S"][x]
        if X["qkw"]:
            # the row's bias on ONE partition; the scores' PSUM adds it to every head by a K = 1 fp32 matmul (a DMA
            # broadcasting it to the H head partitions wrote H x 10 KB per row)
            br = X["BR"][x]
            nisa.dma_copy(dst=br.ap(pattern=[[T, 1], [KP * 128, NCH], [128, KP], [1, 128]], offset=0),
                          src=X["bias"].ap(pattern=[[T, 1], [128 * KP, NCH], [1, KP], [KP, 128]], offset=k * T,
                                           scalar_offset=it, indirect_dim=0))
        else:
            # S starts as the row's bias, broadcast over its H head partitions (one DMA, stride 0 over the partitions)
            nisa.dma_copy(dst=S.ap(pattern=[[T, H], [KP * 128, NCH], [128, KP], [1, 128]], offset=0),
                          src=X["bias"].ap(pattern=[[0, H], [128 * KP, NCH], [1, KP], [KP, 128]], offset=k * T,
                                           scalar_offset=it, indirect_dim=0))
        # q_lat^T [128 r, LC, H]
        for lc in range(LC):
            nisa.nc_matmul(dst=PQ[:, lc, :], stationary=qr[:, lc * 128:(lc + 1) * 128], moving=IB[0:H, 0:H],
                           accumulate=False)
        qt = X["QT"][x]
        nisa.tensor_copy(dst=qt, src=PQ, engine=VE)
        if X["qkw"]:
            # every block's K^T into the row's [128 r, LC, T] tile, then the scores 512 tokens per matmul (one
            # stationary load per (group, lc) instead of per (block, lc)), each group one group behind its transposes
            kta = X["KTA"][x]
            NG = T // 512
            for g in range(NG + 1):
                if g < NG:
                    for b in range(4):
                        i = g * 4 + b
                        ch = i // KP
                        t = i % KP
                        pt = PT[i % 2]
                        for lc in range(LC):
                            c0 = t * R + lc * 128
                            nisa.nc_matmul(dst=pt[:, lc, :], stationary=Kv[ch][:, c0:c0 + 128], moving=IB,
                                           accumulate=False)
                        if i % 2 == 0:
                            nisa.tensor_copy(dst=kta[:, :, i * 128:(i + 1) * 128], src=pt, engine=VE)
                        else:
                            nisa.activation(dst=kta[:, :, i * 128:(i + 1) * 128], op=nl.copy, data=pt)
                if g >= 1:
                    j = (g - 1) * 512
                    ps = PS[(g - 1) % 2]
                    for lc in range(LC):
                        nisa.nc_matmul(dst=ps, stationary=qt[:, lc, :], moving=kta[:, lc, j:j + 512],
                                       accumulate=True if lc > 0 else False)
                    # + bias / scale on every head (0 stays an exact 0; a masked token's NEG_INF / scale), then x scale
                    nisa.nc_matmul(dst=ps, stationary=X["RSC"][:, 0:H], moving=br[:, j:j + 512], accumulate=True)
                    nisa.activation(dst=S[:, j:j + 512], op=nl.copy, data=ps, scale=X["scale"])
        else:
            for ch in range(NCH):
                for t in range(KP):
                    i = ch * KP + t
                    pt = PT[i % 2]
                    for lc in range(LC):  # the block's latent, R on the partitions
                        c0 = t * R + lc * 128
                        nisa.nc_matmul(dst=pt[:, lc, :], stationary=Kv[ch][:, c0:c0 + 128], moving=IB,
                                       accumulate=False)
                    kt = X["KT"][i % KTR]
                    if i % 2 == 0:
                        nisa.tensor_copy(dst=kt, src=pt, engine=VE)
                    else:
                        nisa.activation(dst=kt, op=nl.copy, data=pt)
                    ps = PS[i % 2]
                    for lc in range(LC):
                        nisa.nc_matmul(dst=ps[:, 0:128], stationary=qt[:, lc, :], moving=kt[:, lc, :],
                                       accumulate=(lc > 0))
                    j = i * 128
                    nisa.scalar_tensor_tensor(dst=S[:, j:j + 128], data=ps[:, 0:128], op0=nl.multiply,
                                              operand0=X["scale"], op1=nl.add, operand1=S[:, j:j + 128])
        # softmax over the row's T tokens per head: P = exp(S - max) in bf16 and its fp32 sum in one ACT instruction
        mx, nmx, den, rden = X["MX"][x], X["NMX"][x], X["DEN"][x], X["RDEN"][x]
        nisa.tensor_reduce(dst=mx, op=nl.maximum, data=S, axis=1)
        nisa.tensor_scalar(dst=nmx, data=mx, op0=nl.multiply, operand0=-1.0, engine=VE)
        P = X["P"][x]
        nisa.activation(dst=P, op=nl.exp, data=S, bias=nmx, scale=1.0, reduce_op=nl.add, reduce_res=den,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        # o = P K: P's 128-token blocks transposed to [128, H] against the gathered latent as it is
        for ch in range(NCH):
            for t in range(KP):
                i = ch * KP + t
                j = i * 128
                g = 512 // H
                pp = PP[:, (i % g) * H:(i % g + 1) * H]
                nisa.nc_matmul(dst=pp, stationary=P[:, j:j + 128], moving=IB[0:H, 0:H], accumulate=False)
                pT = X["PTb"][i % 2]
                nisa.activation(dst=pT, op=nl.copy, data=pp)
                nisa.nc_matmul(dst=PO, stationary=pT, moving=Kv[ch][:, t * R:(t + 1) * R], accumulate=(i > 0))
        nisa.reciprocal(dst=rden, data=den)
        ob = X["OB"][x]
        nisa.tensor_scalar(dst=ob, data=PO, op0=nl.multiply, operand0=rden, engine=VE)
        nisa.dma_copy(dst=X["o"].ap(pattern=[[R, H], [1, R]], offset=k * H * R, scalar_offset=it, indirect_dim=0),
                      src=ob)
        if X["want_lse"]:
            ls = X["LS"][x]
            nisa.activation(dst=ls, op=nl.log, data=den)
            nisa.tensor_tensor(dst=ls, data1=ls, data2=mx, op=nl.add, engine=VE)
            nisa.dma_copy(dst=X["lse"].ap(pattern=[[1, H], [1, 1]], offset=k * H, scalar_offset=it, indirect_dim=0),
                          src=ls)


    @nki.jit
    def dsa_slots_n_s_kernel(q_lat, kc, rows_t, bias, identb, n_iter, scale: float, fp8: int, want_lse: int,
                                  rev: int, qkw: int = 1):
        """dsa_slots_n_kernel's arguments and outputs; at grid 2 each program runs half of every iteration's
        rows."""
        NI, rpi, H, R = q_lat.shape
        NCH = rows_t.shape[2]
        LC = R // 128
        T = NCH * KP * 128
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
                 want_lse=want_lse, qkw=qkw)
        IB = _sb((128, 128), BF16)
        nisa.dma_copy(dst=IB, src=identb)
        X["IB"] = IB
        X["scale"] = scale
        X["K"] = []
        for _ in range(2):
            kv = []
            for _ in range(NCH):
                kv.append(_sb((128, KP * R), KD))
            X["K"].append(kv)
        X["KT"] = []
        if qkw:
            X["KTA"] = _pair((128, LC, T), BF16)
            X["BR"] = _pair((1, T), F32)
            X["RSC"] = _sb((1, 128))
            nisa.memset(dst=X["RSC"], value=1.0 / scale)
        else:
            for _ in range(KTR):
                X["KT"].append(_sb((128, LC, 128), BF16))
        X["RS"] = _pair((128, NCH), I32)
        X["QR"] = _pair((H, R), BF16)
        X["QT"] = _pair((128, LC, H), BF16)
        X["S"] = _pair((H, T), F32)
        X["P"] = _pair((H, T), BF16)
        X["OB"] = _pair((H, R), F32)
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
            X["psum"] = _psum(H, LC)
            for k in range(k_lo, k_hi):
                _row_nat(X, it, k)

        nl.fori_loop(0, rt, body)
        if split:  # every row written, both halves, before either program returns
            nisa.core_barrier(data=o, cores=(0, 1))
            if want_lse:
                nisa.core_barrier(data=lse, cores=(0, 1))
        if want_lse:
            return o, lse
        return o
else:
    dsa_slots_n_s_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source (its _row_nat included), XOR dsa_slots_n_s.REV: the static `rev`."""
    src = open(__file__).read()
    a = src.index("if dsn.dsa_slots_n_kernel is not None:")
    b = src.index("    dsa_slots_n_s_kernel = None", a)
    return zlib.crc32(src[a:b].encode()) ^ dns.REV


REV = _kernel_rev()
