"""kernels/dsa_slots.py with the number of rows given at run time: the same arithmetic and layout, over the first n rows
of its inputs only, n a device tensor (the loop register is loaded from it instead of from the static row count).

Why: under context parallelism (models/dsa_long.py, models/mla.py attention_cp) a rank's real share of a row's
selected pools is about keep / A, but a row may hold more, up to keep. Sized for the worst case, every row pays for
640 slots (39.5 us per row at 64 heads, fp8, trn1); 128 slots cost 14.9 us (14.9 / 20.5 / 27.5 / 39.5 us per row at
128 / 256 / 384 / 640 slots). The context-parallel prefill's slot classes (models/mla.py _cp_attend_classes) put the
rows that fit in 128 slots into one buffer and the rest into another,
each a full-size static buffer with its live rows first, and call this kernel once per buffer with that buffer's row
count: every row is attended over all of its slots (exact for any split), and the time follows the real split.

Rows at or past n are not computed and their outputs are not written (the caller never reads them). The rows
processed are ceil(n / RPI) RPI: the last iteration's extra rows are the buffer's own (finite) rows.

At grid 2 (trn2, LNC 2) the calls take kernels/dsa_slots_n_s.py (each iteration's rows split over the two physical
cores), and a decode call's over-capacity fallback kernels/dsa_slots_n_s2.py (the natural layouts). Its REV is this
module's kernel source and dsa_slots.REV.
"""

from __future__ import annotations

import torch

from . import shell as SH

from . import dsa_slots

NEG_INF = dsa_slots.NEG_INF
KP = dsa_slots.KP
# A prefill call's bias (0 / NEG_INF) reaches the split kernel in bf16, whose K = 1 bias matmul then runs at the bf16 rate
# instead of fp32's quarter (kernels/dsa_slots_n_s.py, BB_TAG). 0 is exact in bf16, and a masked score still underflows
# to an exact 0 in exp, so o is the fp32 form's bit for bit; a fully masked row's lse moves with bf16(NEG_INF), and the
# context-parallel combine weighs such a row 0 (or every rank equally).
BIAS_BF16 = True


def emulate(q_lat: torch.Tensor, kc: torch.Tensor, rows: torch.Tensor, bias: torch.Tensor, scale: float,
            n: torch.Tensor, lse: bool = False):
    """dsa_slots.emulate over the first n rows; the rows past n are zero (the kernel leaves them unwritten, so a caller
    that read one would see garbage on the device and zeros here)."""
    out = dsa_slots.emulate(q_lat, kc, rows, bias, scale, lse=lse)
    N = q_lat.shape[0]
    live = torch.arange(N, device=q_lat.device) < n.reshape(()).to(torch.int64)
    if lse:
        o, ls = out
        return o * live.view(N, 1, 1), ls * live.view(N, 1)
    return out * live.view(N, 1, 1)


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None and dsa_slots.nki is not None:
    from .dsa_slots import KTR, _pair, _psum, _row, _sb  # the kernel tracer resolves names, not module attributes

    F32, BF16, I32 = nl.float32, nl.bfloat16, nl.int32

    @nki.jit
    def dsa_slots_n_kernel(q_lat, kc, rows_t, bias, identb, n_iter, scale: float, fp8: int, want_lse: int,
                                rev: int, qkw: int = 1):
        """q_lat bf16 [NI, RPI, H, R]; kc [cache rows / KP, KP R] the latent cache's pool rows (fp8 = 1: e4m3, else
        bf16); rows_t int32 [NI, RPI, 128, NCH] slot ch 128 + p of a row at [.., p, ch]; bias fp32 [NI, RPI, NCH KP 128]
        (token t of slot ch 128 + p at (ch KP + t) 128 + p); identb bf16 [128, 128]; n_iter int32 [1, 1]: the
        device-loop iterations to run (RPI rows each), at most NI. Returns o fp32 [NI, RPI, H, R] (and lse fp32
        [NI, RPI, H, 1])."""
        NI, rpi, H, R = q_lat.shape
        NCH = rows_t.shape[3]
        LC = R // 128
        T = NCH * KP * 128
        o = nl.ndarray((NI, rpi, H, R), dtype=F32, buffer=nl.shared_hbm)
        lse = None
        if want_lse:
            lse = nl.ndarray((NI, rpi, H, 1), dtype=F32, buffer=nl.shared_hbm)
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
        nisa.dma_copy(dst=cnt, src=n_iter)  # the run-time iteration count (dsa_slots: iota of the static NI)
        rt = nisa.register_alloc()
        nisa.register_load(rt, cnt)

        def body(it):
            X["psum"] = _psum(H, LC)
            for k in range(rpi):
                _row(X, it, k)

        nl.fori_loop(0, rt, body)
        if want_lse:
            return o, lse
        return o
else:
    dsa_slots_n_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source and dsa_slots.REV (the row code it runs is dsa_slots')."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    dsa_slots_n_kernel = None", a)
    return zlib.crc32(src[a:b].encode() + str(dsa_slots.REV).encode())


REV = _kernel_rev()


def _rpi(N: int) -> int:
    """Rows per device-loop iteration of a call of N rows: dsa_slots.RPI_PREFILL for a prefill chunk's calls (more than
    dsa_slots.PREFILL_ROWS rows), else RPI (min(RPI, N) under dsa_slots.FIT)."""
    if dsa_slots.RPI_PREFILL and N > dsa_slots.PREFILL_ROWS:
        return dsa_slots.RPI_PREFILL
    return min(dsa_slots.RPI, N) if dsa_slots.FIT else dsa_slots.RPI


def attend(q_lat: torch.Tensor, kc: torch.Tensor, rows: torch.Tensor, bias: torch.Tensor, scale: float,
           n: torch.Tensor, lse: bool = False, fallback=False):
    """The slot attention over the first n rows (n an int tensor of one element, 0 .. N): o [N, H, R] fp32 (and lse
    [N, H]); the rows past n are unwritten on the device (zeros on the CPU). fallback: a decode call's over-capacity
    fallback, by site ("c" dsa_slots_c.attend's, "shell" the DSA shell's; True any), which may take
    kernels/dsa_slots_n_s2.py (_nat)."""
    kc2 = kc.reshape(kc.shape[0], -1)
    if q_lat.device.type == "cpu" and not SH.HOST_SIM:
        return emulate(q_lat, kc2, rows, bias, scale, n, lse)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if dsa_slots_n_kernel is None:
        raise RuntimeError("the NKI DSA slots kernel needs the nki package (the Neuron venv)")
    N, H, R = q_lat.shape
    NS = rows.shape[1]
    if NS % 128:
        raise ValueError(f"dsa_slots_n: {NS} slots per row is not a multiple of 128")
    NCH = NS // 128
    rpi = _rpi(N)
    Np = -(-N // rpi) * rpi
    q = q_lat.to(torch.bfloat16)
    rw, bs = rows, bias.float()
    bb = BIAS_BF16 and bool(dsa_slots.QKW) and N > dsa_slots.PREFILL_ROWS and not _nat(fallback) and _split()
    if bb:  # BIAS_BF16: the prefill call's bias in bf16 (the padding rows' NEG_INF too)
        bs = bs.to(torch.bfloat16)
    if Np != N:  # padded rows read pool row 0, every slot masked
        q = torch.cat([q, q.new_zeros(Np - N, H, R)])
        rw = torch.cat([rw, rw.new_zeros(Np - N, NS)])
        bs = torch.cat([bs, bs.new_full((Np - N, NS, KP), NEG_INF)])
    fp8 = kc2.dtype == torch.float8_e4m3fn
    kcp = kc2.reshape(kc2.shape[0] // KP, KP * R)
    NI = Np // rpi
    if _nat(fallback):  # the natural layouts (plain reshapes), read strided in the kernel's loop
        rows_t = rw.to(torch.int32).reshape(NI, rpi, NCH, 128).contiguous()
        bias_t = bs.reshape(NI, rpi, NCH * 128 * KP).contiguous()
    else:
        rows_t = rw.to(torch.int32).reshape(NI, rpi, NCH, 128).permute(0, 1, 3, 2).contiguous()
        bias_t = bs.reshape(NI, rpi, NCH, 128, KP).permute(0, 1, 2, 4, 3).reshape(NI, rpi, NCH * KP * 128).contiguous()
    eye = torch.eye(128, device=q_lat.device).to(torch.bfloat16)
    # ceil(n / rpi) in exact fp32 arithmetic (n <= 2^24)
    it = torch.floor((n.reshape(1, 1).to(torch.float32) + (rpi - 1)) * (1.0 / rpi)).to(torch.int32)
    out = wrap_nki(_kernel(fallback))[platform.nki_grid()](
        q_lat=q.reshape(NI, rpi, H, R).contiguous(), kc=kcp, rows_t=rows_t, bias=bias_t, identb=eye, n_iter=it,
        scale=float(scale), fp8=int(fp8), want_lse=int(lse), rev=_rev(fallback, bb), qkw=dsa_slots.QKW)
    if lse:
        o, ls = out
        return o.reshape(Np, H, R)[:N], ls.reshape(Np, H)[:N]
    return out.reshape(Np, H, R)[:N]


def _split() -> bool:
    """platform.lnc_split names dsa_slots_n at grid 2 (trn2, LNC=2): kernels/dsa_slots_n_s.py, each iteration's rows
    split over the two physical cores. Read when attend is traced."""
    from .. import platform

    return platform.nki_grid() == 2 and platform.lnc_split("dsa_slots_n")


def _nat(fallback) -> bool:
    """A decode fallback's split kernel reads its rows and bias in their natural layout (kernels/dsa_slots_n_s2.py), so
    attend skips the two permutes; never the context-parallel prefill's call. fallback is the call's site: False (the
    prefill), "c" (dsa_slots_c's fallback), "shell" (the DSA shell's), or True (any site); dsa_slots_n_s2.SITES scopes
    the sites."""
    from . import dsa_slots_n_s2

    site = fallback is True or fallback in dsa_slots_n_s2.SITES
    return bool(fallback) and site and dsa_slots_n_s2.NAT and dsa_slots.QKW and _split()


def _kernel(fallback: bool = False):
    """attend's kernel: the split one under _split() (the natural-layout one under _nat), else this module's."""
    if _nat(fallback):
        from . import dsa_slots_n_s2

        return dsa_slots_n_s2.dsa_slots_n_s_kernel
    if _split():
        from . import dsa_slots_n_s

        return dsa_slots_n_s.dsa_slots_n_s_kernel
    return dsa_slots_n_kernel


def _rev(fallback: bool = False, bb: bool = False) -> int:
    """attend's kernel's static `rev`: the natural-layout kernel's under _nat, the split kernel's REV under _split()
    (REV_BB, its BIAS_BF16 lines included, for a bf16-bias call), else this module's REV."""
    if _nat(fallback):
        from . import dsa_slots_n_s2

        return dsa_slots_n_s2.REV
    if _split():
        from . import dsa_slots_n_s

        return dsa_slots_n_s.REV_BB if bb else dsa_slots_n_s.REV
    return REV
