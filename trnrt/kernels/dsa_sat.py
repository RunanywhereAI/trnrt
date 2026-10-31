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
