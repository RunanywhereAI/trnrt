"""kernels/dsa_topk.py's selection under a run-time trip count of 0 or 1 (models/mla.py attention_cp_rows' failed-row
merge, EngineConfig.dsa_cp_fail_trip): the same instructions on the same data when trip is 1, so the same selection bit
for bit; nothing computed and nothing written when it is 0 (the output then holds whatever its HBM held; the caller
masks it).

Why: under the context-parallel prefill's local top-K (models/mla.py CP_LOCAL_K) a DSA layer re-selects the rows whose
certificate fails from every rank's whole keep list, CP_LOCAL_F (128) rows at a time, and merges them with a selection
over [128, A keep]. On real text no row fails, yet that merge ran at full size in every DSA layer of every prefill
chunk; under the trip it costs the loop's entry.

The form is kernels/dsa_long_select_t.py's: the body is a device loop of `trip` iterations (register_load +
nl.fori_loop), every PSUM tile allocated inside it (dsa_topk._select's own allocations), and both LNC programs run the
same loop count, each on its own row tiles as in dsa_topk (a count the programs disagree on hangs the call at LNC=2).
A separate module, so kernels/dsa_topk.py's source, its REV and the keys of every graph that calls it are unchanged.
"""

from __future__ import annotations

import zlib

import torch

from . import dsa_topk as T

try:  # the Neuron venv
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None and T.nki is not None:
    from .dsa_topk import _consts, _programs, _sb, _select  # noqa: F401  (the tracer resolves names in this module)

    I32 = nl.int32
    F32 = nl.float32

    @nki.jit
    def dsa_topk_t_kernel(sc, trip, keep: int, g: int, lg: int, nbits: int, vis_only: int, kp: int, tail: int,
                               tiles: int, rev: int, rbits: int, spl: int = 1):
        """dsa_topk_kernel's arguments plus trip int32 [1, 1] (0 or 1): its row-tile loop runs trip times."""
        N, W = sc.shape
        out = nl.ndarray((N, W * kp), dtype=F32, buffer=nl.shared_hbm)
        wio, po, G, bits = _consts(W, g, lg)
        NT = (N + 127) // 128
        npg, pid = _programs(spl)
        cnt = _sb((1, 1), I32)
        nisa.dma_copy(dst=cnt, src=trip)
        rt = nisa.register_alloc()
        nisa.register_load(rt, cnt)

        def body(it):
            for t0 in range(0, NT, tiles):
                if (t0 // tiles) % npg != pid:  # LNC: groups of row tiles alternate between the programs (dsa_topk's)
                    continue
                R0 = []
                NS = []
                for t in range(t0, min(t0 + tiles, NT)):
                    R0.append(t * 128)
                    NS.append(min(128, N - t * 128))
                S = []
                for j in range(len(NS)):
                    s = _sb((NS[j], W))
                    nisa.dma_copy(dst=s, src=sc.ap(pattern=[[W, NS[j]], [1, W]], offset=R0[j] * W))
                    S.append(s)
                _select(S, R0, NS, W, g, G, po, wio, bits, keep, nbits, vis_only, kp, tail, out, rbits)

        nl.fori_loop(0, rt, body)
        if npg > 1:  # LNC: every program's rows written (or none) before either ends
            nisa.core_barrier(data=out, cores=(0, 1))
        return out

else:
    dsa_topk_t_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source mixed with dsa_topk's REV (the _select code it runs), the static `rev`:
    the compile-cache key does not hash NKI kernel source."""
    import hashlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    dsa_topk_t_kernel = None", a)
    return zlib.crc32(hashlib.sha256(repr((src[a:b], T.REV)).encode()).digest())


REV = _kernel_rev()


def select(scores: torch.Tensor, keep: int, trip: torch.Tensor, vis_only: bool = True, group: bool = True, kp: int = 1,
           tail: bool = False) -> torch.Tensor:
    """dsa_topk.select when trip (an int tensor of one element, 0 or 1) is 1; when it is 0 the output is unwritten device
    memory, which the caller masks. On the host (no simulator) and for keep >= P, dsa_topk.select itself (its emulation:
    the caller's mask makes the trip-0 output irrelevant)."""
    *lead, P = scores.shape
    R = 1
    for d in lead:
        R *= d
    host = scores.device.type == "cpu" and not T.SH.HOST_SIM
    if keep >= P or host:
        return T.select(scores, keep, vis_only, group, kp, tail)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if dsa_topk_t_kernel is None:
        raise RuntimeError("the NKI DSA top-k kernel needs the nki package (the Neuron venv)")
    kin = T.kernel_inputs(scores.float().reshape(R, P), keep, vis_only, group, kp, tail)
    kin["rev"] = REV
    out = wrap_nki(dsa_topk_t_kernel)[platform.nki_grid()](trip=trip.reshape(1, 1).to(torch.int32), **kin)
    return out.reshape(*lead, P * kp)
