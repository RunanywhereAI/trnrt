"""KDA's gated RMSNorm (KimiLinearRMSNormGated) after the prefill delta rule as one NKI kernel at LNC = 2 (trn2), the
128-row tiles split between the two programs (models/linear_attn.py _mix_rows, the chunk and sequence forms).

What it computes is _mix_rows's tail before the output projection, per row t and v head h:
    y = w * (o * rsqrt(mean(o^2) + eps));  out[t, h] = bf16(y * sigmoid(z[t, h]))
with o the delta rule's fp32 [T, Hv, Dv] (kernels/delta_rule.py), z the output gate bf16 [T, Hv, Dv] and w the o_norm weight
fp32 [Dv]; out is bf16 [T, Hv * Dv], the output projection's input.

Why a kernel: neuronx-cc lays these few elementwise ops on 2 of 128 partitions (this rank's 2 v heads), with 2,048 fp32
transposes into that layout: ~1.07 ms of engine time a KDA layer in the 4,096-row prefill pieces, against 0.17 ms here. Rows
are tokens on the partitions and (head, d) on the free axis, the delta rule's own output layout, so every reduction runs
along the free axis and nothing is transposed. The work is row-local, so the programs exchange nothing and share no control
flow: program p takes tiles p, p + programs, ...

Arithmetic per 128-row tile, the served form (fix 4): per head the sum of squares by activation_reduce (square, add) on fp32
o, a = ss / Dv + eps on the vector engine, the scalar engine's rsqrt with one vector Newton step, (o r) w on the vector
engine, sigmoid(z) on the scalar engine and one multiply into the bf16 output (rounded once). fp32 o, with no bf16 copy, is
how the device's XLA norm reads it (neuronx-cc folds the source's o.to(bf16).float() away), and the Newton step brings the
scalar engine's rsqrt to the vector's precision. What differs from XLA is the fp32 summation order of the mean and the
scalar engine's sigmoid, so the result sits inside the floor, not bit-identical. The other forms (0-3: the first build,
the Newton step alone, sigmoid as 1 / (1 + exp(-z)), sigmoid on fp32 z) stay selectable as the static `fix` argument for
precision lanes.
"""

from __future__ import annotations

import torch

P = 128
FIX = 4  # the served form: fp32 o, the Newton-refined rsqrt


def reference(o: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """_mix_rows's KDA tail in torch (o fp32 [T, Hv, Dv], z bf16 [T, Hv, Dv], w fp32 [Dv]): bf16 [T, Hv * Dv]."""
    of = o.to(torch.bfloat16).float()
    of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    of = w * of
    out = (of * torch.sigmoid(z.float())).to(torch.bfloat16)
    return out.reshape(o.shape[0], -1)


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

# -- the kernel (its span is what REV hashes)
if nki is not None:
    @nki.jit
    def gated_norm2_kernel(o, z, w, eps: float, rev: int, fix: int):
        """The gated norm, tile for tile, with the 128-row tiles split between the LNC programs (program p takes tiles
        p, p + programs, ...). o fp32 [To >= T, Hv, Dv], z bf16 [T, Hv, Dv] with T a multiple of 128 x programs (takes2),
        w fp32 [Dv]; returns bf16 [T, Hv * Dv]."""
        f32, bf16 = nl.float32, nl.bfloat16
        T, Hv, Dv = z.shape
        F = Hv * Dv
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        out = nl.ndarray((T, F), dtype=bf16, buffer=nl.shared_hbm)
        W = nl.ndarray((P, Dv), dtype=f32, buffer=nl.sbuf)  # w on every partition (a stride-0 read)
        nisa.dma_copy(dst=W, src=w.ap(pattern=[[0, P], [1, Dv]], offset=0))
        E = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=E, value=eps)
        for j in range(T // P // npg):
            r0 = (j * npg + pid) * P  # this program's tile (whole tiles only: takes2)
            ot = nl.ndarray((P, F), dtype=f32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ot, src=o.ap(pattern=[[F, P], [1, F]], offset=r0 * F))
            zt = nl.ndarray((P, F), dtype=bf16, buffer=nl.sbuf)
            nisa.dma_copy(dst=zt, src=z.ap(pattern=[[F, P], [1, F]], offset=r0 * F))
            if fix == 4:  # fp32 o as the served device path reads it (no bf16 copy)
                ob = ot
            else:
                ob = nl.ndarray((P, F), dtype=bf16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ob, src=ot, engine=nisa.vector_engine)  # bf16(o), rounded once
            sg = nl.ndarray((P, F), dtype=f32, buffer=nl.sbuf)
            if fix == 2:  # 1 / (1 + exp(-z)): exp on the scalar engine, + 1 and the reciprocal on the vector engine
                ez = nl.ndarray((P, F), dtype=f32, buffer=nl.sbuf)
                nisa.activation(dst=ez, op=nl.exp, data=zt, scale=-1.0)
                nisa.tensor_scalar(dst=ez, data=ez, op0=nl.add, operand0=1.0, engine=nisa.vector_engine)
                nisa.reciprocal(dst=sg, data=ez)
            elif fix == 3:  # the scalar engine's sigmoid on fp32 z
                zf = nl.ndarray((P, F), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=zf, src=zt, engine=nisa.vector_engine)
                nisa.activation(dst=sg, op=nl.sigmoid, data=zf)
            else:
                nisa.activation(dst=sg, op=nl.sigmoid, data=zt)
            y = nl.ndarray((P, F), dtype=f32, buffer=nl.sbuf)
            for h in range(Hv):
                sq = nl.ndarray((P, Dv), dtype=f32, buffer=nl.sbuf)
                ss = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
                nisa.activation_reduce(dst=sq, op=nl.square, data=ob[:, h * Dv:(h + 1) * Dv], reduce_op=nl.add,
                                       reduce_res=ss)
                r = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
                if fix == 0:
                    nisa.activation(dst=r, op=nl.rsqrt, data=ss, scale=1.0 / Dv, bias=E)
                else:  # a = ss / Dv + eps on the vector engine, the scalar engine's rsqrt, one vector Newton step
                    a = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=a, data=ss, op0=nl.multiply, operand0=1.0 / Dv, op1=nl.add, operand1=eps,
                                       engine=nisa.vector_engine)
                    rq = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)  # (not r0: the tile's row offset)
                    nisa.activation(dst=rq, op=nl.rsqrt, data=a)
                    nt = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=nt, data1=rq, data2=rq, op=nl.multiply, engine=nisa.vector_engine)
                    nisa.tensor_tensor(dst=nt, data1=nt, data2=a, op=nl.multiply, engine=nisa.vector_engine)
                    nisa.tensor_scalar(dst=nt, data=nt, op0=nl.multiply, operand0=-0.5, op1=nl.add, operand1=1.5,
                                       engine=nisa.vector_engine)
                    nisa.tensor_tensor(dst=r, data1=rq, data2=nt, op=nl.multiply, engine=nisa.vector_engine)
                nisa.scalar_tensor_tensor(dst=y[:, h * Dv:(h + 1) * Dv], data=ob[:, h * Dv:(h + 1) * Dv],
                                          op0=nl.multiply, operand0=r, op1=nl.multiply, operand1=W)
            res = nl.ndarray((P, F), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=res, data1=y, data2=sg, op=nl.multiply, engine=nisa.vector_engine)
            nisa.dma_copy(dst=out.ap(pattern=[[F, P], [1, F]], offset=r0 * F), src=res)
        return out
else:
    gated_norm2_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of the kernel's source span and P, the constant it reads."""
    import zlib

    src = open(__file__).read()
    a = src.index("# -- the kernel (its span is what REV hashes)")
    b = src.index("    gated_norm2_kernel = None", a)
    return zlib.crc32(repr((P,)).encode(), zlib.crc32(src[a:b].encode()))


REV = _kernel_rev()


def takes2(dv: int, T: int) -> bool:
    """Whether the kernel runs this layer's prefill tail: head_v_dim <= 512, the runtime's grid above 1 and T whole tiles
    for every program. Ints only."""
    if dv > 512:
        return False
    from .. import platform

    g = platform.nki_grid()
    return g > 1 and T > 0 and T % (P * g) == 0


def apply2(o: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float, fix: int | None = None) -> torch.Tensor:
    """The norm at the runtime's grid (takes2): o fp32 [To >= T, Hv, Dv], z [T, Hv, Dv], w [Dv]; returns bf16
    [T, Hv * Dv]."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if gated_norm2_kernel is None:
        raise RuntimeError("the NKI gated-norm kernel needs the nki package (the Neuron venv)")
    return wrap_nki(gated_norm2_kernel)[platform.nki_grid()](
        o=o.float().contiguous(), z=z.to(torch.bfloat16).contiguous(), w=w.float().contiguous(), eps=float(eps), rev=REV,
        fix=FIX if fix is None else int(fix))
