"""The per-half decode shell frame: what every fused half kernel shares.

Host side (the model's wiring and the launch): the program split, the decode-only arranged weights, the read/write
slot rows, the per-program partials' sum, the ring size of a row pipeline, the kernel-source revision. In-kernel (NKI,
traced into the caller's kernel): static and slot DMAs with their descriptor modes, the one cross-program exchange,
rsqrt with a Newton step broadcast over the partitions, and sigmoid through the exp table. The NKI helpers are
kernels/kda_half.py's, device-proven on trn2.
"""

from __future__ import annotations

import os
import zlib

import torch

P = 128


# -- host side ---------------------------------------------------------------------------------------------------------

# HOST_SIM (tests only: a module attribute, never read from the environment): host tensors take the decode kernels'
# device path (hybrid's kda_shell and ffn_half guards, mhc_pre, mhc_mix), so a harness that routes every launch to
# nki.simulate runs the served composition with its kernels in it. Only that harness sets it; nothing on serving's path
# does. Each guard reads it inline, `t.device.type != "cpu" or SH.HOST_SIM`, never through a helper taking the tensor:
# dynamo realizes a helper call's tensor arguments in the call's order, which can reorder a graph's inputs (hybrid.py's
# note on sp / dec).
HOST_SIM = False


def programs() -> int:
    """The programs a shell's arranged weights are laid out for: the runtime's NKI grid, or, on a meta build before the
    runtime is configured (tools that count a model's bytes), the target family's LNC."""
    from .. import platform

    try:
        return platform.nki_grid()
    except RuntimeError:
        t = platform.target()
        return platform.lnc(platform.family_of(t)) if t is not None else 1


def program_units(n: int, npg: int, pid: int) -> tuple[int, int]:
    """(first, count) of program pid's share of n units (heads, experts, output column tiles): the same count each, so
    both programs trace the same shapes. A shell splits its weight bytes this way rather than duplicating them."""
    if npg < 1 or n % npg:
        raise ValueError(f"shell: {n} units do not split evenly over {npg} programs")
    k = n // npg
    return pid * k, k


def ring_rows(pd: int, skew: int, units_per_row: int) -> int:
    """Ring entries a row pipeline needs: the row loading PD ahead, the rows between, and the row whose stage B still
    reads its entry, SKEW units = ceil(SKEW / units per row) rows back. A fixed 4 is exact at 2 units per
    row; at 1 (a head per program) it must be 5 (kernels/kda_half.py ring_rows)."""
    return pd + (skew + units_per_row - 1) // units_per_row + 1


def register_decode_only(layer, prefix: str, shapes: dict) -> None:
    """Create a shell's arranged weights on `layer` as parameters named prefix + name, each (shape, dtype) from
    `shapes`, and list them in layer._decode_only: decoder._layer_split then keeps them out of every prefill and
    piecewise graph, so no other key moves; the whole-step decode (the traced module) reads them. Created on the build
    device (meta in the tools, which then count them); the loader moves parameters to the device. Zeroed, not empty:
    a packed blob's fill writes only its views, so its alignment padding would otherwise hold whatever the allocator
    gave, and the inputs' bytes would vary run to run (an input digest did)."""
    from torch import nn

    for n, (shape, dt) in shapes.items():
        setattr(layer, prefix + n, nn.Parameter(torch.zeros(*shape, dtype=dt), requires_grad=False))
    layer._decode_only = tuple(getattr(layer, "_decode_only", ())) + tuple(prefix + n for n in shapes)


ALIGN = 256  # elements: every packed tensor starts at a multiple (512 B of bf16, 1 KiB of fp32)
DT_NAMES = {torch.bfloat16: "bf16", torch.float32: "f32", torch.float8_e4m3fn: "f8", torch.int32: "i32"}


def packed_layout(shapes: dict) -> dict:
    """{dtype: (numel, {name: (offset, shape)})}: a half's decode-only tensors ({name: (shape, dtype)}) packed one
    after another in insertion order into one flat blob per dtype, each from a multiple of ALIGN elements. A pure
    function of the shapes, so the loader (register_packed, packed_views) and the kernel's trace (its static .ap
    offsets) compute the same layout from the same dims and nothing has to pass it.

    Why: every graph input costs the launch ~1-2 us on the host, and an unpacked decode step had 1,314 of them. A
    half's arranged weights as one tensor per dtype instead of one per layout take a KDA half from ~10 weight inputs
    to 2."""
    lay: dict = {}
    for name, (shape, dt) in shapes.items():
        n = 1
        for d in shape:
            n *= int(d)
        tot, idx = lay.get(dt, (0, {}))
        idx[name] = (tot, tuple(int(d) for d in shape))
        lay[dt] = (tot + -(-n // ALIGN) * ALIGN, idx)
    return lay


def register_packed(layer, prefix: str, shapes: dict) -> dict:
    """register_decode_only for a packed half: one decode-only parameter per dtype, prefix + "blob_" + bf16 / f32 /
    ..., flat, holding every tensor of `shapes` at packed_layout's offsets. Returns the layout."""
    lay = packed_layout(shapes)
    register_decode_only(layer, prefix, {"blob_" + DT_NAMES[dt]: ((n,), dt) for dt, (n, _) in lay.items()})
    return lay


def packed_views(layer, prefix: str, shapes: dict) -> dict:
    """{name: view} of a packed half's blobs, for fill() to write into and for tests to read."""
    out = {}
    for dt, (_, idx) in packed_layout(shapes).items():
        blob = getattr(layer, prefix + "blob_" + DT_NAMES[dt]).data
        for name, (off, shape) in idx.items():
            n = 1
            for d in shape:
                n *= d
            out[name] = blob[off:off + n].view(shape)
    return out


def arrange_fn(fn: torch.Tensor, hc: int) -> torch.Tensor:
    """A block's hyper-connection projection fn [O, hc H] for mhc_pre_h: fnh [128, hc, W, O] with
    fnh[p, n, j, o] = fn[o, n H + p W + j] (h-layout per stream, W = H / 128)."""
    O, F = fn.shape
    H = F // hc
    return fn.reshape(O, hc, P, H // P).permute(2, 1, 3, 0).contiguous()


def slots(state_slot: torch.Tensor, positions: torch.Tensor, rows: int) -> torch.Tensor:
    """int32 [2, T]: each row's read row (rows, i.e. out of range, when the row starts its sequence: a zero state, as
    torch.where(keep, S, 0) does) and its write row. Callers pass start_row(pool rows)."""
    rd = torch.where(positions > 0, state_slot, torch.full_like(state_slot, rows))
    return torch.stack([rd, state_slot]).to(torch.int32)


# The zero row: the state pools have one more row, the last, zeros at allocation and in no free list
# (engine/state_pool.py), and a row starting its sequence (warmup's rows, padded rows) reads it, in range. A read of
# row R past the pool, which the KDA kernels' hardware-DGE reads skip (oob_mode skip), hung the serve warmup at four
# decode rows after a 4,096-row prefill.


def start_row(rows: int) -> int:
    """The row a row starting its sequence reads, of pools with `rows` rows: the zero row, the last."""
    return rows - 1


def add_partials(yp: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[npg, T, H] fp32 per-program partials of an output projection -> [T, H] in dtype: their fp32 sum rounded once,
    as one matmul accumulating both programs' rows in PSUM would round it."""
    return yp.sum(0).to(dtype)


def kernel_rev(path: str, begin: str, end: str) -> int:
    """CRC-32 of a module's kernel source between two markers, passed as the kernel's static `rev`: the compile-cache
    key ignores NKI kernel source, so an edited kernel would reuse a stale NEFF without it."""
    src = open(path).read()
    a = src.index(begin)
    return zlib.crc32(src[a:src.index(end, a)].encode())


# -- in-kernel (NKI) ---------------------------------------------------------------------------------------------------

try:  # the Neuron venv
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nisa = None

if nisa is not None:
    F32 = nl.float32

    def dma_static(dst, src):
        """A DMA with every address static: descriptors made before the kernel runs (dge_mode none). Left to the
        compiler, a kernel's bytes went through the GpSimd engine's software DGE (kda_half: 98%, 64 -> 54.8 us)."""
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.none)

    def dma_slot(dst, src):
        """A DMA at a run-time row (an .ap with scalar_offset from a slots tile): hardware DGE, skipped out of range."""
        nisa.dma_copy(dst=dst, src=src, oob_mode=nisa.oob_mode.skip, dge_mode=nisa.dge_mode.hwdge)

    def rsqrt_bcast(ones_c, ones_r, sq, n: int, scale: float, add: float):
        """rsqrt(scale * (sum over the partitions of sq [128, n] fp32) + add) on every partition, [128, n] fp32.
        ones_c [128, 1] and ones_r [1, 128] fp32 tiles of ones. The scalar engine's rsqrt, then one Newton step on the
        vector engine (the rsqrt alone is ~1e-5 off: kernels/mhc_pre.py)."""
        ps = nl.ndarray((1, n), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps, stationary=ones_c, moving=sq, accumulate=False)
        a = nl.ndarray((1, n), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=a, data=ps, op0=nl.multiply, operand0=scale, op1=nl.add, operand1=add,
                           engine=nisa.vector_engine)
        r0 = nl.ndarray((1, n), dtype=F32, buffer=nl.sbuf)
        nisa.activation(dst=r0, op=nl.rsqrt, data=a)
        nt = nl.ndarray((1, n), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=nt, data1=r0, data2=r0, op=nl.multiply, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=nt, data1=nt, data2=a, op=nl.multiply, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=nt, data=nt, op0=nl.multiply, operand0=-0.5, op1=nl.add, operand1=1.5,
                           engine=nisa.vector_engine)
        r = nl.ndarray((1, n), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r, data1=r0, data2=nt, op=nl.multiply, engine=nisa.vector_engine)
        pb = nl.ndarray((128, n), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=pb, stationary=ones_r, moving=r, accumulate=False)
        rb = nl.ndarray((128, n), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rb, src=pb, engine=nisa.vector_engine)
        return rb

    def exchange_buffer(shape, dtype):
        """The cross-program exchange buffer: shared HBM, allocated with the kernel's outputs before any branch on the
        program id, and RETURNED by the kernel as an extra output (the caller drops it). Proven on device (trn2, grid
        2): a probe on one core was exact in 20 of 20 calls at 8 and 512 columns with new values each call, while its
        control (every program putting at block 0) failed every call; and kda_half's gated-norm exchange gave y
        bit-equal to its emulation at B = 1. Not returned, or allocated inside the program-count branch, it was exact on
        device too: the rule fixes one proven form, not the only working one."""
        return nl.ndarray(shape, dtype=dtype, buffer=nl.shared_hbm)

    def exchange(buf, put, src, get, dst, npg: int):
        """The one cross-program exchange a shell may use. This program's part goes in (put, a view or .ap of buf: its
        own region <- src), then the core barrier on buf (grid 2), then dst <- get (a view of buf: what this program
        needs of the whole). Static DMAs both ways. At grid 1 the read after the write on one HBM tensor is ordered by
        the compiler's dependency tracking alone [device unproven: shells run at grid 2 on trn2]."""
        dma_static(dst=put, src=src)
        if npg > 1:
            nisa.core_barrier(data=buf, cores=(0, 1))
        dma_static(dst=dst, src=get)

    def sigmoid_exp(dst, src, e):
        """dst = sigmoid(src) = 1 / (1 + exp(-src)): the scalar engine's exp, + 1 and the reciprocal on the vector
        engine (e: an fp32 scratch of src's shape). With it a shell's activations need only the exp and rsqrt tables:
        each table switch costs ~1.3 us on the scalar engine. silu(x) = x * sigmoid_exp(x)."""
        nisa.activation(dst=e, op=nl.exp, data=src, scale=-1.0)
        nisa.tensor_scalar(dst=e, data=e, op0=nl.add, operand0=1.0, engine=nisa.vector_engine)
        nisa.reciprocal(dst=dst, data=e)

    def finite_or_zero(src, dst, f):
        """dst (a native uint16 tile [p, n]) = the bf16 bits of src (a uint16 view of a bf16 tile [p, n]) with every NaN
        and +-inf set to +0 and every finite one's bits kept (-0 included); read it back as dst.view(bfloat16). A
        bitcast view works as an operand (the PE takes fp8 views of uint8 tiles), but a write through one converts by
        value (nki.simulate: bits 1 came out as bf16 1.0), so the result goes to a native tile. The bits are read as
        integers 0 .. 65535, so every step is fp32 arithmetic on values below 2^24 (exact) and no NaN is ever compared
        (what the vector engine does with a NaN operand is unmeasured): the exponent is all ones iff (b mod 2^15) >=
        0x7F80. Why: a PE transpose that contracts over the rows (an identity matmul) carries one row's NaN into every
        row, as NaN * 0 = NaN; zeroed first, a non-finite row stays in its row. f: an fp32 scratch of src's shape."""
        VE = nisa.vector_engine
        nisa.tensor_scalar(dst=f, data=src, op0=nl.greater, operand0=32767.5, engine=VE)             # the sign bit
        nisa.scalar_tensor_tensor(dst=f, data=f, op0=nl.multiply, operand0=-32768.0, op1=nl.add, operand1=src)  # mod 2^15
        nisa.tensor_scalar(dst=f, data=f, op0=nl.greater, operand0=32639.5, engine=VE)               # exponent all ones
        nisa.tensor_scalar(dst=f, data=f, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0, engine=VE)  # keep
        nisa.tensor_tensor(dst=dst, data1=src, data2=f, op=nl.multiply, engine=VE)                   # b or 0, exact

    # -- the hyper-connection around a half, in h-layout (h = p W + j, W = H / 128): the shell's prologue -------------

    def streams_h(S, sh, hc: int, T: int, H: int):
        """The streams S (HBM [T, hc H] bf16) into sh [128, hc, T, W] bf16: 64-byte runs per partition."""
        W, F = H // P, hc * H
        for n in range(hc):
            dma_static(dst=sh[:, n, :, :], src=S.ap(pattern=[[W, P], [F, T], [1, W]], offset=n * H))

    def mix_in_h(sh, S, post_p, comb_p, Yp, S_out, hc: int, T: int, H: int, npg: int):
        """The previous block's output mix (kernels/mhc_mix.py's arithmetic) into sh, and S_out (HBM [T, hc H]):
        y = round_bf16(sum of the npg all-reduced partials Yp [npg, T, H]); stream n = post_n y + sum_m comb[m, n]
        S_m in fp32, m in order, through _round's barrier to bf16. S: the streams before that block (HBM [T, hc H])."""
        W, F = H // P, hc * H
        pb0 = nl.ndarray((P, T, hc), dtype=nl.float32, buffer=nl.sbuf)
        dma_static(dst=pb0, src=post_p.ap(pattern=[[0, P], [hc, T], [1, hc]], offset=0))
        cb0 = nl.ndarray((P, T, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
        dma_static(dst=cb0, src=comb_p.ap(pattern=[[0, P], [hc * hc, T], [1, hc * hc]], offset=0))
        st0 = nl.ndarray((P, hc, T, W), dtype=nl.bfloat16, buffer=nl.sbuf)
        streams_h(S, st0, hc, T, H)
        ya = nl.ndarray((P, T, W), dtype=nl.float32, buffer=nl.sbuf)
        for g in range(npg):
            yg = nl.ndarray((P, T, W), dtype=nl.bfloat16, buffer=nl.sbuf)
            dma_static(dst=yg, src=Yp.ap(pattern=[[W, P], [H, T], [1, W]], offset=g * T * H))
            if g == 0:
                nisa.tensor_copy(dst=ya, src=yg, engine=nisa.vector_engine)
            else:
                nisa.tensor_tensor(dst=ya, data1=ya, data2=yg, op=nl.add, engine=nisa.vector_engine)
        ybf = nl.ndarray((P, T, W), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ybf, src=ya, engine=nisa.vector_engine)
        for n in range(hc):
            o = nl.ndarray((P, T, W), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=o, data1=ybf, data2=pb0.ap(pattern=[[T * hc, P], [hc, T], [0, W]], offset=n),
                               op=nl.multiply, engine=nisa.vector_engine)
            for m in range(hc):
                pr = nl.ndarray((P, T, W), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=pr, data1=st0[:, m, :, :],
                                   data2=cb0.ap(pattern=[[T * hc * hc, P], [hc * hc, T], [0, W]], offset=m * hc + n),
                                   op=nl.multiply, engine=nisa.vector_engine)
                nisa.tensor_tensor(dst=o, data1=o, data2=pr, op=nl.add, engine=nisa.vector_engine)
            c = nl.ndarray((P, T, W), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=c, data=o, op0=nl.multiply, operand0=65537.0, engine=nisa.vector_engine)
            d = nl.ndarray((P, T, W), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=d, data1=c, data2=o, op=nl.subtract, engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=c, data1=c, data2=d, op=nl.subtract, engine=nisa.vector_engine)
            nisa.tensor_copy(dst=sh[:, n, :, :], src=c, engine=nisa.vector_engine)
            dma_static(dst=S_out.ap(pattern=[[W, P], [F, T], [1, W]], offset=n * H), src=sh[:, n, :, :])

    def mhc_pre_h(sh, fw, bf, o_base: int, o_scale: int, post_out, comb_out, hc: int, T: int, H: int,
                  iters: int, hc_eps: float, rms_eps: float):
        """kernels/mhc_pre.py on the streams in h-layout: returns x [128, T, W] bf16 (h-layout, the half's input) and
        writes post_out [T, hc] and comb_out [T, hc hc] (HBM fp32). fw: fn as arrange_fn lays it out, [128, hc, W, O]
        bf16 in SBUF; bf: an fp32 blob holding base [O] at o_base and scale [3] at o_scale. The projection is hc W
        matmuls of [128, T] by [128, O] (mhc_pre's NJ k-tiles in another order); sigmoids are sigmoid_exp's; the
        Sinkhorn is mhc_pre's eps form. pre reaches every partition on chip: D [T, T hc] holds pre[k, c] at column
        (k, c) of row k and an exact 0 elsewhere (a select, not a product, so a NaN or inf row reaches no other row's
        column), then ones [T, 128]^T D on the PE, exact in fp32 (one nonzero term per sum). It was an HBM round trip
        through a private scratch: ~13.5 us of kda_shell's critical path."""
        f32, bf16 = nl.float32, nl.bfloat16
        W, F = H // P, hc * H
        O = 2 * hc + hc * hc
        shj = nl.ndarray((P, hc, W, T), dtype=bf16, buffer=nl.sbuf)
        for n in range(hc):
            nisa.tensor_copy(dst=shj[:, n, :, :],
                             src=sh.ap(pattern=[[hc * T * W, P], [1, W], [W, T]], offset=n * T * W),
                             engine=nisa.vector_engine)
        pm = nl.ndarray((T, O), dtype=f32, buffer=nl.psum)
        for n in range(hc):
            for j in range(W):
                nisa.nc_matmul(dst=pm, stationary=shj[:, n, j, :], moving=fw[:, n, j, :], accumulate=(n + j > 0))
        ones = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=1.0)
        ones_t = nl.ndarray((T, P), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ones_t, value=1.0)
        part = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
        for n in range(hc):
            sqn = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sqn, data1=sh[:, n, :, :], data2=sh[:, n, :, :], op=nl.multiply,
                               engine=nisa.vector_engine)
            if n == 0:
                nisa.tensor_reduce(dst=part, op=nl.add, data=sqn, axis=2)
            else:
                pn = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=pn, op=nl.add, data=sqn, axis=2)
                nisa.tensor_tensor(dst=part, data1=part, data2=pn, op=nl.add, engine=nisa.vector_engine)
        pss = nl.ndarray((T, 1), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pss, stationary=part, moving=ones, accumulate=False)
        raw = nl.ndarray((T, O), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=raw, src=pm, engine=nisa.vector_engine)
        ss = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ss, src=pss, engine=nisa.vector_engine)
        a = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=a, data=ss, op0=nl.multiply, operand0=1.0 / F, op1=nl.add, operand1=rms_eps,
                           engine=nisa.vector_engine)
        r0 = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=r0, op=nl.rsqrt, data=a)
        nt = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=nt, data1=r0, data2=r0, op=nl.multiply, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=nt, data1=nt, data2=a, op=nl.multiply, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=nt, data=nt, op0=nl.multiply, operand0=-0.5, op1=nl.add, operand1=1.5,
                           engine=nisa.vector_engine)
        r = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r, data1=r0, data2=nt, op=nl.multiply, engine=nisa.vector_engine)
        mix = nl.ndarray((T, O), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mix, data=raw, op0=nl.multiply, operand0=r, engine=nisa.vector_engine)
        bt = nl.ndarray((T, O), dtype=f32, buffer=nl.sbuf)
        dma_static(dst=bt, src=bf.ap(pattern=[[0, T], [1, O]], offset=o_base))
        stt = nl.ndarray((T, 3), dtype=f32, buffer=nl.sbuf)
        dma_static(dst=stt, src=bf.ap(pattern=[[0, T], [1, 3]], offset=o_scale))
        z = nl.ndarray((T, 2 * hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=z[:, 0:hc], data=mix[:, 0:hc], op0=nl.multiply, operand0=stt[:, 0:1],
                           engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=z[:, hc:2 * hc], data=mix[:, hc:2 * hc], op0=nl.multiply, operand0=stt[:, 1:2],
                           engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=z, data1=z, data2=bt[:, 0:2 * hc], op=nl.add, engine=nisa.vector_engine)
        sgz = nl.ndarray((T, 2 * hc), dtype=f32, buffer=nl.sbuf)
        ez = nl.ndarray((T, 2 * hc), dtype=f32, buffer=nl.sbuf)
        sigmoid_exp(sgz, z, ez)
        pre = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pre, data=sgz[:, 0:hc], op0=nl.add, operand0=hc_eps, engine=nisa.vector_engine)
        D = nl.ndarray((T, T * hc), dtype=f32, buffer=nl.sbuf)
        nisa.affine_select(dst=D, pattern=[[1, T], [0, hc]], channel_multiplier=-1, offset=0,
                           on_true_tile=pre.ap(pattern=[[hc, T], [0, T], [1, hc]], offset=0), on_false_value=0.0,
                           cmp_op=nl.equal)
        pbp = nl.ndarray((P, T * hc), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pbp, stationary=ones_t, moving=D, accumulate=False)
        pb = nl.ndarray((P, T * hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pb, src=pbp, engine=nisa.vector_engine)
        post = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=post, data=sgz[:, hc:2 * hc], op0=nl.multiply, operand0=2.0, engine=nisa.vector_engine)
        dma_static(dst=post_out, src=post)
        lg = nl.ndarray((T, hc, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lg, data=mix.ap(pattern=[[O, T], [hc, hc], [1, hc]], offset=2 * hc), op0=nl.multiply,
                           operand0=stt[:, 2:3], engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=lg, data1=lg, data2=bt.ap(pattern=[[O, T], [hc, hc], [1, hc]], offset=2 * hc),
                           op=nl.add, engine=nisa.vector_engine)
        mx = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=mx, op=nl.maximum, data=lg, axis=2)
        cm = nl.ndarray((T, hc, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=cm, data1=lg, data2=mx.ap(pattern=[[hc, T], [1, hc], [0, hc]], offset=0),
                           op=nl.subtract, engine=nisa.vector_engine)
        nisa.activation(dst=cm, op=nl.exp, data=cm)
        sm = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sm, op=nl.add, data=cm, axis=2)
        nisa.reciprocal(dst=sm, data=sm)
        nisa.tensor_tensor(dst=cm, data1=cm, data2=sm.ap(pattern=[[hc, T], [1, hc], [0, hc]], offset=0),
                           op=nl.multiply, engine=nisa.vector_engine)
        E1 = hc + 1
        cp = nl.ndarray((T, E1, E1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=cp, value=hc_eps)
        nisa.tensor_scalar(dst=cp[:, 0:hc, 0:hc], data=cm, op0=nl.add, operand0=hc_eps, engine=nisa.vector_engine)
        for k in range(2 * iters - 1):
            dk = nl.ndarray((T, hc), dtype=f32, buffer=nl.sbuf)
            if k % 2 == 0:  # columns: sum over i = 0..hc (the eps row last), per j
                nisa.tensor_reduce(dst=dk, op=nl.add, data=cp.ap(pattern=[[E1 * E1, T], [1, hc], [E1, E1]], offset=0),
                                   axis=2)
                db = dk.ap(pattern=[[hc, T], [0, hc], [1, hc]], offset=0)
            else:  # rows: sum over j = 0..hc (the eps column last), per i
                nisa.tensor_reduce(dst=dk, op=nl.add, data=cp.ap(pattern=[[E1 * E1, T], [E1, hc], [1, E1]], offset=0),
                                   axis=2)
                db = dk.ap(pattern=[[hc, T], [1, hc], [0, hc]], offset=0)
            nisa.reciprocal(dst=dk, data=dk)
            nisa.tensor_tensor(dst=cp[:, 0:hc, 0:hc], data1=cp[:, 0:hc, 0:hc], data2=db, op=nl.multiply,
                               engine=nisa.vector_engine)
        dma_static(dst=comb_out.ap(pattern=[[hc * hc, T], [hc, hc], [1, hc]], offset=0), src=cp[:, 0:hc, 0:hc])
        xa = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=xa, data1=pb.ap(pattern=[[T * hc, P], [hc, T], [0, W]], offset=0), data2=sh[:, 0, :, :],
                           op=nl.multiply, engine=nisa.vector_engine)
        for n in range(1, hc):
            prn = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prn, data1=pb.ap(pattern=[[T * hc, P], [hc, T], [0, W]], offset=n),
                               data2=sh[:, n, :, :], op=nl.multiply, engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=xa, data1=xa, data2=prn, op=nl.add, engine=nisa.vector_engine)
        cx = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=cx, data=xa, op0=nl.multiply, operand0=65537.0, engine=nisa.vector_engine)
        dx = nl.ndarray((P, T, W), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=dx, data1=cx, data2=xa, op=nl.subtract, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=cx, data1=cx, data2=dx, op=nl.subtract, engine=nisa.vector_engine)
        xh = nl.ndarray((P, T, W), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xh, src=cx, engine=nisa.vector_engine)
        return xh

# -- end of the in-kernel helpers
REV = (kernel_rev(__file__, "# -- in-kernel (NKI)", "# -- end of the in-kernel helpers")
       ^ zlib.crc32(repr((P,)).encode()))  # and P, which its helpers read (kda_shell, ffn_half, ...)

# -- the hardware-DGE span (TRNRT_DMA_HW): with TRNRT_DMA_HW=1 this module's static DMA helper (dma_static) issues its DMAs in
# hardware DGE (dge_mode hwdge) instead of static descriptors, which cost a descriptor-ring entry per contiguous run in
# every loaded NEFF (the decode spill rings). Every MTP preset sets it (it is in every build behind the published MTP
# numbers); the default presets do not. Read at import, before any kernel module binds dma_static by name or folds REV, so the
# process environment has to carry it (trnrt/__main__.py sets it for a preset that needs it). Below the hashed span;
# REV folds this span's source when it is set (_dma_hw_rev), so every kernel folding REV re-keys, and unset no source,
# REV or key moves.
if os.environ.get("TRNRT_DMA_HW", "0") == "1" and nisa is not None:
    def dma_static(dst, src):  # noqa: F811
        """dma_static under TRNRT_DMA_HW=1: hardware DGE."""
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge)
# -- the hardware-DGE span end


def _dma_hw_rev() -> int:
    """CRC-32 of the hardware-DGE span above (the TRNRT_DMA_HW variant's source): REV's term when it is set."""
    src = open(os.path.abspath(__file__)).read()
    a = src.index("# -- the hardware-DGE span (TRNRT_DMA_HW)")
    return zlib.crc32(src[a:src.index("# -- the hardware-DGE span end", a)].encode())


if os.environ.get("TRNRT_DMA_HW", "0") == "1":
    REV ^= _dma_hw_rev()

# mix_in_h's served form is kernels/mix_wide.py's wide-cast, rebound here, outside the hashed span. The helper above
# (fin) stays as its reference: wide-cast equals its bits on every normal value. MIX_REV is what each caller of
# mix_in_h (kda_shell, ffn_half, ffn_half_pb) folds into its REV. Kernels that fold REV but never call mix_in_h
# (moe_half, dsa_shell) do not take it.
from . import mix_wide as _MW  # noqa: E402

MIX_REV = _MW.MIX_REV
if nisa is not None:
    mix_in_h_fin = mix_in_h  # fin's form (tests compare the served form against it)
    mix_in_h = _MW.mix_in_h_wide_cast
