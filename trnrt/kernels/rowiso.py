"""Row isolation in fewer vector instructions (fin=2, the served form).

The contract, for every kernel that takes fin=2: no NaN / +-inf value crosses from one row (token) to another, a row
holding one is zeros (+0 bits) where the kernel's isolation put them, and every all-finite row is bit for bit fin 0's.
Its form here is elementwise (finite_or_zero3: each non-finite value +0, the row's finite values kept), where the data
sits on 128 lanes (kda_shell's xn and conv history; ffn_half_pb's xn). Each kernel's own tests pin its exact form.

A kernel that puts a batch's rows on the contraction of a PE transpose (an identity matmul) carries one row's NaN into
every row, as NaN * 0 = NaN. fin=1 (shell.finite_or_zero) zeroes every NaN / inf element of the tile the transposes
read, in five vector instructions. Its tiles have the rows on the partitions (4 or 12 of 128 lanes, 641 to 4,096
elements each), and a four-row decode step paid ~0.88 ms for it (shell.py's five lines, 579 instructions over 34
kda_shell calls x 2 sites and 42 moe_half calls). fin=2 keeps the guarantee, no row's NaN / inf
reaches another row, for less:
- finite_or_zero3: fin=1's result in three instructions (the keep and the multiply fused);
- kda_shell: xn (the normed input, all 128 lanes, W B = 128 elements each) instead of the projection's output pr
  (4 lanes, 641), and of the conv history only the columns this program's transposes read (3 hp 128 of 3 nv 128).
  A finite xn row whose projection overflows to inf is not zeroed (pinned by a test);
- ffn_half_pb: xn in mHC's h-layout, all 128 lanes, before the MoE body's transposes.
Every step reads the bf16 bits as the integers 0 .. 65535 (a uint16 view), so it is fp32 arithmetic on integers below
2^24 (exact) and no NaN or inf is ever an operand (what the vector engine does with one is unmeasured).
"""

from __future__ import annotations

import zlib

P = 128

# The row isolation the decode kernels take (kda_shell, moe_half; a kernel parameter): 0 none, 1 shell.finite_or_zero
# at fin=1's sites, 2 this module's (served).
FIN = 2


def kernel_rev(path: str, begin: str, end: str) -> int:
    """shell.kernel_rev's CRC (not imported: this module stays importable without torch)."""
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
    def finite_or_zero3(src, dst, f):
        """dst (a native uint16 tile of src's shape) = the bf16 bits of src (a uint16 view of a bf16 tile) with every
        NaN and +-inf set to +0 and every finite pattern kept, -0 and the subnormals included; read it back as
        dst.view(bfloat16). shell.finite_or_zero's result in three instructions, not five: the exponent is all ones
        iff b mod 2^15 >= 0x7F80 = 32640, so b is kept iff (b mod 2^15) < 32639.5. f: an fp32 scratch of src's shape."""
        VE = nisa.vector_engine
        nisa.tensor_scalar(dst=f, data=src, op0=nl.greater, operand0=32767.5, engine=VE)  # the sign bit
        nisa.scalar_tensor_tensor(dst=f, data=f, op0=nl.multiply, operand0=-32768.0, op1=nl.add, operand1=src)  # mod 2^15
        nisa.scalar_tensor_tensor(dst=dst, data=f, op0=nl.less, operand0=32639.5, op1=nl.multiply, operand1=src)  # b or 0

# -- end of the in-kernel helpers

REV = zlib.crc32(repr((kernel_rev(__file__, "# -- in-kernel (NKI)", "# -- end of the in-kernel helpers"), P)).encode())

# -- in-kernel (NKI): whole-row isolation of a kernel's input (moe_half's fin 2). Below REV's section, so the kernels that
# fold REV keep their compile-cache keys; REV_ROWS is this section's.
if nisa is not None:

    def row_keep(x, T: int, H: int):
        """keep fp32 [T, 1] on T partitions: 1.0 for each row of x (bf16 HBM [T, H]) whose values are all finite, 0.0
        for a row holding any NaN / inf. Read on all 128 lanes: x again as [P, T, H / P] (h = p H / P + c, pieces of
        H / P contiguous values: 64 bytes at H 4096), b mod 2^15 as in finite_or_zero3, its max over c, then each row's
        count of values >= 32639.5 summed over the partitions by a matmul with a ones column (0 / 1 operands: exact).
        Both programs of a grid call it alike (the same structure in each)."""
        C = H // P
        VE = nisa.vector_engine
        xp = nl.ndarray((P, T, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xp, src=x.ap(pattern=[[C, P], [H, T], [1, C]], offset=0), dge_mode=nisa.dge_mode.none)
        b = xp.view(nl.uint16)
        f = nl.ndarray((P, T, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=f, data=b, op0=nl.greater, operand0=32767.5, engine=VE)  # the sign bit
        nisa.scalar_tensor_tensor(dst=f, data=f, op0=nl.multiply, operand0=-32768.0, op1=nl.add, operand1=b)  # mod 2^15
        mx = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=mx, op=nl.maximum, data=f, axis=2)
        bad = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.sbuf)  # 1 where the partition's piece of row t is not
        nisa.tensor_scalar(dst=bad, data=mx, op0=nl.greater, operand0=32639.5, engine=VE)  # all finite
        ones = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=1.0)
        cnt = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=cnt, stationary=bad, moving=ones, accumulate=False)
        keep = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=keep, data=cnt, op0=nl.less, operand0=0.5, engine=VE)
        return keep

    def zero_rows(src, dst, keep):
        """dst (a native uint16 tile [T, n]) = src (a uint16 view of a bf16 tile [T, n], rows on the partitions) with
        each row whose keep is 0 set to +0 whole: the bits, read as integers, times keep [T, 1] (exact, no NaN)."""
        nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=keep, engine=nisa.vector_engine)

# -- end of the row helpers

REV_ROWS = zlib.crc32(repr((kernel_rev(__file__, "# -- in-kernel (NKI): whole-row", "# -- end of the row helpers"),
                            P)).encode())
