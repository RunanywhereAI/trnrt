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
