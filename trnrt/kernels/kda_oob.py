"""The KDA shell's state-read clamp: an off form (RCLAMP 0), kept as the kernels' bodies reference it.

A row that starts its sequence reads the state pools' zero row (kernels/shell.py start_row), in range. Before the zero
row, such a row read "row R", out of range on purpose, and the DMA dropped the read (oob_mode skip), so the row saw the
zeros its tile was memset to; at four decode rows after a 4,096-row prefill every rank's program 0 stopped at its first
row-batched rec_pool read, a hardware-DGE DMA of exactly such a row, with every engine of both programs waiting on it.

RCLAMP 1 (off) makes no read out of range: right after the read rows are loaded into SBUF (sl's read half; cbat 1's
cidx read column) a new index tile takes them clamped into the pool (the loaded tile is only read: an in-place clamp let
neuronx-cc's scheduler put the write before the mask's reads, NCC_ISCH719 at four to eight rows), so every read lands,
and the rows the clamp moved are zeroed after their reads by a predicated copy: +0 over the memset's +0, the skip's
result bit for bit, whatever the row read holds (NaN included: a select, not a multiply). RCLAMP 2: the clamp without
the zeroing, a test-only form. The kernel lines of these forms end in "# RCLAMP". HW is the DGE mode of the shell's
state DMAs (the rec_pool reads and writes, cbat 0's conv rows): the hardware DGE."""

from __future__ import annotations

RCLAMP = 0


# -- in-kernel (NKI) ---------------------------------------------------------------------------------------------------

try:  # the Neuron venv
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nisa = nl = None

if nisa is not None:
    HW = nisa.dge_mode.hwdge
    VE = nisa.vector_engine

    def _sb(shape, dt=None):
        return nl.ndarray(shape, dtype=dt or nl.float32, buffer=nl.sbuf)

    def _clamp(dst, src, n: int):
        """dst = src clamped into [0, n - 1] (int32 row indices; exact through the vector engine's fp32)."""
        nisa.tensor_scalar(dst=dst, data=src, op0=nl.minimum, operand0=float(n - 1), op1=nl.maximum, operand1=0.0,
                           engine=VE)

    def _moved(x, n: int, p: int, w: int):
        """uint32 [p, w]: 1 where x (int32 [p, w]) is outside [0, n - 1], else 0."""
        c = _sb((p, w), nl.int32)
        _clamp(c, x, n)
        f = _sb((p, w))
        nisa.tensor_tensor(dst=f, data1=x, data2=c, op=nl.not_equal, engine=VE)
        m = _sb((p, w), nl.uint32)
        nisa.tensor_copy(dst=m, src=f, engine=VE)
        return m

    def clamp_rows(sl, slots, B: int, R: int, P: int):
        """sl [1, 2 B] int32 as loaded (each row's read row, then its write row) -> (sl2, nk): sl2 a new [1, 2 B] tile,
        the read half clamped into [0, R - 1] and the write half copied (sl is only read); nk uint32 [P, B], the same on
        every partition: 1 for each row whose read row was out of range (a row starting its sequence, a padded row),
        the rows zero() clears after their reads."""
        slp = _sb((P, B), nl.int32)
        nisa.dma_copy(dst=slp, src=slots.ap(pattern=[[0, P], [1, B]], offset=0), dge_mode=nisa.dge_mode.none)
        nk = _moved(slp, R, P, B)
        sl2 = _sb((1, 2 * B), nl.int32)
        _clamp(sl2[:, 0:B], sl[:, 0:B], R)
        nisa.tensor_copy(dst=sl2[:, B:2 * B], src=sl[:, B:2 * B], engine=VE)
        return sl2, nk

    def clamp_conv(ci, BK: int, RK: int):
        """ci [B (K - 1), 2] int32 as loaded (cbat 1's conv rows: read, write) -> (ci2, nkc): ci2 a new tile, the read
        column clamped into [0, RK - 1] and the write column copied (ci is only read); nkc uint32 [B (K - 1), 1], 1
        where the read row was out of range."""
        nkc = _moved(ci[:, 0:1], RK, BK, 1)
        ci2 = _sb((BK, 2), nl.int32)
        _clamp(ci2[:, 0:1], ci[:, 0:1], RK)
        nisa.tensor_copy(dst=ci2[:, 1:2], src=ci[:, 1:2], engine=VE)
        return ci2, nkc

    def zero(dst, m, p: int, n: int, col: int, w: int):
        """dst (p partitions, n elements each) = +0 where m [., w]'s column col is 1 on that partition, else kept: an
        out-of-range row's read cleared, the form kda_shell's A_log tie uses (a memset zero tile, a uint32 predicate)."""
        pr = _sb((p, n), nl.uint32)
        nisa.tensor_copy(dst=pr, src=m.ap(pattern=[[w, p], [0, n]], offset=col), engine=VE)
        z = _sb((p, n), dst.dtype)
        nisa.memset(dst=z, value=0.0, engine=nisa.gpsimd_engine)
        nisa.tensor_copy_predicated(dst=dst.reshape((p, n)) if len(dst.shape) > 2 else dst, src=z, predicate=pr)
else:
    HW = None
    clamp_rows = clamp_conv = zero = None

# -- end of the in-kernel helpers --------------------------------------------------------------------------------------
