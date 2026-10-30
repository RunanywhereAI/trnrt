"""engine/state_pool.py on the host: the row layout (scratch, running rows, checkpoint rows, the zero row last), the zero
row zeros and never handed out or written, and a non-running holder's row reclaimed when the running rows run out."""

from __future__ import annotations

import types

import pytest
import torch

from trnrt.engine import state_pool as SP


class _Model:
    def state_shapes(self):
        return [((3, 8), (2, 4, 4)), ((3, 8), (2, 4, 4))]

    def bind_state(self, conv, rec):
        self.conv, self.rec = conv, rec


def _pool(rows=3, ckpt=4):
    return SP.StatePool(_Model(), rows, torch.device("cpu"), torch.bfloat16, ckpt)


def _req(rid, status=SP.Status.RUNNING):
    return types.SimpleNamespace(rid=rid, status=status)


def test_the_zero_row_is_the_last_and_zero():
    sp = _pool()
    assert sp.rows == 1 + 3 + 4 + 1 and sp.zero_row == sp.rows - 1
    for t in sp.pools():
        assert t.shape[0] == sp.rows and not t[sp.zero_row].any()
    assert [t.dtype for t in sp.pools()] == [torch.bfloat16] * 2 + [torch.float32] * 2  # conv, then recurrent fp32


def test_no_free_list_hands_out_the_zero_row_or_the_scratch_row():
    """Drain the running rows and the checkpoint rows: running 1 .. 3, checkpoints 4 .. 7, neither the scratch row nor
    the zero row among them."""
    sp = _pool()
    running = [sp.row(_req(f"r{n}")) for n in range(3)]
    with pytest.raises(RuntimeError, match="exhausted"):
        sp.row(_req("one too many"))
    ckpts = []
    while (c := sp.alloc_ckpt()) is not None:
        ckpts.append(c)
    assert running == [1, 2, 3] and sorted(ckpts) == [4, 5, 6, 7]
    assert SP.SCRATCH_ROW not in running + ckpts and sp.zero_row not in running + ckpts


def test_writes_to_the_zero_row_are_refused():
    sp = _pool()
    sp.check_writes([1, 2, sp.zero_row - 1])
    with pytest.raises(RuntimeError, match="zero row"):
        sp.check_writes([1, sp.zero_row])


def test_a_request_keeps_its_row_and_a_finished_holder_is_reclaimed():
    sp = _pool(rows=2)
    a, b = _req("a"), _req("b")
    ra, rb = sp.row(a), sp.row(b)
    assert sp.row(a) == ra and ra != rb
    a.status = SP.Status.FINISHED  # its engine has not released it yet
    c = _req("c")
    assert sp.row(c) == ra  # a running request's row (b's) is never taken
    sp.release(b)
    assert sp.row(_req("d")) == rb


def test_checkpoint_rows_round_trip():
    sp = _pool()
    assert sp.num_free_ckpts == 4 and sp.num_ckpt_rows == 4
    r = sp.alloc_ckpt()
    assert sp.num_free_ckpts == 3 and r not in sp.free_ckpt_rows()
    sp.free_ckpt(r)
    assert sp.num_free_ckpts == 4
    assert sp.bytes_per_row() == 2 * (3 * 8 * 2 + 2 * 4 * 4 * 4)
