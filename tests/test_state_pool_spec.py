"""engine/state_pool.py, speculative seats, on the host: the row layout, which seat a request takes, the plain-to-speculative move, the verify's current
row, rows held back while an overlapped step may still touch them, and reclaim of non-running holders.

The layout the shipped cap-12 configuration relies on: 12 seats of which 4 are speculative (k 2: 3 rows each) and 8 plain (1 row each), 12 checkpoint rows,
the scratch row and the zero row: 1 + 4 x 3 + 8 + 12 + 1 = 34 rows."""

from __future__ import annotations

import types

import pytest
import torch

from trnrt.engine import state_pool as SP

K = 2


class _Model:
    def state_shapes(self):
        return [((3, 8), (2, 4, 4))]

    def bind_state(self, conv, rec):
        self.conv, self.rec = conv, rec


def _pool(rows=12, spec_seats=4, ckpt=12, per_req=1 + K):
    return SP.StatePool(
        _Model(), rows, torch.device("cpu"), torch.bfloat16, ckpt, rows_per_req=per_req, spec_seats=spec_seats
    )


def _req(rid, status=SP.Status.RUNNING):
    return types.SimpleNamespace(rid=rid, status=status)


def test_the_shipped_cap_12_pool_is_34_rows_and_every_range_is_where_it_should_be():
    sp = _pool()
    assert sp.rows == 34 and sp.zero_row == 33 and sp.num_ckpt_rows == 12 and sp.mixed_seats
    assert [t.shape[0] for t in sp.pools()] == [34, 34]
    assert sp._free == [[10, 11, 12], [7, 8, 9], [4, 5, 6], [1, 2, 3]]  # pop() hands out [1, 2, 3] first
    assert sp._pfree == [[20], [19], [18], [17], [16], [15], [14], [13]]
    assert sorted(sp._ckpt_free) == list(range(21, 33))
    rows = [r for rs in sp._free + sp._pfree for r in rs] + sp._ckpt_free
    assert (
        len(rows) == len(set(rows)) == 32 and SP.SCRATCH_ROW not in rows and sp.zero_row not in rows
    )  # nothing hands out the scratch or the zero row


def test_without_spec_seats_every_seat_holds_rows_per_req_and_the_old_pool_is_unchanged():
    sp = _pool(rows=4, spec_seats=None, ckpt=2)
    assert not sp.mixed_seats and sp.rows == 1 + 4 * 3 + 2 + 1
    assert [sp.rows_of(_req(f"r{i}")) for i in range(4)] == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    one = SP.StatePool(_Model(), 3, torch.device("cpu"), torch.bfloat16, 4)  # the OSS signature, no speculation
    assert one.rows == 1 + 3 + 4 + 1 and [one.row(_req(f"x{i}")) for i in range(3)] == [1, 2, 3]


@pytest.mark.parametrize("seats", [0, 13, -1])
def test_spec_seats_outside_1_to_seats_are_refused(seats):
    with pytest.raises(ValueError, match="spec seats"):
        _pool(spec_seats=seats)


def test_a_first_use_takes_the_preferred_kind_then_the_other_and_never_the_zero_row():
    sp = _pool()
    spec = [sp.rows_of(_req(f"s{i}")) for i in range(4)]
    assert spec == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    plain = sp.rows_of(_req("p0"))  # the speculative seats are taken: the next request gets a plain seat
    assert plain == [13]
    sp2 = _pool()
    sp2.prefer_spec = False  # plain mode: plain seats first
    assert sp2.rows_of(_req("a")) == [13] and sp2.rows_of(_req("b")) == [14]
    for i in range(6):
        sp2.rows_of(_req(f"c{i}"))
    assert sp2.rows_of(_req("d")) == [1, 2, 3]  # the 8 plain seats are gone: the speculative ones
    drained = _pool()
    for i in range(12):
        drained.rows_of(_req(f"r{i}"))
    with pytest.raises(RuntimeError, match="exhausted"):
        drained.rows_of(_req("one too many"))


def test_the_verify_sets_the_current_row_and_row_follows_it():
    sp = _pool()
    r = _req("a")
    assert sp.row(r) == 1 and sp.holds(r) and sp.is_spec(r)
    sp.set_current(r, 2)
    assert sp.row(r) == 3 and sp.rows_of(r) == [1, 2, 3]  # the rows stay; the current index moves
    sp.set_current(r, 0)
    assert sp.row(r) == 1


def test_a_move_between_seat_kinds_returns_the_source_and_the_new_first_row_and_frees_the_old_seat():
    sp = _pool()
    r = _req("a")
    sp.prefer_spec = False
    assert sp.rows_of(r) == [13] and not sp.is_spec(r)
    src, dst = sp.move(r, to_spec=True)
    assert (src, dst) == (13, 1) and sp.rows_of(r) == [1, 2, 3] and sp.row(r) == 1  # the new first row is current
    assert sp._pfree[-1] == [13]  # the plain seat went back to its list
    sp.set_current(r, 2)
    src, dst = sp.move(r, to_spec=False)
    assert src == 3 and sp.rows_of(r) == [13]  # the CURRENT row is what the caller copies
    assert sp.move(r, to_spec=False) is None  # already that kind
    assert sp.move(_req("never-held"), to_spec=True) is None


def test_a_move_finds_no_seat_when_the_kind_is_full_and_reclaims_a_non_running_holder_first():
    sp = _pool()
    sp.prefer_spec = False
    r = _req("mover")
    sp.rows_of(r)
    holders = []
    sp.prefer_spec = True
    for i in range(4):
        h = _req(f"s{i}")
        sp.rows_of(h)
        holders.append(h)
    assert sp.spec_available() == 0
    assert sp.move(r, to_spec=True) is None  # every speculative seat belongs to a running request
    holders[1].status = SP.Status.WAITING  # preempted: its seat can be reclaimed
    assert sp.spec_available() == 1
    moved = sp.move(r, to_spec=True)
    assert moved is not None and sp.is_spec(r) and not sp.holds(holders[1])


def test_released_rows_are_held_back_while_an_overlapped_step_may_touch_them():
    sp = _pool(rows=2, spec_seats=1, ckpt=0)
    a, b = _req("a"), _req("b")
    ra, rb = sp.rows_of(a), sp.rows_of(b)
    assert ra == [1, 2, 3] and rb == [4]
    sp.hold = True
    sp.release(a)
    sp.release(b)
    assert sp._free == [] and sp._pfree == []  # nothing is reusable yet
    with pytest.raises(RuntimeError, match="exhausted"):
        sp.rows_of(_req("c"))
    sp.release_cooling()
    assert sp._free == [[1, 2, 3]] and sp._pfree == [[4]]
    sp.hold = False
    sp.release(_req("never-held"))  # releasing a stranger is a no-op


def test_a_non_running_holder_is_reclaimed_when_the_free_lists_are_empty():
    sp = _pool(rows=2, spec_seats=1, ckpt=0)
    a, b = _req("a"), _req("b")
    sp.rows_of(a), sp.rows_of(b)
    a.status = SP.Status.FINISHED
    c = _req("c")
    assert sp.rows_of(c) == [1, 2, 3] and not sp.holds(a)  # a's seat, reclaimed
    with pytest.raises(RuntimeError, match="exhausted"):
        sp.rows_of(_req("d"))  # b and c are running


def test_check_writes_refuses_the_zero_row():
    sp = _pool()
    sp.check_writes([1, 2, 21])
    with pytest.raises(RuntimeError, match="zero row"):
        sp.check_writes([3, sp.zero_row])
