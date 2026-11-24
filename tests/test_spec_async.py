"""Speculative decoding under overlap scheduling (engine/spec_async.py): the board graphs on hand-built boards.

The board is the one fp32 row per request slot that the blind speculative step reads and writes: the tokens after the last
verify, the accepted count, the newest token's position, the state row to continue from and the MTP drafts. Every graph is
checked on a board built by hand, so each expected value below is arithmetic done on the page, not another run of the code."""

import torch

from trnrt.engine import spec_async as sa


def test_board_graphs_follow_the_accepted_count():
    Q, k, ps = 2, 1, 4
    S = 6
    W = sa.width(Q, k)
    board = torch.zeros(S, W)
    # slot 1: newest token 50 at position 9 (T[0], acc 0), state row 7, draft 60; slot 3: last verify accepted
    # (T = 61, 70 with acc 1): newest token 70 at position 13, state row 9, draft 80
    board[1, :Q] = torch.tensor([50.0, 0.0])
    board[1, Q : Q + 4] = torch.tensor([0.0, 9.0, 7.0, 1.0])
    board[1, Q + 4] = 60.0
    board[3, :Q] = torch.tensor([61.0, 70.0])
    board[3, Q : Q + 4] = torch.tensor([1.0, 13.0, 9.0, 1.0])
    board[3, Q + 4] = 80.0
    slot_idx = torch.tensor([1, 3, 5])  # 5: padding (scratch)
    valid = torch.tensor([1.0, 1.0, 0.0])
    table = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [0, 0, 0, 0, 0]])
    rows = torch.tensor([[3, 4], [5, 6], [0, 0]])
    pad = torch.full((3, Q), -7, dtype=torch.int64)
    ids, draft, pos, slot, st = sa.spec_prep(board, slot_idx, table, rows, pad, valid, Q, k, ps, 1)
    assert ids[:2].tolist() == [[50, 60], [70, 80]]
    assert draft.view(3, Q)[:2].tolist() == [[60, 0], [80, 0]]
    assert pos[:2].tolist() == [[9, 10], [13, 14]]
    assert slot[:2].tolist() == [[12 * 4 + 1, 12 * 4 + 2], [23 * 4 + 1, 23 * 4 + 2]] and slot[2].tolist() == [-7, -7]
    assert st[:2].tolist() == [[7, 3, 4], [9, 5, 6]]
    # verify output: row 0 of slot 1 rejects (replacement 55), row 0 of slot 3 accepts, bonus 90
    cols = 6
    out = torch.zeros(3 * Q, cols)
    out[0] = torch.tensor([51.0, 0.0, 55.0, -1, -2, -3])
    out[1] = torch.tensor([52.0, 0.0, 0.0, -1, -2, -3])
    out[2] = torch.tensor([80.0, 1.0, 81.0, -1, -2, -3])
    out[3] = torch.tensor([90.0, 0.0, 91.0, -1, -2, -3])
    res = sa.spec_post(board, slot_idx, out, ids, rows.float(), valid, Q, k)
    assert res[:2].tolist() == [[55.0, 0.0, 0.0, 1.0], [80.0, 90.0, 1.0, 1.0]]
    assert board[1, : Q + 4].tolist() == [55.0, 0.0, 0.0, 10.0, 3.0, 1.0]  # newest 55 at 10, state row rows[0] = 3
    assert board[3, : Q + 4].tolist() == [80.0, 90.0, 1.0, 15.0, 6.0, 1.0]  # newest 90 at 15, state row rows[1] = 6
    T, last, mpos, _mslot = sa.mtp_prep(board, slot_idx, table, pad, valid, Q, ps, 1)
    assert T[:2].tolist() == [[55, 0], [80, 90]] and last[:2].tolist() == [0, 1]
    assert mpos[:2].tolist() == [[9, 10], [13, 14]]
    sa.mtp_post(board, slot_idx, torch.tensor([[66.0], [99.0], [5.0]]), valid, Q, k)
    assert board[1, Q + 4] == 66.0 and board[3, Q + 4] == 99.0 and board[5].abs().sum() == 0
    # the next prep: slot 1's newest is T[0] = 55, slot 3's T[1] = 90
    ids2, _, pos2, _, st2 = sa.spec_prep(board, slot_idx, table, rows, pad, valid, Q, k, ps, 1)
    assert ids2[:2].tolist() == [[55, 66], [90, 99]] and pos2[:2].tolist() == [[10, 11], [15, 16]]
    assert st2[:2, 0].tolist() == [3, 6]
    # after a prefill: the token board's sampled token, acc 0
    tb = torch.zeros(S)
    tb[4] = 77.0
    sa.spec_init(
        board,
        tb,
        torch.tensor([4, 5]),
        torch.tensor([20.0, 0.0]),
        torch.tensor([2.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        Q,
        k,
    )
    assert board[4, : Q + 4].tolist() == [77.0, 0.0, 0.0, 20.0, 2.0, 0.0]  # no drafts yet
    # a draftless row (nd 0) takes the sample y_0 whatever the accept flags
    out2 = torch.zeros(2 * Q, cols)
    out2[0] = torch.tensor([33.0, 1.0, 34.0, -1, -2, -3])
    res2 = sa.spec_post(
        board,
        torch.tensor([4, 5]),
        out2,
        torch.tensor([[77, 0], [0, 0]]),
        torch.tensor([[1.0, 2.0], [0.0, 0.0]]),
        torch.tensor([1.0, 0.0]),
        Q,
        k,
    )
    assert res2[0].tolist() == [33.0, 0.0, 0.0, 0.0] and board[4, Q + 1] == 21.0
    sa.spec_host_init(
        board,
        torch.tensor([2]),
        torch.tensor([44.0]),
        torch.tensor([30.0]),
        torch.tensor([5.0]),
        torch.tensor([1.0]),
        Q,
        k,
    )
    assert board[2, : Q + 4].tolist() == [44.0, 0.0, 0.0, 30.0, 5.0, 0.0]
    got = sa.mtp_prefill_ids(tb, torch.tensor([4, 5]), torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([2, -1]))
    assert got.tolist() == [[1, 2, 77], [4, 5, 6]]


def test_mtp_prep_gives_the_later_passes_positions_for_k_above_1():
    """k = 3: after spec_post the board's newest token sits at base; the first MTP pass covers base_old .. base_old
    + k with last index acc, the later two passes positions base and base + 1 (mtp_launch's last + 1 + s)."""
    Q, k, ps = 4, 3, 4
    board = torch.zeros(4, sa.width(Q, k))
    board[1, :Q] = torch.tensor([61.0, 70.0, 0.0, 0.0])
    board[1, Q : Q + 4] = torch.tensor([1.0, 13.0, 0.0, 1.0])  # acc 1, newest token at 13
    slot_idx, valid = torch.tensor([1, 3]), torch.tensor([1.0, 0.0])
    table = torch.tensor([[20, 21, 22, 23, 24], [0, 0, 0, 0, 0]])
    pad = torch.full((2, Q), -7, dtype=torch.int64)
    _T, last, pos, _slot, rpos, rslot = sa.mtp_prep(board, slot_idx, table, pad, valid, Q, ps, 1, None, k)
    assert last[0] == 1 and pos[0].tolist() == [11, 12, 13, 14]
    assert rpos[0].tolist() == [13, 14] and rslot[0].tolist() == [23 * 4 + 1, 23 * 4 + 2]
    assert rpos[1].tolist() == [0, 0] and rslot[1].tolist() == [-7, -7]
    assert len(sa.mtp_prep(board, slot_idx, table, pad, valid, Q, ps, 1)) == 4  # k 1: the graph is unchanged


def _verify_out(rows, cols=6):
    """A verify's result [R * Q, cols] from rows of (y, accept, replacement): the sampler's first three columns."""
    out = torch.zeros(len(rows), cols)
    for i, (y, acc, rep) in enumerate(rows):
        out[i, :3] = torch.tensor([y, acc, rep])
    return out


def _board_with_drafts(Q=3, k=2):
    board = torch.zeros(4, sa.width(Q, k))
    board[1, :Q] = torch.tensor([50.0, 0.0, 0.0])
    board[1, Q : Q + 4] = torch.tensor([0.0, 9.0, 7.0, 1.0])  # acc 0, newest token at 9, state row 7, real drafts
    board[1, Q + 4 : Q + 6] = torch.tensor([60.0, 61.0])
    return board


def test_a_synthetic_accept_flag_replaces_the_verifys_and_a_rejected_position_emits_the_target_sample():
    """spec_post(synth=...): the host's flags decide the accepted prefix, whatever the sampler said, and the position the
    prefix stops at emits the target's own sample y there, not the recovered token r the sampler computed."""
    Q, k = 3, 2
    slot_idx, valid = torch.tensor([1]), torch.tensor([1.0])
    rows = torch.tensor([[3.0, 4.0, 5.0]])
    out = _verify_out([(51, 1, 91), (52, 1, 92), (53, 0, 93)])  # the sampler accepted both drafts: bonus y_2 = 53
    ids = torch.tensor([[50, 60, 61]])

    real = _board_with_drafts(Q, k)
    res = sa.spec_post(real, slot_idx, out, ids, rows, valid, Q, k)
    assert res[0].tolist() == [60.0, 61.0, 53.0, 2.0, 1.0]  # T = both drafts then the bonus, acc 2
    assert real[1, Q : Q + 4].tolist() == [2.0, 9.0 + 1 + 2, 5.0, 1.0]  # newest at 12, state row rows[acc] = 5

    synth = _board_with_drafts(Q, k)
    res = sa.spec_post(synth, slot_idx, out, ids, rows, valid, Q, k, synth=torch.tensor([[1.0, 0.0]]))
    assert res[0].tolist() == [60.0, 52.0, 0.0, 1.0, 1.0]  # one draft accepted, then y_1 = 52 (not the recovered 92)
    assert synth[1, Q : Q + 4].tolist() == [1.0, 11.0, 4.0, 1.0]

    none = _board_with_drafts(Q, k)
    res = sa.spec_post(none, slot_idx, out, ids, rows, valid, Q, k, synth=torch.tensor([[0.0, 1.0]]))
    assert res[0].tolist() == [51.0, 0.0, 0.0, 0.0, 1.0] and none[1, Q + 1] == 10.0  # a first-draft reject: y_0


def test_a_one_row_call_gives_the_two_row_calls_first_row():
    """neuronx-cc misreads the board in a graph of ONE row, so the readers run such a call on the row twice and return the
    first copy. The results must be the ones of the same row in a two-row call."""
    Q, k, ps = 3, 2, 4
    board = _board_with_drafts(Q, k)
    table = torch.tensor([[10, 11, 12, 13]])
    rows = torch.tensor([[3, 4, 5]])
    pad = torch.full((1, Q), -7, dtype=torch.int64)
    one = sa.spec_prep(board, torch.tensor([1]), table, rows, pad, torch.tensor([1.0]), Q, k, ps, 1)
    two = sa.spec_prep(
        board, torch.tensor([1, 1]), table.repeat(2, 1), rows.repeat(2, 1), pad.repeat(2, 1), torch.ones(2), Q, k, ps, 1
    )
    assert one[0].tolist() == two[0][:1].tolist() == [[50, 60, 61]]
    assert one[1].tolist() == two[1][:Q].tolist()
    for a, b in zip(one[2:], two[2:], strict=True):
        assert a.tolist() == b[:1].tolist()
    mone = sa.mtp_prep(board, torch.tensor([1]), table, pad, torch.tensor([1.0]), Q, ps, 1)
    mtwo = sa.mtp_prep(board, torch.tensor([1, 1]), table.repeat(2, 1), pad.repeat(2, 1), torch.ones(2), Q, ps, 1)
    for a, b in zip(mone, mtwo, strict=True):
        assert a.tolist() == b[:1].tolist()


def test_slots_are_exact_through_a_page_table_at_a_power_of_two_page_size():
    ps = 256
    table = torch.tensor([[7, 3, 9, 4]], dtype=torch.int64)
    pos = torch.tensor([[0.0, 255.0, 256.0, 600.0, 1023.0]])
    got = sa._slots(pos, table, ps)[0].tolist()
    assert got == [7 * ps, 7 * ps + 255, 3 * ps, 9 * ps + 88, 4 * ps + 255]


def test_the_decode_checkpoint_a_verify_crossed_is_copied_on_the_device_and_nothing_else_moves():
    """spec_ckpt: a verify whose accepted positions cross a multiple b of the checkpoint interval t copies the state after
    b - 1 (the request's row b - 1 - old) into the reserved checkpoint row; a verify that crossed none, a row without a
    reserved checkpoint row and a padding row copy the scratch row onto itself."""
    Q, k, t = 3, 2, 4
    board = torch.zeros(4, sa.width(Q, k))
    # old = base - 1 - acc is the verify's first position; its computed positions are old .. old + acc
    board[1, Q : Q + 2] = torch.tensor([2.0, 9.0])  # acc 2, base 9: old 6, positions 6 .. 8, b = 8 > 6: crossed
    board[2, Q : Q + 2] = torch.tensor([1.0, 7.0])  # acc 1, base 7: old 5, positions 5 .. 6, b = 4 <= 5: none
    board[3, Q : Q + 2] = torch.tensor([2.0, 9.0])  # crossed, but no checkpoint row reserved
    pool = torch.arange(12.0).reshape(12, 1).repeat(1, 2)
    before = pool.clone()
    rows_all = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])
    ckpt = torch.tensor([0.0, 11.0, 10.0, 0.0])
    sa.spec_ckpt(board, torch.tensor([0, 1, 2, 3]), rows_all, ckpt, torch.tensor([0.0, 1.0, 1.0, 1.0]), Q, t, [pool])
    assert pool[11].tolist() == before[4].tolist(), (
        "b - 1 - old = 1: slot 1's second row (4) went to its checkpoint row"
    )
    assert pool[10].tolist() == before[10].tolist(), "no boundary crossed: slot 2's reserved row is untouched"
    keep = [r for r in range(12) if r != 11]
    assert pool[keep].tolist() == before[keep].tolist()
