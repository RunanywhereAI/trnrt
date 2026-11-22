"""trnrt/engine/host_arena.py: the shared host arenas of the save-once host cache.

In this process: the layout (offsets aligned, views share the payload, every cache dtype round-trips bit for bit); the
peer and owner maths (world 8, attention TP 8, CP 2: four row groups; a second attention group); the generation fence
(a missing save TIMES OUT, a slot reused under a pending load RAISES, never a stale read; the cell is a seqlock); the
completion board; the tags; the /dev/shm capacity guard; and attaching a segment that is still being made.
With real separate processes (the tests named *_process_*): an arena written in one process and read in another, a reader
waiting for a late writer, a killed rank's segment cleaned at the next start, a rank stopped by SIGTERM unlinking its
segments, and a four-process round in which every rank saves the slots its row group owns and reads all of them back.
Each case with a plant that must trip says so."""

from __future__ import annotations

import contextlib
import itertools
import multiprocessing as mp
import os
import signal
import threading
import time
import types

import numpy as np
import pytest
import torch

from trnrt.engine import host_arena as H

CTX = mp.get_context("spawn")
_TAGS = iter(range(10_000_000))


def _tag() -> str:
    return (
        f"t{os.getpid() % 9973}x{next(_TAGS) % 9973}"  # short: macOS caps a POSIX shared-memory name at 31 characters
    )


def _caches(lps=4):
    """Four paged caches of different dtypes and row shapes, as the runner's (k bf16, v fp8, a state fp32, a pool key bf16)."""
    return [
        torch.zeros(10 * lps, 1, 64, dtype=torch.bfloat16),
        torch.zeros(10 * lps, 1, 32, dtype=torch.float8_e4m3fn),
        torch.zeros(10 * lps, 4, dtype=torch.float32),
        torch.zeros(10 * lps, 128, dtype=torch.bfloat16),
    ]


def _content(layout, slot, gen):
    """The bytes a page of this slot at this generation holds: deterministic per (slot, gen), different per cache."""
    g = torch.Generator().manual_seed(1000 * gen + slot)
    return [torch.randn(layout.lps, *shp, generator=g).to(dt) for dt, shp in zip(layout.dtypes, layout.shapes)]


def _bits(t):
    return t.contiguous().reshape(-1).view(torch.uint8)


def _nbytes(d, s):
    return 4 * int(np.prod(s)) * torch.empty((), dtype=d).element_size()


# -- layout ---------------------------------------------------------------------------------------------------------------


def test_layout_offsets_are_aligned_views_share_the_payload_and_round_trip_every_dtype():
    lay = H.Layout(_caches(), 4)
    assert all(o % H.ALIGN == 0 for o in lay.offsets) and lay.slot_bytes % H.ALIGN == 0
    assert lay.slot_bytes >= sum(_nbytes(d, s) for d, s in zip(lay.dtypes, lay.shapes))
    payload = torch.zeros(lay.slot_bytes, dtype=torch.uint8)
    want = _content(lay, 3, 1)
    for v, w in zip(lay.views(payload), want):
        assert v.shape == w.shape and v.dtype == w.dtype
        v.copy_(w)
    assert all(torch.equal(_bits(v), _bits(w)) for v, w in zip(lay.views(payload), want))
    assert payload.any()  # the views wrote INTO the payload, they are not copies
    regions = [(o, o + _nbytes(d, s)) for o, d, s in zip(lay.offsets, lay.dtypes, lay.shapes)]
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(regions))  # no two caches overlap
    flat = torch.zeros(3 * lay.slot_bytes, dtype=torch.uint8)
    for v, w in zip(lay.arena_views(flat.view(3, lay.slot_bytes)), want):
        v[1].copy_(w)
    assert torch.equal(flat[lay.slot_bytes : 2 * lay.slot_bytes], payload)  # the all-slot views address slot 1 alike


# -- peers and owners -------------------------------------------------------------------------------------------------------


def test_peers_are_the_ranks_of_one_cp_index_across_the_row_groups():
    assert H.peers(5, 8, 8, 2) == [1, 3, 5, 7] and H.peers(0, 8, 8, 2) == [0, 2, 4, 6]
    assert [H.row_group(r, 8, 2) for r in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert H.peers(13, 16, 8, 2) == [9, 11, 13, 15]  # the second attention group of 8 stays inside itself
    assert H.peers(31, 32, 32, 8) == [7, 15, 23, 31]  # GLM-5.3-Flash at tp 32: CP 8, four row groups
    with pytest.raises(ValueError, match="do not nest"):
        H.peers(0, 8, 8, 3)


def test_slot_ownership_is_round_robin_over_the_row_groups():
    lay = H.Layout(_caches(), 4)
    st = H.HostStore(_tag(), 5, 8, 8, 2, lay, 40)
    try:
        assert st.R == 4 and st.mine == 2 and st.per_arena == 10
        assert [st.owner(s) for s in range(8)] == [0, 1, 2, 3, 0, 1, 2, 3]
        with pytest.raises(RuntimeError, match="owned by row group 1"):
            st.commit(1, 1)  # slot 1 belongs to row group 1
    finally:
        st.close()


def test_tag_of_keeps_zero_for_untagged():
    assert H.tag_of(0) == 1 and H.tag_of(-1) == (1 << 64) - 1 and H.tag_of(5) == 5 and H.NOTAG == 0


# -- the generation fence -----------------------------------------------------------------------------------------------------


def test_plant_a_load_whose_save_never_ran_times_out_instead_of_reading_stale_bytes():
    tag, lay = _tag(), H.Layout(_caches(), 4)
    a = H.HostStore(tag, 2, 4, 4, 2, lay, 8)  # rank 2: row group 1, CP index 0, owns slots 1, 3, 5, 7
    b = H.HostStore(tag, 0, 4, 4, 2, lay, 8)
    try:
        t = time.time()
        with pytest.raises(TimeoutError, match="never arrived"):
            b.wait_gen(1, 3, timeout=0.5)
        assert time.time() - t < 10
    finally:
        a.close()
        b.close()


def test_plant_a_slot_reused_under_a_pending_load_raises():
    tag, lay = _tag(), H.Layout(_caches(), 4)
    own = H.HostStore(tag, 2, 4, 4, 2, lay, 8)
    rd = H.HostStore(tag, 0, 4, 4, 2, lay, 8)
    try:
        own.commit(3, 4)  # generation 4 is in the arena; the pending load was issued for generation 2
        with pytest.raises(RuntimeError, match="reused under a pending load"):
            rd.wait_gen(3, 2, timeout=1)
        rd.wait_gen(3, 4, timeout=1)  # and the right generation passes at once
    finally:
        own.close()
        rd.close()


def test_the_cell_is_a_seqlock_a_save_in_progress_is_waited_for_and_a_rewrite_under_a_copy_is_caught():
    tag = _tag()
    lay = H.Layout(_caches(), 4)
    own = H.HostStore(tag, 2, 4, 4, 2, lay, 8)  # rank 2: row group 1, CP index 0
    rd = H.HostStore(tag, 0, 4, 4, 2, lay, 8)  # rank 0: row group 0, CP index 0, reads row group 1's slots from rank 2
    try:
        s = 1  # owner = slot mod 2 = row group 1
        own.begin(s, 1)
        with pytest.raises(TimeoutError, match="never arrived"):
            rd.wait_gen(s, 1, timeout=0.2)  # being written: wait, it is not "reused"
        own.commit(s, 1)
        rd.wait_gen(s, 1, timeout=1)
        rd.check_gen(s, 1)  # nothing moved during a copy
        own.begin(s, 2)  # the owner starts the next generation while the reader copies
        with pytest.raises(RuntimeError, match="being rewritten"):
            rd.check_gen(s, 1)
        with pytest.raises(RuntimeError, match="reused under a pending load"):
            rd.wait_gen(s, 1, timeout=1)
        own.commit(s, 2)
        with pytest.raises(RuntimeError, match="rewritten during the copy"):
            rd.check_gen(s, 1)  # and a finished rewrite is caught too
        rd.wait_gen(s, 2, timeout=1)
        rd.check_gen(s, 2)
    finally:
        rd.close()
        own.close()


def test_a_reader_sees_the_owners_bytes_and_tag():
    tag, lay = _tag(), H.Layout(_caches(), 4)
    own = H.HostStore(tag, 2, 4, 4, 2, lay, 8)
    rd = H.HostStore(tag, 0, 4, 4, 2, lay, 8)
    try:
        want = _content(lay, 5, 1)
        own.begin(5, 1)
        for v, w in zip(own.views(5), want):
            v.copy_(w)
        own.commit(5, 1, H.tag_of(12345))
        rd.wait_gen(5, 1, timeout=1)
        assert all(torch.equal(_bits(v), _bits(w)) for v, w in zip(rd.views(5), want))
        rd.check_tag(5, H.tag_of(12345))
        with pytest.raises(RuntimeError, match="does not hold the page"):
            rd.check_tag(5, H.tag_of(54321))
        own.begin(5, 2)
        own.commit(5, 2)  # saved again untagged: NOTAG, never the previous page's tag
        assert rd.tag_at(5) == H.NOTAG
    finally:
        rd.close()
        own.close()


# -- the completion board ---------------------------------------------------------------------------------------------------


def test_the_board_holds_the_minimum_command_every_rank_finished():
    tag = _tag()
    b = H.Board.create(H.board_name(tag), 4)
    c = H.Board.attach(H.board_name(tag), 4)
    try:
        assert b.min_done() == 0
        for r, s in enumerate([9, 9, 7, 9]):
            c.mark(r, s)
        assert b.min_done() == 7  # rank 2 is a slow reader still inside command 8 or earlier
        c.mark(2, 9)
        assert b.min_done() == 9
    finally:
        c.close()
        b.close()


# -- the capacity guard -----------------------------------------------------------------------------------------------------


def test_the_capacity_guard_refuses_arenas_that_do_not_fit_the_tmpfs(monkeypatch, tmp_path):
    monkeypatch.setattr(H.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=10 * 2**30))
    H.check_capacity(9 * 2**30, path=str(tmp_path))
    with pytest.raises(ValueError, match=r"need up to 11\.0 GiB"):
        H.check_capacity(11 * 2**30, path=str(tmp_path))
    H.check_capacity(11 * 2**30, path=str(tmp_path / "no-such-dir"))  # no tmpfs here: skipped, not refused


def _sized(path, size: int, backed: int = 0) -> None:
    with open(path, "wb") as f:
        f.write(b"x" * backed)
        f.truncate(size)


def test_the_capacity_guard_counts_what_other_engines_arenas_may_still_take(monkeypatch, tmp_path):
    """An engine that started earlier (core base 0) has sized its arenas, 6 GiB, and backed 1 MiB of them: the rest is
    not free for the next engine (core base 64), though the tmpfs reports it free. Its own tag and other files are not
    counted."""
    monkeypatch.setattr(H.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=10 * 2**30))
    other = tmp_path / f"{H.PREFIX}c0_a0"
    _sized(other, 6 * 2**30, backed=2**20)
    _sized(tmp_path / f"{H.PREFIX}c64_a0", 4 * 2**30)
    _sized(tmp_path / "unrelated", 4 * 2**30)
    st = os.stat(other)
    if st.st_blocks * 512 >= st.st_size:
        pytest.skip("this filesystem keeps no sparse files")
    assert H.unbacked_bytes(str(tmp_path), skip_tag="c64") == st.st_size - st.st_blocks * 512
    H.check_capacity(3 * 2**30, path=str(tmp_path), tag="c64")  # about 4 GiB left
    with pytest.raises(ValueError, match=r"other engines' arenas may still fill 6\.0 GiB"):
        H.check_capacity(5 * 2**30, path=str(tmp_path), tag="c64")  # 10 GiB free by the tmpfs's count


# -- attaching a segment that is still being made ----------------------------------------------------------------------------
# The names are derived, not handed over after creation, so a reader can reach a name its creator has made but not yet
# sized (and, for an arena, sized but its header not yet written). multiprocessing.shared_memory fails there and, on macOS,
# unlinks the name from under the creator. attach maps only a sized segment, waits for a published header, never unlinks.


def _raw_unsized(name):
    import _posixshmem

    # named, size 0: what SharedMemory(create=True) shows between shm_open and ftruncate
    return _posixshmem.shm_open("/" + name, os.O_CREAT | os.O_EXCL | os.O_RDWR, mode=0o600)


def _raw_unlink(name):
    import _posixshmem

    with contextlib.suppress(FileNotFoundError):
        _posixshmem.shm_unlink("/" + name)


def test_attach_does_not_map_or_unlink_a_segment_its_creator_has_not_sized_yet():
    import _posixshmem

    name = f"{H.PREFIX}{_tag()}_u"
    fd = _raw_unsized(name)
    try:
        with pytest.raises(TimeoutError, match="no size yet"):
            H._Seg.attach(name, timeout=0.2)
        os.close(_posixshmem.shm_open("/" + name, os.O_RDWR, mode=0o600))  # still there: attach did not unlink it
    finally:
        os.close(fd)
        _raw_unlink(name)


def test_an_arena_attach_waits_for_the_size_and_then_for_the_header():
    name = f"{H.PREFIX}{_tag()}_w"
    cap, sb = 3, 4096
    fd = _raw_unsized(name)
    out = {}

    def late():
        time.sleep(0.15)
        os.ftruncate(fd, H.Arena.total_bytes(cap, sb))  # sized: the header is still zeros
        time.sleep(0.15)
        seg = H._Seg.attach(name, 5.0)
        ctl = np.ndarray((2,), dtype=np.uint64, buffer=seg.shm.buf, offset=0)
        ctl[1] = sb
        ctl[0] = cap  # published
        out["t"] = time.time()
        del ctl
        seg.close()

    t = threading.Thread(target=late)
    t0 = time.time()
    t.start()
    try:
        a = H.Arena.attach(name, timeout=10.0)
        assert (a.capacity, a.slot_bytes) == (cap, sb)
        assert time.time() - t0 >= 0.3 - 0.02  # it waited for both
        assert out["t"] <= time.time()
        a.close()
    finally:
        t.join()
        os.close(fd)
        _raw_unlink(name)


def test_plant_the_stock_shared_memory_attach_fails_in_that_window():
    """Why the attach is our own: the standard library's, on a named but unsized segment, raises (and on macOS unlinks the
    name). Not a test of ours: a test that the window is real."""
    from multiprocessing import shared_memory

    name = f"{H.PREFIX}{_tag()}_s"
    fd = _raw_unsized(name)
    try:
        with pytest.raises((ValueError, OSError)):
            shared_memory.SharedMemory(name=name, track=False)
    finally:
        os.close(fd)
        _raw_unlink(name)


# -- with real processes ------------------------------------------------------------------------------------------------------


def _reader(name, slot, lay_args, q):
    lay = H.Layout(*lay_args)
    a = H.Arena.attach(name)
    got = lay.views(a.payload(slot))
    q.put((int(a.gen[slot]), [bytes(_bits(v).numpy().tobytes()) for v in got]))
    a.close()


def test_process_an_arena_written_here_is_read_bit_for_bit_in_another_process():
    lay = H.Layout(_caches(), 4)
    name = H.arena_name(_tag(), 0)
    a = H.Arena.create(name, 6, lay.slot_bytes)
    try:
        want = _content(lay, 2, 5)
        for v, w in zip(lay.views(a.payload(2)), want):
            v.copy_(w)
        a.gen[2] = 5
        q = CTX.Queue()
        p = CTX.Process(target=_reader, args=(name, 2, (_caches(), 4), q))
        p.start()
        gen, got = q.get(timeout=60)
        p.join(30)
        assert gen == 5 and got == [bytes(_bits(w).numpy().tobytes()) for w in want]
    finally:
        a.close()
        _raw_unlink(name)


def _late_writer(tag, rank, world, attn, cp, lay_args, capacity, slot, gen, delay, q):
    lay = H.Layout(*lay_args)
    st = H.HostStore(tag, rank, world, attn, cp, lay, capacity)
    q.put("created")
    time.sleep(delay)
    for v, w in zip(st.views(slot), _content(lay, slot, gen)):
        v.copy_(w)
    st.commit(slot, gen)
    q.put("committed")
    time.sleep(30)
    st.close()


def test_process_a_reader_waits_for_the_save_and_reads_the_bytes_that_were_committed():
    tag, lay = _tag(), H.Layout(_caches(), 4)
    q = CTX.Queue()
    # world 4, attention TP 4, CP 2: row groups 0 = {0, 1}, 1 = {2, 3}; slot 1 is owned by row group 1, rank 2 for CP 0
    p = CTX.Process(target=_late_writer, args=(tag, 2, 4, 4, 2, (_caches(), 4), 8, 1, 7, 0.6, q), daemon=True)
    p.start()
    assert q.get(timeout=60) == "created"
    me = H.HostStore(tag, 0, 4, 4, 2, lay, 8)
    try:
        t = time.time()
        me.wait_gen(1, 7)  # blocks until the writer commits (about 0.6 s)
        assert 0.3 < time.time() - t < 20
        assert all(torch.equal(_bits(v), _bits(w)) for v, w in zip(me.views(1), _content(lay, 1, 7)))
    finally:
        me.close()
        p.terminate()  # its SIGTERM hook unlinks its segment
        p.join(30)
        H.cleanup_stale(tag, names=[H.arena_name(tag, r) for r in range(4)])


def _make_and_wait(name, q):
    H.Arena.create(name, 4, 4096)
    q.put("up")
    time.sleep(60)


def test_process_a_killed_ranks_segment_is_cleaned_at_the_next_start():
    tag = _tag()
    name = H.arena_name(tag, 3)
    q = CTX.Queue()
    p = CTX.Process(target=_make_and_wait, args=(name, q), daemon=True)
    p.start()
    assert q.get(timeout=60) == "up"
    os.kill(p.pid, signal.SIGKILL)  # no atexit, no unlink: the segment survives its owner
    p.join(30)
    H.Arena.attach(name, timeout=2).close()  # still there
    assert H.cleanup_stale(tag, names=[name, H.arena_name(tag, 9)]) == [name]  # the second never existed: skipped
    with pytest.raises(TimeoutError):
        H.Arena.attach(name, timeout=0.3)


def _term_child(tag, marker, q):
    if marker:

        def prev(signum, frame):
            with open(marker, "w") as f:
                f.write("the previous handler ran")

        signal.signal(signal.SIGTERM, prev)
    st = H.HostStore(tag, 0, 4, 4, 2, H.Layout(_caches(), 4), 8)
    st.attach_board(create=True)
    q.put("ready")
    time.sleep(120)


def _gone(names):
    for n in names:
        with pytest.raises(TimeoutError):
            H._Seg.attach(n, timeout=0.2)


@pytest.mark.parametrize("with_previous_handler", [False, True])
def test_process_a_rank_stopped_by_sigterm_unlinks_its_segments(tmp_path, with_previous_handler):
    """atexit does not run on a signal: the hook unlinks the segments, then the rank dies by the signal (or the handler it
    had before runs)."""
    tag = _tag()
    names = [H.arena_name(tag, 0), H.board_name(tag)]
    marker = str(tmp_path / "prev") if with_previous_handler else None
    q = CTX.Queue()
    p = CTX.Process(target=_term_child, args=(tag, marker, q))
    p.start()
    try:
        assert q.get(timeout=60) == "ready"
        for n in names:
            H._Seg.attach(n, timeout=1.0).close()  # both exist while it runs
        os.kill(p.pid, signal.SIGTERM)
        if marker:
            end = time.time() + 30
            while not os.path.exists(marker) and time.time() < end:
                time.sleep(0.05)
            assert os.path.exists(marker), "the handler the process had before was not called"
        else:
            p.join(30)
            assert p.exitcode == -signal.SIGTERM, p.exitcode  # the default action was kept
        _gone(names)
    finally:
        p.kill()
        p.join()
        H.cleanup_stale(tag, names=names)


def _wait_board(st, seq, timeout=30.0):
    """The busy rule, as the host cache applies it: a slot is rewritten only once every rank finished the round that read
    it (Board.min_done() >= seq)."""
    end = time.time() + timeout
    while st.board.min_done() < seq:
        if time.time() > end:
            raise TimeoutError(f"board stuck at {st.board.min_done()}, wanted {seq}")
        time.sleep(0.002)


def _rank(rank, tag, lay_args, nslots, q, rounds, skip_busy_rule=False):
    """world 4, attention TP 4, CP 2: row groups {0, 1} and {2, 3}. Round r: save the slots this row group owns at
    generation r (after every rank finished round r - 1's reads: the busy rule), then read EVERY slot back and report
    whether the bytes equal what the owner wrote. With skip_busy_rule the owner rewrites at once: a slow reader is then
    overtaken, and the fence must RAISE, never hand back the new bytes as if they were the old."""
    lay = H.Layout(*lay_args)
    st = H.HostStore(tag, rank, 4, 4, 2, lay, nslots)
    st.attach_board(create=rank == 0)
    q.put(("up", rank))
    st.board.mark(rank, 0)
    ok, err = True, ""
    try:
        for rd in range(1, rounds + 1):
            if not skip_busy_rule:
                _wait_board(st, rd - 1)
            for s in range(nslots):
                if st.owner(s) == st.mine:
                    st.begin(s, rd)  # the runner's order (ModelRunner._kv_save_many_so): mark, write, stamp
                    for v, w in zip(st.views(s), _content(lay, s * 10 + rank % 2, rd)):
                        v.copy_(w)
                    st.commit(s, rd)
            if skip_busy_rule and rank == 3:
                time.sleep(0.5)  # a slow reader: the others race ahead into round 2
            for s in range(nslots):
                st.wait_gen(s, rd, timeout=20)
                writer = st.peer[st.owner(s)]  # the owner row group's rank at MY CP index wrote what I read
                want = _content(lay, s * 10 + writer % 2, rd)
                ok &= all(torch.equal(_bits(v), _bits(w)) for v, w in zip(st.views(s), want))
                st.check_gen(s, rd)  # after the copy too: a slot rewritten under it is an error, never a wrong byte
            st.board.mark(rank, rd)
    except (RuntimeError, TimeoutError) as e:
        ok, err = False, f"{type(e).__name__}: {e}"
    q.put(("done", rank, ok, err))
    time.sleep(1.0)
    st.close()


def _run_four(rounds, nslots, **kw):
    tag, lay_args = _tag(), (_caches(), 4)
    q = CTX.Queue()
    procs = [
        CTX.Process(target=_rank, args=(r, tag, lay_args, nslots, q, rounds), kwargs=kw, daemon=True) for r in range(4)
    ]
    for p in procs:
        p.start()
    done = []
    try:
        deadline = time.time() + 90
        while len(done) < 4 and time.time() < deadline:
            m = q.get(timeout=90)
            if m[0] == "done":
                done.append(m)
    finally:
        for p in procs:
            p.join(20)
            if p.is_alive():
                p.terminate()
                p.join(20)
        H.cleanup_stale(tag, names=[H.arena_name(tag, r) for r in range(4)] + [H.board_name(tag)])
    return done


def test_process_four_ranks_each_save_what_they_own_and_all_read_everything_back_bit_for_bit():
    done = _run_four(rounds=3, nslots=12)
    assert len(done) == 4 and all(d[2] for d in done), done


def test_process_plant_without_the_busy_rule_a_slow_reader_is_overtaken_and_the_fence_raises():
    done = _run_four(rounds=2, nslots=12, skip_busy_rule=True)
    assert len(done) == 4
    bad = [d for d in done if not d[2]]
    overtaken = ("reused under a pending load", "rewritten during the copy")  # before the copy, or under it
    assert any(any(t in d[3] for t in overtaken) for d in bad), done  # the overtaken reader raised
    assert all(d[3] for d in bad), done  # and no rank failed SILENTLY (wrong bytes with no error)
