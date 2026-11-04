"""Rank 0's calls through a shared-memory segment instead of gloo's broadcast_object_list (trnrt/engine/tp.py
ShmChannel, on unless TRNRT_TP_SHM=0).

- One writer and its readers: every reader gets every call in order, and the writer waits for every reader's
  acknowledgement before it overwrites a call; a reader never touches the resource tracker.
- A call larger than the segment goes through broadcast_object_list in its turn (the fallback), and a segment that
  cannot be created sends every call over gloo; drop_channel unlinks a channel the shutdown call did not close (a send
  still waiting on a dead rank then fails), and a reader whose rank 0 is gone stops instead of waiting forever.
- The module's send / recv over a real gloo group of three CPU processes (no model): the hello, the calls in order with
  the large ones over gloo, the shutdown closing the segment; and the same calls with the channel off.
- Tensor-parallel engines generate the same tokens and logprobs with the channel, with a segment small enough that the
  large calls take the fallback, and without it.
"""

from __future__ import annotations

import logging
import os
import queue
import socket
import threading
import time
import types

import numpy as np
import pytest

from trnrt.engine import tp


def _exchange(ch, world: int, msgs: list, bind=None, timeout: float = 30.0) -> dict:
    """Readers 1 .. world - 1 on threads, each attached to `ch` until the shutdown call (None), and the writer sending
    `msgs` on a thread of its own; bind(rank) first on each. Returns {rank: the calls it received}. A side that raises,
    or is still waiting after `timeout`, fails the test instead of hanging it: the waiting sides' views are dropped, so
    each wait raises and its thread ends."""
    got, errors, readers = {r: [] for r in range(1, world)}, [], []

    def reader(rank):
        try:
            if bind is not None:
                bind(rank)
            r = tp.ShmChannel.attach(ch.shm.name, world, rank)
            readers.append(r)
            try:
                while True:
                    m = r.recv()
                    got[rank].append(m)
                    if m is None:
                        return
            finally:
                r.close()
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append((rank, repr(e)))

    def writer():
        try:
            if bind is not None:
                bind(0)
            for m in msgs:
                ch.send(m)  # each send waits for every reader's ack of the last: never overwrites an unread call
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append((0, repr(e)))

    ts = [threading.Thread(target=reader, args=(r,), daemon=True) for r in range(1, world)]
    ts.append(threading.Thread(target=writer, daemon=True))
    for t in ts:
        t.start()
    deadline = time.monotonic() + timeout
    for t in ts:
        t.join(max(0.0, deadline - time.monotonic()))
    stuck = [t.name for t in ts if t.is_alive()]
    if stuck:
        ch.acks = None
        for r in readers:
            r.ctl = None
    assert not stuck and not errors, (stuck, errors)
    return got


def test_one_writer_two_readers_in_order():
    ch = tp.ShmChannel.create(3, 1 << 16)
    msgs = [("decode", ("decode", 1, 8), [list(range(i, i + 50)), {"k": i}]) for i in range(200)] + [None]
    try:
        got = _exchange(ch, 3, msgs)
    finally:
        ch.close()
    assert got[1] == msgs and got[2] == msgs


def test_a_reader_never_touches_the_resource_tracker(monkeypatch):
    """The ranks share the creator's tracker, which holds a name once: a reader's register / unregister dropped the
    creator's entry and raised KeyError in the tracker for the next reader. Both attach paths (track=False; Python
    3.12's, which has no track) leave the tracker alone; a plain attach shows the recorder sees a register, so an empty
    record is not a broken detector."""
    from multiprocessing import resource_tracker, shared_memory

    ch = tp.ShmChannel.create(2, 1 << 14)
    calls = []
    monkeypatch.setattr(resource_tracker, "register", lambda name, rtype: calls.append(("register", name)))
    monkeypatch.setattr(resource_tracker, "unregister", lambda name, rtype: calls.append(("unregister", name)))
    try:
        shared_memory.SharedMemory(name=ch.shm.name).close()
        assert calls == [("register", ch.shm._name)]
        calls.clear()
        tp.ShmChannel.attach(ch.shm.name, 2, 1).close()
        assert calls == []
        real = shared_memory.SharedMemory

        class NoTrack(real):  # Python 3.12's signature
            def __init__(self, name=None, create=False, size=0, **kw):
                if kw:
                    raise TypeError("SharedMemory.__init__() got an unexpected keyword argument 'track'")
                super().__init__(name, create, size)

        recorder = resource_tracker.register
        monkeypatch.setattr(shared_memory, "SharedMemory", NoTrack)
        tp.ShmChannel.attach(ch.shm.name, 2, 1).close()
        assert calls == [] and resource_tracker.register is recorder  # suppressed during the open, then restored
    finally:
        monkeypatch.undo()
        ch.close()


class _Bcast:
    """dist.broadcast_object_list between threads of one process: rank 0's call hands its object to every reader's
    queue, a reader's call takes the next one from its own. The calling thread's rank is RANK.r."""

    RANK = threading.local()

    def __init__(self, world: int):
        self.q = {r: queue.Queue() for r in range(1, world)}
        self.sent = []

    def __call__(self, objs, src=0, **kw):
        if self.RANK.r == 0:
            self.sent.append(objs[0])
            for q in self.q.values():
                q.put(objs[0])
        else:
            objs[0] = self.q[self.RANK.r].get(timeout=30)


def test_a_call_larger_than_the_segment_goes_through_the_broadcast(monkeypatch):
    """Mode 2: the header says the call follows by broadcast_object_list, and both sides take it there in its turn;
    the small calls around it stay in the segment."""
    bc = _Bcast(3)
    monkeypatch.setattr(tp.dist, "broadcast_object_list", bc)
    ch = tp.ShmChannel.create(3, 4096)
    assert ch.cap == 4096 - ch.off
    big = ("prefill", ("prefill", 4096), [np.arange(4096, dtype=np.int64)])  # 32 KiB of ids: past the segment
    msgs = [("decode", ("decode", 1), [np.arange(8)]), big, ("decode", ("decode", 2), [np.arange(16)]), big, None]

    def bind(rank):
        _Bcast.RANK.r = rank

    try:
        got = _exchange(ch, 3, msgs, bind)
    finally:
        ch.close()
    for rank in (1, 2):
        assert len(got[rank]) == len(msgs) and got[rank][-1] is None
        for a, b in zip(got[rank][:-1], msgs[:-1]):
            assert a[:2] == b[:2] and np.array_equal(a[2][0], b[2][0])
    assert len(bc.sent) == 2 and all(s[0] == "prefill" for s in bc.sent)  # only the large calls took the fallback


def _fake_dist(world: int, sent: list):
    return types.SimpleNamespace(
        broadcast_object_list=lambda objs, src=0: sent.append(objs[0]), get_world_size=lambda: world, get_rank=lambda: 0
    )


def test_a_segment_that_cannot_be_made_sends_every_call_over_gloo(monkeypatch, caplog):
    """Nothing is announced before the segment exists, so a failed create (no usable /dev/shm) leaves the workers on
    the gloo path: every call goes there, said once; the engine's stop (drop_channel) clears the failure."""
    sent = []
    monkeypatch.setattr(tp, "dist", _fake_dist(2, sent))
    monkeypatch.setattr(tp, "SHM", True)

    def fail(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(tp.ShmChannel, "create", staticmethod(fail))
    try:
        with caplog.at_level(logging.INFO, logger="trnrt"):
            tp.send(("decode", 1))
            tp.send(("decode", 2))
            tp.send(None)
        assert sent == [("decode", 1), ("decode", 2), None] and tp._CH["ch"] is None and tp._CH["failed"]
        assert caplog.text.count("trnrt: no shared-memory channel") == 1
    finally:
        tp.drop_channel()
    assert tp._CH == {"ch": None, "failed": False}


def test_the_channel_is_off_with_trnrt_tp_shm_0(monkeypatch):
    sent = []
    monkeypatch.setattr(tp, "dist", _fake_dist(2, sent))
    monkeypatch.setattr(tp, "SHM", False)
    made = []
    monkeypatch.setattr(tp.ShmChannel, "create", staticmethod(lambda *a, **k: made.append(1)))
    tp.send(("decode", 1))
    tp.send(None)
    assert sent == [("decode", 1), None] and made == [] and tp._CH["ch"] is None


def test_the_default_is_on_and_only_0_turns_it_off():
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = "from trnrt.engine import tp; print(tp.SHM, tp.SHM_BYTES)"
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRNRT_TP_SHM")}
    out = {}
    for v in (None, "0", "1", ""):
        e = dict(env, PYTHONPATH=root, **({} if v is None else {"TRNRT_TP_SHM": v}))
        out[v] = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=root, env=e
        ).stdout.split()
    on, off = ["True", str(64 << 20)], ["False", str(64 << 20)]
    assert out == {None: on, "0": off, "1": on, "": on}


def test_drop_channel_unlinks_and_frees_a_send_waiting_on_a_dead_rank(monkeypatch):
    """An engine that stops without the shutdown call (a rank died): its channel is forgotten and unlinked, and the
    send that waits for the dead rank's acknowledgement raises instead of waiting forever."""
    from multiprocessing import shared_memory

    ch = tp.ShmChannel.create(2, 1 << 14)
    monkeypatch.setitem(tp._CH, "ch", ch)
    name = ch.shm.name
    ch.send(("decode", 1))  # no reader acknowledges it: the next send waits
    err = []

    def waiting():
        try:
            ch.send(("decode", 2))
        except RuntimeError as e:
            err.append(e)

    t = threading.Thread(target=waiting, daemon=True)
    t.start()
    time.sleep(0.2)
    assert t.is_alive()  # waiting for rank 1
    tp.drop_channel()
    t.join(10)
    assert not t.is_alive() and err and "closed" in str(err[0])
    assert tp._CH["ch"] is None
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name, create=False)
    tp.drop_channel()  # idempotent


def test_a_reader_whose_rank_0_is_gone_stops(monkeypatch):
    """A worker left polling the segment of a dead rank 0 would hold its NeuronCore for good (over gloo its broadcast
    fails instead): once the wait has backed off it asks, and stops. A live rank 0 keeps it waiting (the control)."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = [0]

    def never():  # true only long after the wait should have asked (so a wait that never asks ends the test too)
        calls[0] += 1
        return calls[0] > 100000

    with pytest.raises(RuntimeError, match="rank 0 exited"):
        tp.ShmChannel._wait(never, gone=lambda: True)
    assert calls[0] == 20000  # at its first ask, once backed off to 1 ms sleeps
    asked = []

    def gone():
        asked.append(1)
        return False

    n = [0]

    def cond():
        n[0] += 1
        return n[0] > 30000

    tp.ShmChannel._wait(cond, gone=gone)
    assert asked  # asked, answered no, and the wait went on until its condition held
    assert not tp._parent_gone()  # the test process has no parent process record


def _port() -> int:
    for _ in range(50):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
        if tp.port_free(p) and tp.port_free(p + 1):
            return p
    raise RuntimeError("no free port")


def _worker(rank: int, world: int, port: int, out) -> None:
    tp.init_rank(rank, world, port)
    got = []
    try:
        while True:
            m = tp.recv()
            if m is None:
                break
            got.append(m)
        out.put((rank, got, tp._CH["ch"] is None))
    finally:
        tp.dist.destroy_process_group()


@pytest.mark.parametrize("shm", [True, False], ids=["channel", "gloo"])
def test_send_and_recv_over_a_real_process_group(monkeypatch, shm):
    """Three CPU ranks, rank 0 in this process: the calls arrive in order on both workers; with the channel, only the
    hello and the call larger than the 4 KiB segment go over gloo, and the shutdown closes the segment on every rank."""
    import multiprocessing as mp
    from multiprocessing import shared_memory

    world, port = 3, _port()
    monkeypatch.setattr(tp, "SHM", shm)
    monkeypatch.setattr(tp, "SHM_BYTES", 4096)
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, port, out), daemon=True) for r in range(1, world)]
    for p in procs:
        p.start()
    saved = tp.process_state()
    msgs = [("decode", ("decode", i % 4), [np.arange(i + 1)], i) for i in range(20)]
    msgs.insert(7, ("prefill", ("prefill", 4096), [np.arange(4096, dtype=np.int64)], 7))  # past the segment
    try:
        tp.init_rank(0, world, port)
        bcasts, names, errors = [], [], []
        real = tp.dist.broadcast_object_list

        def bcast(objs, src=0, **k):  # rank 0's gloo broadcasts
            bcasts.append(objs[0])
            return real(objs, src=src, **k)

        monkeypatch.setattr(tp.dist, "broadcast_object_list", bcast)

        def rank0():  # on a thread: a send stuck on a dead worker fails the test below instead of hanging it
            try:
                for m in [*msgs, None]:
                    tp.send(m)
                    if not names and tp._CH["ch"] is not None:
                        names.append(tp._CH["ch"].shm.name)
            except BaseException as e:  # noqa: BLE001 - reported below
                errors.append(repr(e))

        t = threading.Thread(target=rank0, daemon=True)
        t.start()
        t.join(90)
        assert not t.is_alive() and not errors, ("rank 0's sends", t.is_alive(), errors)
        got = sorted(out.get(timeout=60) for _ in procs)
        for p in procs:
            p.join(30)
        name = names[0] if names else None
    finally:
        tp.drop_channel()
        if tp.dist.is_initialized():
            tp.dist.destroy_process_group()
        tp.restore_process_state(saved)
        for p in procs:
            if p.is_alive():
                p.kill()
    for rank, calls, closed in got:
        assert closed and len(calls) == len(msgs), (rank, len(calls))
        for a, b in zip(calls, msgs):
            assert a[0] == b[0] and a[1] == b[1] and a[3] == b[3] and np.array_equal(a[2][0], b[2][0])
    assert tp._CH["ch"] is None
    if shm:
        assert name is not None and len(bcasts) == 2 and bcasts[0][0] == tp._SHM_HELLO and bcasts[1][0] == "prefill"
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name, create=False)
    else:
        assert name is None and len(bcasts) == len(msgs) + 1  # every call and the shutdown


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    glm = pytest.importorskip("tests.test_glm5_next")
    path = str(tmp_path_factory.mktemp("tpshm") / "model")
    # tp 2 runs the DSA caches context-parallel: the served form's constraints (tests/test_glm5_next.py
    # test_tp2_matches_tp1)
    glm.build(path, seed=7, layers=4, index_topk=16, hidden_size=512)
    return glm, path


def _gen(model, tp_n):
    from trnrt.engine.request import SamplingParams

    glm, path = model
    eng = glm.engine(path, tp=tp_n, max_prefill_tokens=16, max_num_seqs=2, page_size=8, kv_cache_dtype="fp8")
    try:
        reqs = eng.generate(glm.prompts(4, (19, 37)), SamplingParams(max_new_tokens=6, ignore_eos=True, logprobs=0))
        return [(r.output_ids, [x[0] for x in r.logprobs]) for r in reqs]
    finally:
        eng.close()


@pytest.mark.parametrize(
    "nbytes", [64 << 20, 512], ids=["tp2", "tp2, a 512-byte segment: most calls take the fallback"]
)
def test_engines_match_with_the_channel(model, monkeypatch, nbytes):
    monkeypatch.setattr(tp, "SHM", False)
    want = _gen(model, 2)
    monkeypatch.setattr(tp, "SHM", True)
    monkeypatch.setattr(tp, "SHM_BYTES", nbytes)
    made = []
    real = tp.ShmChannel.create

    def create(*a, **k):
        made.append(1)
        return real(*a, **k)

    monkeypatch.setattr(tp.ShmChannel, "create", staticmethod(create))
    bcasts = []
    real_b = tp.dist.broadcast_object_list

    def bcast(objs, src=0, **k):  # rank 0's broadcasts: the hello, then only the fallback's large calls
        bcasts.append(1)
        return real_b(objs, src=src, **k)

    monkeypatch.setattr(tp.dist, "broadcast_object_list", bcast)
    assert _gen(model, 2) == want
    assert made and tp._CH["ch"] is None  # the channel was used and closed at shutdown
    fallback = len(bcasts) - len(made)  # minus each channel's hello
    assert (fallback > 0) == (nbytes < 4096), (fallback, nbytes)  # the small segment really took the fallback
