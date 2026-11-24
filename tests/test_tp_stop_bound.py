"""LLMEngine.close() bounds the tensor-parallel stop (engine.TP_STOP_WAIT_S): a worker that never takes rank 0's stop no
longer holds the process (gloo's broadcast is bounded only by the process group's timeout, the shared-memory channel's
acknowledgement wait by nothing). A healthy stop is unchanged: the same call, nothing logged, the workers joined, not
terminated.

- _bounded_stop on threads: a hung stop returns False after the bound and logs who never answered, a raising one
  returns False at once, a healthy one True, and 0 waits without a bound; unacked_ranks names the ranks that lag the
  shared-memory channel and waits for a close in progress, and a diagnostic that raises still lets close() go on;
- close() in a child process with one real spawned worker (the process tests): a hung stop is bounded and the worker
  terminated, a raising one too, a healthy one joins its worker, an engine left open is closed by the exit hook
  (its stop thread starts at interpreter exit), and without the bound close() hangs (the control); over gloo, the
  group close() destroys under a stuck stop broadcast does not hold close() and the next group of the process works;
- a start that fails after its workers started (stand-in workers) stops them and gives rank 0 its state back."""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
import textwrap
import threading
import time
import types

import pytest
import torch

from trnrt.engine import engine as E
from trnrt.engine import tp as TP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_a_hung_stop_returns_after_the_bound_and_logs(caplog):
    release = threading.Event()
    with caplog.at_level(logging.WARNING, logger="trnrt"):
        t0 = time.monotonic()
        assert E._bounded_stop(lambda msg: release.wait(), timeout=0.2) is False
        dt = time.monotonic() - t0
    release.set()
    assert 0.15 < dt < 5
    assert "the tensor-parallel stop no answer in 0.2 s" in caplog.text and "ranks not acknowledged" in caplog.text


def test_a_raising_stop_returns_false(caplog):
    def send(msg):
        raise ConnectionError("planted: a peer reset the broadcast")

    with caplog.at_level(logging.WARNING, logger="trnrt"):
        assert E._bounded_stop(send, timeout=10) is False
    assert "raised ConnectionError" in caplog.text


def test_a_healthy_stop_is_unchanged(caplog):
    got = []
    with caplog.at_level(logging.WARNING, logger="trnrt"):
        assert E._bounded_stop(got.append, timeout=10) is True
    assert got == [None] and "engine close" not in caplog.text


def test_a_failing_diagnostic_does_not_stop_close_from_stopping_the_workers(monkeypatch, caplog):
    def broken():
        raise TypeError("planted: the channel went away under the read")

    def send(msg):
        raise ConnectionError("planted: a peer reset the broadcast")

    monkeypatch.setattr(TP, "unacked_ranks", broken)
    with caplog.at_level(logging.WARNING, logger="trnrt"):
        assert E._bounded_stop(send, timeout=10) is False
    assert "ranks not acknowledged: unknown (TypeError(" in caplog.text


def test_no_bound_calls_the_stop_on_this_thread():
    me = []
    assert E._bounded_stop(lambda msg: me.append(threading.current_thread()), timeout=0) is True
    assert me == [threading.current_thread()]


PROG = textwrap.dedent(
    """
    import multiprocessing as mp, os, sys, time, types
    sys.path.insert(0, {root!r})

    def hang(path):
        open(path, "w").write(str(os.getpid()))
        while True:
            time.sleep(1)

    def healthy(path, ev):
        open(path, "w").write(str(os.getpid()))
        ev.wait()          # the stop (rank 0's broadcast, here an Event) arrives
        sys.exit(0)

    if __name__ == "__main__":
        mode, pid_file, wait = sys.argv[1], sys.argv[2], float(sys.argv[3])
        import atexit
        workers = []

        def report():
            if workers:
                print(f"AT EXIT exitcode {{workers[0].exitcode}}", flush=True)

        atexit.register(report)  # before the engine module's own exit hook, so after it at exit (last in, first out)
        from trnrt import log
        from trnrt.engine import engine as E
        log.configure()
        E.TP_STOP_WAIT_S = wait
        ctx = mp.get_context("spawn")
        eng = E.LLMEngine.__new__(E.LLMEngine)
        if mode in ("healthy", "exit"):
            ev = ctx.Event()
            w = ctx.Process(target=healthy, args=(pid_file, ev), daemon=True)
            send = lambda msg: ev.set()
        else:
            w = ctx.Process(target=hang, args=(pid_file,), daemon=True)
            def send(msg):
                if mode == "raise":
                    raise ConnectionError("planted: a peer reset the broadcast")
                while True:   # the stuck stop: never returns
                    time.sleep(1)
        w.start()
        workers.append(w)
        while not os.path.exists(pid_file):
            time.sleep(0.05)
        eng.__dict__.update(_workers=[w], runner=types.SimpleNamespace(tp_send=send), _tp_saved=None)
        E._OPEN.add(eng)
        if mode == "exit":  # left open: the engine module's exit hook closes it
            sys.exit(0)
        t0 = time.time()
        eng.close()
        print(f"CLOSED in {{time.time() - t0:.1f}} s exitcode {{w.exitcode}}", flush=True)
    """
).format(root=ROOT)


def _run(tmp_path, mode: str, wait: str, timeout: float):
    script = tmp_path / "prog.py"
    script.write_text(PROG)
    pid_file = tmp_path / f"w-{mode}-{wait}.pid"
    env = dict(os.environ, PYTHONPATH=ROOT)
    t0 = time.time()
    try:
        p = subprocess.run(
            [sys.executable, str(script), mode, str(pid_file), wait],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        out = (p.returncode, p.stdout + p.stderr, time.time() - t0)
    except subprocess.TimeoutExpired:
        out = (None, "TIMEOUT", time.time() - t0)
    if pid_file.exists():  # never leave the test's worker behind
        with contextlib.suppress(ProcessLookupError, ValueError):
            os.kill(int(pid_file.read_text()), 9)
    return out


@pytest.mark.parametrize(
    ("mode", "wait", "want"),
    [
        ("hang", "2", "the tensor-parallel stop no answer in 2 s"),
        ("raise", "120", "stop raised ConnectionError"),
    ],
)
def test_process_a_hung_or_raising_stop_is_bounded_logged_and_terminated(tmp_path, mode, wait, want):
    rc, out, dt = _run(tmp_path, mode, wait, 60)
    assert rc == 0, out[-800:]
    assert want in out, out[-800:]
    assert "CLOSED in" in out and "exitcode -15" in out, out[-800:]  # terminated (SIGTERM), not joined
    assert dt < 30, dt


def test_process_a_healthy_stop_joins_its_worker(tmp_path):
    rc, out, dt = _run(tmp_path, "healthy", "120", 60)
    assert rc == 0, out[-800:]
    assert "engine close:" not in out, out[-800:]  # nothing logged
    assert "exitcode 0" in out, out[-800:]  # the worker exited on its own, joined, not terminated
    assert dt < 30, dt


def test_process_an_engine_left_open_releases_its_worker_through_the_exit_hook(tmp_path):
    """engine._close_open closes an engine still open at interpreter exit: its bounded stop starts its thread there, and
    the worker takes the stop and exits 0 (joined, not terminated)."""
    rc, out, dt = _run(tmp_path, "exit", "120", 60)
    assert rc == 0, out[-800:]
    assert "engine close:" not in out and "AT EXIT exitcode 0" in out, out[-800:]
    assert dt < 30, dt


def test_process_the_control_without_a_bound_hangs(tmp_path):
    """TP_STOP_WAIT_S = 0 is the unbounded stop: the same hung worker holds close()."""
    rc, out, _ = _run(tmp_path, "hang", "0", 12)
    assert rc is None and out == "TIMEOUT", (rc, out[-300:])


GLOO = textwrap.dedent(
    """
    import multiprocessing as mp, os, sys, time, types
    sys.path.insert(0, {root!r})

    def stuck(port):
        from trnrt.engine import tp
        tp.init_rank(1, 2, port)
        while True:  # never takes rank 0's stop
            time.sleep(1)

    def takes(port):
        import torch.distributed as dist
        from trnrt.engine import tp
        tp.init_rank(1, 2, port)
        for _ in range(2):
            box = [None]
            dist.broadcast_object_list(box, src=0)
            print("WORKER GOT", box[0], flush=True)
        dist.destroy_process_group()

    if __name__ == "__main__":
        os.environ["TRNRT_TP_SHM"] = "0"  # the calls over gloo: the stop is a broadcast on its thread
        import torch.distributed as dist
        from trnrt import log
        from trnrt.engine import engine as E
        from trnrt.engine import tp
        log.configure()
        E.TP_STOP_WAIT_S = 2.0
        ctx = mp.get_context("spawn")
        port = tp.free_port()
        w = ctx.Process(target=stuck, args=(port,), daemon=True)
        w.start()
        eng = E.LLMEngine.__new__(E.LLMEngine)
        eng.__dict__.update(_workers=[w], runner=types.SimpleNamespace(tp_send=tp.send), _tp_saved=tp.process_state())
        tp.init_rank(0, 2, port)
        E._OPEN.add(eng)
        t0 = time.time()
        eng.close()  # past the bound: the worker terminated, the group destroyed under the stop's broadcast
        print(f"CLOSED in {{time.time() - t0:.1f}} s exitcode {{w.exitcode}} group {{dist.is_initialized()}}", flush=True)
        port = tp.free_port()
        w2 = ctx.Process(target=takes, args=(port,), daemon=True)
        w2.start()
        tp.init_rank(0, 2, port)  # the next engine of this process
        dist.broadcast_object_list(["hello"], src=0)
        dist.broadcast_object_list(["again"], src=0)
        w2.join(30)
        dist.destroy_process_group()
        print(f"NEW GROUP exitcode {{w2.exitcode}}", flush=True)
    """
).format(root=ROOT)


def test_process_a_group_destroyed_under_a_stuck_stop_broadcast_leaves_the_next_group_working(tmp_path):
    """Over gloo, close() past the bound terminates the workers and destroys the process group while the stop's thread is
    still inside its broadcast (it does not return when its peer dies, so waiting for it would only hold close()). The
    destroy must not wait for that thread, and the next engine's group in the same process must work."""
    script = tmp_path / "gloo.py"
    script.write_text(GLOO)
    p = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=90,
        env=dict(os.environ, PYTHONPATH=ROOT),
        check=False,
    )
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-1200:]
    assert "the tensor-parallel stop no answer in 2 s" in out, out[-1200:]
    assert "exitcode -15 group False" in out, out[-1200:]  # terminated, and the group is gone
    assert out.count("WORKER GOT") == 2 and "NEW GROUP exitcode 0" in out, out[-1200:]


def test_unacked_ranks_names_the_lagging_ranks(monkeypatch):
    monkeypatch.setitem(TP._CH, "ch", None)
    assert "gloo" in str(TP.unacked_ranks())  # no channel open
    ch = TP.ShmChannel.create(4, 1 << 16)
    try:
        ch.seen = 5
        ch.acks[1], ch.acks[2], ch.acks[3] = 5, 4, 5
        monkeypatch.setitem(TP._CH, "ch", ch)
        assert TP.unacked_ranks() == [2]
    finally:
        monkeypatch.setitem(TP._CH, "ch", None)
        ch.close()


def test_unacked_ranks_waits_for_a_close_in_progress_and_never_reads_the_unmapped_channel(monkeypatch):
    """A stop that finishes just after the bound closes the channel on its own thread while close() logs who lagged:
    the read waits for that close (the channel's lock), then says the channel closed. Unlocked, it read the
    acknowledgements while the segment was being unmapped (numpy views do not keep the mapping)."""
    ch = TP.ShmChannel.create(4, 1 << 16)
    monkeypatch.setitem(TP._CH, "ch", ch)
    inside, release = threading.Event(), threading.Event()
    unmap = ch.shm.close

    def slow_unmap():
        inside.set()
        release.wait(10)
        unmap()

    monkeypatch.setattr(ch.shm, "close", slow_unmap)
    closer = threading.Thread(target=ch.close)
    closer.start()
    try:
        assert inside.wait(10)
        got: list = []
        reader = threading.Thread(target=lambda: got.append(TP.unacked_ranks()))
        reader.start()
        reader.join(0.3)
        assert reader.is_alive() and not got, got  # held behind the close in progress
    finally:
        release.set()
        closer.join(10)
    reader.join(10)
    assert got == ["unknown (the channel closed meanwhile)"]
    monkeypatch.setitem(TP._CH, "ch", None)


class _Proc:
    """A worker process that runs until it is terminated (multiprocessing.Process's surface that LLMEngine uses). It
    sets its ready event (worker_main's last argument) when it starts, as a worker past its own start does."""

    def __init__(self, **kw):
        self.pid, self.exitcode, self.terminated = None, None, False
        self.args = kw.get("args", ())

    def start(self):
        self.pid = 1 << 22
        self.args[-1].set()

    def is_alive(self):
        return self.pid is not None and self.exitcode is None

    def join(self, timeout=None):
        pass

    def terminate(self):
        self.terminated, self.exitcode = True, -15

    def kill(self):
        self.exitcode = -9


def test_a_start_that_fails_after_its_workers_started_stops_them_and_gives_rank_0_its_state_back(monkeypatch):
    """Past build_shard (here the save-once /dev/shm refusal) the workers run and the process group is up, and the
    caller gets no engine to close: the constructor stops the workers (terminated: no broadcast to them), gives this
    process its state back and closes rank 0's arena, then raises the start's own error."""
    import multiprocessing

    from trnrt.config import EngineConfig

    procs: list[_Proc] = []
    events: list[str] = []

    def proc(**kw):
        procs.append(_Proc(**kw))
        return procs[-1]

    def refuse(tag, pages):
        raise ValueError("planted: the save-once host arenas need more of /dev/shm than is free")

    mcfg = types.SimpleNamespace(eos_token_ids=(0,), vocab_size=8)
    runner = types.SimpleNamespace(
        device=torch.device("cpu"),
        tp_send=None,
        watchdog=None,
        state=types.SimpleNamespace(free_ckpt=lambda *a: None, bytes_per_row=lambda: 1024),
        kv_page_bytes=lambda: 4096,
        so_possible=lambda: True,
        kv_arena_init=refuse,
        kv_arena_close=lambda: events.append("arena closed"),
        release_slot=lambda req: None,
    )
    monkeypatch.setattr(
        multiprocessing, "get_context", lambda kind: types.SimpleNamespace(Process=proc, Event=threading.Event)
    )
    monkeypatch.setattr(TP, "free_port", lambda core_base=None: 0)
    monkeypatch.setattr(TP, "init_rank", lambda rank, world, port: events.append("process group up"))
    monkeypatch.setattr(TP, "restore_process_state", lambda saved: events.append("process restored"))
    monkeypatch.setattr(E, "resolve_model_path", lambda path, revision=None: path)
    monkeypatch.setattr(E.ModelConfig, "from_pretrained", classmethod(lambda cls, path: mcfg))
    monkeypatch.setattr(E, "resolve_attention_tp", lambda m, c: 2)
    monkeypatch.setattr(E, "pool_pages", lambda c, m: 16)
    monkeypatch.setattr(E, "build_shard", lambda c, path, n, rank: (types.SimpleNamespace(cp_rows=2), runner))

    def no_tokenizer(path):
        raise OSError("no tokenizer here")

    tok = types.SimpleNamespace(from_pretrained=no_tokenizer)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=tok))
    cfg = EngineConfig(
        model_path="no-model", device="cpu", tp=2, host_cache_gb=0.01, host_cache_batch=True, host_cache_save_once=True
    )
    n_open = len(E._OPEN)
    with pytest.raises(ValueError, match="planted"):
        E.LLMEngine(cfg)
    assert len(procs) == 1 and procs[0].terminated
    assert events == ["process group up", "process restored", "arena closed"]
    assert len(E._OPEN) == n_open and runner.tp_send is None  # nothing left for the exit hook to close


class _DeadProc(_Proc):
    """A worker that exits during its own start (its NUMA pin raised), before it is ready to join the process group."""

    def start(self):
        self.pid, self.exitcode = 1 << 22, 1


def test_a_worker_that_dies_before_joining_fails_the_start_at_once(monkeypatch):
    """A worker whose start fails (numa_pin raising NumaPinError, say) never joins the process group. Rank 0 used to
    wait for it in the rendezvous for the gloo timeout (4 h); it now sees the worker gone before it joins, raises naming
    the rank, stops the other workers and gives this process its state back."""
    import multiprocessing

    from trnrt.config import EngineConfig

    procs: list[_Proc] = []
    events: list[str] = []

    def proc(**kw):
        procs.append((_DeadProc if kw["args"][0] == 2 else _Proc)(**kw))
        return procs[-1]

    mcfg = types.SimpleNamespace(eos_token_ids=(0,), vocab_size=8)
    monkeypatch.setattr(
        multiprocessing, "get_context", lambda kind: types.SimpleNamespace(Process=proc, Event=threading.Event)
    )
    monkeypatch.setattr(TP, "free_port", lambda core_base=None: 0)
    monkeypatch.setattr(TP, "init_rank", lambda rank, world, port: events.append("process group up"))
    monkeypatch.setattr(TP, "restore_process_state", lambda saved: events.append("process restored"))
    monkeypatch.setattr(E, "resolve_model_path", lambda path, revision=None: path)
    monkeypatch.setattr(E.ModelConfig, "from_pretrained", classmethod(lambda cls, path: mcfg))
    monkeypatch.setattr(E, "resolve_attention_tp", lambda m, c: 4)
    monkeypatch.setattr(E, "pool_pages", lambda c, m: 16)
    cfg = EngineConfig(model_path="no-model", device="cpu", tp=4)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match=r"rank\(s\) 2 \(exit code 1\) exited before joining the process group"):
        E.LLMEngine(cfg)
    assert time.monotonic() - t0 < 10
    assert events == ["process restored"]  # rank 0 never entered the rendezvous
    assert [p.terminated for p in procs] == [True, False, True]  # the live workers stopped; the dead one is gone


def _exit_at_once(code: int) -> None:
    os._exit(code)


def _ready_then_wait(ev) -> None:
    ev.set()
    time.sleep(30)


def test_wait_joining_with_real_workers():
    """tp.wait_joining over spawned processes: it returns once every worker is ready, and raises within a poll or two
    when one exits first, naming its rank and exit code."""
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    ready = [ctx.Event(), ctx.Event()]
    ws = [ctx.Process(target=_ready_then_wait, args=(ready[0],), daemon=True)]
    ws.append(ctx.Process(target=_exit_at_once, args=(3,), daemon=True))
    for w in ws:
        w.start()
    try:
        t0 = time.monotonic()
        with pytest.raises(RuntimeError, match=r"rank\(s\) 2 \(exit code 3\) exited before joining"):
            TP.wait_joining(ws, ready)
        assert time.monotonic() - t0 < 30
        ready[1].set()  # both ready: it returns
        TP.wait_joining(ws, ready)
    finally:
        for w in ws:
            w.kill()
            w.join(10)
