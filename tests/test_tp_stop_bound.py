"""LLMEngine.close() bounds the tensor-parallel stop (engine.TP_STOP_WAIT_S): a worker that never takes rank 0's stop no
longer holds the process (gloo's broadcast is bounded only by the process group's timeout, the shared-memory channel's
acknowledgement wait by nothing). A healthy stop is unchanged: the same call, nothing logged, the workers joined, not
terminated.

- _bounded_stop on threads: a hung stop returns False after the bound and logs who never answered, a raising one
  returns False at once, a healthy one True, and 0 waits without a bound; unacked_ranks names the ranks that lag the
  shared-memory channel and waits for a close in progress, and a diagnostic that raises still lets close() go on;
- close() in a child process with one real spawned worker (the process tests): a hung stop is bounded and the worker
  terminated, a raising one too, a healthy one joins its worker, and without the bound close() hangs (the control);
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
        from trnrt import log
        from trnrt.engine import engine as E
        log.configure()
        E.TP_STOP_WAIT_S = wait
        ctx = mp.get_context("spawn")
        eng = E.LLMEngine.__new__(E.LLMEngine)
        if mode == "healthy":
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
        while not os.path.exists(pid_file):
            time.sleep(0.05)
        eng.__dict__.update(_workers=[w], runner=types.SimpleNamespace(tp_send=send), _tp_saved=None)
        E._OPEN.add(eng)
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


def test_process_the_control_without_a_bound_hangs(tmp_path):
    """TP_STOP_WAIT_S = 0 is the unbounded stop: the same hung worker holds close()."""
    rc, out, _ = _run(tmp_path, "hang", "0", 12)
    assert rc is None and out == "TIMEOUT", (rc, out[-300:])


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
    """A worker process that runs until it is terminated (multiprocessing.Process's surface that LLMEngine uses)."""

    def __init__(self, **kw):
        self.pid, self.exitcode, self.terminated = None, None, False

    def start(self):
        self.pid = 1 << 22

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

    mcfg = types.SimpleNamespace(eos_token_ids=(0,))
    runner = types.SimpleNamespace(
        device=torch.device("cpu"),
        tp_send=None,
        watchdog=None,
        state=types.SimpleNamespace(free_ckpt=lambda *a: None, bytes_per_row=lambda: 1024),
        kv_page_bytes=lambda: 4096,
        so_possible=lambda: True,
        kv_arena_init=refuse,
        kv_arena_close=lambda: events.append("arena closed"),
    )
    monkeypatch.setattr(multiprocessing, "get_context", lambda kind: types.SimpleNamespace(Process=proc))
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
