"""LLMEngine.close() bounds the tensor-parallel stop (engine.TP_STOP_WAIT_S): a worker that never takes rank 0's stop no
longer holds the process (gloo's broadcast is bounded only by the process group's timeout, the shared-memory channel's
acknowledgement wait by nothing). A healthy stop is unchanged: the same call, nothing logged, the workers joined, not
terminated.

- _bounded_stop on threads: a hung stop returns False after the bound and logs who never answered, a raising one
  returns False at once, a healthy one True, and 0 waits without a bound; unacked_ranks names the ranks that lag the
  shared-memory channel;
- close() in a child process with one real spawned worker (the process tests): a hung stop is bounded and the worker
  terminated, a raising one too, a healthy one joins its worker, and without the bound close() hangs (the control)."""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest

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
