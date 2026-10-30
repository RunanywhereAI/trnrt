"""TRNRT_START_LOCK: one engine at a time builds its collectives on a shared host.

Two tensor-parallel engines whose ranks load collective NEFFs at the same moment collide in the runtime's CCOM bootstrap
(`ncclInitGlobalComm failed`, "Failed to build and load collectives resources"). Holding a host-wide lock from engine
start to the end of warmup would also serialize the minutes of weight loading two engines can do side by side.

With TRNRT_START_LOCK=<path> (`python -m trnrt serve` sets it for its engines), rank 0's ModelRunner takes the lock
around each graph's FIRST call only (engine/model_runner.py _exec): before it sends the call to the other ranks, until
its own call returns, which a collective graph does only once every rank has loaded it. Everything else (weights, steady
state) runs unlocked. Unset: nothing happens.

The lock is a directory made with mkdir (atomic) holding an `owner` file "<who> <pid> <step> <UTC>"; it is stale when
that pid is dead, or when the directory has had no owner file for 60 s.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import time

POLL_S = 2.0
NO_OWNER_STALE_S = 60.0


def _owner_pid(path: str) -> int | None:
    try:
        return int(open(os.path.join(path, "owner")).read().split()[1])
    except (OSError, IndexError, ValueError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # alive, another user's
        return True
    return True


def acquire(path: str, who: str, step: str) -> None:
    """Take the lock (blocking). `who` is written as one word (whitespace folded to "-"), since the pid is read from
    the owner line's second field."""
    who = "-".join(str(who).split()) or "trnrt"
    while True:
        try:
            os.mkdir(path)
        except FileExistsError:
            pid = _owner_pid(path)
            if pid is not None and not _alive(pid):
                shutil.rmtree(path, ignore_errors=True)  # stale: its holder is gone
                continue
            if pid is None:
                try:
                    age = time.time() - os.stat(path).st_mtime
                except OSError:
                    continue
                if age > NO_OWNER_STALE_S:
                    shutil.rmtree(path, ignore_errors=True)  # stale: made, never owned
                    continue
            time.sleep(POLL_S)
            continue
        with open(os.path.join(path, "owner"), "w") as f:
            f.write(f"{who} {os.getpid()} {step} {time.strftime('%H:%M:%SZ', time.gmtime())}\n")
        return


def release(path: str) -> None:
    """Drop the lock if this process holds it."""
    if _owner_pid(path) == os.getpid():
        shutil.rmtree(path, ignore_errors=True)


@contextlib.contextmanager
def held(step: str):
    """The lock around one step, when TRNRT_START_LOCK is set; a no-op otherwise."""
    path = os.environ.get("TRNRT_START_LOCK")
    if not path:
        yield
        return
    acquire(path, os.environ.get("TRNRT_START_LOCK_OWNER", "trnrt"), step)
    try:
        yield
    finally:
        release(path)
