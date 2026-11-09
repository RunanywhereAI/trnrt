"""The production log: each finished request's record and the engine's gauges, on disk, when TRNRT_PROD_DIR names a
directory (host only; unset, the default, nothing is written). Each engine writes under a directory of its own:

    <TRNRT_PROD_DIR>/<engine>/<YYYY-MM-DD, UTC>/requests.jsonl   one line per finished request (serve/stats.py's record)
    <TRNRT_PROD_DIR>/<engine>/<YYYY-MM-DD, UTC>/gauges.jsonl     one line per 10 s: running, waiting, KV use, output
                                                                 tok/s, RPM, TPM

A record holds counts and times only (token counts, timings, the finish reason, the request id), never a prompt or an
output. Each file rotates at the UTC day and at ROTATE_BYTES to <name>.<n>.jsonl, and every rotated file (and any left
uncompressed by an earlier process on an earlier day) is gzipped in the background. At start, today's records and
gauges are read back into the live statistics, so GET /v1/stats's "today" window spans restarts.

The serving threads only enqueue: put() is a put_nowait on a bounded queue (QUEUE), and a full queue drops the record
and counts it. One writer thread formats and writes, so neither the engine's step loop nor the event loop waits on the
disk. A write error (a full disk) is counted, logged at most once a minute, and the record dropped: serving goes on.
"""

from __future__ import annotations

import calendar
import contextlib
import gzip
import json
import logging
import os
import queue
import shutil
import threading
import time
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from .stats import Stats

logger = logging.getLogger(__name__)

ENV = "TRNRT_PROD_DIR"
ROTATE_BYTES = 1 << 30
QUEUE = 8192
NAMES = {"requests": "requests.jsonl", "gauges": "gauges.jsonl"}
GAUGES_RELOAD_S = 1800  # the series covers 30 minutes: older gauges are not read back


def utc_day(t: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(t))


def read_jsonl(path: str) -> list[dict]:
    """A .jsonl or .jsonl.gz file's records; a torn line (a write cut short) is skipped."""
    out = []
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        pass
    return out


def day_files(day_dir: str, kind: str) -> list[str]:
    """A day directory's files of one kind, oldest first: the rotated segments <stem>.<n>.jsonl[.gz], then the
    current one."""
    stem = NAMES[kind][: -len(".jsonl")]
    try:
        names = os.listdir(day_dir)
    except OSError:
        return []
    segs: list[tuple[float, str]] = []
    for n in names:
        base = n.removesuffix(".gz")
        if base == NAMES[kind]:
            segs.append((float("inf"), n))
        elif base.startswith(stem + ".") and base.endswith(".jsonl"):
            mid = base[len(stem) + 1 : -len(".jsonl")]
            if mid.isdigit():
                segs.append((int(mid), n))
    return [os.path.join(day_dir, n) for _, n in sorted(segs)]


def gzip_file(path: str) -> None:
    try:
        with open(path, "rb") as f, gzip.open(path + ".gz", "wb") as g:
            shutil.copyfileobj(f, g, 1 << 20)
        os.remove(path)
    except OSError as e:
        logger.warning("production log: could not gzip %s: %s", path, e)


class ProdLog:
    def __init__(
        self,
        root: str,
        stats: Stats | None = None,
        rotate_bytes: int = ROTATE_BYTES,
        queue_size: int = QUEUE,
        clock: Callable[[], float] = time.time,
    ):
        self.root = os.path.expanduser(root)
        self.rotate_bytes = rotate_bytes
        self.clock = clock
        self.q: queue.Queue = queue.Queue(queue_size)
        self.dropped = dict.fromkeys(NAMES, 0)
        self.written = dict.fromkeys(NAMES, 0)
        self.write_errors = 0
        self.last_error: str | None = None
        self._last_log = float("-inf")
        self._open: dict[str, tuple[str, str, IO[bytes], int]] = {}  # kind -> (day, path, file, size)
        self._stop = object()
        self._gzips: list[threading.Thread] = []
        self.thread = threading.Thread(target=self._run, args=(stats,), name="trnrt-prodlog", daemon=True)
        self.thread.start()

    @classmethod
    def from_env(cls, name: str, stats: Stats | None = None) -> ProdLog | None:
        """The production log of the engine `name` under TRNRT_PROD_DIR, or None when it is unset."""
        root = os.environ.get(ENV)
        if not root:
            return None
        return cls(os.path.join(root, name), stats=stats)

    # -- any thread ---------------------------------------------------------------------------------------------------

    def put(self, kind: str, rec: dict) -> bool:
        """Enqueue one record; never blocks. False when it was dropped (the queue is full)."""
        try:
            self.q.put_nowait((kind, rec))
        except queue.Full:
            self.dropped[kind] += 1
            return False
        return True

    def close(self, timeout: float = 10.0) -> None:
        """Write what is queued and close the files (at exit)."""
        with contextlib.suppress(queue.Full):
            self.q.put((self._stop, None), timeout=timeout)
        self.thread.join(timeout)
        for t in self._gzips:
            t.join(timeout)

    def status(self) -> dict:
        """GET /v1/stats's view of it: where it writes, today's files, and what was written, dropped and failed."""
        day = utc_day(self.clock())
        d = os.path.join(self.root, day)
        files = {}
        try:
            for n in sorted(os.listdir(d)):
                files[n] = os.path.getsize(os.path.join(d, n))
        except OSError:
            pass
        return {
            "on": True,
            "dir": self.root,
            "day": day,
            "files": files,
            "written": dict(self.written),
            "dropped": dict(self.dropped),
            "write_errors": self.write_errors,
            "last_error": self.last_error,
            "queued": self.q.qsize(),
        }

    # -- the writer thread --------------------------------------------------------------------------------------------

    def _run(self, stats: Stats | None) -> None:
        try:
            os.makedirs(self.root, exist_ok=True)
        except OSError as e:
            self._error(f"cannot create {self.root}: {e}")
        threading.Thread(target=self._gzip_old_days, name="trnrt-prodlog-gzip", daemon=True).start()
        if stats is not None:
            try:
                self.load_today(stats)
            except Exception:  # a bad file must not stop the recording
                logger.exception("production log: reading today's records back failed")
        while True:
            batch = [self.q.get()]
            while len(batch) < 512:
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            stop = False
            touched = set()
            for kind, rec in batch:
                if kind is self._stop:
                    stop = True
                    continue
                try:
                    line = (json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                    if self._roll_due(kind, len(line)):
                        self._roll(kind)
                    self._write(kind, line)
                    touched.add(kind)
                    self.written[kind] += 1
                except OSError as e:
                    self.dropped[kind] += 1
                    self._error(f"writing {kind}: {e}")
                except Exception as e:  # noqa: BLE001  one bad record must not end the recording
                    self.dropped[kind] += 1
                    self._error(f"formatting {kind}: {e!r}")
            for kind in touched:
                try:
                    self._open[kind][2].flush()
                except (OSError, KeyError) as e:
                    self._error(f"flushing {kind}: {e}")
            if stop:
                for kind in list(self._open):
                    self._close(kind, gz=False)
                return

    def _error(self, msg: str) -> None:
        self.write_errors += 1
        self.last_error = msg
        now = time.monotonic()
        if now - self._last_log >= 60:
            self._last_log = now
            logger.warning("production log: %s (records dropped; serving continues)", msg)

    def _roll_due(self, kind: str, n: int) -> bool:
        cur = self._open.get(kind)
        return cur is not None and (cur[0] != utc_day(self.clock()) or cur[3] + n > self.rotate_bytes)

    def _roll(self, kind: str) -> None:
        """Close the kind's file (a new UTC day, or the size limit) and gzip it."""
        same_day = self._open[kind][0] == utc_day(self.clock())
        self._close(kind, gz=True, rotate=same_day)

    def _write(self, kind: str, data: bytes) -> None:
        cur = self._open.get(kind)
        if cur is None:
            day = utc_day(self.clock())
            d = os.path.join(self.root, day)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, NAMES[kind])
            f = open(path, "ab")  # noqa: SIM115  held open across writes, closed by _close
            cur = (day, path, f, f.tell())
        cur[2].write(data)
        self._open[kind] = (cur[0], cur[1], cur[2], cur[3] + len(data))

    def _close(self, kind: str, gz: bool, rotate: bool = False) -> None:
        _, path, f, _ = self._open.pop(kind)
        try:
            f.close()
        except OSError as e:
            self._error(f"closing {path}: {e}")
        if not gz:
            return
        if rotate:  # the same day, past the size limit: the current file becomes the next numbered segment
            stem = path[: -len(".jsonl")]
            n = 1
            while os.path.exists(f"{stem}.{n}.jsonl") or os.path.exists(f"{stem}.{n}.jsonl.gz"):
                n += 1
            dest = f"{stem}.{n}.jsonl"
            os.rename(path, dest)
            path = dest
        t = threading.Thread(target=gzip_file, args=(path,), name="trnrt-prodlog-gzip", daemon=True)
        t.start()
        self._gzips.append(t)

    def _gzip_old_days(self) -> None:
        today = utc_day(self.clock())
        try:
            days = [d for d in os.listdir(self.root) if d != today and os.path.isdir(os.path.join(self.root, d))]
        except OSError:
            return
        for d in days:
            for n in os.listdir(os.path.join(self.root, d)):
                if n.endswith(".jsonl"):
                    gzip_file(os.path.join(self.root, d, n))

    def load_today(self, stats: Stats) -> None:
        """Today's records and gauges (earlier processes') into the live statistics: the "today" window and the
        series."""
        now = self.clock()
        day = utc_day(now)
        d = os.path.join(self.root, day)
        recs = [r for p in day_files(d, "requests") for r in read_jsonl(p) if isinstance(r.get("done"), (int, float))]
        gauges = [
            g
            for p in day_files(d, "gauges")
            for g in read_jsonl(p)
            if isinstance(g.get("t"), (int, float)) and g["t"] >= now - GAUGES_RELOAD_S
        ]
        stats.merge_day(recs, gauges, calendar.timegm(time.strptime(day, "%Y-%m-%d")))  # from UTC midnight
