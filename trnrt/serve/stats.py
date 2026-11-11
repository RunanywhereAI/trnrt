"""Live serving statistics of one engine: GET /v1/stats, the page GET /dashboard draws (serve/dashboard.py), and the
records the production log keeps (serve/prodlog.py). Host only: nothing here touches the engine or its graphs.

Cost on the engine thread, O(1) each, under an uncontended lock:
- per finished request, record(): one dict built from the Request, appended to a bounded ring (RING);
- per loop iteration (a step, or an idle wait), tick(): the tokens the step handed out and the running and waiting
  counts, added into the current BUCKET_S bucket.
Everything else (windows, percentiles, the time series) is computed by snapshot() off the event loop, from a copy taken
under the lock, and cached for SNAPSHOT_TTL_S so that several viewers share one computation.

Times are wall-clock seconds (time.time()): the engine's monotonic Request times plus one offset taken at start, so
records reloaded from the production log of an earlier process share the timeline. All are measured inside the server,
from when the engine received the request (after the chat template and tokenization) to its last token, without the
client's network round trip.

Per request: TTFT = first token - arrival; end-to-end = done - arrival; decode tok/s (per stream) = (output - 1) /
(done - first token), and the mean inter-token latency its inverse (requests with at least 2 output tokens). Rates per
window count requests by when they finished: RPM = requests / minutes; TPM input, cached, output, and total (input +
output). The aggregate output tok/s counts tokens as the steps hand them out (BUCKET_S buckets), so it stays current
while long requests are still running.
"""

from __future__ import annotations

import collections
import math
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

RING = 10_000  # finished requests kept for the windows and percentiles (~2.5 MB of dicts)
BUCKET_S = 10
SERIES_BUCKETS = 180  # the time series: 30 min of 10-s points
WINDOWS = (("1m", 60), ("5m", 300), ("15m", 900))
RECENT = 50  # the newest finished requests listed
SNAPSHOT_TTL_S = 1.0
OK_FINISH = ("stop", "length", "tool_calls")
RECENT_KEYS = (
    "rid",
    "arrival",
    "first",
    "done",
    "prompt",
    "cached",
    "output",
    "finish",
    "stream",
    "ttft",
    "e2e",
    "decode_tps",
    "itl",
    "running_at_arrival",
    "waiting_at_arrival",
    "spec_proposed",
    "spec_accepted",
)


def pct(xs: list[float], q: float) -> float | None:
    """The q-quantile (0 .. 1) of xs, interpolating linearly between order statistics (numpy's default)."""
    if not xs:
        return None
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summary(xs: list[float], low: bool = False) -> dict:
    """n, mean, p50, p90 and p99 (and p10 first with low=True: a rate's slowest streams)."""
    out: dict = {"n": len(xs), "mean": (sum(xs) / len(xs)) if xs else None}
    for q in ((0.1,) if low else ()) + (0.5, 0.9, 0.99):
        out[f"p{round(q * 100)}"] = pct(xs, q)
    return out


def derive(rec: dict) -> dict:
    """A record's derived times (seconds) and rate, in place: ttft, e2e, decode_tps, itl (None when undefined)."""
    first, done, arr, n = rec.get("first"), rec.get("done"), rec.get("arrival"), rec.get("output", 0)
    rec["ttft"] = (first - arr) if first is not None and arr is not None else None
    rec["e2e"] = (done - arr) if done is not None and arr is not None else None
    span = (done - first) if first is not None and done is not None else None
    rec["decode_tps"] = (n - 1) / span if span and span > 0 and n > 1 else None
    rec["itl"] = span / (n - 1) if span is not None and n > 1 else None
    return rec


def _kind(finish: str) -> str:
    return "ok" if finish in OK_FINISH else "aborted" if finish == "abort" else "failed"


class Stats:
    def __init__(
        self, ring: int = RING, clock: Callable[[], float] = time.time, mono: Callable[[], float] = time.monotonic
    ):
        self.lock = threading.Lock()
        self.clock = clock
        self.offset = clock() - mono()  # wall = monotonic + offset
        self.started = clock()
        self.records: collections.deque = collections.deque(maxlen=ring)
        self.counts = {"ok": 0, "aborted": 0, "failed": 0}
        self.recorded = 0  # finished requests recorded since start (the ring keeps only the newest RING)
        self.failed_kinds: dict[str, int] = {}
        self.tokens = {"prompt": 0, "cached": 0, "output": 0}
        # Earlier processes' records of today (the production log's reload): totals and the first record's time.
        self.day: dict | None = None
        # BUCKET_S buckets in a ring of SERIES_BUCKETS slots: [bucket id, ticks, running sum, running max, waiting
        # sum, waiting max, tokens handed out].
        self.buckets: list[list | None] = [None] * SERIES_BUCKETS
        self.cur: list | None = None
        self.past: dict[int, dict] = {}  # bucket id -> a gauges record reloaded from today's log (before this start)
        self.on_bucket: Callable[[dict], object] | None = (
            None  # (the closed bucket's gauges record): the production log
        )
        self.kv: Callable[[], tuple[int, int]] | None = None  # () -> (pages used, pages total), for the gauges records
        self._snap: tuple[float, dict] | None = None
        self._snap_lock = threading.Lock()  # snapshot() runs in worker threads: one computes, the others reuse it
        self.handed = 0  # output tokens handed out since this process started

    # -- the engine thread --------------------------------------------------------------------------------------------

    def record(self, req, stream: bool, at_arrival: tuple[int, int] | None, finish: str | None = None) -> dict:
        """A finished request's record: appended to the ring, counted, and returned for the production log."""
        off = self.offset
        rec = {
            "rid": req.rid,
            "arrival": req.arrival_time + off,
            "first": None if req.first_token_time is None else req.first_token_time + off,
            "done": (req.finish_time if req.finish_time is not None else time.monotonic()) + off,
            "prompt": req.num_prompt,
            "cached": max(req.num_cached_tokens, 0),
            "output": len(req.output_ids),
            "finish": finish or req.finish_reason or "unknown",
            "stream": bool(stream),
            "running_at_arrival": None if at_arrival is None else at_arrival[0],
            "waiting_at_arrival": None if at_arrival is None else at_arrival[1],
        }
        if req.spec_proposed:
            rec["spec_proposed"], rec["spec_accepted"] = req.spec_proposed, req.spec_accepted
        derive(rec)
        kind = _kind(rec["finish"])
        with self.lock:
            self.records.append(rec)
            self.recorded += 1
            self.counts[kind] += 1
            if kind == "failed":
                self.failed_kinds[rec["finish"]] = self.failed_kinds.get(rec["finish"], 0) + 1
            for k in self.tokens:
                self.tokens[k] += rec[k]
        return rec

    def failed(self, kind: str) -> None:
        """A request that never ran (a full queue, a refusal) or that an engine failure ended."""
        with self.lock:
            self.counts["failed"] += 1
            self.failed_kinds[kind] = self.failed_kinds.get(kind, 0) + 1

    def tick(self, now_mono: float, tokens: int, running: int, waiting: int) -> None:
        """One engine-loop iteration: the tokens its step handed out (0 when idle) and the scheduler's counts."""
        b = int((now_mono + self.offset) // BUCKET_S)
        closed = None
        with self.lock:
            cur = self.cur
            if cur is None or cur[0] != b:
                closed = cur
                cur = self.cur = [b, 0, 0, 0, 0, 0, 0]
                self.buckets[b % SERIES_BUCKETS] = cur
            cur[1] += 1
            cur[2] += running
            cur[3] = max(cur[3], running)
            cur[4] += waiting
            cur[5] = max(cur[5], waiting)
            cur[6] += tokens
            self.handed += tokens
        if closed is not None and self.on_bucket is not None:
            self.on_bucket(self.gauge_record(closed))

    def gauge_record(self, b: list) -> dict:
        """A closed bucket as the production log's gauges record (requests counted by their finish time in it)."""
        lo, hi = b[0] * BUCKET_S, (b[0] + 1) * BUCKET_S
        done = []
        with self.lock:
            for r in reversed(self.records):  # newest first: stop at the first one older than the bucket
                if r["done"] < lo:
                    break
                if r["done"] < hi and r["finish"] in OK_FINISH:
                    done.append(r)
        rec = {
            "t": lo,
            "running": b[2] / b[1],
            "running_max": b[3],
            "waiting": b[4] / b[1],
            "waiting_max": b[5],
            "out_tok_s": b[6] / BUCKET_S,
            "rpm": len(done) * 60 / BUCKET_S,
            "input_tpm": sum(r["prompt"] for r in done) * 60 / BUCKET_S,
            "output_tpm": sum(r["output"] for r in done) * 60 / BUCKET_S,
        }
        if self.kv is not None:
            used, total = self.kv()
            rec["kv_used_pct"] = 100.0 * used / total if total else None
        return rec

    # -- the production log's reload, at start ------------------------------------------------------------------------

    def merge_day(self, records: list[dict], gauges: list[dict], day_start: float) -> None:
        """Today's records and gauges from earlier processes: the "today" window, the ring's older end, the series."""
        for r in records:
            derive(r)
        n_ok = sum(1 for r in records if r.get("finish") in OK_FINISH)
        n_abort = sum(1 for r in records if r.get("finish") == "abort")
        with self.lock:
            maxlen = self.records.maxlen or 0
            older = [r for r in records if r["done"] < self.started][-maxlen:]
            room = maxlen - len(self.records)
            if room > 0:
                self.records.extendleft(reversed(older[-room:]))
            self.day = {
                "start": day_start,
                "first": min((r["arrival"] for r in records if r.get("arrival") is not None), default=None),
                "ok": n_ok,
                "aborted": n_abort,
                "failed": len(records) - n_ok - n_abort,
                "n": len(records),
                **{k: sum(r.get(k, 0) for r in records) for k in ("prompt", "cached", "output")},
            }
            for g in gauges:
                self.past[int(g["t"] // BUCKET_S)] = g
        self._snap = None

    # -- the HTTP side ------------------------------------------------------------------------------------------------

    def snapshot(self, gauges: dict | None = None, now: float | None = None, ttl: float = SNAPSHOT_TTL_S) -> dict:
        """GET /v1/stats: the windows, the time series and the recent requests, cached for `ttl` seconds, with `gauges`
        (the server's live state, never cached)."""
        now = self.clock() if now is None else now
        with self._snap_lock:
            snap = self._snap
            if not (ttl and snap is not None and 0 <= now - snap[0] < ttl):
                snap = self._snap = (now, self._compute(now))
        return dict(snap[1], gauges=gauges or {})

    def _compute(self, now: float) -> dict:
        with self.lock:
            recs = list(self.records)
            buckets = [list(b) for b in self.buckets if b is not None]
            counts, failed_kinds, tokens = dict(self.counts), dict(self.failed_kinds), dict(self.tokens)
            handed, recorded = self.handed, self.recorded
            day, past = (dict(self.day) if self.day else None), dict(self.past)
        out: dict = {
            "now": now,
            "started": self.started,
            "uptime_s": now - self.started,
            "totals": {
                "requests": counts,
                "failed_kinds": failed_kinds,
                "tokens": tokens,
                "processed": tokens["prompt"] + tokens["output"],
            },
        }
        by_bucket = {b[0]: b for b in buckets}
        windows = {}
        for name, w in WINDOWS:
            lo = now - w
            span = now - max(lo, self.started)
            sel = [r for r in recs if r["done"] >= max(lo, self.started)]
            b_lo = math.ceil(lo / BUCKET_S)
            toks = sum(b[6] for k, b in by_bucket.items() if k >= b_lo)
            tspan = now - max(b_lo * BUCKET_S, self.started)
            windows[name] = self._window(sel, span, toks / tspan if tspan > 0 else None)
        span = now - self.started
        start_sel = [r for r in recs if r["done"] >= self.started]
        windows["start"] = self._window(
            start_sel, span, handed / span if span > 0 else None, totals=(recorded, counts["ok"], tokens)
        )
        if day is not None and now - day["start"] < 86400:  # a process that ran past midnight (UTC) drops "today"
            first = day["first"] if day["first"] is not None else self.started
            dspan = now - first
            dtok = {k: day[k] + tokens[k] for k in ("prompt", "cached", "output")}
            dsel = [r for r in recs if r["done"] >= day["start"]]
            # handed out today: earlier processes' gauges (out_tok_s x BUCKET_S each) plus this process's
            dhanded = handed + BUCKET_S * sum(g.get("out_tok_s") or 0 for g in past.values() if g["t"] >= day["start"])
            windows["today"] = self._window(
                dsel,
                dspan,
                dhanded / dspan if dspan > 0 else None,
                totals=(day["n"] + recorded, day["ok"] + counts["ok"], dtok),
            )
            windows["today"]["first"] = first
        out["windows"] = windows
        out["series"] = self._series(recs, by_bucket, past, now)
        out["recent"] = [{k: r.get(k) for k in RECENT_KEYS} for r in recs[-RECENT:][::-1]]
        return out

    @staticmethod
    def _window(sel: list[dict], span: float, out_tok_s: float | None, totals=None) -> dict:
        """One window: rates over `span` seconds, and the percentiles of the requests that finished in it. `totals`
        (requests, ok requests, tokens) counts a window longer than the ring holds ("start", "today")."""
        minutes = span / 60 if span > 0 else None
        ok = [r for r in sel if r["finish"] in OK_FINISH]
        if totals is None:
            n, n_ok = len(sel), len(ok)
            prompt, cached, output = (sum(r[k] for r in sel) for k in ("prompt", "cached", "output"))
        else:
            n, n_ok, t = totals
            prompt, cached, output = t["prompt"], t["cached"], t["output"]

        def per_min(x: float) -> float | None:
            return x / minutes if minutes else None

        itl = [r["itl"] for r in ok if r["itl"] is not None]
        return {
            "span_s": span,
            "n": n,
            "ok": n_ok,
            "rpm": per_min(n_ok),
            "tpm": {
                "input": per_min(prompt),
                "cached": per_min(cached),
                "output": per_min(output),
                "total": per_min(prompt + output),
            },
            "cache_hit_pct": 100.0 * cached / prompt if prompt else None,
            "out_tok_s": out_tok_s,
            "ttft": summary([r["ttft"] for r in ok if r["ttft"] is not None]),
            "decode_tps": summary([r["decode_tps"] for r in ok if r["decode_tps"] is not None], low=True),
            "e2e": summary([r["e2e"] for r in ok if r["e2e"] is not None]),
            "itl_mean": sum(itl) / len(itl) if itl else None,
        }

    @staticmethod
    def _series(recs: list[dict], by_bucket: dict, past: dict, now: float) -> list[dict]:
        """SERIES_BUCKETS points ending at the current one: live buckets, else today's reloaded gauges, else gaps."""
        last = int(now // BUCKET_S)
        first = last - SERIES_BUCKETS + 1
        done: dict[int, list[dict]] = collections.defaultdict(list)
        for r in recs:
            k = int(r["done"] // BUCKET_S)
            if k >= first and r["finish"] in OK_FINISH:
                done[k].append(r)
        gauge_keys = ("running", "running_max", "waiting", "waiting_max", "out_tok_s")
        out = []
        for k in range(first, last + 1):
            t0 = k * BUCKET_S
            b, g, rs = by_bucket.get(k), past.get(k), done.get(k, [])
            pt: dict = {"t": t0}
            if b is not None:
                width = min(BUCKET_S, now - t0) if k == last else BUCKET_S
                pt.update(
                    running=b[2] / b[1],
                    running_max=b[3],
                    waiting=b[4] / b[1],
                    waiting_max=b[5],
                    out_tok_s=b[6] / width if width > 0 else None,
                )
            else:
                pt.update({x: g.get(x) if g is not None else None for x in gauge_keys})
            if b is not None or rs:
                pt["rpm"] = len(rs) * 60 / BUCKET_S
            else:
                pt["rpm"] = g.get("rpm") if g is not None else None
            ttft = [r["ttft"] for r in rs if r["ttft"] is not None]
            dec = [r["decode_tps"] for r in rs if r["decode_tps"] is not None]
            pt.update(
                ttft_p50=pct(ttft, 0.5), ttft_p90=pct(ttft, 0.9), decode_p50=pct(dec, 0.5), decode_p10=pct(dec, 0.1)
            )
            out.append(pt)
        return out
