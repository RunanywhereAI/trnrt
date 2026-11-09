"""OpenAI-compatible HTTP server for one GLM-5.3-Flash engine.

- POST /v1/chat/completions: the prompt rendered by the model's chat template (the Hugging Face tokenizer's), the
  output's reasoning (`reasoning_content`) and tool calls (`tool_calls`) parsed out of it (serve/parsers.py), as one
  response or as server-sent events;
- POST /v1/completions: a prompt as text or token ids, through the same generation path;
- POST /v1/responses (serve/responses.py) and POST /v1/messages, /v1/messages/count_tokens (serve/anthropic.py): the
  OpenAI Responses and Anthropic Messages APIs, translated into the chat path;
- POST /tokenize and /detokenize (also under /v1);
- GET /v1/models;
- GET /health: 200 while the engine can serve, 503 with the reason once it cannot;
- GET /v1/stats: the KV-cache gauges the router (serve/router.py) uses to break ties between engines;
- GET /metrics: Prometheus counters and histograms (trnrt/metrics.py);
- SGLang's native API (serve/native.py): POST /generate, the info endpoints /model_info and /server_info, /get_load,
  /health_generate, /abort_request and /flush_cache.

A request may ask for `n` choices: each is its own engine request (Run), sharing the prompt through the prefix cache.

A request may carry a `session_id` (a string): the engine keeps that session's prefix referenced in its prefix cache,
and the router pins the session to one engine.

The engine is single-threaded and owned by one thread that runs the step loop (EngineLoop). HTTP handlers talk to it
only through a queue, and each request streams its tokens back through an asyncio queue. A client that disconnects
aborts its request, so the engine stops spending device time on output nobody will read.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import copy
import dataclasses
import functools
import importlib
import json
import logging
import math
import os
import queue
import re
import signal
import sys
import threading
import time
import traceback
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from fastapi import FastAPI, HTTPException
from fastapi import Request as HTTPRequest
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from ..engine.request import Request, SamplingParams, Status
from ..metrics import Metrics, kv_pages, spec_stats
from .parsers import (
    THINK_CLOSE,
    THINK_OPEN,
    StreamParser,
    chat_finish_reason,
    parse_full,
    template_messages,
    tool_choice_error,
)

if TYPE_CHECKING:  # only annotations: importing this module does not load the engine and its model code
    from ..engine.engine import LLMEngine

logger = logging.getLogger(__name__)


@dataclass
class _Stream:
    """A request the engine runs: its tokens go to `out` on the event loop `loop`; `sent` of them so far."""

    req: Request
    loop: asyncio.AbstractEventLoop
    out: asyncio.Queue | _Tagged
    sent: int = 0
    metrics: bool = True  # counted in /metrics when it finishes (a /health_generate probe is not)


@dataclass
class _Submit:
    """A request on its way to the engine thread. `out` receives, in order: ("accepted", Request), or ("error", why,
    HTTP status) if it was not accepted; then ("tokens", new ids, finish reason or None, their logprobs or None) until
    the finish reason is set, or ("error", why, HTTP status) if the engine fails. The status is 400 for a request the
    engine refuses, 500 for an internal error that leaves the engine serving and 503 once the engine has failed. An
    aborted request (EngineLoop.abort, abort_all) ends with ("tokens", its undelivered ids, "abort", their logprobs)."""

    rid: str
    prompt_ids: list[int]
    params: SamplingParams
    loop: asyncio.AbstractEventLoop
    session_id: str | None = None
    out: asyncio.Queue | _Tagged = field(default_factory=asyncio.Queue)
    metrics: bool = True


class _Tagged:
    """One choice's view of a shared queue: each item put here arrives on `q` as (index, item). The engine thread only
    ever calls put_nowait on a request's queue (on its event loop), so the n requests of one HTTP request (`n` > 1) can
    share one queue and be read in arrival order."""

    def __init__(self, q: asyncio.Queue, index: int):
        self.q, self.index = q, index

    def put_nowait(self, item) -> None:
        self.q.put_nowait((self.index, item))


@dataclass
class _Flush:
    """A /flush_cache on its way to the engine thread, which answers `fut` with (flushed, message): at once when the
    engine is idle, else once it is idle, or with a refusal past `deadline` (time.monotonic())."""

    deadline: float
    loop: asyncio.AbstractEventLoop
    fut: asyncio.Future


def _resolve(fut: asyncio.Future, value) -> None:
    if not fut.done():  # its handler may have been cancelled (the client went away)
        fut.set_result(value)


# Engine loops of this process, held weakly: at exit each loop's thread stops after its step, then its engine closes
# (LLMEngine.close is idempotent), which stops its tensor-parallel workers.
_LOOPS: weakref.WeakSet = weakref.WeakSet()


@atexit.register
def _stop_loops() -> None:
    if DETOK_CHECK["checked"]:  # the streaming self-check's total (_Detok._check), for the engine log
        logger.info(
            "detok self-check at exit: %d streams checked, %d mismatches",
            DETOK_CHECK["checked"],
            DETOK_CHECK["mismatch"],
        )
    for lp in list(_LOOPS):
        lp.stop()
        lp.engine.close()


def serve(app: FastAPI, **kwargs) -> None:
    """uvicorn.run(app, **kwargs), with a SIGTERM that ends the process through SystemExit (status 143) once the server
    has shut down. After its graceful shutdown uvicorn restores the previous handler and raises the signal again
    (Server.capture_signals); under SIG_DFL that killed the process there, before the atexit hook that closes the
    engine, and the tensor-parallel workers then died in a broadcast instead of being told to exit."""
    import uvicorn

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    uvicorn.run(app, **kwargs)


class EngineLoop:
    """Owns the engine: one thread runs its step loop and is the only one that touches it. Handlers hand it requests
    and aborts through `inbox`; it hands each request's tokens to that request's asyncio queue after every step."""

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._stopping = threading.Event()
        self.metrics = Metrics()
        self.vocab = engine.mcfg.vocab_size  # the embedding's rows: a prompt id outside them fails the step
        self.inbox: queue.Queue = queue.Queue()
        self.streams: dict[str, _Stream] = {}
        # Why the engine can no longer serve (a rank or this loop failed), None while it can: /health and every new
        # request answer 503 from then on. _gate orders setting it against queueing a request, so none goes unanswered.
        self.failed: str | None = None
        self._gate = threading.Lock()
        # The local-K counters for /metrics (_read_local_k): whether a step ran since they were last read, and whether
        # the engine has them at all (an engine without context-parallel row groups has none).
        self._local_k_stale = True
        self._local_k_none = False
        # time.monotonic() at the end of the last step that ran (/health_generate: a step that ends after a probe was
        # sent shows the engine is serving), and the /flush_cache waiting for the engine to be idle, if any.
        self.last_step = 0.0
        self._flush: _Flush | None = None
        self.thread = threading.Thread(target=self._run, name="trnrt-engine", daemon=True)
        self.thread.start()
        if engine.cfg.tp > 1:
            threading.Thread(target=self._watch_ranks, name="trnrt-rank-watch", daemon=True).start()
        _LOOPS.add(self)

    def stop(self, wait: float = 120.0) -> None:
        """End the engine thread after its current step (at exit, before the engine closes). Idempotent."""
        self._stopping.set()
        self.inbox.put(("wake", None))
        # A failed loop's thread is gone, or blocked for good in a collective with a dead rank (_watch_ranks)
        if self.failed is None and self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(wait)
            if self.thread.is_alive():
                logger.warning("trnrt: the engine thread did not stop within %.0f s; closing the engine anyway", wait)

    def health_error(self) -> str | None:
        """Why the engine cannot serve, or None. A worker that died while this loop waits in a step (a device collective
        with a dead peer) shows here before the loop can notice it."""
        if self.failed is not None:
            return self.failed
        dead = self.engine.dead_ranks()
        if dead:
            return f"tensor-parallel rank(s) {dead} exited"
        if not self.thread.is_alive() and not self._stopping.is_set():
            return "the engine thread exited"
        return None

    def _put(self, kind: str, payload) -> None:
        with self._gate:
            err = self.health_error()
            if err is not None:
                raise HTTPException(status_code=503, detail=f"engine failed: {err}")
            self.inbox.put((kind, payload))

    def submit(
        self,
        prompt_ids: list[int],
        params: SamplingParams,
        session_id: str | None = None,
        rid: str | None = None,
        metrics: bool = True,
        out: _Tagged | None = None,
    ) -> _Submit:
        """Queue a request for the engine thread. `rid`: the client's request id (SGLang's), which /abort_request
        names; a new one when None. `metrics`: whether /metrics counts it. `out`: a choice's view of a shared queue
        (`n` > 1), instead of its own."""
        # Checked here, before the engine sees the request: an id outside the vocabulary fails the step, and at tp > 1
        # one failed step fails the engine for every request (_fail).
        if not prompt_ids or not all(type(t) is int and 0 <= t < self.vocab for t in prompt_ids):
            raise HTTPException(
                status_code=400, detail=f"prompt must be a non-empty list of token ids in [0, {self.vocab})"
            )
        if session_id is not None and not isinstance(session_id, str):
            raise HTTPException(status_code=400, detail="session_id must be a string")
        if rid is not None and not (isinstance(rid, str) and rid):
            raise HTTPException(status_code=400, detail="rid must be a non-empty string")
        sub = _Submit(
            rid or uuid.uuid4().hex, prompt_ids, params, asyncio.get_running_loop(), session_id, metrics=metrics
        )
        if out is not None:
            sub.out = out
        self._put("add", sub)
        return sub

    def abort(self, rid: str) -> None:
        self.inbox.put(("abort", rid))

    def abort_all(self) -> None:
        """Abort every request the engine holds, and those queued for it before this call."""
        self.inbox.put(("abort_all", None))

    def flush(self, timeout_s: float) -> _Flush:
        """Ask the engine thread to drop its prefix cache (LLMEngine.flush_cache) once it is idle, waiting up to
        `timeout_s` for that (SGLang's /flush_cache: 0 flushes only an engine idle when the request arrives)."""
        loop = asyncio.get_running_loop()
        f = _Flush(time.monotonic() + timeout_s, loop, loop.create_future())
        self._put("flush", f)
        return f

    def _drain(self, block: bool) -> None:
        try:
            item = self.inbox.get(timeout=0.05) if block else self.inbox.get_nowait()
        except queue.Empty:
            return
        while True:
            kind, payload = item
            if kind == "add":
                self._add(payload)
            elif kind == "abort":
                st = self.streams.pop(payload, None)
                if st is not None:
                    self._abort(st)
            elif kind == "abort_all":
                streams, self.streams = self.streams, {}
                for st in streams.values():
                    self._abort(st)
            elif kind == "flush":
                self._start_flush(payload)
            # "wake": nothing to do, the loop re-checks whether it is stopping
            try:
                item = self.inbox.get_nowait()
            except queue.Empty:
                return

    def _add(self, sub: _Submit) -> None:
        post = sub.loop.call_soon_threadsafe
        if sub.rid in self.streams:  # a client's rid (SGLang's /generate) names one request at a time
            post(sub.out.put_nowait, ("error", f"rid {sub.rid!r} is already in use", 400))
            return
        try:
            req = self.engine.add_request(sub.prompt_ids, sub.params, rid=sub.rid, session_id=sub.session_id)
        except ValueError as e:
            post(sub.out.put_nowait, ("error", str(e), 400))
        except Exception as e:  # noqa: BLE001  add_request raised before the request reached the scheduler
            traceback.print_exc()  # (its last step): that request fails, the engine keeps serving
            post(sub.out.put_nowait, ("error", f"request refused: {e!r}", 500))
        else:
            self.streams[sub.rid] = _Stream(req, sub.loop, sub.out, metrics=sub.metrics)
            post(sub.out.put_nowait, ("accepted", req))

    def _abort(self, st: _Stream) -> None:
        """Abort a request taken out of `streams`. Its handler, if still listening (an /abort_request; a disconnected
        client's is gone), gets the tokens not yet delivered and the finish reason "abort"."""
        req = st.req
        if req.status is not Status.FINISHED:
            self.engine.abort(req)
        out = req.output_ids
        with_lp = req.params.logprobs is not None
        ready = min(len(out), len(req.logprobs)) if with_lp else len(out)
        new = out[st.sent : ready]
        lps = req.logprobs[st.sent : ready] if with_lp else None
        st.loop.call_soon_threadsafe(st.out.put_nowait, ("tokens", new, "abort", lps))

    def _start_flush(self, f: _Flush) -> None:
        """A /flush_cache, in inbox order: done now if the engine is idle, refused if another is waiting or the request
        set no time to wait, else kept until the engine is idle or its deadline passes (_check_flush; SGLang's
        scheduler_components/flush_wrapper.py)."""
        if self._flush is not None:
            f.loop.call_soon_threadsafe(_resolve, f.fut, (False, "Another flush_cache is already in progress."))
            return
        self._flush = f
        self._check_flush()

    def _check_flush(self) -> None:
        f = self._flush
        if f is None:
            return
        if not self.engine.has_work():
            self._flush = None
            try:
                self.engine.flush_cache()
            except Exception as e:  # noqa: BLE001  a failed flush is the request's answer, never the engine's end
                traceback.print_exc()
                result = (False, f"flush_cache failed: {e!r}")
            else:
                logger.info("trnrt: prefix cache flushed")
                result = (True, "")
            f.loop.call_soon_threadsafe(_resolve, f.fut, result)
        elif time.monotonic() >= f.deadline:
            self._flush = None
            busy = len(self.engine.scheduler.running) + len(self.engine.scheduler.waiting)
            f.loop.call_soon_threadsafe(
                _resolve, f.fut, (False, f"Cache not flushed: {busy} requests are running or waiting.")
            )

    def _run(self) -> None:
        try:
            self._loop()
        except BaseException as e:  # noqa: BLE001  outside a step (a step's own failure is _loop's): nothing would serve requests
            traceback.print_exc()
            self._fail(f"the engine loop failed: {e!r}")

    def _watch_ranks(self) -> None:
        """A rank that dies while the loop waits in a step leaves no thread to notice it: a collective with a dead peer
        can block instead of raising (gloo on macOS does until its timeout; a device collective too). This thread is the
        only check: the loop does not poll its workers between steps (each poll's waitpid calls release the GIL early in
        the step's gap, and the engine thread then waits behind the streams' detokenize). So it polls every 0.25 s and
        fails at first sight: a dead rank does not come back."""
        while not self._stopping.wait(0.25) and self.failed is None:
            dead = self.engine.dead_ranks()
            if dead:
                self._fail(f"tensor-parallel rank(s) {dead} exited")
                return

    def _read_local_k(self) -> None:
        """While the engine is idle (no step in flight, so the read waits for nothing): the local-K selection's counters
        for /metrics (metrics.local_k_lines), read here because this thread alone touches the engine, and only when a
        render asked for them and a step ran since the last read. An engine without them is asked once."""
        if self._local_k_none or not self._local_k_stale or not self.metrics.local_k_wanted:
            return
        self._local_k_stale = False
        read = getattr(self.engine, "local_k_counts", None)
        try:
            counts = read() if read is not None else None
        except Exception:  # noqa: BLE001  a counter, never a reason to stop serving
            traceback.print_exc()
            return
        self._local_k_none = counts is None
        self.metrics.set_local_k(counts)

    def _fail(self, reason: str) -> None:
        """A rank or this loop failed: every open and queued request gets the error, and the engine stops its workers
        without another word to them (LLMEngine.close_failed): a broadcast to a dead peer raises, and one to a live peer
        waits behind a step that will never finish. Requests are not aborted for the same reason: an abort releases the
        request's cached state, which can broadcast to the workers. The first caller wins (this loop's thread or
        _watch_ranks)."""
        with self._gate:
            if self.failed is not None:
                return
            self.failed = reason
        logger.error("trnrt: %s; /health answers 503 and the engine shuts down", reason)
        streams, self.streams = self.streams, {}
        for st in streams.values():
            st.loop.call_soon_threadsafe(st.out.put_nowait, ("error", f"engine failed: {reason}", 503))
        flushes = [self._flush] if self._flush is not None else []
        self._flush = None
        while True:
            try:
                kind, payload = self.inbox.get_nowait()
            except queue.Empty:
                break
            if kind == "add":
                payload.loop.call_soon_threadsafe(payload.out.put_nowait, ("error", f"engine failed: {reason}", 503))
            elif kind == "flush":
                flushes.append(payload)
        for f in flushes:
            f.loop.call_soon_threadsafe(_resolve, f.fut, (False, f"engine failed: {reason}"))
        try:
            self.engine.close_failed()
        except Exception:  # noqa: BLE001  the requests are answered; a teardown error must not hide the reason
            traceback.print_exc()

    def _loop(self) -> None:
        while not self._stopping.is_set() and self.failed is None:
            self._drain(block=not self.engine.has_work())
            self._check_flush()
            if not self.engine.has_work():
                self._read_local_k()
                continue
            self._local_k_stale = True
            try:
                self.engine.step()
            except Exception as e:  # noqa: BLE001  never let the engine thread die silently
                traceback.print_exc()
                if self.engine.cfg.tp > 1:  # rank 0 left a step its workers ran or wait in: the ranks are out of step
                    dead = self.engine.dead_ranks()
                    gone = f"; tensor-parallel rank(s) {dead} exited" if dead else ""
                    self._fail(f"engine step failed: {e!r}{gone}")
                    return
                for st in self.streams.values():
                    st.loop.call_soon_threadsafe(st.out.put_nowait, ("error", f"engine step failed: {e}", 500))
                    if st.req.status is not Status.FINISHED:
                        self.engine.abort(st.req)
                self.streams.clear()
                continue
            self.last_step = time.monotonic()
            batch: dict = {}  # one cross-thread call per event loop and step, not one per stream
            for rid, st in list(self.streams.items()):
                req = st.req
                out = req.output_ids
                finished = req.status is Status.FINISHED
                with_lp = req.params.logprobs is not None
                # With logprobs on, only the tokens whose logprobs are recorded go out, so every token carries its own.
                ready = len(out) if finished or not with_lp else min(len(out), len(req.logprobs))
                if ready > st.sent or finished:
                    new = out[st.sent : ready]
                    lps = req.logprobs[st.sent : ready] if with_lp else None
                    st.sent = ready
                    batch.setdefault(st.loop, []).append(
                        (st.out, ("tokens", new, req.finish_reason if finished else None, lps))
                    )
                if finished:
                    del self.streams[rid]
                    if st.metrics:
                        self.metrics.on_finish(req)
            for loop, items in batch.items():
                loop.call_soon_threadsafe(_deliver, items)


def _deliver(items) -> None:
    for q, item in items:
        q.put_nowait(item)


# The streaming self-check: every stream checks, when it ends, that what it streamed is a prefix of the full decode of its
# ids, and all of it when no stop string cut it. Counted here, one warning logged per mismatch, the total at exit.
DETOK_CHECK = {"checked": 0, "mismatch": 0}


class _Detok:
    """A streamed request's text, pushed a few tokens at a time. Not an incremental detokenizer: each push decodes
    every token id received so far, from the first, and returns the text past what it returned before, so the decoding
    work of a request grows with the square of its length.

    Never emits half of a multi-byte character, and with stop strings it holds back the
    last `max(len(stop)) - 1` characters until they cannot start a stop string, so a stop
    sequence is never partially streamed. Output is cut before the first stop string, as
    the OpenAI API specifies, or after it with `keep` (vLLM's include_stop_str_in_output,
    SGLang's no_stop_trim).

    `regexes` are SGLang's stop_regex patterns, as (compiled, the longest text a match can span or None): the output
    ends at the end of the first match, the matched text included, as SGLang's does (its trim looks for the pattern
    itself in the text, which a pattern rarely is). A match never takes back text already streamed. `matched`: the stop
    string or pattern that ended the output, None until one does.
    """

    def __init__(
        self,
        tokenizer,
        stops: tuple[str, ...] = (),
        incremental: bool = False,
        rid: str = "",
        regexes: tuple[tuple[re.Pattern, int | None], ...] = (),
        keep: bool = False,
        skip_special: bool = True,
    ):
        self.tok = tokenizer
        self.rid = rid
        self.stops = stops
        self.keep = keep
        # With keep the text up to a stop string's end is emitted either way, so nothing needs holding back.
        self.hold = 0 if keep else max((len(x) for x in stops), default=1) - 1
        self.regexes = regexes
        self.skip_special = skip_special
        self.matched: str | None = None
        self._searched = 0  # how much of the text the regexes have been searched over
        self.ids: list[int] = []
        self.emitted = 0
        self.stopped = False
        # incremental (the server's streams): the text so far is kept and extended by decoding a window of the newest ids
        # (from the last point where the text was complete, TGI's prefix / read offsets), instead of decoding every id on
        # every push, which is O(n^2) per stream and holds the GIL while the engine thread waits. The final push decodes
        # the whole output once, so the complete text is the full decode. incremental=False (decode it all, every push) is
        # the reference the tests compare against.
        self.incremental = incremental
        self._text = ""
        self._po = self._ro = 0
        self._streamed: list[str] | None = [] if incremental else None  # the self-check's record (DETOK_CHECK)

    def _check(self, full: str | None) -> None:
        """DETOK_CHECK: everything this stream emitted is a prefix of the full decode of its ids, and all of it if the
        stream ended without a stop string. `full`: that decode when the caller has it (the final push), else decoded
        here."""
        if full is None:
            full = self.tok.decode(self.ids, skip_special_tokens=self.skip_special)
        got = "".join(self._streamed or ())  # set for an incremental stream, the only caller (push)
        DETOK_CHECK["checked"] += 1
        if full.startswith(got) and (self.stopped or got == full):
            return
        DETOK_CHECK["mismatch"] += 1
        i = next((k for k, (a, b) in enumerate(zip(got, full)) if a != b), min(len(got), len(full)))
        logger.warning(
            "detok self-check mismatch %d of %d: rid %s, streamed %d chars, the full decode %d, first difference at "
            "offset %d",
            DETOK_CHECK["mismatch"],
            DETOK_CHECK["checked"],
            self.rid,
            len(got),
            len(full),
            i,
        )

    def _decoded(self, final: bool) -> str:
        skip = self.skip_special
        if final or not self.incremental:
            return self.tok.decode(self.ids, skip_special_tokens=skip)
        prefix = self.tok.decode(self.ids[self._po : self._ro], skip_special_tokens=skip)
        new = self.tok.decode(self.ids[self._po :], skip_special_tokens=skip)
        if new.startswith(prefix) and len(new) > len(prefix) and not new.endswith("\ufffd"):
            self._text += new[len(prefix) :]
            self._po, self._ro = self._ro, len(self.ids)
            return self._text
        # Not committed (the newest character is incomplete, or the window added no text): the text as the full decode
        # reads it now, the incomplete character's U+FFFD included, so a stop string before that character is found at
        # this push, as the full decode finds it (returning only the committed text + U+FFFD finds it one push late).
        # The offsets stay, so the next push re-decodes the window from the same complete point.
        if new.startswith(prefix):
            return self._text + new[len(prefix) :]
        return self._text + "\ufffd"  # an incomplete character: held back below, as the full decode's would be

    def _stop_at(self, text: str, final: bool) -> int | None:
        """Where the output ends, if a stop string or a stop regex is in `text`: the earliest to start wins (a stop
        string on a tie). Sets `matched`."""
        best: tuple[int, int, str] | None = None  # (start, end of the output, what matched)
        for x in self.stops:
            i = text.find(x)
            if i >= 0 and (best is None or i < best[0]):
                best = (i, i + len(x) if self.keep else i, x)
        if self.regexes:
            # An incomplete character's U+FFFD is not text yet: a pattern must not match it.
            body = text if final else text.rstrip("\ufffd")
            for rx, width in self.regexes:
                # Only a match that ends in the new text is new, and a bounded one starts at most `width` before it.
                # Searching from a position (not a slice) keeps anchors and lookbehinds reading the whole text.
                start = 0 if width is None else max(0, self._searched - width - 8)
                m = rx.search(body, start)
                if m is not None and (best is None or m.start() < best[0]):
                    best = (m.start(), m.end(), rx.pattern)
            self._searched = len(body)
        if best is None:
            return None
        self.matched = best[2]
        return max(best[1], self.emitted)

    def push(self, ids: list[int], final: bool, drop_last: bool = False) -> str:
        """The text these ids add. drop_last: the last id is the stop token that ended the request (EOS or a
        stop_token_ids entry), which is not part of the text (vLLM and SGLang leave it out)."""
        if self.stopped:
            return ""
        self.ids.extend(ids)
        if drop_last and ids:
            self.ids.pop()
        text = self._decoded(final)
        cut = self._stop_at(text, final)
        if cut is not None:
            self.stopped = True
            end = cut
        elif final:
            end = len(text)
        else:
            if text.endswith("\ufffd"):  # an incomplete multi-byte character
                return ""
            end = max(self.emitted, len(text) - self.hold)
        delta = text[self.emitted : end]
        self.emitted = end
        if self._streamed is not None:
            self._streamed.append(delta)
            if final or self.stopped:
                self._check(text if final else None)  # the final push's text is already the full decode
        return delta


# Request fields that would change the output in a way no served graph computes (guided decoding, prompt logprobs,
# logit biases, an echoed or suffixed prompt, a minimum length): refused with a 400, never ignored. Each is accepted at
# its neutral values, absent or listed here, which OpenAI SDKs and agent harnesses send by default. A plain-text
# response_format is neutral too; penalties and a thinking budget are the engine's to refuse (engine.unsupported),
# also only when they would change the output. min_tokens (vLLM) and min_new_tokens (SGLang) need EOS and the stop
# tokens masked out of the distribution a token is drawn from, and every served graph draws on the device from the
# logits alone, with no per-row token mask (engine/sampler.py, sample_candidates): ignoring EOS on the host instead
# would put the EOS in the output and let the model write past it.
_NEUTRAL: dict[str, tuple[object, ...]] = {
    "prompt_logprobs": (None,),
    "guided_json": (None,),
    "guided_regex": (None,),
    "guided_choice": (None,),
    "guided_grammar": (None,),
    "structured_outputs": (None,),
    "logit_bias": (None, {}),
    "echo": (None, False),
    "suffix": (None, ""),
    "min_tokens": (None, 0),
    "min_new_tokens": (None, 0),
    "add_generation_prompt": (None, True),
    "continue_final_message": (None, False),
    # A slow tokenizer's decode option; the fast tokenizer the server decodes with has no such switch.
    "spaces_between_special_tokens": (None, True),
    "functions": (None, []),
    "function_call": (None, "none", "auto"),
}


def _stream_flags(body: dict) -> tuple[bool, bool]:
    """stream, and stream_options.include_usage."""
    opts = body.get("stream_options") or {}
    return _flag(body, "stream"), isinstance(opts, dict) and _flag(opts, "include_usage")


def _refused(body: dict) -> list[str]:
    out = [k for k, neutral in _NEUTRAL.items() if body.get(k) not in neutral]
    rf = body.get("response_format")
    if rf is not None and not (isinstance(rf, dict) and rf.get("type", "text") == "text"):
        out.append("response_format")
    return out


_INT64_MAX = 2**63 - 1


def _bad(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _int(body: dict, key: str, default, lo: int, hi: int = _INT64_MAX):
    """body[key] as an int in [lo, hi] (a float with no fraction counts), or `default` when it is absent or null. 400
    otherwise: a value the engine would only fail on in a step must not get there (numpy refuses a negative or
    fractional seed, and the sampling inputs are int64 arrays), since at tp > 1 one failed step fails the engine for
    every request (EngineLoop._fail)."""
    v = body.get(key)
    if v is None:
        return default
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if type(v) is not int or not lo <= v <= hi:
        raise _bad(f"{key} must be an integer in [{lo}, {hi}]")
    return v


def _num(
    body: dict, key: str, default: float, lo: float = -math.inf, hi: float = math.inf, lo_open: bool = False
) -> float:
    """body[key] as a finite number in [lo, hi] ((lo, hi] with lo_open), or `default` when it is absent or null (null
    is the OpenAI SDKs' way to leave a field unset). 400 otherwise: a NaN temperature reaches the device's top-k."""
    v = body.get(key)
    if v is None:
        return default
    try:
        ok = type(v) in (int, float) and math.isfinite(v) and (lo < v if lo_open else lo <= v) and v <= hi
    except OverflowError:  # an int too large for a float
        ok = False
    if not ok:
        raise _bad(f"{key} must be a number in {'(' if lo_open else '['}{lo}, {hi}]")
    return float(v)


def _flag(body: dict, key: str, default: bool = False) -> bool:
    v = body.get(key)
    if v is None:
        return default
    if not isinstance(v, bool):
        raise _bad(f"{key} must be true or false")
    return v


# The sampling fields a checkpoint's generation_config.json may set defaults for, with the range each must be in.
# Penalties are left out: a default the engine refuses (engine.unsupported) would fail every request.
_GEN_DEFAULTS = {"temperature": (0.0, math.inf), "top_p": (0.0, 1.0), "top_k": (-1, _INT64_MAX), "min_p": (0.0, 1.0)}


def generation_defaults(model_path: str | None) -> dict[str, float]:
    """The checkpoint's sampling defaults, from its generation_config.json, for a request that sets none: vLLM's
    `--generation-config auto`, its default. GLM-5.3-Flash's (eb9eb208) sets temperature 1.0 and top_p 0.95, so a
    request without top_p sampled the whole candidate set at top_p 1.0. A value of the wrong type or out of range is
    left out (the server's own default stands) and logged; no file, no defaults."""
    if not model_path:
        return {}
    path = os.path.join(model_path, "generation_config.json")
    try:
        with open(path) as f:
            gen = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning("trnrt: %s is unreadable (%s); no sampling defaults from it", path, e)
        return {}
    out: dict[str, float] = {}
    for k, (lo, hi) in _GEN_DEFAULTS.items():
        v = gen.get(k) if isinstance(gen, dict) else None
        if v is None:
            continue
        if type(v) in (int, float) and math.isfinite(v) and lo <= v <= hi and (k != "top_p" or v > 0):
            out[k] = v
        else:
            logger.warning("trnrt: %s sets %s to %r, out of range; not used as a default", path, k, v)
    return out


def _params(
    body: dict, default_max: int, chat: bool, vocab: int, defaults: dict[str, float] | None = None
) -> SamplingParams:
    """A request's sampling. Every field is checked here, before the request is queued: 400 for a value of the wrong
    type or out of range, and the default for a null. default_max is used when the request sets no max_tokens;
    `defaults` (generation_defaults) replace the server's own temperature, top_p, top_k and min_p defaults."""
    d = defaults or {}
    asked = _refused(body)
    if asked:
        raise _bad(f"not supported: {', '.join(asked)}")
    stop = body.get("stop")
    stop = () if stop is None else (stop,) if isinstance(stop, str) else stop
    if not (isinstance(stop, (list, tuple)) and all(isinstance(x, str) and x for x in stop)):
        raise _bad("stop must be a non-empty string or a list of them")
    stop_ids = body.get("stop_token_ids") or ()
    if not (isinstance(stop_ids, (list, tuple)) and all(type(t) is int and 0 <= t < vocab for t in stop_ids)):
        raise _bad(f"stop_token_ids must be a list of token ids in [0, {vocab})")
    if chat:  # chat: `logprobs` turns them on, `top_logprobs` asks for alternatives
        top = _int(body, "top_logprobs", 0, 0, 20)  # the device returns at most 20 alternatives per token
        if top and not _flag(body, "logprobs"):
            raise _bad("top_logprobs needs logprobs set to true")
        logprobs = top if _flag(body, "logprobs") else None
    else:  # completions: `logprobs` is the number of alternatives (false: none, as absent)
        logprobs = None if body.get("logprobs") is False else _int(body, "logprobs", None, 0, 20)
    max_new = _int(body, "max_tokens", None, 1)
    if max_new is None:
        max_new = _int(body, "max_completion_tokens", default_max, 1)
    top_k = _int(body, "top_k", int(d.get("top_k", 0)), -1)  # -1 and 0 both turn it off (vLLM's -1, the engine's 0)
    return SamplingParams(
        max_new_tokens=max_new,
        temperature=_num(body, "temperature", d.get("temperature", 1.0), lo=0.0),
        top_p=_num(body, "top_p", d.get("top_p", 1.0), lo=0.0, hi=1.0, lo_open=True),
        top_k=max(top_k, 0),
        min_p=_num(body, "min_p", d.get("min_p", 0.0), lo=0.0, hi=1.0),
        repetition_penalty=_num(body, "repetition_penalty", 1.0, lo=0.0, lo_open=True),
        frequency_penalty=_num(body, "frequency_penalty", 0.0),
        presence_penalty=_num(body, "presence_penalty", 0.0),
        logprobs=logprobs,
        thinking_token_budget=_int(body, "thinking_token_budget", None, 0),  # vLLM's; the engine refuses any value
        stop=tuple(stop),
        stop_token_ids=tuple(stop_ids),
        ignore_eos=_flag(body, "ignore_eos"),
        seed=_int(body, "seed", None, 0),
    )


# `n` choices per request at most: each is its own engine request, holding a decode seat.
MAX_N = 128
# stop_regex: patterns per request, and characters per pattern, at most. Each pattern is searched on the event loop at
# every streamed piece of text (over the new text and the longest span a match can have, or the whole text for a
# pattern with an unbounded repeat).
MAX_STOP_REGEX = 16
MAX_STOP_REGEX_CHARS = 1024


@dataclass(frozen=True)
class StopRules:
    """A request's host-side stop rules (the engine matches stop strings and stop tokens itself): SGLang's stop_regex
    as (compiled, the longest text a match spans or None); keep: the stop text stays in the output (vLLM's
    include_stop_str_in_output, SGLang's no_stop_trim); skip_special: skip_special_tokens."""

    regexes: tuple[tuple[re.Pattern, int | None], ...] = ()
    keep: bool = False
    skip_special: bool = True


def _match_width(pattern: str) -> int | None:
    """The longest text a match of `pattern` can span, None if unbounded (or unknown): the window a new match can start
    in. Read from the regex parser's own width (SGLang computes the same bound, sampling_params.get_max_seq_length)."""
    try:
        parser = importlib.import_module("re._parser")
        hi = parser.parse(pattern).getwidth()[1]
    except Exception:  # noqa: BLE001  a private module: without it every pattern is searched over the whole text
        return None
    return int(hi) if hi < 1 << 20 else None


def _stop_rules(body: dict) -> StopRules:
    """The request's StopRules, checked: 400 for a pattern that does not compile, and for keep or skip_special_tokens
    that is not a boolean."""
    rx = body.get("stop_regex")
    rx = [] if rx is None else [rx] if isinstance(rx, str) else rx
    if not (isinstance(rx, list) and all(isinstance(x, str) and x for x in rx)):
        raise _bad("stop_regex must be a non-empty string or a list of them")
    if len(rx) > MAX_STOP_REGEX or any(len(x) > MAX_STOP_REGEX_CHARS for x in rx):
        raise _bad(f"stop_regex takes at most {MAX_STOP_REGEX} patterns of at most {MAX_STOP_REGEX_CHARS} characters")
    compiled = []
    for x in rx:
        try:
            compiled.append((re.compile(x), _match_width(x)))
        except (re.error, RecursionError, OverflowError) as e:
            raise _bad(f"stop_regex {x!r} is not a valid regular expression: {e}") from None
    keep = _flag(body, "include_stop_str_in_output") or _flag(body, "no_stop_trim")
    return StopRules(tuple(compiled), keep, _flag(body, "skip_special_tokens", True))


def _fanout(body: dict, params: SamplingParams) -> list[SamplingParams]:
    """One SamplingParams per choice (`n`, default 1). best_of is accepted only when it asks for no more than n (it would
    rank extra candidates by their logprob, which no request here returns). A seed gives choice i the seed + i, as vLLM
    does (v1/engine/parallel_sampling.py), so the choices differ and each is reproducible; without one each choice draws
    its own noise, keyed by its own request id."""
    n = _int(body, "n", 1, 1, MAX_N)
    best_of = _int(body, "best_of", n, 1)
    if best_of != n:
        raise _bad(f"not supported: best_of {best_of} with n {n}; send best_of equal to n, or leave it out")
    if params.seed is not None and params.seed + n - 1 > _INT64_MAX:
        raise _bad(f"seed must be at most {_INT64_MAX - n + 1} with n {n}: choice i draws with seed + i")
    if n == 1:
        return [params]
    return [params if params.seed is None else dataclasses.replace(params, seed=params.seed + i) for i in range(n)]


# A role test naming "developer" in a Jinja template: `m.role == 'developer'`, `message['role'] != "developer"`,
# `role in ['system', 'developer']`. The word in prose or in an emitted string is not one.
_DEVELOPER_ROLE = re.compile(r"""role['"\]]*\s*(?:==|!=|in)\s*[\[(]?[^\])\n%}]*?['"]developer['"]""")


def template_knows_developer(tok) -> bool:
    """Whether the tokenizer's chat template (one, or a dict of named ones) has a branch for role "developer"."""
    t = getattr(tok, "chat_template", None)
    return any(
        _DEVELOPER_ROLE.search(x) for x in ([t] if isinstance(t, str) else (t or {}).values()) if isinstance(x, str)
    )


def _template_ids(tok, messages: list, kwargs: dict) -> list[int]:
    import jinja2

    try:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kwargs)
    except jinja2.TemplateError as e:  # the template's own raise_exception, on messages it cannot render
        raise _bad(f"the chat template refused the messages: {e}") from None
    if hasattr(ids, "keys"):  # transformers 5 returns a BatchEncoding, not a list
        ids = ids["input_ids"]
    return list(ids)


# apply_chat_template's own arguments: the server sets them from the request (messages, tools) or fixes them.
_TEMPLATE_ARGS = {
    "conversation",
    "messages",
    "tools",
    "documents",
    "add_generation_prompt",
    "continue_final_message",
    "tokenize",
    "chat_template",
    "return_dict",
    "return_assistant_tokens_mask",
    "tokenizer_kwargs",
    "padding",
    "truncation",
    "max_length",
    "return_tensors",
}
# GLM-5.3-Flash's template (chat_template.jinja at eb9eb208) renders reasoning_effort "low" or "high" as that effort
# and any other value as "Max", its default: every other value would be answered at Max without a word.
REASONING_EFFORTS = ("low", "high", "max")


@functools.lru_cache(maxsize=8)
def _variables(src: str) -> frozenset[str] | None:
    import jinja2.meta
    from transformers.utils.chat_template_utils import _compile_jinja_template

    try:
        return frozenset(jinja2.meta.find_undeclared_variables(_compile_jinja_template(src).environment.parse(src)))
    except Exception:  # noqa: BLE001  an unparseable template is answered by apply_chat_template itself
        return None


def template_variables(tok) -> set[str] | None:
    """The variables the tokenizer's chat template (one, or a dict of named ones) reads from its context, parsed with
    transformers' own Jinja environment; None if one cannot be parsed."""
    t = getattr(tok, "chat_template", None)
    out: set[str] = set()
    for src in [t] if isinstance(t, str) else (t or {}).values():
        v = _variables(src)
        if v is None:
            return None
        out |= v
    return out


def template_kwargs(tok, body: dict) -> tuple[dict, bool | None]:
    """The chat template's extra variables for a request: its chat_template_kwargs, plus a top-level reasoning_effort
    (the OpenAI field). 400 for a variable the template never reads, which would be ignored, and for a reasoning
    effort GLM-5.3-Flash's template does not honour.

    enable_thinking (vLLM's convention) is the exception: GLM-5.3-Flash's template has none and always opens the
    reasoning block. On a template without it, it is taken out and returned second, so chat_prompt can check it against
    the rendered prompt: the value the template renders anyway is accepted, the other refused."""
    kw = body.get("chat_template_kwargs")
    if kw is None:
        kw = {}
    if not isinstance(kw, dict):
        raise _bad("chat_template_kwargs must be an object")
    kw = dict(kw)
    taken = sorted(set(kw) & _TEMPLATE_ARGS)
    if taken:
        raise _bad(f"chat_template_kwargs cannot set {', '.join(taken)}: the server sets them from the request")
    effort = body.get("reasoning_effort")
    if effort is not None:
        if kw.get("reasoning_effort", effort) != effort:
            raise _bad("reasoning_effort and chat_template_kwargs.reasoning_effort differ")
        kw["reasoning_effort"] = effort
    used = template_variables(tok)
    if used is None:
        return kw, None
    thinking = None
    if "enable_thinking" in kw and "enable_thinking" not in used:
        thinking = kw.pop("enable_thinking")
        if not isinstance(thinking, bool) and thinking is not None:
            raise _bad("chat_template_kwargs.enable_thinking must be true or false")
    unread = sorted(k for k in kw if k not in used)
    if unread:
        raise _bad(f"the model's chat template does not use {', '.join(unread)}; it would be ignored")
    if "reasoning_effort" in kw and kw["reasoning_effort"] not in REASONING_EFFORTS:
        raise _bad(
            f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)} (max is the default); the chat "
            f"template answers any other value at max"
        )
    return kw, thinking


def chat_prompt(tok, body: dict) -> tuple[list[int], dict]:
    """A /v1/chat/completions request's prompt ids, rendered by the model's chat template from its messages (assistant
    tool_calls arguments as objects: parsers.template_messages), and the StreamParser / parse_full arguments its output
    is parsed with. HTTPException 400 on a malformed request."""
    messages = body.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")
    try:
        messages = template_messages(messages, developer_to_system=not template_knows_developer(tok))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    kwargs, thinking = template_kwargs(tok, body)
    tool_choice = body.get("tool_choice")
    error = tool_choice_error(tool_choice)
    if error:
        raise HTTPException(status_code=400, detail=error)
    tools = body.get("tools") if tool_choice != "none" else None
    if tools:
        kwargs["tools"] = tools
    ids = _template_ids(tok, messages, kwargs)
    # With thinking on, the template opens the reasoning block in the prompt and the output starts inside it; with
    # thinking off it closes the block there, and the output starts with the answer.
    tail = tok.decode(ids[-4:]).rstrip()
    if thinking is not None and thinking != tail.endswith(THINK_OPEN):
        raise _bad(
            f"this model's chat template has no enable_thinking: thinking is always "
            f'{"on" if tail.endswith(THINK_OPEN) else "off"}. For less reasoning send reasoning_effort "low"'
        )
    # parallel_tool_calls=False keeps only the first call (vLLM maybe_filter_parallel_tool_calls,
    # entrypoints/serve/utils/tool_calls_utils.py).
    parse = {
        "tools": bool(tools),
        "thinking_open": tail.endswith(THINK_OPEN),
        "tool_defs": tools,
        "parallel_tool_calls": body.get("parallel_tool_calls") is not False,
    }
    return ids, parse


async def _body(http: HTTPRequest) -> dict:
    """The request's JSON object; 400 for anything else (FastAPI's raw request would answer malformed JSON 500)."""
    try:
        body = await http.json()
    except ValueError:  # json.JSONDecodeError and UnicodeDecodeError
        raise _bad("the request body must be a JSON object") from None
    if not isinstance(body, dict):
        raise _bad("the request body must be a JSON object")
    return body


async def _disconnected(http: HTTPRequest) -> None:
    """Returns once the client has gone away (vLLM's listen_for_disconnect). The body is read already, so the next
    ASGI message is the disconnect."""
    while (await http.receive())["type"] != "http.disconnect":
        pass


async def _unless_disconnected(http: HTTPRequest, aw):
    """`aw`'s result, or None if the client disconnected first (aw is then cancelled). aw's exceptions propagate."""
    task = asyncio.ensure_future(aw)
    gone = asyncio.ensure_future(_disconnected(http))
    try:
        await asyncio.wait({task, gone}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:  # this handler was cancelled itself
        task.cancel()
        raise
    finally:
        gone.cancel()
    if not task.done():  # the client went away first
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return None
    return task.result()


class EngineError(Exception):
    """An engine request failed after it was accepted: `detail` and the HTTP `status` (_Submit)."""

    def __init__(self, detail: str, status: int):
        super().__init__(detail)
        self.detail, self.status = detail, status


class Run:
    """The engine requests of one HTTP request: `n` choices of one prompt (OpenAI's `n`), each its own engine request,
    sharing the prompt. The prefix cache makes the copies cheap: their shared prompt pages become a checkpoint target of
    the first one the scheduler admits (_junctions_ahead when they are queued together; _mark_running_junctions when one
    arrives while the first prefills), and the others wait until they can resume from it (_prefix_in_progress,
    engine/scheduler.py), so the prompt is computed once and each other choice computes only its last partial page.
    Their tokens arrive tagged with the choice's index on one queue.

    Each choice has its own detokenizer (stop strings, stop_regex, skip_special_tokens) and, for a chat request, its own
    GLM parser. A choice the host stops (a stop_regex match) is aborted in the engine. Whatever ends the run early (an
    error, a client gone, a consumer that stops reading) aborts every choice still running (close)."""

    def __init__(
        self,
        loop_: EngineLoop,
        tok,
        prompt_ids: list[int],
        params: list[SamplingParams],
        session_id,
        rules: StopRules,
        eos: tuple[int, ...] = (),
        parse: dict | None = None,
    ):
        self.loop_, self.tok, self.prompt_ids, self.params = loop_, tok, prompt_ids, params
        self.session_id, self.rules, self.parse = session_id, rules, parse
        n = len(params)
        self.q: asyncio.Queue = asyncio.Queue()
        self.subs: list[_Submit] = []
        self.reqs: list[Request | None] = [None] * n
        self.detoks: list[_Detok] = []
        self.parsers = [StreamParser(**parse) for _ in range(n)] if parse is not None else []
        # The ids whose token ends a request without being part of its text (EOS and the request's stop ids).
        self.stop_ids = [frozenset(eos) | frozenset(p.stop_token_ids) for p in params]
        self.ids: list[list[int]] = [[] for _ in range(n)]
        self.text = [""] * n
        self.lps: list[list] = [[] for _ in range(n)]
        self.finish: list[str | None] = [None] * n
        self.created = int(time.time())
        self._early: list = []  # tokens that arrived while start() still waited for another choice's acceptance
        self._open = n
        self.completed = False

    @property
    def n(self) -> int:
        return len(self.params)

    @property
    def rid(self) -> str:
        return self.subs[0].rid

    async def start(self) -> None:
        """Hand every choice to the engine and wait until each is accepted. HTTPException (the engine's status) if one is
        not: the others are aborted, so no choice runs for a request that is answered with an error."""
        try:
            for i, p in enumerate(self.params):
                self.subs.append(self.loop_.submit(self.prompt_ids, p, self.session_id, out=_Tagged(self.q, i)))
            waiting = set(range(self.n))
            while waiting:
                i, item = await self.q.get()
                if item[0] == "error":
                    raise HTTPException(status_code=item[2], detail=item[1])
                if item[0] == "accepted":
                    self.reqs[i] = item[1]
                    waiting.discard(i)
                else:
                    self._early.append((i, item))
        except BaseException:
            self.close()
            raise
        for sub in self.subs:
            self.detoks.append(
                _Detok(
                    self.tok,
                    sub.params.stop,
                    incremental=True,
                    rid=sub.rid,
                    regexes=self.rules.regexes,
                    keep=self.rules.keep,
                    skip_special=self.rules.skip_special,
                )
            )

    def close(self) -> None:
        """Abort every choice that has not finished (idempotent; nothing once the run completed)."""
        if self.completed:
            return
        for i, sub in enumerate(self.subs):
            if self.finish[i] is None:
                self.loop_.abort(sub.rid)

    async def deliveries(self):
        """(choice index, new text, finish reason or None, (id, logprob entry) pairs or None) per delivery, until every
        choice has finished. EngineError if the engine fails a choice; close() runs on any early exit."""
        try:
            while self._open:
                i, item = self._early.pop(0) if self._early else await self.q.get()
                if item[0] == "error":
                    raise EngineError(item[1], item[2])
                if self.finish[i] is not None:  # a choice the host stopped: the tokens it made before its abort
                    continue
                _, ids, finish, lps = item
                self.ids[i] += ids
                if lps is not None:
                    self.lps[i] += zip(ids, lps)
                d = self.detoks[i]
                drop = finish == "stop" and not self.rules.keep and bool(ids) and ids[-1] in self.stop_ids[i]
                text = d.push(ids, finish is not None, drop_last=drop)
                if d.matched is not None:
                    if finish is None:  # a stop_regex match: the engine knows nothing of it
                        self.loop_.abort(self.subs[i].rid)
                    finish = "stop"
                self.text[i] += text
                if finish is not None:
                    self.finish[i] = finish
                    self._open -= 1
                yield i, text, finish, (list(zip(ids, lps)) if lps is not None else None)
            self.completed = True
        finally:
            self.close()

    async def parsed(self):
        """A chat run's deliveries through each choice's GLM parser: (index, delta dicts (StreamParser.push), finish
        reason as the chat API gives it (chat_finish_reason), logprob pairs or None)."""
        async for i, text, finish, pairs in self.deliveries():
            p = self.parsers[i]
            deltas = p.push(text, finish is not None)
            yield i, deltas, chat_finish_reason(finish, p.saw_tool), pairs

    async def wait(self, http: HTTPRequest) -> bool:
        """Every choice to its end (non-streaming); False if the client disconnected first (every choice is then
        aborted). EngineError if the engine fails one."""

        async def drain() -> bool:
            async for _ in self.deliveries():
                pass
            return True

        try:
            done = await _unless_disconnected(http, drain())
        except BaseException:
            self.close()
            raise
        return done is not None

    def message(self, i: int) -> tuple[dict, str | None]:
        """A finished chat choice's message (parse_full of its whole text, as the non-streaming chat API returns it) and
        its finish reason."""
        msg = parse_full(self.text[i], **(self.parse or {}))
        return msg, chat_finish_reason(self.finish[i], bool(msg.get("tool_calls")))

    def matched_stop(self, i: int) -> str | int | None:
        """SGLang's matched_stop for a finished choice: the stop string or stop_regex pattern that ended it, else the
        token id (EOS or a stop id) that did, else None (a length or an abort)."""
        d = self.detoks[i] if i < len(self.detoks) else None
        if d is not None and d.matched is not None:
            return d.matched
        ids = self.ids[i]
        if self.finish[i] == "stop" and ids and ids[-1] in self.stop_ids[i]:
            return ids[-1]
        return None

    def cached_tokens(self) -> int:
        """The prompt tokens resumed from the prefix cache: the fewest of any choice's, since the prompt is counted once
        (the choice that computed it)."""
        return min((max(r.num_cached_tokens, 0) for r in self.reqs if r is not None), default=0)

    def usage(self) -> dict:
        """OpenAI usage: the prompt counted once (as OpenAI counts it for n choices), every choice's output tokens."""
        out = sum(len(x) for x in self.ids)
        return {
            "prompt_tokens": len(self.prompt_ids),
            "completion_tokens": out,
            "total_tokens": len(self.prompt_ids) + out,
            "prompt_tokens_details": {"cached_tokens": self.cached_tokens()},
        }

    def reasoning_tokens(self, i: int) -> int | None:
        """The output tokens choice i spent reasoning: those before its first `</think>` (between `<think>` and it
        when the prompt did not open the block), counted on its ids; None if the tokenizer has no single id for the
        tags."""
        close = _tag_id(self.tok, THINK_CLOSE)
        ids = self.ids[i]
        if close is None:
            return None
        start = 0
        if not (self.parse or {}).get("thinking_open"):
            opening = _tag_id(self.tok, THINK_OPEN)
            if opening is None or opening not in ids:
                return 0
            start = ids.index(opening) + 1
        end = ids.index(close, start) if close in ids[start:] else len(ids)
        return end - start


def _tag_id(tok, tag: str) -> int | None:
    """The id of a tag the tokenizer holds as one token (GLM's `<think>` and `</think>` are added tokens), else None."""
    ids = tok.encode(tag, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def build_app(engine: LLMEngine, model_name: str, preset: str | None = None) -> FastAPI:
    """The server of one engine. `preset`: the --preset it was started with, for /server_info."""
    app = FastAPI(title="TrnRT")
    loop_ = EngineLoop(engine)
    tok = engine.tokenizer
    # A chat request's prompt is rendered and tokenized on one worker thread with its own tokenizer, not on the event
    # loop: GLM-5.3-Flash's template takes tens of milliseconds at 100K tokens, which would stall every other request's
    # stream, and two threads in one Hugging Face fast tokenizer can raise "Already borrowed".
    prompt_pool = ThreadPoolExecutor(1, thread_name_prefix="trnrt-prompt")
    prompt_tok = copy.deepcopy(tok)
    gen_defaults = generation_defaults(getattr(engine, "model_path", None))
    if gen_defaults:
        logger.info("trnrt: sampling defaults from the checkpoint's generation_config.json: %s", gen_defaults)

    @app.get("/health")
    async def health():
        err = loop_.health_error()
        if err is not None:  # the router and any supervisor stop sending to a dead engine
            return JSONResponse(status_code=503, content={"status": "failed", "error": err})
        return {"status": "ok"}

    @app.get("/v1/stats")
    async def stats():
        """The router's tie-break between engines: KV pages in use and usable (metrics.kv_pages)."""
        used, total = kv_pages(engine)
        gauges: dict = {"kv_pages_used": used, "kv_pages_total": total}
        spec = spec_stats(engine)
        if spec is not None:
            gauges["spec"] = spec
        return {"gauges": gauges}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(loop_.metrics.render(engine), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "trnrt"}]}

    eos = tuple(getattr(engine.mcfg, "eos_token_ids", ()) or ())

    async def start_run(body: dict, prompt_ids: list[int], params: SamplingParams, parse: dict | None = None) -> Run:
        """A request's choices handed to the engine and accepted (Run.start). parse: a chat request's parser arguments
        (chat_prompt), None for a completion. Every request field is checked before anything is queued."""
        rules = _stop_rules(body)
        choices = _fanout(body, params)
        run = Run(loop_, tok, prompt_ids, choices, body.get("session_id"), rules, eos, parse)
        await run.start()
        return run

    async def start_chat(body: dict) -> Run:
        """A /v1/chat/completions body's run: its prompt rendered on the prompt thread, its sampling checked, its choices
        accepted. The Responses and Messages APIs (serve/responses.py, serve/anthropic.py) translate into this."""
        ids, parse = await asyncio.get_running_loop().run_in_executor(prompt_pool, chat_prompt, prompt_tok, body)
        # max_tokens defaults to the rest of the context (vLLM's default): reasoning at the template's default effort
        # (Max) can run long, and a smaller hidden default would cut the answer off
        room = max(engine.cfg.max_model_len - len(ids), 1)
        return await start_run(
            body, ids, _params(body, room, chat=True, vocab=loop_.vocab, defaults=gen_defaults), parse
        )

    async def _generate(http: HTTPRequest, body: dict, run: Run, chat: bool):
        stream, include_usage = _stream_flags(body)  # checked before the run started: raises nothing here
        created = run.created
        oid = ("chatcmpl-" if chat else "cmpl-") + run.rid
        obj = "chat.completion" if chat else "text_completion"

        def lp_block(lps):
            """OpenAI logprobs from (token id, (logprob, top ids, top logprobs)) pairs."""
            if lps is None:
                return None
            if chat:

                def entry(t, lp):
                    s = cast("str", tok.decode([t]))  # one id decodes to one string; the stubs also allow a list
                    return {"token": s, "logprob": lp, "bytes": list(s.encode())}

                return {
                    "content": [
                        dict(entry(t, lp), top_logprobs=[entry(i, v) for i, v in zip(ti, tl)])
                        for t, (lp, ti, tl) in lps
                    ]
                }
            return {
                "tokens": [tok.decode([t]) for t, _ in lps],
                "token_logprobs": [lp for _, (lp, _, _) in lps],
                "top_logprobs": [{tok.decode([i]): v for i, v in zip(ti, tl)} for _, (_, ti, tl) in lps],
            }

        def chunk(i: int, text: str, finish: str | None, lps=None) -> dict:
            if chat:
                choice = {"index": i, "delta": {"content": text} if text else {}, "finish_reason": finish}
            else:
                choice = {"index": i, "text": text, "finish_reason": finish}
            if lps is not None:
                choice["logprobs"] = lp_block(lps)
            if finish is not None:
                choice["matched_stop"] = run.matched_stop(i)  # SGLang's field
            return {
                "id": oid,
                "object": obj + (".chunk" if chat else ""),
                "created": created,
                "model": model_name,
                "choices": [choice],
            }

        def chat_chunks(i: int, deltas: list[dict], finish: str | None, pairs) -> list[dict]:
            if not deltas:
                return [chunk(i, "", finish, pairs)] if (finish or pairs) else []
            out = []
            for k, d in enumerate(deltas):
                last = k == len(deltas) - 1
                c = chunk(i, "", finish if last else None, pairs if last else None)
                c["choices"][0]["delta"] = d
                out.append(c)
            return out

        async def events():
            try:
                if chat:  # OpenAI's first chunk of each choice names the role
                    for i in range(run.n):
                        c = chunk(i, "", None)
                        c["choices"][0]["delta"] = {"role": "assistant", "content": ""}
                        yield f"data: {json.dumps(c)}\n\n"
                if chat:
                    async for i, deltas, finish, pairs in run.parsed():
                        for c in chat_chunks(i, deltas, finish, pairs):
                            yield f"data: {json.dumps(c)}\n\n"
                else:
                    async for i, text, finish, pairs in run.deliveries():
                        if text or finish or pairs:
                            yield f"data: {json.dumps(chunk(i, text, finish, pairs))}\n\n"
                if include_usage:  # OpenAI stream_options.include_usage: one last chunk, no choices
                    last = {
                        "id": oid,
                        "object": obj + (".chunk" if chat else ""),
                        "created": created,
                        "model": model_name,
                        "choices": [],
                        "usage": run.usage(),
                    }
                    yield f"data: {json.dumps(last)}\n\n"
                yield "data: [DONE]\n\n"
            except EngineError as e:
                yield f"data: {json.dumps({'error': {'message': e.detail, 'code': e.status}})}\n\n"
            finally:
                run.close()  # the generator is closed early when the client goes away: abort what still runs

        if stream:  # StreamingResponse closes events() when the client goes away, and its finally aborts
            return StreamingResponse(events(), media_type="text/event-stream")

        # Nothing cancels a plain response's handler when its client disconnects, so the request would hold its seat
        # until max_tokens: Run.wait watches for the disconnect, and aborts.
        try:
            if not await run.wait(http):
                return Response(status_code=499)  # nginx's "client closed request"; nobody reads it
        except EngineError as e:
            raise HTTPException(status_code=e.status, detail=e.detail) from None
        choices = []
        for i in range(run.n):
            if chat:
                msg, finish = run.message(i)
                choice = {"index": i, "message": msg, "finish_reason": finish}
            else:
                choice = {"index": i, "text": run.text[i], "finish_reason": run.finish[i]}
            if run.params[i].logprobs is not None:
                choice["logprobs"] = lp_block(run.lps[i])
            choice["matched_stop"] = run.matched_stop(i)
            choices.append(choice)
        return JSONResponse(
            {
                "id": oid,
                "object": obj,
                "created": created,
                "model": model_name,
                "choices": choices,
                "usage": run.usage(),
            }
        )

    @app.post("/v1/completions")
    async def completions(http: HTTPRequest):
        body = await _body(http)
        prompt = body.get("prompt", "")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            ids = prompt
        elif isinstance(prompt, str):
            ids = tok(prompt)["input_ids"]
        else:
            raise HTTPException(status_code=400, detail="prompt must be a string or a list of token ids")
        # max_tokens defaults to 16, the OpenAI completions API's default
        params = _params(body, 16, chat=False, vocab=loop_.vocab, defaults=gen_defaults)
        _stream_flags(body)  # checked before anything is queued
        return await _generate(http, body, await start_run(body, ids, params), chat=False)

    @app.post("/v1/chat/completions")
    async def chat(http: HTTPRequest):
        body = await _body(http)
        _stream_flags(body)  # checked before anything is queued
        return await _generate(http, body, await start_chat(body), chat=True)

    # SGLang's and vLLM's /tokenize and /detokenize (also under /v1). A chat request is rendered exactly as
    # /v1/chat/completions renders it (chat_prompt), so `count` is what the engine would be sent; a prompt is encoded as
    # /v1/completions encodes it. Both run on the prompt thread with its tokenizer, as a chat prompt does: never on the
    # event loop, and never in a second thread inside one tokenizer.
    def tokenize(body: dict) -> dict:
        if (body.get("prompt") is None) == (body.get("messages") is None):
            raise _bad("exactly one of prompt or messages is required")
        if body.get("messages") is not None:
            # The fields chat refuses unless neutral (_NEUTRAL) and that change the rendered prompt.
            for key in ("add_generation_prompt", "continue_final_message"):
                if body.get(key) not in _NEUTRAL[key]:
                    raise _bad(f"{key} is not supported; send the request without it")
            ids, _ = chat_prompt(prompt_tok, body)
            seqs, batch = [ids], False
        else:
            prompt = body["prompt"]
            batch = isinstance(prompt, list)
            texts = prompt if batch else [prompt]
            if not texts or not all(isinstance(t, str) for t in texts):
                raise _bad("prompt must be a string or a non-empty list of strings")
            special = _flag(body, "add_special_tokens", True)
            seqs = [list(prompt_tok(t, add_special_tokens=special)["input_ids"]) for t in texts]
        out: dict = {
            "tokens": seqs if batch else seqs[0],
            "count": [len(s) for s in seqs] if batch else len(seqs[0]),
            "max_model_len": engine.cfg.max_model_len,
        }
        if _flag(body, "return_token_strs"):  # vLLM's
            strs = [prompt_tok.convert_ids_to_tokens(s) for s in seqs]
            out["token_strs"] = strs if batch else strs[0]
        return out

    def detokenize(body: dict) -> dict:
        tokens = body.get("tokens")
        batch = isinstance(tokens, list) and bool(tokens) and all(isinstance(s, list) for s in tokens)
        seqs: list = cast("list", tokens) if batch else [tokens]
        if not all(isinstance(s, list) and all(type(t) is int and 0 <= t < loop_.vocab for t in s) for s in seqs):
            raise _bad(f"tokens must be a list of token ids in [0, {loop_.vocab}), or a list of such lists")
        skip = _flag(body, "skip_special_tokens", True)  # SGLang's default
        texts = [prompt_tok.decode(s, skip_special_tokens=skip) for s in seqs]
        text = texts if batch else texts[0]
        return {"text": text, "prompt": text}  # SGLang answers `text`, vLLM `prompt`

    @app.post("/tokenize")
    @app.post("/v1/tokenize")
    async def tokenize_route(http: HTTPRequest):
        body = await _body(http)
        return await asyncio.get_running_loop().run_in_executor(prompt_pool, tokenize, body)

    @app.post("/detokenize")
    @app.post("/v1/detokenize")
    async def detokenize_route(http: HTTPRequest):
        body = await _body(http)
        return await asyncio.get_running_loop().run_in_executor(prompt_pool, detokenize, body)

    from . import native  # here, not at the top: native imports this module

    native.add_routes(
        app,
        loop_,
        model_name=model_name,
        preset=preset,
        prompt_pool=prompt_pool,
        prompt_tok=prompt_tok,
        gen_defaults=gen_defaults,
    )

    # The Responses and Messages APIs, translated into the chat path (start_chat). Imported here: both import this module.
    from . import anthropic, responses

    @app.post("/v1/responses")
    async def responses_route(http: HTTPRequest):
        body = await _body(http)
        return await responses.respond(http, body, start_chat, model_name)

    @app.get("/v1/responses/{response_id}")
    @app.delete("/v1/responses/{response_id}")
    @app.post("/v1/responses/{response_id}/cancel")
    async def stored_response(response_id: str):
        raise responses.not_stored(response_id)

    @app.post("/v1/messages")
    async def messages_route(http: HTTPRequest):
        try:
            body = await _body(http)
            return await anthropic.respond(http, body, start_chat, model_name)
        except HTTPException as e:  # Anthropic's error shape
            return anthropic.error_response(e)

    def count_tokens(body: dict) -> dict:
        """Anthropic's count_tokens: the prompt the same request to /v1/messages would send the engine, rendered on the
        prompt thread as a chat prompt is (/tokenize counts the same way)."""
        ids, _ = chat_prompt(prompt_tok, anthropic.to_chat(body))
        return {"input_tokens": len(ids)}

    @app.post("/v1/messages/count_tokens")
    async def count_tokens_route(http: HTTPRequest):
        try:
            body = await _body(http)
            return await asyncio.get_running_loop().run_in_executor(prompt_pool, count_tokens, body)
        except HTTPException as e:
            return anthropic.error_response(e)

    return app
