"""OpenAI-compatible HTTP server for one GLM-5.3-Flash engine.

- POST /v1/chat/completions: the prompt rendered by the model's chat template (the Hugging Face tokenizer's), the
  output's reasoning (`reasoning_content`) and tool calls (`tool_calls`) parsed out of it (serve/parsers.py), as one
  response or as server-sent events;
- POST /v1/completions: a prompt as text or token ids, through the same generation path;
- GET /v1/models;
- GET /health: 200 while the engine can serve, 503 with the reason once it cannot;
- GET /v1/stats: the KV-cache gauges the router (serve/router.py) uses to break ties between engines;
- GET /metrics: Prometheus counters and histograms (trnrt/metrics.py).

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
import functools
import json
import logging
import math
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
from .parsers import THINK_OPEN, StreamParser, chat_finish_reason, parse_full, template_messages, tool_choice_error

if TYPE_CHECKING:  # only annotations: importing this module does not load the engine and its model code
    from ..engine.engine import LLMEngine

logger = logging.getLogger(__name__)


@dataclass
class _Stream:
    """A request the engine runs: its tokens go to `out` on the event loop `loop`; `sent` of them so far."""

    req: Request
    loop: asyncio.AbstractEventLoop
    out: asyncio.Queue
    sent: int = 0


@dataclass
class _Submit:
    """A request on its way to the engine thread. `out` receives, in order: ("accepted", Request), or ("error", why,
    HTTP status) if it was not accepted; then ("tokens", new ids, finish reason or None, their logprobs or None) until
    the finish reason is set, or ("error", why, HTTP status) if the engine fails. The status is 400 for a request the
    engine refuses, 500 for an internal error that leaves the engine serving and 503 once the engine has failed."""

    rid: str
    prompt_ids: list[int]
    params: SamplingParams
    loop: asyncio.AbstractEventLoop
    session_id: str | None = None
    out: asyncio.Queue = field(default_factory=asyncio.Queue)


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

    def submit(self, prompt_ids: list[int], params: SamplingParams, session_id: str | None = None) -> _Submit:
        # Checked here, before the engine sees the request: an id outside the vocabulary fails the step, and at tp > 1
        # one failed step fails the engine for every request (_fail).
        if not prompt_ids or not all(type(t) is int and 0 <= t < self.vocab for t in prompt_ids):
            raise HTTPException(
                status_code=400, detail=f"prompt must be a non-empty list of token ids in [0, {self.vocab})"
            )
        if session_id is not None and not isinstance(session_id, str):
            raise HTTPException(status_code=400, detail="session_id must be a string")
        sub = _Submit(uuid.uuid4().hex, prompt_ids, params, asyncio.get_running_loop(), session_id)
        self._put("add", sub)
        return sub

    def abort(self, rid: str) -> None:
        self.inbox.put(("abort", rid))

    def _drain(self, block: bool) -> None:
        try:
            item = self.inbox.get(timeout=0.05) if block else self.inbox.get_nowait()
        except queue.Empty:
            return
        while True:
            kind, payload = item
            if kind == "add":
                post = payload.loop.call_soon_threadsafe
                try:
                    req = self.engine.add_request(
                        payload.prompt_ids, payload.params, rid=payload.rid, session_id=payload.session_id
                    )
                except ValueError as e:
                    post(payload.out.put_nowait, ("error", str(e), 400))
                except Exception as e:  # noqa: BLE001  add_request raised before the request reached the scheduler
                    traceback.print_exc()  # (its last step): that request fails, the engine keeps serving
                    post(payload.out.put_nowait, ("error", f"request refused: {e!r}", 500))
                else:
                    self.streams[payload.rid] = _Stream(req, payload.loop, payload.out)
                    post(payload.out.put_nowait, ("accepted", req))
            elif kind == "abort":
                st = self.streams.pop(payload, None)
                if st is not None and st.req.status is not Status.FINISHED:
                    self.engine.abort(st.req)
            # "wake": nothing to do, the loop re-checks whether it is stopping
            try:
                item = self.inbox.get_nowait()
            except queue.Empty:
                return

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
        while True:
            try:
                kind, payload = self.inbox.get_nowait()
            except queue.Empty:
                break
            if kind == "add":
                payload.loop.call_soon_threadsafe(payload.out.put_nowait, ("error", f"engine failed: {reason}", 503))
        try:
            self.engine.close_failed()
        except Exception:  # noqa: BLE001  the requests are answered; a teardown error must not hide the reason
            traceback.print_exc()

    def _loop(self) -> None:
        while not self._stopping.is_set() and self.failed is None:
            self._drain(block=not self.engine.has_work())
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
    the OpenAI API specifies.
    """

    def __init__(self, tokenizer, stops: tuple[str, ...] = (), incremental: bool = False, rid: str = ""):
        self.tok = tokenizer
        self.rid = rid
        self.stops = stops
        self.hold = max((len(x) for x in stops), default=1) - 1
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
            full = self.tok.decode(self.ids, skip_special_tokens=True)
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
        if final or not self.incremental:
            return self.tok.decode(self.ids, skip_special_tokens=True)
        prefix = self.tok.decode(self.ids[self._po : self._ro], skip_special_tokens=True)
        new = self.tok.decode(self.ids[self._po :], skip_special_tokens=True)
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

    def push(self, ids: list[int], final: bool) -> str:
        if self.stopped:
            return ""
        self.ids.extend(ids)
        text = self._decoded(final)
        cut = min((i for i in (text.find(x) for x in self.stops) if i >= 0), default=-1)
        if cut >= 0:
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
# several choices, logit biases, an echoed or suffixed prompt): refused with a 400, never ignored. Each is accepted at
# its neutral values, absent or listed here, which OpenAI SDKs and agent harnesses send by default. A plain-text
# response_format is neutral too; penalties and a thinking budget are the engine's to refuse (engine.unsupported),
# also only when they would change the output.
_NEUTRAL: dict[str, tuple[object, ...]] = {
    "prompt_logprobs": (None,),
    "guided_json": (None,),
    "guided_regex": (None,),
    "guided_choice": (None,),
    "guided_grammar": (None,),
    "structured_outputs": (None,),
    "n": (None, 1),
    "best_of": (None, 1),
    "logit_bias": (None, {}),
    "echo": (None, False),
    "suffix": (None, ""),
    "min_tokens": (None, 0),
    "include_stop_str_in_output": (None, False),
    "add_generation_prompt": (None, True),
    "continue_final_message": (None, False),
    "skip_special_tokens": (None, True),
    "functions": (None, []),
    "function_call": (None, "none", "auto"),
}


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


def _params(body: dict, default_max: int, chat: bool, vocab: int) -> SamplingParams:
    """A request's sampling. Every field is checked here, before the request is queued: 400 for a value of the wrong
    type or out of range, and the default for a null. default_max is used when the request sets no max_tokens."""
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
    top_k = _int(body, "top_k", 0, -1)  # -1 and 0 both turn it off (vLLM's -1, the engine's 0)
    return SamplingParams(
        max_new_tokens=max_new,
        temperature=_num(body, "temperature", 1.0, lo=0.0),
        top_p=_num(body, "top_p", 1.0, lo=0.0, hi=1.0, lo_open=True),
        top_k=max(top_k, 0),
        min_p=_num(body, "min_p", 0.0, lo=0.0, hi=1.0),
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


def build_app(engine: LLMEngine, model_name: str) -> FastAPI:
    app = FastAPI(title="TrnRT")
    loop_ = EngineLoop(engine)
    tok = engine.tokenizer
    # A chat request's prompt is rendered and tokenized on one worker thread with its own tokenizer, not on the event
    # loop: GLM-5.3-Flash's template takes tens of milliseconds at 100K tokens, which would stall every other request's
    # stream, and two threads in one Hugging Face fast tokenizer can raise "Already borrowed".
    prompt_pool = ThreadPoolExecutor(1, thread_name_prefix="trnrt-prompt")
    prompt_tok = copy.deepcopy(tok)

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

    async def _generate(
        http: HTTPRequest,
        body: dict,
        prompt_ids: list[int],
        params: SamplingParams,
        chat: bool,
        parse: dict | None = None,
    ):
        parse_args = parse or {}
        stream = _flag(body, "stream")
        opts = body.get("stream_options") or {}
        include_usage = isinstance(opts, dict) and _flag(opts, "include_usage")
        sub = loop_.submit(prompt_ids, params, body.get("session_id"))
        # Admission errors (a prompt that does not fit, an unsupported parameter) arrive before any token: surface
        # them as HTTP status codes rather than inside a 200 stream.
        first = await sub.out.get()
        if first[0] == "error":
            raise HTTPException(status_code=first[2], detail=first[1])
        req = first[1]
        created = int(time.time())
        oid = ("chatcmpl-" if chat else "cmpl-") + sub.rid
        obj = "chat.completion" if chat else "text_completion"
        detok = _Detok(tok, params.stop, incremental=True, rid=sub.rid)

        def usage_of(n_out: int) -> dict:
            """OpenAI usage, with prompt_tokens_details.cached_tokens: the prompt tokens this request resumed from the
            prefix cache (0 cold). Final once the request has finished."""
            return {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": n_out,
                "total_tokens": len(prompt_ids) + n_out,
                "prompt_tokens_details": {"cached_tokens": max(req.num_cached_tokens, 0)},
            }

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

        def chunk(text: str, finish: str | None, lps=None) -> dict:
            if chat:
                choice = {"index": 0, "delta": {"content": text} if text else {}, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": text, "finish_reason": finish}
            if lps is not None:
                choice["logprobs"] = lp_block(lps)
            return {
                "id": oid,
                "object": obj + (".chunk" if chat else ""),
                "created": created,
                "model": model_name,
                "choices": [choice],
            }

        sparser = StreamParser(**parse_args) if chat else None

        def chat_chunks(text: str, finish: str | None, pairs):
            if sparser is None:  # only a chat request builds chunks through the parser
                raise ValueError("no stream parser for a completions request")
            deltas = sparser.push(text, finish is not None)
            finish = chat_finish_reason(finish, sparser.saw_tool)
            if not deltas:
                return [chunk("", finish, pairs)] if (finish or pairs) else []
            out = []
            for i, d in enumerate(deltas):
                last = i == len(deltas) - 1
                c = chunk("", finish if last else None, pairs if last else None)
                c["choices"][0]["delta"] = d
                out.append(c)
            return out

        async def events():
            completed = False
            n_streamed = 0
            try:
                if chat:  # OpenAI's first chunk names the role
                    c = chunk("", None)
                    c["choices"][0]["delta"] = {"role": "assistant", "content": ""}
                    yield f"data: {json.dumps(c)}\n\n"
                while True:
                    kind, *rest = await sub.out.get()
                    if kind == "error":
                        completed = True
                        yield f"data: {json.dumps({'error': {'message': rest[0], 'code': rest[1]}})}\n\n"
                        return
                    ids, finish, lps = rest
                    n_streamed += len(ids)
                    text = detok.push(ids, finish is not None)
                    pairs = list(zip(ids, lps)) if lps is not None else None
                    if text or finish or pairs:
                        for c in chat_chunks(text, finish, pairs) if chat else [chunk(text, finish, pairs)]:
                            yield f"data: {json.dumps(c)}\n\n"
                    if finish is not None:
                        completed = True
                        if include_usage:  # OpenAI stream_options.include_usage: one last chunk, no choices
                            yield (
                                "data: "
                                + json.dumps(
                                    {
                                        "id": oid,
                                        "object": obj + (".chunk" if chat else ""),
                                        "created": created,
                                        "model": model_name,
                                        "choices": [],
                                        "usage": usage_of(n_streamed),
                                    }
                                )
                                + "\n\n"
                            )
                        yield "data: [DONE]\n\n"
                        return
            finally:
                # The generator is closed early when the client goes away.
                if not completed:
                    loop_.abort(sub.rid)

        if stream:  # StreamingResponse closes events() when the client goes away, and its finally aborts
            return StreamingResponse(events(), media_type="text/event-stream")

        async def collect():
            text, finish, n_out = "", None, 0
            all_lps: list[tuple[int, tuple[float, list[int], list[float]]]] = []
            while finish is None:
                kind, *rest = await sub.out.get()
                if kind == "error":
                    raise HTTPException(status_code=rest[1], detail=rest[0])
                ids, finish, lps = rest
                n_out += len(ids)
                if lps is not None:
                    all_lps.extend(zip(ids, lps))
                text += detok.push(ids, finish is not None)
            return text, finish, n_out, all_lps

        # Nothing cancels a plain response's handler when its client disconnects, so the request would hold its seat
        # until max_tokens: watch for the disconnect, and abort.
        try:
            done = await _unless_disconnected(http, collect())
        except asyncio.CancelledError:
            loop_.abort(sub.rid)
            raise
        if done is None:
            loop_.abort(sub.rid)
            return Response(status_code=499)  # nginx's "client closed request"; nobody reads it
        text, finish, n_out, all_lps = done
        if chat:
            msg = parse_full(text, **parse_args)
            finish = chat_finish_reason(finish, bool(msg.get("tool_calls")))
            choice = {"index": 0, "message": msg, "finish_reason": finish}
        else:
            choice = {"index": 0, "text": text, "finish_reason": finish}
        if params.logprobs is not None:
            choice["logprobs"] = lp_block(all_lps)
        return JSONResponse(
            {
                "id": oid,
                "object": obj,
                "created": created,
                "model": model_name,
                "choices": [choice],
                "usage": usage_of(n_out),
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
        return await _generate(http, body, ids, _params(body, 16, chat=False, vocab=loop_.vocab), chat=False)

    @app.post("/v1/chat/completions")
    async def chat(http: HTTPRequest):
        body = await _body(http)
        ids, parse = await asyncio.get_running_loop().run_in_executor(prompt_pool, chat_prompt, prompt_tok, body)
        # max_tokens defaults to the rest of the context (vLLM's default): reasoning at the template's default effort
        # (Max) can run long, and a smaller hidden default would cut the answer off
        room = max(engine.cfg.max_model_len - len(ids), 1)
        return await _generate(
            http, body, ids, _params(body, room, chat=True, vocab=loop_.vocab), chat=True, parse=parse
        )

    return app
