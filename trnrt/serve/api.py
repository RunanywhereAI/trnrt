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
import copy
import json
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
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request as HTTPRequest
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from ..engine.request import Request, SamplingParams, Status
from ..metrics import Metrics, kv_pages
from .parsers import THINK_OPEN, StreamParser, chat_finish_reason, parse_full, template_messages, tool_choice_error

if TYPE_CHECKING:  # only annotations: importing this module does not load the engine and its model code
    from ..engine.engine import LLMEngine


@dataclass
class _Stream:
    """A request the engine runs: its tokens go to `out` on the event loop `loop`; `sent` of them so far."""
    req: Request
    loop: asyncio.AbstractEventLoop
    out: asyncio.Queue
    sent: int = 0


@dataclass
class _Submit:
    """A request on its way to the engine thread. `out` receives, in order: ("accepted", Request), or ("error", why)
    if the engine refused it (400); then ("tokens", new ids, finish reason or None, their logprobs or None) until the
    finish reason is set, or ("error", why) if the engine fails."""
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
        self.inbox: queue.Queue = queue.Queue()
        self.streams: dict[str, _Stream] = {}
        # Why the engine can no longer serve (a rank or this loop failed), None while it can: /health and every new
        # request answer 503 from then on. _gate orders setting it against queueing a request, so none goes unanswered.
        self.failed: str | None = None
        self._gate = threading.Lock()
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
                print(f"trnrt: the engine thread did not stop within {wait:.0f} s; closing the engine anyway",
                      flush=True)

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
        if not prompt_ids or not all(isinstance(t, int) for t in prompt_ids):
            raise HTTPException(status_code=400, detail="prompt must be a non-empty list of token ids")
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
                    req = self.engine.add_request(payload.prompt_ids, payload.params, rid=payload.rid,
                                                  session_id=payload.session_id)
                except ValueError as e:
                    post(payload.out.put_nowait, ("error", str(e)))
                except Exception as e:  # noqa: BLE001  add_request raised before the request reached the scheduler
                    traceback.print_exc()  # (its last step): that request fails, the engine keeps serving
                    post(payload.out.put_nowait, ("error", f"request refused: {e!r}"))
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
        except BaseException as e:  # outside a step (a step's own failure is _loop's): nothing would serve requests
            traceback.print_exc()
            self._fail(f"the engine loop failed: {e!r}")

    def _watch_ranks(self) -> None:
        """A rank that dies while the loop waits in a step leaves no thread to notice it: a collective with a dead peer
        can block instead of raising (gloo on macOS does until its timeout; a device collective too). The loop's own
        check between steps gets the first second."""
        seen = None
        while not self._stopping.wait(0.5) and self.failed is None:
            dead = self.engine.dead_ranks()
            if not dead:
                seen = None
            elif seen is None:
                seen = time.monotonic()
            elif time.monotonic() - seen > 1.0:
                self._fail(f"tensor-parallel rank(s) {dead} exited")
                return

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
        print(f"trnrt: {reason}; /health answers 503 and the engine shuts down", flush=True)
        streams, self.streams = self.streams, {}
        for st in streams.values():
            st.loop.call_soon_threadsafe(st.out.put_nowait, ("error", f"engine failed: {reason}"))
        while True:
            try:
                kind, payload = self.inbox.get_nowait()
            except queue.Empty:
                break
            if kind == "add":
                payload.loop.call_soon_threadsafe(payload.out.put_nowait, ("error", f"engine failed: {reason}"))
        try:
            self.engine.close_failed()
        except Exception:  # noqa: BLE001  the requests are answered; a teardown error must not hide the reason
            traceback.print_exc()

    def _loop(self) -> None:
        while not self._stopping.is_set() and self.failed is None:
            dead = self.engine.dead_ranks()
            if dead:
                self._fail(f"tensor-parallel rank(s) {dead} exited")
                return
            self._drain(block=not self.engine.has_work())
            if not self.engine.has_work():
                continue
            try:
                self.engine.step()
            except Exception as e:  # never let the engine thread die silently
                traceback.print_exc()
                if self.engine.cfg.tp > 1:  # rank 0 left a step its workers ran or wait in: the ranks are out of step
                    dead = self.engine.dead_ranks()
                    gone = f"; tensor-parallel rank(s) {dead} exited" if dead else ""
                    self._fail(f"engine step failed: {e!r}{gone}")
                    return
                for st in self.streams.values():
                    st.loop.call_soon_threadsafe(st.out.put_nowait, ("error", f"engine step failed: {e}"))
                    if st.req.status is not Status.FINISHED:
                        self.engine.abort(st.req)
                self.streams.clear()
                continue
            for rid, st in list(self.streams.items()):
                req = st.req
                out = req.output_ids
                finished = req.status is Status.FINISHED
                with_lp = req.params.logprobs is not None
                # With logprobs on, only the tokens whose logprobs are recorded go out, so every token carries its own.
                ready = len(out) if finished or not with_lp else min(len(out), len(req.logprobs))
                if ready > st.sent or finished:
                    new = out[st.sent:ready]
                    lps = req.logprobs[st.sent:ready] if with_lp else None
                    st.sent = ready
                    st.loop.call_soon_threadsafe(st.out.put_nowait,
                                                 ("tokens", new, req.finish_reason if finished else None, lps))
                if finished:
                    del self.streams[rid]
                    self.metrics.on_finish(req)


class _Detok:
    """Incremental detokenizer for streaming.

    Never emits half of a multi-byte character, and with stop strings it holds back the
    last `max(len(stop)) - 1` characters until they cannot start a stop string, so a stop
    sequence is never partially streamed. Output is cut before the first stop string, as
    the OpenAI API specifies.
    """

    def __init__(self, tokenizer, stops: tuple[str, ...] = ()):
        self.tok = tokenizer
        self.stops = stops
        self.hold = max((len(x) for x in stops), default=1) - 1
        self.ids: list[int] = []
        self.emitted = 0
        self.stopped = False

    def push(self, ids: list[int], final: bool) -> str:
        if self.stopped:
            return ""
        self.ids.extend(ids)
        text = self.tok.decode(self.ids, skip_special_tokens=True)
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
        delta = text[self.emitted:end]
        self.emitted = end
        return delta


def _params(body: dict, default_max: int, chat: bool) -> SamplingParams:
    stop = body.get("stop") or ()
    stop = (stop,) if isinstance(stop, str) else tuple(stop)
    if chat:  # chat: `logprobs` turns them on, `top_logprobs` asks for alternatives
        logprobs = int(body.get("top_logprobs") or 0) if body.get("logprobs") else None
    else:  # completions: `logprobs` is the number of alternatives
        logprobs = int(body["logprobs"]) if body.get("logprobs") is not None else None
    if logprobs is not None and not 0 <= logprobs <= 20:  # the device returns at most 20 alternatives per token
        raise HTTPException(status_code=400, detail="logprobs / top_logprobs must be between 0 and 20")
    budget = body.get("thinking_token_budget")  # vLLM's: the reasoning is closed after this many tokens
    return SamplingParams(
        max_new_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or default_max),
        temperature=float(body.get("temperature", 1.0)),
        top_p=float(body.get("top_p", 1.0)),
        top_k=int(body.get("top_k", 0) or 0),
        min_p=float(body.get("min_p", 0.0) or 0.0),
        repetition_penalty=float(body.get("repetition_penalty", 1.0) or 1.0),
        frequency_penalty=float(body.get("frequency_penalty", 0.0) or 0.0),
        presence_penalty=float(body.get("presence_penalty", 0.0) or 0.0),
        logprobs=logprobs,
        thinking_token_budget=int(budget) if budget is not None else None,
        stop=stop,
        stop_token_ids=tuple(body.get("stop_token_ids") or ()),
        ignore_eos=bool(body.get("ignore_eos", False)),
        seed=body.get("seed"),
    )


# A role test naming "developer" in a Jinja template: `m.role == 'developer'`, `message['role'] != "developer"`,
# `role in ['system', 'developer']`. The word in prose or in an emitted string is not one.
_DEVELOPER_ROLE = re.compile(r"""role['"\]]*\s*(?:==|!=|in)\s*[\[(]?[^\])\n%}]*?['"]developer['"]""")


def template_knows_developer(tok) -> bool:
    """Whether the tokenizer's chat template (one, or a dict of named ones) has a branch for role "developer"."""
    t = getattr(tok, "chat_template", None)
    return any(_DEVELOPER_ROLE.search(x) for x in ([t] if isinstance(t, str) else (t or {}).values())
               if isinstance(x, str))


def _template_ids(tok, messages: list, kwargs: dict) -> list[int]:
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kwargs)
    if hasattr(ids, "keys"):  # transformers 5 returns a BatchEncoding, not a list
        ids = ids["input_ids"]
    return list(ids)


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
    kwargs = dict(body.get("chat_template_kwargs") or {})
    tool_choice = body.get("tool_choice")
    error = tool_choice_error(tool_choice, body.get("tools"))
    if error:
        raise HTTPException(status_code=400, detail=error)
    tools = body.get("tools") if tool_choice != "none" else None
    if tools:
        kwargs["tools"] = tools
    ids = _template_ids(tok, messages, kwargs)
    # With thinking on, the template opens the reasoning block in the prompt and the output starts inside it; with
    # thinking off it closes the block there, and the output starts with the answer.
    tail = tok.decode(ids[-4:]).rstrip()
    # parallel_tool_calls=False keeps only the first call (vLLM maybe_filter_parallel_tool_calls,
    # entrypoints/serve/utils/tool_calls_utils.py).
    parse = dict(tools=bool(tools), thinking_open=tail.endswith(THINK_OPEN), tool_defs=tools,
                 parallel_tool_calls=body.get("parallel_tool_calls") is not False)
    return ids, parse


def build_app(engine: LLMEngine, model_name: str) -> FastAPI:
    app = FastAPI(title="trnRT")
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
        return {"gauges": {"kv_pages_used": used, "kv_pages_total": total}}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(loop_.metrics.render(engine), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "trnrt"}]}

    async def _generate(body: dict, prompt_ids: list[int], params: SamplingParams, chat: bool,
                        parse: dict | None = None):
        stream = bool(body.get("stream"))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        tool_choice = body.get("tool_choice")
        sub = loop_.submit(prompt_ids, params, body.get("session_id"))
        # Admission errors (a prompt that does not fit, an unsupported parameter) arrive before any token: surface
        # them as HTTP status codes rather than inside a 200 stream.
        first = await sub.out.get()
        if first[0] == "error":
            raise HTTPException(status_code=400, detail=first[1])
        req = first[1]
        created = int(time.time())
        oid = ("chatcmpl-" if chat else "cmpl-") + sub.rid
        obj = "chat.completion" if chat else "text_completion"
        detok = _Detok(tok, params.stop)

        def usage_of(n_out: int) -> dict:
            """OpenAI usage, with prompt_tokens_details.cached_tokens: the prompt tokens this request resumed from the
            prefix cache (0 cold). Final once the request has finished."""
            return {"prompt_tokens": len(prompt_ids), "completion_tokens": n_out,
                    "total_tokens": len(prompt_ids) + n_out,
                    "prompt_tokens_details": {"cached_tokens": max(req.num_cached_tokens, 0)}}

        def lp_block(lps):
            """OpenAI logprobs from (token id, (logprob, top ids, top logprobs)) pairs."""
            if lps is None:
                return None
            if chat:
                def entry(t, lp):
                    s = tok.decode([t])
                    return {"token": s, "logprob": lp, "bytes": list(s.encode())}
                return {"content": [dict(entry(t, lp), top_logprobs=[entry(i, v) for i, v in zip(ti, tl)])
                                    for t, (lp, ti, tl) in lps]}
            return {"tokens": [tok.decode([t]) for t, _ in lps],
                    "token_logprobs": [lp for _, (lp, _, _) in lps],
                    "top_logprobs": [{tok.decode([i]): v for i, v in zip(ti, tl)} for _, (_, ti, tl) in lps]}

        def chunk(text: str, finish: str | None, lps=None) -> dict:
            if chat:
                choice = {"index": 0, "delta": {"content": text} if text else {}, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": text, "finish_reason": finish}
            if lps is not None:
                choice["logprobs"] = lp_block(lps)
            return {"id": oid, "object": obj + (".chunk" if chat else ""), "created": created,
                    "model": model_name, "choices": [choice]}

        sparser = StreamParser(**parse) if chat else None

        def chat_chunks(text: str, finish: str | None, pairs):
            deltas = sparser.push(text, finish is not None)
            finish = chat_finish_reason(finish, sparser.saw_tool, tool_choice)
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
                while True:
                    kind, *rest = await sub.out.get()
                    if kind == "error":
                        completed = True
                        yield f"data: {json.dumps({'error': {'message': rest[0]}})}\n\n"
                        return
                    ids, finish, lps = rest
                    n_streamed += len(ids)
                    text = detok.push(ids, finish is not None)
                    pairs = list(zip(ids, lps)) if lps is not None else None
                    if text or finish or pairs:
                        for c in (chat_chunks(text, finish, pairs) if chat else [chunk(text, finish, pairs)]):
                            yield f"data: {json.dumps(c)}\n\n"
                    if finish is not None:
                        completed = True
                        if include_usage:  # OpenAI stream_options.include_usage: one last chunk, no choices
                            yield ("data: " + json.dumps({"id": oid, "object": obj + (".chunk" if chat else ""),
                                                          "created": created, "model": model_name, "choices": [],
                                                          "usage": usage_of(n_streamed)}) + "\n\n")
                        yield "data: [DONE]\n\n"
                        return
            finally:
                # The generator is closed early when the client goes away.
                if not completed:
                    loop_.abort(sub.rid)

        if stream:
            return StreamingResponse(events(), media_type="text/event-stream")

        text, finish, n_out, all_lps = "", None, 0, []
        try:
            while finish is None:
                kind, *rest = await sub.out.get()
                if kind == "error":
                    raise HTTPException(status_code=400, detail=rest[0])
                ids, finish, lps = rest
                n_out += len(ids)
                if lps is not None:
                    all_lps.extend(zip(ids, lps))
                text += detok.push(ids, finish is not None)
        except asyncio.CancelledError:
            loop_.abort(sub.rid)
            raise
        if chat:
            msg = parse_full(text, **parse)
            finish = chat_finish_reason(finish, bool(msg.get("tool_calls")), tool_choice)
            choice = {"index": 0, "message": msg, "finish_reason": finish}
        else:
            choice = {"index": 0, "text": text, "finish_reason": finish}
        if params.logprobs is not None:
            choice["logprobs"] = lp_block(all_lps)
        return JSONResponse({"id": oid, "object": obj, "created": created, "model": model_name,
                             "choices": [choice], "usage": usage_of(n_out)})

    @app.post("/v1/completions")
    async def completions(http: HTTPRequest):
        body = await http.json()
        prompt = body.get("prompt", "")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            ids = prompt
        elif isinstance(prompt, str):
            ids = tok(prompt)["input_ids"]
        else:
            raise HTTPException(status_code=400, detail="prompt must be a string or a list of token ids")
        return await _generate(body, ids, _params(body, 16, chat=False), chat=False)

    @app.post("/v1/chat/completions")
    async def chat(http: HTTPRequest):
        body = await http.json()
        ids, parse = await asyncio.get_running_loop().run_in_executor(prompt_pool, chat_prompt, prompt_tok, body)
        return await _generate(body, ids, _params(body, 512, chat=True), chat=True, parse=parse)

    return app
