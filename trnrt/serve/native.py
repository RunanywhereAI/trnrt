"""SGLang's native HTTP API on one engine, beside the OpenAI API of serve/api.py (whose build_app adds these routes).

| Route | What it does |
|---|---|
| POST, PUT /generate | SGLang's GenerateReqInput: `text` or `input_ids`, one prompt or a batch, SGLang's `sampling_params`, `stream`, output logprobs; SGLang's response (`text`, `output_ids`, `meta_info`) |
| GET /model_info, /get_model_info | the model: its path as the operator passed it, served name, parsers, sampling defaults |
| GET /server_info, /get_server_info | the engine's configuration, its version and preset, and its load |
| GET /get_load | requests running and waiting, KV tokens in use (SGLang's legacy /get_load shape) |
| GET /health_generate | a one-token generation through the engine (SGLang's semantics: any step ending meanwhile counts) |
| POST /abort_request | abort a request by its `rid`, or every request (`abort_all`) |
| GET, POST /flush_cache | drop the prefix cache, only while the engine is idle (`?timeout=` seconds to wait for that) |

Shapes follow SGLang's main branch as of September 2026 (after v0.5.18): python/sglang/srt/managers/io_struct.py
(GenerateReqInput, AbortReq), python/sglang/srt/entrypoints/http_server.py, python/sglang/srt/managers/
tokenizer_manager.py (meta_info and its logprob tuples), python/sglang/srt/managers/schedule_batch.py (finish reasons) and
python/sglang/srt/sampling/sampling_params.py. Every route reaches the engine through EngineLoop's inbox (AGENTS.md:
the engine thread owns engine state). What the served graphs cannot compute (prompt logprobs, guided decoding, several
samples per prompt, multimodal input, LoRA) is refused with a 400 that names the field, as on the OpenAI routes.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import os
import time
from typing import TYPE_CHECKING, Any, cast

from fastapi import HTTPException
from fastapi import Request as HTTPRequest
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from .. import __version__
from ..engine.request import SamplingParams
from ..metrics import kv_pages
from .api import _NEUTRAL, _bad, _body, _Detok, _error, _flag, _int, _params, _unless_disconnected

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from fastapi import FastAPI

    from .api import EngineLoop, _Submit

logger = logging.getLogger(__name__)

# SGLang's SamplingParams.max_new_tokens default.
GENERATE_MAX_NEW_TOKENS = 128
# SGLang's HEALTH_CHECK_TIMEOUT default (http_server.py): how long /health_generate waits for a step to end.
HEALTH_GENERATE_TIMEOUT_S = 20.0
# SGLang's names for parsers equivalent to serve/parsers.py's (function_call_parser.py, reasoning_parser.py).
REASONING_PARSER, TOOL_CALL_PARSER = "glm45", "glm47"
# SGLang's /flush_cache answer on success (http_server.py).
FLUSHED = (
    "Cache flushed.\nPlease check backend logs for more details. (When there are running or waiting requests, the "
    "operation will not be performed.)\n"
)

# SGLang sampling_params -> the field api._params reads it from. The mapped fields _NEUTRAL lists are refused unless
# neutral, named here by their SGLang names.
_SAMPLING: dict[str, str] = {
    "max_new_tokens": "max_tokens",
    "stop": "stop",
    "stop_token_ids": "stop_token_ids",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "repetition_penalty": "repetition_penalty",
    "ignore_eos": "ignore_eos",
    "sampling_seed": "seed",
    "logit_bias": "logit_bias",
    "min_new_tokens": "min_tokens",
    "json_schema": "guided_json",
    "regex": "guided_regex",
    "ebnf": "guided_grammar",
}
# SGLang sampling parameters with no counterpart here, or not served on /generate (n choices, kept stop text and special
# tokens in the text are the OpenAI endpoints' Run path), accepted only at these values.
_SAMPLING_NEUTRAL: dict[str, tuple[object, ...]] = {
    "n": (None, 1),
    "no_stop_trim": (None, False),
    "skip_special_tokens": (None, True),
    "stop_regex": (None, [], ""),
    "structural_tag": (None,),
    "spaces_between_special_tokens": (None, True),
    "stream_interval": (None, 1),
    "beam_width": (None, 1),
    "custom_params": (None, {}),
}
# GenerateReqInput fields this server does not serve (multimodal input, LoRA, hidden states, disaggregation, priority
# scheduling, cache salts, ...), accepted only absent or at a neutral value. Fields that change nothing a client sees
# (conversation_id, a trace header) are accepted as unknown fields are on every route.
_GENERATE_REFUSED = (
    "input_embeds",
    "image_data",
    "video_data",
    "audio_data",
    "mm_hashes",
    "mm_content_hashes",
    "use_audio_in_video",
    "video_config",
    "modalities",
    "token_ids_logprob",
    "return_sampling_mask",
    "return_flat_raw_top_logprobs",
    "return_flat_raw_top_logprobs_b64",
    "return_hidden_states",
    "return_routed_experts",
    "routed_experts_start_len",
    "return_indexer_topk",
    "return_entropy",
    "return_bytes",
    "session_params",
    "lora_path",
    "lora_id",
    "custom_logit_processor",
    "positional_embed_overrides",
    "bootstrap_host",
    "bootstrap_port",
    "bootstrap_room",
    "bootstrap_pair_key",
    "decode_tp_size",
    "routed_dp_rank",
    "data_parallel_rank",
    "disagg_prefill_dp_rank",
    "routing_key",
    "background",
    "require_reasoning",
    "max_thinking_tokens",
    "priority",
    "extra_key",
    "cache_salt",
    "custom_labels",
    "multi_item_delimiter_indices",
    "max_dynamic_patch",
    "min_dynamic_patch",
    "image_max_dynamic_patch",
    "video_max_dynamic_patch",
    "images_config",
    "need_wait_for_mm_inputs",
    "num_items_assigned",
    "encoder_urls",
)
_UNSET: tuple[object, ...] = (None, False, 0, "", [], {})  # 0 == False: a dp rank or priority of 0 is the default


@dataclasses.dataclass
class _Item:
    """One prompt of a /generate request: its ids, sampling, rid (None: the server's) and logprob options."""

    ids: list[int]
    params: SamplingParams
    rid: str | None
    top: int  # top logprobs per output token (with return_logprob)
    logprob: bool


def _sampling(sp: Any, logprobs: int | None, vocab: int, defaults: dict[str, float]) -> SamplingParams:
    """SGLang sampling_params as SamplingParams, checked by api._params under SGLang's names. 400 for an unknown key
    (SGLang's SamplingParams(**sp) raises on one), and for a parameter not served at a value that is not neutral."""
    if sp is None:
        sp = {}
    if not isinstance(sp, dict):
        raise _bad("sampling_params must be an object")
    unknown = sorted(k for k in sp if k not in _SAMPLING and k not in _SAMPLING_NEUTRAL)
    if unknown:
        raise _bad(f"unknown sampling parameters: {', '.join(unknown)}")
    refused = [k for k, neutral in _SAMPLING_NEUTRAL.items() if sp.get(k) not in neutral]
    refused += [k for k, v in _SAMPLING.items() if v in _NEUTRAL and sp.get(k) not in _NEUTRAL[v]]
    if refused:
        raise _bad(f"not supported: {', '.join(refused)}")
    body: dict = {v: sp[k] for k, v in _SAMPLING.items() if k in sp}
    body["logprobs"] = logprobs
    try:
        return _params(body, GENERATE_MAX_NEW_TOKENS, chat=False, vocab=vocab, defaults=defaults)
    except HTTPException as e:  # its message names the field as _params reads it: name SGLang's
        detail = str(e.detail)
        for sg, ours in _SAMPLING.items():
            if sg != ours and detail.startswith(ours + " "):
                detail = sg + detail[len(ours) :]
                break
        raise _bad(detail) from None


def _per_prompt(body: dict, key: str, n: int, single: bool) -> list:
    """body[key] for each of n prompts: one value for all, or (a batch only) a list of n."""
    v = body.get(key)
    if isinstance(v, list):
        if single or len(v) != n:
            raise _bad(f"{key} must be one value, or a list of one per prompt of a batch ({n})")
        return v
    return [v] * n


def _prompts(body: dict, tok) -> tuple[list[list[int]], bool]:
    """The request's prompts as token ids, and whether it is one prompt (SGLang's is_single)."""
    text, ids = body.get("text"), body.get("input_ids")
    if (text is None) == (ids is None):
        raise _bad("exactly one of text or input_ids is required")
    if text is not None:
        if isinstance(text, str):
            return [list(tok(text)["input_ids"])], True
        if isinstance(text, list) and text and all(isinstance(t, str) for t in text):
            return [list(tok(t)["input_ids"]) for t in text], False
        raise _bad("text must be a string or a non-empty list of strings")
    if isinstance(ids, list) and ids and all(type(t) is int for t in ids):
        return [ids], True
    if isinstance(ids, list) and ids and all(isinstance(s, list) and s for s in ids):
        return ids, False  # each one's ids are checked by EngineLoop.submit
    raise _bad("input_ids must be a non-empty list of token ids, or a list of such lists")


def parse_generate(body: dict, tok, vocab: int, defaults: dict[str, float]) -> tuple[list[_Item], bool]:
    """A /generate request (SGLang's GenerateReqInput) as its prompts, and whether it is one prompt. 400 for anything
    malformed or not served (_GENERATE_REFUSED, _sampling, prompt logprobs). Tokenizes `text`: run it off the event
    loop."""
    refused = [k for k in _GENERATE_REFUSED if body.get(k) not in _UNSET]
    if refused:
        raise _bad(f"not supported: {', '.join(refused)}")
    seqs, single = _prompts(body, tok)
    n = len(seqs)
    rid = body.get("rid")
    if rid is None:
        rids: list[str | None] = [None] * n
    elif isinstance(rid, str) and rid:
        rids = [rid] if single else [f"{rid}_{i}" for i in range(n)]  # SGLang's batch rids (_normalize_rid)
    elif not single and isinstance(rid, list) and len(rid) == n and all(isinstance(r, str) and r for r in rid):
        rids = rid
    else:
        raise _bad("rid must be a non-empty string, or for a batch a list of one per prompt")
    if len({r for r in rids if r is not None}) < sum(r is not None for r in rids):
        raise _bad("the rids of a batch must differ")
    for key in ("stream", "return_text_in_logprobs", "return_prompt_token_ids", "log_metrics", "no_logs"):
        _flag(body, key)
    if body.get("session_id") is not None and not isinstance(body.get("session_id"), str):
        raise _bad("session_id must be a string")
    items = []
    for i, (ids, sp, want_v, start_v, top_v, r) in enumerate(
        zip(
            seqs,
            _per_prompt(body, "sampling_params", n, single),
            _per_prompt(body, "return_logprob", n, single),
            _per_prompt(body, "logprob_start_len", n, single),
            _per_prompt(body, "top_logprobs_num", n, single),
            rids,
        )
    ):
        one = {"return_logprob": want_v, "logprob_start_len": start_v, "top_logprobs_num": top_v}
        want = _flag(one, "return_logprob")
        start = _int(one, "logprob_start_len", -1, -1)
        top = _int(one, "top_logprobs_num", 0, 0, 20)  # the device returns at most 20 alternatives per token
        if want and start >= 0:
            raise _bad(
                f"logprob_start_len {start} asks for prompt logprobs (prompt {i}), which no served graph computes; "
                "send -1 for output logprobs only"
            )
        params = _sampling(sp, top if want else None, vocab, defaults)
        items.append(_Item(list(ids), params, r, top, want))
    return items, single


class _Run:
    """One prompt of a /generate request in the engine: its output as it arrives, and SGLang's view of it."""

    def __init__(self, item: _Item, sub: _Submit, tok, text_in_logprobs: bool, prompt_ids: bool, t0: float):
        self.item, self.sub, self.tok = item, sub, tok
        self.detok = _Detok(tok, item.params.stop, incremental=True, rid=sub.rid)
        self.text_in, self.prompt_ids, self.t0 = text_in_logprobs, prompt_ids, t0
        self.ids: list[int] = []
        # SGLang's logprob entries of the output so far, built once per token (a stream's events repeat them all)
        self.token_lps: list[list] = []
        self.top_lps: list[list[list]] = []
        self.text = ""
        self.finish: str | None = None
        self.req: Any = None  # the engine's Request, once accepted

    @property
    def rid(self) -> str:
        return self.sub.rid

    def push(self, ids: list[int], finish: str | None, lps) -> bool:
        """The engine's next tokens; whether anything changed."""
        if lps is not None:  # one per id (the last delivery of a finished request may have fewer)
            self.token_lps += [self._entry(lp, t) for t, (lp, _, _) in zip(ids, lps)]
            if self.item.top:
                self.top_lps += [[self._entry(v, i) for i, v in zip(ti, tl)] for _, ti, tl in lps]
        self.ids += ids
        delta = self.detok.push(ids, finish is not None)
        self.text += delta
        self.finish = finish
        return bool(ids or delta or finish)

    async def run(self) -> None:
        """Read the engine's tokens until the request finishes; HTTPException on an engine error."""
        while self.finish is None:
            kind, *rest = await cast("asyncio.Queue", self.sub.out).get()
            if kind == "error":
                raise _error(*rest)  # a displaced request's 503 keeps QUEUE_FULL_HEADER
            self.push(*rest)

    def _finish_reason(self) -> dict | None:
        """SGLang's finish reason (schedule_batch.py FINISH_*.to_json): a stop names what matched, the stop string or the
        token id (EOS or a stop_token_ids entry); a length, the tokens generated."""
        f = self.finish
        if f is None:
            return None
        if f == "length":
            return {"type": "length", "length": len(self.ids)}
        if f == "abort":
            return {"type": "abort", "message": "Aborted"}
        if f == "stop" and self.detok.stopped:
            full = self.tok.decode(self.ids, skip_special_tokens=True)
            hits = [(full.find(x), x) for x in self.item.params.stop if x in full]
            if hits:
                return {"type": "stop", "matched": min(hits)[1]}
        return {"type": f, "matched": self.ids[-1] if self.ids else None}

    def _entry(self, lp: float, t: int) -> list:
        return [lp, t, self.tok.decode([t]) if self.text_in else None]

    def result(self) -> dict:
        """SGLang's output so far (tokenizer_manager.py _handle_batch_output; a stream's events are cumulative, as with
        SGLang's default incremental_streaming_output off)."""
        req = self.req
        meta: dict = {
            "id": self.rid,
            "finish_reason": self._finish_reason(),
            "prompt_tokens": len(self.item.ids),
            "completion_tokens": len(self.ids),
            "cached_tokens": max(req.num_cached_tokens, 0),
            "num_retractions": req.num_preemptions,
        }
        if self.finish is not None:
            meta["e2e_latency"] = time.perf_counter() - self.t0
        if self.item.logprob:
            meta["input_token_logprobs"] = []  # output logprobs only: SGLang's logprob_start_len -1
            meta["output_token_logprobs"] = list(self.token_lps)
            if self.item.top:
                meta["input_top_logprobs"] = []
                meta["output_top_logprobs"] = list(self.top_lps)
        out = {"text": self.text, "output_ids": list(self.ids), "meta_info": meta}
        if self.prompt_ids:
            out["prompt_token_ids"] = self.item.ids
        return out


def _jsonable(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return str(v)  # a torch dtype


def _checkpoint_ids(path: str | None) -> tuple[str | None, list | None]:
    """model_type and architectures from the checkpoint's config.json (None when unreadable)."""
    if not path:
        return None, None
    try:
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None, None
    if not isinstance(cfg, dict):
        return None, None
    return cfg.get("model_type"), cfg.get("architectures")


def add_routes(
    app: FastAPI,
    loop_: EngineLoop,
    *,
    model_name: str,
    preset: str | None,
    prompt_pool: ThreadPoolExecutor,
    prompt_tok,
    gen_defaults: dict[str, float],
) -> None:
    """SGLang's native routes on the app of one engine (serve/api.py build_app)."""
    engine = loop_.engine
    tok = engine.tokenizer
    model_type, architectures = _checkpoint_ids(getattr(engine, "model_path", None))

    def load() -> dict:
        """SGLang's legacy /get_load entry for this engine (http_server.py get_load: num_tokens is the KV tokens in use
        plus the waiting requests' tokens, num_pending_tokens the latter). Read off the engine thread, as /v1/stats
        is: list() copies each queue whole while holding the GIL, so the engine thread's changes cannot interleave."""
        sched = engine.scheduler
        running, waiting = list(sched.running), list(sched.waiting)
        used, total = kv_pages(engine)
        pending = sum(r.num_tokens for r in waiting)
        return {
            "dp_rank": 0,
            "num_reqs": len(running) + len(waiting),
            "num_running_reqs": len(running),
            "num_waiting_reqs": len(waiting),
            "num_tokens": used * engine.cfg.page_size + pending,
            "num_pending_tokens": pending,
            "token_usage": round(used / total, 4) if total else 0.0,
            "ts_tic": time.perf_counter(),
        }

    @app.get("/get_load")
    async def get_load():
        return [load()]

    @app.get("/model_info")
    @app.get("/get_model_info")
    async def model_info():
        """SGLang's /model_info. model_path is the checkpoint as the operator passed it (a directory or a Hugging Face
        repo id), never the download cache it resolved to."""
        return {
            "model_path": engine.cfg.model_path,
            "tokenizer_path": engine.cfg.model_path,
            "served_model_name": model_name,
            "is_generation": True,
            "preferred_sampling_params": gen_defaults or None,
            "reasoning_parser": REASONING_PARSER,
            "tool_call_parser": TOOL_CALL_PARSER,
            "has_image_understanding": False,
            "has_audio_understanding": False,
            "model_type": model_type,
            "architectures": architectures,
        }

    @app.get("/server_info")
    @app.get("/get_server_info")
    async def server_info():
        """SGLang's /server_info: the engine's configuration (EngineConfig: shapes, pools and routes, all from the
        command line; it holds no credential), SGLang's names for the ones its clients read, the version and preset,
        and the load (internal_states)."""
        cfg = engine.cfg
        fields = (
            {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
            if dataclasses.is_dataclass(cfg)
            else dict(vars(cfg))
        )
        return {
            **{k: _jsonable(v) for k, v in fields.items()},
            "served_model_name": model_name,
            "tp_size": cfg.tp,
            "dp_size": 1,
            "context_length": cfg.max_model_len,
            "preset": preset,
            "version": __version__,
            "startup_seconds": getattr(engine, "startup_seconds", None),
            "internal_states": [load()],
        }

    @app.get("/health_generate")
    async def health_generate():
        """SGLang's /health_generate: one token generated from prompt [0] (greedy, not counted in /metrics). A step that
        ends after the probe is sent shows the engine serving, so a busy engine answers as soon as its current step ends
        (and the probe is aborted); 503 if no step ends within HEALTH_GENERATE_TIMEOUT_S, or once the engine failed."""
        tic = time.monotonic()
        err = loop_.health_error()
        if err is None:
            try:
                sub = loop_.submit([0], SamplingParams(max_new_tokens=1, temperature=0.0), metrics=False)
            except HTTPException as e:  # the engine failed meanwhile
                err = str(e.detail)
        if err is not None:
            return JSONResponse(status_code=503, content={"status": "failed", "error": err})

        async def finished() -> str | None:
            while True:
                kind, *rest = await cast("asyncio.Queue", sub.out).get()
                if kind == "error":
                    return str(rest[0])
                if kind == "tokens" and rest[1] is not None:
                    return None

        done = asyncio.ensure_future(finished())
        try:
            while not done.done() and loop_.last_step <= tic and time.monotonic() < tic + HEALTH_GENERATE_TIMEOUT_S:
                await asyncio.wait({done}, timeout=0.05)
            if done.done():
                err = done.result()
            elif loop_.last_step <= tic:
                err = f"no engine step ended within {HEALTH_GENERATE_TIMEOUT_S:.0f} s"
        finally:
            if not done.done():
                done.cancel()
                loop_.abort(sub.rid)
        err = loop_.health_error() or err
        if err is not None:
            return JSONResponse(status_code=503, content={"status": "failed", "error": err})
        return {"status": "ok"}

    @app.post("/abort_request")
    async def abort_request(http: HTTPRequest):
        """SGLang's AbortReq: `rid` (exact: a batch's rids are rid_0, rid_1, ...) or `abort_all`. An aborted request
        answers with what it generated and finish_reason {"type": "abort"}. 200 whether or not the rid is running, as
        SGLang."""
        body = await _body(http)
        for key in ("finished_reason", "abort_message"):
            if body.get(key) is not None:
                raise _bad(f"not supported: {key}")
        if _flag(body, "abort_all"):
            loop_.abort_all()
            return Response(status_code=200)
        rid = body.get("rid")
        if not (isinstance(rid, str) and rid):
            raise _bad("rid (a non-empty string) is required unless abort_all is true")
        loop_.abort(rid)
        return Response(status_code=200)

    @app.api_route("/flush_cache", methods=["GET", "POST"])
    async def flush_cache(http: HTTPRequest):
        """SGLang's /flush_cache: the prefix cache (device pages, state checkpoints and the host tier) dropped, only
        while no request runs or waits; `timeout` (seconds, default 0) waits that long for the engine to be idle. 200
        with SGLang's text when flushed, 400 with the reason when not."""
        raw = http.query_params.get("timeout", "0")
        try:
            timeout = float(raw)
        except ValueError:
            timeout = math.nan
        if not (math.isfinite(timeout) and timeout >= 0):
            raise _bad("timeout must be a number of seconds >= 0")
        f = loop_.flush(timeout)
        flushed, message = await f.fut
        if flushed:
            return PlainTextResponse(FLUSHED)
        return PlainTextResponse(message + "\n", status_code=400)

    @app.api_route("/generate", methods=["POST", "PUT"])
    async def generate(http: HTTPRequest):
        body = await _body(http)
        items, single = await asyncio.get_running_loop().run_in_executor(
            prompt_pool, parse_generate, body, prompt_tok, loop_.vocab, gen_defaults
        )
        t0 = time.perf_counter()
        subs: list[_Submit] = []
        log_metrics = _flag(body, "log_metrics", True)
        try:  # extend() keeps the requests queued before one raised: they are aborted
            subs.extend(
                loop_.submit(it.ids, it.params, body.get("session_id"), rid=it.rid, metrics=log_metrics) for it in items
            )
        except HTTPException:
            for s in subs:
                loop_.abort(s.rid)
            raise
        # Admission errors (a prompt that does not fit, a rid in use) arrive before any token: an HTTP status, not a
        # 200 stream, and the batch's other prompts are aborted.
        # /generate's requests own their queues (api._Submit.out is a _Tagged view only for `n` > 1 choices)
        firsts = [await cast("asyncio.Queue", s.out).get() for s in subs]
        bad = next((f for f in firsts if f[0] == "error"), None)
        if bad is not None:
            for s, f in zip(subs, firsts):
                if f[0] == "accepted":
                    loop_.abort(s.rid)
            raise _error(*bad[1:])  # a full queue's 503 keeps QUEUE_FULL_HEADER: the router keeps the engine up
        text_in, prompt_ids = _flag(body, "return_text_in_logprobs"), _flag(body, "return_prompt_token_ids")
        runs = [_Run(it, s, tok, text_in, prompt_ids, t0) for it, s in zip(items, subs)]
        for r, f in zip(runs, firsts):
            r.req = f[1]

        def abort_unfinished() -> None:
            for r in runs:
                if r.finish is None:
                    loop_.abort(r.rid)

        if _flag(body, "stream"):
            return StreamingResponse(_events(runs, single, abort_unfinished), media_type="text/event-stream")

        async def collect() -> bool:
            await asyncio.gather(*(r.run() for r in runs))
            return True

        try:  # a client that goes away aborts its requests (api._generate)
            done = await _unless_disconnected(http, collect())
        finally:
            abort_unfinished()
        if done is None:
            return Response(status_code=499)  # nginx's "client closed request"; nobody reads it
        return runs[0].result() if single else [r.result() for r in runs]


async def _events(runs: list[_Run], single: bool, abort_unfinished):
    """A /generate stream: SGLang's server-sent events, one per prompt that has new output after a step (cumulative
    text, output_ids and meta_info; a batch's carry its prompt's `index`), then `data: [DONE]`. An engine error ends the
    stream with SGLang's error event; a client that goes away closes this generator, which aborts what is left."""
    merged: asyncio.Queue = asyncio.Queue()

    async def pump(i: int, r: _Run) -> None:
        while True:
            m = await cast("asyncio.Queue", r.sub.out).get()
            merged.put_nowait((i, m))
            if m[0] == "error" or m[2] is not None:
                return

    pumps = [asyncio.ensure_future(pump(i, r)) for i, r in enumerate(runs)]
    left = len(runs)
    try:
        while left:
            i, (kind, *rest) = await merged.get()
            if kind == "error":
                err = {"message": rest[0], "type": "server_error", "code": rest[1], "retryable": False}
                yield f"data: {json.dumps({'error': err})}\n\n"
                break
            r = runs[i]
            if r.push(*rest):
                out = r.result()
                if not single:
                    out["index"] = i
                yield f"data: {json.dumps(out)}\n\n"
            left -= r.finish is not None
        yield "data: [DONE]\n\n"
    finally:
        for p in pumps:
            p.cancel()
        abort_unfinished()
