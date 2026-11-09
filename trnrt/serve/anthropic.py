"""The Anthropic Messages API, `POST /v1/messages` and `POST /v1/messages/count_tokens`, served by the chat path
(serve/api.py).

A request is translated into a /v1/chat/completions body, and its output, parsed by the same GLM parsers as a chat
response, back into Anthropic content blocks:

| Messages API | Chat path |
|---|---|
| `system` (a string or text blocks) | a leading system message |
| `text` blocks | message content |
| an assistant's `thinking` blocks | its `reasoning_content` (GLM's template replays it) |
| an assistant's `tool_use` blocks | its `tool_calls` |
| a user's `tool_result` blocks | `tool` messages, in their place among the user's text |
| `tools` (`name`, `description`, `input_schema`) | function tools |
| `tool_choice` `auto` / `none`; `disable_parallel_tool_use` | `tool_choice` `auto` / `none`; `parallel_tool_calls` false |
| `max_tokens` (required), `temperature`, `top_p`, `top_k`, `stop_sequences` | the same sampling fields; `stop` |
| `thinking` `{"type": "disabled"}` / `{"type": "enabled", "budget_tokens": N}` | `enable_thinking` false / `thinking_token_budget` N, served by an engine started with `--structured-outputs` and refused otherwise |
| `output_config.effort` | `reasoning_effort` |
| output `reasoning_content`, `content`, `tool_calls` | `thinking`, `text` and `tool_use` blocks, in that order |

GLM-5.3-Flash always reasons (its template has no switch), so its reasoning comes back as a `thinking` block whether or
not the request set `thinking`, with an empty `signature` (nothing here signs it). `tool_choice` `any` and `tool` need
guided decoding and are refused, as the chat path refuses `required` and a named tool. Usage follows Anthropic's split:
`input_tokens` are the prompt tokens not read from the prefix cache, `cache_read_input_tokens` those that were.

Errors use Anthropic's shape, `{"type": "error", "error": {"type": ..., "message": ...}}`. The translation follows
SGLang's Anthropic adapter (python/sglang/srt/entrypoints/anthropic/serving.py), except that SGLang accepts and ignores a
thinking budget, which this server enforces (with `--structured-outputs`) or refuses, never ignores (AGENTS.md).
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from fastapi import Request as HTTPRequest
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .api import EngineError, Run

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# Anthropic's error types by HTTP status (https://docs.anthropic.com/en/api/errors).
ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    503: "overloaded_error",
    529: "overloaded_error",
}
# Request fields that change the output in a way this server cannot serve: refused unless absent or null.
_UNSUPPORTED = ("mcp_servers", "container", "context_management")


def _bad(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def error_body(status: int, detail) -> dict:
    return {
        "type": "error",
        "error": {"type": ERROR_TYPES.get(status, "api_error"), "message": str(detail)},
    }


def error_response(e: HTTPException) -> JSONResponse:
    """An HTTPException in Anthropic's error shape, at its status and with its headers (a full queue's 503 keeps
    api.QUEUE_FULL_HEADER, so the router does not take the engine for failed)."""
    return JSONResponse(error_body(e.status_code, e.detail), status_code=e.status_code, headers=e.headers)


def _text_of(content, where: str) -> str:
    """A string, or text blocks joined (as Anthropic concatenates a turn's text blocks); 400 for any other block."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _bad(f"{where} must be a string or a list of text blocks")
    out = []
    for b in content:
        if not (isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)):
            kind = b.get("type") if isinstance(b, dict) else type(b).__name__
            raise _bad(f"{where}: block type {kind!r} is not supported here; this model reads text only")
        out.append(b["text"])
    return "".join(out)


def _system(system) -> str | None:
    if system is None:
        return None
    return _text_of(system, "system")


def messages(body: dict) -> list[dict]:
    """The chat messages of a request's system and messages."""
    out: list[dict] = []
    sys = _system(body.get("system"))
    if sys:
        out.append({"role": "system", "content": sys})
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise _bad("messages must be a non-empty list")
    for k, m in enumerate(msgs):
        where = f"messages[{k}]"
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            raise _bad(f"{where}: role must be user or assistant")
        role, content = m["role"], m.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise _bad(f"{where}: content must be a string or a list of blocks")
        text: list[str] = []
        thinking: list[str] = []
        calls: list[dict] = []
        for j, b in enumerate(content):
            at = f"{where}.content[{j}]"
            kind = b.get("type") if isinstance(b, dict) else None
            if kind == "text" and isinstance(b.get("text"), str):
                text.append(b["text"])
            elif kind == "thinking" and role == "assistant":
                if isinstance(b.get("thinking"), str):
                    thinking.append(b["thinking"])
            elif kind == "tool_use" and role == "assistant":
                name, args = b.get("name"), b.get("input", {})
                if not isinstance(name, str) or not name or not isinstance(args, dict):
                    raise _bad(f"{at}: tool_use needs a name and an input object")
                calls.append(
                    {
                        "id": b.get("id") or "call_" + uuid.uuid4().hex[:24],
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                    }
                )
            elif kind == "tool_result" and role == "user":
                if text:  # the user's text before the result stays before it
                    out.append({"role": "user", "content": "".join(text)})
                    text = []
                result = b.get("content", "")
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": b.get("tool_use_id"),
                        "content": result if isinstance(result, str) else _text_of(result, at),
                    }
                )
            else:
                raise _bad(f"{at}: block type {kind!r} is not supported in a {role} message")
        if role == "assistant":
            msg: dict = {"role": "assistant", "content": "".join(text)}
            if thinking:
                msg["reasoning_content"] = "\n".join(thinking)
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif text:
            out.append({"role": "user", "content": "".join(text)})
    return out


def _tools(tools) -> list[dict] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise _bad("tools must be a list")
    out = []
    for t in tools:
        ok = isinstance(t, dict) and isinstance(t.get("name"), str) and isinstance(t.get("input_schema"), dict)
        if not ok or t.get("type", "custom") != "custom":
            kind = t.get("type") if isinstance(t, dict) else None
            raise _bad(f"tool {kind!r} is not supported: only custom tools with a name and an input_schema")
        fn = {"name": t["name"], "parameters": t["input_schema"]}
        if t.get("description") is not None:
            fn["description"] = t["description"]
        out.append({"type": "function", "function": fn})
    return out


def to_chat(body: dict) -> dict:
    """The /v1/chat/completions body for a Messages request (count_tokens takes the same, without max_tokens). 400 for
    what cannot be served."""
    for key in _UNSUPPORTED:
        if body.get(key) not in (None, []):
            raise _bad(f"{key} is not supported")
    chat: dict[str, Any] = {"messages": messages(body)}
    tools = _tools(body.get("tools"))
    if tools:
        chat["tools"] = tools
    tc = body.get("tool_choice")
    if tc is not None:
        kind = tc.get("type") if isinstance(tc, dict) else None
        if kind == "auto" or kind == "none":
            chat["tool_choice"] = kind
        elif kind == "any":
            chat["tool_choice"] = "required"  # refused by the chat path: it needs guided decoding
        elif kind == "tool":
            chat["tool_choice"] = {"type": "function", "function": {"name": tc.get("name")}}  # refused, as above
        else:
            raise _bad('tool_choice type must be "auto", "any", "tool" or "none"')
        if tc.get("disable_parallel_tool_use") is True:
            chat["parallel_tool_calls"] = False
    thinking = body.get("thinking")
    if thinking is not None:
        kind = thinking.get("type") if isinstance(thinking, dict) else None
        if kind == "disabled":
            chat["chat_template_kwargs"] = {"enable_thinking": False}
        elif kind == "enabled":
            budget = thinking.get("budget_tokens")
            if type(budget) is not int:
                raise _bad("thinking.budget_tokens is required with thinking enabled")
            chat["thinking_token_budget"] = budget  # the engine refuses any budget, with its reason
        elif kind != "adaptive":
            raise _bad('thinking type must be "enabled", "disabled" or "adaptive"')
    oc = body.get("output_config")
    if oc is not None:
        if not isinstance(oc, dict) or set(oc) - {"effort"}:
            raise _bad("output_config supports only effort")
        if oc.get("effort") is not None:
            chat["reasoning_effort"] = oc["effort"]
    for src, dst in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("stop_sequences", "stop"),
    ):
        if body.get(src) is not None:
            chat[dst] = body[src]
    if body.get("session_id") is not None:  # the router's and the engine's session affinity
        chat["session_id"] = body["session_id"]
    return chat


def _usage(run: Run) -> dict:
    u = run.usage()
    cached = u["prompt_tokens_details"]["cached_tokens"]
    return {
        "input_tokens": u["prompt_tokens"] - cached,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
        "output_tokens": u["completion_tokens"],
    }


def _stop(run: Run, finish: str | None, stops: list[str]) -> tuple[str | None, str | None]:
    """Anthropic's stop_reason and stop_sequence for a finished chat choice."""
    if finish == "length":
        return "max_tokens", None
    if finish == "tool_calls":
        return "tool_use", None
    matched = run.matched_stop(0)
    if isinstance(matched, str) and matched in stops:
        return "stop_sequence", matched
    return "end_turn", None


def content_blocks(msg: dict) -> list[dict]:
    """A parsed chat message (parsers.parse_full) as Anthropic content blocks: thinking, text, then tool_use."""
    out: list[dict] = []
    if msg.get("reasoning_content"):
        out.append({"type": "thinking", "thinking": msg["reasoning_content"], "signature": ""})
    if msg.get("content"):
        out.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or ():
        fn = tc["function"]
        try:
            args = json.loads(fn["arguments"]) if fn["arguments"] else {}
        except ValueError:  # a call cut off by max_tokens: its arguments as written, under one key
            args = {"_raw": fn["arguments"]}
        out.append({"type": "tool_use", "id": tc["id"], "name": fn["name"], "input": args})
    return out


def _check(body: dict) -> list[str]:
    """The checks of a Messages request beyond the translation: max_tokens is required; stream is a boolean. Returns
    stop_sequences."""
    mt = body.get("max_tokens")
    if type(mt) is not int or mt < 1:
        raise _bad("max_tokens is required: an integer of at least 1")
    stream = body.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise _bad("stream must be true or false")
    stops = body.get("stop_sequences") or []
    return stops if isinstance(stops, list) else []


async def respond(http: HTTPRequest, body: dict, start_chat: Callable[[dict], Awaitable[Run]], model: str) -> Response:
    """POST /v1/messages: the chat path's run of the translated request, answered as a Message or its events."""
    stops = _check(body)
    chat = to_chat(body)
    chat["max_tokens"] = body["max_tokens"]
    run = await start_chat(chat)
    if body.get("stream"):
        return StreamingResponse(_events(run, stops, model), media_type="text/event-stream")
    try:
        if not await run.wait(http):
            return Response(status_code=499)  # the client went away; nobody reads it
    except EngineError as e:
        raise e.http() from None
    msg, finish = run.message(0)
    reason, seq = _stop(run, finish, stops)
    return JSONResponse(
        {
            "id": "msg_" + run.rid,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content_blocks(msg),
            "stop_reason": reason,
            "stop_sequence": seq,
            "usage": _usage(run),
        }
    )


def _ev(kind: str, **fields) -> str:
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **fields})}\n\n"


async def _events(run: Run, stops: list[str], model: str):
    """Anthropic's stream: message_start, then per block content_block_start, its deltas and content_block_stop, then
    message_delta (stop_reason and the final usage) and message_stop. message_start waits for the first delivery: by
    then the engine has scheduled the request, so its prefix-cache hit (cache_read_input_tokens) is known."""
    index = -1
    open_kind: str | None = None  # the open block's kind: thinking, text or tool_use
    call_index: int | None = None  # the open tool_use block's call (the parser's index)
    started = False
    finish = None
    space = ""  # whitespace-only text not yet in a text block
    try:
        async for _, deltas, fin, _ in run.parsed():
            if not started:
                started = True
                start: dict[str, Any] = {
                    "id": "msg_" + run.rid,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": dict(_usage(run), output_tokens=0),
                }
                yield _ev("message_start", message=start)
            for d in deltas:
                if "tool_calls" in d:
                    for call in d["tool_calls"]:
                        fn = call["function"]
                        if "id" in call:
                            space = ""
                            if open_kind is not None:
                                yield _ev("content_block_stop", index=index)
                            index, open_kind, call_index = index + 1, "tool_use", call["index"]
                            use = {"type": "tool_use", "id": call["id"], "name": fn["name"], "input": {}}
                            yield _ev("content_block_start", index=index, content_block=use)
                        if open_kind == "tool_use" and call_index == call["index"] and fn["arguments"]:
                            part = {"type": "input_json_delta", "partial_json": fn["arguments"]}
                            yield _ev("content_block_delta", index=index, delta=part)
                    continue
                kind, key = ("thinking", "reasoning_content") if "reasoning_content" in d else ("text", "content")
                piece = d[key]
                if open_kind != kind:
                    if kind == "text":
                        if not piece.strip():  # whitespace between two calls opens no text block: held
                            space += piece
                            continue
                        piece, space = space + piece, ""
                    if open_kind is not None:
                        yield _ev("content_block_stop", index=index)
                    index, open_kind = index + 1, kind
                    block = (
                        {"type": "thinking", "thinking": "", "signature": ""}
                        if kind == "thinking"
                        else {"type": "text", "text": ""}
                    )
                    yield _ev("content_block_start", index=index, content_block=block)
                delta = (
                    {"type": "thinking_delta", "thinking": piece}
                    if kind == "thinking"
                    else {"type": "text_delta", "text": piece}
                )
                yield _ev("content_block_delta", index=index, delta=delta)
            finish = fin or finish
        if open_kind is not None:
            yield _ev("content_block_stop", index=index)
        reason, seq = _stop(run, finish, stops)
        yield _ev("message_delta", delta={"stop_reason": reason, "stop_sequence": seq}, usage=_usage(run))
        yield _ev("message_stop")
    except EngineError as e:
        yield _ev("error", error=error_body(e.status, e.detail)["error"])
    finally:
        run.close()  # the client went away (the generator is closed early): abort the request
