"""The OpenAI Responses API, `POST /v1/responses`, stateless, served by the chat path (serve/api.py).

A request is translated into a /v1/chat/completions body: `instructions` and `input` become messages, function tools
become chat tools, `max_output_tokens` becomes max_tokens and `reasoning.effort` becomes reasoning_effort. Its output,
parsed by the same GLM parsers as a chat response, is translated back into Responses output items:

| Chat output | Responses item |
|---|---|
| `reasoning_content` | a `reasoning` item, its text as `reasoning_text` content (and as a `summary_text` when `reasoning.summary` is set, as SGLang mirrors it) |
| `content` | a `message` item with one `output_text` part |
| each of `tool_calls` | a `function_call` item; its `call_id` is the chat call's id |

Every other field goes to the chat path unchanged, which serves or refuses it as /v1/chat/completions does (sampling
extras, `session_id`, `chat_template_kwargs`). A streamed response sends the Responses events: `response.created`,
`response.in_progress`, then per item `response.output_item.added`, its deltas and its `.done` events, then
`response.completed` (`response.incomplete` when max_output_tokens cut it off, `response.failed` on an engine error).

Nothing is stored: `store` true, `previous_response_id` and `background` are refused with a 400, and so are the fields
whose meaning needs what no served graph computes (structured output in `text.format`, `top_logprobs`) or a server-side
tool. The fields that matter follow SGLang's /v1/responses (python/sglang/srt/entrypoints/openai/serving_responses.py and
protocol.py ResponsesRequest), and the event shapes the OpenAI API's.
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

# The Responses fields translated here. Any other field goes to the chat path as it is. Four are dropped because they
# change nothing a client gets back: `metadata` and `user` (echoed), `safety_identifier` (names the end user for OpenAI's
# abuse monitoring, as `user` does) and `prompt_cache_key` (a hint for OpenAI's prompt-cache routing; the prefix cache
# here matches every prompt by its tokens, and the router routes by session_id or prefix). `service_tier` is served
# only as "auto" or "default" (to_chat).
_OWN = frozenset(
    {
        "input",
        "instructions",
        "max_output_tokens",
        "tools",
        "tool_choice",
        "reasoning",
        "text",
        "store",
        "previous_response_id",
        "background",
        "include",
        "truncation",
        "max_tool_calls",
        "top_logprobs",
        "metadata",
        "user",
        "service_tier",
        "stream",
        "stream_options",
        "prompt_cache_key",
        "safety_identifier",
    }
)
# Chat fields with no meaning here, or another name in this API: refused unless absent or neutral.
_CHAT_ONLY: dict[str, tuple[object, ...]] = {
    "messages": (None,),
    "n": (None, 1),
    "best_of": (None, 1),
    "logprobs": (None, False),
    "max_tokens": (None,),
    "max_completion_tokens": (None,),
    "prompt": (None,),
    "conversation": (None,),
}
# include values served: the reasoning text, returned as the reasoning item's encrypted_content (not encrypted: this
# server keeps no key) so a stateless client can send its reasoning back with the next turn.
_INCLUDE = frozenset({"reasoning.encrypted_content"})
_TEXT_PARTS = ("input_text", "output_text", "text")


def _bad(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _text_of(content, where: str) -> str:
    """A message item's content as text: a string, or a list of text parts (input_text, output_text), joined as GLM's
    template joins a list of text parts. 400 for any other part (an image or a file)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _bad(f"{where}: content must be a string or a list of parts")
    out = []
    for part in content:
        if not (isinstance(part, dict) and part.get("type") in _TEXT_PARTS and isinstance(part.get("text"), str)):
            kind = part.get("type") if isinstance(part, dict) else type(part).__name__
            raise _bad(f"{where}: content part {kind!r} is not supported; this model reads text only")
        out.append(part["text"])
    return "".join(out)


def _reasoning_text(item: dict) -> str:
    """A reasoning item's text: its reasoning_text content, else its summary, else its encrypted_content (which this
    server fills with the reasoning text itself)."""
    for key, kind in (("content", "reasoning_text"), ("summary", "summary_text")):
        parts = item.get(key)
        if isinstance(parts, list):
            texts = [p.get("text") for p in parts if isinstance(p, dict) and p.get("type") == kind]
            if any(isinstance(t, str) and t for t in texts):
                return "\n".join(t for t in texts if isinstance(t, str))
    enc = item.get("encrypted_content")
    return enc if isinstance(enc, str) else ""


def input_messages(body: dict) -> list[dict]:
    """The chat messages of a request's instructions and input. An assistant turn's items (reasoning, an assistant
    message, function calls) join into one assistant message, as the chat API carries a turn; a function_call_output
    becomes a tool message."""
    msgs: list[dict] = []
    ins = body.get("instructions")
    if ins is not None:
        if not isinstance(ins, str):
            raise _bad("instructions must be a string")
        msgs.append({"role": "system", "content": ins})
    inp = body.get("input")
    if isinstance(inp, str):
        msgs.append({"role": "user", "content": inp})
        return msgs
    if not isinstance(inp, list) or not inp:
        raise _bad("input must be a string or a non-empty list of items")

    def assistant() -> dict:
        """The assistant message the next assistant-side item joins: the last message if it is one, else a new one."""
        if msgs and msgs[-1].get("role") == "assistant":
            return msgs[-1]
        msgs.append({"role": "assistant", "content": ""})
        return msgs[-1]

    for k, item in enumerate(inp):
        where = f"input[{k}]"
        if not isinstance(item, dict):
            raise _bad(f"{where} must be an object")
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role")
            if role not in ("user", "system", "developer", "assistant"):
                raise _bad(f"{where}: role must be user, system, developer or assistant")
            text = _text_of(item.get("content"), where)
            if role == "assistant":
                m = assistant()
                m["content"] = (m.get("content") or "") + text
            else:
                msgs.append({"role": role, "content": text})
        elif kind == "reasoning":
            m = assistant()
            text = _reasoning_text(item)
            m["reasoning_content"] = (m.get("reasoning_content") or "") + text
        elif kind == "function_call":
            name, args = item.get("name"), item.get("arguments", "")
            if not isinstance(name, str) or not name or not isinstance(args, str):
                raise _bad(f"{where}: a function_call needs a name and its arguments as a JSON string")
            call_id = item.get("call_id") or item.get("id") or "call_" + uuid.uuid4().hex[:24]
            m = assistant()
            m.setdefault("tool_calls", []).append(
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}
            )
        elif kind == "function_call_output":
            out = item.get("output")
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": out if isinstance(out, str) else _text_of(out, where),
                }
            )
        elif kind == "item_reference":
            raise _bad(f"{where}: item_reference is not supported; this server stores no items")
        else:
            raise _bad(f"{where}: input item type {kind!r} is not supported")
    return msgs


def _chat_tools(tools) -> list[dict] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise _bad("tools must be a list")
    out = []
    for t in tools:
        if not (isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("name"), str)):
            kind = t.get("type") if isinstance(t, dict) else None
            raise _bad(f"tool type {kind!r} is not supported: only function tools with a name")
        fn = {k: t[k] for k in ("name", "description", "parameters", "strict") if k in t}
        out.append({"type": "function", "function": fn})
    return out


def _tool_choice(tc):
    """The chat API's tool_choice: "auto" and "none" are served; "required" and a named function are passed on for the
    chat path to refuse with its reason (they need guided decoding)."""
    if isinstance(tc, dict) and tc.get("type") == "function" and "function" not in tc:
        return {"type": "function", "function": {"name": tc.get("name")}}
    return tc


def to_chat(body: dict) -> tuple[dict, dict]:
    """The /v1/chat/completions body for a Responses request, and what translating its output back needs: wants_summary,
    encrypted (include reasoning.encrypted_content). 400 for what cannot be served."""
    if body.get("store") is True:
        raise _bad("store is not supported: this server keeps no responses; send store false")
    if body.get("previous_response_id") is not None:
        raise _bad("previous_response_id is not supported: this server keeps no responses; send the whole input")
    if body.get("background") is True:
        raise _bad("background is not supported")
    if body.get("truncation") not in (None, "disabled"):
        raise _bad('truncation is not supported; send "disabled" or leave it out')
    if body.get("max_tool_calls") is not None:
        raise _bad("max_tool_calls is not supported")
    if body.get("top_logprobs") not in (None, 0):
        raise _bad("top_logprobs is not supported on /v1/responses")
    # flex, priority and scale are processing classes with their own latency, which one engine's queue does not have
    if body.get("service_tier") not in (None, "auto", "default"):
        raise _bad('service_tier is not supported; send "auto" or "default", or leave it out')
    include = body.get("include")
    if include is None:
        include = []
    if not isinstance(include, list) or any(x not in _INCLUDE for x in include):
        raise _bad(f"include supports only {', '.join(sorted(_INCLUDE))}")
    text = body.get("text")
    if text is not None:
        fmt = text.get("format") if isinstance(text, dict) else None
        if not isinstance(text, dict) or (fmt is not None and (not isinstance(fmt, dict) or fmt.get("type") != "text")):
            raise _bad('text.format other than {"type": "text"} is not supported: it needs guided decoding')
        if text.get("verbosity") is not None:
            raise _bad("text.verbosity is not supported")
    for key, neutral in _CHAT_ONLY.items():
        if body.get(key) not in neutral:
            raise _bad(f"{key} is not a /v1/responses field; see the Responses API")
    reasoning = body.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, dict):
        raise _bad("reasoning must be an object")
    reasoning = reasoning or {}
    extra = sorted(k for k in reasoning if k not in ("effort", "summary", "generate_summary"))
    if extra:
        raise _bad(f"reasoning.{extra[0]} is not supported: reasoning takes effort and summary")
    summary = reasoning.get("summary", reasoning.get("generate_summary"))
    if summary not in (None, "auto", "concise", "detailed"):
        raise _bad('reasoning.summary must be "auto", "concise" or "detailed"')

    chat = {k: v for k, v in body.items() if k not in _OWN}
    chat["messages"] = input_messages(body)
    tools = _chat_tools(body.get("tools"))
    if tools:
        chat["tools"] = tools
    if body.get("tool_choice") is not None:
        chat["tool_choice"] = _tool_choice(body["tool_choice"])
    if body.get("max_output_tokens") is not None:
        chat["max_tokens"] = body["max_output_tokens"]
    if reasoning.get("effort") is not None:
        chat["reasoning_effort"] = reasoning["effort"]
    return chat, {"wants_summary": summary is not None, "encrypted": bool(include)}


def _id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex


def _usage(run: Run) -> dict:
    u = run.usage()
    return {
        "input_tokens": u["prompt_tokens"],
        "input_tokens_details": {"cached_tokens": u["prompt_tokens_details"]["cached_tokens"]},
        "output_tokens": u["completion_tokens"],
        "output_tokens_details": {"reasoning_tokens": run.reasoning_tokens(0) or 0},
        "total_tokens": u["total_tokens"],
    }


def _reasoning_item(item_id: str, text: str, info: dict, status: str = "completed") -> dict:
    item: dict = {
        "id": item_id,
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text}] if info["wants_summary"] and text else [],
        "content": [{"type": "reasoning_text", "text": text}] if text else [],
        "status": status,
    }
    if info["encrypted"]:
        item["encrypted_content"] = text
    return item


def _message_item(item_id: str, text: str, status: str = "completed") -> dict:
    content = (
        [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}] if status == "completed" else []
    )
    return {"id": item_id, "type": "message", "role": "assistant", "status": status, "content": content}


def _call_item(item_id: str, call_id: str, name: str, arguments: str, status: str = "completed") -> dict:
    return {
        "id": item_id,
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    }


def output_items(msg: dict, info: dict) -> list[dict]:
    """A parsed chat message (parsers.parse_full) as Responses output items: reasoning, the answer, then the calls."""
    out = []
    if msg.get("reasoning_content"):
        out.append(_reasoning_item(_id("rs_"), msg["reasoning_content"], info))
    if msg.get("content"):
        out.append(_message_item(_id("msg_"), msg["content"]))
    for tc in msg.get("tool_calls") or ():
        fn = tc["function"]
        out.append(_call_item(_id("fc_"), tc["id"], fn["name"], fn["arguments"]))
    return out


def _response(body: dict, model: str, rid: str, created: int, status: str, output: list, usage) -> dict:
    """The Response object, echoing the request's settings as the OpenAI API does."""
    r = body.get("reasoning")
    reasoning: dict = r if isinstance(r, dict) else {}
    return {
        "id": "resp_" + rid,
        "object": "response",
        "created_at": created,
        "status": status,
        "background": False,
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "instructions": body.get("instructions"),
        "max_output_tokens": body.get("max_output_tokens"),
        "max_tool_calls": None,
        "model": model,
        "output": output,
        "parallel_tool_calls": body.get("parallel_tool_calls") is not False,
        "previous_response_id": None,
        "reasoning": {"effort": reasoning.get("effort"), "summary": reasoning.get("summary")},
        "store": False,
        "temperature": body.get("temperature"),
        "text": {"format": {"type": "text"}},
        "tool_choice": body.get("tool_choice") if body.get("tool_choice") is not None else "auto",
        "tools": body.get("tools") or [],
        "top_p": body.get("top_p"),
        "truncation": "disabled",
        "usage": usage,
        "user": body.get("user"),
        "metadata": body.get("metadata") or {},
    }


class _Stream:
    """The Responses events of one streamed run, from its parsed chat deltas. One output item is open at a time (GLM
    writes reasoning, then the answer, then its calls): opening the next closes it. Whitespace-only content between two
    calls opens no message item (it is held, and leads the next content if any comes)."""

    def __init__(self, body: dict, info: dict, model: str, run: Run):
        self.body, self.info, self.model, self.run = body, info, model, run
        self.seq = 0
        self.output: list[dict] = []  # the items closed so far
        self.cur: dict[str, Any] | None = None  # the open item: kind, id, index, text, and a call's call_id, name
        self.space = ""  # whitespace-only content not yet in a message item

    def ev(self, kind: str, **fields) -> str:
        data = {"type": kind, "sequence_number": self.seq, **fields}
        self.seq += 1
        return f"event: {kind}\ndata: {json.dumps(data)}\n\n"

    def response(self, status: str, usage=None) -> dict:
        return _response(self.body, self.model, self.run.rid, self.run.created, status, list(self.output), usage)

    def _open(self, kind: str, **extra) -> list[str]:
        out = self.close()
        index = len(self.output)
        cur: dict[str, Any] = {
            "kind": kind,
            "id": _id({"reasoning": "rs_", "message": "msg_", "call": "fc_"}[kind]),
            "index": index,
        }
        cur.update(text="", **extra)
        self.cur = cur
        if kind == "reasoning":
            item = _reasoning_item(cur["id"], "", self.info, "in_progress")
        elif kind == "message":
            item = _message_item(cur["id"], "", "in_progress")
        else:
            item = _call_item(cur["id"], cur["call_id"], cur["name"], "", "in_progress")
        out.append(self.ev("response.output_item.added", output_index=index, item=item))
        if kind == "message":
            part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
            out.append(
                self.ev(
                    "response.content_part.added", item_id=cur["id"], output_index=index, content_index=0, part=part
                )
            )
        elif kind == "reasoning" and self.info["wants_summary"]:
            part = {"type": "summary_text", "text": ""}
            out.append(
                self.ev(
                    "response.reasoning_summary_part.added",
                    item_id=cur["id"],
                    output_index=index,
                    summary_index=0,
                    part=part,
                )
            )
        return out

    def close(self) -> list[str]:
        """The .done events of the open item, if any."""
        cur, self.cur = self.cur, None
        if cur is None:
            return []
        at = {"item_id": cur["id"], "output_index": cur["index"]}
        text = cur["text"]
        out = []
        if cur["kind"] == "reasoning":
            item = _reasoning_item(cur["id"], text, self.info)
            if self.info["wants_summary"]:
                out.append(self.ev("response.reasoning_summary_text.done", **at, summary_index=0, text=text))
                part = {"type": "summary_text", "text": text}
                out.append(self.ev("response.reasoning_summary_part.done", **at, summary_index=0, part=part))
            else:
                out.append(self.ev("response.reasoning_text.done", **at, content_index=0, text=text))
        elif cur["kind"] == "message":
            item = _message_item(cur["id"], text)
            out.append(self.ev("response.output_text.done", **at, content_index=0, text=text, logprobs=[]))
            out.append(self.ev("response.content_part.done", **at, content_index=0, part=item["content"][0]))
        else:
            item = _call_item(cur["id"], cur["call_id"], cur["name"], text)
            out.append(self.ev("response.function_call_arguments.done", **at, arguments=text, name=cur["name"]))
        out.append(self.ev("response.output_item.done", output_index=cur["index"], item=item))
        self.output.append(item)
        return out

    def _current(self) -> dict[str, Any]:
        if self.cur is None:
            raise RuntimeError("no output item is open")
        return self.cur

    def delta(self, d: dict) -> list[str]:
        """The events of one parsed chat delta (StreamParser: reasoning_content, content or tool_calls)."""
        out: list[str] = []
        if "reasoning_content" in d:
            if self.cur is None or self.cur["kind"] != "reasoning":
                out += self._open("reasoning")
            cur = self._current()
            cur["text"] += d["reasoning_content"]
            at = {"item_id": cur["id"], "output_index": cur["index"]}
            if self.info["wants_summary"]:
                out.append(
                    self.ev(
                        "response.reasoning_summary_text.delta", **at, summary_index=0, delta=d["reasoning_content"]
                    )
                )
            else:
                out.append(
                    self.ev("response.reasoning_text.delta", **at, content_index=0, delta=d["reasoning_content"])
                )
        elif "content" in d:
            text = d["content"]
            if self.cur is None or self.cur["kind"] != "message":
                if not text.strip():
                    self.space += text
                    return out
                text, self.space = self.space + text, ""
                out += self._open("message")
            cur = self._current()
            cur["text"] += text
            at = {"item_id": cur["id"], "output_index": cur["index"]}
            out.append(self.ev("response.output_text.delta", **at, content_index=0, delta=text, logprobs=[]))
        else:
            for call in d.get("tool_calls", ()):
                fn = call["function"]
                if "id" in call:  # a new call: its name is complete
                    self.space = ""
                    out += self._open("call", call_id=call["id"], name=fn["name"], call_index=call["index"])
                call_item = self.cur  # the call's item, or None (a muted or skipped call)
                if call_item is None or call_item.get("call_index") != call["index"] or not fn["arguments"]:
                    continue
                call_item["text"] += fn["arguments"]
                out.append(
                    self.ev(
                        "response.function_call_arguments.delta",
                        item_id=call_item["id"],
                        output_index=call_item["index"],
                        delta=fn["arguments"],
                    )
                )
        return out


async def respond(http: HTTPRequest, body: dict, start_chat: Callable[[dict], Awaitable[Run]], model: str) -> Response:
    """POST /v1/responses: the chat path's run of the translated request, answered as a Response or its events."""
    chat, info = to_chat(body)
    stream = body.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise _bad("stream must be true or false")
    run = await start_chat(chat)
    if stream:
        return StreamingResponse(_events(run, body, info, model), media_type="text/event-stream")
    try:
        if not await run.wait(http):
            return Response(status_code=499)  # the client went away; nobody reads it
    except EngineError as e:
        raise e.http() from None
    msg, finish = run.message(0)
    status = "incomplete" if finish == "length" else "completed"
    return JSONResponse(_response(body, model, run.rid, run.created, status, output_items(msg, info), _usage(run)))


async def _events(run: Run, body: dict, info: dict, model: str):
    s = _Stream(body, info, model, run)
    try:
        yield s.ev("response.created", response=s.response("in_progress"))
        yield s.ev("response.in_progress", response=s.response("in_progress"))
        finish = None
        async for _, deltas, fin, _ in run.parsed():
            for d in deltas:
                for e in s.delta(d):
                    yield e
            finish = fin or finish
        for e in s.close():
            yield e
        status = "incomplete" if finish == "length" else "completed"
        yield s.ev(f"response.{status}", response=s.response(status, _usage(run)))
    except EngineError as e:
        for ev in s.close():
            yield ev
        failed = s.response("failed")
        failed["error"] = {"code": "server_error" if e.status >= 500 else "invalid_request", "message": e.detail}
        yield s.ev("response.failed", response=failed)
    finally:
        run.close()  # the client went away (the generator is closed early): abort the request


def not_stored(response_id: str) -> HTTPException:
    """GET, DELETE or cancel of a stored response: 404, since nothing is stored."""
    return HTTPException(
        status_code=404, detail=f"response {response_id!r} not found: this server stores no responses (store is false)"
    )
