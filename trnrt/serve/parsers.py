"""Reasoning and tool-call parsing for GLM-5.3-Flash chat completions.

GLM-5.3-Flash writes its reasoning as `<think> ... </think>` ahead of the answer, and its tool calls as
`<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...</tool_call>` with no newlines between the tags
(the GLM-4.7 / GLM-5 layout). With thinking on, the chat template opens `<think>` in the PROMPT, so the output starts
inside the reasoning block (`thinking_open`); with thinking off it closes the block in the prompt, and the output starts
with the answer.

The streaming parser is one state machine over the detokenized text. It holds back any suffix that could still become
a tag, so a tag is never streamed as content or reasoning. A tool call is announced (index, id, name) once its name is
complete, and its arguments then stream as JSON deltas that only ever extend what was sent. The non-streaming parse
runs the same machine over the whole text, so both paths return the same calls.

Behaviour follows vLLM v0.30.0 (commit ced6857afa0ea7b2e3f0846a62e1394e90f15607: vllm/parser/glm47_moe.py,
vllm/parser/engine/parser_engine.py, vllm/tool_parsers/utils.py) where vLLM and SGLang v0.5.21 (commit
e00930c5489053f26d86b179cee0d087f846acbb: python/sglang/srt/function_call/glm4_moe_detector.py,
parser/reasoning_parser.py) agree or SGLang has no equivalent. Portions adapted from vLLM,
Copyright contributors to the vLLM project, Apache License 2.0; see THIRD_PARTY_NOTICES.md.
Deliberate differences, each with its reason:
- Whether the output starts inside a reasoning block is read from the rendered prompt, not from an `enable_thinking`
  default (vLLM assumes thinking on). The two agree whenever the model follows its chat template, and this way a
  request that turns thinking off through chat_template_kwargs needs no special case.
- A tool block that yields no call (no function name, or a name the request did not offer) is returned verbatim as
  content instead of being dropped (vLLM and SGLang drop it), so the client sees what the model wrote.
- A call cut off by max_tokens keeps its partial last argument in both paths (vLLM's streaming and non-streaming
  disagree there).
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"

# GLM-4.7 and GLM-5 write `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value></tool_call>`
# (https://huggingface.co/zai-org/GLM-4.7/blob/602d01efcdd332c5238ca4bcede555defbe83eb7/chat_template.jinja,
# https://huggingface.co/zai-org/GLM-5/blob/c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2/chat_template.jinja). GLM-4.5 and
# GLM-4.6 put a newline after the name and after each pair
# (https://huggingface.co/zai-org/GLM-4.5/blob/cbb2c7cfb52fa128a9660cb1a7a78e017899e115/chat_template.jinja), which the
# parser accepts too, as vLLM's does. Strings are written raw, other values with `tojson`. `<think>`, `<tool_call>` and
# the four arg tags are added tokens with special=false (tokenizer_config.json), so the server's skip_special_tokens
# decode keeps them (tests/test_glm53_chat.py checks this on GLM-5.3-Flash's tokenizer). Regexes: vLLM
# https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/parser/glm47_moe.py, SGLang
# https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/function_call/glm4_moe_detector.py.
ARG_KEY_OPEN, ARG_KEY_CLOSE = "<arg_key>", "</arg_key>"
ARG_VALUE_OPEN, ARG_VALUE_CLOSE = "<arg_value>", "</arg_value>"
_ARG_RE = re.compile(r"<arg_key>(?P<key>.*?)</arg_key>\s*<arg_value>(?P<value>.*?)</arg_value>", re.DOTALL)
_PARTIAL_ARG_RE = re.compile(r"<arg_key>(?P<key>.*?)</arg_key>\s*<arg_value>(?P<value>.*)$", re.DOTALL)

# Inside reasoning: the end tag ends it; a repeated start tag is absorbed (vLLM glm47 (REASONING, THINK_START)); a tool
# call opener also ends it, without the end tag (vLLM's (REASONING, TOOL_START) transition, SGLang's tool_start_token).
_THINK_TAGS = (THINK_CLOSE, THINK_OPEN, TOOL_OPEN)
# Every tag that can appear inside a tool block, held back when the text so far ends with part of one.
_BLOCK_TAGS = (TOOL_CLOSE, ARG_KEY_OPEN, ARG_KEY_CLOSE, ARG_VALUE_OPEN, ARG_VALUE_CLOSE)


# ---------------------------------------------------------------------------------------
# Argument typing (vLLM ParserEngine._fix_arg_types over vllm/tool_parsers/utils.py)
# ---------------------------------------------------------------------------------------

# utils.py _TYPE_ALIASES
_TYPE_ALIASES = {
    "str": "string", "text": "string", "varchar": "string", "char": "string", "enum": "string",
    "int": "integer", "int32": "integer", "int64": "integer", "uint": "integer", "uint32": "integer",
    "uint64": "integer", "long": "integer", "short": "integer", "unsigned": "integer",
    "float": "number", "float32": "number", "float64": "number", "double": "number",
    "bool": "boolean", "dict": "object", "arr": "array", "list": "array", "sequence": "array",
}


def _schema_types(schema) -> set[str]:
    """utils.py extract_types_from_schema: `type` (string or list), types implied by `enum`,
    and anyOf / oneOf / allOf, recursively; {"string"} when nothing is declared."""
    if not isinstance(schema, dict):
        return {"string"}
    types: set[str] = set()
    t = schema.get("type")
    if isinstance(t, str):
        types.add(t)
    elif isinstance(t, list):
        types.update(x for x in t if isinstance(x, str))
    enum = schema.get("enum")
    for v in enum if isinstance(enum, list) else ():
        types.add("null" if v is None else "boolean" if isinstance(v, bool) else "integer" if isinstance(v, int)
                  else "number" if isinstance(v, float) else "array" if isinstance(v, list)
                  else "object" if isinstance(v, dict) else "string")
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(schema.get(key), list):
            for choice in schema[key]:
                types |= _schema_types(choice)
    return types or {"string"}


def _finite(obj) -> bool:
    try:
        json.dumps(obj, allow_nan=False)
        return True
    except (ValueError, TypeError):
        return False


def _coerce(value: str, types: set[str]):
    """utils.py coerce_to_schema_type: the first type in null > integer > number > boolean >
    object > array > string order that the raw value converts to; else JSON, else the string."""
    norm = {_TYPE_ALIASES.get(t.strip().lower(), t.strip().lower()) for t in types}
    for t in ("null", "integer", "number", "boolean", "object", "array", "string"):
        if t not in norm:
            continue
        if t == "null":
            if value.lower() == "null":
                return None
        elif t == "string":
            return value
        elif t == "integer":
            try:
                return int(value)
            except ValueError:
                pass
        elif t == "number":
            try:
                f = float(value)
            except ValueError:
                continue
            if math.isfinite(f):
                return f if f != int(f) else int(f)
        elif t == "boolean":
            low = value.lower().strip()
            if low in ("true", "1"):
                return True
            if low in ("false", "0"):
                return False
        else:
            try:
                parsed = json.loads(value)
            except ValueError:
                continue
            if _finite(parsed):
                return parsed
    try:
        parsed = json.loads(value)
    except ValueError:
        return value
    return parsed if _finite(parsed) else value


def _args_json(items: list[tuple[str, str, bool]], complete: bool, props: dict) -> str:
    """The arguments JSON of a call from its (key, raw value, closed) items, values typed by the tool's schema (vLLM
    ParserEngine._fix_arg_types: only declared keys are retyped, the rest stay strings). While the call is open this is
    the prefix that cannot change (vLLM _safe_arg_prefix): closed values are final; the value still being written
    streams only when it will stay a string, otherwise its key waits with it; the closing brace waits for the call to
    end. Same separators as json.dumps, so the complete text is json.dumps of the arguments. A repeated key is written
    again rather than overwritten, so the stream stays a prefix of the final text (a JSON reader keeps the last one, as
    vLLM's dict does)."""
    parts = []
    for k, v, closed in items:
        key = json.dumps(k, ensure_ascii=False) + ": "
        schema = props.get(k)
        if closed or complete:
            value = _coerce(v, _schema_types(schema)) if isinstance(schema, dict) else v
            parts.append(key + json.dumps(value, ensure_ascii=False))
            continue
        if not isinstance(schema, dict) or _schema_types(schema) == {"string"}:
            parts.append(key + json.dumps(v, ensure_ascii=False)[:-1])
        else:
            parts.append(key)
        return "{" + ", ".join(parts)
    if complete:
        return "{" + ", ".join(parts) + "}"
    return "{" + ", ".join(parts) if parts else ""


# ---------------------------------------------------------------------------------------
# The GLM tool block
# ---------------------------------------------------------------------------------------

def _scan(body: str, done: bool) -> tuple[str | None, str]:
    """A tool block's (function name, raw argument text) from its text after `<tool_call>` (vLLM glm47_moe.py, which
    reads both the inline layout and GLM-4.5's newline one). The name ends at the first `<arg_key>`, newline or the end
    of the block; None while it may still be being written, "" when there is none."""
    i = body.find(ARG_KEY_OPEN)
    head = (body if i < 0 else body[:i]).lstrip()
    if i < 0 and not done and "\n" not in head:
        return None, ""
    return head.split("\n", 1)[0].strip(), body[i:] if i >= 0 else ""


def _arguments(args: str, complete: bool, props: dict) -> str:
    """The call's arguments JSON so far (_args_json) from its `<arg_key>` / `<arg_value>` pairs."""
    items = [(m.group("key").strip(), m.group("value"), True) for m in _ARG_RE.finditer(args)]
    m = _PARTIAL_ARG_RE.search(_ARG_RE.sub("", args))
    if m and m.group("key").strip():
        items.append((m.group("key").strip(), m.group("value"), False))
    return _args_json(items, complete, props)


# ---------------------------------------------------------------------------------------
# Request-level rules
# ---------------------------------------------------------------------------------------

def template_messages(messages: list, developer_to_system: bool = False) -> list:
    """The request's messages as the chat template takes them: an assistant turn's tool_calls arguments as an object (a
    dict), not the JSON string the OpenAI API carries. GLM's templates write a call key by key from
    `tc.arguments.items()`, and an agent replays its earlier calls, string arguments and all, on every turn. vLLM's
    rules (chat_utils.py _postprocess_messages, vllm-project/vllm 36fa72d2d0): None or "" is {}; a dict is kept; a
    string is parsed, and one that is not JSON, or is JSON but not an object, becomes {} so that one malformed call in
    the history does not fail every later turn; an empty tool_calls list is dropped; an entry that is not a function
    call is refused (ValueError). SGLang (serving_chat.py normalize_assistant_tool_call_arguments) parses the same way
    but refuses the malformed string; this follows vLLM, since an agent session cannot recover from an error on its own
    history. New dicts; the request's are left as they were.

    developer_to_system: role "developer" (the OpenAI API's successor to "system" for instructions) becomes "system",
    for a chat template that has no branch for it. GLM-5.3-Flash's renders only user / assistant / tool / system, so a
    developer turn would vanish from the prompt while the request is still answered 200, and an agent harness that
    probes the role would then trust it. api.chat_prompt sets it when the template never tests for "developer"."""
    out = []
    for m in messages:
        if developer_to_system and isinstance(m, dict) and m.get("role") == "developer":
            m = dict(m, role="system")
        if not (isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list)):
            out.append(m)
            continue
        m = dict(m)
        if not m["tool_calls"]:
            del m["tool_calls"]
            out.append(m)
            continue
        calls = []
        for tc in m["tool_calls"]:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if not isinstance(tc, dict) or tc.get("type", "function") != "function" or not isinstance(fn, dict):
                raise ValueError("assistant tool_calls entries must be objects of type 'function' with a function")
            a = fn.get("arguments")
            if isinstance(a, dict):
                parsed = a
            elif not a:
                parsed = {}
            else:
                try:
                    parsed = json.loads(a) if isinstance(a, str) else a
                except ValueError:
                    parsed = None
                if not isinstance(parsed, dict):
                    parsed = {}
            calls.append(dict(tc, function=dict(fn, arguments=parsed)))
        m["tool_calls"] = calls
        out.append(m)
    return out


def tool_choice_error(tool_choice) -> str | None:
    """None for the tool_choice values this server honours, "none" and "auto" (or absent), else why not. "required" and
    a named function are what vLLM enforces with guided decoding, which no served graph runs: accepting them would
    answer 200 with whatever the model chose, so they are refused. vLLM's check (chat_completion/protocol.py
    check_tool_usage, v0.30.0) without its refusal of "auto" when no tools are sent."""
    if tool_choice is None or tool_choice in ("none", "auto"):
        return None
    if tool_choice == "required" or isinstance(tool_choice, dict):
        return ("`tool_choice` \"required\" and named tools are not supported: they need guided decoding, which this "
                "server does not run. Send \"auto\" or \"none\".")
    return f"Invalid value for `tool_choice`: {tool_choice}! Only \"none\" and \"auto\" are supported."


def chat_finish_reason(finish: str | None, called: bool) -> str | None:
    """vLLM chat_completion/serving.py: "tool_calls" once a call was parsed, but only when the model stopped by itself.
    A request cut off by its token limit stays "length" even when it holds a call: that call may be truncated (the
    parser closes its partial last argument), and an agent must not run it as complete. (vLLM kept the engine's reason
    for a named tool_choice; this server refuses one: tool_choice_error.)"""
    if finish == "stop" and called:
        return "tool_calls"
    return finish


# ---------------------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------------------

def _find(text: str, tags) -> tuple[int, str | None]:
    """The earliest tag in text (the longest one on a tie)."""
    best, found = -1, None
    for tag in tags:
        i = text.find(tag)
        if i >= 0 and (best < 0 or i < best or (i == best and len(tag) > len(found))):
            best, found = i, tag
    return best, found


def _safe_len(text: str, tags, final: bool) -> int:
    """How much of text can be emitted without risking a split tag."""
    if final:
        return len(text)
    hold = 0
    for tag in tags:
        for k in range(min(len(tag) - 1, len(text)), hold, -1):
            if tag.startswith(text[-k:]):
                hold = k
                break
    return len(text) - hold


@dataclass
class _Slot:
    """The call of the tool block being parsed."""
    index: int | None = None  # set once the call is announced
    skip: bool = False  # the block yields no call (no name, or a name the request did not offer)
    muted: bool = False  # parallel_tool_calls=False: calls after the first are dropped
    id: str = ""
    name: str = ""
    sent: str = ""  # arguments streamed so far


class StreamParser:
    """Feed detokenized text with push(); returns OpenAI chat `delta` dicts to emit.

    tools: parse tool calls (the request sent tools and tool_choice is not "none"); without it a tool block is
    content. thinking_open: the prompt ended inside a reasoning block. tool_defs: the request's OpenAI `tools`, used to
    type arguments by their schema and to refuse names the request did not offer (None: any name)."""

    def __init__(self, tools: bool = False, thinking_open: bool = False, tool_defs: list | None = None,
                 parallel_tool_calls: bool = True):
        self.tools = bool(tools)
        self.tool_defs = tool_defs
        self.parallel = parallel_tool_calls
        self.saw_tool = False
        self.reasoning_opened = False
        self._buf = ""
        self._state = "start"
        self._closed = False  # a reasoning block ended, so a further end tag is a duplicate
        self._skip_nl = False  # drop the newlines that follow the end tag
        self._slot = _Slot()
        self._calls = 0
        if thinking_open:
            self._enter_think()

    def _enter_think(self):
        self._state = "think"
        self.reasoning_opened = True

    def push(self, delta: str, final: bool) -> list[dict]:
        self._buf += delta
        out: list[dict] = []
        while True:
            st = self._state
            if st == "start":
                s = self._buf.lstrip()
                if s.startswith(THINK_OPEN):
                    self._buf = s[len(THINK_OPEN):]
                    self._enter_think()
                    continue
                if not final and THINK_OPEN.startswith(s):
                    return _coalesce(out)  # might still become the start tag
                self._state = "content"
                continue
            if st == "think":
                i, tag = _find(self._buf, _THINK_TAGS)
                if tag is None:
                    n = _safe_len(self._buf, _THINK_TAGS, final)
                    if n:
                        out.append({"reasoning_content": self._buf[:n]})
                        self._buf = self._buf[n:]
                    return _coalesce(out)
                if i:
                    out.append({"reasoning_content": self._buf[:i]})
                if tag == THINK_OPEN:  # a repeated start tag is absorbed
                    self._buf = self._buf[i + len(tag):]
                    continue
                self._closed = True
                self._state = "content"
                if tag == THINK_CLOSE:
                    self._buf = self._buf[i + len(tag):]
                    self._skip_nl = True
                else:  # the tool opener stays for the content state
                    self._buf = self._buf[i:]
                continue
            if st == "content":
                if self._skip_nl:
                    self._buf = self._buf.lstrip("\n")
                    if not self._buf and not final:
                        return _coalesce(out)
                    self._skip_nl = False
                tags = [THINK_OPEN]  # a later start tag opens another reasoning block (vLLM glm47 (CONTENT, THINK_START))
                if self.tools:
                    tags.append(TOOL_OPEN)
                if self._closed:
                    tags.append(THINK_CLOSE)  # vLLM drops a duplicate end tag ((CONTENT, THINK_END))
                i, tag = _find(self._buf, tags)
                if tag is None:
                    n = _safe_len(self._buf, tags, final)
                    if n:
                        out.append({"content": self._buf[:n]})
                        self._buf = self._buf[n:]
                    return _coalesce(out)
                before, self._buf = self._buf[:i], self._buf[i + len(tag):]
                if tag == TOOL_OPEN:
                    if before.strip():
                        out.append({"content": before})
                    self._state, self._slot = "block", _Slot()
                    continue
                if before:
                    out.append({"content": before})
                if tag == THINK_OPEN:
                    self._enter_think()
                continue
            # st == "block"
            j = self._buf.find(TOOL_CLOSE)
            if j >= 0 or final:
                body = self._buf[:j] if j >= 0 else self._buf
                self._buf = self._buf[j + len(TOOL_CLOSE):] if j >= 0 else ""
                out += self._block(body, True, TOOL_CLOSE if j >= 0 else "")
                self._state = "content"
                if not self._buf:
                    return _coalesce(out)
                continue
            out += self._block(self._buf[:_safe_len(self._buf, _BLOCK_TAGS, False)], False, "")
            return _coalesce(out)

    def _props(self, name: str) -> dict:
        for t in self.tool_defs or ():
            fn = t.get("function") if isinstance(t, dict) else None
            if isinstance(fn, dict) and fn.get("name") == name:
                params = fn.get("parameters")
                return (params.get("properties") or {}) if isinstance(params, dict) else {}
        return {}

    def _offered(self, name: str) -> bool:
        """vLLM glm47_moe validate_tool_names=True, SGLang parse_base_json: a name the request did not offer is no
        call."""
        if self.tool_defs is None:
            return True
        return any(isinstance(t, dict) and (t.get("function") or {}).get("name") == name for t in self.tool_defs)

    def _block(self, body: str, done: bool, closer: str) -> list[dict]:
        """The deltas of the current tool block so far. body: its text after `<tool_call>`; done: the block ended
        (closer: its `</tool_call>`, or "" when the output ended inside it)."""
        slot, out = self._slot, []
        name, args = _scan(body, done)
        announce = False
        if not slot.skip and slot.index is None and name is not None:
            if name and self._offered(name):
                slot.index, self._calls = self._calls, self._calls + 1
                slot.id, slot.name = "call_" + uuid.uuid4().hex[:24], name
                slot.muted = not self.parallel and slot.index > 0
                announce = True
            else:
                slot.skip = True
        if slot.index is not None:
            text = _arguments(args, done, self._props(slot.name))
            diff = ""
            if text.startswith(slot.sent):  # never retract what was streamed
                diff, slot.sent = text[len(slot.sent):], text
            if not slot.muted and (announce or diff):
                call = {"index": slot.index, "function": {"arguments": diff}}
                if announce:
                    call = {"index": slot.index, "id": slot.id, "type": "function",
                            "function": {"name": slot.name, "arguments": diff}}
                out.append({"tool_calls": [call]})
                self.saw_tool = True
        if done:
            if slot.index is None:
                out.append({"content": TOOL_OPEN + body + closer})
            self._slot = _Slot()
        return out


def _coalesce(deltas: list[dict]) -> list[dict]:
    """Merge neighbouring deltas of one kind; argument pieces of one call become one."""
    out: list[dict] = []
    for d in deltas:
        (key,) = d
        if out and key in out[-1] and len(out[-1]) == 1:
            if key != "tool_calls":
                out[-1][key] += d[key]
                continue
            calls = out[-1]["tool_calls"]
            for c in d["tool_calls"]:
                if calls and calls[-1]["index"] == c["index"] and "id" not in c:
                    calls[-1]["function"]["arguments"] += c["function"]["arguments"]
                else:
                    calls.append(c)
            continue
        out.append(d)
    return out


def parse_full(text: str, tools: bool = False, thinking_open: bool = False, tool_defs: list | None = None,
               parallel_tool_calls: bool = True) -> dict:
    """Non-streaming: split a whole completion into content, reasoning_content, tool_calls."""
    p = StreamParser(tools, thinking_open, tool_defs, parallel_tool_calls)
    deltas = p.push(text, final=True)
    msg: dict = {"role": "assistant", "content": None}
    if p.reasoning_opened:
        msg["reasoning_content"] = "".join(d.get("reasoning_content", "") for d in deltas).strip("\n")
    content = "".join(d.get("content", "") for d in deltas)
    by_index: dict[int, dict] = {}
    for d in deltas:
        for c in d.get("tool_calls", ()):
            if "id" in c:
                by_index[c["index"]] = {"id": c["id"], "type": "function",
                                        "function": {"name": c["function"]["name"], "arguments": ""}}
            by_index[c["index"]]["function"]["arguments"] += c["function"]["arguments"]
    calls = [by_index[i] for i in sorted(by_index)]
    if calls:
        content = content.strip()  # vLLM strips content when calls were made
    elif "reasoning_content" in msg:
        content = content.strip("\n")
    msg["content"] = content if content else None
    if calls:
        msg["tool_calls"] = calls
    return msg
