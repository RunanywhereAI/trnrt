"""The streaming reasoning / tool-call parser (trnrt/serve/parsers.py) and its request-level rules, on GLM-5.3-Flash's
output format: `<think>R</think>C<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`."""

import json
import random

from tests.parser_util import collect
from trnrt.serve.parsers import StreamParser, parse_full

OUT = ("<think>The user wants weather.</think>Let me check.<tool_call>get_weather<arg_key>city</arg_key>"
       "<arg_value>Paris</arg_value><arg_key>unit</arg_key><arg_value>c</arg_value></tool_call>")


def test_full_parse_splits_reasoning_content_and_tools():
    m = parse_full(OUT, tools=True, thinking_open=False)
    assert m["reasoning_content"] == "The user wants weather."
    assert m["content"] == "Let me check."
    (call,) = m["tool_calls"]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "unit": "c"}


def test_thinking_open_in_prompt():
    m = parse_full("reasoning here</think>\n\nanswer", tools=False, thinking_open=True)
    assert m["reasoning_content"] == "reasoning here" and m["content"] == "answer"


def test_plain_text_is_untouched():
    m = parse_full("just an answer", tools=True, thinking_open=False)
    assert m["content"] == "just an answer" and "tool_calls" not in m and "reasoning_content" not in m


def test_a_tool_block_without_tools_is_content():
    """A request that sent no tools (or tool_choice "none") gets a tool block back as text."""
    out = "Sure.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
    m = parse_full(out, tools=False)
    assert "tool_calls" not in m and m["content"] == out


def stream(text, chunks, **kw):
    p = StreamParser(**kw)
    deltas = []
    i = 0
    for n in chunks:
        deltas += p.push(text[i : i + n], final=False)
        i += n
    deltas += p.push(text[i:], final=True)
    return deltas


def test_streaming_matches_full_parse_for_any_chunking():
    rng = random.Random(0)
    for _ in range(200):
        chunks = []
        left = len(OUT)
        while left > 0:
            n = rng.randint(1, 7)
            chunks.append(min(n, left))
            left -= n
        reasoning, content, calls = collect(stream(OUT, chunks[:-1], tools=True, thinking_open=False))
        assert reasoning.strip("\n") == "The user wants weather."
        assert content.strip() == "Let me check."
        assert [(c["name"], json.loads(c["arguments"])) for c in calls] == [("get_weather", {"city": "Paris",
                                                                                            "unit": "c"})]
        assert "<" not in content


def test_tool_choice_rules():
    from trnrt.serve.parsers import chat_finish_reason, tool_choice_error

    assert tool_choice_error("auto") is None and tool_choice_error("none") is None
    assert tool_choice_error(None) is None
    # Enforcing these needs guided decoding, which no served graph runs: refused, never served unenforced.
    assert "guided decoding" in tool_choice_error("required")
    assert "guided decoding" in tool_choice_error({"type": "function", "function": {"name": "get_weather"}})
    assert "Invalid value for `tool_choice`" in tool_choice_error("sometimes")
    # "tool_calls" once a call was parsed and the model stopped; a call cut off by the token limit stays "length".
    assert chat_finish_reason("stop", True) == "tool_calls"
    assert chat_finish_reason("length", True) == "length"
    assert chat_finish_reason("stop", False) == "stop" and chat_finish_reason(None, True) is None
