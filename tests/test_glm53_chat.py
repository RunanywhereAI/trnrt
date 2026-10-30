"""GLM-5.3-Flash chat with tools as an agent client drives it, against the model's own chat template and tokenizer
(zai-org/GLM-5.3-Flash at a pinned revision, read from the local Hugging Face cache: no network unless
TRNRT_TEST_NETWORK=1; skipped unless the tokenizer files and chat_template.jinja are all cached).

- The prompt: an agent replays its earlier calls with arguments as the OpenAI JSON string, and the template writes
  them key by key from `tc.arguments.items()`, so the raw history cannot render (the positive control);
  api.chat_prompt (parsers.template_messages) renders it, every call and tool result in the template's own form, and
  the generation prompt ends `<|assistant|><think>`, so the output starts inside reasoning (thinking_open).
- The output: the assistant turn the template itself writes (reasoning, text, two calls with string, integer and
  object arguments) parses back to the same reasoning, text and calls, whole and in any chunking.
- The detokenizer: that turn as the model's token ids, decoded one token at a time by the server's _Detok
  (skip_special_tokens=True), keeps every tag: `<think>`, `<tool_call>` and the four arg tags
  are added tokens with special=false in tokenizer.json (ids 154841-154850), which this measures.
"""

from __future__ import annotations

import json

import pytest

from trnrt.serve.parsers import StreamParser, parse_full, template_messages
from tests.parser_util import check_streaming

REPO, REV = "zai-org/GLM-5.3-Flash", "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {
        "type": "object", "properties": {"filePath": {"type": "string"}, "offset": {"type": "integer"},
                                         "limit": {"type": "integer"}}, "required": ["filePath"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a command", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"},
                                         "env": {"type": "object"}}, "required": ["command"]}}},
]
KW = dict(tools=True, tool_defs=TOOLS)


NEEDS = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")  # the tokenizer and the chat template


@pytest.fixture(scope="module")
def tok():
    import os

    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    from transformers import AutoTokenizer

    missing = [f for f in NEEDS if not isinstance(try_to_load_from_cache(REPO, f, revision=REV), str)]
    if missing and os.environ.get("TRNRT_TEST_NETWORK") == "1":  # opted in: fetch the few small files
        for f in missing:
            hf_hub_download(REPO, f, revision=REV)
        missing = []
    if missing:  # a partial cache (another test fetched only the config) renders no template: skip, not fail
        pytest.skip(f"{REPO}@{REV[:12]}: {', '.join(missing)} not in the local Hugging Face cache "
                    "(TRNRT_TEST_NETWORK=1 fetches them)")
    try:
        tok = AutoTokenizer.from_pretrained(REPO, revision=REV, local_files_only=True)
    except (OSError, ValueError) as e:  # e.g. "Couldn't instantiate the backend tokenizer": an older tokenizers
        pytest.skip(f"{REPO}@{REV[:12]}'s tokenizer does not load here: {e}")
    if not tok.chat_template:
        pytest.skip(f"{REPO}@{REV[:12]}: the tokenizer loaded without its chat template")
    return tok


def agent_body(arguments_as) -> dict:
    """An agent's second turn: system, the user's task, the assistant's two calls (arguments as the OpenAI
    string, or as given), their results, and the next instruction."""
    calls = [("call_1", "read", {"filePath": "src/app.py", "limit": 20}),
             ("call_2", "bash", {"command": "pytest -q tests/test_app.py", "env": {"CI": "1"}})]
    return {"model": "glm", "stream": True, "tools": TOOLS, "messages": [
        {"role": "system", "content": "You are a coding agent, an interactive CLI tool."},
        {"role": "user", "content": "Fix the failing test in tests/test_app.py."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": i, "type": "function", "function": {"name": n, "arguments": arguments_as(a)}} for i, n, a in calls]},
        {"role": "tool", "tool_call_id": "call_1", "content": "def add(a, b):\n    return a - b\n"},
        {"role": "tool", "tool_call_id": "call_2", "content": "FAILED tests/test_app.py::test_add"},
        {"role": "user", "content": "Go on."}]}


def test_template_messages_follows_vllms_rules():
    raw = [{"role": "user", "content": "hi"},
           {"role": "assistant", "content": None, "tool_calls": [
               {"id": "a", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}},
               {"id": "b", "type": "function", "function": {"name": "g", "arguments": {"y": 2}}},
               {"id": "c", "type": "function", "function": {"name": "h", "arguments": ""}},
               {"id": "d", "type": "function", "function": {"name": "i", "arguments": None}},
               {"id": "e", "type": "function", "function": {"name": "j", "arguments": '{"cut": "off'}},
               {"id": "f", "type": "function", "function": {"name": "k", "arguments": "[1, 2]"}}]},
           {"role": "assistant", "content": "done", "tool_calls": []}]
    before = json.dumps(raw)
    out = template_messages(raw)
    assert [tc["function"]["arguments"] for tc in out[1]["tool_calls"]] == [{"x": 1}, {"y": 2}, {}, {}, {}, {}]
    assert out[0] is raw[0] and "tool_calls" not in out[2] and out[2]["content"] == "done"
    assert json.dumps(raw) == before, "the request's messages are not changed"
    for bad in ("not a dict", {"type": "code_interpreter", "function": {"name": "x"}}, {"type": "function"}):
        with pytest.raises(ValueError, match="tool_calls entries"):
            template_messages([{"role": "assistant", "tool_calls": [bad]}])


def test_the_agents_second_turn_renders_in_glm53s_template(tok):
    from trnrt.serve.api import chat_prompt

    body = agent_body(json.dumps)
    with pytest.raises(Exception, match="items"):   # the raw history: string arguments have no .items()
        tok.apply_chat_template(body["messages"], tools=TOOLS, add_generation_prompt=True, tokenize=False)
    ids, parse = chat_prompt(tok, body)
    text = tok.decode(ids)
    assert ("<tool_call>read<arg_key>filePath</arg_key><arg_value>src/app.py</arg_value><arg_key>limit</arg_key>"
            "<arg_value>20</arg_value></tool_call>") in text
    assert ("<tool_call>bash<arg_key>command</arg_key><arg_value>pytest -q tests/test_app.py</arg_value>"
            '<arg_key>env</arg_key><arg_value>{"CI": "1"}</arg_value></tool_call>') in text
    assert "<|observation|><tool_response>def add(a, b):" in text and "test_add</tool_response>" in text
    assert text.endswith("<|assistant|><think>") and '"name": "read"' in text  # the tools listed in the system turn
    assert parse["thinking_open"] and parse["tools"] and parse["tool_defs"] == TOOLS
    assert ids == chat_prompt(tok, agent_body(lambda a: a))[0], "string and dict arguments agree"


def _template_turn(tok) -> str:
    """The assistant turn the template writes for reasoning, text and two calls, as the model would emit it after the
    generation prompt's `<think>`: the rendered history's assistant part from `<think>` on, without that tag."""
    msgs = [{"role": "user", "content": "Fix it."},
            {"role": "assistant", "content": "Reading the file first.", "reasoning_content": "The test fails on add.",
             "tool_calls": [{"id": "c1", "type": "function", "function": {
                 "name": "read", "arguments": {"filePath": "src/app.py", "offset": 1, "limit": 40}}},
                 {"id": "c2", "type": "function", "function": {
                     "name": "bash", "arguments": {"command": "ls -la src", "env": {"LC_ALL": "C"}}}}]}]
    text = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False)
    turn = text[text.rindex("<|assistant|>") + len("<|assistant|>"):]
    assert turn.startswith("<think>")
    return turn[len("<think>"):].rstrip("\n")


WANT = [("read", {"filePath": "src/app.py", "offset": 1, "limit": 40}),
        ("bash", {"command": "ls -la src", "env": {"LC_ALL": "C"}})]


def _calls(msg: dict) -> list:
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in msg.get("tool_calls") or []]


def test_the_templates_own_assistant_turn_parses_back(tok):
    out = _template_turn(tok)
    msg = parse_full(out, thinking_open=True, **KW)
    assert msg["reasoning_content"] == "The test fails on add." and msg["content"] == "Reading the file first."
    assert _calls(msg) == WANT
    streamed = check_streaming(out, thinking_open=True, **KW)  # every chunking gives the full parse's message
    assert _calls(streamed) == WANT


def test_the_servers_detokenizer_keeps_every_tag(tok):
    from trnrt.serve.api import _Detok

    out = _template_turn(tok)
    ids = tok(out, add_special_tokens=False)["input_ids"]
    assert {154842, 154843, 154844, 154847, 154848, 154849, 154850} <= set(ids)  # </think> and the tool tags, one id each
    d, sp, deltas = _Detok(tok), StreamParser(thinking_open=True, **KW), []
    for n, i in enumerate(ids):
        last = n == len(ids) - 1
        deltas += sp.push(d.push([i], final=last), final=last)
    assert tok.decode(d.ids, skip_special_tokens=True) == out, "a tag was dropped or changed by the decode"
    reasoning = "".join(x.get("reasoning_content") or "" for x in deltas)
    content = "".join(x.get("content") or "" for x in deltas)
    calls: dict[int, dict] = {}
    for x in deltas:
        for tc in x.get("tool_calls") or ():
            c = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
            c["name"] += (tc.get("function") or {}).get("name") or ""
            c["arguments"] += (tc.get("function") or {}).get("arguments") or ""
    assert reasoning == "The test fails on add." and content.strip() == "Reading the file first."
    assert [(c["name"], json.loads(c["arguments"])) for _, c in sorted(calls.items())] == WANT


def test_a_developer_turn_reaches_glm53s_prompt_as_system(tok):
    """GLM-5.3-Flash's template renders only user / assistant / tool / system: a developer turn vanished from the prompt
    (the positive control, the template alone), while the request was answered. chat_prompt sends it as system."""
    from trnrt.serve.api import chat_prompt, template_knows_developer

    msgs = [{"role": "developer", "content": "Always answer in French."}, {"role": "user", "content": "hi"}]
    raw = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    assert "French" not in raw and not template_knows_developer(tok)
    ids, _ = chat_prompt(tok, {"messages": msgs})
    assert "<|system|>Always answer in French.<|user|>hi<|assistant|><think>" in tok.decode(ids)


def test_a_template_that_knows_developer_keeps_it():
    from trnrt.serve.api import template_knows_developer

    msgs = [{"role": "developer", "content": "x"}]
    assert template_messages(msgs) == msgs                            # off unless asked
    assert template_messages(msgs, developer_to_system=True) == [{"role": "system", "content": "x"}]
    assert msgs == [{"role": "developer", "content": "x"}]            # the request's own dicts are not changed

    class Tok:
        chat_template = "{% for m in messages %}{% if m.role == 'developer' %}<dev>{{ m.content }}{% endif %}{% endfor %}"

    class Named:
        chat_template = {"default": "{{ messages }}", "tool_use": "{% if m.role == 'developer' %}{% endif %}"}

    class Prose:  # mentions the word, has no branch for the role: its developer turn would vanish
        chat_template = ("You are a coding assistant for a developer.{% for m in messages %}{% if m.role == 'user' %}"
                         "{{ m.content }}{% endif %}{% endfor %}")

    class Listed:
        chat_template = "{% for m in messages %}{% if message['role'] in ['system', \"developer\"] %}{% endif %}{% endfor %}"

    assert template_knows_developer(Tok()) and template_knows_developer(Named()) and template_knows_developer(Listed())
    assert not template_knows_developer(Prose())

