"""GLM-5.3-Flash's own chat template behind the server (trnrt/serve/api.py template_kwargs and chat_prompt): the
template is tests/fixtures/GLM-5.3-Flash-chat_template.jinja, chat_template.jinja of zai-org/GLM-5.3-Flash at revision
eb9eb208, byte-identical.

That template has no enable_thinking. It always opens `<think>` after `<|assistant|>`, and it renders
`Reasoning Effort: Low` or `High` for reasoning_effort "low" or "high" and `Max` for any other value. So:
- reasoning_effort reaches the template, top-level (the OpenAI field) or in chat_template_kwargs. "low", "high" and
  "max" are served; any other value is a 400, since the template would answer it at Max;
- enable_thinking true is what the template does anyway, and is served; false would be ignored, so it is a 400;
- a chat_template_kwargs variable the template never reads is a 400; clear_thinking, which it reads, is served.
The server runs over tests/test_serve_api.py's stand-in engine, its tokenizer given GLM's role and think tags (and the
three effort lines) as added tokens, so the rendered prompt decodes back to the tags the real tokenizer would see.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.test_serve_api import PLAIN, SDK_CHAT, TOOLS, ScriptedEngine

TEMPLATE = os.path.join(os.path.dirname(__file__), "fixtures", "GLM-5.3-Flash-chat_template.jinja")
TAGS = [
    "[gMASK]",
    "<sop>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<think>",
    "</think>",
    "Reasoning Effort: Low",
    "Reasoning Effort: High",
    "Reasoning Effort: Max",
]
MSG = {"messages": [{"role": "user", "content": "hi"}]}


def _glm_engine() -> ScriptedEngine:
    from tokenizers import AddedToken

    eng = ScriptedEngine(PLAIN)
    tok = eng.tokenizer
    tok.add_tokens([AddedToken(t, normalized=False) for t in TAGS])
    with open(TEMPLATE) as f:
        tok.chat_template = f.read()
    eng.mcfg.vocab_size = len(tok)
    return eng


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from trnrt.serve import api

    eng = _glm_engine()
    with TestClient(api.build_app(eng, "glm")) as c:
        c.engine = eng
        yield c
    for lp in [lp for lp in api._LOOPS if lp.engine is eng]:
        lp.stop()


def _prompt(c) -> str:
    """The last request's prompt as the engine got it, decoded."""
    return c.engine.tokenizer.decode(c.engine.added[-1].prompt_ids)


def test_the_template_reads_reasoning_effort_and_has_no_enable_thinking():
    from trnrt.serve.api import template_variables

    eng = _glm_engine()
    used = template_variables(eng.tokenizer)
    assert {"reasoning_effort", "clear_thinking", "tools", "messages"} <= used and "enable_thinking" not in used
    text = eng.tokenizer.apply_chat_template(MSG["messages"], add_generation_prompt=True, tokenize=False)
    assert "Reasoning Effort: Max" in text and text.endswith("<|assistant|><think>")


@pytest.mark.parametrize("where", ["top", "kwargs", "both"])
@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_reasoning_effort_reaches_the_template(client, effort, where):
    body = dict(MSG)
    if where in ("top", "both"):
        body["reasoning_effort"] = effort
    if where in ("kwargs", "both"):
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    lines = [t for t in TAGS if t.startswith("Reasoning Effort")]
    assert [t for t in lines if t in _prompt(client)] == [f"Reasoning Effort: {effort.capitalize()}"]
    # The prompt opened the reasoning block, so the output is reasoning (the stand-in never closes it).
    assert r.json()["choices"][0]["message"]["reasoning_content"] == "Hello there again!"


def test_no_reasoning_effort_is_the_templates_max(client):
    assert client.post("/v1/chat/completions", json=MSG).status_code == 200
    assert "Reasoning Effort: Max" in _prompt(client) and _prompt(client).endswith("<|assistant|><think>")


def test_values_the_template_would_ignore_are_refused(client):
    for extra in (
        {"reasoning_effort": "medium"},
        {"reasoning_effort": "minimal"},
        {"reasoning_effort": "none"},
        {"reasoning_effort": "xhigh"},
        {"reasoning_effort": "Low"},
        {"chat_template_kwargs": {"reasoning_effort": "medium"}},
        {"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "high"}},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"chat_template_kwargs": {"enable_thinking": "no"}},
        {"chat_template_kwargs": {"thinking": True}},
    ):
        r = client.post("/v1/chat/completions", json={**MSG, **extra})
        assert r.status_code == 400, (extra, r.text)
    r = client.post("/v1/chat/completions", json={**MSG, "chat_template_kwargs": {"enable_thinking": False}})
    assert "no enable_thinking" in r.json()["detail"] and "reasoning_effort" in r.json()["detail"]
    assert client.engine.added == []


def test_values_the_template_honours_are_served(client):
    for kwargs in ({"enable_thinking": True}, {"clear_thinking": True}, {"clear_thinking": False}, {}):
        r = client.post("/v1/chat/completions", json={**MSG, "chat_template_kwargs": kwargs})
        assert r.status_code == 200, (kwargs, r.text)
    # What an OpenAI SDK sends by default, tools and all
    r = client.post("/v1/chat/completions", json={**SDK_CHAT, **MSG, "tools": TOOLS})
    assert r.status_code == 200, r.text
