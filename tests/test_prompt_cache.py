"""The prompt token cache (serve/prompt_cache.py): every id list it returns must equal a full encode of the same text,
and the served path's (apply_chat_template(tokenize=True)) ids, in every case it serves, or it must turn itself off and
serve those.

The tokenizer is a byte-level BPE trained here on chat-like text (merges across spaces, punctuation, digits and
multi-byte characters, as GLM-5.3-Flash's does), with GLM-5.3-Flash's tags as added tokens and its own chat template
(tests/fixtures). Conversations grow turn by turn as an agent's do (tool calls and their results, reasoning, sessions
that branch, contents that contain the tags' text or a part of it), and every turn's ids are compared. A second
tokenizer adds tokens that overlap one another, the case the cut's margin exists for. The real GLM-5.3-Flash tokenizer
runs too when it is in the local Hugging Face cache (tests/test_glm53_chat.py's fixture)."""

from __future__ import annotations

import json
import os
import random

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from tests import test_glm53_chat as glm
from trnrt.serve import prompt_cache as PC

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
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<tool_response>",
    "</tool_response>",
    "<tools>",
    "</tools>",
]
WORDS = [
    "the",
    "a",
    "of",
    "to",
    "and",
    "in",
    "is",
    "it",
    "you",
    "that",
    "was",
    "for",
    "on",
    "are",
    "with",
    "as",
    "I",
    "his",
    "they",
    "be",
    "at",
    "one",
    "have",
    "this",
    "from",
    "or",
    "had",
    "by",
    "hot",
    "word",
    "but",
    "what",
    "some",
    "we",
    "can",
    "out",
    "other",
    "were",
    "all",
    "there",
    "when",
    "up",
    "use",
    "your",
    "how",
    "said",
    "an",
    "each",
    "she",
    "which",
    "do",
    "their",
    "time",
    "if",
    "will",
    "way",
    "about",
    "many",
    "then",
    "them",
    "write",
    "would",
    "like",
    "so",
    "these",
    "her",
    "long",
    "make",
    "thing",
    "see",
    "him",
    "two",
    "has",
    "look",
    "more",
    "day",
    "could",
    "go",
    "come",
    "did",
    "number",
    "sound",
    "no",
    "most",
    "people",
    "my",
    "over",
    "know",
    "water",
    "than",
    "call",
    "first",
    "who",
    "may",
    "down",
    "side",
    "been",
    "now",
    "find",
    "def",
    "class",
    "return",
    "import",
    "self",
    "None",
    "True",
    "False",
    "print",
    "json",
    "path",
    "file",
    "error",
    "test",
    "pytest",
    "assert",
    "value",
    "key",
]
EXTRA = [
    "中文",
    "café",
    "naïve",
    "🙂",
    "Ω",
    "\t",
    "\n",
    "  ",
    "\n\n",
    "!!",
    "...",
    "{}",
    '{"a": 1}',
    "x=1;",
    "€100",
    "<",
    ">",
    "|",
    "<|",
    "|>",
    "</",
    "think",
    "user|>",
]


def _corpus(rng, n=1500):
    return [" ".join(rng.choice(WORDS + EXTRA) for _ in range(rng.randrange(3, 30))) for _ in range(n)]


def _tokenizer(added=TAGS, template=True):
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=700, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False
    )
    tk.train_from_iterator(_corpus(random.Random(0)), trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, clean_up_tokenization_spaces=False)
    tok.add_tokens([AddedToken(t, normalized=False) for t in added])
    if template:
        with open(TEMPLATE) as f:
            tok.chat_template = f.read()
    return tok


@pytest.fixture(scope="module")
def tok():
    return _tokenizer()


def full(tok, text):
    return tok._tokenizer.encode(text, add_special_tokens=False).ids


def served(tok, messages, **kw):
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kw)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def text_of(tok, messages, **kw):
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kw)


def _content(rng):
    """A message's text: words, multi-byte characters, whitespace runs, and now and then a tag's text or a part of one
    (a user pasting a transcript)."""
    parts = []
    for _ in range(rng.randrange(1, 25)):
        r = rng.random()
        if r < 0.06:
            parts.append(rng.choice(TAGS))
        elif r < 0.12:
            t = rng.choice(TAGS)
            parts.append(t[: rng.randrange(1, len(t))])
        elif r < 0.25:
            parts.append(rng.choice(EXTRA))
        else:
            parts.append(rng.choice(WORDS))
    return rng.choice(["", " ", "\n"]).join(parts)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
]


def _turn(rng, messages, k):
    """The agent's next turn: its reply (text, reasoning, tool calls), the tool results, and the next user message."""
    msgs = list(messages)
    reply = {"role": "assistant", "content": _content(rng)}
    if rng.random() < 0.5:
        reply["reasoning_content"] = _content(rng)
    if rng.random() < 0.5:
        calls = [
            {
                "id": f"call_{k}_{j}",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": _content(rng)})},
            }
            for j in range(rng.randrange(1, 3))
        ]
        reply["tool_calls"] = calls
        msgs.append(reply)
        msgs.extend({"role": "tool", "tool_call_id": c["id"], "content": _content(rng)} for c in calls)
    else:
        msgs.append(reply)
    if rng.random() < 0.8 or msgs[-1]["role"] == "assistant":
        msgs.append({"role": "user", "content": _content(rng)})
    return msgs


def _template_messages(messages):
    from trnrt.serve.parsers import template_messages

    return template_messages(messages)


def test_the_cache_is_on_for_a_byte_level_bpe_with_plain_added_tokens(tok):
    cache = PC.PromptCache(tok, 1 << 26)
    assert cache.off is None, cache.off


@pytest.mark.parametrize("seed", range(6))
def test_agent_sessions_turn_by_turn_give_the_served_ids(tok, seed):
    """Several sessions, interleaved, each growing turn by turn; some requests carry their session id and some do not
    (the content key), and now and then a session branches from an earlier turn (an agent retrying). Every turn's ids
    are the served path's, and the cache hits on most turns (it is not vacuously falling back)."""
    rng = random.Random(seed)
    cache = PC.PromptCache(tok, 1 << 26, audit_every=0)  # no audits: the comparison here is the check
    system = {"role": "system", "content": "You are a coding agent. " + _content(rng)}
    sessions = {f"s{i}": [system, {"role": "user", "content": _content(rng)}] for i in range(4)}
    history: dict[str, list] = {s: [] for s in sessions}
    for k in range(60):
        sid = rng.choice(list(sessions))
        if history[sid] and rng.random() < 0.1:  # a branch: retry from an earlier turn
            sessions[sid] = rng.choice(history[sid])
        msgs = sessions[sid] = _turn(rng, sessions[sid], k)
        history[sid].append(msgs)
        tm = _template_messages(msgs)
        kw = {"tools": TOOLS} if seed % 2 else {}
        text = text_of(tok, tm, **kw)
        key = sid if rng.random() < 0.7 else None
        got = cache.encode(text, key)
        assert got == served(tok, tm, **kw) == full(tok, text), (seed, k)
    assert cache.off is None
    assert cache.counts["hits"] > cache.counts["misses"], cache.counts
    assert cache.counts["reused_tokens"] > cache.counts["tail_tokens"], cache.counts


@pytest.mark.parametrize("seed", range(4))
def test_random_text_pairs_around_the_added_tokens(tok, seed):
    """Texts built of words, tags and tag fragments, each next one sharing a random prefix of the last (cut anywhere:
    inside a tag, right after one, inside a multi-byte run); every encode equals the full encode."""
    rng = random.Random(100 + seed)
    cache = PC.PromptCache(tok, 1 << 24, audit_every=0)
    text = _content(rng)
    for k in range(400):
        cut = rng.randrange(0, len(text) + 1)
        text = text[:cut] + _content(rng) if rng.random() < 0.8 else _content(rng) + text
        assert cache.encode(text, "one") == full(tok, text), (seed, k, text[-80:])
    assert cache.counts["hits"] > 100


def test_overlapping_added_tokens_keep_the_margin():
    """Added tokens that overlap one another (one a prefix, a suffix or a part of another): the cut needs the longer
    shared margin. Random texts over their fragments, every encode the full one. With the margin reduced to the token's
    own length (the mutant need = len), the same texts produce a wrong cut."""
    added = ["<a>", "<a>b", "b<c>", "<c>", "xx<a", "a>yy", "<|z|>", "q<a>zz", "w<|z|>w"]
    t = _tokenizer(added, template=False)
    frags = [*added, "<", ">", "a", "b", "c", "x", "y", " ", "word", "<a", "a>", "b<", "<|", "|>", "z", "q", "w", "zz"]

    def run(cache, rng):
        text = "".join(rng.choice(frags) for _ in range(30))
        for _ in range(1500):
            text = text[: rng.randrange(0, len(text) + 1)] + "".join(
                rng.choice(frags) for _ in range(rng.randrange(1, 12))
            )
            if cache.encode(text, "k") != full(t, text):
                return False
        return True

    for seed in range(3):
        cache = PC.PromptCache(t, 1 << 24, audit_every=0)
        assert cache.off is None, cache.off
        assert run(cache, random.Random(seed)), seed

    def naive():
        c = PC.PromptCache(t, 1 << 24, audit_every=0)
        c.need = {i: len(t.convert_ids_to_tokens(i)) for i in c.need}  # no margin past the token itself
        return c

    assert not all(run(naive(), random.Random(seed)) for seed in range(3)), "the control: some cut must be wrong"


def test_sessions_and_content_keys_find_their_own_entries(tok):
    cache = PC.PromptCache(tok, 1 << 24, audit_every=0)
    long = " ".join(WORDS)  # a system prompt longer than KEY_CHARS: the content key is the same on the next turn
    a = text_of(tok, [{"role": "system", "content": "Alpha. " + long}, {"role": "user", "content": "first task here"}])
    b = text_of(tok, [{"role": "system", "content": "Beta. " + long}, {"role": "user", "content": "other task"}])
    cache.encode(a, "s1")
    cache.encode(b, None)
    a2 = a + "more words of the first session <|user|> again"
    b2 = b + " and the next turn"
    assert cache.encode(a2, "s1") == full(tok, a2) and cache.counts["hits"] == 1
    assert (
        cache.encode(b2, None) == full(tok, b2) and cache.counts["hits"] == 2
    )  # same first KEY_CHARS: the content key
    assert (
        cache.encode(a2 + " x", "s2") == full(tok, a2 + " x") and cache.counts["hits"] == 2
    )  # another session: a miss


def test_memory_bound_and_buckets(tok):
    cache = PC.PromptCache(tok, 12_000, audit_every=0)
    rng = random.Random(3)
    for i in range(200):
        t = text_of(tok, [{"role": "user", "content": f"{i} " + _content(rng)}])
        assert cache.encode(t, f"s{i % 50}") == full(tok, t)
        assert cache.bytes <= cache.max_bytes
    assert cache.counts["evictions"] > 0 and all(len(b) <= PC.BUCKET["s"] for b in cache.buckets.values())


def test_an_audit_that_disagrees_turns_the_cache_off_and_serves_the_served_ids(tok):
    """The served path is the reference: if it ever differs from the cut (a tokenizer the checks could not see), the
    cache serves the served ids, turns off for good, and passes every later request through."""
    cache = PC.PromptCache(tok, 1 << 24, audit_every=1)
    t1 = text_of(tok, [{"role": "user", "content": "hello there"}])
    cache.encode(t1, "s")
    t2 = t1 + "general kenobi <|user|> next"
    wrong = [*full(tok, t2), 5]
    assert cache.encode(t2, "s", served=lambda: wrong) == wrong
    assert cache.off is not None and "audit mismatch" in cache.off and not cache.entries
    t3 = t2 + " more"
    assert cache.encode(t3, "s", served=lambda: full(tok, t3)) == full(tok, t3)
    assert cache.counts["hits"] == 1


def test_audits_run_on_the_first_hits_and_then_every_nth(tok):
    calls = []
    cache = PC.PromptCache(tok, 1 << 24, audit_every=4)
    t = text_of(tok, [{"role": "user", "content": "start"}])
    cache.encode(t, "s")
    for i in range(20):
        t = t + f" step {i} <|observation|> ok"
        cache.encode(t, "s", served=lambda t=t: calls.append(1) or full(tok, t))
    assert cache.counts["hits"] == 20 and cache.counts["audits"] == len(calls)
    assert len(calls) == sum(1 for h in range(1, 21) if h <= PC.AUDIT_FIRST or h % 4 == 0)


def test_the_cache_stays_off_for_tokenizers_it_cannot_cut_exactly():
    from tokenizers import AddedToken

    strip = _tokenizer(TAGS[:4])
    strip.add_tokens([AddedToken("<|strip|>", lstrip=True, normalized=False)])
    assert "strip, normalize or match single words" in PC.PromptCache(strip, 1 << 20).off
    # a whole-word token matches by what follows it: cached "x foo bar", a cut at "foo" would keep "x " apart from
    # it, while in "x foobar" no token matches and " f" merges. The self-check does not catch it; the guard must.
    word = _tokenizer(TAGS[:4])
    word.add_tokens([AddedToken("foo", single_word=True, normalized=False)])
    assert "match single words: ['foo']" in PC.PromptCache(word, 1 << 20).off
    assert PC.PromptCache(word, 1 << 20).encode("x foobar", "s") == full(word, "x foobar")
    trunc = _tokenizer(TAGS[:4])
    trunc._tokenizer.enable_truncation(64)
    assert "truncates or pads" in PC.PromptCache(trunc, 1 << 20).off
    none = _tokenizer([])
    assert PC.PromptCache(none, 1 << 20).off == "no added tokens to cut at"
    assert PC.PromptCache(object(), 1 << 20).off == "not a fast tokenizer"
    off = PC.PromptCache(strip, 1 << 20)
    text = "plain <|user|> text"
    assert off.encode(text, "s", served=lambda: [1, 2, 3]) == [1, 2, 3] and not off.entries  # off: the served ids


def test_the_template_check_compares_the_served_path(tok, monkeypatch):
    """At construction the served path (apply_chat_template(tokenize=True)) must equal the rendered text's encode on a
    sample chat; a transformers that added a special token on one path only turns the cache off."""
    t = _tokenizer()
    real = t.apply_chat_template

    def plus_bos(*a, **kw):
        out = real(*a, **kw)
        if kw.get("tokenize"):
            ids = list(out["input_ids"] if hasattr(out, "keys") else out)
            return [0, *ids]
        return out

    monkeypatch.setattr(t, "apply_chat_template", plus_bos)
    assert "template check" in PC.PromptCache(t, 1 << 20).off


def test_from_env(tok, monkeypatch):
    monkeypatch.delenv(PC.ENV_MB, raising=False)
    assert PC.PromptCache.from_env(tok) is None
    monkeypatch.setenv(PC.ENV_MB, "0")
    assert PC.PromptCache.from_env(tok) is None
    monkeypatch.setenv(PC.ENV_MB, "1.5")
    monkeypatch.setenv(PC.ENV_AUDIT, "0")
    c = PC.PromptCache.from_env(tok)
    assert c is not None and c.max_bytes == int(1.5 * 2**20) and c.audit_every == 0
    assert PC.PromptCache.from_env(None) is None


# The real GLM-5.3-Flash tokenizer and template, when cached locally (tests/test_glm53_chat.py's fixture).
glm_tok = glm.tok


def test_the_real_tokenizer_turn_by_turn(glm_tok):
    cache = PC.PromptCache(glm_tok, 1 << 26, audit_every=0)
    assert cache.off is None, cache.off
    rng = random.Random(1)
    msgs = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "Fix the test."}]
    for k in range(25):
        msgs = _turn(rng, msgs, k)
        tm = _template_messages(msgs)
        text = text_of(glm_tok, tm, tools=glm.TOOLS)
        assert cache.encode(text, "s") == served(glm_tok, tm, tools=glm.TOOLS) == full(glm_tok, text), k
    assert cache.counts["hits"] >= 20
