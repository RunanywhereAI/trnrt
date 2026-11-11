"""Structured outputs through xgrammar, the grammar engine SGLang and vLLM both default to, and the packed token masks
the sampler takes (engine/sampler.py shaped_logits).

The grammar runs on the host: a matcher per request fills its row of a packed token bitmask (int32 [rows, ceil(V / 32)],
bit t % 32 of word t // 32 set when token t is allowed) before each step that samples for it, and accepting the sampled
token advances the matcher (engine/engine.py). The same mask carries a thinking budget's forced token (one bit set) and
min_tokens (the end-of-sequence and stop ids cleared). The graphs that take a mask exist only on an engine started with
--structured-outputs (engine/host_inputs.py).

xgrammar is an optional dependency (`pip install 'trnrt[grammar]'`): without it a guided-decoding request is refused, and
nothing else needs it.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

# Compiled grammars kept, the least recently used dropped first: an agent sends the same few schemas on every turn, and
# a client sending a new schema each request must not grow the engine's memory without end.
CACHE_SIZE = 32
MISSING = "guided decoding needs the xgrammar package on the engine's host: pip install 'trnrt[grammar]'"


@dataclass(frozen=True)
class GrammarSpec:
    """One request's output constraint. kind: "json_schema" (value: the schema as JSON text), "regex" or "ebnf"
    (value: the pattern or grammar). A JSON object without a schema is the schema {"type": "object"} (SGLang's reading of
    response_format json_object), and a choice of strings is a regex of their alternatives (vLLM's guided_choice)."""

    kind: str
    value: str

    KINDS = ("json_schema", "regex", "ebnf")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ValueError(f"grammar kind {self.kind!r}: one of {', '.join(self.KINDS)}")

    @staticmethod
    def choice(options: list[str]) -> GrammarSpec:
        return GrammarSpec("regex", "(" + "|".join(re.escape(o) for o in options) + ")")


def words(vocab: int) -> int:
    """32-bit words per mask row for a vocabulary of `vocab` ids."""
    return -(-vocab // 32)


def full_mask(rows: int, vocab: int) -> np.ndarray:
    """Every token allowed, for `rows` rows."""
    return np.full((rows, words(vocab)), -1, np.int32)


def allow_only(row: np.ndarray, token: int) -> None:
    """Allow exactly `token` in one mask row (a forced token)."""
    row[:] = 0
    row.view(np.uint32)[token // 32] = np.uint32(1 << (token % 32))


def forbid(row: np.ndarray, tokens) -> None:
    """Clear `tokens` in one mask row."""
    u = row.view(np.uint32)
    for t in tokens:
        u[t // 32] &= np.uint32(~(1 << (t % 32)) & 0xFFFFFFFF)


def allowed(row: np.ndarray, token: int) -> bool:
    return bool((row.view(np.uint32)[token // 32] >> np.uint32(token % 32)) & np.uint32(1))


class GrammarBackend:
    """xgrammar's compiler over the engine's tokenizer, with the last CACHE_SIZE compiled grammars cached by spec (an
    agent sends the same schema on every turn). xgrammar's own compiler cache is turned off (GrammarCompiler
    cache_enabled, where the installed xgrammar has it), so this bound holds for both. Built on the first guided
    request; raises ValueError(MISSING) without xgrammar."""

    def __init__(self, tokenizer, vocab_size: int, stop_ids=()):
        try:
            import xgrammar as xgr
        except ImportError as e:
            raise ValueError(MISSING) from e
        if tokenizer is None:
            raise ValueError("guided decoding needs the model's tokenizer, and this engine has none")
        self.xgr = xgr
        self.vocab_size = vocab_size
        # The model's end-of-sequence ids are the grammar's stop tokens: a finished grammar allows exactly those, and
        # the engine stops on them as on any EOS.
        info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=vocab_size, stop_token_ids=list(stop_ids) or None
        )
        try:
            self.compiler = xgr.GrammarCompiler(info, cache_enabled=False)
        except TypeError:  # an xgrammar without the setting
            self.compiler = xgr.GrammarCompiler(info)
        self._cache: OrderedDict[GrammarSpec, Any] = OrderedDict()

    def compile(self, spec: GrammarSpec):
        hit = self._cache.get(spec)
        if hit is not None:
            self._cache.move_to_end(spec)
            return hit
        c = self.compiler
        try:
            if spec.kind == "json_schema":
                g = c.compile_json_schema(spec.value)
            elif spec.kind == "regex":
                g = c.compile_regex(spec.value)
            else:
                g = c.compile_grammar(spec.value)
        except Exception as e:  # xgrammar raises RuntimeError (and others) on a malformed spec
            raise ValueError(f"invalid {spec.kind} for guided decoding: {e}") from e
        self._cache[spec] = g
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return g

    def matcher(self, spec: GrammarSpec):
        return self.xgr.GrammarMatcher(self.compile(spec))

    def fill(self, mask: np.ndarray, rows: list[tuple[int, Any]]) -> None:
        """Write each (row index, matcher) pair's allowed tokens into `mask` (int32 [n, words]) in place."""
        if not rows:
            return
        import torch

        t = torch.from_numpy(mask)  # shares memory: the matcher writes the numpy rows
        for i, m in rows:
            m.fill_next_token_bitmask(t, i)
