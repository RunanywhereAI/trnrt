"""The host side of guided decoding, thinking budgets and min_tokens: what a request needs at admission, and what each
committed token changes (a grammar matcher accepts it, a thinking budget counts it). The token masks these produce are
built per call in engine/host_inputs.py.

Reasoning. GLM-5.3-Flash writes its reasoning between `<think>` and `</think>` (serve/parsers.py THINK_OPEN /
THINK_CLOSE), and its chat template opens the block in the prompt. As in vLLM:
- a grammar constrains the output after the reasoning (structured outputs are not enabled in reasoning): under a block
  the prompt opened, the matcher waits for `</think>` and constrains from the token after it;
- thinking_token_budget counts the reasoning tokens after `<think>` and, at the budget, forces `</think>` (one forced
  token a step, through a one-bit mask); a model that closes the block itself ends the count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .request import Status

if TYPE_CHECKING:
    from .grammar import GrammarBackend
    from .request import Request

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def ends_at(seq: list[int], end: int, tail: list[int]) -> bool:
    """Whether seq[:end] ends with `tail` (non-empty), without copying seq."""
    return bool(tail) and end >= len(tail) and seq[end - len(tail) : end] == tail


def in_reasoning(prompt: list[int], start: list[int], close: list[int]) -> bool:
    """Whether a prompt leaves the output inside a reasoning block: its last start tag comes after its last close tag.
    Scanned from the end, so a long agent prompt is read only back to its last start tag."""
    if not start:
        return False
    s = next((i for i in range(len(prompt) - len(start), -1, -1) if prompt[i : i + len(start)] == start), -1)
    if s < 0:
        return False
    return not any(prompt[i : i + len(close)] == close for i in range(s, len(prompt) - len(close) + 1))


class HostSampling:
    """One engine's guided-decoding state: the reasoning tags' ids and the grammar backend (engine/grammar.py), built on
    the first guided request. `runner.grammar` is set to the backend, where host_inputs fills the masks from it."""

    def __init__(self, tokenizer, vocab_size: int, eos_ids, runner):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.eos_ids = tuple(eos_ids)
        self.runner = runner
        enc = (lambda t: list(tokenizer(t, add_special_tokens=False)["input_ids"])) if tokenizer is not None else None
        self.start_ids: list[int] = enc(THINK_OPEN) if enc else []
        self.close_ids: list[int] = enc(THINK_CLOSE) if enc else []
        self.backend: GrammarBackend | None = None

    def admit(self, req: Request) -> None:
        """Set a request up before it is queued; ValueError (the API's 400) for what it cannot serve: a thinking budget
        or a grammar without the reasoning tags or the tokenizer, or a grammar xgrammar refuses (or no xgrammar)."""
        p = req.params
        if p.thinking_token_budget is not None:
            if not self.start_ids or not self.close_ids:
                raise ValueError(f"a thinking budget needs the tokenizer to encode {THINK_OPEN} and {THINK_CLOSE}")
            req.think_in = in_reasoning(req.prompt_ids, self.start_ids, self.close_ids)
            if req.think_in and p.thinking_token_budget == 0:
                req.think_force = list(self.close_ids)  # no reasoning at all: the first token closes the block
        if p.grammar is not None:
            if self.backend is None:
                from .grammar import GrammarBackend

                self.backend = GrammarBackend(self.tokenizer, self.vocab_size, self.eos_ids)
                self.runner.grammar = self.backend
            req.matcher = self.backend.matcher(p.grammar)  # ValueError: a spec xgrammar refuses
            req.grammar_armed = not (self.close_ids and in_reasoning(req.prompt_ids, self.start_ids, self.close_ids))

    def advance(self, seqs, finish_now) -> list[Request]:
        """After a commit: each sampling entry's new output tokens counted against its thinking budget and accepted by its
        grammar. A grammar that terminated, or ended at a dead end, finishes its request ("stop", through
        finish_now(req, reason)); returns those requests."""
        done = []
        for s in seqs:
            r = s.req
            if not s.sample:
                continue
            if r.params.thinking_token_budget is not None and not r.think_done:
                self._think(r)
            if r.matcher is not None and self._grammar(r) and r.status is Status.RUNNING:
                finish_now(r, "stop")
                done.append(r)
        return done

    def _think(self, r: Request) -> None:
        """vLLM's thinking budget: count reasoning tokens after the start tag; at the budget queue the close tag, forced
        one token a step (host_inputs allows only the next forced token)."""
        out = r.output_ids
        budget = r.params.thinking_token_budget or 0
        for i in range(r.think_seen, len(out)):
            if r.think_force:
                if out[i] != r.think_force[0]:
                    raise RuntimeError(f"{r.rid}: forced token {r.think_force[0]} not sampled ({out[i]})")
                r.think_force.pop(0)
                if not r.think_force:
                    r.think_done = True
                    break
                continue
            if not r.think_in:
                r.think_in = ends_at(out, i + 1, self.start_ids)
                if r.think_in and budget == 0:
                    r.think_force = list(self.close_ids)
                continue
            if ends_at(out, i + 1, self.close_ids):  # the model ended its reasoning itself
                r.think_done = True
                break
            r.think_count += 1
            if r.think_count >= budget:
                r.think_force = self._close_rest(out, i + 1)
        r.think_seen = len(out)

    def _close_rest(self, out: list[int], end: int) -> list[int]:
        """The close tag's ids still to force after out[:end]: the rest of it when out[:end] ends with a partial close
        tag (a multi-token tag cut at the budget), else the whole tag. Forcing the whole tag then would write its
        first ids twice."""
        close = self.close_ids
        for k in range(len(close) - 1, 0, -1):
            if ends_at(out, end, close[:k]):
                return close[k:]
        return list(close)

    def _grammar(self, r: Request) -> bool:
        """Accept r's new output tokens into its matcher (arming it first at the reasoning's end). True once the grammar
        has terminated, or a dead end's EOS ended it (host_inputs._mask_row)."""
        m, out = r.matcher, r.output_ids
        for i in range(r.grammar_seen, len(out)):
            if not r.grammar_armed:  # the close tag arms the grammar for the token after it
                r.grammar_armed = ends_at(out, i + 1, self.close_ids)
                continue
            if m.is_terminated():
                break
            # Sampled under this matcher's own mask, so a refusal means the mask was not applied: fail loudly. The one
            # exception is the end-of-sequence id a dead end allows (host_inputs._mask_row), which ends the request.
            if not m.accept_token(out[i]):
                if r.grammar_dead and out[i] in self.eos_ids:
                    # The dead end's EOS ends the request here, as "stop": under ignore_eos the commit does not stop on
                    # it, and the next step would meet the same dead end again, until the length limit.
                    r.grammar_seen = len(out)
                    return True
                raise RuntimeError(f"{r.rid}: token {out[i]} rejected by its own grammar")
        r.grammar_seen = len(out)
        return bool(m.is_terminated())
