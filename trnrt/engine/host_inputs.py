"""The host inputs of the graph variants that take them (engine/sampler.py shaped_logits): a packed token mask and the
penalty inputs, built per call from its sampled rows, and the variant tag that keys the call's graph.

A variant exists only on an engine started with it (EngineConfig.structured_outputs, EngineConfig.penalties), and a call
takes one only when one of its rows needs it: every other call passes no extra argument, so it traces and runs the very
graph an engine without the flags runs.

Tags (the suffix of the runner's call key, e.g. ("decode", B, P, "grammar")):
- ("grammar",): a token mask [rows, ceil(V / 32)] (guided decoding, a thinking budget's forced token, min_tokens);
- ("pen{M}",): penalty inputs over M distinct tokens a row (EngineConfig.penalty_buckets, the vocabulary always the last
  rung, so every request fits);
- ("grammar", "pen{M}"): both. With both flags on, every penalty variant carries the mask (all ones when no row needs
  it), so a batch mixing guided and penalized requests needs no other form.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..config import pick_bucket
from . import grammar

if TYPE_CHECKING:
    from ..config import EngineConfig
    from .request import Request

logger = logging.getLogger(__name__)

GRAMMAR = "grammar"


def penalty_rungs(ecfg: EngineConfig, vocab: int) -> tuple[int, ...]:
    """The distinct-token counts the penalty variants are compiled at: the configured rungs below the vocabulary size,
    then the vocabulary size (no row can hold more distinct tokens)."""
    return (*sorted({m for m in ecfg.penalty_buckets if 0 < m < vocab}), vocab)


def variant_tags(ecfg: EngineConfig, vocab: int) -> list[tuple[str, ...]]:
    """Every variant an engine warms, in warmup order (none without the flags)."""
    tags: list[tuple[str, ...]] = []
    if ecfg.structured_outputs:
        tags.append((GRAMMAR,))
    if ecfg.penalties:
        g = (GRAMMAR,) if ecfg.structured_outputs else ()
        tags += [(*g, f"pen{m}") for m in penalty_rungs(ecfg, vocab)]
    return tags


def _out_index(r: Request) -> int:
    """The output index of the token a call launched now samples for r: its tokens so far, in-flight placeholders
    included (each holds one output token)."""
    return r.num_tokens - r.num_prompt


def _mask_row(r: Request, row: np.ndarray, eos_ids, backend) -> bool:
    """Fill r's mask row (all ones on entry) for the token sampled now; False if it needs none."""
    if r.think_force:  # a thinking budget's next forced token, and nothing else
        grammar.allow_only(row, r.think_force[0])
        return True
    used = False
    if r.matcher is not None and r.grammar_armed and r.grammar_active:
        backend.fill(row[None], [(0, r.matcher)])
        if not row.any():
            # A dead end: the grammar goes on only through text no token spells (xgrammar allows a token whose bytes
            # start a match, not knowing whether the vocabulary can finish it). The request ends here, on an
            # end-of-sequence id, rather than drawing from an empty set (guided.py lets that end through).
            logger.warning("request %s: its grammar cannot continue with this vocabulary; ending it", r.rid)
            for e in eos_ids:
                row.view(np.uint32)[e // 32] |= np.uint32(1 << (e % 32))
            r.grammar_dead = True
        used = True
    p = r.params
    if _out_index(r) < p.min_tokens:
        ends = [*(() if p.ignore_eos else eos_ids), *p.stop_token_ids]
        before = row.copy()
        grammar.forbid(row, ends)
        if not row.any():  # a grammar that can only end now: its end wins over min_tokens (vLLM's conflict too)
            row[:] = before
        used = True
    return used


def _penalty_row(r: Request) -> tuple[list[int], list[float]]:
    """r's distinct prompt-or-output tokens (prompt tokens only under a repetition penalty: vLLM applies frequency and
    presence to the output alone) and each one's count in the output. The output counts are kept on the request and
    extended as tokens arrive, and the prompt's distinct ids are taken once."""
    st = r.__dict__.get("_pen")
    out = r.output_ids
    if st is None or st[0] > len(out):
        seen = set(r.prompt_ids) if r.params.repetition_penalty != 1.0 else set()
        st = r.__dict__["_pen"] = [0, {}, seen]
    counts: dict[int, int] = st[1]
    for t in out[st[0] :]:
        counts[t] = counts.get(t, 0) + 1
    st[0] = len(out)
    ids = sorted(counts.keys() | st[2])
    return ids, [float(counts.get(t, 0)) for t in ids]


def penalty_inputs(rows: list[Request | None], m: int):
    """(hist_ids int64 [n, M], out_counts fp32 [n, M], rep, freq, pres fp32 [n]) at M = m (sampler.apply_penalties).
    A row pads with its first entry (the scatter writes the same value twice); a row without penalties gathers and
    scatters id 0 unchanged (repetition 1, frequency and presence 0)."""
    n = len(rows)
    ids = np.zeros((n, m), np.int64)
    cnt = np.zeros((n, m), np.float32)
    rep = np.ones(n, np.float32)
    freq = np.zeros(n, np.float32)
    pres = np.zeros(n, np.float32)
    for i, r in enumerate(rows):
        if r is None or not r.params.has_penalties:
            continue
        toks, counts = _penalty_row(r)
        if toks:
            ids[i, : len(toks)] = toks
            cnt[i, : len(toks)] = counts
            ids[i, len(toks) :] = toks[0]
            cnt[i, len(toks) :] = counts[0]
        p = r.params
        rep[i], freq[i], pres[i] = p.repetition_penalty, p.frequency_penalty, p.presence_penalty
    return ids, cnt, rep, freq, pres


def extras(ecfg: EngineConfig, vocab: int, eos_ids, backend, rows: list[Request | None]) -> tuple[list, tuple]:
    """([bitmask, penalties], tag) for a call whose sampled rows are `rows` (None: padding or a row that does not sample),
    or ([], ()) when none needs either: the call then takes the graph without them."""
    live = [r for r in rows if r is not None]
    pen = ecfg.penalties and any(r.params.has_penalties for r in live)
    mask = None
    if ecfg.structured_outputs and any(r.params.needs_mask for r in live):
        mask = grammar.full_mask(len(rows), vocab)
        used = False
        for i, r in enumerate(rows):
            if r is not None and r.params.needs_mask:
                used |= _mask_row(r, mask[i], eos_ids, backend)
        if not used:
            mask = None
    if mask is None and not pen:
        return [], ()
    if not pen:
        return [mask, None], (GRAMMAR,)
    rungs = penalty_rungs(ecfg, vocab)
    need = max((len(_penalty_row(r)[0]) for r in live if r.params.has_penalties), default=1)
    m = pick_bucket(max(need, 1), rungs)
    if ecfg.structured_outputs and mask is None:
        mask = grammar.full_mask(len(rows), vocab)
    tag = ((GRAMMAR,) if mask is not None else ()) + (f"pen{m}",)
    return [mask, penalty_inputs(rows, m)], tag


def dummy(tag: tuple[str, ...], rows: int, vocab: int) -> list:
    """Warmup's extras for a variant: every token allowed, neutral penalties."""
    mask = grammar.full_mask(rows, vocab) if GRAMMAR in tag else None
    pen = next((int(t[3:]) for t in tag if t.startswith("pen")), None)
    return [mask, penalty_inputs([None] * rows, pen) if pen is not None else None]
