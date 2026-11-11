"""The prompt token cache (TRNRT_PROMPT_CACHE_MB, host only, off by default): a chat turn's prompt ids built from an
earlier turn's ids and an encode of the new text's tail, instead of an encode of the whole rendered prompt.

Why: an agent's next turn is its previous prompt plus a few messages, and the chat endpoint renders and encodes the
whole conversation on every turn. Rendering is the cheap half: on one host CPU core of a trn2.48xlarge, GLM-5.3-Flash's
chat template rendered a ~100K-token conversation in about 5 ms, and its tokenizer then took 195-239 ms to encode
97K-114K-token prompts. An encode holds the GIL, so a wider prompt pool would not run two at once. This keeps each key's
last (text, ids) and encodes only what follows the text both turns share.

Why a cut is exact. A Hugging Face fast tokenizer splits its added tokens out of the raw text first, then normalizes,
pre-tokenizes and encodes each piece between them on its own. So when no added token strips its neighbours' whitespace,
is matched after normalization or matches only as a whole word (lstrip, rstrip, normalized and single_word all false, as
for every added token of GLM-5.3-Flash), the text before an added token encodes the same whatever follows it:

    ids(new) = ids(old)[:i] + encode(new[c:])

for an added token T at character c and token index i of the old encoding, provided its match is the same in the new
text. Matching is leftmost-longest, so it is when every match that could start at or overlap c lies wholly inside the
text both share (L characters): c + need(T) <= L, need(T) the longest added token that starts with T's text; and, when
some added token could overlap T from the left (a proper suffix of it a prefix of T, or T inside it), c + M <= L for M
the longest added token.

The key (the request's session_id, else a hash of the text's first KEY_CHARS characters: the system prompt and tools)
only finds up to BUCKET candidates, of which the one sharing the longest prefix is used; the texts are compared every
time, so a stale entry or a key several chats share costs a miss, never wrong ids. Guards, any of which turns the cache
off for good (logged, and the served path's ids served):
- at construction, a self-check of the cut against the full encode on strings built from the tokenizer's own added
  tokens, and a check that the served path (apply_chat_template(tokenize=True)) equals the rendered text's encode on a
  sample chat; and off outright for a tokenizer that truncates or pads, or whose added tokens strip, normalize or
  match single words (their match depends on the characters around them, which the self-check samples only);
- an audit of the first AUDIT_FIRST hits, and of every TRNRT_PROMPT_CACHE_AUDIT-th hit after them (default 64; 0:
  none), against the served path's ids for that request.
Used by one thread (the API server's prompt pool), so it takes no lock.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import os
import sys
from array import array
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

ENV_MB = "TRNRT_PROMPT_CACHE_MB"
ENV_AUDIT = "TRNRT_PROMPT_CACHE_AUDIT"
KEY_CHARS = 256  # an agent's system prompt and tools: the same on every turn, and on every session of that agent
BUCKET = {"s": 8, "h": 32}  # candidates per key: a session id's, and all the sessions behind one content key
AUDIT_FIRST = 8  # every one of a process's first hits is audited: a wrong cut would otherwise serve 63 more


def _lcp(a: str, b: str) -> int:
    """The length of a's and b's common prefix: one compare when b extends a (the usual case), else a gallop and then a
    binary search, comparing chunks in place (b.startswith(a[lo:hi], lo))."""
    if b.startswith(a):
        return len(a)
    n, lo, step = min(len(a), len(b)), 0, 1024
    while True:  # a[:lo] == b[:lo]
        hi = min(n, lo + step)
        if hi == lo or not b.startswith(a[lo:hi], lo):
            break
        lo, step = hi, 2 * step
    hi -= 1  # a[:hi + 1] != b[:hi + 1]: the prefix length is in [lo, hi]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if b.startswith(a[lo:mid], lo):
            lo = mid
        else:
            hi = mid - 1
    return lo


class _Entry:
    __slots__ = ("bstart", "btok", "ids", "key", "nbytes", "text")

    def __init__(self, text: str, ids: array, bstart: array, btok: array):
        self.text, self.ids, self.bstart, self.btok = text, ids, bstart, btok
        self.key = ""
        self.nbytes = sys.getsizeof(text) + sum(x.itemsize * len(x) for x in (ids, bstart, btok)) + 200


class PromptCache:
    def __init__(self, tok, max_bytes: int, audit_every: int = 64):
        self.fast = getattr(tok, "_tokenizer", None)  # a PreTrainedTokenizerFast's tokenizers.Tokenizer
        self.max_bytes, self.audit_every = int(max_bytes), int(audit_every)
        self.entries: OrderedDict[int, _Entry] = OrderedDict()  # least recently used first
        self.buckets: dict[str, list[int]] = {}
        self.bytes, self._next = 0, 0
        self.counts = dict.fromkeys(("hits", "misses", "reused_tokens", "tail_tokens", "audits", "evictions"), 0)
        self.off: str | None = None  # why the cache is off, once it is
        self.margin = 0
        self.need: dict[int, int] = {}  # token id -> characters past its start both texts must share before a cut there
        self.added_ids: frozenset[int] = frozenset()
        if self.fast is None:
            self.off = "not a fast tokenizer"
            logger.warning("prompt cache off: %s", self.off)
            return
        added = list(self.fast.get_added_tokens_decoder().values())
        texts = [t.content for t in added]
        self.margin = max((len(x) for x in texts), default=0)
        for x in texts:
            over = any(y != x and (x in y or any(x.startswith(y[k:]) for k in range(1, len(y)))) for y in texts)
            self.need[self.fast.token_to_id(x)] = self.margin if over else max(len(y) for y in texts if y.startswith(x))
        self.added_ids = frozenset(self.need)
        # each makes a token's match depend on the characters around it. single_word: cached "x foo bar", a cut at
        # "foo" keeps "x " apart from it, while in "x foobar" no token matches and " f" merges; the self-check passes
        bad = [t.content for t in added if t.lstrip or t.rstrip or t.normalized or t.single_word]
        if self.fast.truncation is not None or self.fast.padding is not None:
            # transformers sets these per call and a raw encode keeps them: a truncating tokenizer.json would cut the
            # cached ids and never the served ones
            self.off = f"the tokenizer truncates or pads ({self.fast.truncation}, {self.fast.padding})"
        elif bad:
            self.off = f"added tokens that strip, normalize or match single words: {bad[:4]}"
        elif not added:
            self.off = "no added tokens to cut at"
        else:
            self.off = self._self_check(texts) or self._template_check(tok)
        if self.off is not None:
            logger.warning("prompt cache off: %s", self.off)

    @classmethod
    def from_env(cls, tok) -> PromptCache | None:
        """The cache TRNRT_PROMPT_CACHE_MB sizes (MiB of host memory), or None when it is unset or 0."""
        mb = float(os.environ.get(ENV_MB, "0") or 0)
        if mb <= 0 or tok is None:
            return None
        return cls(tok, int(mb * 2**20), int(os.environ.get(ENV_AUDIT, "64") or 64))

    def _full(self, text: str):
        return self.fast.encode(text, add_special_tokens=False)  # type: ignore[union-attr]  # off when it is None

    def _bounds(self, enc, c0: int = 0, t0: int = 0) -> tuple[list[int], list[int]]:
        """(character start, token index) of every added token in an encoding of text[c0:], shifted to the whole
        text."""
        s, t = [], []
        for j, i in enumerate(enc.ids):
            if i in self.added_ids:
                s.append(enc.offsets[j][0] + c0)
                t.append(j + t0)
        return s, t

    def _cut(self, e: _Entry, text: str, shared: int | None = None) -> tuple[int, int] | None:
        """(character, token index) to cut at: the last added token of e's text at c with c + need <= the common prefix;
        None if there is none."""
        n = _lcp(e.text, text) if shared is None else shared
        j = bisect.bisect_left(e.bstart, n) - 1
        while j >= 0:
            c, i = e.bstart[j], e.btok[j]
            if c + self.need[e.ids[i]] <= n:
                return c, i
            j -= 1
        return None

    def encode(self, text: str, key: str | None = None, served: Callable[[], list[int]] | None = None) -> list[int]:
        """text's ids, as the tokenizer's encode(text, add_special_tokens=False) gives them. served: () -> the ids the
        uncached path serves for the same request (api.chat_prompt passes apply_chat_template(tokenize=True)), what an
        audit compares with; without it, the text's full encode."""
        if self.off is not None:
            return list(served()) if served is not None else self._full(text).ids
        if key is not None:
            k = f"s:{key}"
        else:
            k = (
                "h:"
                + hashlib.sha1(text[:KEY_CHARS].encode("utf-8", "surrogatepass"), usedforsecurity=False).hexdigest()
            )
        best: tuple[int, int] | None = None
        for eid in self.buckets.get(k, ()):  # the candidate sharing the longest prefix
            n = _lcp(self.entries[eid].text, text)
            if best is None or n > best[1]:
                best = (eid, n)
        cut = self._cut(self.entries[best[0]], text, best[1]) if best is not None else None
        if best is not None and cut is not None:
            e = self.entries[best[0]]
            c, i = cut
            tail = self._full(text[c:])
            ids = e.ids[:i].tolist() + tail.ids
            ts, tt = self._bounds(tail, c, i)
            j = bisect.bisect_left(e.btok, i)
            bstart, btok = e.bstart[:j].tolist() + ts, e.btok[:j].tolist() + tt
            self.counts["hits"] += 1
            self.counts["reused_tokens"] += i
            self.counts["tail_tokens"] += len(tail.ids)
            h = self.counts["hits"]
            if self.audit_every > 0 and (h <= AUDIT_FIRST or h % self.audit_every == 0):
                self.counts["audits"] += 1
                want = list(served()) if served is not None else self._full(text).ids
                if want != ids:
                    self.off = f"audit mismatch at hit {h} ({len(ids)} ids cached, {len(want)} served)"
                    logger.error("prompt cache off: %s", self.off)
                    self.entries.clear()
                    self.buckets.clear()
                    self.bytes = 0
                    return want
            self._drop(best[0])  # the new text continues that one
        else:
            enc = self._full(text)
            ids = enc.ids
            bstart, btok = self._bounds(enc)
            self.counts["misses"] += 1
        self._put(k, _Entry(text, array("I", ids), array("q", bstart), array("q", btok)))
        return ids

    def _drop(self, eid: int) -> None:
        e = self.entries.pop(eid)
        self.bytes -= e.nbytes
        b = self.buckets[e.key]
        b.remove(eid)
        if not b:
            del self.buckets[e.key]

    def _put(self, k: str, e: _Entry) -> None:
        if e.nbytes > self.max_bytes:
            return
        e.key = k
        eid, self._next = self._next, self._next + 1
        self.entries[eid] = e
        self.bytes += e.nbytes
        b = self.buckets.setdefault(k, [])
        b.append(eid)
        if len(b) > BUCKET[k[0]]:
            self._drop(b[0])
        while self.bytes > self.max_bytes:
            self._drop(next(iter(self.entries)))
            self.counts["evictions"] += 1

    def _self_check(self, added: list[str]) -> str | None:
        """The cut against the full encode on strings built from the tokenizer's own added tokens, at every cut point
        of them: None, or what differed."""
        a, b = added[0], added[min(1, len(added) - 1)]
        base = f"System line, café 中文 🙂\n{a}\nuser text  with  spaces\t{b} more words{a}\n\n tail{b}"
        texts = [base + extra for extra in ("", " and then some", "\n" + b + "x", "y" + a + " z", a, a + b)]
        pairs = [(old, new) for old in [base[:n] for n in range(1, len(base) + 1, 3)] + texts for new in texts]
        # the new text parting inside an added token that punctuation runs into: a cut there would split "!<|" in two
        pairs += [(f"x{p}{t}hello", f"x{p}{t[:k]}zz more") for t in added[:4] for p in "!}>" for k in (1, 2)]
        for old, new in pairs:
            enc = self._full(old)
            bs, bt = self._bounds(enc)
            cut = self._cut(_Entry(old, array("I", enc.ids), array("q", bs), array("q", bt)), new)
            if cut is None:
                continue
            c, i = cut
            if enc.ids[:i] + self._full(new[c:]).ids != self._full(new).ids:
                return f"self-check: a cut at character {c} of {old[:40]!r} -> {new[:40]!r} differs"
        return None

    def _template_check(self, tok) -> str | None:
        """The chat endpoint serves apply_chat_template(tokenize=True); the cache encodes the text it renders. The two
        must agree on a sample chat, or a transformers version adding special tokens on one path only would change the
        served ids (the audits catch it on real chats too): None, or what differed."""
        if not getattr(tok, "chat_template", None):
            return None  # the chat endpoint cannot render without one either
        m = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello, 世界! Ready?"},
            {"role": "assistant", "content": "Yes."},
            {"role": "user", "content": "Then go on."},
        ]
        try:
            want = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True)
            want = list(want["input_ids"] if hasattr(want, "keys") else want)
            text = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        except Exception as e:  # noqa: BLE001  any failure of the sample turns the cache off, never the server
            return f"template check failed: {type(e).__name__}: {e}"[:200]
        if want != self._full(text).ids:
            return "template check: apply_chat_template(tokenize=True) differs from the rendered text's encode"
        return None

    def stats(self) -> dict:
        return dict(
            self.counts,
            on=self.off is None,
            off=self.off,
            entries=len(self.entries),
            buckets=len(self.buckets),
            bytes=self.bytes,
            max_bytes=self.max_bytes,
            audit_every=self.audit_every,
        )
