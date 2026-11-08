"""trnrt/serve/api.py's _Detok: the incremental mode (TGI's prefix / read offsets, the server's streams) emits exactly
what the full decode emits, push by push, with stop strings, on byte-level tokens that split multi-byte characters. A
byte-level tokenizer stub (each token a byte string, decoded with errors="replace", as a byte-level BPE decoder replaces an
incomplete character) stands in for the served one.

GLM-5.3-Flash's own split below is the case an earlier form failed: a window ending inside a character returned only the
committed text, so a stop string before that character was found one push late. The control re-runs that form."""

from __future__ import annotations

import random

import pytest


class ByteTok:
    def __init__(self, vocab: list[bytes], special=()):
        self.vocab, self.special = vocab, set(special)

    def decode(self, ids, skip_special_tokens=True):
        return b"".join(self.vocab[i] for i in ids if not (skip_special_tokens and i in self.special)).decode(
            "utf-8", "replace"
        )


# GLM-5.3-Flash's own split of "mixed 中文 English 日本語 한국어 العربية עברית" (token bytes from the served tokenizer)
GLM = [
    b"mixed",
    b" \xe4\xb8\xad\xe6\x96\x87",
    b" English",
    b" \xe6\x97\xa5\xe6\x9c\xac",
    b"\xe8\xaa\x9e",
    b" \xed\x95\x9c\xea\xb5\xad",
    b"\xec\x96\xb4",
    b" \xd8\xa7\xd9\x84\xd8\xb9\xd8\xb1\xd8\xa8\xd9\x8a\xd8\xa9",
    b" \xd7",
    b"\xa2",
    b"\xd7\x91",
    b"\xd7\xa8",
    b"\xd7\x99\xd7",
    b"\xaa",
]


def _run(D, tok, ids, stops, chunks, incremental):
    d, pos, out = D(tok, stops, incremental=incremental), 0, []
    for j, k in enumerate(chunks):
        out.append(d.push(ids[pos : pos + k], j == len(chunks) - 1))
        pos += k
    return out, d.stopped


def _old_branch(D):
    """The pre-fix _decoded (the control): an uncommitted window returns only the committed text + U+FFFD."""

    class Old(D):
        def _decoded(self, final):
            if final or not self.incremental:
                return self.tok.decode(self.ids, skip_special_tokens=True)
            prefix = self.tok.decode(self.ids[self._po : self._ro], skip_special_tokens=True)
            new = self.tok.decode(self.ids[self._po :], skip_special_tokens=True)
            if len(new) > len(prefix) and not new.endswith("\ufffd"):
                self._text += new[len(prefix) :]
                self._po, self._ro = self._ro, len(self.ids)
                return self._text
            return self._text + "\ufffd"

    return Old


def test_glms_split_stop_case_now_matches_and_the_old_branch_does_not():
    from trnrt.serve.api import _Detok

    tok, ids, stops, ch = ByteTok(GLM), list(range(len(GLM))), ("本語 한국어 ",), (3, 6, 1, 4)
    full = _run(_Detok, tok, ids, stops, ch, False)
    assert full == (["mixed 中文 E", "nglish 日", "", ""], True)
    assert _run(_Detok, tok, ids, stops, ch, True) == full
    assert _run(_old_branch(_Detok), tok, ids, stops, ch, True) == (["mixed 中文 E", "", "nglish 日", ""], True)


TEXTS = [
    "mixed 中文 English 日本語 한국어 العربية עברית",
    "emoji 👩‍💻🚀🇮🇳 then ascii",
    "µs × — … “q” ½ °C 數據",  # noqa: RUF001  the ambiguous characters are the test data
    "def f(x):\n    return {'a': [1, 2]}  # 注释",
    "é ñ ﬁ 𝕏𝔸𝔹 𓀀 ꙮ 🫠",  # noqa: RUF001  the ambiguous characters are the test data
]


@pytest.mark.parametrize("seed", range(4))
def test_random_byte_splits_chunkings_and_stops_match_push_by_push(seed):
    """Random byte-level tokenizations (cut anywhere, inside characters too), special tokens that decode to nothing, random
    chunkings, and stop strings from the text: every push equal. The old branch must differ somewhere (the control)."""
    from trnrt.serve.api import _Detok

    rng, old_diffs, runs = random.Random(seed), 0, 0
    for _ in range(100):
        b = rng.choice(TEXTS).encode()
        cuts = sorted(rng.sample(range(1, len(b)), k=min(len(b) - 1, rng.randint(3, 20))))
        vocab = [b[a:z] for a, z in zip([0, *cuts], [*cuts, len(b)])] + [b"<SPECIAL>"]
        sp = len(vocab) - 1
        ids = list(range(len(vocab) - 1))
        for _ in range(rng.randint(0, 3)):
            ids.insert(rng.randrange(len(ids) + 1), sp)
        tok = ByteTok(vocab, special=(sp,))
        full_text = tok.decode(ids)
        i = rng.randrange(len(full_text))
        stop_sets = [(), (full_text[i : i + rng.randint(1, 6)],)]
        if rng.random() < 0.3:
            ids = ids[: rng.randint(1, len(ids))]  # a stream cut by max_tokens, possibly inside a character
        for stops in stop_sets:
            ch, left = [], len(ids)
            while left:
                k = min(left, rng.randint(1, 5))
                ch.append(k)
                left -= k
            runs += 1
            want = _run(_Detok, tok, ids, stops, ch, False)
            assert _run(_Detok, tok, ids, stops, ch, True) == want, (vocab, ids, stops, ch)
            old_diffs += _run(_old_branch(_Detok), tok, ids, stops, ch, True) != want
    assert runs == 200 and old_diffs > 0, (
        f"the control never differed in {runs} runs: this test can't see the bug class"
    )
