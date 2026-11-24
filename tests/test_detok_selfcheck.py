"""The streaming self-check (trnrt/serve/api.py DETOK_CHECK): an incremental stream, when it ends, checks that what it
streamed is a prefix of the full decode of its ids (all of it when no stop string cut it), counts each check, and prints
the rid and offset of a mismatch. A corrupted delta must be counted; a stop-string cut must not count as one; the reference
(non-incremental) mode never checks. A WordLevel tokenizer whose word t<i> is token i."""

from __future__ import annotations

import logging

import pytest

from tests.test_api_prompt_thread import _save_tokenizer


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    from transformers import AutoTokenizer

    d = tmp_path_factory.mktemp("detok_check")
    _save_tokenizer(d, 64)
    return AutoTokenizer.from_pretrained(str(d))


@pytest.fixture
def check(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="trnrt")  # the self-check logs its mismatches
    from trnrt.serve import api

    c = {"checked": 0, "mismatch": 0}
    monkeypatch.setattr(api, "DETOK_CHECK", c)
    return c


def _stream(D, tok, ids, stops=(), chunk=2, **kw):
    d = D(tok, stops, **kw)
    out = [d.push(ids[i : i + chunk], i + chunk >= len(ids)) for i in range(0, len(ids), chunk)]
    return d, "".join(out)


IDS = [5, 17, 42, 9, 3, 25, 11, 7, 30, 12]


def test_a_clean_stream_checks_once_and_matches(tok, check, caplog):
    from trnrt.serve.api import _Detok

    _, text = _stream(_Detok, tok, IDS, incremental=True, rid="r-clean")
    assert text == tok.decode(IDS, skip_special_tokens=True)
    assert check == {"checked": 1, "mismatch": 0}
    assert "mismatch" not in caplog.text


def test_a_stop_string_cut_compares_the_stopped_prefix(tok, check, caplog):
    from trnrt.serve.api import _Detok

    full = tok.decode(IDS, skip_special_tokens=True)
    stop = "t25"
    assert stop in full
    d, text = _stream(_Detok, tok, IDS, stops=(stop,), incremental=True, rid="r-stop")
    assert d.stopped and text == full[: full.index(stop)]
    assert check == {"checked": 1, "mismatch": 0}  # one check, at the stop push; no false mismatch
    assert "mismatch" not in caplog.text


def test_a_corrupted_delta_is_counted_with_its_rid_and_offset(tok, check, caplog):
    from trnrt.serve.api import _Detok

    class Corrupt(
        _Detok
    ):  # the planted fault: the newest character of the second push's text changed before it is streamed
        n = 0

        def _decoded(self, final):
            t = super()._decoded(final)
            if not final:
                self.n += 1
                if self.n == 2:
                    t = t[:-1] + "X"
            return t

    at = len(tok.decode(IDS[:4], skip_special_tokens=True)) - 1  # the second push (2 tokens a push) ends at IDS[:4]
    _, text = _stream(Corrupt, tok, IDS, incremental=True, rid="r-planted")
    assert text[at] == "X"
    assert check == {"checked": 1, "mismatch": 1}
    out = caplog.text
    assert "detok self-check mismatch 1 of 1: rid r-planted" in out and f"first difference at offset {at}" in out, out


def test_the_reference_mode_never_checks(tok, check):
    from trnrt.serve.api import _Detok

    _stream(_Detok, tok, IDS, incremental=False, rid="r-default")
    _stream(_Detok, tok, IDS, stops=("t25",), incremental=False)
    assert check == {"checked": 0, "mismatch": 0}
