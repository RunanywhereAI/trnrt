"""Speculative decoding's counters as the server exposes them (trnrt/metrics.py): the spec_decode_* lines of GET /metrics, under
vLLM's names, and the spec block of GET /v1/stats. Nothing is exposed for an engine that does not speculate."""

from __future__ import annotations

import types

from trnrt.metrics import spec_lines, spec_stats


def engine(**kw):
    cfg = types.SimpleNamespace(spec_method="mtp", spec_k=2, spec_max_batch=0, spec_hysteresis=1, spec_synth_al=None)
    for k in ("spec_max_batch", "spec_hysteresis", "spec_synth_al"):
        if k in kw:
            setattr(cfg, k, kw.pop(k))
    base = {
        "cfg": cfg,
        "spec_proposed": 40,
        "spec_accepted": 26,
        "spec_verifies": 20,
        "spec_pos_reached": [20, 14],
        "spec_pos_accepted": [17, 9],
        "dyn_flips": 3,
        "dyn_plain_steps": 11,
        "dyn_spec_steps": 29,
        "dyn_plain": False,
    }
    return types.SimpleNamespace(**{**base, **kw})


def test_the_counters_use_vllms_spec_decode_names():
    lines = spec_lines(engine())
    assert "trnrt:spec_decode_num_drafts_total 20" in lines
    assert "trnrt:spec_decode_num_draft_tokens_total 40" in lines
    assert "trnrt:spec_decode_num_accepted_tokens_total 26" in lines
    assert 'trnrt:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 17' in lines
    assert 'trnrt:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 9' in lines
    assert not [x for x in lines if "spec_mode" in x or "spec_steps" in x], "no dynamic counters for a static engine"


def test_a_dynamic_engine_adds_its_flips_and_the_steps_of_each_mode():
    lines = spec_lines(engine(spec_max_batch=4))
    assert "trnrt:spec_mode_flips_total 3" in lines
    assert (
        'trnrt:spec_steps_total{mode="speculative"} 29' in lines and 'trnrt:spec_steps_total{mode="plain"} 11' in lines
    )


def test_the_stats_block_gives_the_acceptance_rate_and_the_mean_acceptance_length():
    s = spec_stats(engine(spec_synth_al=2.518506))
    assert s["acceptance"] == 26 / 40 and s["accept_length"] == 1 + 26 / 20 and s["synth_al"] == 2.518506
    assert s["per_position"] == [{"reached": 20, "accepted": 17}, {"reached": 14, "accepted": 9}] and "dyn" not in s
    assert spec_stats(engine(spec_max_batch=4))["dyn"] == {
        "max_batch": 4,
        "hysteresis": 1,
        "plain_now": False,
        "flips": 3,
        "plain_steps": 11,
        "spec_steps": 29,
    }


def test_before_the_first_verify_the_rates_are_none_not_a_division_by_zero():
    s = spec_stats(engine(spec_proposed=0, spec_accepted=0, spec_verifies=0, spec_pos_reached=[], spec_pos_accepted=[]))
    assert s["acceptance"] is None and s["accept_length"] is None and s["per_position"] == []


def test_an_engine_that_does_not_speculate_exposes_nothing():
    plain = types.SimpleNamespace(cfg=types.SimpleNamespace(spec_method=None))
    assert spec_lines(plain) == [] and spec_stats(plain) is None
