"""The speculative configurations (trnrt/__main__.py PRESETS mtp-*, EngineConfig's speculation fields):

- each mtp preset sets exactly its documented values: the ones below are the shipped configurations', row for row, so a
  change to a preset shows up here;
- the serving defaults and the batch presets are untouched by the new fields (speculation off, nothing else moved);
- the recurrent-state pool a preset implies has the row counts its graphs were compiled at (34, 34, 34 and 58 rows);
- every inconsistent combination is refused, each with a positive control that the valid one builds.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from trnrt import __main__ as cli

if TYPE_CHECKING:
    from trnrt.config import EngineConfig

SPEC_FIELDS = (
    "spec_method",
    "spec_k",
    "spec_synth_al",
    "spec_max_batch",
    "spec_hysteresis",
    "spec_bucket_max",
    "plain_bucket_min",
    "spec_page_buckets",
    "mtp_fill_rows",
    "mtp_chunk_q",
    "mtp_kv_only",
    "dyn_mtp_kv",
    "state_seats",
    "state_spec_seats",
    "dense_half_skip_rows",
)

# What the shipped configurations set, by preset: engine shapes, then speculation.
MTP = {
    "spec_method": "mtp",
    "spec_k": 2,
    "mtp_chunk_q": 64,
    "mtp_fill_rows": 1024,
    "max_num_seqs": 16,
    "prefill_token_buckets": (512, 1024, 2048, 4096),
    "prefill_page_buckets": (1024, 4096),
    "prefill_page_min_rows": 2048,
    "kda_shell_max_rows": 12,
    "dsa_layer_max_rows": 16,
    "ffn_half_below": 17,
    "dsa_index_rows": 8,
    "kv_cache_gb": 4.0,
    "page_size": 256,
}
DYN = {
    **MTP,
    "page_buckets": (1024, 3072, 4096),
    "spec_max_batch": 4,
    "spec_hysteresis": 1,
    "spec_bucket_max": 4,
    "plain_bucket_min": 4,
    "spec_page_buckets": (1024, 4096),
    "mtp_kv_only": True,
    "dyn_mtp_kv": True,
}
SEATS8 = {"max_running": 8, "state_seats": 8, "state_checkpoints": 8, "host_cache_gb": 4.0}
EXPECTED = {
    "mtp-batch8": {**DYN, **SEATS8, "decode_batch_buckets": (1, 2, 4, 6, 8)},
    "mtp-batch2": {
        **MTP,
        **SEATS8,
        "page_buckets": (1024, 4096),
        "prefill_token_buckets": (512, 1024, 4096),
        "decode_batch_buckets": (1, 2, 4, 6, 8),
    },
    "mtp-batch12": {
        **DYN,
        "decode_batch_buckets": (1, 2, 4, 8, 12),
        "max_running": 12,
        "state_seats": 12,
        "state_spec_seats": 4,
        "state_checkpoints": 12,
        "host_cache_gb": 8.0,
    },
    "mtp-batch24": {
        **DYN,
        "max_num_seqs": 32,
        "decode_batch_buckets": (1, 2, 4, 16, 24),
        "max_running": 24,
        "state_seats": 24,
        "state_spec_seats": 4,
        "state_checkpoints": 24,
        "plain_bucket_min": 16,
        "host_cache_gb": 12.0,
        "dsa_layer_max_rows": 24,
        "ffn_half_below": 25,
        "dense_half_skip_rows": (2, 18, 24),
    },
}
# The recurrent-state pool's rows: the scratch row, the running seats' rows (1 + spec_k for a seat that speculates), the
# checkpoint rows and the zero row.
STATE_ROWS = {"mtp-batch8": 34, "mtp-batch2": 34, "mtp-batch12": 34, "mtp-batch24": 58}


def _cfg(*argv) -> EngineConfig:
    return cli.engine_config(["--model", "m", *argv])


def seat_rows(c: EngineConfig) -> int:
    seats = c.state_seats or c.max_num_seqs
    spec = seats if c.state_spec_seats is None else c.state_spec_seats
    return 1 + spec * (1 + c.spec_k) + (seats - spec) + (c.state_checkpoints or 2 * c.max_num_seqs) + 1


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_a_preset_sets_the_shipped_values(name):
    c = _cfg("--preset", name)
    for field, want in EXPECTED[name].items():
        assert getattr(c, field) == want, (name, field)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_a_preset_changes_nothing_but_what_it_names(name):
    served = _cfg()
    c = _cfg("--preset", name)
    named = {f.name for f in dataclasses.fields(c) if f.name in EXPECTED[name]}
    moved = {f.name for f in dataclasses.fields(c) if getattr(c, f.name) != getattr(served, f.name)}
    assert moved <= named | set(SPEC_FIELDS), moved - named - set(SPEC_FIELDS)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_state_pool_has_the_rows_the_graphs_were_compiled_at(name):
    assert seat_rows(_cfg("--preset", name)) == STATE_ROWS[name]


def test_speculation_is_off_by_default_and_in_the_batch_presets():
    for argv in ((), ("--preset", "batch12"), ("--preset", "batch8")):
        c = _cfg(*argv)
        assert c.spec_method is None and not c.spec_max_batch and not c.mtp_kv_only and not c.dyn_mtp_kv
        assert c.spec_synth_al is None and c.state_seats is None and c.dense_half_skip_rows == (2,)


def test_an_explicit_argument_wins_over_a_preset():
    c = _cfg("--preset", "mtp-batch12", "--host-cache-gb", "12", "--spec-max-batch", "3", "--no-mtp-kv-only")
    assert (c.host_cache_gb, c.spec_max_batch, c.mtp_kv_only) == (12.0, 3, False)


def test_the_synthetic_acceptance_comes_from_the_flag_or_the_environment(monkeypatch):
    monkeypatch.delenv("TRNRT_SPEC_SYNTH_AL", raising=False)
    assert _cfg("--preset", "mtp-batch8").spec_synth_al is None
    assert _cfg("--preset", "mtp-batch8", "--spec-synth-al", "2.5").spec_synth_al == 2.5
    monkeypatch.setenv("TRNRT_SPEC_SYNTH_AL", "2.518506")
    assert _cfg("--preset", "mtp-batch8").spec_synth_al == 2.518506
    assert _cfg("--preset", "mtp-batch8", "--spec-synth-al", "2.0").spec_synth_al == 2.0
    with pytest.raises(ValueError, match="needs spec_method"):
        _cfg()  # the variable is set but speculation is off: refused, not silently ignored


@pytest.mark.parametrize(
    ("argv", "why"),
    [
        (("--spec-max-batch", "4"), "spec_max_batch needs spec_method"),
        (("--spec-page-buckets", "1024,4096"), "spec_page_buckets needs spec_method"),
        (("--mtp-kv-only",), "mtp_kv_only needs spec_method"),
        (("--spec-method", "mtp", "--page-size", "96"), "power-of-two page size"),
        (("--spec-method", "mtp", "--spec-synth-al", "3.5"), r"in \[1, spec_k \+ 1\]"),
        (("--spec-method", "mtp", "--spec-bucket-max", "4"), "need spec_max_batch"),
        (("--spec-method", "mtp", "--state-spec-seats", "4"), "needs spec_max_batch"),
        (("--spec-method", "mtp", "--dyn-mtp-kv"), "needs spec_max_batch"),
        (("--spec-method", "mtp", "--mtp-kv-only"), "needs mtp_chunk_q"),
        (("--spec-method", "mtp", "--spec-max-batch", "4", "--state-spec-seats", "3"), "must cover spec_max_batch 4"),
        (("--spec-method", "mtp", "--spec-max-batch", "4", "--state-spec-seats", "99"), "fit the"),
        (("--state-seats", "4"), "below the running cap 12"),
        (("--state-seats", "99"), "above max_num_seqs"),
    ],
)
def test_an_inconsistent_combination_is_refused(argv, why, monkeypatch):
    monkeypatch.delenv("TRNRT_SPEC_SYNTH_AL", raising=False)
    with pytest.raises(ValueError, match=why):
        _cfg(*argv)


def test_the_valid_neighbours_of_the_refused_combinations_build():
    ok = (
        ("--spec-method", "mtp", "--spec-max-batch", "4", "--state-spec-seats", "4"),
        ("--spec-method", "mtp", "--spec-max-batch", "4", "--spec-bucket-max", "4", "--plain-bucket-min", "4"),
        ("--spec-method", "mtp", "--spec-synth-al", "3.0"),
        ("--spec-method", "mtp", "--mtp-chunk-q", "64", "--mtp-kv-only"),
        ("--state-seats", "12"),
    )
    for argv in ok:
        _cfg(*argv)
