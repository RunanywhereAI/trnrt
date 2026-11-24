"""The graphs --structured-outputs and --penalties add to warmup (and so to a capture), per shipped preset, on
tests/test_spec_runner_stub.py's recording stub model: the variants exactly at the plain decode family's buckets and page
rungs and at every prefill bucket and rung, none beside a verify, and nothing at all without the flags."""

from __future__ import annotations

import types

import pytest
import torch

from tests import test_spec_runner_stub as S
from trnrt import __main__ as cli
from trnrt.config import prefill_pages
from trnrt.engine import model_runner as mr

VOCAB = 154_880  # GLM-5.3-Flash's
PRESETS = ["batch12", "batch8", "mtp-batch8", "mtp-batch12", "mtp-batch24"]


def build(preset, monkeypatch, *extra):
    ecfg = cli.engine_config(["--model", "m", "--preset", preset, "--device", "cpu", "--tp", "1", *extra])
    log: list = []
    monkeypatch.setattr(mr, "_piecewise", lambda *a, **k: S._recorders(log))
    mcfg = types.SimpleNamespace(vocab_size=VOCAB, eos_token_ids=(1,))
    return mr.ModelRunner(S.StubModel(), mcfg, ecfg, 8, torch.device("cpu")), ecfg


def tagged(runner):
    """The warmed keys that carry a variant tag."""
    return {k for k in runner.compile_seconds if any(isinstance(x, str) for x in k[1:])}


def expected(ecfg, tags):
    dec = ecfg.resolved_decode_batch_buckets()
    plain = [b for b in dec if not ecfg.plain_bucket_min or b >= ecfg.plain_bucket_min]
    out = {("decode", B, P, *t) for B in plain for P in ecfg.resolved_page_buckets() for t in tags}
    for C in ecfg.resolved_prefill_token_buckets():
        for P in prefill_pages(ecfg.resolved_prefill_page_buckets(), ecfg.prefill_page_min_rows, C):
            out |= {("prefill", C, P, *t) for t in tags}
    return out


@pytest.mark.parametrize("preset", PRESETS)
def test_without_the_flags_warmup_adds_nothing(preset, monkeypatch):
    runner, _ = build(preset, monkeypatch)
    runner.warmup()
    assert tagged(runner) == set()


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize(
    ("flags", "tags"),
    [
        (["--structured-outputs"], [("grammar",)]),
        (["--penalties"], [("pen1024",), ("pen16384",), (f"pen{VOCAB}",)]),
        (
            ["--structured-outputs", "--penalties"],
            [("grammar",), ("grammar", "pen1024"), ("grammar", "pen16384"), ("grammar", f"pen{VOCAB}")],
        ),
    ],
)
def test_the_flags_warm_exactly_their_variants(preset, flags, tags, monkeypatch):
    runner, ecfg = build(preset, monkeypatch, *flags)
    runner.warmup()
    got, want = tagged(runner), expected(ecfg, tags)
    assert got == want, (sorted(got - want)[:4], sorted(want - got)[:4])
    assert not [k for k in got if k[0] == "verify"]


def test_batch12_counts(monkeypatch):
    """Literal counts for the served preset with --structured-outputs: decode buckets 1 2 4 6 8 12 x pages 512 1024 4096,
    and prefill 512 1024 (the 4096-page rung only, below 2048 rows) and 2048 4096 (both rungs)."""
    runner, _ = build("batch12", monkeypatch, "--structured-outputs")
    runner.warmup()
    keys = tagged(runner)
    assert len([k for k in keys if k[0] == "decode"]) == 18
    assert sorted(k[1:3] for k in keys if k[0] == "prefill") == [
        (512, 4096),
        (1024, 4096),
        (2048, 1024),
        (2048, 4096),
        (4096, 1024),
        (4096, 4096),
    ]
