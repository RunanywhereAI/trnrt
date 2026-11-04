"""The context-parallel prefill's local-K counters (models/mla.py CP_LOCAL_K: each DSA layer's cp_localk buffer,
[rows, failed the certificate, failed past CP_LOCAL_F]), which the graphs add to and the host reads back:

- engine.local_k_counts sums them over the layers (None for a model without them), and local_k_report turns them into
  the line LLMEngine.close prints: EXACT only with no row past CP_LOCAL_F;
- LLMEngine.close prints it once for a tensor-parallel engine with the counters, never after a failure (a step on a
  dead collective would never let the counters be read), and nothing for an engine without them;
- /metrics carries them as three counters, which the engine thread reads only while idle, only once a scrape asked for
  them and a step ran since its last read; an engine without them is asked once.

No model is built: the layers, the engine and its loop are stand-ins.
"""

from __future__ import annotations

import logging
import time
import types

import pytest
import torch

from trnrt.engine import engine as E
from trnrt.metrics import Metrics, local_k_lines
from trnrt.models import mla


class _Model:
    def __init__(self, *counts):
        self.layers = [types.SimpleNamespace(cp_localk=torch.tensor(c, dtype=torch.float32)) for c in counts]
        self.layers.append(types.SimpleNamespace())  # a DSA layer outside the row groups' path: no counters

    def kv_layers(self):
        return self.layers


def test_counts_are_summed_over_the_layers():
    assert E.local_k_counts(_Model([100, 3, 0], [50, 1, 0])) == [150.0, 4.0, 0.0]
    assert E.local_k_counts(_Model()) is None


def test_the_report_says_exact_only_with_nothing_past_f():
    line = E.local_k_report(None, [150.0, 4.0, 0.0], "this engine's run")
    assert line == (
        f"local-K this engine's run: 4 of 150 rows failed the certificate (2.67e-02), 0 past CP_LOCAL_F = "
        f"{mla.CP_LOCAL_F} (EXACT)"
    )
    assert E.local_k_report([100.0, 1.0, 0.0], [150.0, 4.0, 2.0], "x").endswith(
        f"2 past CP_LOCAL_F = {mla.CP_LOCAL_F} (NOT EXACT)"
    )
    assert "3 of 50 rows" in E.local_k_report([100.0, 1.0, 0.0], [150.0, 4.0, 2.0], "x")
    assert E.local_k_report(None, None, "x") is None


def _engine(model):
    """An LLMEngine that went through none of __init__: what close reads of a tensor-parallel engine."""
    eng = object.__new__(E.LLMEngine)
    eng.model = model
    eng.runner = types.SimpleNamespace(tp_send=lambda msg: None, watchdog=None, model=model)
    eng._workers, eng._tp_saved = [], None
    E._OPEN.add(eng)
    return eng


@pytest.fixture
def written(capsys, caplog):
    """What close wrote so far, one entry per line: its log records and anything it printed."""
    caplog.set_level(logging.INFO, logger="trnrt")

    def lines() -> list[str]:
        out = capsys.readouterr().out.splitlines() + caplog.messages
        caplog.clear()
        return out

    return lines


def test_close_prints_the_report_once(written):
    eng = _engine(_Model([100, 3, 0], [50, 1, 0]))
    eng.close()
    assert written() == [
        (
            f"local-K this engine's run: 4 of 150 rows failed the certificate (2.67e-02), 0 past CP_LOCAL_F = "
            f"{mla.CP_LOCAL_F} (EXACT)"
        )
    ]
    eng.close()  # idempotent: no second report
    assert written() == []


def test_close_reports_nothing_after_a_failure_or_without_counters(written):
    failed = _engine(_Model([100, 3, 1]))
    failed.close_failed()  # a rank failed: nothing more goes to the workers, and no device read either
    failed.close()
    dead = _engine(_Model([100, 3, 1]))
    dead._workers = [types.SimpleNamespace(exitcode=-9, pid=None)]
    dead.close()
    _engine(_Model()).close()
    assert written() == []
    control = _engine(_Model([100, 3, 1]))  # the same engine, healthy: the report, NOT EXACT
    control.close()
    assert "NOT EXACT" in "\n".join(written())


def _stub():
    return types.SimpleNamespace(
        scheduler=types.SimpleNamespace(running=[], waiting=[], num_preemptions=0),
        pool=types.SimpleNamespace(num_usable=100, num_free=90),
        radix=types.SimpleNamespace(evictable_pages=4),
    )


def test_metrics_render_the_counts_last_read():
    m = Metrics()
    assert local_k_lines(None) == [] and "local_k" not in m.render(_stub())
    assert m.local_k_wanted  # the render asked for them
    m.set_local_k([150.0, 4.0, 0.0])
    assert not m.local_k_wanted
    text = m.render(_stub())
    for line in (
        "# TYPE trnrt:dsa_local_k_rows_total counter",
        "trnrt:dsa_local_k_rows_total 150",
        "trnrt:dsa_local_k_failed_rows_total 4",
        "trnrt:dsa_local_k_inexact_rows_total 0",
    ):
        assert line in text.splitlines(), line


class _LoopEngine:
    """What EngineLoop touches of an idle engine, with local_k_counts counting its reads (None: no counters)."""

    def __init__(self, counts):
        self.cfg = types.SimpleNamespace(tp=1)
        self.mcfg = types.SimpleNamespace(vocab_size=10)
        self.counts, self.reads, self.work = counts, 0, 0
        self.scheduler, self.pool, self.radix = _stub().scheduler, _stub().pool, _stub().radix

    def local_k_counts(self):
        self.reads += 1
        return self.counts

    def has_work(self):
        return self.work > 0

    def step(self):
        self.work -= 1

    def dead_ranks(self):
        return []

    def close(self):
        pass


def _until(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end
        time.sleep(0.01)


@pytest.fixture
def loop():
    pytest.importorskip("fastapi")
    from trnrt.serve import api

    made = []

    def start(engine):
        made.append(api.EngineLoop(engine))
        return made[-1]

    yield start
    for lp in made:
        lp.stop()


def test_the_loop_reads_the_counters_when_idle_and_asked(loop):
    eng = _LoopEngine([150.0, 4.0, 0.0])
    lp = loop(eng)
    time.sleep(0.3)
    assert eng.reads == 0  # no scrape asked yet
    lp.metrics.render(eng)
    _until(lambda: lp.metrics.local_k is not None)
    assert eng.reads == 1 and "trnrt:dsa_local_k_rows_total 150" in lp.metrics.render(eng)
    time.sleep(0.3)
    assert eng.reads == 1  # asked again, but no step ran since the read
    eng.counts = [200.0, 5.0, 0.0]
    eng.work = 2  # two steps, then idle again
    _until(lambda: eng.work == 0)
    _until(lambda: eng.reads == 2)
    assert "trnrt:dsa_local_k_rows_total 200" in lp.metrics.render(eng)


def test_an_engine_without_counters_is_asked_once(loop):
    eng = _LoopEngine(None)
    lp = loop(eng)
    lp.metrics.render(eng)
    _until(lambda: eng.reads == 1)
    eng.work = 1
    _until(lambda: eng.work == 0)
    lp.metrics.render(eng)
    time.sleep(0.3)
    assert eng.reads == 1 and "local_k" not in lp.metrics.render(eng)
