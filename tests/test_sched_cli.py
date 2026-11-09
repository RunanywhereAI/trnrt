"""The scheduling flags (trnrt/__main__.py add_scheduling_args, scheduling): SGLang's names, host only, refused before
any engine starts when the scheduler or the prefix cache would refuse them, and handed by serve to every engine."""

from __future__ import annotations

import pytest

pytest.importorskip("httpx")
pytest.importorskip("uvicorn")

import trnrt.__main__ as cli
from tests import test_serve_cli as base
from trnrt.engine.scheduler import Scheduling

host, engine, stand_in = base.host, base.engine, base.stand_in  # its fixtures: fake engine processes, no device

BASE = ["--model", "m", "--device", "cpu"]


def test_no_flag_is_the_default_scheduling():
    assert cli.scheduling(BASE) == Scheduling()


def test_the_flags_give_the_scheduling_settings():
    s = cli.scheduling(
        [
            *BASE,
            "--schedule-policy",
            "lof",
            "--enable-priority-scheduling",
            "--schedule-low-priority-values-first",
            "--priority-scheduling-preemption-threshold",
            "3",
            "--disable-priority-preemption",
            "--max-queued-requests",
            "64",
            "--max-queued-tokens",
            "100000",
            "--radix-eviction-policy",
            "slru",
            "--radix-eviction-policy-config",
            '{"protected_threshold": 4}',
        ]
    )
    assert s == Scheduling(
        policy="lof",
        priority_scheduling=True,
        low_priority_first=True,
        priority_preemption=False,
        priority_preemption_threshold=3,
        max_queued_requests=64,
        max_queued_tokens=100000,
        eviction_policy="slru",
        eviction_config={"protected_threshold": 4},
    )


def test_the_scheduling_flags_change_no_engine_config_field():
    """Host only: the EngineConfig (whose fields key the captured graphs) is the same with every flag set."""
    flags = [
        "--schedule-policy",
        "priority",
        "--max-queued-requests",
        "8",
        "--radix-eviction-policy",
        "tlru",
        "--radix-eviction-policy-config",
        '{"threshold": 4096, "next_prompt_estimate": 512}',
    ]
    assert cli.engine_config([*BASE, *flags]) == cli.engine_config(BASE)


@pytest.mark.parametrize(
    ("flags", "why"),
    [
        (["--schedule-policy", "lpm", "--enable-priority-scheduling"], "not lpm"),
        (["--radix-eviction-policy", "tlru"], "tlru eviction needs"),
        (["--radix-eviction-policy", "lru", "--radix-eviction-policy-config", '{"threshold": 3}'], "no setting"),
        (
            ["--radix-eviction-policy", "slru", "--radix-eviction-policy-config", '{"protected_threshold": -2}'],
            "non-neg",
        ),
    ],
)
def test_a_combination_the_engine_would_refuse_is_refused_by_serve_before_any_engine_starts(host, flags, why, capsys):
    with pytest.raises(SystemExit) as e:
        cli.cmd_serve([*BASE, *flags])
    assert e.value.code == 2 and why in capsys.readouterr().err
    assert host.engines == []


@pytest.mark.parametrize("flags", [["--radix-eviction-policy-config", "[1]"], ["--radix-eviction-policy-config", "{x"]])
def test_the_eviction_config_must_be_a_json_object(host, flags, capsys):
    with pytest.raises(SystemExit) as e:
        cli.cmd_serve([*BASE, *flags])
    assert e.value.code == 2 and "--radix-eviction-policy-config" in capsys.readouterr().err


def test_serve_hands_the_scheduling_flags_to_every_engine(stand_in):
    flags = ["--enable-priority-scheduling", "--max-queued-requests", "16", "--radix-eviction-policy", "priority"]
    assert cli.cmd_serve([*BASE, "--port", "9100", *flags]) == 0
    want = Scheduling(priority_scheduling=True, max_queued_requests=16, eviction_policy="priority")
    assert stand_in.schedulings == [want, want]
    stand_in.schedulings.clear()
    assert cli.cmd_serve([*BASE, "--port", "9100"]) == 0
    assert stand_in.schedulings == [Scheduling(), Scheduling()]
