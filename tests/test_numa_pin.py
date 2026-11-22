"""engine/tp.py numa_pin fails loud: each tensor-parallel rank is pinned to its NeuronCore's NUMA node, and a device
listing that fails stops the start instead of leaving every rank unpinned with a warning each (easy to miss, and the
ranks' host work then crosses the socket on every step). On the host, with a synthetic trn2.48xlarge listing (16 devices,
4 logical cores each at LNC 2; devices 0-3 and 12-15 on node 1, 4-11 on node 0) and os.sched_getaffinity /
sched_setaffinity faked (macOS has neither):
- any failure of the listing raises NumaPinError before any affinity change; TRNRT_NUMA_PIN_OPTIONAL=1 runs unpinned,
  with a warning and an "off" NUMA-PIN line;
- rank 0 logs the engine's NUMA-PIN line (on, partial), the other ranks none;
- neuron-ls's two JSON forms are read and anything else refused; the newer flag is tried when the old one fails;
- fmt_cpus is parse_cpulist's inverse."""

from __future__ import annotations

import json
import logging
import os
import subprocess

import pytest

from trnrt.engine import tp as TP

NODE0 = "0-47,96-143"
NODE1 = "48-95,144-191"
LS = [
    {
        "neuron_device": d,
        "nc_count": 4,
        "logical_neuroncore_config": 2,
        "cpu_affinity": NODE1 if d < 4 or d >= 12 else NODE0,
    }
    for d in range(16)
]
ALL = set(range(192))


@pytest.fixture
def affinity(monkeypatch):
    state = {"mask": set(ALL), "sets": 0}

    def setaff(pid, mask):
        assert pid == 0
        state["mask"], state["sets"] = set(mask), state["sets"] + 1

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(state["mask"]), raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", setaff, raising=False)
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    monkeypatch.delenv(TP.PIN_OPTIONAL, raising=False)
    return state


@pytest.mark.parametrize(
    "err",
    [
        FileNotFoundError("neuron-ls"),
        ValueError("not JSON"),
        KeyError("nc_count"),
        RuntimeError("neuron-ls gave no device JSON"),
        subprocess.CalledProcessError(2, ["neuron-ls", "--json-output"]),
        subprocess.TimeoutExpired(["neuron-ls"], 60),
    ],
)
def test_a_failing_device_listing_fails_the_start(monkeypatch, affinity, err):
    def broken():
        raise err

    monkeypatch.setattr(TP, "neuron_devices", broken)
    with pytest.raises(TP.NumaPinError, match="TRNRT_NUMA_PIN_OPTIONAL=1"):
        TP.numa_pin(0, 0)
    assert affinity["sets"] == 0


def test_the_override_runs_unpinned_with_a_warning_and_an_off_line(monkeypatch, affinity, caplog):
    def broken():
        raise FileNotFoundError("neuron-ls")

    monkeypatch.setattr(TP, "neuron_devices", broken)
    monkeypatch.setenv(TP.PIN_OPTIONAL, "1")
    with caplog.at_level(logging.WARNING, logger="trnrt"):
        assert TP.numa_pin(0, 0, world=32) is None
    assert affinity["sets"] == 0
    assert "rank 0 NOT PINNED" in caplog.text and "NUMA-PIN: off, 0/32 ranks pinned" in caplog.text


@pytest.mark.parametrize("what", ["lnc", "another rank's record"])
def test_a_bad_lnc_or_another_ranks_bad_record_takes_the_failed_listing_path(affinity, monkeypatch, caplog, what):
    """NEURON_LOGICAL_NC_CONFIG not a number, or (rank 0, which counts the engine's pins) a record of a device only other
    ranks use (device 5: ranks 20-23): NumaPinError, or unpinned under the override, never a bare ValueError."""
    devs = LS
    if what == "lnc":
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "two")
    else:
        devs = [*LS[:5], {**LS[5], "nc_count": None}, *LS[6:]]
    with pytest.raises(TP.NumaPinError, match="NEURON_LOGICAL_NC_CONFIG='two'" if what == "lnc" else "device record"):
        TP.numa_pin(0, 0, devs, world=32)
    monkeypatch.setenv(TP.PIN_OPTIONAL, "1")
    with caplog.at_level(logging.WARNING, logger="trnrt"):
        assert TP.numa_pin(0, 0, devs, world=32) is None
    assert affinity["sets"] == 0 and "NUMA-PIN: off, 0/32 ranks pinned" in caplog.text


@pytest.mark.parametrize(
    ("launched", "line"),
    [
        (ALL, "NUMA-PIN: on, 32/32 ranks pinned"),
        (
            TP.parse_cpulist(NODE0),
            "NUMA-PIN: partial, 16/32 ranks pinned; 16 as launched (no CPU of their node in the launch mask 0-47,96-143)",
        ),
    ],
)
def test_rank_0_logs_the_engines_numa_pin_line(affinity, caplog, launched, line):
    """Core base 0 (devices 0-7: 0-3 on node 1, 4-7 on node 0). A node-0-only launch set leaves ranks 0-15 as launched,
    which a start once did with no error at all."""
    affinity["mask"] = set(launched)
    with caplog.at_level(logging.INFO, logger="trnrt"):
        TP.numa_pin(0, 0, LS, world=32)
    assert line in caplog.text
    if launched == ALL:
        assert affinity["mask"] == TP.parse_cpulist(NODE1)
    else:
        assert "rank 0 not pinned" in caplog.text and affinity["sets"] == 0


def test_ranks_other_than_0_log_no_engine_line(affinity, caplog):
    with caplog.at_level(logging.INFO, logger="trnrt"):
        assert TP.numa_pin(5, 0, LS) == TP.parse_cpulist(NODE1)  # core 5: device 1, node 1
    assert "NUMA-PIN:" not in caplog.text


def test_ls_records_reads_both_forms_and_guesses_nothing():
    assert TP.ls_records(json.dumps(LS)) == LS
    assert TP.ls_records(json.dumps({"instance": {"type": "trn2.48xlarge"}, "neuron_devices": LS})) == LS
    for bad in (
        [],
        [{"x": 1}],
        [*LS, 3],
        [1, 2],
        {"a": LS, "b": LS},
        {"instance": {}},
        {"devices": [{"x": 1}]},
        "text",
        3,
    ):
        with pytest.raises(ValueError, match="neuron-ls JSON"):
            TP.ls_records(json.dumps(bad))


# Records neuron-ls could print that numa_cpus cannot read: device 0's field null, missing, not a number, or zero cores.
BAD_RECORDS = {
    "nc_count null": {"nc_count": None},
    "nc_count zero": {"nc_count": 0},
    "nc_count text": {"nc_count": "four"},
    "cpu_affinity null": {"cpu_affinity": None},
    "neuron_device null": {"neuron_device": None},
    "lnc a list": {"logical_neuroncore_config": [2]},
}


def _bad(case):
    return [{**LS[0], **BAD_RECORDS[case]}, *LS[1:]]


@pytest.mark.parametrize("case", sorted(BAD_RECORDS))
def test_a_device_record_it_cannot_read_fails_the_start_as_a_failed_listing(affinity, monkeypatch, caplog, case):
    with pytest.raises(TP.NumaPinError, match="neuron-ls device record"):
        TP.numa_pin(0, 0, _bad(case))
    monkeypatch.setenv(TP.PIN_OPTIONAL, "1")
    with caplog.at_level(logging.WARNING, logger="trnrt"):
        assert TP.numa_pin(0, 0, _bad(case)) is None  # the override runs it unpinned
    assert affinity["sets"] == 0 and "rank 0 NOT PINNED" in caplog.text


def test_neuron_devices_falls_back_to_the_newer_flag(monkeypatch):
    calls = []

    def run(argv, **kw):
        calls.append(argv[1:])
        if argv[1:] == ["--json-output"]:
            raise subprocess.CalledProcessError(2, argv, stderr="unknown flag --json-output")
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps({"neuron_devices": LS}), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert TP.neuron_devices() == LS and calls == [["--json-output"], ["--output-format", "json"]]

    def refuse(argv, **kw):
        raise subprocess.CalledProcessError(2, argv, stderr="no")

    monkeypatch.setattr(subprocess, "run", refuse)
    with pytest.raises(RuntimeError, match="neuron-ls gave no device JSON"):
        TP.neuron_devices()


@pytest.mark.parametrize("old", ["the --json-output flag is deprecated", json.dumps({"devices": "moved"}), "[]"])
def test_the_newer_flag_is_tried_when_the_old_one_exits_0_with_output_it_cannot_read(monkeypatch, old):
    calls = []

    def run(argv, **kw):
        calls.append(argv[1:])
        out = old if argv[1:] == ["--json-output"] else json.dumps({"neuron_devices": LS})
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert TP.neuron_devices() == LS and calls == [["--json-output"], ["--output-format", "json"]]
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout=old))
    with pytest.raises(RuntimeError, match=r"neuron-ls gave no device JSON: --json-output: .*; --output-format json: "):
        TP.neuron_devices()  # neither readable: a RuntimeError naming both, which numa_pin takes as a failed listing


def test_fmt_cpus_inverts_parse_cpulist():
    for s in ("0-47,96-143", "7", "0-3,5,8-9", ""):
        assert TP.fmt_cpus(TP.parse_cpulist(s)) == s
