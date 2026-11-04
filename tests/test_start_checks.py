"""The checks of its host an engine makes before it starts (trnrt/engine/start_checks.py):

- /proc/buddyinfo's free order-10 blocks, over every zone and in each NUMA node's Normal zone: a start is refused
  below either floor, with the page-cache drop that fixes it and the switch that skips the check; a host whose total
  passes while one node is short is short (a rank allocates on its own node);
- the engine's own Neuron devices (neuron-ls's records at LNC 2: four logical cores a device on a trn2.48xlarge) must
  have no runtime user; processes on the other engine's devices, and processes without a runtime (neuron-ls itself, a
  monitor) do not count; a failing neuron-ls refuses, a missing one only warns;
- nothing is checked on CPU or off Linux, and each check has its own skip switch.
"""

from __future__ import annotations

import logging
import subprocess
import sys

import pytest

from trnrt.engine import start_checks as S

# A host's /proc/buddyinfo at a refused start: node 0's Normal zone had no free order-10 block while the total over
# every zone passed the all-zone floor.
SHORT_NODE = (
    "Node 0, zone      DMA      1      0      1      0      1      1      1      0      1      1      3\n"
    "Node 0, zone    DMA32   1015    979    736    535    373    228    105     44     19     10    700\n"
    "Node 0, zone   Normal  15342   9211   4561   1866    711    222     51      8      1      0      0\n"
    "Node 1, zone   Normal   2011   1302    988    701    522    410    322    250    200    150   1200\n"
)
PLENTY = (
    "Node 0, zone   Normal  15342   9211   4561   1866    711    222     51      8      1      0   9000\n"
    "Node 1, zone   Normal   2011   1302    988    701    522    410    322    250    200    150   7000\n"
)


def test_one_short_node_is_short():
    total, nodes, zones = S.parse_buddyinfo(SHORT_NODE)
    assert (total, nodes, zones) == (1903, {0: 0, 1: 1200}, 4)
    assert total >= S.ORDER10_FLOOR and S.order10_short(total, nodes)  # the total alone would have passed
    assert not S.order10_short(total, {0: 700, 1: 1200})
    assert S.order10_short(900, {0: 700, 1: 1200})  # the all-zone floor
    assert S.order10_short(S.ORDER10_FLOOR, {0: S.ORDER10_NODE_FLOOR - 1})
    assert not S.order10_short(S.ORDER10_FLOOR, {0: S.ORDER10_NODE_FLOOR})


def test_host_memory_refusal_names_the_counts_and_the_fix(tmp_path):
    p = tmp_path / "buddyinfo"
    p.write_text(SHORT_NODE)
    why = S.host_memory_refusal(str(p))
    assert why is not None and "1903 free order-10" in why and "node 0 0, node 1 1200" in why
    assert "sync; echo 1 | sudo tee /proc/sys/vm/drop_caches" in why and "TRNRT_SKIP_HOST_MEM_CHECK=1" in why
    p.write_text(PLENTY)
    assert S.host_memory_refusal(str(p)) is None
    p.write_text("")  # no zone listed: nothing to check
    assert S.host_memory_refusal(str(p)) is None
    assert S.host_memory_refusal(str(tmp_path / "absent")) is None


def _trn2(procs: dict[int, list[dict]] | None = None) -> list[dict]:
    """neuron-ls -j -a on a trn2.48xlarge at LNC 2: 16 devices of 4 logical cores; procs: device -> its processes."""
    out = []
    for d in range(16):
        rec = {
            "neuron_device": d,
            "nc_count": 4,
            "logical_neuroncore_config": 2,
            "neuroncore_ids": list(range(4 * d, 4 * d + 4)),
        }
        if procs and d in procs:
            rec["neuron_processes"] = procs[d]
        out.append(rec)
    return out


def _rt(pid: int, cmd: str = "python") -> dict:
    return {"pid": pid, "command": cmd, "neuron_runtime_version": "2.32.0"}


def test_an_engine_owns_the_devices_of_its_cores():
    devs = _trn2()
    assert S.engine_devices(0, 32, devs, 2) == list(range(8))
    assert S.engine_devices(32, 32, devs, 2) == list(range(8, 16))
    assert S.engine_devices(0, 64, devs, 2) == list(range(16))
    assert S.engine_devices(4, 1, devs, 2) == [1]
    assert S.engine_devices(4, 1, devs, None) == [1]  # neuron-ls's own LNC (2) when none is given
    assert S.engine_devices(4, 1, devs, 1) == [0]  # at LNC 1 a device holds 8 logical cores


def test_only_runtime_users_of_the_engines_own_devices_refuse():
    other = _trn2({2: [_rt(111, "engine 0 rank 2")], 9: [{"pid": 7, "command": "neuron-ls"}]})
    assert S.device_refusal(32, 32, 2, other) is None  # engine 1: engine 0's ranks and a tool are not its users
    busy = _trn2({9: [_rt(222, "python -m trnrt engine"), {"pid": 7, "command": "neuron-ls"}], 12: [_rt(333)]})
    why = S.device_refusal(32, 32, 2, busy)
    assert why is not None
    assert "pid 222 on device 9 (python -m trnrt engine)" in why and "pid 333 on device 12" in why
    assert "pid 7" not in why  # neuron-ls itself: no runtime
    assert "cores 32-63" in why and "TRNRT_SKIP_DEVICE_CHECK=1" in why
    assert S.device_refusal(0, 32, 2, busy) is None  # engine 0's devices are free


def test_a_failing_neuron_ls_refuses_and_a_missing_one_warns(monkeypatch, caplog):
    def failing():
        raise subprocess.CalledProcessError(1, ["neuron-ls", "-j", "-a"])

    monkeypatch.setattr(S, "neuron_ls", failing)
    why = S.device_refusal(0, 32, 2)
    assert why is not None and "never taken as free" in why and "TRNRT_SKIP_DEVICE_CHECK=1" in why
    monkeypatch.setattr(S, "neuron_ls", lambda: None)
    with caplog.at_level(logging.INFO, logger="trnrt"):
        assert S.device_refusal(0, 32, 2) is None
    assert "trnrt: neuron-ls not found" in caplog.text


def test_a_listing_it_cannot_read_refuses():
    why = S.device_refusal(0, 32, 2, [{"neuron_device": 0}])  # no nc_count
    assert why is not None and "could not read neuron-ls's device listing" in why and "TRNRT_SKIP_DEVICE_CHECK" in why


def test_neuron_ls_refuses_an_empty_listing(monkeypatch):
    monkeypatch.setattr(S.shutil, "which", lambda name: "/bin/neuron-ls")
    monkeypatch.setattr(S.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="[]"))
    with pytest.raises(ValueError, match="no Neuron device listed"):
        S.neuron_ls()


@pytest.fixture
def linux_host(monkeypatch, tmp_path):
    """A Linux host with every device busy and the memory short, so each check refuses unless skipped."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(S, "neuron_ls", lambda: _trn2({d: [_rt(1000 + d)] for d in range(16)}))
    p = tmp_path / "buddyinfo"
    p.write_text(SHORT_NODE)
    monkeypatch.setattr(S, "BUDDYINFO", str(p))
    for k in (S.SKIP_DEVICE, S.SKIP_HOST_MEM):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_refusal_checks_devices_then_memory_each_skippable(linux_host):
    assert "are in use" in S.refusal("neuron", 0, 32)
    linux_host.setenv(S.SKIP_DEVICE, "1")
    assert "contiguous free host memory" in S.refusal("neuron", 0, 32)
    linux_host.setenv(S.SKIP_HOST_MEM, "1")
    assert S.refusal("neuron", 0, 32) is None


def test_nothing_is_checked_on_cpu_or_off_linux(linux_host):
    assert S.refusal("cpu", 0, 32) is None
    linux_host.setattr(sys, "platform", "darwin")
    assert S.refusal("neuron", 0, 32) is None
