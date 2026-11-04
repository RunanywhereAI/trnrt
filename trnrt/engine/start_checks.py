"""Checks of the host an engine makes before it starts (`python -m trnrt engine`, which is also how `serve` starts each
of its engines, one at a time). Each failed check refuses the start with what it found and how to fix it, instead of
a failure deep in start-up. Only on Linux with --device neuron: anywhere else there is nothing to check.

- This engine's Neuron devices (those holding its logical cores core_base .. core_base + tp - 1) must have no runtime
  user: per device, the processes `neuron-ls -j -a` lists with a Neuron runtime version. Not the processes that hold
  /dev/neuron* open: every Neuron process opens the first devices' nodes, whatever cores it runs on, so on a host
  shared by two engines that would refuse beside any process of the other one. A device another process's runtime
  holds fails this engine's runtime start. If neuron-ls is not installed this check is skipped with a warning; if it
  fails, the start is refused (a read that fails is never taken as "free"). TRNRT_SKIP_DEVICE_CHECK=1 skips it.
- Contiguous free host memory, from /proc/buddyinfo: at least ORDER10_FLOOR free order-10 (4 MiB) blocks over all
  zones, and at least ORDER10_NODE_FLOOR in each NUMA node's Normal zone. The runtime allocates each rank's host DMA
  rings from such blocks on the rank's own node, and a start failed ("Failed to allocate HOST memory", "dma rings")
  with a node's Normal zone at 0 free blocks while hundreds of GiB were available: clean page cache (a checkpoint just
  read, other large files) breaks the blocks up, and dropping it gives them back. The floors are margins above that
  failure, not measured thresholds. TRNRT_SKIP_HOST_MEM_CHECK=1 skips it.

A start that passes prints one line to stderr naming what each check found, or why it did not run, so an operator can
see the checks ran.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

ORDER10_FLOOR = 1024  # free order-10 blocks over every zone: 4 GiB contiguous
ORDER10_NODE_FLOOR = 256  # free order-10 blocks in each NUMA node's Normal zone (a rank allocates on its own node)
BUDDYINFO = "/proc/buddyinfo"
SKIP_DEVICE = "TRNRT_SKIP_DEVICE_CHECK"
SKIP_HOST_MEM = "TRNRT_SKIP_HOST_MEM_CHECK"
DROP_CACHES = "sync; echo 1 | sudo tee /proc/sys/vm/drop_caches"
NEURON_LS = "/opt/aws/neuron/bin/neuron-ls"


def parse_buddyinfo(text: str) -> tuple[int, dict[int, int], int]:
    """(free order-10 blocks over every zone, {NUMA node: its Normal zone's free order-10 blocks}, zone lines read) of
    /proc/buddyinfo's text: lines "Node 0, zone   Normal   <free blocks of order 0> ... <order 10>"."""
    total, nodes, zones = 0, {}, 0
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 15 and f[0] == "Node":
            zones += 1
            total += int(f[14])
            if f[3] == "Normal":
                nodes[int(f[1].rstrip(","))] = int(f[14])
    return total, nodes, zones


def order10_short(total: int, nodes: dict[int, int]) -> bool:
    """Below ORDER10_FLOOR over every zone, or some node's Normal zone below ORDER10_NODE_FLOOR."""
    return total < ORDER10_FLOOR or any(n < ORDER10_NODE_FLOOR for n in nodes.values())


def host_memory_refusal(path: str | None = None) -> str | None:
    """Why the host's free contiguous memory is too short for the runtime's start, or None (also when `path`, default
    BUDDYINFO, cannot be read or lists no zone: nothing to check)."""
    return host_memory_check(path)[0]


def host_memory_check(path: str | None = None) -> tuple[str | None, str]:
    """(host_memory_refusal's answer, what the check found when it passed: for the start's pass line)."""
    path = path or BUDDYINFO
    try:
        with open(path) as f:
            total, nodes, zones = parse_buddyinfo(f.read())
    except OSError:
        return None, f"host memory not checked ({path} unreadable)"
    if not zones:
        return None, f"host memory not checked (no zone in {path})"
    by_node = ", ".join(f"node {k} {v}" for k, v in sorted(nodes.items())) or "none listed"
    if not order10_short(total, nodes):
        return None, f"host memory ok ({total} free order-10 blocks; Normal zones: {by_node})"
    return (f"too little contiguous free host memory for the Neuron runtime ({path}): {total} free order-10 (4 MiB) "
            f"blocks over all zones (needs {ORDER10_FLOOR}), and in each NUMA node's Normal zone: {by_node} (needs "
            f"{ORDER10_NODE_FLOOR} each). The runtime allocates each rank's host DMA rings from these blocks on the "
            "rank's node, and its start fails without them (\"Failed to allocate HOST memory\"). Clean page cache is "
            f"the usual cause: drop it on the host (`{DROP_CACHES}`) and start again, or set {SKIP_HOST_MEM}=1 to "
            "start anyway."), ""


def neuron_ls() -> list[dict] | None:
    """`neuron-ls -j -a`: one record per Neuron device, with the processes using it; None when neuron-ls is not
    installed. Raises when it fails or lists no device."""
    exe = shutil.which("neuron-ls") or (NEURON_LS if os.path.exists(NEURON_LS) else None)
    if exe is None:
        return None
    out = subprocess.run([exe, "-j", "-a"], capture_output=True, text=True, timeout=60, check=True).stdout
    devices = json.loads(out)
    if not isinstance(devices, list) or not devices:
        raise ValueError(f"no Neuron device listed: {out[:200]!r}")
    return devices


def engine_devices(core_base: int, tp: int, devices: list[dict], lnc: int | None = None) -> list[int]:
    """The devices holding logical NeuronCores core_base .. core_base + tp - 1 at logical config `lnc` (default: each
    device's own, as neuron-ls reports it; tp.numa_cpus reads the records the same way)."""
    mine = set()
    for d in devices:
        ls_lnc = int(d.get("logical_neuroncore_config") or 1)
        per_device = int(d["nc_count"]) * ls_lnc // (lnc or ls_lnc)
        dev = int(d["neuron_device"])
        if any(c // per_device == dev for c in range(core_base, core_base + tp)):
            mine.add(dev)
    return sorted(mine)


def device_users(devices: list[dict], wanted) -> list[tuple[int, object, str]]:
    """(device, pid, command) of every process with a Neuron runtime on a device in `wanted`."""
    out = []
    for d in devices:
        dev = int(d["neuron_device"])
        if dev in wanted:
            for p in d.get("neuron_processes") or []:
                if p.get("neuron_runtime_version"):
                    out.append((dev, p.get("pid", "?"), str(p.get("command") or "?")))
    return out


def device_refusal(core_base: int, tp: int, lnc: int | None = None, devices: list[dict] | None = None) -> str | None:
    """Why this engine's devices are not free, or None (devices: neuron-ls's records, read here when None)."""
    return device_check(core_base, tp, lnc, devices)[0]


def device_check(core_base: int, tp: int, lnc: int | None = None,
                 devices: list[dict] | None = None) -> tuple[str | None, str]:
    """(device_refusal's answer, what the check found when it passed: for the start's pass line)."""
    cores = f"cores {core_base}-{core_base + tp - 1}"
    if devices is None:
        try:
            devices = neuron_ls()
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            return (f"could not list the Neuron devices' users (neuron-ls: {type(e).__name__}: {e}), and a failed read "
                    f"is never taken as free devices. Fix neuron-ls, or set {SKIP_DEVICE}=1 to start without this "
                    "check."), ""
        if devices is None:
            print(f"trnrt: neuron-ls not found: not checking that the Neuron devices of {cores} are free",
                  file=sys.stderr, flush=True)
            return None, f"devices of {cores} not checked (no neuron-ls)"
    try:
        mine = engine_devices(core_base, tp, devices, lnc)
        users = device_users(devices, set(mine))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        return (f"could not read neuron-ls's device listing ({type(e).__name__}: {e}), and a failed read is never "
                f"taken as free devices. Set {SKIP_DEVICE}=1 to start without this check."), ""
    if not users:
        return None, f"devices {', '.join(map(str, mine))} of {cores} free"
    named = "; ".join(f"pid {pid} on device {dev} ({cmd})" for dev, pid, cmd in users[:16])
    more = f"; and {len(users) - 16} more" if len(users) > 16 else ""
    return (f"the Neuron devices of this engine's {cores} (devices "
            f"{', '.join(map(str, mine))}) are in use by another process: {named}{more}. Stop them (an engine of an "
            f"earlier run that did not exit, a neuron-monitor) and start again, or set {SKIP_DEVICE}=1 to start "
            "anyway."), ""


def refusal(device: str, core_base: int, tp: int) -> str | None:
    """Why an engine of `tp` ranks on logical cores from `core_base` must not start on this host, or None. Nothing is
    checked off Linux or off the Neuron device. A pass prints one line to stderr: what each check found, or why it did
    not run."""
    if device != "neuron" or not sys.platform.startswith("linux"):
        return None
    found = f"devices not checked ({SKIP_DEVICE}=1)"
    if os.environ.get(SKIP_DEVICE) != "1":
        from .. import platform

        why, found = device_check(core_base, tp, platform.LNC)
        if why is not None:
            return why
    mem = f"host memory not checked ({SKIP_HOST_MEM}=1)"
    if os.environ.get(SKIP_HOST_MEM) != "1":
        why, mem = host_memory_check()
        if why is not None:
            return why
    print(f"trnrt: start checks passed: {found}; {mem}", file=sys.stderr, flush=True)
    return None
