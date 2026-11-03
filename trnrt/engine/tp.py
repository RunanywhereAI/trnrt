"""Tensor-parallel process group.

One process per rank; rank 0 runs the scheduler and the API, ranks > 0 replay every graph
call rank 0 makes (ModelRunner.serve), so all ranks execute identical graphs on identical
inputs and meet at the same in-graph collectives. Host-side coordination is gloo.

On Neuron each rank sees exactly one NeuronCore (NEURON_RT_VISIBLE_CORES) and all ranks
share one CCOM bootstrap endpoint (NEURON_RT_ROOT_COMM_ID), the arrangement vllm-neuron's
worker uses (vllm_neuron/vllm/worker/neuron_worker.py, _init_neuron_distributed_environment
_and_runtime and rendezvous_ccom_bootstrap). The CCOM port is the gloo port + 1 so the two
never collide (libtorch_neuronx_lite defaults the CCOM root to MASTER_PORT otherwise).
"""

from __future__ import annotations

import datetime
import errno
import os
import socket

import torch.distributed as dist

# Ranks > 0 wait in a broadcast while rank 0 compiles a new bucket, which can take tens of
# minutes for a large model; the default 30-minute gloo timeout would kill a healthy job.
TIMEOUT = datetime.timedelta(hours=4)


def port_free(p: int) -> bool:
    """Whether p can be bound on EVERY local address, as rank 0's TCPStore binds it (it listens on
    all interfaces). Free on 127.0.0.1 is not enough: gloo's own listeners sit on the host's
    address, and a long-lived process collects them (a test session held two dozen gloo listeners
    on the host's address, free_port probing 127.0.0.1 returned one of them, and rank 0's TCPStore
    bind failed)."""
    families = [(socket.AF_INET, "0.0.0.0")] + ([(socket.AF_INET6, "::")] if socket.has_ipv6 else [])
    for fam, host in families:
        try:
            with socket.socket(fam) as s:
                s.bind((host, p))
        except OSError as e:
            if fam == socket.AF_INET6 and e.errno in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):  # no IPv6 here
                continue
            return False
    return True


# Ports of a multi-rank job on a Neuron device: p (rank 0's TCPStore, held for the job's life) and p + 1 (the runtime's
# collectives root, NEURON_RT_ROOT_COMM_ID, bound only while a graph's collectives load). Picked by bind(0), both fell
# in the kernel's ephemeral range (32768-60999 on a trn2), and p + 1 is free between loads: an engine's graph
# first-load found its root port taken ("Failed to bind(127.0.0.1<port>), error: Address already in use", and a rank
# reached a foreign listener there) while another engine on the host was tearing down. So
# a device job takes its pair from a band of its own: PORT_BAND ports per first logical core, below the ephemeral
# range, where neither a bind(0) nor another core base's job can land.
PORT_BAND0 = 21000
PORT_BAND = 20


def port_band(core_base: int) -> range:
    """The ports a job whose first logical core is core_base may use: PORT_BAND of them, disjoint across core bases."""
    lo = PORT_BAND0 + core_base * PORT_BAND
    return range(lo, lo + PORT_BAND)


def free_port(core_base: int | None = None) -> int:
    """A port p such that p and p + 1 were both free on every local address a moment ago. With core_base (a device job:
    rank r runs on logical core core_base + r), the pair comes from port_band(core_base), and none free there is an
    error rather than a fall back into the ephemeral range. Without (CPU runs, tests), any pair bind(0) offers."""
    if core_base is not None:
        b = port_band(core_base)
        for p in range(b.start, b.stop - 1, 2):
            if port_free(p) and port_free(p + 1):
                return p
        raise RuntimeError(f"no free port pair in core base {core_base}'s band {b.start}-{b.stop - 1}: another job on "
                           f"the same cores, or its sockets still in TIME_WAIT")
    for _ in range(50):
        with socket.socket() as a:
            a.bind(("0.0.0.0", 0))
            p = a.getsockname()[1]
        if port_free(p) and port_free(p + 1):
            return p
    raise RuntimeError("no free port pair")


# Process state init_rank / neuron_env change in the calling process, which rank 0 (the engine's
# own process: a server, a test session) restores on LLMEngine.close.
_ENV_KEYS = ("MASTER_ADDR", "MASTER_PORT", "NEURON_RT_VISIBLE_CORES", "NEURON_RT_ROOT_COMM_ID")


def process_state() -> dict:
    import torch

    return {"env": {k: os.environ.get(k) for k in _ENV_KEYS}, "threads": torch.get_num_threads(),
            "affinity": os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None}


def restore_process_state(saved: dict) -> None:
    import torch

    for k, v in saved["env"].items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    torch.set_num_threads(saved["threads"])
    if saved.get("affinity") is not None and os.sched_getaffinity(0) != saved["affinity"]:  # numa_pin's, on rank 0
        os.sched_setaffinity(0, saved["affinity"])


# NUMA pinning (host side only: no graph reads it): each device rank runs on the CPUs of its NeuronCore's NUMA node,
# intersected with the CPUs it was launched on, so a launcher's CPU set (taskset, a cgroup) still bounds it; an empty
# intersection, or no device for the core, leaves the rank as launched. The mapping is the driver's, from
# `neuron-ls --json-output` (each device's neuron_device, nc_count at its logical_neuroncore_config, cpu_affinity):
# the PCI order is not the device order (on a trn2.48xlarge device 0 is 0000:cc:00.0, node 1; devices 0-3 and 12-15
# are node 1 (CPUs 48-95, 144-191) and 4-11 node 0 (0-47, 96-143)).
def parse_cpulist(s: str) -> set[int]:
    """A kernel CPU list ("0-47,96-143") as a set."""
    out: set[int] = set()
    for part in s.split(","):
        if part.strip():
            a, _, b = part.strip().partition("-")
            out.update(range(int(a), int(b or a) + 1))
    return out


def neuron_devices() -> list[dict]:
    """`neuron-ls --json-output`: one dict per Neuron device."""
    import json
    import shutil
    import subprocess

    exe = shutil.which("neuron-ls") or "/opt/aws/neuron/bin/neuron-ls"
    return json.loads(subprocess.run([exe, "--json-output"], capture_output=True, text=True, timeout=60,
                                     check=True).stdout)


def numa_cpus(core: int, devices: list[dict], lnc: int | None = None) -> set[int] | None:
    """The CPUs local to the device holding logical NeuronCore `core` at logical config `lnc` (default: the one
    neuron-ls reports), or None if no device holds it."""
    for d in devices:
        ls_lnc = int(d.get("logical_neuroncore_config") or 1)
        per_device = int(d["nc_count"]) * ls_lnc // (lnc or ls_lnc)
        if int(d["neuron_device"]) == core // per_device:
            return parse_cpulist(d["cpu_affinity"])
    return None


def pin_mask(core: int, devices: list[dict], launched, lnc: int | None = None) -> set[int] | None:
    """numa_cpus(core) & launched, or None when that is empty or no device holds the core."""
    return (numa_cpus(core, devices, lnc) or set()) & set(launched) or None


# What numa_pin saw (core base, devices, launch mask, LNC), for rank_threads: which of the engine's ranks share this
# rank's CPUs. None when the device listing failed (every rank then runs as launched).
_PIN: dict | None = None


def numa_pin(rank: int, core_base: int, devices: list[dict] | None = None) -> set[int] | None:
    """Pin the calling thread, and so every thread it starts after, to pin_mask of logical core core_base + rank
    (the runtime's NEURON_LOGICAL_NC_CONFIG when set, else neuron-ls's). Returns the mask, or None."""
    global _PIN
    import sys

    _PIN = None
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG")
    launched = os.sched_getaffinity(0)
    try:
        devs = neuron_devices() if devices is None else devices
        mask = pin_mask(core_base + rank, devs, launched, int(lnc) if lnc else None)
    except (OSError, ValueError, KeyError, RuntimeError) as e:  # neuron-ls missing or failing: run as launched
        print(f"NUMA pin: rank {rank} not pinned ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
        return None
    _PIN = {"core_base": core_base, "devices": devs, "launched": set(launched), "lnc": int(lnc) if lnc else None}
    if mask is None:
        print(f"NUMA pin: rank {rank} not pinned (no CPU of core {core_base + rank}'s node in the launch "
              f"mask)", file=sys.stderr, flush=True)
        return None
    os.sched_setaffinity(0, mask)
    return mask


def neuron_env(rank: int, port: int, core_base: int) -> None:
    """Must run before libtorch_neuronx_lite is imported in this process."""
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_base + rank)
    os.environ["NEURON_RT_ROOT_COMM_ID"] = f"127.0.0.1:{port + 1}"
    numa_pin(rank, core_base)


def rank_threads(world: int) -> int:
    """Intra-op threads for this rank: the CPUs it may run on (its affinity, after numa_pin) split over the engine's
    ranks that run on those same CPUs (numa_pin's masks for every rank; all `world` ranks when nothing was pinned).
    os.cpu_count() ignores taskset, cgroups and the pin: a tp=32 engine launched on 80 of a host's 192 CPUs got 6 threads
    a rank, 192 threads on 80 CPUs, and its start ran ~13x slower than with 2 (measured). Launched on every CPU and
    unpinned, this is cpu_count() // world."""
    mine = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    sharers = world
    if _PIN is not None:
        p = _PIN
        sharers = sum(1 for r in range(world)
                      if (pin_mask(p["core_base"] + r, p["devices"], p["launched"], p["lnc"]) or p["launched"]) == mine)
        sharers = sharers or world  # the affinity moved since numa_pin: no rank's mask is it, so assume all
    return max(1, len(mine) // sharers)


def init_rank(rank: int, world: int, port: int) -> None:
    # Each rank's torch would otherwise start one intra-op thread per CPU: measured load
    # average 364 on a 128-vCPU host with 8 ranks loading weights.
    import torch
    from torch.distributed import distributed_c10d as c10d

    torch.set_num_threads(rank_threads(world))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # torch 2.11 init_process_group names the default group from _world.group_count BEFORE its
    # rendezvous, so a rendezvous that raises leaves the count at 1 with no group to destroy. The
    # next init in this process then names its group "1" while fresh ranks name theirs "0", and
    # gloo's full mesh waits for keys under the other prefix until TIMEOUT: measured, the ranks of a
    # later test engine all in _new_process_group_helper for 10+ minutes after an earlier engine's
    # rank 0 failed to bind its store. Put the count back so a failed init leaves the process as it
    # found it.
    count = c10d._world.group_count
    # c10d's own init_process_group, not the torch.distributed attribute: vllm_neuron (imported with
    # libtorch_neuronx_lite in some Neuron images) replaces that attribute for tcp://127.0.0.1 init methods
    # (vllm_neuron/vllm/patches/port_hold_patch.py), and its rank 0 may rebind to another port that only a patched
    # rank finds. A process that imported it before starting an engine had a patched rank 0 and unpatched workers,
    # and the gloo mesh never formed (measured: a tp=2 CPU engine hung). The port here is chosen by this module.
    try:
        c10d.init_process_group("gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}",
                                timeout=TIMEOUT)
    except BaseException:
        if not dist.is_initialized():
            c10d._world.group_count = count
        raise


def attention_group(world: int, attn_tp: int):
    """This rank's attention group (models/decoder.py DecoderForCausalLM: attention TP): ranks
    g * attn_tp .. (g + 1) * attn_tp - 1 for g = rank // attn_tp, consecutive so that a group spans
    as few chips as possible (an all-reduce within one chip is several times cheaper than one
    across chips). None when no subgroup is needed (attn_tp 1 runs no attention
    collective, attn_tp == world uses the world group). dist.new_group is collective over the
    world, so EVERY rank must call this, with the same arguments, at the same point."""
    if attn_tp in (1, world):
        return None
    if world % attn_tp:
        raise ValueError(f"attention tp={attn_tp} does not divide tp={world}")
    rank, mine = dist.get_rank(), None
    for g in range(world // attn_tp):
        pg = dist.new_group(list(range(g * attn_tp, (g + 1) * attn_tp)))
        if g == rank // attn_tp:
            mine = pg
    return mine


def send(msg) -> None:
    dist.broadcast_object_list([msg], src=0)


def recv():
    box = [None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def worker_main(rank: int, cfg, path: str, num_pages: int, port: int) -> None:
    if cfg.device == "neuron":
        neuron_env(rank, port, cfg.tp_core_base)
    init_rank(rank, cfg.tp, port)
    from .engine import build_shard

    _, runner = build_shard(cfg, path, num_pages, rank)
    runner.serve(recv)
    dist.destroy_process_group()
