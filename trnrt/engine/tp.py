"""Tensor-parallel process group.

One process per rank; rank 0 runs the scheduler and the API, ranks > 0 replay every graph
call rank 0 makes (ModelRunner.serve), so all ranks execute identical graphs on identical
inputs and meet at the same in-graph collectives. Host-side coordination is gloo; rank 0's
calls reach the other ranks through a shared-memory channel (ShmChannel below; TRNRT_TP_SHM=0
sends them over gloo instead).

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
        raise RuntimeError(
            f"no free port pair in core base {core_base}'s band {b.start}-{b.stop - 1}: another job on "
            f"the same cores, or its sockets still in TIME_WAIT"
        )
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

    return {
        "env": {k: os.environ.get(k) for k in _ENV_KEYS},
        "threads": torch.get_num_threads(),
        "affinity": os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None,
    }


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
    return json.loads(
        subprocess.run([exe, "--json-output"], capture_output=True, text=True, timeout=60, check=True).stdout
    )


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
        print(
            f"NUMA pin: rank {rank} not pinned (no CPU of core {core_base + rank}'s node in the launch mask)",
            file=sys.stderr,
            flush=True,
        )
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
        sharers = sum(
            1
            for r in range(world)
            if (pin_mask(p["core_base"] + r, p["devices"], p["launched"], p["lnc"]) or p["launched"]) == mine
        )
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
        c10d.init_process_group(
            "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}", timeout=TIMEOUT
        )
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


# Rank 0's calls to the other ranks (host only: no graph or cache key reads any of this) go through a shared-memory
# channel unless TRNRT_TP_SHM=0 (rank 0's environment). Every rank of a tensor-parallel engine runs on this one host.
# dist.broadcast_object_list pickles a call, then runs two gloo collectives (its size, then its bytes) over TCP: about
# 3 ms per decode step at 1-4 rows (measured; flat in rows, so latency, not bytes), on the critical path while every
# rank's device idles. Rank 0 writes the pickled call and a sequence number into the segment; each worker polls the
# number (spinning, then backing off), copies the call and acknowledges it, and rank 0 overwrites the payload only once
# every worker has acknowledged the last one. A call larger than the segment goes through broadcast_object_list in its
# turn. The first call announces the segment's name over gloo, so the workers need no setting of their own; None (the
# shutdown) closes it. The segment is TRNRT_TP_SHM_BYTES (64 MiB) in /dev/shm, one per engine.
SHM = os.environ.get("TRNRT_TP_SHM", "1") != "0"
SHM_BYTES = int(os.environ.get("TRNRT_TP_SHM_BYTES", str(64 << 20)))
_SHM_HELLO = "__trnrt_tp_shm__"
# The open channel of this process ("ch"), and whether creating one failed in this engine ("failed": its calls then go
# over gloo, as with TRNRT_TP_SHM=0).
_CH: dict = {"ch": None, "failed": False}
# A waiting party asks whether its peer is still there about once a second once it has backed off to 1 ms sleeps.
_ALIVE_EVERY = 1000


class ShmChannel:
    """One writer (rank 0), world - 1 readers, on a multiprocessing.shared_memory segment: header [seq, length, mode,
    capacity], one acknowledged seq per rank, then the payload. mode 1: the payload holds the call; mode 2: the call
    follows by broadcast_object_list (too large for the segment). The capacity is the creator's request, written into
    the header: the OS may map more (macOS rounds a segment up to its 16 KiB page), and every party must agree."""

    HDR = 64

    def __init__(self, shm, world: int, rank: int, owner: bool, nbytes: int = 0):
        import numpy as np

        self.shm, self.world, self.rank, self.owner = shm, world, rank, owner
        self.ctl = np.ndarray((4,), dtype=np.uint64, buffer=shm.buf, offset=0)
        self.acks = np.ndarray((world,), dtype=np.uint64, buffer=shm.buf, offset=self.HDR)
        self.off = self.HDR + -(-8 * world // 64) * 64
        if owner:
            self.ctl[3] = max(0, nbytes - self.off)
        self.cap = int(self.ctl[3])
        # every party starts at 0: a fresh segment's first call is seq 1. A reader that took the segment's current seq
        # when it attached would skip a call written before it attached (the writer announces the segment, then
        # writes its first call at once) and never acknowledge it: the writer would wait on it forever.
        self.seen = 0

    @classmethod
    def create(cls, world: int, nbytes: int = 0) -> ShmChannel:
        from multiprocessing import shared_memory

        nbytes = nbytes or SHM_BYTES
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        shm.buf[: cls.HDR + 8 * world] = bytes(cls.HDR + 8 * world)
        return cls(shm, world, 0, True, nbytes)

    @classmethod
    def attach(cls, name: str, world: int, rank: int) -> ShmChannel:
        from multiprocessing import resource_tracker, shared_memory

        # A reader never touches the resource tracker: the creator registers the segment and unlinks it. The engine's
        # ranks share the creator's tracker, whose registry holds a name ONCE, so a reader's register-then-unregister
        # (the usual fix for Python < 3.13 tracking every attach) dropped the creator's entry, every later unregister
        # raised KeyError in the tracker, and a killed rank 0 then leaked the segment. Python >= 3.13: track=False;
        # 3.12: register is a no-op during the open.
        try:
            shm = shared_memory.SharedMemory(name=name, track=False)
        except TypeError:
            real = resource_tracker.register
            resource_tracker.register = lambda *a, **k: None
            try:
                shm = shared_memory.SharedMemory(name=name)
            finally:
                resource_tracker.register = real
        return cls(shm, world, rank, False)

    @staticmethod
    def _wait(cond, gone=None) -> None:
        """Until cond(). Once the wait has backed off to 1 ms sleeps, gone() (a reader's: whether rank 0's process has
        exited) is asked about once a second, and raises when it answers yes: over gloo a worker's broadcast fails once
        rank 0's connection drops, and a worker left polling the segment of a dead rank 0 would hold its NeuronCore for
        good."""
        import time

        n = 0
        while not cond():
            n += 1
            if n > 2000:  # ~0.5 ms of spinning, then back off: idle workers do not hold a core each
                time.sleep(5e-5 if n < 20000 else 1e-3)
                if gone is not None and n >= 20000 and n % _ALIVE_EVERY == 0 and gone():
                    raise RuntimeError("tensor-parallel rank 0 exited: this rank stops")

    def _acked(self, seq: int) -> bool:
        """Whether every reader has acknowledged call `seq`. Raises once drop_channel closed the channel under a send
        still waiting (a rank had died): that send then fails instead of waiting forever."""
        acks = self.acks
        if acks is None:
            raise RuntimeError("the tensor-parallel channel was closed while rank 0 waited for the other ranks")
        return all(int(acks[r]) >= seq for r in range(1, self.world))

    def send(self, msg) -> None:
        import pickle

        seq = self.seen
        self._wait(lambda: self._acked(seq))
        data = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
        big = len(data) > self.cap
        if not big:
            self.shm.buf[self.off : self.off + len(data)] = data
        self.ctl[1] = len(data)
        self.ctl[2] = 2 if big else 1
        self.seen = seq + 1
        self.ctl[0] = seq + 1  # last: a reader that sees it sees the payload, length and mode
        if big:
            dist.broadcast_object_list([msg], src=0)

    def recv(self):
        import pickle

        self._wait(lambda: int(self.ctl[0]) > self.seen, _parent_gone)
        self.seen += 1
        n, mode = int(self.ctl[1]), int(self.ctl[2])
        data = bytes(self.shm.buf[self.off : self.off + n]) if mode == 1 else None
        self.acks[self.rank] = self.seen
        if mode == 2:
            box = [None]
            dist.broadcast_object_list(box, src=0)
            return box[0]
        return pickle.loads(data)

    def close(self) -> None:
        self.ctl = self.acks = None
        self.shm.close()
        if self.owner:
            self.shm.unlink()


def _parent_gone() -> bool:
    """In a spawned worker (multiprocessing.parent_process: rank 0, the engine's process): whether rank 0 has exited.
    False in a process with no parent process record (rank 0 itself, a test's reader threads)."""
    import multiprocessing

    p = multiprocessing.parent_process()
    return p is not None and not p.is_alive()


def send(msg) -> None:
    ch = _CH["ch"]
    if ch is None and SHM and not _CH["failed"] and msg is not None and dist.get_world_size() > 1:
        try:
            ch = ShmChannel.create(dist.get_world_size())
        except (OSError, ValueError) as e:  # no usable /dev/shm: nothing was announced yet, so the calls go over gloo
            _CH["failed"] = True
            import sys

            print(
                f"trnrt: no shared-memory channel for the tensor-parallel calls ({type(e).__name__}: {e}); they go "
                "over gloo, which is slower (TRNRT_TP_SHM=0 chooses gloo from the start)",
                file=sys.stderr,
                flush=True,
            )
        else:
            _CH["ch"] = ch
            dist.broadcast_object_list([(_SHM_HELLO, ch.shm.name)], src=0)
    if ch is None:
        dist.broadcast_object_list([msg], src=0)
        return
    ch.send(msg)
    if msg is None:
        ch._wait(lambda: ch._acked(ch.seen))  # every rank has read it
        ch.close()
        _CH["ch"] = None


def recv():
    ch = _CH["ch"]
    if ch is not None:
        msg = ch.recv()
        if msg is None:
            ch.close()
            _CH["ch"] = None
        return msg
    box = [None]
    dist.broadcast_object_list(box, src=0)
    msg = box[0]
    if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == _SHM_HELLO:
        _CH["ch"] = ShmChannel.attach(msg[1], dist.get_world_size(), dist.get_rank())
        return recv()
    return msg


def drop_channel() -> None:
    """Rank 0, when its engine stops: forget and unlink a channel the shutdown call did not close (an engine whose rank
    failed, or that failed to start), so that the next engine of this process opens its own. A send still waiting for
    a dead rank's acknowledgement then raises instead of waiting forever. A no-op after a normal close.

    The mapping is not closed here: that send's thread may be reading the header at this moment, and numpy views do
    not hold the buffer, so unmapping under it would crash the process. The mapping goes with the channel's last
    reference (that thread's, once its send has raised)."""
    ch, _CH["ch"], _CH["failed"] = _CH["ch"], None, False
    if ch is None:
        return
    ch.ctl = ch.acks = None
    try:
        ch.shm.unlink()
    except OSError:  # already gone
        pass


def worker_main(rank: int, cfg, path: str, num_pages: int, port: int) -> None:
    if cfg.device == "neuron":
        neuron_env(rank, port, cfg.tp_core_base)
    init_rank(rank, cfg.tp, port)
    from .engine import build_shard

    _, runner = build_shard(cfg, path, num_pages, rank)
    runner.serve(recv)
    dist.destroy_process_group()
