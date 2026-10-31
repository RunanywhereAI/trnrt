"""engine/tp.py rank_threads: each rank's torch intra-op threads are the CPUs it may run on, split over the engine's ranks
that run on the same CPUs (numa_pin's masks replayed for every rank), not os.cpu_count() // world, which ignores taskset,
cgroups and the pin. A synthetic trn2.48xlarge listing (16 devices, 4 logical cores each at LNC 2; devices 0-3 and 12-15 on
node 1 = CPUs 48-95,144-191, devices 4-11 on node 0 = CPUs 0-47,96-143), with os.sched_getaffinity / sched_setaffinity
faked (macOS has neither)."""

from __future__ import annotations

import inspect
import os

import pytest

from trnrt.engine import tp as TP

NODE0 = "0-47,96-143"
NODE1 = "48-95,144-191"
LS = [{"neuron_device": d, "nc_count": 4, "logical_neuroncore_config": 2,
       "cpu_affinity": NODE1 if d < 4 or d >= 12 else NODE0} for d in range(16)]
ALL = set(range(192))
HALF_B = set(range(48, 88)) | set(range(144, 184))  # a half-box engine's launch set: node 1's CPUs within the engine set


@pytest.fixture
def affinity(monkeypatch):
    state = {"mask": set(ALL)}

    def setaff(pid, mask):
        assert pid == 0
        state["mask"] = set(mask)

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(state["mask"]), raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", setaff, raising=False)
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    return state


def test_unpinned_on_every_cpu_keeps_cpu_count_over_world(monkeypatch, affinity):
    """Launched on all 192 CPUs, nothing pinned: 192 // 32 = 6, what init_rank set before."""
    monkeypatch.setattr(TP, "_PIN", None)
    monkeypatch.setattr(os, "cpu_count", lambda: 192)
    assert TP.rank_threads(32) == 6 == os.cpu_count() // 32


def test_unpinned_on_a_narrowed_set_splits_that_set(monkeypatch, affinity):
    """taskset to 80 CPUs, nothing pinned: 80 // 32 = 2 a rank (cpu_count() // world gave 6: 192 threads on 80 CPUs)."""
    monkeypatch.setattr(TP, "_PIN", None)
    affinity["mask"] = set(HALF_B)
    assert TP.rank_threads(32) == 2


@pytest.mark.parametrize("rank", [0, 15, 16, 31])
def test_a_half_box_engine_shares_one_set(affinity, rank):
    """Core base 32 launched on HALF_B: ranks 0-15 (node 0's devices) can't pin and stay on HALF_B, ranks 16-31 pin to node 1
    & HALF_B = HALF_B. All 32 share 80 CPUs: 2 threads each."""
    affinity["mask"] = set(HALF_B)
    TP.numa_pin(rank, 32, LS)
    assert affinity["mask"] == HALF_B and TP.rank_threads(32) == 2


@pytest.mark.parametrize("rank,want_mask,want", [
    (0, set(range(48, 72)), 24 // 16),   # node 1 & 0-71 = 48-71, ranks 0-15 share it: 1
    (16, set(range(0, 48)), 48 // 16),   # node 0 & 0-71 = 0-47, ranks 16-31 share it: 3
])
def test_pinned_across_two_nodes_counts_each_nodes_ranks(affinity, rank, want_mask, want):
    """Launched on 0-71 at core base 0: each node's 16 ranks split their own node's CPUs. Against world (32) the count would
    read 0 and 1, so the per-set count is what is under test."""
    affinity["mask"] = set(range(0, 72))
    TP.numa_pin(rank, 0, LS)
    assert affinity["mask"] == want_mask and TP.rank_threads(32) == want
    assert len(want_mask) // 32 != want  # a world-based split gives another answer (control)


def test_an_affinity_matching_no_rank_falls_back_to_world(affinity):
    affinity["mask"] = set(HALF_B)
    TP.numa_pin(20, 32, LS)
    affinity["mask"] = set(range(0, 64))  # moved after the pin
    assert TP.rank_threads(32) == 64 // 32


def test_a_failed_listing_counts_all_ranks(monkeypatch, affinity):
    """neuron-ls failing leaves every rank as launched, so all `world` ranks share the launch set."""
    def broken():
        raise FileNotFoundError("neuron-ls")

    monkeypatch.setattr(TP, "neuron_devices", broken)
    affinity["mask"] = set(HALF_B)
    assert TP.numa_pin(3, 32) is None and TP.rank_threads(32) == 2


def test_at_least_one_thread(monkeypatch, affinity):
    monkeypatch.setattr(TP, "_PIN", None)
    affinity["mask"] = set(range(8))
    assert TP.rank_threads(32) == 1


def test_init_rank_sets_rank_threads():
    src = inspect.getsource(TP.init_rank)
    assert "torch.set_num_threads(rank_threads(world))" in src and "cpu_count" not in src
