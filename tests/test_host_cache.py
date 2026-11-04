"""The host cache (engine/host_cache.py): its page part (LRU, pinning, write-through) and its state-checkpoint part
(deepest checkpoint a prompt reaches, LRU over its own slots), against a log of the device copies it asks for.
Engine-level restores, bitwise against cold runs, are in tests/test_host_cache_checkpoints.py."""

import pytest

from trnrt.engine.host_cache import HostTier


class Moves:
    def __init__(self):
        self.log = []

    def kv_save(self, slot, page):
        self.log.append(("save", slot, page))

    def kv_load(self, slot, page):
        self.log.append(("load", slot, page))

    def kv_drop(self, slot):
        self.log.append(("drop", slot))

    def state_save(self, slot, row):
        self.log.append(("state_save", slot, row))

    def state_load(self, slot, row):
        self.log.append(("state_load", slot, row))

    def state_drop(self, slot):
        self.log.append(("state_drop", slot))


def test_host_tier_lru_and_pinning():
    mv = Moves()
    t = HostTier(mv, page_size=2, capacity_pages=2, capacity_ckpts=1)
    t.offload([1, 2, 3, 4], 0, [10, 11])  # two pages of one path
    assert t.lookup([1, 2, 3, 4, 5], 0, 2) == [0, 1] and t.lookup([1, 2, 9, 9], 0, 2) == [0]
    t.unpin([0, 1])
    t.lookup([1, 2], 0, 1)  # pins the first page
    t.offload([7, 7], 0, [12])  # full: drops the LRU page that is not pinned
    assert ("drop", 1) in mv.log and t.lookup([1, 2], 0, 1) == [0]


def test_write_through_keeps_a_restored_page():
    mv = Moves()
    t = HostTier(mv, page_size=2, capacity_pages=4, capacity_ckpts=1)
    t.offload([1, 2, 3, 4], 0, [10, 11])
    slots = t.lookup([1, 2, 3, 4], 0, 2)
    t.load(slots, [20, 21])
    assert t.restored == 2 and not t.pinned
    n = len(mv.log)
    t.offload([1, 2, 3, 4], 0, [20, 21])  # evicted again: the host still holds both pages
    assert len(mv.log) == n and t.saved == 2


def test_a_page_matches_only_under_its_own_prefix():
    """The chain hash covers every earlier page: the same page of tokens after another prefix is a miss."""
    t = HostTier(Moves(), page_size=2, capacity_pages=4, capacity_ckpts=1)
    t.offload([1, 2, 3, 4], 0, [10, 11])
    assert t.lookup([9, 9, 3, 4], 1, 2) == []
    assert t.lookup([1, 2, 3, 4], 1, 2) == [1]


def test_the_deepest_host_checkpoint_in_range_is_found():
    mv = Moves()
    t = HostTier(mv, page_size=2, capacity_pages=8, capacity_ckpts=4)
    path = [1, 2, 3, 4, 5, 6, 7, 8]
    t.offload_ckpt(path[:2], 30)  # the state after page 1
    t.offload_ckpt(path[:6], 31)  # after page 3
    assert t.find_ckpt(path, 1, 4) == (3, 1)
    assert t.find_ckpt(path, 1, 2) == (1, 0)
    assert t.find_ckpt(path, 2, 2) is None and t.find_ckpt(path, 4, 4) is None
    assert t.find_ckpt([1, 2, 9, 9, 5, 6], 1, 3) == (1, 0), "a checkpoint matches only its own prefix"
    t.load_ckpt(1, 40)
    assert ("state_load", 1, 40) in mv.log and t.ckpts_restored == 1


def test_checkpoints_are_lru_over_their_own_slots():
    mv = Moves()
    t = HostTier(mv, page_size=2, capacity_pages=8, capacity_ckpts=2)
    t.offload_ckpt([1, 2], 30)
    t.offload_ckpt([3, 4], 31)
    assert t.find_ckpt([1, 2], 1, 1) == (1, 0)  # touches [1, 2]: [3, 4] is now the least recent
    t.offload_ckpt([5, 6], 32)
    assert ("state_drop", 1) in mv.log
    assert t.find_ckpt([3, 4], 1, 1) is None and t.find_ckpt([5, 6], 1, 1) == (1, 1)
    n = t.ckpts_saved
    t.offload_ckpt([1, 2], 33)  # already held: no copy
    assert t.ckpts_saved == n


def test_an_evicted_node_brings_its_checkpoint_along():
    mv = Moves()
    t = HostTier(mv, page_size=2, capacity_pages=8, capacity_ckpts=2)
    t.offload([1, 2, 3, 4], 1, [11], ckpt=7)  # a leaf holding page 1 of the path, with the state after it
    assert ("save", 0, 11) in mv.log and ("state_save", 0, 7) in mv.log
    assert t.find_ckpt([1, 2, 3, 4], 1, 2) == (2, 0)


def test_clear_drops_everything():
    mv = Moves()
    t = HostTier(mv, page_size=2, capacity_pages=2, capacity_ckpts=1)
    t.offload([1, 2], 0, [10], ckpt=5)
    t.clear()
    assert ("drop", 0) in mv.log and ("state_drop", 0) in mv.log
    assert t.lookup([1, 2], 0, 1) == [] and t.find_ckpt([1, 2], 1, 1) is None
    assert len(t.free_slots) == 2 and len(t.free_ckpt_slots) == 1


def test_both_parts_need_room():
    with pytest.raises(ValueError, match="room for at least one page and one state checkpoint"):
        HostTier(Moves(), page_size=2, capacity_pages=1, capacity_ckpts=0)
    with pytest.raises(ValueError, match="room for at least one page and one state checkpoint"):
        HostTier(Moves(), page_size=2, capacity_pages=0, capacity_ckpts=1)
