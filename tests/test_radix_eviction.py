"""The prefix cache's eviction policies (engine/radix_cache.py EVICTION_POLICIES, SGLang's --radix-eviction-policy).

Each test builds prefixes whose recency alone orders them one way and whose policy key orders them the other, so the
policy is what decides; lru on the same tree is the control."""

from __future__ import annotations

import pytest

from trnrt.engine.kv_pool import PagePool
from trnrt.engine.radix_cache import RadixCache, policy_params

PS = 4


def tree(policy="lru", config=None, n=3):
    """n two-page prefixes [k]*8 for k = 1 .. n, inserted (so last used) in that order: lru evicts 1 first."""
    pool = PagePool(64)
    rc = RadixCache(pool, PS, policy, config)
    nodes = [rc.insert([k] * 8, pool.alloc(2)).node for k in range(1, n + 1)]
    return rc, nodes


def evicted_order(rc, n) -> list[int]:
    """Evict one prefix at a time: which ones went, in order (by their token)."""
    out = []
    for _ in range(n):
        left = {k for k in range(1, n + 1) if rc.match_prefix([k] * 8, max_pages=2).num_pages}
        rc.evict(1)
        gone = left - {k for k in left if rc.match_prefix([k] * 8, max_pages=2).num_pages}
        out.extend(sorted(gone))
    return out


def test_lru_control():
    rc, _ = tree()
    assert evicted_order(rc, 3) == [1, 2, 3]


def test_lfu_evicts_the_fewest_hits_first():
    rc, (a, _b, c) = tree("lfu")
    for _ in range(3):
        rc.touch(a, 0)
    rc.touch(c, 0)
    assert evicted_order(rc, 3) == [2, 3, 1]


def test_slru_evicts_every_probationary_prefix_before_a_protected_one():
    rc, (a, _, _) = tree("slru")  # protected from 2 hits: a node starts at 1 (its insert), one touch protects it
    rc.touch(a, 0)
    assert evicted_order(rc, 3) == [2, 3, 1]
    rc, (a, _b, _c) = tree("slru", {"protected_threshold": 3})
    rc.touch(a, 0)  # 2 hits: still probationary under a threshold of 3
    assert evicted_order(rc, 3) == [1, 2, 3]


def test_priority_evicts_the_least_important_first():
    rc, (a, b, _c) = tree("priority")
    rc.touch(a, 5, hit=False)
    rc.touch(b, -1, hit=False)  # c: never touched, the lowest importance of all
    assert evicted_order(rc, 3) == [3, 2, 1]


def test_tlru_evicts_the_tail_a_next_turn_can_recompute_first():
    """Budget 8 tokens (threshold 10, next prompt 2): a 2-page leaf (8 tokens) is safe, a 3-page one (12) is not. The
    older 3-page prefix outlives the newer 2-page one, where lru takes it first (the control)."""
    for policy, config, kept in (("tlru", {"threshold": 10, "next_prompt_estimate": 2}, 1), ("lru", None, 2)):
        pool = PagePool(64)
        rc = RadixCache(pool, PS, policy, config)
        rc.insert([1] * 12, pool.alloc(3))  # the older
        rc.insert([2] * 8, pool.alloc(2))
        rc.evict(1)
        left = [k for k, n in ((1, 12), (2, 8)) if rc.match_prefix([k] * n).num_pages]
        assert left == [kept], (policy, left)


def test_tlru_chain_counts_the_evicted_tail():
    pool = PagePool(64)
    rc = RadixCache(pool, PS, "tlru", {"threshold": 10, "next_prompt_estimate": 2})  # budget 8 tokens
    chain = [3] * 4 + [4] * 4 + [5] * 4
    pages = pool.alloc(3)
    for n in (1, 2, 3):  # three nodes of one page each, the deepest inserted last
        rc.insert(chain[: 4 * n], pages[:n])
    rc.insert([1] * 12, pool.alloc(3))  # the newest, 12 tokens: unsafe from the start
    rc.evict(1)  # the chain's leaf (4 tokens, safe) goes first, though it is older than the long one
    assert rc.total_pages() == 5
    rc.evict(1)  # its parent: 4 + the 4 evicted below = 8 tokens, still safe
    assert rc.total_pages() == 4
    rc.evict(1)  # the chain's first page: 4 + 8 = 12, unsafe as the long one is; by recency the chain is older
    assert rc.total_pages() == 3 and rc.match_prefix([1] * 12).num_pages == 3


def test_a_sequence_that_grows_again_resets_its_evicted_tail_and_a_sibling_does_not():
    pool = PagePool(64)
    rc = RadixCache(pool, PS, "tlru", {"threshold": 10, "next_prompt_estimate": 2})
    rc.insert([3] * 4, pool.alloc(1))
    rc.insert([3] * 4 + [4] * 4, [rc.match_prefix([3] * 4).pages[0], *pool.alloc(1)])
    rc.evict(1)
    parent = rc.match_prefix([3] * 4).node
    assert parent.tail_evicted == 4
    # Another branch under it: the evicted sequence's next turn must still recompute its 4 tokens.
    rc.insert([3] * 4 + [6] * 4, [parent.pages[0], *pool.alloc(1)])
    assert parent.tail_evicted == 4
    rc.insert([3] * 4 + [4] * 4, [parent.pages[0], *pool.alloc(1)])  # the evicted sequence itself, back
    assert parent.tail_evicted == 0


def test_a_sibling_insert_keeps_the_shared_prefix_unsafe():
    """Budget 8 tokens. A 4-token shared prefix whose 8-token branch was evicted must not look safe (4 + 8 > 8) once a
    short sibling branch is inserted and evicted under it: it outlives a newer safe prefix."""
    pool = PagePool(64)
    rc = RadixCache(pool, PS, "tlru", {"threshold": 10, "next_prompt_estimate": 2})
    shared = [3] * 4
    rc.insert(shared, pool.alloc(1))
    page = rc.match_prefix(shared).pages[0]
    rc.insert(shared + [4] * 8, [page, *pool.alloc(2)])
    rc.evict(1)  # the 8-token branch
    rc.insert(shared + [6] * 4, [page, *pool.alloc(1)])
    rc.insert([2] * 8, pool.alloc(2))  # newer, 8 tokens: safe
    rc.evict(1)  # the sibling (4 tokens below an evicted 0: safe, the oldest); a match would refresh the prefix
    assert rc.total_pages() == 3
    rc.evict(1)  # the shared prefix (4 + 8 evicted, unsafe) outlives the newer safe one
    assert rc.match_prefix(shared).num_pages == 1 and rc.match_prefix([2] * 8).num_pages == 0


def test_session_references_still_come_first_under_every_policy():
    for policy, config in (
        ("lfu", None),
        ("slru", None),
        ("priority", None),
        ("tlru", {"threshold": 9, "next_prompt_estimate": 0}),
    ):
        rc, (a, _b, c) = tree(policy, config)
        gen = rc.session_open("s")
        assert rc.session_register("s", gen, c)
        if rc.tracks_use:
            rc.touch(a, 9)
        order = evicted_order(rc, 3)
        assert order[-1] == 3, (policy, order)


def test_policy_settings_are_checked():
    assert policy_params("lru") == {} and policy_params("slru") == {"protected_threshold": 2}
    assert policy_params("slru", {"protected_threshold": 5}) == {"protected_threshold": 5}
    with pytest.raises(ValueError, match="one of"):
        policy_params("fifo")
    with pytest.raises(ValueError, match="no setting 'threshold'"):
        policy_params("slru", {"threshold": 3})
    with pytest.raises(ValueError, match="no setting"):
        policy_params("lru", {"protected_threshold": 3})
    with pytest.raises(ValueError, match="non-negative integer"):
        policy_params("slru", {"protected_threshold": -1})
    with pytest.raises(ValueError, match="non-negative integer"):
        policy_params("slru", {"protected_threshold": 2.5})
    with pytest.raises(ValueError, match="tlru eviction needs"):
        policy_params("tlru")
    with pytest.raises(ValueError, match="tlru eviction needs"):
        policy_params("tlru", {"threshold": 4, "next_prompt_estimate": 4})
    with pytest.raises(ValueError, match="one of"):
        RadixCache(PagePool(8), PS, "mru")


def test_lru_keeps_its_order_key_and_records_nothing():
    """The default's eviction key is the one it had before the policies existed (session references, then recency), and
    it never tracks hits or importance: an engine without the flags evicts exactly as before."""
    rc, (a, _b, _c) = tree()
    assert not rc.tracks_use
    assert rc._key(a) == (a.session_refs > 0, a.session_refs, a.last_access)
