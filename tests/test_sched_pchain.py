"""The restore admission's host work: engine/scheduler.py page_chain / shared_pages_req, the host cache's chain-fed lookup /
find_ckpt and offload_ckpt_hash, and RadixCache.path_hash must give exactly what the slice / rehash forms they replace give."""

from __future__ import annotations

import random

from trnrt.engine.host_cache import HostTier
from trnrt.engine.kv_pool import PagePool
from trnrt.engine.radix_cache import RadixCache
from trnrt.engine.request import Request, SamplingParams
from trnrt.engine.scheduler import page_chain, shared_pages, shared_pages_req

PS = 4


def _req(ids, rid="r"):
    return Request(rid=rid, prompt_ids=list(ids), params=SamplingParams(max_new_tokens=4))


def test_shared_pages_req_equals_shared_pages_on_random_prefixes():
    rng = random.Random(0)
    for _ in range(400):
        base = [rng.randrange(5) for _ in range(rng.randrange(0, 60))]
        a = base + [rng.randrange(5) for _ in range(rng.randrange(0, 30))]
        b = base[: rng.randrange(0, len(base) + 1)] + [rng.randrange(5) for _ in range(rng.randrange(0, 30))]
        a, b = a or [rng.randrange(5)], b or [rng.randrange(5)]  # a Request needs a prompt
        ra, rb = _req(a, "a"), _req(b, "b")
        for limit in (0, 1, 3, 7, 100):
            assert shared_pages_req(ra, rb, PS, limit) == shared_pages(a, b, PS, limit)


def test_placeholders_and_appends_stay_exact():
    """-1 placeholders (in-flight steps, replaced in place) are never cached: a page holding one ends the chain and the
    rest is compared by slices; replacing the placeholder and appending tokens extends the chain."""
    rng = random.Random(1)
    for _ in range(200):
        base = [rng.randrange(3) for _ in range(rng.randrange(8, 40))]
        a, b = _req(base + [rng.randrange(3) for _ in range(12)], "a"), _req(list(base), "b")
        b.token_ids.extend([-1] * rng.randrange(1, 6))
        for limit in (2, 5, 50):
            assert shared_pages_req(a, b, PS, limit) == shared_pages(a.token_ids, b.token_ids, PS, limit)
        # the step lands: placeholders replaced in place, then more tokens
        for i, t in enumerate(b.token_ids):
            if t == -1:
                b.token_ids[i] = rng.randrange(3)
        b.token_ids.extend(rng.randrange(3) for _ in range(9))
        for limit in (2, 5, 50):
            assert shared_pages_req(a, b, PS, limit) == shared_pages(a.token_ids, b.token_ids, PS, limit)


def test_page_chain_rebuilds_on_a_new_or_shorter_token_list():
    r = _req(list(range(20)))
    c1 = list(page_chain(r, PS))
    r.token_ids = list(range(100, 120))  # a new list object
    assert page_chain(r, PS) != c1
    r.token_ids = r.token_ids[:6]  # shorter: one page
    assert len(page_chain(r, PS)) == 1


def test_page_chain_equals_the_host_tiers_hashes():
    t = HostTier(None, PS, 8, 4)
    ids = [random.Random(2).randrange(9) for _ in range(37)]
    assert page_chain(_req(ids), PS) == t.hashes(ids, len(ids) // PS)


class _Moves:
    def __init__(self):
        self.log = []

    def __getattr__(self, name):
        return lambda *a: self.log.append((name,) + a)


def test_lookup_and_find_ckpt_take_the_chain():
    t = HostTier(_Moves(), PS, 8, 4)
    ids = list(range(24))
    t.offload(ids, 0, [10, 11, 12, 13])
    t.offload_ckpt(ids[: 3 * PS], 7)
    ch = page_chain(_req(ids), PS)
    assert t.lookup(ids, 1, 4, chain=ch) == t.lookup(ids, 1, 4)
    assert t.find_ckpt(ids, 1, 5, chain=ch) == t.find_ckpt(ids, 1, 5)
    assert t.find_ckpt(ids, 1, 5, chain=ch[:2]) == t.find_ckpt(ids, 1, 5)  # a short chain falls back


def test_path_hash_equals_the_tiers_hash_of_the_path_through_splits():
    pool = PagePool(64)
    rad = RadixCache(pool, PS)
    t = HostTier(None, PS, 8, 4)
    rng = random.Random(3)
    common = [rng.randrange(9) for _ in range(3 * PS)]
    seqs = [common + [rng.randrange(9) for _ in range(rng.randrange(1, 4) * PS)] for _ in range(4)]
    seqs.append(common[: 2 * PS])  # splits the shared node after 2 pages
    for s in seqs:
        pages = pool.alloc(len(s) // PS)
        m = rad.insert(s, pages)
        n = m.node
        h = rad.path_hash(n)
        path, x = [], n
        while x is not None and x is not rad.root:
            path = x.key + path
            x = x.parent
        assert h == t.hashes(path, len(path) // PS)[-1]
    # every cached node hash is still its path's after the splits
    stack = list(rad.root.children.values())
    while stack:
        n = stack.pop()
        stack.extend(n.children.values())
        if "_hend" in n.__dict__:
            path, x = [], n
            while x is not None and x is not rad.root:
                path = x.key + path
                x = x.parent
            assert n.__dict__["_hend"] == t.hashes(path, len(path) // PS)[-1]


def test_drop_ckpt_offloads_by_hash_when_wired_and_by_path_otherwise():
    pool = PagePool(16)
    rad = RadixCache(pool, PS)
    ids = list(range(3 * PS))
    m = rad.insert(ids, pool.alloc(3))
    got = []
    rad.offload_ckpt = lambda path, row: got.append(("path", len(path), row))
    m.node.ckpt, rad.num_ckpts = 5, max(1, rad.num_ckpts)
    rad._drop_ckpt(m.node, offload=True)
    assert got == [("path", 3 * PS, 5)]
    rad.offload_ckpt_hash = lambda h, row: got.append(("hash", h, row))
    m.node.ckpt, rad.num_ckpts = 6, max(1, rad.num_ckpts)
    rad._drop_ckpt(m.node, offload=True)
    assert got[-1] == ("hash", HostTier(None, PS, 1, 1).hashes(ids, 3)[-1], 6)
