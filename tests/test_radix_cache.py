from trnrt.engine.kv_pool import NULL_PAGE, PagePool
from trnrt.engine.radix_cache import RadixCache

PS = 4


def make(num_pages=64):
    pool = PagePool(num_pages)
    return pool, RadixCache(pool, PS)


def test_pool_never_hands_out_null_page_and_rejects_double_free():
    pool = PagePool(8)
    pages = pool.alloc(7)
    assert pages is not None and NULL_PAGE not in pages and len(set(pages)) == 7
    assert pool.alloc(1) is None
    pool.free(pages[:1])
    try:
        pool.free(pages[:1])
    except ValueError:
        pass
    else:
        raise AssertionError("double free accepted")


def test_match_is_page_aligned_and_splits_on_partial_edge():
    pool, rc = make()
    ids = list(range(12))  # 3 pages
    pages = pool.alloc(3)
    rc.insert(ids, pages)
    # Shares the first page and a half: only one FULL page matches.
    m = rc.match_prefix([0, 1, 2, 3, 4, 5, 99, 99, 99])
    assert m.pages == pages[:1]
    # The edge was split at the match, so the node ends exactly there.
    assert m.node.pages == pages[:1]
    assert rc.match_prefix(ids).pages == pages
    assert rc.total_pages() == 3


def test_insert_returns_tree_pages_for_an_existing_prefix():
    pool, rc = make()
    ids = list(range(8))
    first = pool.alloc(2)
    rc.insert(ids, first)
    dup = pool.alloc(3)
    res = rc.insert(ids + [8, 9, 10, 11], dup)
    assert res.pages[:2] == first and res.pages[2] == dup[2]


def test_max_pages_limits_the_match():
    pool, rc = make()
    ids = list(range(16))
    pages = pool.alloc(4)
    rc.insert(ids, pages)
    assert rc.match_prefix(ids, max_pages=3).pages == pages[:3]


def test_locked_nodes_survive_eviction_and_lru_goes_first():
    pool, rc = make(num_pages=16)
    a = pool.alloc(2)
    b = pool.alloc(2)
    rc.insert([1] * 8, a)
    rc.insert([2] * 8, b)
    locked = rc.match_prefix([1] * 8).node  # also makes [1]* most recently used
    rc.lock(locked)
    free_before = pool.num_free
    assert rc.evict(10) == 2  # only b is evictable
    assert pool.num_free == free_before + 2
    assert rc.match_prefix([2] * 8).pages == []
    rc.unlock(locked)
    assert rc.evict(10) == 2
    assert rc.total_pages() == 0 and pool.num_free == pool.num_usable


def test_lru_evicts_the_least_recently_used_prefix():
    pool, rc = make()
    old, new = pool.alloc(2), pool.alloc(2)
    rc.insert([1] * 8, old)
    rc.insert([2] * 8, new)
    rc.match_prefix([1] * 8)  # a hit makes the older prefix the most recent
    assert rc.evict(2) == 2
    assert rc.match_prefix([1] * 8).pages == old and rc.match_prefix([2] * 8).pages == []


def test_eviction_cascades_to_parent_once_children_are_gone():
    pool, rc = make()
    base = list(range(8))
    p = pool.alloc(3)
    rc.insert(base + [50, 51, 52, 53], p)
    q = pool.alloc(1)
    rc.insert(base + [60, 61, 62, 63], [p[0], p[1], q[0]])
    assert rc.total_pages() == 4
    assert rc.evict(100) == 4
    assert pool.num_free == pool.num_usable


def test_evicting_a_node_frees_and_offloads_its_checkpoint():
    pool, rc = make()
    freed, offloaded = [], []
    rc.free_ckpt = freed.append
    rc.offload = lambda path, first, pages, ckpt: offloaded.append((len(path), first, list(pages), ckpt))
    ids = list(range(12))
    pages = pool.alloc(3)
    rc.insert(ids, pages)
    assert rc.attach_checkpoint(ids, 3, 7) and rc.num_ckpts == 1
    assert not rc.attach_checkpoint(ids, 3, 8), "a node keeps its first checkpoint"
    m, kv = rc.match_checkpoint(ids + [99])
    assert m.node.ckpt == 7 and m.num_pages == 3 and kv == 3
    assert rc.evict(3) == 3
    assert offloaded == [(12, 0, pages, 7)] and freed == [7] and rc.num_ckpts == 0


def test_a_hit_stops_at_the_deepest_checkpoint():
    pool, rc = make()
    ids = list(range(24))  # 6 pages
    rc.insert(ids, pool.alloc(6))
    m, kv = rc.match_checkpoint(ids)
    assert m.num_pages == 0 and m.node is rc.root and kv == 6  # no checkpoint: position 0, zero state
    assert rc.attach_checkpoint(ids, 2, 10)
    m, kv = rc.match_checkpoint(ids, max_pages=5)
    assert m.num_pages == 2 and m.node.ckpt == 10 and kv == 5


def _all_refs_zero(rc):
    stack = [rc.root]
    while stack:
        n = stack.pop()
        assert n.session_refs == 0 and not n.sessions, (n.key, n.session_refs, n.sessions)
        stack.extend(n.children.values())


def test_session_referenced_prefix_is_evicted_last_and_released_on_close():
    """SGLang #29173 semantics: unreferenced first (whatever its recency), referenced KV still
    evictable when nothing else is, and close drops references without freeing."""
    pool, rc = make()
    a, b = list(range(100, 108)), list(range(200, 208))  # two 2-page prefixes
    na = rc.insert(a, pool.alloc(2)).node  # older: plain LRU would take it first
    rc.insert(b, pool.alloc(2))
    gen = rc.session_open("s")
    assert rc.session_register("s", gen, na)
    assert rc.evict(2) == 2
    assert rc.match_prefix(a).num_pages == 2 and rc.match_prefix(b).num_pages == 0
    assert rc.evict(1) == 2  # soft protection: the referenced prefix goes when nothing else is left
    assert rc.total_pages() == 0
    _all_refs_zero(rc)

    na = rc.insert(a, pool.alloc(2)).node
    rc.insert(b, pool.alloc(2))
    assert rc.session_register("s", gen, na)
    assert rc.session_close("s") and rc.total_pages() == 4  # nothing freed by the close
    _all_refs_zero(rc)
    rc.match_prefix(b)  # b is now the more recent; a is first again under LRU
    rc.evict(2)
    assert rc.match_prefix(a).num_pages == 0 and rc.match_prefix(b).num_pages == 2


def test_open_sessions_are_bounded_least_recently_used_first():
    """A server runs for weeks and clients seldom close sessions: past MAX_OPEN_SESSIONS the least recently used
    session is closed (its references dropped, nothing freed), so the bookkeeping stays bounded and a stale session
    stops protecting its prefix. A request or a registration counts as use."""
    pool, rc = make()
    rc.MAX_OPEN_SESSIONS = 3
    nodes = [rc.insert(list(range(100 * i, 100 * i + 4)), pool.alloc(1)).node for i in range(1, 6)]
    gens = {}
    for i, sid in enumerate(["s0", "s1", "s2"]):
        gens[sid] = rc.session_open(sid)
        assert rc.session_register(sid, gens[sid], nodes[i])
    rc.session_open("s0")  # s0's next request: now s1 is the least recently used
    gens["s3"] = rc.session_open("s3")
    assert list(rc._session_open) == ["s2", "s0", "s3"] and rc.num_sessions_expired == 1
    assert nodes[1].session_refs == 0 and not nodes[1].sessions and "s1" not in rc._session_nodes
    assert nodes[0].session_refs == 1 and nodes[2].session_refs == 1
    assert rc.session_register("s2", gens["s2"], nodes[3])  # a registration is use too: s0 is now the oldest
    gens["s4"] = rc.session_open("s4")
    assert list(rc._session_open) == ["s3", "s2", "s4"] and nodes[0].session_refs == 0
    assert rc.total_pages() == 5  # closing by the cap frees nothing
    assert not rc.session_register("s1", gens["s1"], nodes[1])  # an expired session's late request cannot re-attach
    # The same id sent again (an agent that paused and resumed) is a fresh session: a new generation at the most
    # recently used end, whose request registers as any other's; the oldest open session is closed instead.
    again = rc.session_open("s1")
    assert again == gens["s1"] + 1 and list(rc._session_open)[-1] == "s1" and "s3" not in rc._session_open
    assert rc.session_register("s1", again, nodes[1]) and nodes[1].session_refs == 1 and nodes[1].sessions == {"s1"}
    rc.session_close("s1")
    for k in range(100):  # any number of sessions: the open map and the registrations stay at the cap
        g = rc.session_open(f"x{k}")
        rc.session_register(f"x{k}", g, nodes[k % 5])
    assert len(rc._session_open) == 3 and len(rc._session_nodes) <= 3
    assert sum(n.session_refs for n in nodes) <= 3 * rc.MAX_SESSION_NODES


def test_session_registration_is_generation_checked_and_split_safe():
    pool, rc = make()
    ids = list(range(12))
    old = rc.session_open("s")
    rc.session_close("s")
    new = rc.session_open("s")  # reopened by a later request
    assert new == old + 1
    pages = pool.alloc(3)
    node = rc.insert(ids, pages).node
    assert not rc.session_register("s", old, node)  # in flight across the close: stale
    assert rc.session_register("s", new, node)
    # A later turn extends the prefix: one registration, at the deeper node.
    deeper = rc.insert(ids + list(range(12, 16)), pages + pool.alloc(1)).node
    assert rc.session_register("s", new, deeper)
    assert rc._session_nodes["s"] == [deeper] and node.session_refs == 1
    # Splitting a referenced edge keeps the count on both halves.
    m = rc.match_prefix(ids[:4] + [999] * 4)
    assert m.node.session_refs == 1 and m.node.sessions == set()
    # Evicting the referenced leaf moves the registration to what is left of the prefix.
    pool2_free = pool.num_free
    rc.evict(1)
    assert pool.num_free == pool2_free + 1 and rc._session_nodes["s"] == [node]
    assert node.sessions == {"s"} and node.session_refs == 1
    rc.session_close("s")
    _all_refs_zero(rc)
