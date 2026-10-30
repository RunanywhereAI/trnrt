"""RadixCache.free_dominated, called at a request's release: a state checkpoint that a deeper one on the same unbranched
radix path dominates is freed.

Why: a session resumes each turn from its own state checkpoint, and every turn leaves one more on the session's path.
Older turns' checkpoints would only hold rows until eviction took them, so a full hit needed ~2.5 x sessions + seats
rows; freeing the dominated ones brings that toward sessions + seats. What it loses: a request that BRANCHES from an
older position (an agent retrying an earlier turn) finds the older checkpoint gone and resumes from an earlier one,
unless the host cache took it (the free goes through _drop_ckpt with offload, as eviction does).
"""

from __future__ import annotations

from trnrt.engine.kv_pool import PagePool
from trnrt.engine.radix_cache import RadixCache

PS = 4


def _tree():
    pool = PagePool(64)
    rad = RadixCache(pool, PS)
    freed = []
    rad.free_ckpt = freed.append
    return pool, rad, freed


def _insert(pool, rad, ids):
    return rad.insert(ids, pool.alloc(len(ids) // PS))


def test_a_deeper_checkpoint_frees_the_older_ones_on_its_unbranched_path():
    pool, rad, freed = _tree()
    path = list(range(1, 25))                   # 6 pages
    res = _insert(pool, rad, path)
    assert rad.attach_checkpoint(path, 2, 10) and rad.attach_checkpoint(path, 4, 11) and rad.attach_checkpoint(path, 5, 12)
    assert rad.free_dominated(res.node) == 2
    assert sorted(freed) == [10, 11] and rad.num_ckpts == 1 and rad.num_dominated_freed == 2
    m, _ = rad.match_checkpoint(path)
    assert m.node.ckpt == 12 and m.num_pages == 5


def test_a_branch_keeps_its_checkpoint_and_everything_above_it():
    """A second request's path leaves the first at page 2: the checkpoint AT the branch (page 2) serves both, and one
    above it (page 1) is not dominated by either branch alone."""
    pool, rad, freed = _tree()
    a = list(range(1, 25))
    b = a[:8] + [100 + i for i in range(16)]   # shares pages 1-2
    ra = _insert(pool, rad, a)
    _insert(pool, rad, b)
    assert rad.attach_checkpoint(a, 1, 9) and rad.attach_checkpoint(a, 2, 10) and rad.attach_checkpoint(a, 5, 12)
    assert rad.free_dominated(ra.node) == 0
    assert freed == [] and rad.num_ckpts == 3


def test_a_locked_checkpoint_is_kept():
    """A request holding the node (admission keeps req.ckpt_restore until its copy launches) must find its row."""
    pool, rad, freed = _tree()
    path = list(range(1, 25))
    res = _insert(pool, rad, path)
    assert rad.attach_checkpoint(path, 2, 10) and rad.attach_checkpoint(path, 5, 12)
    held, _ = rad.match_checkpoint(path, 2)
    rad.lock(held.node)
    assert rad.free_dominated(res.node) == 0 and freed == []
    rad.unlock(held.node)
    assert rad.free_dominated(res.node) == 1 and freed == [10]


def test_the_nearest_prompt_end_checkpoint_is_kept():
    """The anchor rule: of the prompt-end checkpoints on the path, the nearest one at or above the deepest checkpoint
    stays (a replay that differs inside the reply resumes there); older ones are dominated by it and go."""
    pool, rad, freed = _tree()
    path = list(range(1, 25))
    res = _insert(pool, rad, path)
    assert rad.attach_checkpoint(path, 1, 9, anchor=True)   # an older turn's prompt end
    assert rad.attach_checkpoint(path, 3, 10, anchor=True)  # this turn's prompt end
    assert rad.attach_checkpoint(path, 5, 12)               # this turn's finish checkpoint
    assert rad.free_dominated(res.node) == 1 and freed == [9]
    assert rad.num_ckpts == 2
    # Positive control: the same tree without the anchors frees both.
    pool, rad, freed = _tree()
    res = _insert(pool, rad, path)
    assert rad.attach_checkpoint(path, 1, 9) and rad.attach_checkpoint(path, 3, 10) and rad.attach_checkpoint(path, 5, 12)
    assert rad.free_dominated(res.node) == 2 and sorted(freed) == [9, 10]


def test_the_host_cache_gets_a_freed_row():
    pool, rad, freed = _tree()
    path = list(range(1, 25))
    res = _insert(pool, rad, path)
    assert rad.attach_checkpoint(path, 2, 10) and rad.attach_checkpoint(path, 5, 12)
    kept = []
    rad.offload_ckpt = lambda ids, row: kept.append((len(ids), row))
    assert rad.free_dominated(res.node) == 1
    assert kept == [(2 * PS, 10)] and freed == [10], "the freed row's state goes to the host cache first"
