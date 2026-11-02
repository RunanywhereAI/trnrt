"""In-chunk state checkpoints (ScheduledSeq.mid) on GLM-5.3-Flash's architecture (tests/test_glm5_next.py's small random
config: KDA, pooled-DSA MLA, MoE; page size 4), fp32 on CPU. A checkpoint target strictly inside a prefill chunk (the
prompt's last page boundary, a junction) is written by the chunk's own graph instead of splitting the chunk there:
- the checkpoint it leaves at the prompt's last page boundary is BIT-equal, every KDA layer's recurrent state and conv
  window, to the one a chunk ending at that boundary leaves (a copy of the state row after it), both chunks padded to
  the same rows; a checkpoint one page off is not (positive control); the square [K, V] state is not its transpose;
- a returning turn resumes from it and its output equals a cold prefill's;
- it costs no prefill call;
- with no free checkpoint row the target is dropped, not split, and the scheduler counts both outcomes.
"""

import pytest
import torch

pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

PS = 4


def sp(n=12, **kw):
    from trnrt.engine.request import SamplingParams

    return SamplingParams(max_new_tokens=n, ignore_eos=True, **kw)


@pytest.fixture(scope="module")
def glm(tmp_path_factory):
    from tests.test_glm5_next import build

    path = str(tmp_path_factory.mktemp("glm5_next_ckpt_mid"))
    build(path, seed=1, index_topk=16)
    return path


def eng_(path, **kw):
    from tests.test_glm5_next import engine

    return engine(path, **kw)


def mk(seed, lengths):
    from tests.test_glm5_next import prompts

    return prompts(seed, lengths)


def _checkpoint_states(eng, prompt, n_pages, monkeypatch):
    """Run prompt (2 tokens out) and return [(rec_state, conv_state)] per KDA layer of the checkpoint attached at
    n_pages pages."""
    from trnrt.config import LinearSpec
    from trnrt.engine.radix_cache import RadixCache

    rows = {}
    real = RadixCache.attach_checkpoint

    def spy(self, ids, n, row, anchor=False):
        ok = real(self, ids, n, row, anchor)
        if ok:
            rows[n] = row
        return ok
    monkeypatch.setattr(RadixCache, "attach_checkpoint", spy)
    try:
        eng.generate([prompt], sp(2))
    finally:
        monkeypatch.setattr(RadixCache, "attach_checkpoint", real)
    assert n_pages in rows, (n_pages, rows)
    row = rows[n_pages]
    return [(layer.rec_state[row].clone(), layer.conv_state[row].clone()) for layer in eng.runner.model.layers
            if isinstance(layer.spec, LinearSpec)]


# One chunk holds the whole prompt, so its last page boundary is strictly inside the chunk.
WIDE = dict(max_prefill_tokens=64)


def test_the_in_chunk_checkpoint_is_bit_equal_to_one_a_chunk_ends_at(glm, monkeypatch):
    (p,) = mk(91, (29,))   # last page boundary 28
    # The first chunk ends at 28 (its state row is copied there), padded to the 32 rows the in-chunk prompt takes: on an
    # x86 host a 28-row chunk rounds differently from a 32-row one (up to 25 ULP from the second KDA layer on), and with
    # equal rows the two are bit-equal there too, as everywhere on Apple's BLAS.
    ends = eng_(glm, max_prefill_tokens=28, prefill_token_buckets=(32,))
    try:
        want = _checkpoint_states(ends, p, 7, monkeypatch)
        (q,) = mk(91, (25,))  # the same tokens' first 25: its boundary is one page earlier (24)
        off_one = _checkpoint_states(ends, list(q), 6, monkeypatch)
    finally:
        ends.close()
    mid = eng_(glm, **WIDE)
    try:
        got = _checkpoint_states(mid, p, 7, monkeypatch)
        assert mid.scheduler.ckpt_mid_placed == 1
    finally:
        mid.close()
    assert len(got) == len(want) > 0
    for i, ((gs, gc), (ws, wc)) in enumerate(zip(got, want)):
        assert torch.equal(gc, wc), (i, (gc - wc).abs().max())
        assert torch.equal(gs, ws), (i, (gs - ws).abs().max())
        assert gs.shape[-1] == gs.shape[-2] and not torch.allclose(gs, gs.transpose(-1, -2))  # square, not symmetric
    # positive control: one page off differs (the comparison can see a wrong k)
    assert any(not torch.equal(gs, os_) for (gs, _), (os_, _) in zip(got, off_one))


def test_a_returning_turn_resumes_from_it_and_equals_cold(glm):
    """A reply that differs on replay resumes from the in-chunk checkpoint at turn 1's prompt's last page boundary, and
    turn 2 equals a cold prefill token for token."""
    (p1,), (tail,) = mk(81, (29,)), mk(82, (6,))
    eng = eng_(glm, **WIDE)
    try:
        (r1,) = eng.generate([p1], sp(13))
        reply = list(r1.output_ids)
        reply[2] = (reply[2] + 1) % 384
        p2 = p1 + reply + tail
        (r2,) = eng.generate([p2], sp(8))
        assert r2.num_cached_tokens == (len(p1) - 1) // PS * PS, (r2.num_cached_tokens, len(p1))
    finally:
        eng.close()
    fresh = eng_(glm, **WIDE)
    try:
        (c2,) = fresh.generate([p2], sp(8))
    finally:
        fresh.close()
    assert r2.output_ids == c2.output_ids


def test_it_costs_no_prefill_call(glm):
    """A 29-token prompt in a 64-token budget is one prefill call, with its checkpoint at 28 written inside it."""
    (p,) = mk(93, (29,))
    eng = eng_(glm, **WIDE)
    try:
        eng.generate([p], sp(2))
        assert sum(n for k, n in eng.runner.calls.items() if k[0] == "prefill") == 1
        assert eng.scheduler.ckpt_mid_placed == 1 and eng.radix.num_ckpts >= 1
    finally:
        eng.close()


def test_the_counters_say_when_a_checkpoint_was_placed_or_skipped(glm, monkeypatch):
    """With no free checkpoint row the in-chunk target is dropped (no split fallback), and the scheduler counts it."""
    (p,) = mk(93, (29,))
    eng = eng_(glm, **WIDE)
    try:
        eng.generate([p], sp(2))
        s = eng.scheduler
        assert (s.ckpt_mid_placed, s.ckpt_mid_skipped) == (1, 0)
        monkeypatch.setattr(type(s), "_alloc_ckpt", lambda self: None)
        (q,) = mk(94, (29,))
        eng.generate([q], sp(2))
        assert (s.ckpt_mid_placed, s.ckpt_mid_skipped) == (1, 1)
    finally:
        eng.close()
