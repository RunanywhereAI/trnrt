"""With the host cache, a state checkpoint on a RUNNING request's path may go to host memory when no checkpoint row is
free (radix_cache.py evict_checkpoints `keep`), on GLM-5.3-Flash's architecture (tests/test_glm5_next.py's small random
config), fp32 on CPU.

Run: python3 -m pytest tests/test_ckpt_offload_running.py -q -s

Why: without it a running turn holds the checkpoint it resumed from and every one it attached until it ends, so a
multi-session load's full prefix hit needs ~2.5 x seats checkpoint rows; with it ~seats.

- A checkpoint offloaded while its request ran comes back BITWISE: request 1 checkpoints the state at b = 40 (its
  prompt end, as tests/test_host_cache_checkpoints.py), and with ONE checkpoint row its first decode checkpoint needs
  that row while it runs, so the b checkpoint goes to the host from the running path. (A decode checkpoint attaches to
  the tree only once its page is there, so it is never itself taken while pending.) Request 2 (same 40 tokens, another
  suffix) restores it from the host and equals the cold run over the same chunks, bit for bit.
- Mutations that must fail: restoring another host checkpoint's slot; and ignoring `keep` -- two requests admitted in
  one step, one row: the first's restored row, not yet copied, is taken by the second's restore, so the first resumes
  from the second's state.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

from tests.test_host_cache_checkpoints import B, HOST, NEW, PrefillSpy, _engine, _poison, _prompts, _reference  # noqa: E402


@pytest.fixture(scope="module")
def glm(tmp_path_factory):
    from tests.test_glm5_next import build

    path = tmp_path_factory.mktemp("glm5_next_offload")
    hf = build(str(path), seed=1, index_topk=16)
    return str(path), hf, hf.config.text_config.vocab_size


def _run(path, first, second, mode: str, flush: bool = False, **kw):
    """Request 2's (output ids, logprobs, prefill starts, counts) after request 1 on an engine with one checkpoint row;
    mode "hit", or "cold" (request 2 alone on a fresh engine). flush: every device checkpoint to the host before
    request 2 (request 1's decode checkpoint joins the b one there)."""
    from trnrt.engine.request import SamplingParams

    hit = mode != "cold"
    if hit:
        kw.update(host_cache_gb=HOST, state_checkpoints=1)
    eng = _engine(path, max_prefill_tokens=B, max_model_len=128, **kw)
    try:
        sp = SamplingParams(max_new_tokens=NEW, ignore_eos=True, logprobs=1)
        counts = None
        if hit:
            eng.generate([first], sp)
            rad, tier = eng.radix, eng.host_tier
            counts = dict(running_offloaded=rad.num_running_offloaded, saved_before=tier.ckpts_saved)
            if flush:
                rad.evict_checkpoints(rad.num_ckpts)
                counts["host_ckpts"] = len(tier.ckpts)
            _poison(eng, rad)
        spy = PrefillSpy(eng.runner)
        (req,) = eng.generate([second], sp)
        if hit:
            counts.update(restored=tier.ckpts_restored)
        return list(req.output_ids), [x[0] for x in req.logprobs], [x for q, x, _ in spy.starts if q == req.rid], counts
    finally:
        eng.close()


def test_a_checkpoint_offloaded_while_its_request_ran_restores_bitwise(glm):
    path, ref, vocab = glm
    first, second = _prompts(vocab, B)
    hit = _run(path, first, second, "hit")
    cold = _run(path, first, second, "cold")
    c = hit[3]
    assert hit[2] == [B], f"request 2 should resume at {B}, prefilled from {hit[2]}"
    assert cold[2] == [0, B], cold[2]
    assert c["running_offloaded"] >= 1 and c["saved_before"] >= 1 and c["restored"] == 1, c
    assert hit[0] == cold[0] and hit[1] == cold[1], "a hit restored from the host must be bitwise the cold run"
    want, lp_ref = _reference(ref, second + hit[0], len(second))
    err = (torch.tensor(hit[1]) - lp_ref.gather(1, torch.tensor(want).unsqueeze(1))[:, 0]).abs().max().item()
    print(f"\n  resume at {B}, {c}; hit == cold bitwise, vs reference {err:.2e}")
    assert hit[0] == want and err < 1e-4


def test_restoring_the_wrong_offloaded_slot_is_caught(glm, monkeypatch):
    """The host cache hands back another checkpoint's slot (request 1's decode checkpoint, flushed to the host beside
    the b one): the hit still resumes at b, and must not equal cold."""
    from trnrt.engine.host_cache import HostTier

    path, _, vocab = glm
    real = HostTier.find_ckpt

    def wrong(self, ids, lo, hi):
        got = real(self, ids, lo, hi)
        others = [s for s in self.ckpts.values() if got is not None and s != got[1]]
        return got if got is None or not others else (got[0], others[-1])

    first, second = _prompts(vocab, B)
    cold = _run(path, first, second, "cold")
    monkeypatch.setattr(HostTier, "find_ckpt", wrong)
    hit = _run(path, first, second, "hit", flush=True)
    assert hit[2] == [B] and hit[3]["restored"] == 1 and hit[3]["host_ckpts"] >= 2, hit[3]
    dl = max(abs(p - q) for p, q in zip(hit[1], cold[1]))
    print(f"\n  wrong slot restored: tokens {'equal' if hit[0] == cold[0] else 'DIFFER'}, max |dlogprob| {dl:.2e}")
    assert hit[0] != cold[0] or dl > 1e-3


def _two_at_once(path, ref, vocab):
    """Two prefixes' b checkpoints on the host, then X (prefix 1 + 4) and Y (prefix 2 + 4) added together on an engine
    with one row: X's admission restores prefix 1's checkpoint into the row (X has yet to copy from it), then Y's
    restore wants a row. -> {X, Y: (tokens equal the reference, max |dlogprob| against it, cached tokens)}."""
    from trnrt.engine.request import SamplingParams

    g = torch.Generator().manual_seed(77)
    p1, p2 = (torch.randint(0, vocab, (B,), generator=g).tolist() for _ in range(2))
    a1, a2, x, y = (torch.randint(0, vocab, (n,), generator=g).tolist() for n in (3, 3, 4, 4))
    eng = _engine(path, max_prefill_tokens=B, max_model_len=128, max_num_seqs=2, host_cache_gb=HOST,
                  state_checkpoints=1)
    try:
        sp = SamplingParams(max_new_tokens=NEW, ignore_eos=True, logprobs=1)
        eng.generate([p1 + a1], sp)   # prompt end at B: each leaves its b checkpoint
        eng.generate([p2 + a2], sp)
        rad = eng.radix
        rad.evict_checkpoints(rad.num_ckpts)   # every checkpoint to the host: the row is free
        _poison(eng, rad)
        reqs = [eng.add_request(p1 + x, sp), eng.add_request(p2 + y, sp)]
        eng.step()
        assert {r.status.value for r in reqs} == {"running"}, "both should be admitted in one step"
        while any(r.status.value != "finished" for r in reqs):
            eng.step()
        out = {}
        for tag, r in zip("XY", reqs):
            want, lp_ref = _reference(ref, list(r.prompt_ids) + list(r.output_ids), len(r.prompt_ids))
            lp = torch.tensor([v[0] for v in r.logprobs])
            out[tag] = (list(r.output_ids) == want,
                        (lp - lp_ref.gather(1, torch.tensor(r.output_ids).unsqueeze(1))[:, 0]).abs().max().item(),
                        r.num_cached_tokens)
        return out
    finally:
        eng.close()


@pytest.mark.parametrize("keep", ["honoured", "ignored"])
def test_a_row_a_request_has_yet_to_copy_from_stays(glm, monkeypatch, keep):
    """keep honoured: X resumes at b from its restored row, Y finds no row it may take and recomputes; both equal the
    reference. keep ignored (the mutation): Y's restore takes X's row and X resumes from prefix 2's state."""
    from trnrt.engine.radix_cache import RadixCache

    path, ref, vocab = glm
    if keep == "ignored":
        real = RadixCache.evict_checkpoints
        monkeypatch.setattr(RadixCache, "evict_checkpoints",
                            lambda self, n, keep=None: real(self, n, None if keep is None else set()))
    out = _two_at_once(path, ref, vocab)
    print(f"\n  keep {keep}: X (tokens equal, max |dlogprob|, cached) {out['X']}, Y {out['Y']}")
    assert out["X"][2] == B, "X resumes from its restored checkpoint"
    if keep == "honoured":
        assert all(ok and err < 1e-4 for ok, err, _ in out.values())
    else:
        assert not out["X"][0] or out["X"][1] > 1e-3
