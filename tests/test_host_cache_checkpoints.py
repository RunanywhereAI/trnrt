"""A prefix hit restored from the host cache (engine/host_cache.py) is BITWISE the same request run cold, on
GLM-5.3-Flash's architecture (tests/test_glm5_next.py's small random config: KDA states, MLA + DSA pages), fp32 on CPU.

Run: python3 -m pytest tests/test_host_cache_checkpoints.py -q -s

Request 1 (a preamble of b tokens and 3 more) leaves its prompt-end checkpoint at b, its prompt's last page boundary,
which ends its first chunk (max_prefill_tokens = b); request 2 (the same preamble and 4 other tokens) resumes there.
Cold, on a fresh engine, request 2 computes exactly [0, b) and then its suffix, as the hit does after its restore, so
the two must agree bit for bit. Between the requests the checkpoint is forced out to the host cache:
- "rows": the checkpoints alone (RadixCache.evict_checkpoints, what running out of checkpoint rows does; the pages
  stay on the device);
- "all": the checkpoints and every page under them (RadixCache.evict).
Request 2 must then restore the checkpoint (and in "all" the pages) from the host, resume at b, and equal cold.
"device" (nothing evicted) is the control. Before request 2 every free checkpoint row and free page is overwritten with
garbage: a freed row still holds the evicted state and the restore may be handed that same row, so without it a
restore that copied nothing would pass. Two mutations pin that the hit reads the host's copy: no state copy back, and
no page copy back.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

NEW = 8
B = 40   # past the model's index_topk 16: the sparse attention regime
PS = 4   # tests/test_glm5_next.py engine(): page_size 4
HOST = 0.05   # GiB per rank: ample for the tiny model's rows and pages


@pytest.fixture(scope="module")
def glm(tmp_path_factory):
    from tests.test_glm5_next import build

    path = tmp_path_factory.mktemp("glm5_next_host_cache")
    hf = build(str(path), seed=1, index_topk=16)
    return str(path), hf, hf.config.text_config.vocab_size


def _engine(path, **kw):
    from tests.test_glm5_next import engine

    return engine(path, **kw)


def _prompts(vocab: int, b: int):
    """(request 1, request 2): a preamble of b tokens, then 3 tokens (its prompt-end checkpoint is at b) and 4 others
    (no checkpoint target past b, so the hit's suffix chunk and the cold one have the same form)."""
    g = torch.Generator().manual_seed(31 + b)
    pre = torch.randint(0, vocab, (b,), generator=g).tolist()
    s1 = torch.randint(0, vocab, (3,), generator=g).tolist()
    s2 = torch.randint(0, vocab, (4,), generator=g).tolist()
    assert (b + len(s1) - 1) // PS * PS == b
    return pre + s1, pre + s2


class PrefillSpy:
    """(request id, start, end) of every prefill entry the runner runs."""

    def __init__(self, runner):
        self.starts = []
        orig = runner.prefill

        def wrap(s, *a, _orig=orig):
            for x in (s if isinstance(s, list) else [s]):
                self.starts.append((x.req.rid, x.start, x.end))
            return _orig(s, *a)

        runner.prefill = wrap


POISON = 3.0   # finite: harmless where a read is masked (a page's unwritten tail), wrong anywhere it is not


def _poison(eng, rad) -> None:
    run, st = eng.runner, eng.runner.state
    for row in st.free_ckpt_rows():
        for p in st.pools():
            p[row].fill_(POISON)
    for page in rad.pool._free:
        for c in run._paged_caches():
            c[page * run.lps : (page + 1) * run.lps].fill_(POISON)


def _run(path, first, second, mode: str, **kw):
    """Request 2's (output ids, logprobs, prefill starts, host cache counts) after request 1 on the same engine, its
    checkpoints evicted per `mode` in between; "cold": request 2 alone on a fresh engine."""
    from trnrt.engine.request import SamplingParams

    hit = mode != "cold"
    if hit:
        kw.setdefault("host_cache_gb", HOST)
    eng = _engine(path, max_prefill_tokens=B, max_model_len=128, **kw)
    try:
        sp = SamplingParams(max_new_tokens=NEW, ignore_eos=True, logprobs=1)
        if hit:
            eng.generate([first], sp)
            rad, tier = eng.radix, eng.host_tier
            assert rad.num_ckpts >= 1, "request 1 should leave its checkpoint at B in the tree"
            assert tier.ckpt_capacity >= 2 and tier.capacity >= 64, (tier.ckpt_capacity, tier.capacity)
            if mode == "rows":
                rad.evict_checkpoints(rad.num_ckpts)
                assert rad.num_ckpts == 0 and rad.total_pages() > 0
            elif mode == "all":
                rad.evict(rad.total_pages())
                assert rad.num_ckpts == 0 and rad.total_pages() == 0
            _poison(eng, rad)
        spy = PrefillSpy(eng.runner)
        (req,) = eng.generate([second], sp)
        counts = None
        if hit:
            counts = dict(ckpts_saved=tier.ckpts_saved, ckpts_restored=tier.ckpts_restored, pages_restored=tier.restored)
        return list(req.output_ids), [x[0] for x in req.logprobs], [x for q, x, _ in spy.starts if q == req.rid], counts
    finally:
        eng.close()


def _reference(ref, seq, start):
    """(argmax at positions start - 1 .. len(seq) - 2, their log-softmax rows), teacher forced in one call."""
    with torch.no_grad():
        logits = ref(torch.tensor([seq]), use_cache=False).logits[0]
    rows = logits[start - 1 : len(seq) - 1].float()
    return rows.argmax(-1).tolist(), torch.log_softmax(rows, -1)


@pytest.mark.parametrize("mode", ["rows", "all", "device"])
def test_a_hit_restored_from_the_host_cache_is_bitwise_the_cold_run(glm, mode):
    path, ref, vocab = glm
    first, second = _prompts(vocab, B)
    hit = _run(path, first, second, mode)
    cold = _run(path, first, second, "cold")
    assert hit[2] == [B], f"request 2 should resume at {B}, prefilled from {hit[2]}"
    assert cold[2] == [0, B], cold[2]
    c = hit[3]
    if mode == "device":
        assert c["ckpts_restored"] == 0 and c["pages_restored"] == 0
    else:
        assert c["ckpts_saved"] >= 1 and c["ckpts_restored"] == 1, c
        assert (c["pages_restored"] == B // PS) if mode == "all" else (c["pages_restored"] == 0), c
    assert hit[0] == cold[0] and hit[1] == cold[1], "a hit restored from the host must be bitwise the cold run"
    want, lp_ref = _reference(ref, second + hit[0], len(second))
    err = (torch.tensor(hit[1]) - lp_ref.gather(1, torch.tensor(want).unsqueeze(1))[:, 0]).abs().max().item()
    print(f"\n  {mode}: resume at {B}, hit == cold bitwise, host {c}; vs reference max |dlogprob| {err:.2e}")
    assert hit[0] == want and err < 1e-4


@pytest.mark.parametrize("broken", ["state_load", "kv_load"])
def test_the_hit_reads_the_host_copy(glm, monkeypatch, broken):
    """Each mutation still resumes at B from a "restored" checkpoint, and must not equal cold."""
    from trnrt.engine.model_runner import ModelRunner

    path, _, vocab = glm
    mode = "all" if broken == "kv_load" else "rows"
    if broken == "state_load":
        monkeypatch.setattr(ModelRunner, "_state_load", lambda self, slot, row: None)
    else:
        monkeypatch.setattr(ModelRunner, "_kv_load", lambda self, slot, page: None)
    first, second = _prompts(vocab, B)
    hit = _run(path, first, second, mode)
    cold = _run(path, first, second, "cold")
    assert hit[2] == [B] and hit[3]["ckpts_restored"] == 1
    dl = max(abs(p - q) for p, q in zip(hit[1], cold[1]))
    print(f"\n  {broken} broken: tokens {'equal' if hit[0] == cold[0] else 'DIFFER'}, max |dlogprob| {dl:.2e}")
    assert hit[0] != cold[0] or dl > 1e-3
