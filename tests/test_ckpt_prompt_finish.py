"""The prompt-end and finish state checkpoints a returning agent turn resumes from, on GLM-5.3-Flash's architecture
(glm5_next: KDA, pooled-DSA MLA, MoE; tests/test_glm5_next.py's small random config, page size 4, prefill chunks of 8),
fp32 on CPU. Turn 2 is turn 1's prompt + its reply + new tokens, as an agent's next request is:
- a replay of the reply token for token resumes from the finish checkpoint, at turn 1's last page boundary (the state
  after prompt + generated[:-1], floored to a page: the tail under a page is recomputed);
- a replay that differs inside the reply (a re-rendered tool call) resumes from the prompt-end checkpoint, which
  free_dominated keeps (the finish checkpoint is deeper on the same path and would otherwise free it); without that
  rule, further back;
- every turn's output equals a cold prefill's (a fresh engine), token for token.
"""

import pytest

pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

PS = 4  # tests/test_glm5_next.py engine(): page_size 4, max_prefill_tokens 8


def sp(n=12, **kw):
    from trnrt.engine.request import SamplingParams

    return SamplingParams(max_new_tokens=n, ignore_eos=True, **kw)


@pytest.fixture(scope="module")
def glm(tmp_path_factory):
    from tests.test_glm5_next import build

    path = str(tmp_path_factory.mktemp("glm5_next_ckpt"))
    build(path, seed=1, index_topk=16)
    return path


def eng_(path, **kw):
    from tests.test_glm5_next import engine

    return engine(path, **kw)


def mk(seed, lengths):
    from tests.test_glm5_next import prompts

    return prompts(seed, lengths)


def page_floor(n: int) -> int:
    return n // PS * PS


def cold(glm, prompt, n):
    fresh = eng_(glm)
    try:
        (r,) = fresh.generate([prompt], sp(n))
        return r.output_ids
    finally:
        fresh.close()


def two_turns(eng, p1, n1, tail, edit=None):
    """Turn 1 on `eng`, then turn 2 = p1 + its reply (edited by `edit` when given) + tail; both requests."""
    (r1,) = eng.generate([p1], sp(n1))
    reply = list(r1.output_ids)
    if edit is not None:
        reply = edit(reply)
    (r2,) = eng.generate([p1 + reply + tail], sp(8))
    return r1, r2, p1 + reply + tail


def altered(reply):
    """The reply with its third token changed: a replay that differs inside the reply."""
    out = list(reply)
    out[2] = (out[2] + 1) % 384  # tests/test_glm5_next.py SMALL vocab
    return out


@pytest.mark.parametrize("p_len", [29, 28])  # unaligned, and ending exactly on a page boundary
def test_a_replayed_reply_resumes_from_the_finish_checkpoint(glm, p_len):
    (p1,), (tail,) = mk(81, (p_len,)), mk(82, (6,))
    eng = eng_(glm)
    try:
        r1, r2, p2 = two_turns(eng, p1, 13, tail)
        fed = len(p1) + len(r1.output_ids) - 1  # the last sampled token is never fed
        assert r2.num_cached_tokens == page_floor(fed), (r2.num_cached_tokens, fed)
        assert r2.output_ids == cold(glm, p2, 8)
        assert r1.output_ids == cold(glm, p1, 13)
        print(f"prompt {len(p1)}: turn 2 resumed at {r2.num_cached_tokens} of {fed} fed tokens")
    finally:
        eng.close()


@pytest.mark.parametrize("p_len", [29, 28])
def test_a_reply_that_differs_resumes_from_the_prompt_end_checkpoint(glm, p_len):
    """The finish checkpoint dominates the prompt-end one on the same path, and the nearest prompt-end checkpoint is
    kept anyway. Its position is the prompt's last page boundary strictly before the newest token, so a prompt ending
    on a boundary keeps it one page back."""
    (p1,), (tail,) = mk(81, (p_len,)), mk(82, (6,))
    eng = eng_(glm)
    try:
        r1, r2, p2 = two_turns(eng, p1, 13, tail, edit=altered)
        assert r2.num_cached_tokens == page_floor(len(p1) - 1), (r2.num_cached_tokens, len(p1))
        assert r2.output_ids == cold(glm, p2, 8)
        print(f"prompt {len(p1)}: the altered replay resumed at {r2.num_cached_tokens}")
    finally:
        eng.close()


def test_free_dominated_without_the_anchor_rule_would_lose_the_prompt_end_checkpoint(glm, monkeypatch):
    """Positive control for the rule: a prompt-end checkpoint attached as an ordinary one (anchor False) is freed when
    turn 1 releases, so the altered replay falls back further."""
    from trnrt.engine.radix_cache import RadixCache

    (p1,), (tail,) = mk(81, (29,)), mk(82, (6,))
    real = RadixCache.attach_checkpoint
    monkeypatch.setattr(
        RadixCache, "attach_checkpoint", lambda self, ids, n, row, anchor=False: real(self, ids, n, row, anchor=False)
    )
    eng = eng_(glm)
    try:
        _, r2, p2 = two_turns(eng, p1, 13, tail, edit=altered)
        assert r2.num_cached_tokens < page_floor(len(p1) - 1), r2.num_cached_tokens
        assert r2.output_ids == cold(glm, p2, 8)
    finally:
        eng.close()


def test_free_dominated_keeps_one_prompt_end_checkpoint_per_path(glm):
    """Three turns of one session: after turn 3 the path holds turn 3's checkpoints, and of the prompt-end ones only
    the latest; turn 1's and turn 2's were dominated and freed."""
    (p1,), tails = mk(81, (29,)), mk(83, (5, 6))
    eng = eng_(glm)
    try:
        prompt = p1
        (r,) = eng.generate([prompt], sp(9))
        for tail in tails:
            prompt = prompt + list(r.output_ids) + tail
            (r,) = eng.generate([prompt], sp(9))
        anchors = [n for n in _nodes(eng.scheduler.radix) if n.ckpt is not None and n.anchor]
        assert len(anchors) == 1 and _depth(anchors[0]) == page_floor(len(prompt) - 1)
    finally:
        eng.close()


def _nodes(radix):
    stack = [radix.root]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children.values())


def _depth(node) -> int:
    d = 0
    while node is not None and node.parent is not None:
        d += len(node.key)
        node = node.parent
    return d
