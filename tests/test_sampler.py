"""engine/sampler.py on the host: the draw's semantics, the vocabulary-parallel form (sample_candidates on the union of
the shards' top max(K, N) is sample() on the whole logits), and the NaN guard every selection reads through."""

import pytest
import torch
import torch.nn.functional as F

from trnrt.engine import sampler as S
from trnrt.engine.sampler import NUM_TOP_LOGPROBS, apply_penalties, sample, unpack


def run(logits, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, K=64, seed=0):
    B = logits.shape[0]
    g = torch.Generator().manual_seed(seed)
    f = lambda v: torch.full((B,), v, dtype=torch.float32)  # noqa: E731
    return sample(logits, f(temperature), f(top_p), torch.full((B,), top_k, dtype=torch.int64),
                  f(min_p), torch.rand(B, K, generator=g))


def test_greedy_and_raw_logprobs():
    logits = torch.randn(3, 1000)
    tokens, lps, top_ids, top_lps = unpack(run(logits), 3)
    ref = torch.log_softmax(logits, -1)
    assert tokens == logits.argmax(-1).tolist()
    for b in range(3):
        assert abs(lps[b] - ref[b, tokens[b]].item()) < 1e-4
        assert top_ids[b] == torch.topk(logits[b], NUM_TOP_LOGPROBS).indices.tolist()
        assert abs(top_lps[b][0] - ref[b].max().item()) < 1e-4


def test_top_k_one_is_greedy_even_with_temperature():
    logits = torch.randn(4, 500)
    tokens, *_ = unpack(run(logits, temperature=1.5, top_k=1), 4)
    assert tokens == logits.argmax(-1).tolist()


def test_min_p_drops_tokens_below_threshold():
    logits = torch.full((1, 100), -10.0)
    logits[0, 5], logits[0, 6], logits[0, 7] = 5.0, 4.9, 0.0  # p(7) << 0.5 * p(5)
    seen = set()
    for seed in range(200):
        tokens, *_ = unpack(run(logits, temperature=1.0, min_p=0.5, seed=seed), 1)
        seen.add(tokens[0])
    assert seen == {5, 6}


def test_sampling_matches_the_distribution():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.2] + [1e-9] * 61]))
    counts = torch.zeros(64)
    n = 4000
    for seed in range(n // 50):
        out = run(logits.repeat(50, 1), temperature=1.0, seed=seed)
        tokens, *_ = unpack(out, 50)
        for t in tokens:
            counts[t] += 1
    freq = counts[:3] / n
    assert torch.allclose(freq, torch.tensor([0.5, 0.3, 0.2]), atol=0.03), freq


def test_top_p_keeps_the_smallest_set_covering_p():
    logits = torch.log(torch.tensor([[0.6, 0.25, 0.1, 0.05] + [1e-9] * 60]))
    seen = set()
    for seed in range(300):
        tokens, *_ = unpack(run(logits, temperature=1.0, top_p=0.8, seed=seed), 1)
        seen.add(tokens[0])
    assert seen == {0, 1}


def test_penalties_match_vllm_semantics():
    logits = torch.tensor([[2.0, -1.0, 0.5, 3.0, 1.0]])
    hist = torch.tensor([[0, 1, 3, 0]])  # padded by repeating the first id
    counts = torch.tensor([[2.0, 0.0, 1.0, 2.0]])  # token 1 seen in the prompt only
    out = apply_penalties(logits, hist, counts, torch.tensor([2.0]), torch.tensor([0.1]), torch.tensor([0.5]))
    expect = torch.tensor([[2.0 / 2 - 0.2 - 0.5, -1.0 * 2, 0.5, 3.0 / 2 - 0.1 - 0.5, 1.0]])
    assert torch.allclose(out, expect)


def test_penalized_token_is_not_sampled_but_logprobs_stay_raw():
    logits = torch.tensor([[5.0, 4.0] + [-10.0] * 62])
    B = 1
    f = lambda v: torch.full((B,), v)  # noqa: E731
    pen = (torch.tensor([[0]]), torch.tensor([[3.0]]), f(1.0), f(1.0), f(0.0))
    out = sample(logits, f(0.0), f(1.0), torch.zeros(B, dtype=torch.int64), f(0.0), torch.rand(B, 64),
                 penalties=pen)
    tokens, lps, top_ids, _ = unpack(out, 1)
    assert tokens == [1] and top_ids[0][0] == 0
    assert abs(lps[0] - torch.log_softmax(logits, -1)[0, 1].item()) < 1e-5


# -- the vocabulary-parallel form ----------------------------------------------------------------------------------

def shards(logits: torch.Tensor, tp: int, M: int):
    """The union of tp shards' top M (values, fp32 ids) and their log-sum-exps combined, as the vocabulary-parallel
    sampler gathers them."""
    B, V = logits.shape
    Vr = -(-V // tp)
    pad = F.pad(logits, (0, tp * Vr - V), value=-1e30)
    vals, ids, lses = [], [], []
    for r in range(tp):
        local = pad[:, r * Vr : (r + 1) * Vr]
        v, i = S.topk_large(local, M)
        vals.append(v)
        ids.append(i.float() + r * Vr)
        lses.append(S.logsumexp_large(local))
    return torch.cat(vals, 1), torch.cat(ids, 1), torch.logsumexp(torch.cat(lses, 1), dim=-1, keepdim=True)


def params(B, temperature, top_k, top_p, min_p=0.0):
    return (torch.full((B,), temperature), torch.full((B,), top_p), torch.full((B,), top_k, dtype=torch.int64),
            torch.full((B,), min_p))


def check_same(got, want, N=NUM_TOP_LOGPROBS):
    assert torch.equal(got[:, 0], want[:, 0])  # the token
    assert torch.equal(got[:, 2 : 2 + N], want[:, 2 : 2 + N])  # the top-N ids
    lp = torch.cat([got[:, 1:2], got[:, 2 + N :]], 1), torch.cat([want[:, 1:2], want[:, 2 + N :]], 1)
    torch.testing.assert_close(lp[0], lp[1], rtol=0, atol=8 * torch.finfo(torch.float32).eps * 32)  # ~a few ulps


SETTINGS = [(0.0, 0, 1.0, 0.0), (0.7, 20, 0.9, 0.0), (1.0, 50, 0.95, 0.0), (0.7, 0, 0.9, 0.05), (1.0, 0, 1.0, 0.0)]


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("B", [1, 2, 8])
@pytest.mark.parametrize("setting", SETTINGS)
def test_candidates_sample_as_the_logits(tp, B, setting):
    """Greedy, and sampled under one noise stream with top-k / top-p / min-p, over a vocabulary that does not divide by
    tp (the last shard has padding columns): the same token and top-N ids, logprobs within a few fp32 ulps (the
    shards' log-sum-exps combined)."""
    g = torch.Generator().manual_seed(tp * 100 + B)
    V, K = 1003, 64
    logits = torch.randn(B, V, generator=g) * 3 - 2
    noise = torch.rand(B, K, generator=g)
    p = params(B, *setting)
    want = sample(logits, *p, noise)
    cv, ci, lse = shards(logits, tp, max(K, NUM_TOP_LOGPROBS))
    check_same(S.sample_candidates(cv, ci, lse, *p, noise), want)


@pytest.mark.parametrize("tp", [2, 4])
def test_one_candidate_k(tp):
    g = torch.Generator().manual_seed(tp)
    logits = torch.randn(3, 517, generator=g)
    noise = torch.rand(3, 1, generator=g)
    p = params(3, 0.0, 0, 1.0)
    cv, ci, lse = shards(logits, tp, NUM_TOP_LOGPROBS)
    check_same(S.sample_candidates(cv, ci, lse, *p, noise), sample(logits, *p, noise))


# -- the NaN guard (_selectable) ------------------------------------------------------------------------------------

def _topk_large_unguarded(x, k, chunk=128):
    """topk_large without the guard: what a selection over a NaN row reads on a device that returns out-of-range ids."""
    B, V = x.shape
    if -(-V // chunk) < k:
        return torch.topk(x, k, dim=-1)
    pad = (-V) % chunk
    if pad:
        x = F.pad(x, (0, pad), value=-1e30)
    xc = x.view(B, -1, chunk)
    cmax = xc.amax(dim=-1)
    _, ci = torch.topk(cmax, k, dim=-1)
    cand = torch.gather(xc, 1, ci.unsqueeze(-1).expand(B, k, chunk)).reshape(B, k * chunk)
    vals, j = torch.topk(cand, k, dim=-1)
    idx = torch.gather(ci, 1, torch.div(j, chunk, rounding_mode="floor")) * chunk + j % chunk
    return vals, idx


def _args(B=4, K=8, V=4096, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, V, generator=g) * 3
    return (logits, torch.tensor([0.0, 0.7, 1.0, 1.3])[:B], torch.full((B,), 0.9), torch.zeros(B, dtype=torch.long),
            torch.full((B,), 0.02), torch.rand(B, K, generator=g))


@pytest.mark.parametrize("V", [300, 4096, 32000])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_topk_large_is_unchanged_on_finite_rows(V, seed):
    x = torch.randn(6, V, generator=torch.Generator().manual_seed(seed))
    for k in (1, 8, 24):
        v, i = S.topk_large(x, k)
        v0, i0 = _topk_large_unguarded(x, k)
        assert torch.equal(v, v0) and torch.equal(i, i0)


@pytest.mark.parametrize("V", [300, 4096])
def test_nan_and_neg_inf_rows_select_in_range(V):
    logits, *rest = _args(V=V)
    logits[1] = float("nan")
    logits[2] = float("-inf")
    out = S.sample(logits, *rest)
    ids = torch.cat([out[:, :1], out[:, 2:2 + NUM_TOP_LOGPROBS]], 1)
    assert ((ids >= 0) & (ids < V)).all()
    ref = S.sample(*(_args(V=V)[0][[0, 3]],) + tuple(r[[0, 3]] for r in rest))
    assert torch.equal(out[[0, 3]], ref), "a NaN row changed its neighbours' samples"
