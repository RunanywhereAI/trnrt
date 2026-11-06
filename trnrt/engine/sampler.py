"""On-device sampling, compiled into the same graph as the model.

Greedy, temperature, top-k, top-p and min-p over a fixed candidate set of the K highest
logits (K = noise.shape[1], EngineConfig.sampling_candidates). Exact for top_k <= K; for a
wider distribution the mass outside the top K is dropped, which is the price of a static
shape. Randomness comes in as uniform noise generated on the host per request, so a seeded
request is reproducible and the graph has no RNG state. Rows with temperature <= 0 are greedy
over the full vocab.

Logprobs follow vLLM's default (`logprobs_mode=raw_logprobs`): log-softmax of the raw
model logits, before temperature, penalties and filtering.

Two forms of the same draw:
- sample(): from the full logits [B, V];
- sample_candidates(): from a candidate set that holds the logits' top max(K, N). This is the
  vocabulary-parallel form the served decode and prefill graphs use (models/decoder.py):
  each tensor-parallel rank takes its vocabulary shard's top max(K, N) and log-sum-exp, and one
  all-reduce gathers B x tp x (1 + 2 max(K, N)) floats instead of the logits over the whole
  vocabulary. On GLM-5.3-Flash at TP 32 the gathered form costs, from the gather to the step's end,
  0.44 ms at batch 1 and 1.70 ms at batch 2 per decode step: the gather is bf16 [B, 154,880] and
  the sampler reads it in at least three full passes from HBM. The candidates, the token and the
  top N are the same as sample()'s; the log-sum-exp is combined from the shards', so a logprob may
  differ in its last fp32 bits. A call with penalties samples from the gathered logits (sample()).

The graph returns ONE fp32 tensor per call, so a step is a single device-to-host read:
    out[:, 0]          sampled token id (exact in fp32 for ids < 2**24)
    out[:, 1]          its logprob
    out[:, 2:2+N]      top-N token ids
    out[:, 2+N:2+2N]   top-N logprobs

A speculative verify (verify_sample) returns VERIFY_COLS columns per row instead: y, accept, r, then the raw logprobs of
y, r and the draft, then the top N ids and logprobs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

NUM_TOP_LOGPROBS = 20  # the OpenAI API maximum for top_logprobs


def _selectable(x: torch.Tensor) -> torch.Tensor:
    """x with NaN and -inf as the dtype's finite minimum: what a top-k or argmax selects from. On a Neuron device a
    top-k over a NaN (or all -inf) row may return out-of-range ids (0xFFFFFFFF on trn2), and every id here feeds a
    gather; a padded row's logits can be NaN. So each selection reads a sanitized copy, its ids are clamped, and every
    VALUE is read from the original tensor: on finite rows the output is unchanged bit for bit. The minimum is x's own
    dtype's: a wider dtype's minimum cast down becomes -inf, which would undo the sanitizing."""
    return torch.where(torch.isnan(x) | torch.isneginf(x), torch.full_like(x, torch.finfo(x.dtype).min), x)


def topk_large(x: torch.Tensor, k: int, chunk: int = 128):
    """Exact top-k over a long last axis, for [B small, V large] on Trainium.

    A plain topk / argmax / logsumexp over [B, V] puts B rows on 128 partitions, so a decode
    batch uses a few lanes (measured on trn1: a top-64 over a 151,936-entry vocabulary took
    3.37 ms at B = 6). Folding to [B, V / chunk, chunk] makes the rows plentiful: take each
    chunk's max, the k best chunks, then top-k inside those k * chunk candidates. Exact,
    because every element of the true top-k lies in a chunk whose max is among the top k.
    """
    B, V = x.shape
    s = _selectable(x)
    if -(-V // chunk) < k:  # fewer chunks than k: nothing to gain
        idx = torch.topk(s, k, dim=-1).indices.clamp(0, V - 1)
        return torch.gather(x, 1, idx), idx
    pad = (-V) % chunk
    if pad:
        s = F.pad(s, (0, pad), value=-1e30)
    sc = s.view(B, -1, chunk)
    cmax = sc.amax(dim=-1)
    ci = torch.topk(cmax, k, dim=-1).indices.clamp(0, sc.shape[1] - 1)
    cand = torch.gather(sc, 1, ci.unsqueeze(-1).expand(B, k, chunk)).reshape(B, k * chunk)
    j = torch.topk(cand, k, dim=-1).indices.clamp(0, k * chunk - 1)
    idx = (torch.gather(ci, 1, torch.div(j, chunk, rounding_mode="floor")) * chunk + j % chunk).clamp(0, V - 1)
    return torch.gather(x, 1, idx), idx


def logsumexp_large(x: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    """logsumexp over a long last axis, folded the same way. Returns [B, 1]."""
    B, V = x.shape
    if chunk >= V:
        return torch.logsumexp(x, dim=-1, keepdim=True)
    pad = (-V) % chunk
    if pad:
        x = F.pad(x, (0, pad), value=-1e30)
    return torch.logsumexp(torch.logsumexp(x.view(B, -1, chunk), dim=-1), dim=-1, keepdim=True)


def apply_penalties(
    logits: torch.Tensor,
    hist_ids: torch.Tensor,
    out_counts: torch.Tensor,
    rep: torch.Tensor,
    freq: torch.Tensor,
    pres: torch.Tensor,
) -> torch.Tensor:
    """vLLM semantics. hist_ids [B, M]: every distinct token in the prompt or the output
    (rows padded by repeating their first entry, which makes the scatter idempotent);
    out_counts [B, M]: occurrences in the output only. repetition_penalty divides positive
    and multiplies negative logits of any seen token; frequency and presence penalties
    subtract per output occurrence and once per distinct output token."""
    g = torch.gather(logits, 1, hist_ids)
    r = rep.unsqueeze(1)
    g = torch.where(g > 0, g / r, g * r)
    g = g - freq.unsqueeze(1) * out_counts - pres.unsqueeze(1) * (out_counts > 0).float()
    return torch.scatter(logits, 1, hist_ids, g)


def _draw(
    cv: torch.Tensor,
    ci: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    min_p: torch.Tensor,
    noise: torch.Tensor,
):
    """One sampled candidate per row from the K candidates cv [B, K] (fp32 logits, sorted descending) whose ids are
    ci [B, K]: the distribution at the row's temperature, filtered by top-p, top-k and min-p (the best candidate
    always kept), drawn with the Gumbel-max trick on `noise`. Returns (its id from ci, its index [B, 1] into cv)."""
    K = cv.shape[1]
    t = torch.clamp(temperature, min=1e-5).unsqueeze(1)
    probs = torch.softmax(cv / t, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    rank = torch.arange(K, device=cv.device).unsqueeze(0)
    keep = (cum - probs) < top_p.unsqueeze(1)
    keep = keep & ((rank < top_k.unsqueeze(1)) | (top_k.unsqueeze(1) <= 0))
    keep = keep & (probs >= min_p.unsqueeze(1) * probs[:, :1])
    keep = keep | (rank == 0)  # never an empty set
    gumbel = -torch.log(-torch.log(torch.clamp(noise, 1e-10, 1.0 - 1e-7)))
    scores = torch.where(keep, torch.log(probs) + gumbel, -1e30)
    choice = torch.argmax(_selectable(scores), dim=-1, keepdim=True).clamp(0, K - 1)
    return torch.gather(ci, 1, choice).squeeze(1), choice


def sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    min_p: torch.Tensor,
    noise: torch.Tensor,
    bitmask: torch.Tensor | None = None,
    penalties: tuple | None = None,
) -> torch.Tensor:
    """logits [B, V] fp32; temperature/top_p/min_p [B] fp32; top_k [B] int64 (0 = off);
    noise [B, K] uniform in (0, 1); bitmask: always None (no guided decoding; the parameter keeps
    the models' call sites); penalties: optional (hist_ids, out_counts, rep, freq, pres) for
    apply_penalties, which shape what is sampled, never the reported raw logprobs.
    Returns [B, 2 + 2N] fp32."""
    if bitmask is not None:
        raise ValueError("guided decoding (a token bitmask) is not supported")
    K = noise.shape[1]
    N = NUM_TOP_LOGPROBS
    lse = logsumexp_large(logits)
    raw_vals, raw_idx = topk_large(logits, max(K, N))  # sorted descending
    if penalties is None:
        vals, idx = raw_vals, raw_idx
    else:
        vals, idx = topk_large(apply_penalties(logits, *penalties), max(K, 1))
    greedy = idx[:, 0]
    if K > 1:
        sampled, _ = _draw(vals[:, :K], idx[:, :K], temperature, top_p, top_k, min_p, noise)
        token = torch.where(temperature <= 0, greedy, sampled)
    else:
        token = greedy
    token_lp = torch.gather(logits, 1, token.unsqueeze(1)) - lse
    top_lp = raw_vals[:, :N] - lse
    return torch.cat([token.unsqueeze(1).float(), token_lp, raw_idx[:, :N].float(), top_lp], dim=1)


def sample_candidates(
    cand_vals: torch.Tensor,
    cand_ids: torch.Tensor,
    lse: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    min_p: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """sample() without penalties, from a candidate set instead of the logits: cand_vals [B, C] fp32 logits,
    cand_ids [B, C] their vocabulary ids as fp32 (exact below 2^24), C >= max(K, N), holding the logits' top
    max(K, N), and lse [B, 1] the logits' log-sum-exp. The vocabulary-parallel sampler passes the union of every
    shard's top max(K, N), which holds the logits' top max(K, N) (each of those is in its own shard's top
    max(K, N)). So the candidates sample() draws from, its token and its top N are the same, and the token's logit is
    read from the candidates. The ids stay fp32 throughout: no int cast in the graph."""
    K = noise.shape[1]
    N = NUM_TOP_LOGPROBS
    M = max(K, N)
    j = torch.topk(_selectable(cand_vals), M, dim=-1).indices.clamp(0, cand_vals.shape[1] - 1)  # sorted descending
    vals, idx = torch.gather(cand_vals, 1, j), torch.gather(cand_ids, 1, j)
    greedy = idx[:, 0]
    tok = vals[:, :1]
    if K > 1:
        cv = vals[:, :K]
        sampled, choice = _draw(cv, idx[:, :K], temperature, top_p, top_k, min_p, noise)
        greedy_t = temperature <= 0
        token = torch.where(greedy_t, greedy, sampled)
        tok = torch.where(greedy_t.unsqueeze(1), tok, torch.gather(cv, 1, choice))
    else:
        token = greedy
    token_lp = tok - lse
    top_lp = vals[:, :N] - lse
    return torch.cat([token.unsqueeze(1), token_lp, idx[:, :N], top_lp], dim=1)


VERIFY_COLS = 6 + 2 * NUM_TOP_LOGPROBS


def verify_sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    min_p: torch.Tensor,
    noise: torch.Tensor,
    u_accept: torch.Tensor,
    draft: torch.Tensor,
) -> torch.Tensor:
    """Speculative verification for one draft token per row (R = B x Q rows).

    Row i scores the draft for the NEXT position, d = draft[i], under the request's own
    sampling distribution p (temperature, top-k, top-p, min-p over the K candidates):
    accept with probability p(d) (greedy rows: d == argmax), else the replacement is a
    sample from p with d removed. For a deterministic proposal, such as the MTP head's greedy draft,
    this is exactly the rejection sampler vLLM uses, so the output distribution is unchanged.

    Columns: y (sample from p, the bonus token after a fully accepted draft), accept,
    r (replacement), raw logprobs of y, r and d, then top-N ids and logprobs.
    """
    K = noise.shape[1]
    N = NUM_TOP_LOGPROBS
    lse = logsumexp_large(logits)
    vals, idx = topk_large(logits, max(K, N))
    cv, ci = vals[:, :K], idx[:, :K]
    t = torch.clamp(temperature, min=1e-5).unsqueeze(1)
    probs = torch.softmax(cv / t, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    rank = torch.arange(K, device=logits.device).unsqueeze(0)
    keep = (cum - probs) < top_p.unsqueeze(1)
    keep = keep & ((rank < top_k.unsqueeze(1)) | (top_k.unsqueeze(1) <= 0))
    keep = keep & (probs >= min_p.unsqueeze(1) * probs[:, :1])
    keep = keep | (rank == 0)
    kept = torch.where(keep, probs, 0.0)
    kept = kept / kept.sum(dim=-1, keepdim=True)
    greedy = ci[:, 0]
    is_d = ci == draft.unsqueeze(1)
    p_d = (kept * is_d.float()).sum(dim=-1)
    sampling = temperature > 0
    accept = torch.where(sampling, u_accept < p_d, draft == greedy)
    gumbel = -torch.log(-torch.log(torch.clamp(noise, 1e-10, 1.0 - 1e-7)))
    scores = torch.where(keep, torch.log(kept) + gumbel, -1e30)
    pick = lambda sc: torch.argmax(_selectable(sc), -1, keepdim=True).clamp(0, K - 1)  # noqa: E731
    y = torch.where(sampling, torch.gather(ci, 1, pick(scores)).squeeze(1), greedy)
    rscores = torch.where(is_d, -1e30, scores)
    rsampled = torch.gather(ci, 1, pick(rscores)).squeeze(1)
    greedy_r = torch.where(draft == greedy, ci[:, 1], greedy)
    r = torch.where(sampling, rsampled, greedy_r)
    lp = lambda tok: torch.gather(logits, 1, tok.unsqueeze(1)) - lse  # noqa: E731
    return torch.cat(
        [
            y.unsqueeze(1).float(),
            accept.unsqueeze(1).float(),
            r.unsqueeze(1).float(),
            lp(y),
            lp(r),
            lp(draft),
            idx[:, :N].float(),
            vals[:, :N] - lse,
        ],
        dim=1,
    )


def unpack(out, n: int):
    """Host side: the first `n` rows of a sampler output as (tokens, logprobs, top_ids, top_lps)."""
    out = out[:n]
    N = NUM_TOP_LOGPROBS
    return (
        out[:, 0].long().tolist(),
        out[:, 1].tolist(),
        out[:, 2 : 2 + N].long().tolist(),
        out[:, 2 + N : 2 + 2 * N].tolist(),
    )
