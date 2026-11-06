"""Synthetic acceptance for speculative decoding: a benchmark mode, off by default.

InferenceX fixes the acceptance length (AL) of every submission at a committed "golden" value per model and draft length,
and engines force it with synthetic acceptance: the draft runs, verification is real, and the number of accepted drafts a
step is drawn from a schedule whose mean is the golden AL. This module is that schedule, the same one vLLM merged for
this (vllm-project/vllm pull 40662; see THIRD_PARTY_NOTICES.md, "Designs followed"). GLM-5.3-Flash with 2 draft tokens has
a golden AL of 2.518506.

Set it with `TRNRT_SPEC_SYNTH_AL` (or `EngineConfig.spec_synth_al`). Use it only to compare with a published InferenceX
number: with it on, an accepted position emits the DRAFT token whether or not the target agrees, so the text a request
returns is the draft head's, not the target's. Real acceptance is the default and the only mode for any quality check.

The semantics:
- The configured per-position rates are UNCONDITIONAL: entry i is the marginal probability that the first i + 1 drafts are
  all accepted. A target AL becomes rates with the minimum-variance schedule `[1] * floor(AL - 1) + [AL - 1 - floor(AL - 1)]
  + [0]...` (length k), so the accepted count takes only the two values around AL - 1.
- The sampler converts them to CONDITIONAL rates c_i = p_i / p_(i-1) (0 where p_(i-1) = 0) and walks the draft positions in
  order, one uniform u per position, until the first rejection: accept position i iff u < c_i.
- An accepted position emits the DRAFT token. The first rejected position emits the TARGET's own sample at that position;
  if nothing is rejected the bonus token is the target's sample after the last draft (engine/spec_async.py `spec_post` and
  `Engine._accept` apply this rule).

Differences from vLLM, on purpose: vLLM's uniform at a position is a pure function of (request seed, absolute token
position); here it is a per-request stream, k draws a verify step (drafts or not), seeded as the sampler's noise is
(`SamplingParams.seed`, else (`EngineConfig.seed`, crc32(rid))), with a salt so the two streams differ. The distribution
of the accepted count is the same, and a run is reproducible given the request seeds and the same sequence of steps.
"""

from __future__ import annotations

import zlib

import numpy as np

SALT = 0x53594E54  # "SYNT": the synthetic stream is never the sampling noise stream


def min_variance_rates(al: float, n: int) -> list[float]:
    """Mean acceptance length -> unconditional per-position rates for n draft positions, with a range check:
    al in [1, n + 1]; al - 1 is the expected number of accepted drafts."""
    if not 1.0 <= al <= float(n + 1):
        raise ValueError(f"the synthetic acceptance length must be in [1, {n + 1}] for {n} draft token(s), got {al}")
    drafts = al - 1.0
    full = int(drafts)
    return ([1.0] * full + [drafts - full] + [0.0] * (n - full - 1))[:n]


def to_conditional(rates: list[float]) -> list[float]:
    """Unconditional -> conditional rates for the early-terminating rejection loop: c_i = p_i / p_(i-1), 0 where p_(i-1) is 0."""
    return [p / q if q > 0.0 else 0.0 for p, q in zip(rates, [1.0, *rates[:-1]], strict=True)]


class SynthAccept:
    """The accept flags of a request's verify steps. flags() gives one step's k flags; the engine takes the accepted PREFIX of
    them (the first 0 ends it)."""

    def __init__(self, al: float, k: int, seed: int = 0):
        if k < 1:
            raise ValueError("synthetic acceptance needs at least one draft token")
        self.al, self.k, self.seed = float(al), int(k), int(seed)
        self.rates = min_variance_rates(self.al, self.k)
        self.cond = to_conditional(self.rates)
        self._rngs: dict[str, np.random.Generator] = {}

    def _rng(self, rid: str, req_seed: int | None) -> np.random.Generator:
        r = self._rngs.get(rid)
        if r is None:
            ent = (
                [SALT, int(req_seed) & 0xFFFFFFFFFFFFFFFF]
                if req_seed is not None
                else [SALT, self.seed & 0xFFFFFFFFFFFFFFFF, zlib.crc32(rid.encode())]
            )
            r = self._rngs[rid] = np.random.default_rng(ent)
        return r

    def flags(self, rid: str, req_seed: int | None = None) -> list[float]:
        """One verify step of request `rid`: k flags, flag j = 1.0 iff the j-th uniform is below the conditional rate c_j.
        Always k uniforms are drawn (so the stream advances one step at a time whether or not the step had drafts, and the
        synchronous and the asynchronous engine consume it identically)."""
        u = self._rng(rid, req_seed).random(self.k)
        return [1.0 if x < c else 0.0 for x, c in zip(u.tolist(), self.cond, strict=True)]

    def forget(self, rid: str) -> None:
        self._rngs.pop(rid, None)

    @staticmethod
    def accepted(flags: list[float]) -> int:
        """The accepted prefix length of one step's flags."""
        n = 0
        for f in flags:
            if f <= 0.5:
                break
            n += 1
        return n
