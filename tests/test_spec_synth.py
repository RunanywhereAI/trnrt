"""Synthetic acceptance (trnrt/engine/spec_synth.py, `EngineConfig.spec_synth_al` / `TRNRT_SPEC_SYNTH_AL`): the schedule that InferenceX's golden
acceptance length is forced with.

  1. the schedule equals vLLM's, formula for formula (the two reference functions below are vLLM's, from the pull request that added it:
     vllm-project/vllm pull 40662; see THIRD_PARTY_NOTICES.md), and the pull request's own logged case;
  2. the REALIZED accepted-count distribution is the schedule's, not only its mean: the minimum-variance schedule takes exactly two values, so a schedule with
     the right mean and the wrong shape must fail (a plant).

The engine-level tests (spec_post takes the host's flags; CPU engines realize the schedule) are not written yet; tests/test_spec_metrics.py covers
the synthetic acceptance's counters.

    python3 -m pytest tests/test_spec_synth.py -q"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from trnrt.engine import spec_synth as SS

# -- vLLM's reference (vllm/config/speculative.py, vllm/v1/spec_decode/utils.py, pull request 40662), copied -------------------------------------------------------


def vllm_rates(length: float, n: int) -> list[float]:
    num_drafts = length - 1
    num_full = int(num_drafts)
    return ([1.0] * num_full + [num_drafts - num_full] + [0.0] * (n - num_full - 1))[:n]


def vllm_conditional(rates: list[float]) -> list[float]:
    return [p / q if q > 0.0 else 0.0 for p, q in zip(rates, [1.0, *rates[:-1]])]


# InferenceX's glm-5.3-nvfp4__mtp__native curve (golden-AL table of the benchmark)
GOLDEN = {1: 1.84, 2: 2.49, 3: 2.94, 4: 3.28}


def test_the_schedule_equals_vllms_for_every_length_and_draft_count():
    for n in range(1, 9):
        for al in [*np.linspace(1.0, n + 1.0, 41).tolist(), 1.0, n + 1.0, 1.84, 2.6]:
            if al > n + 1:
                continue
            r = SS.min_variance_rates(al, n)
            assert r == vllm_rates(al, n), (al, n)
            assert SS.to_conditional(r) == vllm_conditional(r), (al, n)
            assert len(r) == n and all(0.0 <= x <= 1.0 for x in r) and all(a >= b for a, b in itertools.pairwise(r))
            assert abs(1.0 + sum(r) - al) < 1e-9  # the mean acceptance length is 1 + the sum of the unconditional rates


def test_the_golden_curve_and_the_prs_logged_case():
    assert SS.min_variance_rates(1.84, 1) == pytest.approx([0.84])
    assert SS.min_variance_rates(2.94, 3) == pytest.approx([1.0, 0.94, 0.0])
    assert SS.min_variance_rates(3.28, 4) == pytest.approx([1.0, 1.0, 0.28, 0.0])
    assert SS.to_conditional(SS.min_variance_rates(3.28, 4)) == pytest.approx([1.0, 1.0, 0.28, 0.0])
    # the PR's log: synthetic_acceptance_length 2.6 with 3 speculative tokens measured per-position rates 1.000, 0.599, 0.000
    assert SS.min_variance_rates(2.6, 3) == pytest.approx([1.0, 0.6, 0.0])


@pytest.mark.parametrize(("al", "n"), [(0.99, 3), (4.01, 3), (2.0, 0), (-1.0, 1)])
def test_a_length_outside_1_to_k_plus_1_is_refused(al, n):
    with pytest.raises(ValueError, match="synthetic acceptance length"):
        SS.min_variance_rates(al, n)


def test_streams_are_reproducible_and_differ_by_request_and_from_the_sampling_noise():
    a, b = SS.SynthAccept(2.5, 3, seed=7), SS.SynthAccept(2.5, 3, seed=7)
    s1 = [a.flags("r1") for _ in range(50)]
    assert s1 == [b.flags("r1") for _ in range(50)]
    assert s1 != [b.flags("r2") for _ in range(50)]
    # the request's own seed decides when given, whatever the request id
    c, d = SS.SynthAccept(2.5, 3, seed=1), SS.SynthAccept(2.5, 3, seed=2)
    assert [c.flags("x", 99) for _ in range(30)] == [d.flags("y", 99) for _ in range(30)]
    # not the sampling noise stream of the same seed (the salt)
    u = np.random.default_rng(99).random(3 * 30).reshape(30, 3)
    f = SS.SynthAccept(2.5, 3).flags
    assert [f("z", 99) for _ in range(30)] != [
        [float(x < cc) for x, cc in zip(row, SS.to_conditional(SS.min_variance_rates(2.5, 3)))] for row in u
    ]
    a.forget("r1")
    assert [a.flags("r1") for _ in range(50)] == s1  # forgotten, it restarts


# -- the realized accepted-count distribution ----------------------------------------------------------------------------------------------------------------------


def realized(sx: SS.SynthAccept, n_steps: int, rid="q") -> np.ndarray:
    """Accepted-draft counts of n_steps verify steps (counts[m] = steps that accepted m drafts)."""
    counts = np.zeros(sx.k + 1, np.int64)
    for _ in range(n_steps):
        counts[SS.SynthAccept.accepted(sx.flags(rid))] += 1
    return counts


def check_distribution(counts: np.ndarray, al: float, k: int, sigmas: float = 5.0) -> None:
    """The min-variance schedule takes exactly two values, floor(AL - 1) with probability 1 - frac and floor(AL - 1) + 1 with probability frac: nothing else may occur, each
    frequency sits within `sigmas` binomial deviations, and the mean AL sits within the mean's own."""
    n = int(counts.sum())
    drafts = al - 1.0
    lo = int(drafts)
    frac = drafts - lo
    want = np.zeros(k + 1)
    want[lo] = 1.0 - frac
    if frac > 0:
        want[lo + 1] = frac
    off_support = counts[want == 0].sum()
    assert off_support == 0, f"accepted counts outside the schedule's two values: {counts.tolist()}"
    for m in range(k + 1):
        if want[m] > 0:
            p = want[m]
            assert abs(counts[m] / n - p) <= sigmas * math.sqrt(p * (1 - p) / n) + 1e-12, (m, counts.tolist(), p)
    mean_al = 1.0 + float((counts * np.arange(k + 1)).sum()) / n
    sd = math.sqrt(frac * (1 - frac) / n)
    assert abs(mean_al - al) <= sigmas * sd + 1e-12, (mean_al, al)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_the_realized_accepted_count_distribution_is_the_golden_schedules(k):
    al = GOLDEN[k]
    check_distribution(realized(SS.SynthAccept(al, k, seed=3), 40000), al, k)


@pytest.mark.parametrize(
    ("al", "k"), [(1.0, 3), (4.0, 3), (2.0, 3), (2.6, 3), (2.518506, 2)]
)  # the last: AgentX's golden length
def test_the_edges_of_the_schedule_hold_too(al, k):
    check_distribution(realized(SS.SynthAccept(al, k, seed=5), 20000), al, k)


def test_PLANT_the_checker_sees_a_schedule_with_the_right_mean_and_the_wrong_shape():
    # k = 3, AL 2.94: the same mean from the same rate at every position (0.64 each, unconditional-as-conditional): the accepted count takes 0..3, the min-variance schedule only 1 and 2
    k, al = 3, 2.94
    sx = SS.SynthAccept(al, k, seed=3)
    sx.cond = [(al - 1.0) / k] * k
    counts = realized(sx, 40000)
    assert (
        1.0 + (counts * np.arange(k + 1)).sum() / counts.sum() < al
    )  # (even the mean misses: the conditional chain shortens it)
    with pytest.raises(AssertionError):
        check_distribution(counts, al, k)
    # and the geometric schedule vLLM used before this PR, calibrated to the same mean, is a different shape too
    sx2 = SS.SynthAccept(al, k, seed=3)
    sx2.cond = [0.9, 0.9, 0.6]
    with pytest.raises(AssertionError):
        check_distribution(realized(sx2, 40000), al, k)
