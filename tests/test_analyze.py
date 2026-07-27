"""Statistics tests against hand-computable values."""

from __future__ import annotations

import math

import pytest

from idq.analyze import (
    accuracy,
    bootstrap_ci,
    grounding_decomposition,
    holm_correct,
    mcnemar_exact,
    token_profile,
)


def rows(correct_flags, *, condition="clean", model="m", valid=None):
    valid = valid if valid is not None else [True] * len(correct_flags)
    return [
        {"question_id": f"q{i}", "correct": c, "valid": v, "category": "perception",
         "condition": condition, "model_label": model}
        for i, (c, v) in enumerate(zip(correct_flags, valid))
    ]


def test_mcnemar_matches_hand_computed_value():
    """b=1, c=5 discordant. Exact two-sided p = 2 * P(X<=1 | n=6, p=0.5)
    = 2 * (1 + 6)/64 = 7/32 = 0.21875."""
    a = [True] + [False] * 5 + [True] * 10
    b = [False] + [True] * 5 + [True] * 10
    res = mcnemar_exact(a, b)
    assert (res.b, res.c) == (1, 5)
    assert res.p_value == pytest.approx(7 / 32)


def test_mcnemar_no_discordant_pairs_is_p_one():
    res = mcnemar_exact([True, False], [True, False])
    assert (res.b, res.c) == (0, 0)
    assert res.p_value == 1.0


def test_mcnemar_strongly_discordant_is_significant():
    a = [True] * 20 + [False] * 2
    b = [False] * 20 + [True] * 2
    assert mcnemar_exact(a, b).p_value < 0.001


def test_mcnemar_is_symmetric_in_magnitude():
    a = [True] * 8 + [False] * 3
    b = [False] * 8 + [True] * 3
    assert mcnemar_exact(a, b).p_value == pytest.approx(mcnemar_exact(b, a).p_value)


def test_holm_correction_is_monotone_and_bounded():
    adj = holm_correct({"p1": 0.01, "p2": 0.04, "p3": 0.03})
    assert adj["p1"] == pytest.approx(0.03)
    assert all(0.0 <= v <= 1.0 for v in adj.values())
    assert adj["p1"] <= adj["p3"] <= adj["p2"]


def test_holm_never_reduces_a_pvalue():
    raw = {"a": 0.2, "b": 0.3, "c": 0.5}
    adj = holm_correct(raw)
    assert all(adj[k] >= raw[k] for k in raw)


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bootstrap_ci([1.0] * 60 + [0.0] * 40, n_boot=2000, seed=0)
    assert lo < 0.60 < hi
    assert hi - lo < 0.30


def test_bootstrap_ci_is_deterministic_given_seed():
    vals = [1.0, 0.0] * 50
    assert bootstrap_ci(vals, n_boot=500, seed=7) == bootstrap_ci(vals, n_boot=500, seed=7)


def test_invalid_output_counts_as_wrong_and_is_reported():
    r = rows([True, True, False, False], valid=[True, True, True, False])
    res = accuracy(r, n_boot=500)
    assert res.accuracy == pytest.approx(0.5)
    assert res.invalid_rate == pytest.approx(0.25)
    assert res.n_valid == 3


def test_grounding_decomposition_splits_the_gap():
    """clean .90, blind_tags .70, blind_notags .50 ->
    image contribution .20, tag leakage .20, total .40."""
    n = 100
    clean = rows([i < 90 for i in range(n)], condition="clean")
    btags = rows([i < 70 for i in range(n)], condition="blind_tags")
    bnotags = rows([i < 50 for i in range(n)], condition="blind_notags")

    d = grounding_decomposition(clean, btags, bnotags, n_boot=800)
    assert d.total_gap == pytest.approx(0.40)
    assert d.image_contribution == pytest.approx(0.20)
    assert d.tag_leakage == pytest.approx(0.20)
    assert d.n_shared == n
    assert d.ci_total[0] <= d.total_gap <= d.ci_total[1]


def test_grounding_decomposition_requires_shared_questions():
    d = grounding_decomposition(rows([True]), [], [])
    assert d.n_shared == 0
    assert math.isnan(d.total_gap)


def test_token_profile_flags_proxy_when_reasoning_tokens_absent():
    """A missing measurement and a measured zero are different claims."""
    unreported = [{"completion_tokens": 100, "reasoning_tokens": None,
                   "reasoning_tokens_reported": False, "latency_ms": 500.0}] * 10
    prof = token_profile(unreported)
    assert prof["is_proxy"] is True
    assert prof["token_metric"] == "completion_tokens"

    reported = [{"completion_tokens": 100, "reasoning_tokens": 80,
                 "reasoning_tokens_reported": True, "latency_ms": 500.0}] * 10
    prof = token_profile(reported)
    assert prof["is_proxy"] is False
    assert prof["token_metric"] == "reasoning_tokens"
    assert prof["mean_tokens"] == pytest.approx(80.0)
