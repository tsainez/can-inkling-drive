"""Cross-model comparison, robustness, ranking stability, efficiency frontier.

These are the four things a reviewer will attack, so each test pins a value that
can be checked by hand rather than asserting the code agrees with itself.
"""

from __future__ import annotations

import math

import pytest

from helpers import flags, rows

from idq.analyze import (
    chance_accuracy,
    efficiency_frontier,
    kendall_tau_b,
    pairwise_comparisons,
    pareto_frontier,
    provenance,
    ranking_stability,
    robustness,
    summarize_run,
)


# --------------------------------------------------------- pairwise / Holm

def test_primary_comparison_is_exempt_from_holm_family():
    """Three models is three pairs. Naming one primary leaves a family of two,
    so the surviving pairs are multiplied by at most 2, not 3."""
    data = (
        rows(flags(80, 100), model="a")
        + rows(flags(60, 100), model="b")
        + rows(flags(58, 100), model="c")
    )
    res = pairwise_comparisons(data, condition="clean", primary=("a", "b"))

    assert res["family_size"] == 2
    assert res["primary"] == "a vs b"

    by_pair = {c["pair"]: c for c in res["comparisons"]}
    primary = by_pair["a vs b"]
    assert primary["is_primary"] is True
    assert primary["p_adjusted"] is None
    assert primary["significant"] is (primary["p_raw"] < 0.05)

    other = by_pair["b vs c"]
    assert other["p_adjusted"] >= other["p_raw"]
    assert other["p_adjusted"] <= min(1.0, 2 * other["p_raw"]) + 1e-12


def test_primary_may_be_given_in_either_order():
    data = rows(flags(80, 100), model="a") + rows(flags(50, 100), model="b")
    res = pairwise_comparisons(data, condition="clean", primary=("b", "a"))
    assert res["primary"] == "a vs b"


def test_missing_primary_is_a_warning_not_a_silent_full_correction():
    """Correcting every pair when the preregistered one is absent is the right
    behaviour, but doing it silently would let the paper claim a primary it
    never had."""
    data = rows(flags(80, 100), model="a") + rows(flags(50, 100), model="b")
    res = pairwise_comparisons(data, condition="clean", primary=("a", "nonexistent"))

    assert res["primary"] is None
    assert res["family_size"] == 1
    assert any("not present" in w for w in res["warnings"])


def test_comparisons_pair_on_question_id_not_position():
    """Model b answered a disjoint question set; nothing is comparable."""
    data = rows(flags(8, 10), model="a") + rows(flags(2, 10), model="b", offset=100)
    res = pairwise_comparisons(data, condition="clean")

    assert res["comparisons"] == []
    assert any("no questions answered by both" in w for w in res["warnings"])


def test_comparisons_only_use_the_named_condition():
    data = (
        rows(flags(90, 100), model="a", condition="clean")
        + rows(flags(30, 100), model="a", condition="blind_tags")
        + rows(flags(50, 100), model="b", condition="clean")
    )
    res = pairwise_comparisons(data, condition="clean")
    assert res["comparisons"][0]["accuracy_a"] == pytest.approx(0.90)


# -------------------------------------------------------------- robustness

def test_robustness_reports_above_chance_retention_not_just_a_delta():
    """0.90 -> 0.50 against chance 0.25 retains (0.50-0.25)/(0.90-0.25) = 0.3846.

    The same -0.40 delta from a lower baseline would retain far less, which is
    exactly why the raw delta is not sufficient.
    """
    data = (
        rows(flags(90, 100), condition="clean")
        + rows(flags(50, 100), condition="corrupt", corruption="motion_blur",
               corruption_severity=3)
    )
    res = robustness(data, model="m", n_boot=800)

    assert res.n_shared == 100
    assert res.delta == pytest.approx(0.40)
    assert res.chance == pytest.approx(0.25)
    assert res.above_chance_retention == pytest.approx(0.25 / 0.65)
    assert res.ci_delta[0] <= res.delta <= res.ci_delta[1]
    assert res.corruption == "motion_blur"
    assert res.corruption_severity == 3


def test_retention_is_undefined_when_the_baseline_is_at_chance():
    """No signal to retain means the ratio is noise over noise; nan, not a number."""
    data = rows(flags(25, 100), condition="clean") + rows(flags(20, 100), condition="corrupt")
    res = robustness(data, model="m", n_boot=400)
    assert math.isnan(res.above_chance_retention)


def test_robustness_surfaces_a_shift_into_invalid_output():
    """Corruption can make a model stop answering rather than answer wrongly.
    Accuracy alone cannot tell those apart."""
    clean = rows(flags(60, 100), condition="clean")
    corrupt = rows(
        flags(40, 100), condition="corrupt",
        valid=[True] * 60 + [False] * 40,
    )
    res = robustness(clean + corrupt, model="m", n_boot=400)

    assert res.invalid_rate_baseline == pytest.approx(0.0)
    assert res.invalid_rate_degraded == pytest.approx(0.40)
    assert res.invalid_rate_delta == pytest.approx(0.40)


def test_robustness_with_no_overlap_reports_nan_rather_than_zero():
    data = rows(flags(5, 10), condition="clean") + rows(
        flags(1, 10), condition="corrupt", offset=50
    )
    res = robustness(data, model="m")
    assert res.n_shared == 0
    assert math.isnan(res.delta)


# ------------------------------------------------------- ranking stability

def test_kendall_tau_b_matches_hand_computed_values():
    assert kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # One adjacent swap in four items: 5 concordant, 1 discordant of 6 pairs.
    assert kendall_tau_b([1, 2, 3, 4], [2, 1, 3, 4]) == pytest.approx(4 / 6)


def test_kendall_tau_b_corrects_for_ties():
    """A pair tied in one variable leaves the denominator, so a full tie in x
    against distinct y is undefined rather than zero."""
    assert math.isnan(kendall_tau_b([1, 1, 1], [1, 2, 3]))
    assert kendall_tau_b([1, 1, 2], [1, 2, 3]) == pytest.approx(2 / math.sqrt(2 * 3))


def test_an_unsupported_order_flip_does_not_count_as_a_ranking_change():
    """Two models one point apart swap places. Neither ordering is significant,
    so the ranking has not been shown to change."""
    data = (
        rows(flags(50, 100), model="a", condition="clean")
        + rows(flags(49, 100), model="b", condition="clean")
        + rows(flags(40, 100), model="a", condition="corrupt")
        + rows(flags(41, 100), model="b", condition="corrupt")
    )
    rs = ranking_stability(data)

    assert len(rs["inversions"]) == 1
    assert rs["inversions"][0]["supported"] is False
    assert rs["n_supported_inversions"] == 0
    assert rs["ranking_preserved"] is True


def test_a_decisive_reversal_is_a_supported_inversion():
    """a beats b on every question when clean; b beats a on every question when
    corrupted. Both orderings are individually significant."""
    n = 60
    a_clean = [True] * n
    b_clean = [False] * n
    data = (
        rows(a_clean, model="a", condition="clean")
        + rows(b_clean, model="b", condition="clean")
        + rows(b_clean, model="a", condition="corrupt")
        + rows(a_clean, model="b", condition="corrupt")
    )
    rs = ranking_stability(data)

    assert rs["n_supported_inversions"] == 1
    assert rs["ranking_preserved"] is False
    assert rs["kendall_tau_b"] == pytest.approx(-1.0)
    assert rs["rank_baseline"] == ["a", "b"]
    assert rs["rank_degraded"] == ["b", "a"]


def test_ranking_uses_questions_all_models_answered():
    """b skipped the questions a got wrong. Ranking b above a on its easier
    subset would be an artifact of coverage, not accuracy."""
    data = (
        rows(flags(50, 100), model="a", condition="clean")
        + rows([True] * 50, model="b", condition="clean")  # only q0..q49
        + rows(flags(50, 100), model="a", condition="corrupt")
        + rows([True] * 50, model="b", condition="corrupt")
    )
    rs = ranking_stability(data)
    # On the shared 50 questions a is also perfect, so the two tie and there is
    # no inversion to report.
    assert rs["n_shared_baseline"] == 50
    assert rs["accuracy_baseline"]["a"] == pytest.approx(1.0)
    assert rs["inversions"] == []


def test_ranking_stability_needs_two_models():
    rs = ranking_stability(rows(flags(5, 10)) + rows(flags(3, 10), condition="corrupt"))
    assert rs["ranking_preserved"] is None
    assert "fewer than two models" in rs["note"]


# --------------------------------------------------------------- efficiency

def test_pareto_frontier_keeps_only_non_dominated_points():
    points = [
        ("cheap_bad", 100.0, 0.50),
        ("mid", 500.0, 0.70),
        ("expensive_worse", 900.0, 0.65),   # dominated by mid
        ("expensive_best", 1200.0, 0.80),
    ]
    assert pareto_frontier(points) == ["cheap_bad", "expensive_best", "mid"]


def test_pareto_frontier_keeps_both_of_two_identical_points():
    """Neither is strictly better, so eliminating one would be arbitrary."""
    assert pareto_frontier([("a", 100.0, 0.6), ("b", 100.0, 0.6)]) == ["a", "b"]


def test_pareto_frontier_excludes_points_with_a_missing_axis():
    """A model with no token count cannot be placed; treating nan as 0 would put
    it on the frontier for free."""
    pts = [("known", 500.0, 0.60), ("unmeasured", float("nan"), 0.90)]
    assert pareto_frontier(pts) == ["known"]


def test_efficiency_falls_back_to_a_shared_axis_when_metrics_differ():
    """One model reporting reasoning tokens and another not must not be plotted
    on two different quantities."""
    data = (
        rows(flags(70, 100), model="reports", tokens=300, reasoning=True)
        + rows(flags(60, 100), model="silent", tokens=200, reasoning=False)
    )
    eff = efficiency_frontier(data, condition="clean", n_boot=400)

    assert eff["metric_is_uniform"] is False
    assert eff["token_metric"] == "completion_tokens"
    assert eff["is_proxy"] is True
    assert eff["fallback_reason"]
    by_model = {p["model"]: p for p in eff["points"]}
    # completion_tokens for both, not 290 vs 200.
    assert by_model["reports"]["mean_tokens"] == pytest.approx(300.0)
    assert by_model["silent"]["mean_tokens"] == pytest.approx(200.0)


def test_efficiency_uses_reasoning_tokens_when_every_model_reports_them():
    data = (
        rows(flags(70, 100), model="a", tokens=300)
        + rows(flags(60, 100), model="b", tokens=200)
    )
    eff = efficiency_frontier(data, condition="clean", n_boot=400)

    assert eff["metric_is_uniform"] is True
    assert eff["token_metric"] == "reasoning_tokens"
    assert eff["is_proxy"] is False
    # a is more accurate but thinks more; both are on the frontier.
    assert eff["pareto_frontier"] == ["a", "b"]


# --------------------------------------------------------------- provenance

def test_provenance_flags_a_provider_that_swapped_builds():
    data = (
        rows(flags(5, 10), served_model="inferact/inkling-nvfp4", quantization="nvfp4")
        + rows(flags(5, 10), offset=10, served_model="inferact/inkling-bf16",
               quantization="bf16")
    )
    prov = provenance(data)["m"]
    assert "served_model" in prov["inconsistent_fields"]
    assert "quantization" in prov["inconsistent_fields"]


def test_provenance_does_not_flag_prompt_key_varying_by_condition():
    """The blind template legitimately differs from the image template."""
    data = (
        rows(flags(5, 10), condition="clean", prompt_key="mcq@v1")
        + rows(flags(5, 10), condition="blind_tags", prompt_key="mcq_blind@v1")
    )
    assert provenance(data)["m"]["inconsistent_fields"] == []


def test_chance_accuracy_counts_each_question_once():
    """A question collected under several seeds must not vote twice."""
    two = rows(flags(1, 2), n_options=2)
    four = rows(flags(1, 2), n_options=4, offset=2)
    assert chance_accuracy(two + four) == pytest.approx(0.375)
    assert chance_accuracy(two + two + four) == pytest.approx(0.375)


# ------------------------------------------------------------ full report

def test_summarize_run_emits_every_research_question_section():
    data = []
    for model, acc in (("a", 80), ("b", 60)):
        data += rows(flags(acc, 100), model=model, condition="clean")
        data += rows(flags(acc - 20, 100), model=model, condition="blind_tags")
        data += rows(flags(acc - 30, 100), model=model, condition="blind_notags")
        data += rows(flags(acc - 15, 100), model=model, condition="corrupt",
                     corruption="fog", corruption_severity=3)

    sections = summarize_run(
        data, primary_comparison=("a", "b"), n_boot=300
    ).sections

    assert set(sections["comparisons"]) == {"clean", "blind_tags", "blind_notags", "corrupt"}
    assert sections["comparisons"]["clean"]["primary"] == "a vs b"
    assert set(sections["grounding"]) == {"a", "b"}
    assert set(sections["robustness"]) == {"a", "b"}
    assert sections["ranking_stability"]["ranking_preserved"] is True
    assert sections["efficiency"]["clean"]["token_metric"] == "reasoning_tokens"
    assert set(sections["provenance"]) == {"a", "b"}


def test_summarize_run_omits_sections_the_data_cannot_support():
    """A single model with no corrupt condition must not produce an empty
    comparison table for someone to quote."""
    sections = summarize_run(rows(flags(50, 100)), n_boot=200).sections
    assert "comparisons" not in sections
    assert "robustness" not in sections
    assert "ranking_stability" not in sections


def test_report_warns_about_an_unstamped_price():
    data = rows(flags(50, 100), usd_per_1m_input=1.0, usd_per_1m_output=2.0,
                price_quoted_on="")
    sections = summarize_run(data, n_boot=200).sections
    assert sections["by_model_condition"]["m|clean"]["cost"]["price_unstamped"] is True
    assert any("quote date" in w for w in sections["warnings"])


def test_report_warns_when_no_primary_comparison_was_designated():
    data = (
        rows(flags(80, 100), model="a")
        + rows(flags(60, 100), model="b")
        + rows(flags(40, 100), model="c")
    )
    sections = summarize_run(data, n_boot=200).sections
    assert any("no primary comparison designated" in w for w in sections["warnings"])


def test_pooling_two_severities_into_one_number_is_flagged():
    """Severity is part of the cache key, so fog s2 and fog s5 coexist in one
    cache file. Averaging them would report a corruption strength that was never
    run."""
    data = (
        rows(flags(80, 100), condition="clean")
        + rows(flags(60, 50), condition="corrupt", corruption="fog", corruption_severity=2)
        + rows(flags(30, 50), condition="corrupt", corruption="fog",
               corruption_severity=5, offset=50)
    )
    res = robustness(data, model="m", n_boot=300)
    assert "severities=[2, 5]" in res.pooling_warning

    sections = summarize_run(data, n_boot=300).sections
    assert any("pools more than one degradation" in w for w in sections["warnings"])


def test_a_single_corruption_at_one_severity_is_not_flagged():
    data = (
        rows(flags(80, 100), condition="clean")
        + rows(flags(50, 100), condition="corrupt", corruption="fog", corruption_severity=3)
    )
    assert robustness(data, model="m", n_boot=300).pooling_warning == ""
