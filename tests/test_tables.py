"""Markdown rendering. The tables go in the paper, so what they must not do is
turn a missing measurement into a number.
"""

from __future__ import annotations

from helpers import flags, rows

from idq.analyze import summarize_run
from idq.tables import DASH, ci, num, pct, pval, render_markdown, table, usd


def test_missing_values_render_as_a_dash_not_as_zero():
    """'we did not measure this' and 'this was zero' are different claims."""
    for fn in (pct, num, usd):
        assert fn(float("nan")) == DASH
        assert fn(None) == DASH
        assert fn(float("inf")) == DASH
    assert ci(float("nan"), 0.5) == DASH
    assert pval(None) == DASH


def test_zero_still_renders_as_zero():
    assert pct(0.0) == "0.0"
    assert num(0) == "0"
    assert usd(0.0) == "$0.0000"


def test_small_pvalues_are_not_rendered_as_zero():
    """'p = 0.000' invites the reading that p is exactly zero."""
    assert pval(0.0000001) == "<0.001"
    assert pval(0.031) == "0.031"


def test_empty_table_says_so_rather_than_rendering_a_header_only():
    assert "No data" in table(["a", "b"], [])


def test_render_puts_warnings_before_the_tables():
    """A proxy metric or an unstamped price has to be read before the number it
    applies to, not after."""
    data = rows(flags(50, 100), usd_per_1m_input=1.0, usd_per_1m_output=2.0,
                price_quoted_on="", reasoning=False)
    md = render_markdown(summarize_run(data, n_boot=200).sections)

    assert md.index("Read before quoting") < md.index("## Main results")
    assert "proxy" in md
    assert "quote date" in md


def test_every_section_renders_for_a_complete_run():
    data = []
    for model, acc in (("a", 80), ("b", 55)):
        data += rows(flags(acc, 100), model=model, condition="clean",
                     usd_per_1m_input=1.87, usd_per_1m_output=4.68,
                     price_quoted_on="2026-07-24", served_by="baseten",
                     quantization="nvfp4", thinking_effort="medium",
                     git_sha="abc1234")
        data += rows(flags(acc - 25, 100), model=model, condition="blind_tags")
        data += rows(flags(acc - 35, 100), model=model, condition="blind_notags")
        data += rows(flags(acc - 20, 100), model=model, condition="corrupt",
                     corruption="motion_blur", corruption_severity=3)

    md = render_markdown(
        summarize_run(data, primary_comparison=("a", "b"), n_boot=300).sections,
        title="T",
    )

    for heading in (
        "## Main results",
        "## RQ1 — pairwise model comparisons",
        "## RQ2 — accuracy per thinking token",
        "## RQ3 — grounding decomposition",
        "## RQ4 — robustness under degradation",
        "## RQ4 — ranking stability",
        "## Appendix A — serving configuration",
        "## Appendix B — cost",
    ):
        assert heading in md, f"missing section {heading}"

    assert "**(primary)**" in md
    assert "exempt" in md            # primary is not Holm-corrected
    assert "motion_blur s3" in md    # the corruption is named, not implied
    assert "2026-07-24" in md        # cost table is date-stamped
    assert "nvfp4" in md             # serving config is reported
    assert "not a property of the" in md  # the price caveat travels with the table


def test_robustness_table_refuses_to_report_a_degradation_that_never_happened():
    """Regression: two blind conditions were once passed as baseline/degraded.

    No corruption was applied, so the table printed a "degradation" that was
    really a comparison of two undegraded conditions — and, because both sat at
    chance, an above-chance-retained figure of 157%.
    """
    data = (
        rows(flags(315, 600), model="inkling", condition="blind_notags", n_options=2)
        + rows(flags(323, 600), model="inkling", condition="blind_tags", n_options=2)
    )
    md = render_markdown(
        summarize_run(
            data, baseline_condition="blind_notags",
            degraded_condition="blind_tags", n_boot=300,
        ).sections,
        title="T",
    )

    assert "## RQ4 — robustness under degradation" in md
    assert "not evaluable" in md
    assert "Suppressed" in md
    assert "157" not in md


def test_cost_table_marks_an_unstamped_price_in_the_table_itself():
    """Not only in the warnings list: someone will copy just the table."""
    data = rows(flags(50, 100), usd_per_1m_input=1.0, usd_per_1m_output=2.0,
                price_quoted_on="")
    md = render_markdown(summarize_run(data, n_boot=200).sections)
    assert "**unstamped**" in md


def test_sections_the_data_cannot_support_say_why():
    md = render_markdown(summarize_run(rows(flags(50, 100)), n_boot=200).sections)
    assert "no pairwise tests" in md
    assert "ranking comparison" in md


def test_a_field_that_was_never_recorded_is_not_a_blank_cell():
    """An empty provenance cell reads as 'nothing to report'. A model whose
    served_model the provider never disclosed has to be visibly unknown, since
    §4 of the methods forbids implying a quantization we did not observe."""
    md = render_markdown(summarize_run(rows(flags(50, 100)), n_boot=200).sections)
    provenance_row = [
        line for line in md.splitlines() if line.startswith("| m |")
    ][-1]
    assert "|  |" not in provenance_row, provenance_row
    assert DASH in provenance_row
