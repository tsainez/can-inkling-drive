"""Paper-ready Markdown tables rendered from a summarize_run report.

The JSON report is the machine-readable artefact; this is what goes into the
paper. Keeping the two in one place matters more than it looks: a number typed
by hand into a table is a number nobody can trace back to a cache record, and
every table here is generated from the same report the reproducibility appendix
points at.

Nothing is rounded away silently. A missing measurement renders as an em dash
rather than as zero, because "we did not measure this" and "this was zero" are
different claims and the second one is a stronger statement than the data
supports.
"""

from __future__ import annotations

import math
from typing import Sequence

DASH = "—"


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def pct(x, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}" if _is_num(x) else DASH


def num(x, digits: int = 0) -> str:
    return f"{x:,.{digits}f}" if _is_num(x) else DASH


def usd(x, digits: int = 4) -> str:
    return f"${x:,.{digits}f}" if _is_num(x) else DASH


def ci(low, high, digits: int = 1) -> str:
    if not (_is_num(low) and _is_num(high)):
        return DASH
    return f"[{100 * low:.{digits}f}, {100 * high:.{digits}f}]"


def pval(p) -> str:
    if p is None:
        return DASH
    if not _is_num(p):
        return DASH
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Markdown pipe table. Alignment is left for the first column, right for
    the rest, which is how numeric columns want to read."""
    if not rows:
        return "_No data for this table._"
    align = ["---"] + ["---:"] * (len(headers) - 1)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _split_key(key: str) -> tuple[str, str]:
    model, _, condition = key.partition("|")
    return model, condition


# ------------------------------------------------------------------ tables

def main_results(sections: dict) -> str:
    """Accuracy, chance, invalid rate and tokens per model and condition."""
    rows = []
    for key, s in sorted(sections.get("by_model_condition", {}).items()):
        model, condition = _split_key(key)
        acc, tok = s["accuracy"], s["tokens"]
        rows.append([
            model, condition, num(acc["n"]),
            pct(acc["accuracy"]), ci(acc["ci_low"], acc["ci_high"]),
            pct(s.get("chance")), pct(acc["invalid_rate"]),
            num(tok["mean_tokens"]),
            tok["token_metric"] + (" (proxy)" if tok["is_proxy"] else ""),
        ])
    return table(
        ["Model", "Condition", "n", "Acc %", "95% CI", "Chance %", "Invalid %",
         "Mean tokens", "Token metric"],
        rows,
    )


def grounding(sections: dict) -> str:
    """RQ3: the three-way split of the grounding gap."""
    rows = []
    for model, g in sorted(sections.get("grounding", {}).items()):
        ci_total = g.get("ci_total") or ()
        rows.append([
            model, num(g["n_shared"]),
            pct(g["clean"]), pct(g["blind_tags"]), pct(g["blind_notags"]),
            pct(g["total_gap"]),
            ci(ci_total[0], ci_total[1]) if len(ci_total) == 2 else DASH,
            pct(g["image_contribution"]), pct(g["tag_leakage"]),
        ])
    return table(
        ["Model", "n", "clean %", "blind_tags %", "blind_notags %",
         "Total gap pp", "95% CI", "Image pp", "Tag leakage pp"],
        rows,
    )


def comparisons(sections: dict) -> str:
    """RQ1: pairwise McNemar, Holm-corrected except the primary comparison."""
    blocks = []
    for condition, comp in sorted(sections.get("comparisons", {}).items()):
        rows = []
        for c in comp["comparisons"]:
            rows.append([
                c["pair"] + (" **(primary)**" if c["is_primary"] else ""),
                num(c["n_paired"]),
                pct(c["accuracy_a"]), pct(c["accuracy_b"]), pct(c["delta"]),
                f"{c['b']}/{c['c']}",
                pval(c["p_raw"]),
                "exempt" if c["is_primary"] else pval(c["p_adjusted"]),
                "yes" if c["significant"] else "no",
            ])
        head = (
            f"**Condition `{condition}`** — {comp['family_size']} comparisons in the "
            f"Holm family, primary: {comp['primary'] or 'none designated'}, "
            f"α = {comp['alpha']}"
        )
        blocks.append(head + "\n\n" + table(
            ["Pair", "n", "Acc A %", "Acc B %", "Δ pp", "b/c", "p", "p (Holm)", "Sig."],
            rows,
        ))
    return "\n\n".join(blocks) if blocks else "_Only one model collected; no pairwise tests._"


def efficiency(sections: dict) -> str:
    """RQ2: the accuracy-versus-thinking-tokens frontier."""
    blocks = []
    for condition, eff in sorted(sections.get("efficiency", {}).items()):
        rows = []
        for p in eff["points"]:
            rows.append([
                p["model"] + (" ★" if p["model"] in eff["pareto_frontier"] else ""),
                num(p["n"]), pct(p["accuracy"]), ci(p["ci_low"], p["ci_high"]),
                num(p["mean_tokens"]), num(p["median_tokens"]),
                pct(p["reasoning_reported_frac"]),
                usd(p["usd_per_correct"]),
            ])
        head = (
            f"**Condition `{condition}`** — axis: `{eff['token_metric']}`"
            + (" (proxy)" if eff["is_proxy"] else "")
            + ("; ★ = on the Pareto frontier" if eff["pareto_frontier"] else "")
        )
        if eff["fallback_reason"]:
            head += f"\n\n> {eff['fallback_reason']}"
        blocks.append(head + "\n\n" + table(
            ["Model", "n", "Acc %", "95% CI", "Mean tok", "Median tok",
             "Reasoning reported %", "USD/correct"],
            rows,
        ))
    return "\n\n".join(blocks) if blocks else "_No efficiency data._"


def robustness(sections: dict) -> str:
    """RQ4, part one: per-model accuracy loss under degradation."""
    rows = []
    for model, r in sorted(sections.get("robustness", {}).items()):
        ci_d = r.get("ci_delta") or ()
        rows.append([
            model, f"{r['corruption'] or DASH} s{r['corruption_severity']}",
            num(r["n_shared"]),
            pct(r["accuracy_baseline"]), pct(r["accuracy_degraded"]),
            pct(r["delta"]),
            ci(ci_d[0], ci_d[1]) if len(ci_d) == 2 else DASH,
            pval(r["p_value"]),
            pct(r["above_chance_retention"]),
            pct(r["invalid_rate_delta"]),
        ])
    return table(
        ["Model", "Corruption", "n", "Baseline %", "Degraded %", "Δ pp", "95% CI",
         "p", "Above-chance retained %", "Δ invalid pp"],
        rows,
    )


def ranking_stability(sections: dict) -> str:
    """RQ4, part two: whether the ordering of models survives degradation."""
    rs = sections.get("ranking_stability")
    if not rs:
        return "_Both a baseline and a degraded condition with two or more models are "\
               "needed for a ranking comparison._"

    lines = [
        f"- Baseline (`{rs['baseline_condition']}`) order: "
        + " > ".join(rs.get("rank_baseline") or []),
        f"- Degraded (`{rs['degraded_condition']}`) order: "
        + " > ".join(rs.get("rank_degraded") or []),
        f"- Kendall τ-b: {rs['kendall_tau_b']:.3f}" if _is_num(rs["kendall_tau_b"])
        else f"- Kendall τ-b: {DASH}",
        f"- Supported inversions: {rs['n_supported_inversions']} of "
        f"{len(rs['inversions'])} order flips",
        f"- Ranking preserved: **{rs['ranking_preserved']}**",
    ]
    body = "\n".join(lines)

    if rs["inversions"]:
        rows = [[
            inv["pair"], inv["baseline_leader"], inv["degraded_leader"],
            pct(inv["baseline_delta"]), pct(inv["degraded_delta"]),
            pval(inv["p_baseline"]), pval(inv["p_degraded"]),
            "yes" if inv["supported"] else "no",
        ] for inv in rs["inversions"]]
        body += "\n\n" + table(
            ["Pair", "Baseline leader", "Degraded leader", "Δ baseline pp",
             "Δ degraded pp", "p baseline", "p degraded", "Supported"],
            rows,
        )
    return body + f"\n\n> {rs['note']}"


def cost_appendix(sections: dict) -> str:
    """Appendix: dollars, stamped with provider and quote date."""
    rows = []
    for key, s in sorted(sections.get("by_model_condition", {}).items()):
        model, condition = _split_key(key)
        c = s["cost"]
        if not c.get("priced"):
            rows.append([model, condition] + [DASH] * 7)
            continue
        rows.append([
            model, condition,
            "/".join(c.get("served_by") or []) or DASH,
            usd(c.get("usd_per_1m_input"), 2), usd(c.get("usd_per_1m_output"), 2),
            "/".join(c.get("price_quoted_on") or []) or "**unstamped**",
            num(c["input_tokens"]), num(c["output_tokens"]),
            usd(c["total_usd"], 2), usd(c["usd_per_correct"]),
        ])
    return table(
        ["Model", "Condition", "Provider", "$/1M in", "$/1M out", "Quoted on",
         "Input tok", "Output tok", "Total", "USD/correct"],
        rows,
    ) + (
        "\n\n> Prices are a dated snapshot of one provider, not a property of the "
        "model. Open-weights hosting prices move with promotions; readers should not "
        "expect these figures to hold."
    )


def provenance(sections: dict) -> str:
    """Serving configuration, which §4 of the methods requires reporting."""
    def fmt(v):
        if isinstance(v, list):
            # More than one distinct value is the finding, so show all of them.
            return "/".join(v) if v else DASH
        return str(v) if v not in (None, "") else DASH

    rows = []
    for model, p in sorted(sections.get("provenance", {}).items()):
        rows.append([
            model, num(p["n_rows"]), fmt(p.get("model_string")),
            fmt(p.get("served_model")), fmt(p.get("quantization")),
            fmt(p.get("thinking_effort")), fmt(p.get("temperature")),
            fmt(p.get("max_tokens")), fmt(p.get("git_sha")),
            ", ".join(p["inconsistent_fields"]) or "none",
        ])
    return table(
        ["Model", "n", "Model string", "Served as", "Quant.", "Thinking effort",
         "Temp.", "Max tok", "Harness commit", "Inconsistent"],
        rows,
    )


# ------------------------------------------------------------------ render

SECTIONS = (
    ("Main results", main_results),
    ("RQ1 — pairwise model comparisons", comparisons),
    ("RQ2 — accuracy per thinking token", efficiency),
    ("RQ3 — grounding decomposition", grounding),
    ("RQ4 — robustness under degradation", robustness),
    ("RQ4 — ranking stability", ranking_stability),
    ("Appendix A — serving configuration", provenance),
    ("Appendix B — cost", cost_appendix),
)


def render_markdown(sections: dict, *, title: str = "Results") -> str:
    """Assemble every table into one document, warnings first.

    Warnings lead rather than trail. Anything that makes a number unquotable —
    a proxy token metric, an unstamped price, a provider that swapped builds —
    should be read before the table it applies to, not discovered afterwards.
    """
    parts = [f"# {title}", "", f"Scored rows: {sections.get('n_rows', 0)}", ""]

    warnings = sections.get("warnings") or []
    if warnings:
        parts += ["## Read before quoting any number", ""]
        parts += [f"- {w}" for w in warnings]
        parts += [""]

    for heading, fn in SECTIONS:
        parts += [f"## {heading}", "", fn(sections), ""]

    return "\n".join(parts).rstrip() + "\n"
