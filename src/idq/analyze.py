"""Analysis: accuracy with CIs, paired tests, grounding gap, cost per correct.

Statistical choices worth defending in review:

  - Bootstrap percentile CIs over questions, because accuracy on a fixed
    question set is not a simple binomial when questions differ in difficulty
    and option count.
  - Exact McNemar (binomial), not the chi-square approximation, because
    discordant-pair counts on a few hundred questions are often small enough
    that the approximation misbehaves.
  - Holm correction across the family of pairwise comparisons, with one
    preregistered primary comparison exempt. Five models is ten pairs per
    condition; uncorrected, a spurious significant result is close to expected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------- accuracy

@dataclass(frozen=True)
class AccuracyResult:
    n: int
    n_valid: int
    accuracy: float
    ci_low: float
    ci_high: float
    invalid_rate: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def accuracy(rows: Sequence[dict], *, n_boot: int = 10000, seed: int = 0) -> AccuracyResult:
    """Invalid output counts as incorrect, and is also reported separately.

    Scoring an unparseable answer as wrong is the conservative choice; hiding
    how often it happens would not be.
    """
    if not rows:
        return AccuracyResult(0, 0, float("nan"), float("nan"), float("nan"), float("nan"))
    correct = [1.0 if r.get("correct") else 0.0 for r in rows]
    n_valid = sum(1 for r in rows if r.get("valid"))
    lo, hi = bootstrap_ci(correct, n_boot=n_boot, seed=seed)
    return AccuracyResult(
        n=len(rows),
        n_valid=n_valid,
        accuracy=float(np.mean(correct)),
        ci_low=lo,
        ci_high=hi,
        invalid_rate=1.0 - (n_valid / len(rows)),
    )


# ------------------------------------------------------------ paired tests

@dataclass(frozen=True)
class McNemarResult:
    n_paired: int
    b: int  # a correct, b wrong
    c: int  # a wrong, b correct
    p_value: float
    accuracy_a: float
    accuracy_b: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> McNemarResult:
    if len(a_correct) != len(b_correct):
        raise ValueError("paired test requires equal-length inputs")
    n = len(a_correct)
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)

    m = b + c
    if m == 0:
        p = 1.0
    else:
        k = min(b, c)
        tail = sum(math.comb(m, i) for i in range(0, k + 1)) * (0.5 ** m)
        p = min(1.0, 2.0 * tail)

    return McNemarResult(
        n_paired=n, b=b, c=c, p_value=float(p),
        accuracy_a=float(np.mean([1.0 if x else 0.0 for x in a_correct])) if n else float("nan"),
        accuracy_b=float(np.mean([1.0 if x else 0.0 for x in b_correct])) if n else float("nan"),
    )


def paired_by_question(rows_a: Sequence[dict], rows_b: Sequence[dict]):
    """Align two row sets on question_id. Only questions present in both count."""
    a = {r["question_id"]: bool(r.get("correct")) for r in rows_a}
    b = {r["question_id"]: bool(r.get("correct")) for r in rows_b}
    shared = sorted(set(a) & set(b))
    return [a[q] for q in shared], [b[q] for q in shared], shared


def holm_correct(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni. Returns adjusted p-values, monotone-enforced."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (label, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)
        adjusted[label] = running
    return adjusted


def pair_label(model_a: str, model_b: str) -> str:
    return f"{model_a} vs {model_b}"


def pairwise_comparisons(
    rows: Sequence[dict],
    *,
    condition: str,
    primary: tuple[str, str] | None = None,
    alpha: float = 0.05,
) -> dict:
    """Every pairwise model comparison within one condition.

    The preregistered primary comparison is reported uncorrected and is
    excluded from the Holm family. That is the point of designating it in
    advance: one hypothesis gets full power, and the exploratory remainder pays
    for its own multiplicity. Adding the primary to the family afterwards would
    penalise it for comparisons it was never competing with.

    Comparisons are paired on question_id, so a model missing a question drops
    that question from the pair rather than from the study.
    """
    subset = filter_rows(rows, condition=condition)
    models = sorted({r.get("model_label") for r in subset if r.get("model_label")})
    warnings: list[str] = []
    want_primary = tuple(sorted(primary)) if primary else None

    raw: dict[str, dict] = {}
    for i, a in enumerate(models):
        for b_model in models[i + 1:]:
            xa, xb, shared = paired_by_question(
                filter_rows(subset, model_label=a), filter_rows(subset, model_label=b_model)
            )
            label = pair_label(a, b_model)
            if not shared:
                warnings.append(
                    f"{label}: no questions answered by both models in {condition}; skipped"
                )
                continue
            res = mcnemar_exact(xa, xb)
            raw[label] = {
                "model_a": a,
                "model_b": b_model,
                "n_paired": res.n_paired,
                "b": res.b,
                "c": res.c,
                "accuracy_a": res.accuracy_a,
                "accuracy_b": res.accuracy_b,
                "delta": res.accuracy_a - res.accuracy_b,
                "p_raw": res.p_value,
                "is_primary": want_primary == tuple(sorted((a, b_model))),
            }

    primary_labels = [lbl for lbl, d in raw.items() if d["is_primary"]]
    if want_primary and not primary_labels:
        warnings.append(
            f"primary comparison {want_primary[0]} vs {want_primary[1]} is not present in "
            f"{condition}; every comparison is Holm-corrected and no result here is the "
            "preregistered primary"
        )

    family = {lbl: d["p_raw"] for lbl, d in raw.items() if not d["is_primary"]}
    adjusted = holm_correct(family) if family else {}

    comparisons = []
    for label, d in sorted(raw.items()):
        p_adj = None if d["is_primary"] else adjusted.get(label)
        decisive = d["p_raw"] if d["is_primary"] else p_adj
        comparisons.append({
            **d,
            "pair": label,
            "p_adjusted": p_adj,
            "significant": bool(decisive is not None and decisive < alpha),
        })

    # Ordering by effect size rather than alphabetically: the reader wants the
    # largest accuracy differences first.
    comparisons.sort(key=lambda d: (-abs(d["delta"]), d["pair"]))

    return {
        "condition": condition,
        "alpha": alpha,
        "models": models,
        "primary": primary_labels[0] if primary_labels else None,
        "family_size": len(family),
        "correction": "holm" if family else "none",
        "comparisons": comparisons,
        "warnings": warnings,
    }


# --------------------------------------------------------- grounding gap

@dataclass(frozen=True)
class GroundingDecomposition:
    clean: float
    blind_tags: float
    blind_notags: float
    total_gap: float          # clean - blind_notags
    image_contribution: float # clean - blind_tags
    tag_leakage: float        # blind_tags - blind_notags
    ci_total: tuple = ()
    n_shared: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def grounding_decomposition(
    rows_clean: Sequence[dict],
    rows_blind_tags: Sequence[dict],
    rows_blind_notags: Sequence[dict],
    *,
    n_boot: int = 10000,
    seed: int = 0,
) -> GroundingDecomposition:
    """Split the grounding gap three ways.

    A single blind condition confounds two things: what the image contributes,
    and what DriveLM's object tags leak in text. Running blind twice separates
    them, so a small total gap can be attributed rather than left ambiguous.
    """
    maps = [
        {r["question_id"]: bool(r.get("correct")) for r in rows}
        for rows in (rows_clean, rows_blind_tags, rows_blind_notags)
    ]
    shared = sorted(set(maps[0]) & set(maps[1]) & set(maps[2]))
    if not shared:
        return GroundingDecomposition(*([float("nan")] * 6), ci_total=(), n_shared=0)

    c = np.array([1.0 if maps[0][q] else 0.0 for q in shared])
    bt = np.array([1.0 if maps[1][q] else 0.0 for q in shared])
    bn = np.array([1.0 if maps[2][q] else 0.0 for q in shared])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shared), size=(n_boot, len(shared)))
    diffs = c[idx].mean(axis=1) - bn[idx].mean(axis=1)
    ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))

    return GroundingDecomposition(
        clean=float(c.mean()),
        blind_tags=float(bt.mean()),
        blind_notags=float(bn.mean()),
        total_gap=float(c.mean() - bn.mean()),
        image_contribution=float(c.mean() - bt.mean()),
        tag_leakage=float(bt.mean() - bn.mean()),
        ci_total=ci,
        n_shared=len(shared),
    )


# ------------------------------------------------- robustness (corruption)

def chance_accuracy(rows: Sequence[dict]) -> float:
    """Chance is mean(1/k) over questions, never assumed to be 0.25.

    Option counts vary in DriveLM, so a fixed 0.25 would misstate every
    "above chance" claim. Computed over unique questions, because a question
    collected under several seeds must not get several votes.
    """
    ks = {
        r.get("question_id"): r.get("n_options")
        for r in rows
        if isinstance(r.get("n_options"), int) and r.get("n_options", 0) > 0
    }
    if not ks:
        return float("nan")
    return float(np.mean([1.0 / k for k in ks.values()]))


@dataclass(frozen=True)
class RobustnessResult:
    model: str
    baseline_condition: str
    degraded_condition: str
    corruption: str
    corruption_severity: int
    n_shared: int
    accuracy_baseline: float
    accuracy_degraded: float
    delta: float                    # baseline - degraded, so positive = degraded
    ci_delta: tuple = ()
    p_value: float = float("nan")
    chance: float = float("nan")
    above_chance_retention: float = float("nan")
    # Why retention is undefined, when it is. Empty when the ratio is reportable.
    retention_note: str = ""
    invalid_rate_baseline: float = float("nan")
    invalid_rate_degraded: float = float("nan")
    invalid_rate_delta: float = float("nan")
    # Non-empty when the degraded rows are not a single corruption at a single
    # severity. Severity is part of the cache key, so two severities live
    # happily in one cache file and would otherwise be pooled into one number.
    pooling_warning: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def robustness(
    rows: Sequence[dict],
    *,
    model: str,
    baseline: str = "clean",
    degraded: str = "corrupt",
    n_boot: int = 10000,
    seed: int = 0,
) -> RobustnessResult:
    """Paired accuracy loss under sensor degradation for one model.

    Two things are reported that a raw accuracy delta hides:

    `above_chance_retention` — the fraction of above-chance accuracy that
    survives, (degraded - chance) / (baseline - chance). A drop from 0.90 to
    0.50 and a drop from 0.50 to 0.30 are both "-0.20" in absolute terms, but
    the first keeps 38% of its signal and the second keeps 8%. Undefined when
    baseline accuracy is at or below chance, because there was no signal to
    retain and the ratio would be noise divided by noise.

    `invalid_rate_delta` — corruption can push a model into refusing or
    rambling past its token limit rather than answering wrongly. That is a
    different failure than a wrong answer and would otherwise be invisible in
    an accuracy-only comparison.
    """
    base_rows = filter_rows(rows, model_label=model, condition=baseline)
    deg_rows = filter_rows(rows, model_label=model, condition=degraded)

    corruptions = sorted({r.get("corruption") or "" for r in deg_rows} - {""})
    severities = sorted({r.get("corruption_severity") or 0 for r in deg_rows})
    pooling = ""
    if len(corruptions) > 1 or len(severities) > 1:
        pooling = (
            f"{model}/{degraded} pools more than one degradation "
            f"(corruptions={corruptions or ['none']}, severities={severities}) into a "
            "single accuracy. Filter to one corruption at one severity before reporting."
        )

    xa, xb, shared = paired_by_question(base_rows, deg_rows)
    if not shared:
        return RobustnessResult(
            model=model, baseline_condition=baseline, degraded_condition=degraded,
            corruption="+".join(corruptions),
            corruption_severity=severities[-1] if severities else 0,
            n_shared=0, accuracy_baseline=float("nan"), accuracy_degraded=float("nan"),
            delta=float("nan"), pooling_warning=pooling,
        )

    a = np.array([1.0 if x else 0.0 for x in xa])
    d = np.array([1.0 if x else 0.0 for x in xb])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shared), size=(n_boot, len(shared)))
    diffs = a[idx].mean(axis=1) - d[idx].mean(axis=1)
    ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))

    shared_set = set(shared)
    base_shared = [r for r in base_rows if r["question_id"] in shared_set]
    deg_shared = [r for r in deg_rows if r["question_id"] in shared_set]
    inv_base = 1.0 - sum(1 for r in base_shared if r.get("valid")) / len(base_shared)
    inv_deg = 1.0 - sum(1 for r in deg_shared if r.get("valid")) / len(deg_shared)

    chance = chance_accuracy(base_shared)
    acc_b, acc_d = float(a.mean()), float(d.mean())
    headroom = acc_b - chance

    # Retention divides one above-chance margin by another. When the baseline
    # margin is small the denominator is mostly noise, and the ratio explodes:
    # a baseline of 52.5% against chance 50% has 2.5pp of headroom, so a
    # degraded score of 53.9% reports "157% retained" - a number that says
    # nothing except that both conditions were at chance. Requiring the
    # baseline to be *statistically* above chance, not merely numerically
    # above it, is what makes the ratio mean something.
    base_ci = bootstrap_ci(a, n_boot=n_boot, seed=seed)
    retention_note = ""
    if headroom <= 0:
        retention = float("nan")
        retention_note = (
            f"baseline {acc_b:.3f} is at or below chance {chance:.3f}; "
            "there was no signal to retain"
        )
    elif base_ci[0] <= chance:
        retention = float("nan")
        retention_note = (
            f"baseline {acc_b:.3f} is not distinguishable from chance {chance:.3f} "
            f"(95% CI lower bound {base_ci[0]:.3f}); the retention ratio would be "
            "noise divided by noise"
        )
    else:
        retention = (acc_d - chance) / headroom

    return RobustnessResult(
        model=model,
        baseline_condition=baseline,
        degraded_condition=degraded,
        corruption="+".join(corruptions),
        corruption_severity=severities[-1] if severities else 0,
        n_shared=len(shared),
        accuracy_baseline=acc_b,
        accuracy_degraded=acc_d,
        delta=acc_b - acc_d,
        ci_delta=ci,
        p_value=mcnemar_exact(xa, xb).p_value,
        chance=chance,
        above_chance_retention=retention,
        retention_note=retention_note,
        invalid_rate_baseline=inv_base,
        invalid_rate_degraded=inv_deg,
        invalid_rate_delta=inv_deg - inv_base,
        pooling_warning=pooling,
    )


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall tau-b, tie-corrected. Implemented here to avoid a SciPy dependency.

    Tau-b rather than tau-a because two models can post identical accuracy on a
    finite question set, and tau-a would treat that tie as a half-disagreement.
    """
    if len(x) != len(y):
        raise ValueError("tau requires equal-length inputs")
    n = len(x)
    if n < 2:
        return float("nan")

    concordant = discordant = tied_x = tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0:
                tied_x += 1
            if dy == 0:
                tied_y += 1
            s = dx * dy
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1

    n0 = n * (n - 1) / 2
    denom = math.sqrt((n0 - tied_x) * (n0 - tied_y))
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def _shared_accuracies(
    rows: Sequence[dict], condition: str
) -> tuple[dict[str, float], list[str]]:
    """Per-model accuracy on the questions every model answered in this condition.

    Ranking models on different question sets would let an easier subset beat a
    better model, so the ranking is computed on the intersection even though the
    headline per-model accuracy in the results table is not.
    """
    subset = filter_rows(rows, condition=condition)
    models = sorted({r.get("model_label") for r in subset if r.get("model_label")})
    maps = {
        m: {
            r["question_id"]: bool(r.get("correct"))
            for r in filter_rows(subset, model_label=m)
        }
        for m in models
    }
    if not maps:
        return {}, []
    shared = sorted(set.intersection(*(set(v) for v in maps.values())))
    if not shared:
        return {}, []
    return (
        {m: float(np.mean([1.0 if maps[m][q] else 0.0 for q in shared])) for m in models},
        shared,
    )


def ranking_stability(
    rows: Sequence[dict],
    *,
    baseline: str = "clean",
    degraded: str = "corrupt",
    alpha: float = 0.05,
) -> dict:
    """Does the accuracy ordering of models survive degradation? (RQ4)

    Reported two ways, because neither alone answers the question:

    `kendall_tau_b` summarises how much the ordering moved. We deliberately do
    not attach a p-value to it — with five models there are only 120 possible
    orderings, and a rank-correlation test on five points has almost no power.
    A non-significant tau would say nothing about stability.

    The `inversions` audit is the load-bearing part. An order flip only counts
    as evidence that the ranking changed if the pair was significantly ordered
    one way in the baseline and significantly ordered the other way after
    degradation. Two models separated by half a point that swap places have not
    changed ranking; they were never distinguishable. `ranking_preserved` is
    therefore about supported inversions only.
    """
    acc_base, shared_base = _shared_accuracies(rows, baseline)
    acc_deg, shared_deg = _shared_accuracies(rows, degraded)
    models = sorted(set(acc_base) & set(acc_deg))

    if len(models) < 2:
        return {
            "baseline_condition": baseline,
            "degraded_condition": degraded,
            "models": models,
            "kendall_tau_b": float("nan"),
            "inversions": [],
            "n_supported_inversions": 0,
            "ranking_preserved": None,
            "note": "fewer than two models present in both conditions; nothing to rank",
        }

    base_p = {
        c["pair"]: c
        for c in pairwise_comparisons(rows, condition=baseline, alpha=alpha)["comparisons"]
    }
    deg_p = {
        c["pair"]: c
        for c in pairwise_comparisons(rows, condition=degraded, alpha=alpha)["comparisons"]
    }

    inversions = []
    for i, a in enumerate(models):
        for b_model in models[i + 1:]:
            sb = np.sign(acc_base[a] - acc_base[b_model])
            sd = np.sign(acc_deg[a] - acc_deg[b_model])
            if sb == 0 or sd == 0 or sb == sd:
                continue
            label = pair_label(a, b_model)
            pb = base_p.get(label, {}).get("p_raw", float("nan"))
            pd_ = deg_p.get(label, {}).get("p_raw", float("nan"))
            supported = bool(pb < alpha and pd_ < alpha)
            inversions.append({
                "pair": label,
                "baseline_leader": a if sb > 0 else b_model,
                "degraded_leader": a if sd > 0 else b_model,
                "baseline_delta": acc_base[a] - acc_base[b_model],
                "degraded_delta": acc_deg[a] - acc_deg[b_model],
                "p_baseline": pb,
                "p_degraded": pd_,
                "supported": supported,
                "reading": (
                    "order reverses and both orderings are individually significant"
                    if supported
                    else "order reverses but the pair is not significantly separated in "
                         "both conditions; consistent with noise"
                ),
            })

    n_supported = sum(1 for inv in inversions if inv["supported"])
    return {
        "baseline_condition": baseline,
        "degraded_condition": degraded,
        "models": models,
        "n_shared_baseline": len(shared_base),
        "n_shared_degraded": len(shared_deg),
        "accuracy_baseline": acc_base,
        "accuracy_degraded": acc_deg,
        "rank_baseline": [m for m in sorted(models, key=lambda m: -acc_base[m])],
        "rank_degraded": [m for m in sorted(models, key=lambda m: -acc_deg[m])],
        "kendall_tau_b": kendall_tau_b(
            [acc_base[m] for m in models], [acc_deg[m] for m in models]
        ),
        "inversions": inversions,
        "n_supported_inversions": n_supported,
        "ranking_preserved": n_supported == 0,
        "note": (
            "ranking_preserved counts only inversions where both orderings are "
            "individually significant; tau is reported without a p-value because a "
            "rank test on this few models has negligible power"
        ),
    }


# ------------------------------------------------------- efficiency & cost

def token_profile(rows: Sequence[dict]) -> dict:
    """Thinking-token profile, with an explicit note when it is a proxy.

    If the provider never reported reasoning tokens, RQ2 falls back to total
    completion tokens. That substitution is recorded rather than assumed, so
    the paper states which quantity the frontier plot is actually on.
    """
    reported = [r for r in rows if r.get("reasoning_tokens_reported")]
    use_proxy = len(reported) == 0
    field = "completion_tokens" if use_proxy else "reasoning_tokens"
    vals = [r.get(field) for r in rows if isinstance(r.get(field), (int, float))]
    lat = [r.get("latency_ms") for r in rows if isinstance(r.get("latency_ms"), (int, float))]

    return {
        "token_metric": field,
        "is_proxy": use_proxy,
        "reasoning_reported_frac": (len(reported) / len(rows)) if rows else 0.0,
        "mean_tokens": float(np.mean(vals)) if vals else float("nan"),
        "median_tokens": float(np.median(vals)) if vals else float("nan"),
        "p95_tokens": float(np.percentile(vals, 95)) if vals else float("nan"),
        "median_latency_ms": float(np.median(lat)) if lat else float("nan"),
        "p95_latency_ms": float(np.percentile(lat, 95)) if lat else float("nan"),
    }


def cost_profile(
    rows: Sequence[dict], *, usd_per_1m_input: float | None, usd_per_1m_output: float | None
) -> dict:
    """Dollar figures are a dated snapshot of one provider, not a model property.

    Prices for open-weights models change with promotions and with who is
    hosting. Report tokens as the durable axis and treat this as an appendix
    table stamped with a date.
    """
    if usd_per_1m_input is None or usd_per_1m_output is None:
        return {"priced": False}

    pin = sum(r.get("prompt_tokens") or 0 for r in rows)
    pout = sum(r.get("completion_tokens") or 0 for r in rows)
    total = pin / 1e6 * usd_per_1m_input + pout / 1e6 * usd_per_1m_output
    n_correct = sum(1 for r in rows if r.get("correct"))

    # An unstamped dollar figure is not reportable. The appendix has to say who
    # was charging what and when, so the quote date and provider travel with the
    # number instead of being reconstructed by hand later.
    quote_dates = sorted({r.get("price_quoted_on") or "" for r in rows} - {""})
    providers = sorted({r.get("served_by") or "" for r in rows} - {""})

    return {
        "priced": True,
        "usd_per_1m_input": usd_per_1m_input,
        "usd_per_1m_output": usd_per_1m_output,
        "price_quoted_on": quote_dates,
        "served_by": providers,
        "price_unstamped": not quote_dates,
        "input_tokens": pin,
        "output_tokens": pout,
        "total_usd": round(total, 6),
        "usd_per_call": round(total / len(rows), 8) if rows else float("nan"),
        "n_correct": n_correct,
        "usd_per_correct": round(total / n_correct, 6) if n_correct else float("nan"),
    }


def pareto_frontier(points: Sequence[tuple[str, float, float]]) -> list[str]:
    """Non-dominated labels from (label, cost, benefit) triples.

    Cost is minimised, benefit maximised. A point survives unless some other
    point is at least as good on both axes and strictly better on one, so two
    genuinely tied points both stay on the frontier rather than one arbitrarily
    eliminating the other.

    Points with a missing coordinate are excluded: a model whose token count was
    never reported cannot be placed on a token axis, and defaulting it to zero
    would put it on the frontier by accident.
    """
    usable = [
        (lbl, c, b) for lbl, c, b in points
        if isinstance(c, (int, float)) and isinstance(b, (int, float))
        and math.isfinite(c) and math.isfinite(b)
    ]
    frontier = []
    for lbl, c, b in usable:
        dominated = any(
            oc <= c and ob >= b and (oc < c or ob > b)
            for olbl, oc, ob in usable
            if olbl != lbl
        )
        if not dominated:
            frontier.append(lbl)
    return sorted(frontier)


def efficiency_frontier(
    rows: Sequence[dict], *, condition: str, n_boot: int = 10000, seed: int = 0
) -> dict:
    """Accuracy against thinking tokens for every model in one condition. (RQ2)

    The trap this guards: if one model reports `reasoning_tokens` and another
    does not, plotting each on its own available metric compares reasoning
    tokens against total completion tokens and the frontier is meaningless. When
    the metric is not uniform across models, every model falls back to
    `completion_tokens`, which all of them report, and the substitution is
    recorded in the output instead of being silently absorbed.

    No scalar accuracy-per-token ratio is produced. It divides a bounded
    quantity by an unbounded one, so it is dominated by whichever model happened
    to think least and is close to uninterpretable.
    """
    subset = filter_rows(rows, condition=condition)
    models = sorted({r.get("model_label") for r in subset if r.get("model_label")})
    per_model = {m: filter_rows(subset, model_label=m) for m in models}
    profiles = {m: token_profile(rs) for m, rs in per_model.items()}

    metrics = {p["token_metric"] for p in profiles.values()}
    uniform = len(metrics) <= 1
    metric = "reasoning_tokens" if metrics == {"reasoning_tokens"} else "completion_tokens"
    fallback_reason = "" if uniform else (
        "at least one model never reported reasoning_tokens, so the shared axis is "
        "completion_tokens for every model; per-model reasoning coverage is in "
        "reasoning_reported_frac"
    )

    points = []
    for m in models:
        rs = per_model[m]
        acc = accuracy(rs, n_boot=n_boot, seed=seed)
        vals = [r.get(metric) for r in rs if isinstance(r.get(metric), (int, float))]
        prices = _prices_for(rs, None)
        cost = cost_profile(
            rs, usd_per_1m_input=prices.get("input"), usd_per_1m_output=prices.get("output")
        )
        points.append({
            "model": m,
            "n": acc.n,
            "accuracy": acc.accuracy,
            "ci_low": acc.ci_low,
            "ci_high": acc.ci_high,
            "chance": chance_accuracy(rs),
            "mean_tokens": float(np.mean(vals)) if vals else float("nan"),
            "median_tokens": float(np.median(vals)) if vals else float("nan"),
            "reasoning_reported_frac": profiles[m]["reasoning_reported_frac"],
            "usd_per_correct": cost.get("usd_per_correct") if cost.get("priced") else None,
        })

    return {
        "condition": condition,
        "token_metric": metric,
        "metric_is_uniform": uniform,
        "is_proxy": metric == "completion_tokens",
        "fallback_reason": fallback_reason,
        "points": sorted(points, key=lambda p: -p["accuracy"]),
        "pareto_frontier": pareto_frontier(
            [(p["model"], p["mean_tokens"], p["accuracy"]) for p in points]
        ),
        "note": (
            "frontier minimises tokens and maximises accuracy; no scalar "
            "accuracy-per-token ratio is reported"
        ),
    }


# ------------------------------------------------------------- breakdowns

def by_category(rows: Sequence[dict], **kw) -> dict[str, dict]:
    cats: dict[str, list] = {}
    for r in rows:
        cats.setdefault(r.get("category") or "unknown", []).append(r)
    return {c: accuracy(rs, **kw).as_dict() for c, rs in sorted(cats.items())}


def extraction_breakdown(rows: Sequence[dict]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("extraction_method") or "unknown"] = (
            counts.get(r.get("extraction_method") or "unknown", 0) + 1
        )
    n = len(rows) or 1
    return {k: round(v / n, 4) for k, v in sorted(counts.items())}


def filter_rows(rows: Sequence[dict], **eq) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in eq.items())]


@dataclass
class Report:
    sections: dict = field(default_factory=dict)

    def to_json(self) -> str:
        import json
        return json.dumps(self.sections, indent=2, default=str)


def _prices_for(subset: Sequence[dict], override: dict | None) -> dict:
    """Prices come from the records themselves unless explicitly overridden."""
    if override:
        return override
    for r in subset:
        if r.get("usd_per_1m_input") is not None and r.get("usd_per_1m_output") is not None:
            return {"input": r["usd_per_1m_input"], "output": r["usd_per_1m_output"]}
    return {}


def provenance(rows: Sequence[dict]) -> dict:
    """Serving configuration per model, with contamination flagged.

    Every field here is one the paper has to report and one where more than one
    distinct value across a model's records is a problem rather than a detail. A
    provider that swapped model builds mid-run, or a thinking-effort setting that
    drifted, breaks comparability; the pilot checks for it on 20 calls and this
    checks the same thing across the whole collected set.
    """
    out: dict = {}
    fields = (
        "model_string", "served_model", "served_by", "served_provider", "quantization",
        "thinking_effort", "max_tokens", "temperature", "reasoning_format",
        "include_sampling_params", "max_tokens_field", "prompt_key", "cohort_id",
        "harness_version", "git_sha",
    )
    for model in sorted({r.get("model_label") for r in rows if r.get("model_label")}):
        subset = filter_rows(rows, model_label=model)
        entry: dict = {"n_rows": len(subset)}
        inconsistent = []
        for f in fields:
            vals = sorted({str(r.get(f)) for r in subset if r.get(f) not in (None, "")})
            # "" for never recorded, the value itself when consistent, the list
            # when it is not. An empty list would read as "several values".
            entry[f] = "" if not vals else (vals[0] if len(vals) == 1 else vals)
            # prompt_key legitimately differs between the image and blind
            # templates, so it is excluded from the contamination check.
            if len(vals) > 1 and f != "prompt_key":
                inconsistent.append(f)
        entry["seeds"] = sorted({r.get("seed") for r in subset if r.get("seed") is not None})
        entry["conditions"] = sorted(
            {r.get("condition") for r in subset if r.get("condition")}
        )
        entry["inconsistent_fields"] = inconsistent
        out[model] = entry
    return out


def _report_warnings(out: dict, rows: Sequence[dict]) -> list[str]:
    """Everything a reader of the JSON should be told before quoting a number."""
    warnings: list[str] = []

    for model, prov in out.get("provenance", {}).items():
        for f in prov.get("inconsistent_fields", []):
            warnings.append(
                f"{model}: more than one value of {f} across records ({prov[f]}). "
                "Records collected under different serving configurations are not "
                "comparable and should be separated before reporting."
            )
        if not prov.get("git_sha"):
            warnings.append(
                f"{model}: no git_sha on any record. The repository had no commit at "
                "collection time, so results cannot be traced to a harness revision. "
                "Commit before collecting."
            )

    for key, section in out.get("by_model_condition", {}).items():
        if section["tokens"]["is_proxy"]:
            warnings.append(
                f"{key}: reasoning_tokens never reported; token figures are "
                "completion_tokens as a proxy. RQ2 must say so explicitly."
            )
        if section["cost"].get("price_unstamped"):
            warnings.append(
                f"{key}: priced with no price_quoted_on. A dollar figure without a "
                "quote date is not reportable; pass --price-date at collection."
            )
        inv = section["accuracy"]["invalid_rate"]
        if isinstance(inv, float) and inv > 0.10:
            warnings.append(
                f"{key}: invalid-output rate {inv:.1%}. Report it in the results "
                "table and check whether max_tokens truncated the reasoning."
            )

    for cond, eff in out.get("efficiency", {}).items():
        if not eff["metric_is_uniform"]:
            warnings.append(f"efficiency/{cond}: {eff['fallback_reason']}")

    for model, rob in out.get("robustness", {}).items():
        if rob.get("pooling_warning"):
            warnings.append(rob["pooling_warning"])

    for cond, comp in out.get("comparisons", {}).items():
        warnings.extend(f"comparisons/{cond}: {w}" for w in comp["warnings"])
        if comp["primary"] is None and comp["family_size"] > 1:
            warnings.append(
                f"comparisons/{cond}: no primary comparison designated, so all "
                f"{comp['family_size']} pairs are Holm-corrected. Preregister one "
                "primary comparison to give it full power."
            )

    return warnings


def summarize_run(
    rows: Sequence[dict],
    *,
    model_prices: dict | None = None,
    primary_comparison: tuple[str, str] | None = None,
    baseline_condition: str = "clean",
    degraded_condition: str = "corrupt",
    alpha: float = 0.05,
    n_boot: int = 10000,
) -> Report:
    """One report over a scored row set: RQ1 through RQ4.

    Sections are emitted only when the data supports them. A run with one model
    gets no pairwise comparisons; a run without the corrupt condition gets no
    robustness section. Emitting an empty or nan-filled section would invite
    someone to quote it.
    """
    model_prices = model_prices or {}
    out: dict = {"n_rows": len(rows), "by_model_condition": {}}

    keys = sorted({(r.get("model_label"), r.get("condition")) for r in rows})
    for model, condition in keys:
        subset = filter_rows(rows, model_label=model, condition=condition)
        prices = _prices_for(subset, model_prices.get(model))
        out["by_model_condition"][f"{model}|{condition}"] = {
            "accuracy": accuracy(subset, n_boot=n_boot).as_dict(),
            # Every "above chance" claim needs the measured value, because the
            # MCQ subset mixes option counts.
            "chance": chance_accuracy(subset),
            "by_category": by_category(subset, n_boot=n_boot),
            "extraction_methods": extraction_breakdown(subset),
            "tokens": token_profile(subset),
            "cost": cost_profile(
                subset,
                usd_per_1m_input=prices.get("input"),
                usd_per_1m_output=prices.get("output"),
            ),
        }

    models = sorted({r.get("model_label") for r in rows if r.get("model_label")})
    conditions = sorted({r.get("condition") for r in rows if r.get("condition")})

    # RQ3: grounding decomposition, one per model with all three conditions.
    for model in models:
        c = filter_rows(rows, model_label=model, condition="clean")
        bt = filter_rows(rows, model_label=model, condition="blind_tags")
        bn = filter_rows(rows, model_label=model, condition="blind_notags")
        if c and bt and bn:
            out.setdefault("grounding", {})[model] = grounding_decomposition(
                c, bt, bn, n_boot=n_boot
            ).as_dict()

    # RQ1: paired model comparisons, corrected within condition.
    if len(models) > 1:
        for condition in conditions:
            out.setdefault("comparisons", {})[condition] = pairwise_comparisons(
                rows, condition=condition, primary=primary_comparison, alpha=alpha
            )

    # RQ2: accuracy against thinking tokens.
    for condition in conditions:
        out.setdefault("efficiency", {})[condition] = efficiency_frontier(
            rows, condition=condition, n_boot=n_boot
        )

    # RQ4: does accuracy, and then the ranking, survive degradation.
    if baseline_condition in conditions and degraded_condition in conditions:
        for model in models:
            res = robustness(
                rows, model=model, baseline=baseline_condition,
                degraded=degraded_condition, n_boot=n_boot,
            )
            if res.n_shared:
                out.setdefault("robustness", {})[model] = res.as_dict()
        if len(models) > 1:
            out["ranking_stability"] = ranking_stability(
                rows, baseline=baseline_condition, degraded=degraded_condition, alpha=alpha
            )

    out["provenance"] = provenance(rows)
    out["warnings"] = _report_warnings(out, rows)
    return Report(sections=out)
