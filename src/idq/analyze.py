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

    return {
        "priced": True,
        "input_tokens": pin,
        "output_tokens": pout,
        "total_usd": round(total, 6),
        "usd_per_call": round(total / len(rows), 8) if rows else float("nan"),
        "n_correct": n_correct,
        "usd_per_correct": round(total / n_correct, 6) if n_correct else float("nan"),
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


def summarize_run(rows: Sequence[dict], *, model_prices: dict | None = None) -> Report:
    """One report over a scored row set, sliced by model and condition."""
    model_prices = model_prices or {}
    out: dict = {"n_rows": len(rows), "by_model_condition": {}}

    keys = sorted({(r.get("model_label"), r.get("condition")) for r in rows})
    for model, condition in keys:
        subset = filter_rows(rows, model_label=model, condition=condition)
        prices = _prices_for(subset, model_prices.get(model))
        out["by_model_condition"][f"{model}|{condition}"] = {
            "accuracy": accuracy(subset).as_dict(),
            "by_category": by_category(subset),
            "extraction_methods": extraction_breakdown(subset),
            "tokens": token_profile(subset),
            "cost": cost_profile(
                subset,
                usd_per_1m_input=prices.get("input"),
                usd_per_1m_output=prices.get("output"),
            ),
        }

    for model in sorted({r.get("model_label") for r in rows}):
        c = filter_rows(rows, model_label=model, condition="clean")
        bt = filter_rows(rows, model_label=model, condition="blind_tags")
        bn = filter_rows(rows, model_label=model, condition="blind_notags")
        if c and bt and bn:
            out.setdefault("grounding", {})[model] = grounding_decomposition(c, bt, bn).as_dict()

    return Report(sections=out)
