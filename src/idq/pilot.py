"""The 20-call pilot.

Answers four questions before any real money is committed:

1. Does the provider's payload shape actually work? (the most likely failure)
2. Does it report reasoning_tokens, or do we fall back to a proxy?
3. What does a call really cost, measured rather than assumed?
4. Is the model deterministic at temperature 0 with a fixed seed?

Question 4 decides whether the full study needs three seeds or one. Batched MoE
inference is often nondeterministic even at temperature 0, and many providers
ignore `seed` entirely. If output is reproducible, three seeds is a 3x waste. If
it isn't, run-to-run variance is real and has to be reported rather than
described as seed variation.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from .cache import ResponseCache
from .collect import collect
from .score import score_records


@dataclass
class PilotReport:
    n_requested: int = 0
    stats: dict = field(default_factory=dict)
    payload_ok: bool = False
    reasoning_tokens_reported_frac: float = 0.0
    served_models: list = field(default_factory=list)
    invalid_rate: float = float("nan")
    extraction_methods: dict = field(default_factory=dict)
    mean_prompt_tokens: float = float("nan")
    mean_completion_tokens: float = float("nan")
    mean_reasoning_tokens: float = float("nan")
    median_latency_ms: float = float("nan")
    measured_usd_per_call: float | None = None
    determinism: dict = field(default_factory=dict)
    sizing: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _median(vals):
    vals = sorted(v for v in vals if isinstance(v, (int, float)))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    return float(vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2)


def run_pilot(
    questions,
    cfg,
    provider,
    cache_path: str,
    *,
    n: int = 20,
    n_repeat: int = 5,
    image_root: str = "",
    check_determinism: bool = True,
    budget_usd: float = 15.0,
    n_models: int = 5,
    n_conditions: int = 3,
    target_ci_halfwidth: float = 0.04,
) -> PilotReport:
    report = PilotReport(n_requested=n)
    subset = questions[:n]

    cache = ResponseCache(cache_path)
    stats = collect(subset, cfg, provider, cache, image_root=image_root, verbose=False)
    report.stats = stats.as_dict()

    rows = score_records(ResponseCache(cache_path).records(), subset)
    ok = [r for r in rows if r["status"] == "success"]
    report.payload_ok = len(ok) > 0

    if not ok:
        report.warnings.append(
            "No successful responses. Check the model string, base URL and API key "
            "before changing anything else."
        )
        return report

    report.reasoning_tokens_reported_frac = sum(
        1 for r in ok if r.get("reasoning_tokens_reported")
    ) / len(ok)
    report.served_models = sorted({r.get("served_model") or "" for r in ok} - {""})
    report.invalid_rate = 1.0 - (sum(1 for r in ok if r["valid"]) / len(ok))

    methods: dict[str, int] = {}
    for r in rows:
        methods[r["extraction_method"]] = methods.get(r["extraction_method"], 0) + 1
    report.extraction_methods = dict(sorted(methods.items()))

    report.mean_prompt_tokens = _mean([r.get("prompt_tokens") for r in ok])
    report.mean_completion_tokens = _mean([r.get("completion_tokens") for r in ok])
    report.mean_reasoning_tokens = _mean([r.get("reasoning_tokens") for r in ok])
    report.median_latency_ms = _median([r.get("latency_ms") for r in ok])

    if report.reasoning_tokens_reported_frac == 0.0:
        report.warnings.append(
            "reasoning_tokens never reported. RQ2 falls back to completion_tokens "
            "as a proxy; say so explicitly in the paper."
        )
    if report.invalid_rate > 0.10:
        report.warnings.append(
            f"invalid-output rate {report.invalid_rate:.1%} on the pilot. Consider "
            "raising max_tokens or tightening the prompt before scaling up."
        )
    if len(report.served_models) > 1:
        report.warnings.append(
            f"provider served more than one model build during the pilot: "
            f"{report.served_models}. Results may not be comparable."
        )

    price_in = cfg.model.usd_per_1m_input
    price_out = cfg.model.usd_per_1m_output
    if price_in is not None and price_out is not None:
        per_call = (
            report.mean_prompt_tokens / 1e6 * price_in
            + report.mean_completion_tokens / 1e6 * price_out
        )
        report.measured_usd_per_call = round(per_call, 8)
    else:
        report.warnings.append(
            "No price set on the ModelSpec, so cost is unmeasured. Fill in "
            "usd_per_1m_input/output and price_quoted_on from the provider console."
        )

    if check_determinism and n_repeat > 0:
        report.determinism = _determinism_check(
            subset[:n_repeat], cfg, provider, cache_path, image_root=image_root
        )
        if report.determinism.get("identical_frac") == 1.0:
            report.warnings.append(
                "Output was byte-identical on repeat. One seed is sufficient; "
                "three seeds would triple cost for no information."
            )
        elif report.determinism.get("identical_frac") is not None:
            report.warnings.append(
                "Output varied between identical requests. Multiple runs measure "
                "provider nondeterminism, not seed variation. Report it as "
                "run-to-run variance."
            )

    report.sizing = _sizing(
        report.measured_usd_per_call,
        budget_usd=budget_usd,
        n_models=n_models,
        n_conditions=n_conditions,
        target_ci_halfwidth=target_ci_halfwidth,
    )
    return report


def _determinism_check(subset, cfg, provider, cache_path, *, image_root=""):
    """Re-issue identical requests into a separate cache and compare.

    A separate cache file is the point: the same cache key would otherwise be a
    hit and no second call would be made.
    """
    second_path = cache_path + ".determinism"
    if os.path.exists(second_path):
        os.remove(second_path)

    collect(subset, cfg, provider, ResponseCache(second_path),
            image_root=image_root, verbose=False)

    first = {r["question_id"]: r for r in ResponseCache(cache_path).records()
             if r.get("status") == "success"}
    second = {r["question_id"]: r for r in ResponseCache(second_path).records()
              if r.get("status") == "success"}

    shared = sorted(set(first) & set(second))
    if not shared:
        return {"n_compared": 0, "identical_frac": None,
                "note": "no overlapping successful calls to compare"}

    identical_text = sum(
        1 for q in shared if first[q]["response_text"] == second[q]["response_text"]
    )
    same_letter = 0
    for q in shared:
        a = (first[q].get("response_text") or "")[-80:]
        b = (second[q].get("response_text") or "")[-80:]
        same_letter += int(a == b)

    return {
        "n_compared": len(shared),
        "identical_frac": identical_text / len(shared),
        "identical_tail_frac": same_letter / len(shared),
        "note": (
            "identical_frac 1.0 means temperature 0 plus seed is reproducible on "
            "this provider; below 1.0 means it is not"
        ),
    }


def _sizing(usd_per_call, *, budget_usd, n_models, n_conditions, target_ci_halfwidth):
    """Two independent limits on sample size: money and precision.

    The precision figure uses the worst-case binomial half-width at p=0.5,
    1.96*sqrt(0.25/n), which is the conservative bound. Real driving-QA accuracy
    away from 0.5 gives a tighter interval than this.
    """
    out: dict = {
        "cells": n_models * n_conditions,
        "target_ci_halfwidth": target_ci_halfwidth,
        "questions_for_target_precision": math.ceil(
            0.25 * (1.96 / target_ci_halfwidth) ** 2
        ),
    }
    if usd_per_call == 0:
        # A free model is not a budget signal. Say so rather than silently
        # dropping the budget half of the sizing analysis.
        out["note"] = "cost is zero (mock or free tier); only precision constrains n"
        out["recommended_questions"] = out["questions_for_target_precision"]
        out["binding_constraint"] = "precision"
    elif usd_per_call:
        affordable_calls = int(budget_usd / usd_per_call)
        out.update(
            {
                "budget_usd": budget_usd,
                "measured_usd_per_call": usd_per_call,
                "affordable_calls": affordable_calls,
                "questions_at_budget_one_seed": affordable_calls
                // max(1, n_models * n_conditions),
            }
        )
        out["recommended_questions"] = min(
            out["questions_for_target_precision"],
            out["questions_at_budget_one_seed"],
        )
        out["binding_constraint"] = (
            "budget"
            if out["questions_at_budget_one_seed"] < out["questions_for_target_precision"]
            else "precision"
        )
    return out
