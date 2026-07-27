"""Pilot tests. The pilot's job is to answer four questions before money is spent."""

from __future__ import annotations

import math

from idq.adapters import FixtureAdapter
from idq.config import MOCK_MODEL, DecodeParams, ModelSpec, RunConfig
from idq.pilot import run_pilot
from idq.providers import MockProvider

PRICED = ModelSpec(
    label="mock-priced", model_string="mock/uniform", served_by="mock",
    usd_per_1m_input=1.87, usd_per_1m_output=4.68, price_quoted_on="2026-07-24",
)


def cfg(model=MOCK_MODEL):
    return RunConfig(model=model, condition="blind_tags", seed=0,
                     decode=DecodeParams(temperature=0.0, max_tokens=512))


def test_pilot_reports_payload_health(tmp_path):
    qs = FixtureAdapter(n=100).load()
    r = run_pilot(qs, cfg(), MockProvider(seed=5), str(tmp_path / "p.jsonl"), n=20)

    assert r.payload_ok is True
    assert r.stats["successes"] == 20
    assert r.reasoning_tokens_reported_frac == 1.0
    assert r.invalid_rate == 0.0
    assert r.mean_completion_tokens > 0
    assert not math.isnan(r.median_latency_ms)


def test_pilot_detects_determinism(tmp_path):
    """The mock is deterministic given the same prompt and seed, so a repeat
    run must come back byte-identical and the pilot must say so."""
    qs = FixtureAdapter(n=50).load()
    r = run_pilot(qs, cfg(), MockProvider(seed=5), str(tmp_path / "p.jsonl"),
                  n=20, n_repeat=5)

    assert r.determinism["n_compared"] == 5
    assert r.determinism["identical_frac"] == 1.0
    assert any("One seed is sufficient" in w for w in r.warnings)


def test_pilot_measures_cost_from_real_usage(tmp_path):
    qs = FixtureAdapter(n=50).load()
    r = run_pilot(qs, cfg(PRICED), MockProvider(seed=5), str(tmp_path / "p.jsonl"), n=20)

    assert r.measured_usd_per_call is not None and r.measured_usd_per_call > 0
    expected = (
        r.mean_prompt_tokens / 1e6 * 1.87 + r.mean_completion_tokens / 1e6 * 4.68
    )
    # measured_usd_per_call is rounded to 8dp for reporting, so allow for that.
    assert abs(r.measured_usd_per_call - expected) < 1e-8


def test_pilot_warns_when_no_price_is_set(tmp_path):
    qs = FixtureAdapter(n=50).load()
    unpriced = ModelSpec(label="u", model_string="mock/uniform", served_by="mock")
    r = run_pilot(qs, cfg(unpriced), MockProvider(seed=5), str(tmp_path / "p.jsonl"), n=20)
    assert r.measured_usd_per_call is None
    assert any("No price set" in w for w in r.warnings)


def test_sizing_reports_both_constraints(tmp_path):
    qs = FixtureAdapter(n=50).load()
    r = run_pilot(qs, cfg(PRICED), MockProvider(seed=5), str(tmp_path / "p.jsonl"),
                  n=20, budget_usd=15.0, n_models=5, n_conditions=3)

    s = r.sizing
    assert s["cells"] == 15
    # 1.96^2 * 0.25 / 0.04^2 = 600.25 -> 601
    assert s["questions_for_target_precision"] == 601
    assert s["binding_constraint"] in ("budget", "precision")
    assert s["recommended_questions"] == min(
        s["questions_for_target_precision"], s["questions_at_budget_one_seed"]
    )


def test_pilot_reports_failure_clearly_when_everything_errors(tmp_path):
    qs = FixtureAdapter(n=50).load()
    refuser = ModelSpec(label="r", model_string="mock/refuser", served_by="mock")
    r = run_pilot(qs, cfg(refuser), MockProvider(seed=5), str(tmp_path / "p.jsonl"), n=20)

    assert r.payload_ok is False
    assert any("No successful responses" in w for w in r.warnings)
    assert r.stats["terminal_errors"] == 20


def test_pilot_flags_high_invalid_rate(tmp_path):
    qs = FixtureAdapter(n=100).load()
    r = run_pilot(qs, cfg(), MockProvider(seed=5, style="messy"),
                  str(tmp_path / "p.jsonl"), n=40, n_repeat=0)
    assert r.invalid_rate > 0.10
    assert any("invalid-output rate" in w for w in r.warnings)


def test_pilot_reuses_cache_on_rerun(tmp_path):
    """Rerunning the pilot must not pay twice."""
    qs = FixtureAdapter(n=50).load()
    path = str(tmp_path / "p.jsonl")
    run_pilot(qs, cfg(), MockProvider(seed=5), path, n=20, n_repeat=0)

    p = MockProvider(seed=5)
    r2 = run_pilot(qs, cfg(), p, path, n=20, n_repeat=0)
    assert r2.stats["calls_made"] == 0
    assert r2.stats["cached_hits"] == 20
    assert p.calls == 0
