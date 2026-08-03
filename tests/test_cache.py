"""Cache behaviour: never pay twice, survive interruption, fork on change."""

from __future__ import annotations

import os

import pytest

from idq.adapters import FixtureAdapter
from idq.cache import ResponseCache
from idq.config import MOCK_MODEL, DecodeParams, RunConfig
from idq.collect import append_run_log, collect, git_is_dirty
from idq.providers import MockProvider


def cfg(**kw):
    base = dict(model=MOCK_MODEL, condition="blind_tags", seed=0)
    base.update(kw)
    return RunConfig(**base)


def test_second_run_makes_zero_calls(tmp_path):
    qs = FixtureAdapter(n=40).load()
    cache_path = str(tmp_path / "cache.jsonl")

    p1 = MockProvider(seed=1)
    s1 = collect(qs, cfg(), p1, ResponseCache(cache_path), verbose=False)
    assert s1.calls_made == 40 and s1.successes == 40

    p2 = MockProvider(seed=1)
    s2 = collect(qs, cfg(), p2, ResponseCache(cache_path), verbose=False)
    assert s2.calls_made == 0
    assert s2.cached_hits == 40
    assert p2.calls == 0


def test_resume_after_interruption(tmp_path):
    qs = FixtureAdapter(n=50).load()
    cache_path = str(tmp_path / "cache.jsonl")

    collect(qs, cfg(), MockProvider(seed=1), ResponseCache(cache_path), limit=20, verbose=False)
    assert len(ResponseCache(cache_path)) == 20

    s = collect(qs, cfg(), MockProvider(seed=1), ResponseCache(cache_path), verbose=False)
    assert s.cached_hits == 20
    assert s.calls_made == 30
    assert len(ResponseCache(cache_path)) == 50


def test_truncated_trailing_line_does_not_break_resume(tmp_path):
    """A hard kill mid-write costs at most the in-flight call."""
    qs = FixtureAdapter(n=10).load()
    cache_path = str(tmp_path / "cache.jsonl")
    collect(qs, cfg(), MockProvider(seed=1), ResponseCache(cache_path), verbose=False)

    with open(cache_path, "a", encoding="utf-8") as fh:
        fh.write('{"cache_key": "hal')

    cache = ResponseCache(cache_path)
    assert len(cache) == 10
    s = collect(qs, cfg(), MockProvider(seed=1), cache, verbose=False)
    assert s.calls_made == 0


def test_prompt_change_forks_the_cache(tmp_path):
    """Changing a prompt must re-collect, not silently mix two prompts."""
    qs = FixtureAdapter(n=10).load()
    cache_path = str(tmp_path / "cache.jsonl")

    collect(qs, cfg(prompt_key="mcq_blind@v1"), MockProvider(seed=1),
            ResponseCache(cache_path), verbose=False)
    s = collect(qs, cfg(prompt_key="mcq@v1"), MockProvider(seed=1),
                ResponseCache(cache_path), verbose=False)

    assert s.cached_hits == 0
    assert s.calls_made == 10
    assert len(ResponseCache(cache_path)) == 20


def test_decode_param_change_forks_the_cache(tmp_path):
    """Temperature is in the key; changing it must not reuse stale records."""
    qs = FixtureAdapter(n=10).load()
    cache_path = str(tmp_path / "cache.jsonl")

    collect(qs, cfg(decode=DecodeParams(temperature=0.0)), MockProvider(seed=1),
            ResponseCache(cache_path), verbose=False)
    s = collect(qs, cfg(decode=DecodeParams(temperature=0.7)), MockProvider(seed=1),
                ResponseCache(cache_path), verbose=False)
    assert s.calls_made == 10


def test_thinking_effort_change_forks_the_cache(tmp_path):
    qs = FixtureAdapter(n=10).load()
    cache_path = str(tmp_path / "cache.jsonl")

    collect(qs, cfg(decode=DecodeParams(thinking_effort="low")), MockProvider(seed=1),
            ResponseCache(cache_path), verbose=False)
    s = collect(qs, cfg(decode=DecodeParams(thinking_effort="high")), MockProvider(seed=1),
                ResponseCache(cache_path), verbose=False)
    assert s.calls_made == 10


def test_seed_change_forks_the_cache(tmp_path):
    qs = FixtureAdapter(n=10).load()
    cache_path = str(tmp_path / "cache.jsonl")
    collect(qs, cfg(seed=0), MockProvider(seed=1), ResponseCache(cache_path), verbose=False)
    s = collect(qs, cfg(seed=1), MockProvider(seed=1), ResponseCache(cache_path), verbose=False)
    assert s.calls_made == 10


def test_retryable_errors_are_not_cached_and_terminal_errors_are(tmp_path):
    qs = FixtureAdapter(n=30).load()

    # Retryable: not cached, so a later run retries and completes the set.
    flaky_path = str(tmp_path / "flaky.jsonl")
    s1 = collect(qs, cfg(), MockProvider(seed=3, fail_rate=0.5),
                 ResponseCache(flaky_path), verbose=False)
    assert s1.retryable_failures > 0
    assert len(ResponseCache(flaky_path)) == s1.successes

    s2 = collect(qs, cfg(), MockProvider(seed=3, fail_rate=0.0),
                 ResponseCache(flaky_path), verbose=False)
    assert s2.calls_made == s1.retryable_failures
    assert len(ResponseCache(flaky_path)) == 30

    # Terminal: cached, so the second run does not pay to rediscover it.
    from idq.config import ModelSpec
    refuser = ModelSpec(label="refuser", model_string="mock/refuser", served_by="mock")
    term_path = str(tmp_path / "terminal.jsonl")
    t1 = collect(qs, cfg(model=refuser), MockProvider(seed=1),
                 ResponseCache(term_path), verbose=False)
    assert t1.terminal_errors == 30
    t2 = collect(qs, cfg(model=refuser), MockProvider(seed=1),
                 ResponseCache(term_path), verbose=False)
    assert t2.calls_made == 0
    assert t2.cached_hits == 30


def test_full_raw_response_and_prompt_are_stored(tmp_path):
    """Metrics nobody planned for must stay recoverable without re-collecting."""
    qs = FixtureAdapter(n=3).load()
    cache_path = str(tmp_path / "cache.jsonl")
    collect(qs, cfg(), MockProvider(seed=1), ResponseCache(cache_path), verbose=False)

    rec = next(iter(ResponseCache(cache_path).records()))
    for field in ("response_raw", "response_text", "usage", "latency_ms", "ts_iso",
                  "prompt_hash", "prompt_system", "prompt_user", "harness_version"):
        assert field in rec, f"missing {field}"
    assert rec["response_raw"]["choices"][0]["message"]["content"]
    assert rec["usage"]["reasoning_tokens"] is not None


def test_cohort_id_is_carried_into_every_record(tmp_path):
    qs = FixtureAdapter(n=2).load()
    cache_path = str(tmp_path / "cache.jsonl")
    collect(
        qs, cfg(), MockProvider(seed=1), ResponseCache(cache_path),
        cohort_id="cohort-123", verbose=False,
    )
    assert {r["cohort_id"] for r in ResponseCache(cache_path).records()} == {
        "cohort-123"
    }


def test_run_log_is_append_only_and_contains_no_credential(tmp_path, monkeypatch):
    monkeypatch.setenv("IDQ_API_KEY", "sk-never-log-this")
    path = str(tmp_path / "run-log.jsonl")
    append_run_log(path, {"cohort_id": "c1", "stats": {"successes": 2}})
    append_run_log(path, {"cohort_id": "c1", "stats": {"successes": 3}})
    text = open(path, encoding="utf-8").read()
    assert len(text.splitlines()) == 2
    assert "sk-never-log-this" not in text


def test_dirty_tree_detection_fails_closed(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr("idq.collect.subprocess.run", broken)
    assert git_is_dirty() is True


def test_collection_stops_at_measured_budget(tmp_path):
    from idq.config import ModelSpec

    priced = ModelSpec(
        label="priced", model_string="mock/priced", served_by="mock",
        usd_per_1m_input=1000.0, usd_per_1m_output=1000.0,
    )
    qs = FixtureAdapter(n=20).load()
    stats = collect(
        qs,
        cfg(model=priced),
        MockProvider(seed=1),
        ResponseCache(str(tmp_path / "budget.jsonl")),
        max_usd=0.05,
        verbose=False,
    )
    assert stats.budget_exhausted is True
    assert stats.calls_made < 20
    # The ceiling is checked between calls, so overshoot is bounded to one call.
    assert stats.measured_usd >= 0.05


def test_request_pacing_waits_between_call_starts(tmp_path, monkeypatch):
    clock = {"now": 0.0}
    sleeps = []

    def monotonic():
        return clock["now"]

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr("idq.collect.time.monotonic", monotonic)
    monkeypatch.setattr("idq.collect.time.sleep", sleep)
    collect(
        FixtureAdapter(n=3).load(),
        cfg(),
        MockProvider(seed=1),
        ResponseCache(str(tmp_path / "paced.jsonl")),
        requests_per_minute=60,
        verbose=False,
    )
    assert sleeps == [1.0, 1.0]


def test_no_api_key_is_ever_written(tmp_path, monkeypatch):
    monkeypatch.setenv("IDQ_API_KEY", "sk-should-never-appear")
    qs = FixtureAdapter(n=5).load()
    cache_path = str(tmp_path / "cache.jsonl")
    collect(qs, cfg(), MockProvider(seed=1), ResponseCache(cache_path), verbose=False)
    assert "sk-should-never-appear" not in open(cache_path, encoding="utf-8").read()


def test_dry_run_makes_no_calls(tmp_path):
    qs = FixtureAdapter(n=10).load()
    cache_path = str(tmp_path / "cache.jsonl")
    p = MockProvider(seed=1)
    s = collect(qs, cfg(), p, ResponseCache(cache_path), dry_run=True, verbose=False)
    assert p.calls == 0 and s.calls_made == 0
    assert not os.path.exists(cache_path) or len(ResponseCache(cache_path)) == 0


def test_a_missing_credential_aborts_instead_of_poisoning_the_cache(tmp_path, monkeypatch):
    """Regression: an empty API key once marked all 20 pilot questions terminal.

    Terminal records are never retried, so the cache had to be deleted by hand
    before the run could be repaired. A configuration error is identical for
    every question and says nothing about any of them: abort, cache nothing.
    """
    from idq.collect import ConfigurationError
    from idq.providers import OpenAICompatProvider

    monkeypatch.delenv("IDQ_ABSENT_KEY", raising=False)
    qs = FixtureAdapter(n=20).load()
    cache_path = str(tmp_path / "cache.jsonl")
    provider = OpenAICompatProvider(
        base_url="https://example.invalid/v1", api_key_env="IDQ_ABSENT_KEY"
    )

    with pytest.raises(ConfigurationError) as exc:
        collect(qs, cfg(), provider, ResponseCache(cache_path), verbose=False)

    assert "no money was spent" in str(exc.value)
    assert len(ResponseCache(cache_path)) == 0, "configuration error must not be cached"


def test_a_per_request_terminal_error_is_still_cached(tmp_path):
    """The abort must not swallow genuine per-question terminal failures."""
    from idq.config import ModelSpec

    refuser = ModelSpec(label="r", model_string="mock/refuser", served_by="mock")
    qs = FixtureAdapter(n=5).load()
    cache_path = str(tmp_path / "cache.jsonl")

    stats = collect(qs, cfg(model=refuser), MockProvider(seed=1),
                    ResponseCache(cache_path), verbose=False)
    assert stats.terminal_errors == 5
    assert len(ResponseCache(cache_path)) == 5
