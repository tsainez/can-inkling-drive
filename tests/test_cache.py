"""Cache behaviour: never pay twice, survive interruption, fork on change."""

from __future__ import annotations

import os

from idq.adapters import FixtureAdapter
from idq.cache import ResponseCache
from idq.config import MOCK_MODEL, DecodeParams, RunConfig
from idq.collect import collect
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
