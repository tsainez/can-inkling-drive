"""End-to-end offline smoke test. No API key, no network, no cost.

The load-bearing assertion is that a uniformly random mock model scores at
chance. If scoring were subtly wrong - misaligned gold answers, a parser that
prefers the first option, a bug that scores unparseable output as correct -
this test fails. Every other number the harness produces is only trustworthy
because this one holds.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from idq.adapters import FixtureAdapter, expected_chance
from idq.analyze import accuracy, bootstrap_ci, summarize_run
from idq.cache import ResponseCache
from idq.config import MOCK_MODEL, DecodeParams, RunConfig
from idq.collect import collect
from idq.providers import MockProvider
from idq.score import read_rows, score_records, write_rows

BLIND = ("blind_tags", "blind_notags")
GROUNDING_CONDITIONS = ("clean",) + BLIND


def materialize_images(questions, root: str) -> str:
    """Write tiny real JPEGs so the image-bearing conditions run offline."""
    rng = np.random.default_rng(0)
    for q in questions:
        for rel in q.image_paths.values():
            path = os.path.join(root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                arr = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
                Image.fromarray(arr).save(path, format="JPEG", quality=80)
    return root


def run_condition(
    questions, condition, cache_path, *,
    style="uniform", seed=0, mock_seed=11, image_root="",
    corruption="", severity=0,
):
    cfg = RunConfig(
        model=MOCK_MODEL,
        condition=condition,
        seed=seed,
        decode=DecodeParams(temperature=0.0, max_tokens=512),
        corruption=corruption,
        corruption_severity=severity,
        prompt_key="mcq_blind@v1" if condition in BLIND else "mcq@v1",
    )
    provider = MockProvider(style=style, seed=mock_seed)
    cache = ResponseCache(cache_path)
    stats = collect(questions, cfg, provider, cache, image_root=image_root, verbose=False)
    return stats, cache


def test_no_api_key_required(monkeypatch, tmp_path):
    """The whole pipeline must run with a scrubbed environment."""
    for var in [k for k in os.environ if "API" in k.upper() or "KEY" in k.upper()]:
        monkeypatch.delenv(var, raising=False)

    questions = FixtureAdapter(n=100).load()
    stats, _ = run_condition(questions, "blind_tags", str(tmp_path / "c.jsonl"))
    assert stats.successes == 100


def test_fixture_questions_are_distinct():
    """A fixture with few unique prompts would make a prompt-seeded mock
    deterministic and turn the chance test into a test of a handful of draws."""
    questions = FixtureAdapter(n=500).load()
    assert len({q.stem for q in questions}) > 450
    assert len({q.question_id for q in questions}) == 500


def test_mock_model_scores_at_chance(tmp_path):
    """The sanity check that scoring is not lying.

    Chance is mean(1/n_options), not 0.25 - the fixture mixes 2, 3 and 4
    option questions on purpose, so a hardcoded 0.25 would make this test
    itself incorrect.
    """
    questions = FixtureAdapter(n=1200).load()
    chance = expected_chance(questions)
    assert 0.25 < chance < 0.50, "fixture should not be uniformly 4-option"

    _, cache = run_condition(questions, "blind_tags", str(tmp_path / "c.jsonl"))
    rows = score_records(cache.records(), questions)

    assert len(rows) == 1200
    res = accuracy(rows, n_boot=4000)

    # Every answer from the uniform mock is parseable.
    assert res.invalid_rate == 0.0
    # The bootstrap CI must contain chance.
    assert res.ci_low <= chance <= res.ci_high, (
        f"observed {res.accuracy:.4f} CI [{res.ci_low:.4f}, {res.ci_high:.4f}] "
        f"excludes chance {chance:.4f} - scoring is suspect"
    )


def test_mock_accuracy_is_stable_across_mock_seeds(tmp_path):
    """Not a one-seed fluke: several independent mock seeds all land at chance."""
    questions = FixtureAdapter(n=800).load()
    chance = expected_chance(questions)

    for mock_seed in (1, 2, 3, 4, 5):
        _, cache = run_condition(
            questions, "blind_tags", str(tmp_path / f"c{mock_seed}.jsonl"), mock_seed=mock_seed
        )
        rows = score_records(cache.records(), questions)
        res = accuracy(rows, n_boot=2000)
        assert res.ci_low <= chance <= res.ci_high, (
            f"mock seed {mock_seed}: {res.accuracy:.4f} "
            f"CI [{res.ci_low:.4f}, {res.ci_high:.4f}] excludes chance {chance:.4f}"
        )


def test_messy_output_still_scores_at_chance_among_valid_answers(tmp_path):
    """Parser robustness: format variety must not bias which letter wins.

    Invalid output is reported, not hidden, and the letters that do parse are
    still uniform.
    """
    questions = FixtureAdapter(n=1500).load()
    chance = expected_chance(questions)

    _, cache = run_condition(questions, "blind_tags", str(tmp_path / "m.jsonl"), style="messy")
    rows = score_records(cache.records(), questions)

    valid = [r for r in rows if r["valid"]]
    invalid_rate = 1 - len(valid) / len(rows)
    assert 0.10 < invalid_rate < 0.45, f"unexpected invalid rate {invalid_rate:.3f}"

    lo, hi = bootstrap_ci([1.0 if r["correct"] else 0.0 for r in valid], n_boot=4000)
    assert lo <= chance <= hi

    methods = {r["extraction_method"] for r in rows}
    for expected in ("answer_tag", "sole_letter", "boxed", "final_line",
                     "unparseable", "ambiguous"):
        assert expected in methods, f"parser branch {expected} never exercised"


def test_full_pipeline_all_conditions_and_report(tmp_path):
    """Collect, score, analyze - the whole loop, offline, including images."""
    questions = FixtureAdapter(n=300).load()
    root = materialize_images(questions, str(tmp_path / "images"))
    cache_path = str(tmp_path / "cache.jsonl")

    total_calls = 0
    for condition in GROUNDING_CONDITIONS:
        stats, _ = run_condition(questions, condition, cache_path, image_root=root)
        total_calls += stats.calls_made
    stats, _ = run_condition(
        questions, "corrupt", cache_path, image_root=root, corruption="fog", severity=3
    )
    total_calls += stats.calls_made
    assert total_calls == 1200

    cache = ResponseCache(cache_path)
    rows = score_records(cache.records(), questions)
    scored_path = str(tmp_path / "scored.jsonl")
    write_rows(rows, scored_path)

    reloaded = read_rows(scored_path)
    assert len(reloaded) == 1200

    report = summarize_run(reloaded).sections
    assert report["n_rows"] == 1200
    for condition in GROUNDING_CONDITIONS + ("corrupt",):
        assert f"mock-uniform|{condition}" in report["by_model_condition"]

    section = report["by_model_condition"]["mock-uniform|blind_tags"]
    assert set(section["by_category"]) == {"perception", "prediction", "planning", "behavior"}
    assert section["tokens"]["is_proxy"] is False  # mock reports reasoning tokens
    assert section["cost"]["priced"] is True

    # Grounding decomposition is computed once all three conditions exist.
    g = report["grounding"]["mock-uniform"]
    assert g["n_shared"] == 300
    # A model that cannot see is not penalised by removing images, so all three
    # conditions sit at chance and the decomposition is near zero.
    assert abs(g["total_gap"]) < 0.15


def test_corrupt_condition_forks_cache_by_severity(tmp_path):
    """Changing severity must re-collect, not reuse images degraded differently."""
    questions = FixtureAdapter(n=20).load()
    root = materialize_images(questions, str(tmp_path / "images"))
    cache_path = str(tmp_path / "cache.jsonl")

    run_condition(questions, "corrupt", cache_path, image_root=root,
                  corruption="fog", severity=2)
    stats, _ = run_condition(questions, "corrupt", cache_path, image_root=root,
                             corruption="fog", severity=5)
    assert stats.calls_made == 20
    assert stats.cached_hits == 0


def test_scoring_never_touches_the_network(tmp_path, monkeypatch):
    """A scoring bug must cost zero dollars to fix."""
    questions = FixtureAdapter(n=50).load()
    cache_path = str(tmp_path / "cache.jsonl")
    run_condition(questions, "blind_tags", cache_path)

    import socket

    def blocked(*args, **kwargs):  # pragma: no cover - only fires on failure
        raise AssertionError("scoring attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)

    rows = score_records(ResponseCache(cache_path).records(), questions)
    summarize_run(rows)
    assert len(rows) == 50


def test_rescoring_requires_no_recollection(tmp_path):
    """Cache survives; scoring can be rerun any number of times for free."""
    questions = FixtureAdapter(n=80).load()
    cache_path = str(tmp_path / "cache.jsonl")
    run_condition(questions, "blind_tags", cache_path)

    cache = ResponseCache(cache_path)
    first = score_records(cache.records(), questions)
    second = score_records(cache.records(), questions)
    assert [r["predicted_letter"] for r in first] == [r["predicted_letter"] for r in second]

    provider = MockProvider(seed=11)
    cfg = RunConfig(model=MOCK_MODEL, condition="blind_tags", seed=0,
                    decode=DecodeParams(temperature=0.0, max_tokens=512),
                    prompt_key="mcq_blind@v1")
    stats = collect(questions, cfg, provider, ResponseCache(cache_path), verbose=False)
    assert stats.calls_made == 0 and provider.calls == 0


def test_blind_conditions_produce_different_prompts(tmp_path):
    """blind_notags must actually differ from blind_tags, or the decomposition
    measures nothing."""
    questions = FixtureAdapter(n=20).load()
    cache_path = str(tmp_path / "cache.jsonl")
    for condition in BLIND:
        run_condition(questions, condition, cache_path)

    records = list(ResponseCache(cache_path).records())
    tagged = [r for r in records if r["condition"] == "blind_tags"]
    untagged = [r for r in records if r["condition"] == "blind_notags"]

    assert any("CAM_" in r["prompt_user"] for r in tagged)
    assert not any("CAM_" in r["prompt_user"] for r in untagged)
    assert len({r["cache_key"] for r in records}) == 40


@pytest.mark.parametrize("condition", GROUNDING_CONDITIONS)
def test_every_record_carries_provenance(tmp_path, condition):
    """Prompt hash, harness version and decode params in every record, so a
    result can always be traced back to the exact configuration that made it."""
    questions = FixtureAdapter(n=10).load()
    root = materialize_images(questions, str(tmp_path / "images"))
    cache_path = str(tmp_path / f"{condition}.jsonl")
    run_condition(questions, condition, cache_path, image_root=root)

    records = list(ResponseCache(cache_path).records())
    assert len(records) == 10
    for rec in records:
        assert rec["prompt_hash"] and len(rec["prompt_hash"]) == 64
        assert rec["harness_version"]
        assert rec["decode"]["temperature"] == 0.0
        assert rec["condition"] == condition
        assert rec["gold_letter"] in rec["letters"]
        if condition == "clean":
            assert rec["image_manifest"] and rec["image_manifest"][0]["source_sha256"]
