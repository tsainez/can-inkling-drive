"""Converter tests.

The converter is the most dangerous code in the repo: a bug here does not
crash, it silently produces a benchmark that measures the wrong thing. So the
properties that matter are asserted directly - class balance, position balance,
gold correctness, and determinism.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from idq.adapters.base import expected_chance
from idq.convert import (
    DriveLMConvertedAdapter,
    answer_is_usable,
    discover_templates,
    normalize_answer,
    normalize_template,
)

BALANCED_Q = "Is <c{i},CAM_FRONT,{x}.5,{y}.5> an object that the ego vehicle should consider?"
SKEWED_Q = "Is <c{i},CAM_FRONT,{x}.5,{y}.5> a traffic sign or a road barrier?"
TAGGED_ANSWER_Q = "What object would consider <c{i},CAM_FRONT,{x}.5,{y}.5> most relevant?"


def build_data(tmp_path, n_frames=400):
    """Synthetic DriveLM-shaped data with a balanced, a skewed and a
    tag-answer template."""
    scenes = {}
    for f in range(n_frames):
        fmt = dict(i=(f % 6) + 1, x=100 + f, y=200 + f)
        balanced_a = "Yes." if f % 2 == 0 else "No."
        skewed_a = "No." if f % 10 else "Yes."          # 90% No
        tagged_a = f"<c{(f % 6) + 1},CAM_BACK,1.5,2.5>." if f % 2 else "The ego vehicle."
        scenes[f"scene-{f // 4}"] = scenes.get(f"scene-{f // 4}", {
            "scene_description": "d", "key_frames": {}
        })
        scenes[f"scene-{f // 4}"]["key_frames"][f"frame-{f}"] = {
            "QA": {
                "planning": [{"Q": BALANCED_Q.format(**fmt), "A": balanced_a, "C": None}],
                "prediction": [
                    {"Q": SKEWED_Q.format(**fmt), "A": skewed_a, "C": None},
                    {"Q": TAGGED_ANSWER_Q.format(**fmt), "A": tagged_a, "C": None},
                ],
            },
            "image_paths": {"CAM_FRONT": f"samples/CAM_FRONT/{f}.jpg"},
        }
    path = tmp_path / "train.json"
    path.write_text(json.dumps(scenes), encoding="utf-8")
    return str(path)


def adapter(path, **kw):
    kw.setdefault("min_total", 50)
    return DriveLMConvertedAdapter(path=path, **kw)


# ------------------------------------------------------------- normalization

def test_template_normalization_collapses_object_tags():
    a = normalize_template("Is <c1,CAM_FRONT,100.5,200.5> relevant?")
    b = normalize_template("Is <c4,CAM_BACK,999.1,3.0> relevant?")
    assert a == b == "Is <OBJ> relevant?"


def test_answer_normalization_strips_trailing_period():
    assert normalize_answer("  Going  ahead. ") == "Going ahead"


def test_answers_containing_object_tags_are_rejected():
    """A tag-bearing answer cannot serve as a shared option across questions."""
    assert answer_is_usable("Moving", max_chars=60)
    assert not answer_is_usable("<c1,CAM_FRONT,1.0,2.0>", max_chars=60)
    assert not answer_is_usable("x" * 100, max_chars=60)
    assert not answer_is_usable("", max_chars=60)


# ------------------------------------------------------------------ discovery

def test_discovery_keeps_balanced_and_drops_skewed(tmp_path):
    data = json.loads(open(build_data(tmp_path), encoding="utf-8").read())
    specs = discover_templates(data, min_total=50, max_majority_share=0.75)

    templates = {s.template for s in specs}
    assert any("should consider" in t for t in templates), "balanced template dropped"
    # 90% "No" must be excluded: balancing it would discard most of the
    # majority class and leave a non-random slice.
    assert not any("traffic sign or a road barrier" in t for t in templates)
    # Tag-bearing answers make the option set non-interchangeable.
    assert not any("most relevant" in t for t in templates)


def test_discovery_respects_min_total(tmp_path):
    data = json.loads(open(build_data(tmp_path), encoding="utf-8").read())
    assert discover_templates(data, min_total=10_000) == []


def test_spec_reports_capacity_and_balance(tmp_path):
    data = json.loads(open(build_data(tmp_path), encoding="utf-8").read())
    spec = discover_templates(data, min_total=50, max_majority_share=0.75)[0]
    assert spec.n_options == 2
    assert spec.majority_share == pytest.approx(0.5, abs=0.01)
    assert spec.balanced_capacity == 400


# ----------------------------------------------------------------- conversion

def test_gold_classes_are_exactly_balanced(tmp_path):
    """Unbalanced gold classes let a majority-guesser beat chance."""
    a = adapter(build_data(tmp_path))
    a.load()
    for _, classes in a.stats["gold_class_balance"].items():
        assert len(set(classes.values())) == 1, f"classes not equal: {classes}"


def test_gold_positions_are_counterbalanced(tmp_path):
    """A model that always answers 'A' must score exactly chance."""
    a = adapter(build_data(tmp_path))
    qs = a.load()
    positions = Counter(q.gold_letter for q in qs)
    assert len(set(positions.values())) == 1, f"position bias: {positions}"

    always_a = sum(1 for q in qs if q.gold_letter == "A") / len(qs)
    assert always_a == pytest.approx(expected_chance(qs), abs=1e-9)


def test_gold_letter_points_at_the_true_answer(tmp_path):
    """The single most important assertion in the file."""
    a = adapter(build_data(tmp_path))
    for q in a.load():
        idx = q.letters.index(q.gold_letter)
        assert q.option_texts[idx] == q.raw["gold_text"]


def test_options_are_the_templates_full_answer_set(tmp_path):
    a = adapter(build_data(tmp_path))
    for q in a.load():
        assert sorted(q.option_texts) == sorted(set(q.option_texts)), "duplicate option"
        assert len(q.option_texts) == len(q.letters)
        assert all(t for t in q.option_texts)


def test_chance_is_one_over_k(tmp_path):
    a = adapter(build_data(tmp_path))
    qs = a.load()
    assert expected_chance(qs) == pytest.approx(0.5)


def test_conversion_is_deterministic(tmp_path):
    path = build_data(tmp_path)
    first = adapter(path).load()
    second = adapter(path).load()
    assert [q.question_id for q in first] == [q.question_id for q in second]
    assert [q.gold_letter for q in first] == [q.gold_letter for q in second]
    assert [q.option_texts for q in first] == [q.option_texts for q in second]


def test_different_seed_changes_sampling_not_correctness(tmp_path):
    path = build_data(tmp_path)
    a = adapter(path, n_per_template=100, seed=1).load()
    b = adapter(path, n_per_template=100, seed=2).load()
    assert {q.question_id for q in a} != {q.question_id for q in b}
    for qs in (a, b):
        for q in qs:
            assert q.option_texts[q.letters.index(q.gold_letter)] == q.raw["gold_text"]


def test_n_per_template_caps_output(tmp_path):
    a = adapter(build_data(tmp_path), n_per_template=100)
    qs = a.load()
    assert len(qs) == 100
    assert len(set(q.gold_letter for q in qs)) == 2


def test_stems_keep_object_tags_for_the_blind_conditions(tmp_path):
    """blind_notags strips them later; they must be present to be stripped."""
    qs = adapter(build_data(tmp_path)).load()
    assert any("CAM_FRONT" in q.stem for q in qs)
    assert all("<OBJ>" not in q.stem for q in qs)
    assert all("CAM_" not in q.stem_for("blind_notags") for q in qs)


def test_image_paths_are_carried_through(tmp_path):
    qs = adapter(build_data(tmp_path)).load()
    assert all(q.image_paths.get("CAM_FRONT") for q in qs)


def test_provenance_is_recorded_on_every_question(tmp_path):
    qs = adapter(build_data(tmp_path)).load()
    for q in qs:
        assert q.raw["source"] == "converted_from_free_form"
        assert q.raw["template_id"] and q.raw["template"]
