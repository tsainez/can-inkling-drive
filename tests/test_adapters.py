"""Adapter tests, including a DriveLM-shaped fixture parsed end to end."""

from __future__ import annotations

import json

from idq.adapters import DriveLMAdapter, FixtureAdapter, expected_chance, summarize

DRIVELM_FIXTURE = {
    "scene-0001": {
        "scene_description": "urban intersection",
        "key_frames": {
            "frame-a": {
                "QA": {
                    "perception": [
                        {
                            "Q": "What is the moving status of object "
                                 "<c1,CAM_FRONT,1088.3,497.5>? Please select the correct "
                                 "answer from the following options: A. Going ahead. "
                                 "B. Turning right. C. Back up. D. Stopped.",
                            "A": "C",
                        },
                        {
                            "Q": "Describe the scene in front of the ego vehicle.",
                            "A": "A busy intersection with pedestrians.",
                        },
                    ],
                    "planning": [
                        {
                            "Q": "What should the ego vehicle do? Please select the correct "
                                 "answer from the following options: A. Brake. B. Accelerate.",
                            "A": "A",
                        },
                        {
                            "Q": "What should the ego vehicle do? Please select the correct "
                                 "answer from the following options: A. Brake. B. Accelerate.",
                            "A": "not a letter or an option",
                        },
                    ],
                },
                "image_paths": {
                    "CAM_FRONT": "samples/CAM_FRONT/x.jpg",
                    "CAM_BACK": "samples/CAM_BACK/x.jpg",
                },
            }
        },
    }
}


def test_drivelm_adapter_filters_to_mcq_and_reports_what_it_dropped(tmp_path):
    path = tmp_path / "drivelm.json"
    path.write_text(json.dumps(DRIVELM_FIXTURE), encoding="utf-8")

    adapter = DriveLMAdapter(path=str(path))
    qs = adapter.load()

    assert len(qs) == 2
    assert adapter.stats["total_qa"] == 4
    assert adapter.stats["not_multiple_choice"] == 1
    # An unreadable gold answer is counted, not silently dropped.
    assert adapter.stats["unresolvable_gold"] == 1
    assert 0.0 < adapter.stats["mcq_share_of_all_qa"] <= 1.0

    # The MCQ subset is not a random sample; both distributions are reported so
    # the paper can say which slice its conclusions cover.
    assert adapter.stats["all_qa_by_category"] == {"perception": 2, "planning": 2}
    assert adapter.stats["mcq_by_category"] == {"perception": 1, "planning": 1}

    perception = [q for q in qs if q.category == "perception"][0]
    assert perception.gold_letter == "C"
    assert perception.option_texts == ("Going ahead", "Turning right", "Back up", "Stopped")
    assert perception.image_paths["CAM_FRONT"].endswith("x.jpg")


def test_question_ids_are_stable_across_loads(tmp_path):
    path = tmp_path / "drivelm.json"
    path.write_text(json.dumps(DRIVELM_FIXTURE), encoding="utf-8")
    a = [q.question_id for q in DriveLMAdapter(path=str(path)).load()]
    b = [q.question_id for q in DriveLMAdapter(path=str(path)).load()]
    assert a == b and len(set(a)) == len(a)


def test_malformed_input_does_not_crash(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"scene": {"key_frames": {"f": {}}}}), encoding="utf-8")
    assert DriveLMAdapter(path=str(path)).load() == []


def test_expected_chance_is_not_hardcoded_quarter():
    """Fixture mixes 2, 3 and 4 option questions, so chance is above 0.25."""
    qs = FixtureAdapter(n=400).load()
    chance = expected_chance(qs)
    assert 0.25 < chance < 0.50
    assert chance == expected_chance(qs)

    counts = summarize(qs)["by_n_options"]
    assert set(counts) == {2, 3, 4}


def test_blind_notags_strips_tags_from_the_stem():
    q = FixtureAdapter(n=4).load()[0]
    assert "CAM_FRONT" in q.stem_for("blind_tags")
    assert "CAM_FRONT" not in q.stem_for("blind_notags")
    assert "clean" and "CAM_FRONT" in q.stem_for("clean")
