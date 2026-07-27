"""Synthetic adapter for offline testing.

Deliberately includes the shapes that break naive parsers:
  - options whose text starts with an option letter ("C. Back up")
  - varying option counts (2, 3, 4), so chance accuracy is not 0.25
  - DriveLM-style object tags in the stem

Every generated question has distinct text. That matters: a fixture with only
a handful of unique prompts makes any prompt-seeded mock deterministic across
thousands of questions, which would quietly turn the chance test into a test of
four random draws.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .base import Question, make_question_id, split_stem_and_options

CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)

# (category, stem template, option pool). Option order is shuffled per question,
# so options whose text begins with an option letter land in every position.
_FAMILIES = [
    (
        "perception",
        "What is the moving status of object <c{cid},{cam},{x},{y}> at {place}?",
        ["Going ahead", "Turning right", "Back up", "Stopped"],
    ),
    (
        "prediction",
        "Will the pedestrian at <c{cid},{cam},{x},{y}> near {place} enter the roadway?",
        ["Yes", "No"],
    ),
    (
        "planning",
        "Given the object at <c{cid},{cam},{x},{y}>, what should the ego vehicle "
        "do on approach to {place}?",
        [
            "Accelerate through the intersection",
            "Brake and yield",
            "Change lanes to the left",
        ],
    ),
    (
        "behavior",
        "What is the ego vehicle doing relative to <c{cid},{cam},{x},{y}> at {place}?",
        [
            "Driving with normal speed",
            "Decelerating",
            "Backing up slowly",
            "Steering to the right",
        ],
    ),
]

_PLACES = (
    "a four-way stop", "a signalised intersection", "a merge ramp",
    "a construction zone", "an unprotected left turn", "a crosswalk",
    "a roundabout", "a parking lot exit",
)

_LEAD = "Please select the correct answer from the following options: "


@dataclass
class FixtureAdapter:
    n: int = 400
    seed: int = 1234
    name: str = "fixture"

    def load(self) -> list[Question]:
        rng = random.Random(self.seed)
        out: list[Question] = []

        for i in range(self.n):
            category, stem_tpl, pool = _FAMILIES[i % len(_FAMILIES)]

            stem = stem_tpl.format(
                cid=rng.randint(1, 9),
                cam=rng.choice(CAMERAS),
                x=round(rng.uniform(0.0, 1600.0), 1),
                y=round(rng.uniform(0.0, 900.0), 1),
                place=rng.choice(_PLACES),
            )

            options = list(pool)
            rng.shuffle(options)
            letters = [chr(ord("A") + j) for j in range(len(options))]
            body = " ".join(f"{ltr}. {txt}." for ltr, txt in zip(letters, options))
            full = f"{stem} {_LEAD}{body}"

            parsed_stem, parsed_letters, parsed_texts = split_stem_and_options(full)
            if parsed_letters != letters or list(parsed_texts) != options:
                raise AssertionError(  # pragma: no cover - fixture must round-trip
                    f"fixture failed to round-trip through the parser: {full!r}"
                )

            out.append(
                Question(
                    question_id=make_question_id("fixture", self.seed, i),
                    scene_id=f"scene_{i // 8:04d}",
                    frame_id=f"frame_{i:05d}",
                    category=category,
                    stem=parsed_stem,
                    letters=tuple(parsed_letters),
                    option_texts=tuple(parsed_texts),
                    gold_letter=rng.choice(parsed_letters),
                    image_paths={"CAM_FRONT": f"synthetic/{i:05d}.jpg"},
                    raw={"synthetic": True, "question_text": full},
                )
            )
        return out
