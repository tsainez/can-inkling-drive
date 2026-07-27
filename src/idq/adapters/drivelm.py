"""DriveLM adapter.

Expected shape (DriveLM-nuScenes train split):

    { scene_token: {
        "scene_description": str,
        "key_frames": { frame_token: {
            "QA": { "perception": [{"Q": ..., "A": ...}, ...],
                    "prediction": [...], "planning": [...], "behavior": [...] },
            "image_paths": { "CAM_FRONT": "...", ... } } } } }

Written defensively: DriveLM releases vary in which keys are present, and the
val/test splits may ship without gold answers because they back a leaderboard.
Anything we cannot parse is counted and reported, never silently dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .base import Question, make_question_id, normalize_gold, split_stem_and_options

QA_CATEGORIES = ("perception", "prediction", "planning", "behavior")


@dataclass
class DriveLMAdapter:
    path: str
    name: str = "drivelm"
    stats: dict = field(default_factory=dict)

    def load(self) -> list[Question]:
        with open(self.path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        questions: list[Question] = []
        counts = {
            "total_qa": 0,
            "not_multiple_choice": 0,
            "unresolvable_gold": 0,
            "missing_image_paths": 0,
            "kept": 0,
        }
        by_cat_total: dict[str, int] = {}

        for scene_id, scene in _as_items(data):
            key_frames = _get(scene, "key_frames") or {}
            for frame_id, frame in _as_items(key_frames):
                qa = _get(frame, "QA") or {}
                image_paths = _get(frame, "image_paths") or {}
                if not image_paths:
                    counts["missing_image_paths"] += 1

                for category in QA_CATEGORIES:
                    items = _get(qa, category) or []
                    for idx, item in enumerate(items):
                        counts["total_qa"] += 1
                        by_cat_total[category] = by_cat_total.get(category, 0) + 1

                        q_text = str(_get(item, "Q") or "")
                        a_text = str(_get(item, "A") or "")

                        stem, letters, option_texts = split_stem_and_options(q_text)
                        if len(letters) < 2:
                            counts["not_multiple_choice"] += 1
                            continue

                        gold = normalize_gold(a_text, letters, option_texts)
                        if gold is None:
                            counts["unresolvable_gold"] += 1
                            continue

                        questions.append(
                            Question(
                                question_id=make_question_id(
                                    "drivelm", scene_id, frame_id, category, idx
                                ),
                                scene_id=str(scene_id),
                                frame_id=str(frame_id),
                                category=category,
                                stem=stem,
                                letters=tuple(letters),
                                option_texts=tuple(option_texts),
                                gold_letter=gold,
                                image_paths={str(k): str(v) for k, v in _as_items(image_paths)},
                                raw={"Q": q_text, "A": a_text},
                            )
                        )
                        counts["kept"] += 1

        mcq_share = counts["kept"] / counts["total_qa"] if counts["total_qa"] else 0.0
        kept_by_cat: dict[str, int] = {}
        for q in questions:
            kept_by_cat[q.category] = kept_by_cat.get(q.category, 0) + 1

        self.stats = {
            **counts,
            "mcq_share_of_all_qa": round(mcq_share, 4),
            # The MCQ subset is not a random sample of DriveLM. Reporting both
            # distributions lets the paper state which slice the conclusions
            # are actually about, rather than have a reviewer state it first.
            "all_qa_by_category": dict(sorted(by_cat_total.items())),
            "mcq_by_category": dict(sorted(kept_by_cat.items())),
        }
        return questions


def _as_items(obj):
    if isinstance(obj, dict):
        return list(obj.items())
    if isinstance(obj, list):
        return list(enumerate(obj))
    return []


def _get(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return None
