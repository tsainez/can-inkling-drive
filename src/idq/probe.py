"""Raw-data probe. Diagnoses why an adapter kept fewer questions than expected.

Deliberately dumb: it does no parsing of its own beyond counting surface
patterns, so it can tell you whether the data lacks multiple-choice questions
or the adapter simply failed to recognise them. Those two look identical from
the adapter's summary and have completely different fixes.

Output is bounded so it can be pasted into a conversation.
"""

from __future__ import annotations

import json
import re
from collections import Counter

# Surface patterns that would indicate a multiple-choice question, tested
# independently of the adapter's own splitter.
PATTERNS = {
    "phrase_select_correct": re.compile(r"select the correct answer", re.I),
    "phrase_following_options": re.compile(r"following options", re.I),
    "word_options": re.compile(r"\boptions?\b", re.I),
    "letter_dot_A": re.compile(r"(?:^|\s)A\.\s"),
    "letter_paren_A": re.compile(r"(?:^|\s)\(?A\)\s"),
    "letter_colon_A": re.compile(r"(?:^|\s)A:\s"),
    "two_letters_dot": re.compile(r"(?:^|\s)A\.\s.*(?:^|\s)B\.\s", re.S),
    "newline_separated": re.compile(r"\n\s*A[.):]"),
}

GOLD_PATTERNS = {
    "answer_is_bare_letter": re.compile(r"^\s*[A-H]\s*[.):\]]?\s*$"),
    "answer_starts_with_letter": re.compile(r"^\s*[A-H]\s*[.):\]]\s+\S"),
}


def probe(path: str, *, n_examples: int = 3, max_chars: int = 420) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    pattern_counts: Counter = Counter()
    gold_counts: Counter = Counter()
    per_category: dict[str, Counter] = {}
    examples: dict[str, list] = {}
    mcq_examples: list = []
    q_len = Counter()

    top_keys = list(data.keys())[:1] if isinstance(data, dict) else []
    schema_sample = {}
    if top_keys:
        scene = data[top_keys[0]]
        schema_sample["scene_keys"] = sorted(scene.keys()) if isinstance(scene, dict) else []
        kf = (scene or {}).get("key_frames") if isinstance(scene, dict) else None
        if isinstance(kf, dict) and kf:
            frame = next(iter(kf.values()))
            schema_sample["frame_keys"] = sorted(frame.keys()) if isinstance(frame, dict) else []
            qa = (frame or {}).get("QA")
            if isinstance(qa, dict):
                schema_sample["qa_categories"] = sorted(qa.keys())
                for cat, items in qa.items():
                    if isinstance(items, list) and items and isinstance(items[0], dict):
                        schema_sample[f"{cat}_item_keys"] = sorted(items[0].keys())
                        break

    for scene in _values(data):
        for frame in _values(_get(scene, "key_frames")):
            qa = _get(frame, "QA") or {}
            for category, items in _items(qa):
                per_category.setdefault(category, Counter())
                for item in items or []:
                    q = str(_get(item, "Q") or "")
                    a = str(_get(item, "A") or "")

                    q_len[_bucket(len(q))] += 1

                    hit_any = False
                    for name, pat in PATTERNS.items():
                        if pat.search(q):
                            pattern_counts[name] += 1
                            per_category[category][name] += 1
                            hit_any = True

                    for name, pat in GOLD_PATTERNS.items():
                        if pat.match(a):
                            gold_counts[name] += 1

                    if hit_any and len(mcq_examples) < n_examples * 2:
                        mcq_examples.append(
                            {"category": category, "Q": q[:max_chars], "A": a[:200]}
                        )

                    bucket = examples.setdefault(category, [])
                    if len(bucket) < n_examples:
                        bucket.append({"Q": q[:max_chars], "A": a[:200]})

    return {
        "schema_sample": schema_sample,
        "question_length_buckets": dict(sorted(q_len.items())),
        "surface_pattern_counts": dict(pattern_counts),
        "gold_answer_shape_counts": dict(gold_counts),
        "surface_patterns_by_category": {
            k: dict(v) for k, v in sorted(per_category.items()) if v
        },
        "examples_matching_any_mcq_pattern": mcq_examples,
        "examples_by_category": examples,
    }


def _bucket(n: int) -> str:
    for edge in (50, 100, 200, 400, 800):
        if n < edge:
            return f"<{edge}"
    return ">=800"


def _values(obj):
    if isinstance(obj, dict):
        return list(obj.values())
    if isinstance(obj, list):
        return obj
    return []


def _items(obj):
    return list(obj.items()) if isinstance(obj, dict) else []


def _get(obj, key):
    return obj.get(key) if isinstance(obj, dict) else None
