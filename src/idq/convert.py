"""Convert DriveLM's free-form train QA into balanced multiple choice.

Why this exists
---------------
DriveLM ships answers in train and multiple-choice formatting in val, and never
both in the same file. Val's answers are empty strings because val backs a
public leaderboard. Objective offline scoring therefore requires constructing
the MCQ form ourselves.

This is not a workaround. DriveLM's own repository contains extract_data.py and
convert_data.py, which generate the val/test multiple-choice format from the
same free-form annotations. We perform the equivalent transformation on the
split where answers are public.

Three properties make the result defensible:

1. **No generated text.** Every option is a verbatim human-written answer that
   annotators gave to that same question template. We select options; we never
   write them. There is no generator whose fingerprints a model could exploit
   (cf. Gururangan et al. 2018 on annotation artifacts).

2. **Balanced gold classes.** Raw DriveLM answer distributions are extreme -
   89% "No", 92% "Going ahead", 99.8% "Low". Sampling equal numbers per gold
   class makes chance exactly 1/k, so a model that always guesses the majority
   class gains nothing. Without this, a blind model scores ~90% on priors alone
   and the grounding gap collapses for reasons that have nothing to do with
   grounding.

3. **Counterbalanced option positions.** The gold letter is distributed evenly
   across positions rather than randomly, so a model with a position bias
   ("always answer A") scores exactly chance instead of getting lucky.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .adapters.base import Question, make_question_id
from .hashing import hash_obj

OBJECT_TAG = re.compile(r"<[a-zA-Z]\d+,\s*CAM_[A-Z_]+,[^>]*>")
QA_CATEGORIES = ("perception", "prediction", "planning", "behavior")

TEMPLATE_PLACEHOLDER = "<OBJ>"


@dataclass(frozen=True)
class TemplateSpec:
    """A question template whose answers form a small closed set."""

    category: str
    template: str
    options: tuple[str, ...]
    counts: tuple[int, ...]
    n_total: int

    @property
    def n_options(self) -> int:
        return len(self.options)

    @property
    def majority_share(self) -> float:
        return max(self.counts) / self.n_total if self.n_total else 1.0

    @property
    def balanced_capacity(self) -> int:
        """Questions available once classes are equalized."""
        return min(self.counts) * self.n_options

    @property
    def template_id(self) -> str:
        return hash_obj([self.category, self.template])[:12]

    def as_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "category": self.category,
            "template": self.template,
            "options": list(self.options),
            "counts": list(self.counts),
            "n_total": self.n_total,
            "majority_share": round(self.majority_share, 4),
            "balanced_capacity": self.balanced_capacity,
        }


def normalize_template(text: str) -> str:
    return re.sub(r"\s+", " ", OBJECT_TAG.sub(TEMPLATE_PLACEHOLDER, text or "")).strip()


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).rstrip(" .").strip()


def answer_is_usable(answer: str, *, max_chars: int) -> bool:
    """Reject answers that cannot serve as a shared option across questions.

    An answer containing an object tag is specific to one question - it could
    not appear as a distractor elsewhere without being either nonsensical or a
    giveaway. Rejecting them keeps every option interchangeable within a
    template, which is what makes the distractors honest.
    """
    if not answer:
        return False
    if OBJECT_TAG.search(answer):
        return False
    return len(answer) <= max_chars


def discover_templates(
    data: dict,
    *,
    min_total: int = 5000,
    max_options: int = 6,
    max_majority_share: float = 0.75,
    max_answer_chars: int = 60,
) -> list[TemplateSpec]:
    """Find templates suitable for conversion.

    max_majority_share is the load-bearing filter. A template answered "No" 89%
    of the time can be balanced by discarding most of the majority class, but
    the survivors are then a heavily non-random slice of the original, and any
    claim about the benchmark stops being a claim about the benchmark. We
    exclude such templates rather than salvage them.
    """
    answers: dict[tuple[str, str], Counter] = defaultdict(Counter)

    for scene in _values(data):
        for frame in _values(_get(scene, "key_frames")):
            qa = _get(frame, "QA") or {}
            for category in QA_CATEGORIES:
                for item in _get(qa, category) or []:
                    q = normalize_template(str(_get(item, "Q") or ""))
                    a = normalize_answer(str(_get(item, "A") or ""))
                    if q and a:
                        answers[(category, q)][a] += 1

    specs: list[TemplateSpec] = []
    for (category, template), counter in answers.items():
        usable = {
            a: c for a, c in counter.items()
            if answer_is_usable(a, max_chars=max_answer_chars)
        }
        # Every observed answer must be usable. If some are not, the closed set
        # is not really closed and a "distractor" might actually be correct.
        if len(usable) != len(counter):
            continue
        if not (2 <= len(usable) <= max_options):
            continue

        total = sum(usable.values())
        if total < min_total:
            continue

        ordered = sorted(usable.items(), key=lambda kv: (-kv[1], kv[0]))
        spec = TemplateSpec(
            category=category,
            template=template,
            options=tuple(a for a, _ in ordered),
            counts=tuple(c for _, c in ordered),
            n_total=total,
        )
        if spec.majority_share > max_majority_share:
            continue
        specs.append(spec)

    specs.sort(key=lambda s: -s.balanced_capacity)
    return specs


@dataclass
class DriveLMConvertedAdapter:
    """Adapter producing balanced MCQs from DriveLM's free-form train answers."""

    path: str
    name: str = "drivelm_converted"
    seed: int = 20260731
    n_per_template: int | None = None
    min_total: int = 5000
    max_options: int = 6
    max_majority_share: float = 0.75
    template_ids: tuple[str, ...] = ()
    stats: dict = field(default_factory=dict)

    def load(self) -> list[Question]:
        with open(self.path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        specs = discover_templates(
            data,
            min_total=self.min_total,
            max_options=self.max_options,
            max_majority_share=self.max_majority_share,
        )
        if self.template_ids:
            wanted = set(self.template_ids)
            specs = [s for s in specs if s.template_id in wanted]

        by_key = {(s.category, s.template): s for s in specs}

        # Gather candidates, bucketed by (template, gold answer), so classes can
        # be equalized afterwards.
        buckets: dict[tuple[str, str, str], list] = defaultdict(list)
        for scene_id, scene in _items(data):
            for frame_id, frame in _items(_get(scene, "key_frames")):
                qa = _get(frame, "QA") or {}
                images = _get(frame, "image_paths") or {}
                for category in QA_CATEGORIES:
                    for idx, item in enumerate(_get(qa, category) or []):
                        raw_q = str(_get(item, "Q") or "")
                        gold = normalize_answer(str(_get(item, "A") or ""))
                        key = (category, normalize_template(raw_q))
                        spec = by_key.get(key)
                        if spec is None or gold not in spec.options:
                            continue
                        buckets[(spec.template_id, gold, category)].append(
                            (scene_id, frame_id, idx, raw_q, images, spec)
                        )

        # Deterministic ordering before sampling, so the study is reproducible
        # from the source file alone.
        questions: list[Question] = []
        per_template: dict[str, int] = {}
        class_counts: dict[str, Counter] = defaultdict(Counter)
        position_counts: Counter = Counter()

        for spec in specs:
            classes = [
                (opt, buckets.get((spec.template_id, opt, spec.category), []))
                for opt in spec.options
            ]
            if any(not items for _, items in classes):
                continue

            take = min(len(items) for _, items in classes)
            if self.n_per_template:
                take = min(take, self.n_per_template // spec.n_options)
            if take < 1:
                continue

            for opt, items in classes:
                ordered = sorted(items, key=lambda r: _row_sort_key(r))
                chosen = _deterministic_sample(ordered, take, self.seed, spec.template_id, opt)

                for rank, (scene_id, frame_id, idx, raw_q, images, sp) in enumerate(chosen):
                    letters, texts, gold_letter = _counterbalanced_options(
                        sp.options, opt, rank
                    )
                    stem = re.sub(r"\s+", " ", raw_q).strip()
                    questions.append(
                        Question(
                            question_id=make_question_id(
                                "drivelm_conv", scene_id, frame_id, sp.category, idx
                            ),
                            scene_id=str(scene_id),
                            frame_id=str(frame_id),
                            category=sp.category,
                            stem=stem,
                            letters=tuple(letters),
                            option_texts=tuple(texts),
                            gold_letter=gold_letter,
                            image_paths={str(k): str(v) for k, v in _items(images)},
                            raw={
                                "template_id": sp.template_id,
                                "template": sp.template,
                                "gold_text": opt,
                                "source": "converted_from_free_form",
                            },
                        )
                    )
                    per_template[sp.template_id] = per_template.get(sp.template_id, 0) + 1
                    class_counts[sp.template_id][opt] += 1
                    position_counts[gold_letter] += 1

        questions.sort(key=lambda q: q.question_id)

        self.stats = {
            "templates_discovered": len(specs),
            "templates_used": len(per_template),
            "questions_built": len(questions),
            "questions_per_template": per_template,
            "gold_class_balance": {k: dict(v) for k, v in class_counts.items()},
            # If this is not near-uniform, a model with a position bias would
            # score above chance for free.
            "gold_position_balance": dict(sorted(position_counts.items())),
            "template_specs": [s.as_dict() for s in specs],
        }
        return questions


def _counterbalanced_options(options: tuple[str, ...], gold: str, rank: int):
    """Place the gold answer at position rank % k; rotate the rest around it.

    Rotation rather than shuffling means option order is a deterministic
    function of rank, so gold lands in every position exactly equally often
    within a class rather than approximately equally.
    """
    k = len(options)
    others = [o for o in options if o != gold]
    gold_pos = rank % k
    texts: list[str] = []
    it = iter(others)
    for pos in range(k):
        texts.append(gold if pos == gold_pos else next(it))
    letters = [chr(ord("A") + i) for i in range(k)]
    return letters, texts, letters[gold_pos]


def _deterministic_sample(ordered: list, take: int, seed: int, template_id: str, opt: str):
    """Stable pseudo-random subset: sort by a hash of (seed, item key) and slice.

    Avoids random.shuffle so the selection depends only on values in the source
    file, not on iteration order or Python version.
    """
    if take >= len(ordered):
        return ordered
    keyed = sorted(
        ordered,
        key=lambda r: hash_obj([seed, template_id, opt, _row_sort_key(r)]),
    )
    return keyed[:take]


def _row_sort_key(row) -> list:
    scene_id, frame_id, idx = row[0], row[1], row[2]
    return [str(scene_id), str(frame_id), int(idx)]


def _values(obj):
    if isinstance(obj, dict):
        return list(obj.values())
    if isinstance(obj, list):
        return obj
    return []


def _items(obj):
    if isinstance(obj, dict):
        return list(obj.items())
    if isinstance(obj, list):
        return list(enumerate(obj))
    return []


def _get(obj, key):
    return obj.get(key) if isinstance(obj, dict) else None
