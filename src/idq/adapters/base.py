"""Benchmark adapter interface and shared MCQ text handling.

The adapter layer exists so the data source is swappable without a rewrite.
Anything downstream of here sees only Question objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Protocol

from ..hashing import hash_obj, short

# Object references DriveLM embeds in question text, e.g.
#   <c1,CAM_FRONT,1088.3,497.5>
# These name a camera and a pixel location. They leak scene information even
# with no image attached, which is why blind_notags exists as a condition.
OBJECT_TAG = re.compile(r"<[a-zA-Z]\d+,\s*CAM_[A-Z_]+,\s*[-\d.]+,\s*[-\d.]+>")

# Option boundary detector.
#
# The naive approach - splitting on a character class like [^A-D] - silently
# eats options whose text begins with an option letter: "C. Back up" loses its
# "B" and the option vanishes. This regex instead requires an option letter to
# sit at a real boundary and be followed by a delimiter and whitespace, so the
# "B" in "Back" can never match.
OPTION_START = re.compile(r"(?:(?<=^)|(?<=[\s(\[]))([A-H])[.):\]]\s+")

# Where an options list tends to begin in DriveLM phrasing.
OPTIONS_LEAD = re.compile(
    r"(please\s+select\s+the\s+correct\s+answer\s+from\s+the\s+following\s+options\s*:?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Question:
    question_id: str
    scene_id: str
    frame_id: str
    category: str
    stem: str
    letters: tuple[str, ...]
    option_texts: tuple[str, ...]
    gold_letter: str
    image_paths: dict[str, str] = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def n_options(self) -> int:
        return len(self.letters)

    def stem_for(self, condition: str) -> str:
        """Question stem as presented under a given condition."""
        if condition == "blind_notags":
            return strip_object_tags(self.stem)
        return self.stem


class BenchmarkAdapter(Protocol):
    name: str

    def load(self) -> list[Question]: ...


def strip_object_tags(text: str) -> str:
    """Replace object references with a neutral placeholder.

    We replace rather than delete so the sentence stays grammatical - deleting
    would change sentence structure and confound the comparison with a
    fluency effect rather than an information effect.
    """
    out = OBJECT_TAG.sub("the referenced object", text)
    return re.sub(r"\s{2,}", " ", out).strip()


def split_stem_and_options(text: str) -> tuple[str, list[str], list[str]]:
    """Split an inline-MCQ question into stem, letters, option texts.

    Returns ("", [], []) shaped output with empty option lists when the text
    does not parse as multiple choice, so callers can filter rather than
    except.
    """
    lead = OPTIONS_LEAD.search(text)
    search_from = lead.end() if lead else 0

    matches = list(OPTION_START.finditer(text, search_from))
    if len(matches) < 2:
        return text.strip(), [], []

    # Option letters must be consecutive from A with no repeats. Anything else
    # is a false positive from prose, not an options list.
    letters = [m.group(1) for m in matches]
    expected = [chr(ord("A") + i) for i in range(len(letters))]
    if letters != expected:
        # Try the longest consecutive-from-A prefix.
        keep = 0
        for i, ltr in enumerate(letters):
            if ltr == chr(ord("A") + i):
                keep = i + 1
            else:
                break
        if keep < 2:
            return text.strip(), [], []
        matches = matches[:keep]
        letters = letters[:keep]

    stem_end = lead.start() if lead else matches[0].start()
    stem = text[:stem_end].strip()

    texts: list[str] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        opt = text[m.end():end].strip()
        opt = opt.rstrip(" .;,")
        texts.append(opt)

    if any(not t for t in texts):
        return text.strip(), [], []

    return stem, letters, texts


def normalize_gold(answer: str, letters: list[str], option_texts: list[str]) -> str | None:
    """Resolve a gold answer to a letter, or None if it cannot be resolved.

    Unresolvable golds are counted and reported by the adapter rather than
    silently dropped - a benchmark whose answers we cannot read is a finding,
    not a nuisance.
    """
    if not answer:
        return None
    a = answer.strip()

    m = re.fullmatch(r"([A-H])\s*[.):\]]?\s*", a)
    if m and m.group(1) in letters:
        return m.group(1)

    m = OPTION_START.match(a)
    if m and m.group(1) in letters:
        return m.group(1)

    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    target = norm(a)
    if target:
        hits = [ltr for ltr, txt in zip(letters, option_texts) if norm(txt) == target]
        if len(hits) == 1:
            return hits[0]

    return None


def make_question_id(*parts: object) -> str:
    """Stable across runs and machines; independent of iteration order."""
    return short(hash_obj([str(p) for p in parts]), 16)


def summarize(questions: Iterable[Question]) -> dict:
    qs = list(questions)
    by_cat: dict[str, int] = {}
    by_nopt: dict[int, int] = {}
    for q in qs:
        by_cat[q.category] = by_cat.get(q.category, 0) + 1
        by_nopt[q.n_options] = by_nopt.get(q.n_options, 0) + 1
    return {
        "n_questions": len(qs),
        "by_category": dict(sorted(by_cat.items())),
        "by_n_options": dict(sorted(by_nopt.items())),
        "expected_chance": expected_chance(qs),
    }


def expected_chance(questions: Iterable[Question]) -> float:
    """Chance accuracy is mean(1/n_options), not 0.25.

    If any question has 2 or 3 options, hardcoding 0.25 would make the smoke
    test's sanity check itself incorrect.
    """
    qs = list(questions)
    if not qs:
        return 0.0
    return sum(1.0 / q.n_options for q in qs) / len(qs)
