"""Answer extraction.

Reports how the letter was extracted, not just what it was. Extraction method
is a first-class field: if 30% of a model's answers only parse via loose
fallbacks, that is a result about the model's instruction following, and it
belongs in the paper rather than being smoothed away.

Unparseable output is a reported metric, never silently scored wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

# Ordered most-trustworthy first. The first method that fires wins.
METHODS = (
    "answer_tag",      # "Answer: C" - what the prompt asks for
    "sole_letter",     # entire reply is "C" or "C."
    "boxed",           # \boxed{C}, common in reasoning-model output
    "final_line",      # last non-empty line is a bare letter
    "option_text",     # reply restates one option's text verbatim
    "first_letter",    # loosest: first standalone option letter anywhere
    "ambiguous",       # multiple distinct candidates, no way to choose
    "unparseable",     # nothing found
)

_ANSWER_TAG = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is)?\s*[:\-]?\s*\(?\s*([A-H])\s*\)?\s*[.)\]]?",
    re.IGNORECASE,
)
_BOXED = re.compile(r"\\boxed\{\s*([A-H])\s*\}")
_SOLE = re.compile(r"^\s*\(?\s*([A-H])\s*\)?\s*[.):\]]?\s*$")
_STANDALONE = re.compile(r"(?:(?<=^)|(?<=[\s(\[*]))([A-H])(?=[\s.):\]*,]|$)")

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ParsedAnswer:
    letter: str | None
    method: str
    candidates: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return self.letter is not None


def strip_thinking(text: str) -> str:
    """Remove delimited reasoning blocks before answer extraction.

    Some providers return the chain of thought inline in content. A letter
    mentioned mid-reasoning is not the answer, so extraction runs on the
    post-thinking text. The raw text is still stored in the cache untouched.
    """
    return _THINK_BLOCK.sub(" ", text)


def count_thinking_chars(text: str) -> int:
    """Fallback signal for RQ2 when a provider does not report reasoning tokens."""
    return sum(len(m.group(0)) for m in _THINK_BLOCK.finditer(text))


def parse_answer(text: str, letters: Sequence[str]) -> ParsedAnswer:
    if text is None:
        return ParsedAnswer(None, "unparseable")

    valid = set(letters)
    body = strip_thinking(text).strip()
    if not body:
        return ParsedAnswer(None, "unparseable")

    tail = body[-400:]

    for method, pattern, scope in (
        ("answer_tag", _ANSWER_TAG, tail),
        ("boxed", _BOXED, body),
    ):
        hits = [m.group(1).upper() for m in pattern.finditer(scope)]
        hits = [h for h in hits if h in valid]
        if hits:
            # Last occurrence: models often restate then correct themselves.
            return ParsedAnswer(hits[-1], method)

    m = _SOLE.match(body)
    if m and m.group(1).upper() in valid:
        return ParsedAnswer(m.group(1).upper(), "sole_letter")

    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if lines:
        m = _SOLE.match(lines[-1])
        if m and m.group(1).upper() in valid:
            return ParsedAnswer(m.group(1).upper(), "final_line")

    return ParsedAnswer(None, "unparseable")


def parse_answer_with_options(
    text: str, letters: Sequence[str], option_texts: Sequence[str]
) -> ParsedAnswer:
    """Full parser: strict methods, then option-text matching, then loose fallback."""
    strict = parse_answer(text, letters)
    if strict.is_valid:
        return strict

    if text is None:
        return ParsedAnswer(None, "unparseable")

    body = strip_thinking(text).strip()
    if not body:
        return ParsedAnswer(None, "unparseable")

    valid = set(letters)

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()

    nbody = norm(body)
    text_hits = []
    for ltr, opt in zip(letters, option_texts):
        nopt = norm(opt)
        if nopt and nopt in nbody:
            text_hits.append(ltr)
    if len(text_hits) == 1:
        return ParsedAnswer(text_hits[0], "option_text")
    if len(text_hits) > 1:
        return ParsedAnswer(None, "ambiguous", tuple(sorted(set(text_hits))))

    standalone = [m.group(1).upper() for m in _STANDALONE.finditer(body)]
    standalone = [h for h in standalone if h in valid]
    distinct = sorted(set(standalone))
    if len(distinct) == 1:
        return ParsedAnswer(distinct[0], "first_letter")
    if len(distinct) > 1:
        return ParsedAnswer(None, "ambiguous", tuple(distinct))

    return ParsedAnswer(None, "unparseable")
