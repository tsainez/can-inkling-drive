"""Parser tests, including the two bugs specified up front."""

from __future__ import annotations

import pytest

from idq.adapters.base import normalize_gold, split_stem_and_options, strip_object_tags
from idq.parse import parse_answer_with_options

TRAP = (
    "What is the moving status of object <c1,CAM_FRONT,1088.3,497.5>? "
    "Please select the correct answer from the following options: "
    "A. Going ahead. B. Turning right. C. Back up. D. Stopped."
)


def test_option_starting_with_option_letter_survives():
    """'C. Back up' must not vanish: a naive [^A-D] class eats the B in 'Back'."""
    stem, letters, texts = split_stem_and_options(TRAP)
    assert letters == ["A", "B", "C", "D"]
    assert texts == ["Going ahead", "Turning right", "Back up", "Stopped"]
    assert "Back up" in texts
    assert stem.startswith("What is the moving status")
    assert "Please select" not in stem


def test_all_options_starting_with_letters():
    text = (
        "Pick one. A. Approach the curb. B. Brake hard. "
        "C. Continue ahead. D. Decelerate gently."
    )
    _, letters, texts = split_stem_and_options(text)
    assert letters == ["A", "B", "C", "D"]
    assert texts == [
        "Approach the curb", "Brake hard", "Continue ahead", "Decelerate gently",
    ]


def test_two_and_three_option_questions():
    _, letters, _ = split_stem_and_options("Q? A. Yes. B. No.")
    assert letters == ["A", "B"]
    _, letters, _ = split_stem_and_options("Q? A. Left. B. Right. C. Straight.")
    assert letters == ["A", "B", "C"]


def test_non_mcq_returns_empty():
    _, letters, texts = split_stem_and_options("What is the ego vehicle doing right now?")
    assert letters == [] and texts == []


def test_prose_with_stray_letters_is_not_mistaken_for_options():
    _, letters, _ = split_stem_and_options("Vitamin A. helps. Then B. something.")
    assert len(letters) < 2 or letters == ["A", "B"]


def test_strip_object_tags():
    out = strip_object_tags(TRAP)
    assert "<c1," not in out
    assert "CAM_FRONT" not in out
    assert "the referenced object" in out


@pytest.mark.parametrize(
    "answer,expected",
    [("C", "C"), ("C.", "C"), ("C)", "C"), (" c ", None), ("Back up", "C"), ("", None),
     ("Nonsense", None)],
)
def test_normalize_gold(answer, expected):
    letters = ["A", "B", "C", "D"]
    texts = ["Going ahead", "Turning right", "Back up", "Stopped"]
    assert normalize_gold(answer, letters, texts) == expected


LETTERS = ["A", "B", "C", "D"]
TEXTS = ["Going ahead", "Turning right", "Back up", "Stopped"]


@pytest.mark.parametrize(
    "text,letter,method",
    [
        ("Answer: C", "C", "answer_tag"),
        ("The final answer is B.", "B", "answer_tag"),
        ("C", "C", "sole_letter"),
        ("(D)", "D", "sole_letter"),
        ("\\boxed{A}", "A", "boxed"),
        ("Reasoning here.\n\nB", "B", "final_line"),
        ("<think>maybe A or B</think>\nAnswer: D", "D", "answer_tag"),
        ("I think the vehicle will Back up.", "C", "option_text"),
    ],
)
def test_extraction_methods(text, letter, method):
    p = parse_answer_with_options(text, LETTERS, TEXTS)
    assert (p.letter, p.method) == (letter, method)


def test_unparseable_is_reported_not_guessed():
    p = parse_answer_with_options("I cannot determine this.", LETTERS, TEXTS)
    assert p.letter is None
    assert p.method == "unparseable"
    assert p.is_valid is False


def test_ambiguous_output_is_flagged():
    p = parse_answer_with_options("It is either A or D, hard to say.", LETTERS, TEXTS)
    assert p.letter is None
    assert p.method == "ambiguous"
    assert set(p.candidates) == {"A", "D"}


def test_thinking_block_letters_do_not_leak_into_the_answer():
    """A letter mentioned mid-reasoning is not the answer."""
    p = parse_answer_with_options("<think>A looks right, no wait</think>\nAnswer: B", LETTERS, TEXTS)
    assert p.letter == "B"


def test_letter_outside_option_range_rejected():
    p = parse_answer_with_options("Answer: G", ["A", "B"], ["Yes", "No"])
    assert p.letter is None
