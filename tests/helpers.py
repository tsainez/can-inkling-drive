"""Synthetic scored rows for the statistics tests.

Shared here rather than imported across test modules: pytest only puts the
tests directory on sys.path, so `from tests.test_x import ...` passes under
`python -m pytest` and fails under `pytest`, which is what CI runs.
"""

from __future__ import annotations


def rows(
    correct_flags,
    *,
    model="m",
    condition="clean",
    n_options=4,
    valid=None,
    tokens=100,
    reasoning=True,
    offset=0,
    **extra,
):
    """Scored rows with the fields analyze/tables read.

    `offset` shifts question ids so two calls can describe disjoint question
    sets, which is how the pairing logic gets tested.
    """
    valid = valid if valid is not None else [True] * len(correct_flags)
    out = []
    for i, (c, v) in enumerate(zip(correct_flags, valid)):
        out.append({
            "question_id": f"q{i + offset}",
            "model_label": model,
            "condition": condition,
            "correct": c,
            "valid": v,
            "category": "perception",
            "n_options": n_options,
            "completion_tokens": tokens,
            "reasoning_tokens": tokens - 10 if reasoning else None,
            "reasoning_tokens_reported": reasoning,
            "prompt_tokens": 50,
            "latency_ms": 500.0,
            **extra,
        })
    return out


def flags(n_correct, n_total):
    """The first n_correct questions right, the rest wrong.

    Deterministic on purpose: two models built this way overlap on exactly the
    questions their accuracies imply, so McNemar's b and c are hand-checkable.
    """
    return [True] * n_correct + [False] * (n_total - n_correct)
