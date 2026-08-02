"""Scoring. Reads the cache, writes scored rows. Never opens a network socket.

Because collection and scoring are separate, a bug here is free to fix: delete
the scored output, change the code, run again.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

from .adapters.base import Question
from .parse import count_thinking_chars, parse_answer_with_options


def score_records(records: Iterable[dict], questions: Iterable[Question]) -> list[dict]:
    qmap = {q.question_id: q for q in questions}
    rows: list[dict] = []

    for rec in records:
        q = qmap.get(rec.get("question_id"))
        letters = list(rec.get("letters") or (q.letters if q else []))
        option_texts = list(q.option_texts) if q else []
        gold = rec.get("gold_letter") or (q.gold_letter if q else None)

        if rec.get("status") != "success":
            rows.append(
                _row(rec, gold, predicted=None, method="provider_error",
                     correct=False, valid=False)
            )
            continue

        parsed = parse_answer_with_options(rec.get("response_text") or "", letters, option_texts)
        rows.append(
            _row(
                rec, gold,
                predicted=parsed.letter,
                method=parsed.method,
                correct=bool(parsed.letter is not None and parsed.letter == gold),
                valid=parsed.is_valid,
                candidates=list(parsed.candidates),
            )
        )
    return rows


def _row(rec, gold, *, predicted, method, correct, valid, candidates=None) -> dict:
    usage = rec.get("usage") or {}
    text = rec.get("response_text") or ""
    decode = rec.get("decode") or {}
    return {
        "cache_key": rec.get("cache_key"),
        "question_id": rec.get("question_id"),
        "scene_id": rec.get("scene_id"),
        "category": rec.get("category"),
        "n_options": rec.get("n_options"),
        "model_label": rec.get("model_label"),
        "model_string": rec.get("model_string"),
        "served_by": rec.get("served_by"),
        # Quantization, thinking effort and the harness commit are carried
        # through to the scored row because the paper's serving-configuration
        # and reproducibility tables are built from these rows, not from the
        # cache. Losing them here would mean re-reading the cache to write an
        # appendix, or worse, reporting "unknown" for something we recorded.
        "quantization": rec.get("quantization", "unknown"),
        "thinking_effort": decode.get("thinking_effort"),
        "max_tokens": decode.get("max_tokens"),
        "temperature": decode.get("temperature"),
        "reasoning_format": decode.get("reasoning_format", "reasoning_effort"),
        "include_sampling_params": decode.get("include_sampling_params", True),
        "max_tokens_field": decode.get("max_tokens_field", "max_tokens"),
        "harness_version": rec.get("harness_version", ""),
        "git_sha": rec.get("git_sha", ""),
        "cohort_id": rec.get("cohort_id", ""),
        "usd_per_1m_input": rec.get("usd_per_1m_input"),
        "usd_per_1m_output": rec.get("usd_per_1m_output"),
        "price_quoted_on": rec.get("price_quoted_on", ""),
        "condition": rec.get("condition"),
        "seed": rec.get("seed"),
        "prompt_key": rec.get("prompt_key"),
        "prompt_hash": rec.get("prompt_hash"),
        "corruption": rec.get("corruption"),
        "corruption_severity": rec.get("corruption_severity"),
        "status": rec.get("status"),
        "error_reason": rec.get("error_reason", ""),
        "gold_letter": gold,
        "predicted_letter": predicted,
        "extraction_method": method,
        "ambiguous_candidates": candidates or [],
        # Invalid output is a reported metric, not a hidden wrong answer.
        "valid": valid,
        "correct": correct,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "reasoning_tokens_reported": usage.get("reasoning_tokens_reported", False),
        # What the provider says it served, e.g. "inferact/inkling-nvfp4".
        # Quantization affects accuracy and providers do not always volunteer
        # it, so whatever they do return is preserved verbatim.
        "served_model": rec.get("served_model", ""),
        "served_provider": rec.get("served_provider", ""),
        # Two independent fallbacks for RQ2 if reasoning_tokens goes missing:
        # the sibling reasoning_content field, and any inline <think> block.
        "reasoning_chars": len(rec.get("reasoning_text") or ""),
        "thinking_chars": count_thinking_chars(text),
        "response_chars": len(text),
        "latency_ms": rec.get("latency_ms"),
    }


def write_rows(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
