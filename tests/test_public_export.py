"""The public result projection is deterministic and fails closed."""

from __future__ import annotations

import json

import pytest

from idq.cli import build_parser
from idq.public_results import (
    PUBLIC_FIELDS,
    PublicExportError,
    export_public_results,
)


def _row(question_id: str = "0123456789abcdef") -> dict:
    return {
        "cache_key": "private-cache-key",
        "question_id": question_id,
        "scene_id": "private-scene",
        "frame_id": "private-frame",
        "category": "planning",
        "n_options": 2,
        "model_label": "inkling",
        "model_string": "thinkingmachines/inkling",
        "served_by": "baseten",
        "served_model": "thinkingmachines/inkling",
        "served_provider": "",
        "quantization": "unknown",
        "thinking_effort": "medium",
        "max_tokens": 2048,
        "temperature": 0.0,
        "reasoning_format": "reasoning_effort",
        "include_sampling_params": True,
        "max_tokens_field": "max_tokens",
        "harness_version": "0.1.0",
        "git_sha": "123abcd",
        "cohort_id": "a" * 64,
        "usd_per_1m_input": 1.0,
        "usd_per_1m_output": 4.05,
        "price_quoted_on": "2026-08-03",
        "condition": "blind_tags",
        "seed": 0,
        "prompt_key": "mcq_blind@v1",
        "prompt_hash": "b" * 64,
        "corruption": "",
        "corruption_severity": 0,
        "status": "success",
        "error_reason": "",
        "gold_letter": "B",
        "predicted_letter": "A",
        "extraction_method": "answer_tag",
        "ambiguous_candidates": [],
        "valid": True,
        "correct": False,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "reasoning_tokens": 40,
        "reasoning_tokens_reported": True,
        "reasoning_chars": 300,
        "thinking_chars": 0,
        "response_chars": 9,
        "latency_ms": 750.5,
    }


def test_export_has_exact_allowlist_and_cannot_copy_private_content(tmp_path):
    row = _row()
    sentinel = "NEVER_PUBLISH_THIS_SENTINEL"
    row.update({
        "prompt_system": sentinel,
        "prompt_user": sentinel,
        "response_text": sentinel,
        "reasoning_text": sentinel,
        "response_raw": {"nested": sentinel},
        "image_manifest": [{"path": f"/Users/person/{sentinel}.jpg"}],
        "credential": f"sk-{sentinel}",
        "source_annotation": sentinel,
        "error_message": sentinel,
    })

    out = tmp_path / "public.jsonl"
    receipt = export_public_results([row], out)
    raw = out.read_text(encoding="utf-8")
    public = json.loads(raw)

    assert set(public) == set(PUBLIC_FIELDS)
    assert sentinel not in raw
    assert "/Users/" not in raw
    assert "scene_id" not in raw and "frame_id" not in raw
    assert "cache_key" not in raw and "ambiguous_candidates" not in raw
    assert "prompt_user" not in raw and "response_text" not in raw
    assert receipt["rows"] == 1
    assert receipt["sha256"]
    assert public["input_cost_usd"] == 0.0001
    assert public["output_cost_usd"] == 0.0002025
    assert public["total_cost_usd"] == 0.0003025


def test_output_is_byte_deterministic_independent_of_input_order(tmp_path):
    first = _row("1111111111111111")
    second = _row("0000000000000000")
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"

    receipt_a = export_public_results([first, second], a)
    receipt_b = export_public_results([second, first], b)

    assert a.read_bytes() == b.read_bytes()
    assert receipt_a["sha256"] == receipt_b["sha256"]


def test_cli_accepts_multiple_scored_inputs():
    args = build_parser().parse_args([
        "export-public",
        "--scored", "results/blind.jsonl",
        "--scored", "results/clean.jsonl",
    ])

    assert args.scored == ["results/blind.jsonl", "results/clean.jsonl"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("question_id", "scene-123", "opaque"),
        ("question_id", "/Users/person/private.jpg", "private path"),
        ("model_label", "/Users/person/private-model", "private path"),
        ("prompt_hash", "not-a-hash", "64-character"),
        ("git_sha", "dirty", "revision"),
        ("latency_ms", float("nan"), "finite"),
    ],
)
def test_unsafe_or_incomplete_allowed_values_fail_before_writing(
    tmp_path, field, value, message
):
    row = _row()
    row[field] = value
    out = tmp_path / "must-not-exist.jsonl"

    with pytest.raises(PublicExportError, match=message):
        export_public_results([row], out)

    assert not out.exists()
