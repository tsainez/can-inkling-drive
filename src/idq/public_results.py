"""Create a publication-safe, per-question results artifact.

The scored file is still a private working artifact: it contains identifiers and
diagnostic fields that are useful locally but are unnecessary for independent
analysis.  This module deliberately builds each public row from a fixed
allowlist.  It never copies an input dictionary and then deletes known-sensitive
keys; a newly-added private field therefore stays private by default.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable


PUBLIC_SCHEMA = "idq.public-result"
PUBLIC_SCHEMA_VERSION = 1

# Keep this tuple explicit and flat.  It is both the JSON schema and the privacy
# boundary: no source annotations, prompts, responses, reasoning, image
# manifests, paths, scene/frame IDs, or cache internals belong here.
PUBLIC_FIELDS = (
    "schema",
    "schema_version",
    "question_id",
    "category",
    "n_options",
    "model_label",
    "model_string",
    "served_by",
    "served_model",
    "served_provider",
    "quantization",
    "condition",
    "status",
    "gold_letter",
    "predicted_letter",
    "correct",
    "valid",
    "extraction_method",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "reasoning_tokens_reported",
    "latency_ms",
    "usd_per_1m_input",
    "usd_per_1m_output",
    "input_cost_usd",
    "output_cost_usd",
    "total_cost_usd",
    "price_quoted_on",
    "seed",
    "prompt_key",
    "prompt_hash",
    "temperature",
    "max_tokens",
    "thinking_effort",
    "reasoning_format",
    "include_sampling_params",
    "max_tokens_field",
    "corruption",
    "corruption_severity",
    "cohort_id",
    "git_sha",
    "harness_version",
)

_OPAQUE_ID_RE = re.compile(r"^[0-9a-f]{16,64}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_LETTER_RE = re.compile(r"^[A-Z]$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CONDITIONS = frozenset({"clean", "blind_tags", "blind_notags", "corrupt"})
_STATUSES = frozenset({"success", "terminal_error"})

# Public identifiers may contain a slash (for example, an exact provider model
# string), so path rejection is intentionally more specific than banning '/'.
_PRIVATE_TEXT_MARKERS = (
    "/Users/",
    "/home/",
    "../",
    "./data/",
    "./results/",
    "file://",
    "data:image/",
    "bearer ",
    "api_key",
    "api-key",
)


class PublicExportError(ValueError):
    """A scored row is unsafe or incomplete for public export."""


def _text(row: dict, field: str, *, required: bool = False, max_len: int = 200) -> str:
    value = row.get(field, "")
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise PublicExportError(f"{field} must be a string")
    if required and not value:
        raise PublicExportError(f"{field} is required")
    if len(value) > max_len or any(ord(char) < 32 for char in value):
        raise PublicExportError(f"{field} contains unsafe text")
    lowered = value.lower()
    if any(marker.lower() in lowered for marker in _PRIVATE_TEXT_MARKERS):
        raise PublicExportError(f"{field} looks like private path or credential material")
    return value


def _number(row: dict, field: str, *, integer: bool = False) -> int | float | None:
    value = row.get(field)
    if value is None:
        return None
    # bool is an int in Python, but accepting it here makes malformed provenance
    # look valid in JSON.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicExportError(f"{field} must be numeric or null")
    if not math.isfinite(float(value)) or value < 0:
        raise PublicExportError(f"{field} must be finite and non-negative")
    if integer:
        if not isinstance(value, int):
            raise PublicExportError(f"{field} must be an integer or null")
        return value
    return value


def _boolean(row: dict, field: str) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise PublicExportError(f"{field} must be a boolean")
    return value


def _cost(tokens: int | None, rate: int | float | None) -> float | None:
    if tokens is None or rate is None:
        return None
    return round(float(tokens) * float(rate) / 1_000_000, 12)


def public_row(row: dict) -> dict:
    """Return one validated row containing only ``PUBLIC_FIELDS``."""
    if not isinstance(row, dict):
        raise PublicExportError("each scored row must be a JSON object")

    question_id = _text(row, "question_id", required=True, max_len=64)
    if not _OPAQUE_ID_RE.fullmatch(question_id):
        raise PublicExportError("question_id must be an opaque 16-64 character hex ID")

    cohort_id = _text(row, "cohort_id", required=True, max_len=64)
    if not _HASH_RE.fullmatch(cohort_id):
        raise PublicExportError("cohort_id must be a 64-character hex digest")

    git_sha = _text(row, "git_sha", required=True, max_len=40)
    if not _GIT_SHA_RE.fullmatch(git_sha):
        raise PublicExportError("git_sha must be a 7-40 character hex revision")

    prompt_hash = _text(row, "prompt_hash", required=True, max_len=64)
    if not _HASH_RE.fullmatch(prompt_hash):
        raise PublicExportError("prompt_hash must be a 64-character hex digest")

    condition = _text(row, "condition", required=True, max_len=32)
    if condition not in _CONDITIONS:
        raise PublicExportError(f"unsupported condition {condition!r}")
    status = _text(row, "status", required=True, max_len=32)
    if status not in _STATUSES:
        raise PublicExportError(f"unsupported status {status!r}")

    gold = _text(row, "gold_letter", required=True, max_len=1)
    predicted = row.get("predicted_letter")
    if predicted is not None:
        if not isinstance(predicted, str) or not _LETTER_RE.fullmatch(predicted):
            raise PublicExportError("predicted_letter must be one uppercase letter or null")
    if not _LETTER_RE.fullmatch(gold):
        raise PublicExportError("gold_letter must be one uppercase letter")

    prompt_tokens = _number(row, "prompt_tokens", integer=True)
    completion_tokens = _number(row, "completion_tokens", integer=True)
    reasoning_tokens = _number(row, "reasoning_tokens", integer=True)
    input_rate = _number(row, "usd_per_1m_input")
    output_rate = _number(row, "usd_per_1m_output")
    input_cost = _cost(prompt_tokens, input_rate)
    output_cost = _cost(completion_tokens, output_rate)
    total_cost = (
        round(input_cost + output_cost, 12)
        if input_cost is not None and output_cost is not None
        else None
    )

    price_date = _text(row, "price_quoted_on", required=True, max_len=10)
    if not _DATE_RE.fullmatch(price_date):
        raise PublicExportError("price_quoted_on must use YYYY-MM-DD")

    exported = {
        "schema": PUBLIC_SCHEMA,
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "question_id": question_id,
        "category": _text(row, "category", required=True, max_len=64),
        "n_options": _number(row, "n_options", integer=True),
        "model_label": _text(row, "model_label", required=True),
        "model_string": _text(row, "model_string", required=True),
        "served_by": _text(row, "served_by", required=True),
        "served_model": _text(row, "served_model", required=status == "success"),
        "served_provider": _text(row, "served_provider"),
        "quantization": _text(row, "quantization", required=True),
        "condition": condition,
        "status": status,
        "gold_letter": gold,
        "predicted_letter": predicted,
        "correct": _boolean(row, "correct"),
        "valid": _boolean(row, "valid"),
        "extraction_method": _text(row, "extraction_method", required=True, max_len=64),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "reasoning_tokens_reported": _boolean(row, "reasoning_tokens_reported"),
        "latency_ms": _number(row, "latency_ms"),
        "usd_per_1m_input": input_rate,
        "usd_per_1m_output": output_rate,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
        "price_quoted_on": price_date,
        "seed": _number(row, "seed", integer=True),
        "prompt_key": _text(row, "prompt_key", required=True),
        "prompt_hash": prompt_hash,
        "temperature": _number(row, "temperature"),
        "max_tokens": _number(row, "max_tokens", integer=True),
        "thinking_effort": _text(row, "thinking_effort", max_len=32),
        "reasoning_format": _text(row, "reasoning_format", required=True, max_len=64),
        "include_sampling_params": _boolean(row, "include_sampling_params"),
        "max_tokens_field": _text(row, "max_tokens_field", required=True, max_len=64),
        "corruption": _text(row, "corruption", max_len=64),
        "corruption_severity": _number(row, "corruption_severity", integer=True),
        "cohort_id": cohort_id,
        "git_sha": git_sha,
        "harness_version": _text(row, "harness_version", required=True, max_len=64),
    }
    if tuple(exported) != PUBLIC_FIELDS:
        raise AssertionError("public exporter and PUBLIC_FIELDS are out of sync")
    return exported


def export_public_results(rows: Iterable[dict], path: str | os.PathLike[str]) -> dict:
    """Validate, sort, and atomically write deterministic public JSONL."""
    public = [public_row(row) for row in rows]
    lines = [
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        for row in public
    ]
    lines.sort()
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as fh:
            temp_name = fh.name
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_name, output)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)

    return {
        "schema": PUBLIC_SCHEMA,
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "rows": len(public),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "out": str(output),
    }
