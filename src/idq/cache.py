"""Content-addressed JSONL response cache.

Properties the study depends on:
  - Never pay twice for the same call.
  - Survive a dropped connection and resume. Append-only JSONL with a flush and
    fsync per record, so a killed process loses at most the in-flight call.
  - Store the FULL raw response. Metrics nobody thought of at collection time
    stay recoverable without spending money again.
  - Successes and terminal failures are cached. Retryable failures are not.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Iterator

from .hashing import hash_obj

SCHEMA_VERSION = 1


def make_cache_key(*, key_fields: dict, question_id: str, prompt_hash: str,
                   image_manifest_hash: str = "") -> str:
    return hash_obj(
        {
            "schema": SCHEMA_VERSION,
            "question_id": question_id,
            "prompt_hash": prompt_hash,
            "image_manifest_hash": image_manifest_hash,
            **key_fields,
        }
    )


@dataclass
class ResponseCache:
    path: str
    _index: dict = None  # cache_key -> status

    def __post_init__(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        self._index = {}
        self._load_index()

    def _load_index(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A partial trailing line from a hard kill. Skip it; the
                    # call will simply be re-made.
                    continue
                key = rec.get("cache_key")
                if key:
                    self._index[key] = rec.get("status", "success")

    def has(self, cache_key: str) -> bool:
        return cache_key in self._index

    def status(self, cache_key: str) -> str | None:
        return self._index.get(cache_key)

    def __len__(self) -> int:
        return len(self._index)

    def append(self, record: dict) -> None:
        key = record["cache_key"]
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._index[key] = record.get("status", "success")

    def records(self) -> Iterator[dict]:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def build_record(
    *,
    cache_key: str,
    status: str,
    question,
    cfg,
    prompt_hash: str,
    system: str,
    user: str,
    response=None,
    error_reason: str = "",
    error_message: str = "",
    image_manifest: list | None = None,
    harness_version: str = "",
    git_sha: str = "",
) -> dict:
    """One cache line. Raw response is stored whole and never trimmed."""
    rec = {
        "schema": SCHEMA_VERSION,
        "cache_key": cache_key,
        "status": status,
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question_id": question.question_id,
        "scene_id": question.scene_id,
        "frame_id": question.frame_id,
        "category": question.category,
        "n_options": question.n_options,
        "letters": list(question.letters),
        "gold_letter": question.gold_letter,
        "model_label": cfg.model.label,
        "model_string": cfg.model.model_string,
        "served_by": cfg.model.served_by,
        "quantization": cfg.model.quantization,
        # Price travels with the data. It is a dated snapshot of one provider,
        # not a property of the model, so it belongs in the record rather than
        # being supplied from a side channel at analysis time.
        "usd_per_1m_input": cfg.model.usd_per_1m_input,
        "usd_per_1m_output": cfg.model.usd_per_1m_output,
        "price_quoted_on": cfg.model.price_quoted_on,
        "condition": cfg.condition,
        "seed": cfg.seed,
        "prompt_key": cfg.prompt_key,
        "prompt_hash": prompt_hash,
        "corruption": cfg.corruption,
        "corruption_severity": cfg.corruption_severity,
        "decode": {
            "temperature": cfg.decode.temperature,
            "top_p": cfg.decode.top_p,
            "max_tokens": cfg.decode.max_tokens,
            "thinking_effort": cfg.decode.thinking_effort,
        },
        "image_manifest": image_manifest or [],
        "prompt_system": system,
        "prompt_user": user,
        "harness_version": harness_version,
        "git_sha": git_sha,
    }

    if response is not None:
        rec.update(
            {
                "response_text": response.text,
                "reasoning_text": response.reasoning_text,
                "served_model": response.served_model,
                "response_raw": response.raw,
                "usage": response.usage,
                "latency_ms": response.latency_ms,
                "http_status": response.status,
                "attempts": response.attempts,
            }
        )
    else:
        rec.update(
            {
                "response_text": None,
                "reasoning_text": "",
                "served_model": "",
                "response_raw": None,
                "usage": {},
                "latency_ms": None,
                "http_status": None,
                "attempts": None,
                "error_reason": error_reason,
                "error_message": error_message[:2000],
            }
        )
    return rec
