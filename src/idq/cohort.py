"""Frozen evaluation cohorts and their publication provenance.

The raw DriveLM file stays gitignored.  A cohort manifest contains only stable
question IDs, source hashes, selection parameters, and aggregate balance
counts, so it can be published without redistributing benchmark data.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

from .adapters.base import Question, expected_chance
from .hashing import hash_obj, sha256_file

SCHEMA_VERSION = 1


class CohortError(ValueError):
    """The frozen cohort cannot be reproduced exactly."""


def _sorted_counter(values) -> dict:
    return dict(sorted(Counter(values).items()))


def audit_questions(questions: list[Question]) -> dict:
    """Publication-facing composition and balance checks for a cohort."""
    if not questions:
        raise CohortError("cannot audit an empty cohort")

    template_counts = Counter()
    class_counts: dict[str, Counter] = defaultdict(Counter)
    joint_counts: dict[str, Counter] = defaultdict(Counter)

    for q in questions:
        template_id = str(q.raw.get("template_id") or "")
        gold_text = str(q.raw.get("gold_text") or "")
        if not template_id or not gold_text:
            raise CohortError(
                f"question {q.question_id} lacks converted-template provenance"
            )
        template_counts[template_id] += 1
        class_counts[template_id][gold_text] += 1
        joint_counts[template_id][f"{gold_text}|{q.gold_letter}"] += 1

    return {
        "n_questions": len(questions),
        "expected_chance": expected_chance(questions),
        "by_category": _sorted_counter(q.category for q in questions),
        "by_template": dict(sorted(template_counts.items())),
        "by_gold_letter": _sorted_counter(q.gold_letter for q in questions),
        "gold_class_by_template": {
            key: dict(sorted(counts.items()))
            for key, counts in sorted(class_counts.items())
        },
        "joint_gold_class_and_letter_by_template": {
            key: dict(sorted(counts.items()))
            for key, counts in sorted(joint_counts.items())
        },
    }


def require_exact_balance(audit: dict) -> None:
    """Reject a cohort where task, class, or option position can confer priors."""
    checks = {
        "templates": audit["by_template"],
        "gold letters": audit["by_gold_letter"],
    }
    checks.update(
        {
            f"gold classes for template {template_id}": counts
            for template_id, counts in audit["gold_class_by_template"].items()
        }
    )
    checks.update(
        {
            f"joint class/letter cells for template {template_id}": counts
            for template_id, counts in
            audit["joint_gold_class_and_letter_by_template"].items()
        }
    )
    for label, counts in checks.items():
        if not counts or len(set(counts.values())) != 1:
            raise CohortError(f"cohort is not exactly balanced across {label}: {counts}")


def _repeat_subset(questions: list[Question], per_joint_cell: int) -> dict:
    """A small balanced subset for measuring provider nondeterminism."""
    cells: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for q in questions:
        key = (
            str(q.raw["template_id"]),
            str(q.raw["gold_text"]),
            q.gold_letter,
        )
        cells[key].append(q.question_id)

    chosen: list[str] = []
    for key in sorted(cells):
        ids = sorted(cells[key])
        if len(ids) < per_joint_cell:
            raise CohortError(
                f"repeat subset needs {per_joint_cell} questions in {key}, found {len(ids)}"
            )
        chosen.extend(ids[:per_joint_cell])
    return {
        "selection": "first IDs sorted within each template x gold class x gold letter cell",
        "per_joint_cell": per_joint_cell,
        "n_questions": len(chosen),
        "question_ids": chosen,
    }


def build_manifest(
    questions: list[Question],
    *,
    data_path: str,
    adapter: str,
    convert_seed: int,
    n_per_template: int,
    created_on: str,
    repeat_per_joint_cell: int = 5,
) -> dict:
    ids = [q.question_id for q in questions]
    if len(ids) != len(set(ids)):
        raise CohortError("cohort contains duplicate question IDs")

    audit = audit_questions(questions)
    require_exact_balance(audit)
    identity = {
        "schema": SCHEMA_VERSION,
        "adapter": adapter,
        "source_sha256": sha256_file(data_path),
        "selection": {
            "convert_seed": convert_seed,
            "n_per_template": n_per_template,
            "task_weighting": "equal_templates",
        },
        "question_ids": ids,
    }
    return {
        "schema": SCHEMA_VERSION,
        "cohort_id": hash_obj(identity),
        "created_on": created_on,
        "benchmark": "DriveLM-nuScenes v1.1 train",
        "adapter": adapter,
        "source": {
            "filename": os.path.basename(data_path),
            "sha256": identity["source_sha256"],
            "redistributed": False,
        },
        "ip_boundary": (
            "Public DriveLM data only; no client data, telemetry, logs, imagery, "
            "observations, employer hardware, or employer compute."
        ),
        "selection": identity["selection"],
        "audit": audit,
        "question_ids": ids,
        "repeat_subset": _repeat_subset(questions, repeat_per_joint_cell),
    }


def write_manifest(manifest: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def read_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("schema") != SCHEMA_VERSION:
        raise CohortError(f"unsupported cohort schema: {manifest.get('schema')!r}")
    ids = manifest.get("question_ids") or []
    if len(ids) != len(set(ids)):
        raise CohortError("cohort manifest contains duplicate question IDs")
    identity = {
        "schema": manifest["schema"],
        "adapter": manifest.get("adapter"),
        "source_sha256": (manifest.get("source") or {}).get("sha256"),
        "selection": manifest.get("selection"),
        "question_ids": ids,
    }
    expected = hash_obj(identity)
    if manifest.get("cohort_id") != expected:
        raise CohortError("cohort manifest identity hash does not match its contents")
    return manifest


def select_questions(
    questions: list[Question],
    manifest: dict,
    *,
    data_path: str | None = None,
) -> list[Question]:
    if data_path:
        actual_source_hash = sha256_file(data_path)
        expected_source_hash = manifest["source"]["sha256"]
        if actual_source_hash != expected_source_hash:
            raise CohortError(
                "DriveLM source hash does not match the frozen cohort manifest"
            )

    by_id = {q.question_id: q for q in questions}
    missing = [qid for qid in manifest["question_ids"] if qid not in by_id]
    if missing:
        raise CohortError(
            f"cannot reproduce frozen cohort; {len(missing)} question IDs are missing"
        )
    selected = [by_id[qid] for qid in manifest["question_ids"]]
    observed = audit_questions(selected)
    if observed != manifest.get("audit"):
        raise CohortError("frozen cohort audit does not match reconstructed questions")
    require_exact_balance(observed)
    return selected

