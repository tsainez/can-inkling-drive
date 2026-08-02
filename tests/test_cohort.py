"""Frozen cohorts must be balanced, reproducible, and tamper-evident."""

from __future__ import annotations

import json

import pytest

from idq.cohort import (
    CohortError,
    build_manifest,
    read_manifest,
    select_questions,
    write_manifest,
)
from idq.convert import DriveLMConvertedAdapter

from test_convert import build_data


def _cohort(tmp_path):
    data_path = build_data(tmp_path)
    questions = DriveLMConvertedAdapter(
        path=data_path, min_total=50, n_per_template=100
    ).load()
    manifest = build_manifest(
        questions,
        data_path=data_path,
        adapter="drivelm_converted",
        convert_seed=20260731,
        n_per_template=100,
        created_on="2026-07-31",
        repeat_per_joint_cell=2,
    )
    return data_path, questions, manifest


def test_manifest_records_exact_joint_balance_and_public_boundary(tmp_path):
    _, _, manifest = _cohort(tmp_path)
    assert manifest["audit"]["n_questions"] == 100
    assert manifest["audit"]["by_gold_letter"] == {"A": 50, "B": 50}
    classes = next(iter(manifest["audit"]["gold_class_by_template"].values()))
    assert sorted(classes.values()) == [50, 50]
    joint = next(
        iter(manifest["audit"]["joint_gold_class_and_letter_by_template"].values())
    )
    assert sorted(joint.values()) == [25, 25, 25, 25]
    assert manifest["source"]["redistributed"] is False
    assert "no client data" in manifest["ip_boundary"].lower()


def test_manifest_round_trip_selects_the_same_order(tmp_path):
    data_path, questions, manifest = _cohort(tmp_path)
    path = tmp_path / "cohort.json"
    write_manifest(manifest, path)
    loaded = read_manifest(path)
    selected = select_questions(questions, loaded, data_path=data_path)
    assert [q.question_id for q in selected] == manifest["question_ids"]


def test_manifest_tampering_is_rejected(tmp_path):
    _, _, manifest = _cohort(tmp_path)
    path = tmp_path / "cohort.json"
    write_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["question_ids"] = payload["question_ids"][1:]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CohortError, match="identity hash"):
        read_manifest(path)


def test_random_slice_is_not_mistaken_for_a_balanced_cohort(tmp_path):
    data_path = build_data(tmp_path, n_frames=402)
    questions = DriveLMConvertedAdapter(path=data_path, min_total=50).load()
    with pytest.raises(CohortError, match="not exactly balanced"):
        build_manifest(
            questions,
            data_path=data_path,
            adapter="drivelm_converted",
            convert_seed=20260731,
            n_per_template=0,
            created_on="2026-07-31",
        )
