"""Image-view selection tests.

DriveLM object coordinates are camera-specific. A clean or corrupted request
must send the view named in the question, not a fixed front camera.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from idq.adapters.base import Question
from idq.cache import ResponseCache
from idq.collect import ConfigurationError, collect
from idq.config import MOCK_MODEL, DecodeParams, RunConfig
from idq.images import build_manifest, referenced_cameras
from idq.providers import MockProvider


def test_referenced_cameras_follow_object_tags_in_order():
    stem = (
        "Compare <c1,CAM_BACK,1.0,2.0> with <c2,CAM_FRONT_LEFT,3.0,4.0> "
        "and <c3,CAM_BACK,5.0,6.0>."
    )
    assert referenced_cameras(stem) == ("CAM_BACK", "CAM_FRONT_LEFT")
    assert referenced_cameras("No object reference here.") == ("CAM_FRONT",)


def test_missing_referenced_camera_path_fails_before_provider_call(tmp_path):
    with pytest.raises(FileNotFoundError, match="CAM_BACK"):
        build_manifest(
            {"CAM_FRONT": "front.jpg"},
            root=str(tmp_path),
            cameras=("CAM_BACK",),
        )


def test_clean_collection_sends_camera_named_in_question(tmp_path):
    image_root = tmp_path / "images"
    paths = {}
    for camera, value in (("CAM_FRONT", 20), ("CAM_BACK", 220)):
        rel = f"samples/{camera}/frame.jpg"
        path = image_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((24, 32, 3), value, dtype=np.uint8)).save(path)
        paths[camera] = rel

    question = Question(
        question_id="camera-selection",
        scene_id="scene",
        frame_id="frame",
        category="perception",
        stem="What is the status of <c1,CAM_BACK,12.0,8.0>?",
        letters=("A", "B"),
        option_texts=("Moving", "Stationary"),
        gold_letter="A",
        image_paths=paths,
    )
    cfg = RunConfig(
        model=MOCK_MODEL,
        condition="clean",
        seed=0,
        decode=DecodeParams(temperature=0.0, max_tokens=64),
        prompt_key="mcq@v1",
    )
    cache = ResponseCache(str(tmp_path / "cache.jsonl"))
    provider = MockProvider(seed=1)

    stats = collect(
        [question], cfg, provider, cache,
        image_root=str(image_root), verbose=False,
    )

    assert stats.successes == 1
    record = next(cache.records())
    assert [item["camera"] for item in record["image_manifest"]] == ["CAM_BACK"]
    assert json.dumps(record["image_manifest"]).find("CAM_FRONT") == -1


def test_all_image_paths_are_checked_before_first_provider_call(tmp_path):
    image_root = tmp_path / "images"
    front = image_root / "front.jpg"
    front.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(front)

    def question(qid, camera, paths):
        return Question(
            question_id=qid,
            scene_id="scene",
            frame_id=qid,
            category="perception",
            stem=f"What is the status of <c1,{camera},1.0,2.0>?",
            letters=("A", "B"),
            option_texts=("Moving", "Stationary"),
            gold_letter="A",
            image_paths=paths,
        )

    questions = [
        question("valid-first", "CAM_FRONT", {"CAM_FRONT": "front.jpg"}),
        question("missing-second", "CAM_BACK", {"CAM_BACK": "absent.jpg"}),
    ]
    cfg = RunConfig(
        model=MOCK_MODEL,
        condition="clean",
        seed=0,
        decode=DecodeParams(temperature=0.0, max_tokens=64),
        prompt_key="mcq@v1",
    )
    provider = MockProvider(seed=1)
    cache = ResponseCache(str(tmp_path / "cache.jsonl"))

    with pytest.raises(ConfigurationError, match="missing-second"):
        collect(
            questions, cfg, provider, cache,
            image_root=str(image_root), verbose=False,
        )

    assert provider.calls == 0
    assert len(cache) == 0
