"""Image loading, corruption, and encoding for image-bearing conditions.

Source files are opened read-only and never written. Corruptions happen in
memory and the corrupted bytes go straight into the request payload.
"""

from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .corruptions import apply_corruption
from .hashing import hash_obj, sha256_file

# Fallback for image questions with no DriveLM object tag. Tagged questions use
# the camera named in the tag; always sending CAM_FRONT would show the wrong
# view for most of the frozen cohort, while sending all six views would multiply
# image cost and change the task unnecessarily.
DEFAULT_CAMERAS = ("CAM_FRONT",)
CAMERA_IN_OBJECT_TAG = re.compile(r"<[a-zA-Z]\d+,\s*(CAM_[A-Z_]+),")


@dataclass(frozen=True)
class ImageRef:
    camera: str
    path: str
    source_sha256: str


def referenced_cameras(stem: str) -> tuple[str, ...]:
    """Return tagged cameras in first-occurrence order, or CAM_FRONT fallback."""
    return tuple(dict.fromkeys(CAMERA_IN_OBJECT_TAG.findall(stem))) or DEFAULT_CAMERAS


def build_manifest(
    image_paths: dict[str, str],
    *,
    root: str = "",
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
) -> list[ImageRef]:
    refs: list[ImageRef] = []
    for cam in cameras:
        rel = image_paths.get(cam)
        if not rel:
            raise FileNotFoundError(f"missing image path for referenced camera {cam}")
        full = os.path.join(root, rel) if root else rel
        if not os.path.exists(full):
            raise FileNotFoundError(f"missing image for {cam}: {full}")
        refs.append(ImageRef(camera=cam, path=full, source_sha256=sha256_file(full)))
    return refs


def manifest_hash(
    refs: list[ImageRef],
    corruption: str,
    severity: int,
    *,
    max_edge: int = 0,
    jpeg_quality: int = 90,
) -> str:
    """Part of the cache key.

    Includes the corruption parameters, so changing severity forks the cache
    instead of silently reusing images degraded at the old setting — and the
    preprocessing parameters, for the same reason.
    """
    if not refs:
        return ""
    return hash_obj(
        {
            "images": [{"camera": r.camera, "sha256": r.source_sha256} for r in refs],
            "corruption": corruption,
            "severity": severity,
            # Preprocessing is part of what the model saw. Omitting it meant a
            # change to resizing silently reused responses collected against
            # differently-sized images.
            "max_edge": max_edge,
            "jpeg_quality": jpeg_quality,
        }
    )


def manifest_to_json(refs: list[ImageRef]) -> list[dict]:
    return [{"camera": r.camera, "path": r.path, "source_sha256": r.source_sha256} for r in refs]


def load_and_encode(
    refs: list[ImageRef],
    *,
    corruption: str = "",
    severity: int = 0,
    seed: int = 0,
    max_edge: int = 0,
    jpeg_quality: int = 90,
) -> list[str]:
    """Return base64 data URLs. Never touches the source file after reading.

    max_edge defaults to 0, meaning **no resizing**, and that default is
    load-bearing rather than incidental.

    DriveLM question text refers to objects by absolute pixel coordinate in the
    source image's frame — `<c1,CAM_FRONT,197.5,517.5>`. Resizing the image
    without rewriting those coordinates points the model at the wrong place: on
    a 1600x900 source scaled to 1024x576, that reference lands 186 pixels above
    the object it names. The model then answers from priors, scores at chance,
    and nothing about the run reveals why.

    Rewriting the coordinates instead is not an option: the blind conditions
    send the original numbers, so rescaling only in the image conditions would
    make the question text differ between conditions and break the paired
    comparison the whole grounding decomposition rests on.

    So the image is sent at native resolution and the coordinates stay true.
    If a future model or budget forces downscaling, the coordinates in the stem
    must be rewritten in the same commit, for every condition.
    """
    out: list[str] = []
    for ref in refs:
        with Image.open(ref.path) as im:
            im = im.convert("RGB")
            if max_edge and max(im.size) > max_edge:
                scale = max_edge / max(im.size)
                im = im.resize(
                    (max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                    Image.BILINEAR,
                )
            arr = np.asarray(im, dtype=np.uint8)

        if corruption:
            arr = apply_corruption(arr, corruption, severity, seed=seed)

        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        out.append(f"data:image/jpeg;base64,{b64}")
    return out
