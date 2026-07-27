"""Image loading, corruption, and encoding for image-bearing conditions.

Source files are opened read-only and never written. Corruptions happen in
memory and the corrupted bytes go straight into the request payload.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .corruptions import apply_corruption
from .hashing import hash_obj, sha256_file

# nuScenes camera order. CAM_FRONT alone is the cheap default: sending six
# cameras multiplies input tokens by six for a question that usually only
# concerns one view.
DEFAULT_CAMERAS = ("CAM_FRONT",)


@dataclass(frozen=True)
class ImageRef:
    camera: str
    path: str
    source_sha256: str


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
            continue
        full = os.path.join(root, rel) if root else rel
        if not os.path.exists(full):
            raise FileNotFoundError(f"missing image for {cam}: {full}")
        refs.append(ImageRef(camera=cam, path=full, source_sha256=sha256_file(full)))
    return refs


def manifest_hash(refs: list[ImageRef], corruption: str, severity: int) -> str:
    """Part of the cache key.

    Includes the corruption parameters, so changing severity forks the cache
    instead of silently reusing images degraded at the old setting.
    """
    if not refs:
        return ""
    return hash_obj(
        {
            "images": [{"camera": r.camera, "sha256": r.source_sha256} for r in refs],
            "corruption": corruption,
            "severity": severity,
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
    max_edge: int = 1024,
    jpeg_quality: int = 90,
) -> list[str]:
    """Return base64 data URLs. Never touches the source file after reading."""
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
