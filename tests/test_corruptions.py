"""Corruption tests: determinism, severity monotonicity, original-file integrity."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from idq.corruptions import CORRUPTIONS, apply_corruption, motion_blur
from idq.hashing import sha256_file


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_deterministic_given_seed(img, name):
    a = apply_corruption(img, name, 3, seed=42)
    b = apply_corruption(img, name, 3, seed=42)
    assert np.array_equal(a, b), f"{name} is not reproducible across calls"


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_shape_dtype_and_range_preserved(img, name):
    out = apply_corruption(img, name, 3, seed=0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert 0 <= out.min() and out.max() <= 255


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_corruption_actually_changes_the_image(img, name):
    assert not np.array_equal(apply_corruption(img, name, 3, seed=0), img)


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_severity_is_monotone(img, name):
    """Severity 5 must be further from the original than severity 1."""
    base = img.astype(np.float32)
    d1 = np.abs(apply_corruption(img, name, 1, seed=0).astype(np.float32) - base).mean()
    d5 = np.abs(apply_corruption(img, name, 5, seed=0).astype(np.float32) - base).mean()
    assert d5 > d1, f"{name}: severity 5 ({d5:.2f}) not stronger than 1 ({d1:.2f})"


def test_originals_are_never_modified(tmp_path, img):
    """The source file on disk must be byte-identical after a corruption pass."""
    path = tmp_path / "frame.jpg"
    Image.fromarray(img).save(path, format="JPEG", quality=95)
    before = sha256_file(str(path))

    from idq.images import ImageRef, load_and_encode
    refs = [ImageRef(camera="CAM_FRONT", path=str(path), source_sha256=before)]
    for name in CORRUPTIONS:
        load_and_encode(refs, corruption=name, severity=4, seed=1)

    assert sha256_file(str(path)) == before


def test_input_array_is_not_mutated(img):
    original = img.copy()
    for name in CORRUPTIONS:
        apply_corruption(img, name, 4, seed=0)
    assert np.array_equal(img, original)


def test_motion_blur_handles_kernels_larger_than_five():
    """PIL's ImageFilter.Kernel only accepts 3x3 and 5x5; ours is numpy, so
    severity 5 uses a 27-pixel streak without special-casing."""
    rng = np.random.default_rng(1)
    im = rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8)
    out = motion_blur(im, severity=5, rng=np.random.default_rng(0), angle_deg=0.0)
    assert out.shape == im.shape
    assert out.std() < im.std(), "a 27-pixel streak should reduce variance"


def test_grayscale_input_supported():
    rng = np.random.default_rng(2)
    im = rng.integers(0, 256, size=(40, 40), dtype=np.uint8)
    for name in CORRUPTIONS:
        assert apply_corruption(im, name, 2, seed=0).shape == im.shape


def test_unknown_corruption_and_severity_rejected(img):
    with pytest.raises(ValueError):
        apply_corruption(img, "snow", 3)
    with pytest.raises(ValueError):
        apply_corruption(img, "fog", 9)


def test_images_are_sent_at_native_resolution_so_coordinates_stay_true(tmp_path):
    """Regression: the clean condition once pointed the model at the wrong pixel.

    DriveLM refers to objects by absolute coordinate in the source frame
    (`<c1,CAM_FRONT,197.5,517.5>`). The harness used to downscale 1600x900 to
    1024x576 without rewriting those numbers, so every reference missed its
    object by ~186 pixels vertically and the model answered from priors.
    """
    import base64, io
    import numpy as np
    from PIL import Image

    from idq.images import ImageRef, load_and_encode
    from idq.hashing import sha256_file

    rng = np.random.default_rng(0)
    src = rng.integers(0, 256, size=(900, 1600, 3), dtype=np.uint8)
    path = tmp_path / "frame.jpg"
    Image.fromarray(src).save(path, format="JPEG", quality=95)

    refs = [ImageRef("CAM_FRONT", str(path), sha256_file(str(path)))]
    data_url = load_and_encode(refs)[0]
    raw = base64.b64decode(data_url.split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as sent:
        assert sent.size == (1600, 900), (
            f"image was resized to {sent.size}; DriveLM coordinates are in "
            "1600x900 space and would no longer point at their objects"
        )


def test_preprocessing_parameters_are_in_the_cache_key(tmp_path):
    """A resize change must fork the cache, not silently reuse old responses."""
    import numpy as np
    from PIL import Image

    from idq.images import ImageRef, manifest_hash
    from idq.hashing import sha256_file

    rng = np.random.default_rng(1)
    path = tmp_path / "f.jpg"
    Image.fromarray(rng.integers(0, 256, size=(900, 1600, 3), dtype=np.uint8)).save(
        path, format="JPEG"
    )
    refs = [ImageRef("CAM_FRONT", str(path), sha256_file(str(path)))]

    native = manifest_hash(refs, "", 0, max_edge=0)
    resized = manifest_hash(refs, "", 0, max_edge=1024)
    requality = manifest_hash(refs, "", 0, max_edge=0, jpeg_quality=60)

    assert native != resized
    assert native != requality
