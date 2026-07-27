"""In-memory sensor-degradation corruptions.

Originals are never modified: images are opened read-only, converted to arrays,
and every transform returns a new array. A test asserts the source file's hash
is unchanged after a full corruption pass.

Severity follows the ImageNet-C convention of 1-5 so the paper can say
"severity 3" and mean something comparable (Hendrycks & Dietterich, ICLR 2019).

Motion blur is implemented in numpy on purpose. PIL's ImageFilter.Kernel only
accepts 3x3 and 5x5 kernels, which is far too small for a realistic motion
streak, and silently constrains the severity scale if you try to use it.
"""

from __future__ import annotations

import numpy as np

from .hashing import hash_obj

SEVERITIES = (1, 2, 3, 4, 5)
CORRUPTIONS = ("fog", "brightness", "motion_blur", "gaussian_noise")


def _check(severity: int) -> int:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity}")
    return severity


def _to_float(img: np.ndarray) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def gaussian_noise(img: np.ndarray, severity: int = 3, rng: np.random.Generator | None = None):
    sigma = [0.04, 0.06, 0.10, 0.16, 0.26][_check(severity) - 1]
    rng = rng or np.random.default_rng(0)
    x = _to_float(img)
    return _to_uint8(x + rng.normal(0.0, sigma, x.shape).astype(np.float32))


def brightness(img: np.ndarray, severity: int = 3, rng: np.random.Generator | None = None):
    """Additive brightness in HSV-value space, approximated in RGB.

    Positive shift only: this models overexposure / sun glare, the failure mode
    that actually shows up in daytime camera data.
    """
    shift = [0.10, 0.20, 0.30, 0.40, 0.50][_check(severity) - 1]
    x = _to_float(img)
    return _to_uint8(x + shift)


def motion_blur(img: np.ndarray, severity: int = 3, rng: np.random.Generator | None = None,
                angle_deg: float | None = None):
    """Linear motion streak via a normalized line kernel, applied in numpy.

    Implemented as a mean of sub-pixel-rounded shifted copies, which is exactly
    a line kernel convolution and avoids both the PIL 3x3/5x5 limit and a scipy
    dependency.
    """
    length = [5, 9, 13, 19, 27][_check(severity) - 1]
    rng = rng or np.random.default_rng(0)
    if angle_deg is None:
        angle_deg = float(rng.uniform(0.0, 180.0))

    x = _to_float(img)
    if x.ndim == 2:
        x = x[:, :, None]

    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)

    offsets = np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, length)
    acc = np.zeros_like(x, dtype=np.float32)
    for t in offsets:
        sx = int(round(t * dx))
        sy = int(round(t * dy))
        acc += np.roll(np.roll(x, sy, axis=0), sx, axis=1)
    acc /= float(length)

    if acc.shape[2] == 1:
        acc = acc[:, :, 0]
    return _to_uint8(acc)


def fog(img: np.ndarray, severity: int = 3, rng: np.random.Generator | None = None):
    """Atmospheric scattering: I' = I * t + A * (1 - t).

    Transmission map t comes from smoothed value noise, so fog is spatially
    varying rather than a flat wash - a flat wash is trivially invertible and
    would understate the difficulty.
    """
    intensity, airlight = [
        (0.20, 0.85), (0.32, 0.87), (0.45, 0.90), (0.58, 0.92), (0.72, 0.95)
    ][_check(severity) - 1]
    rng = rng or np.random.default_rng(0)

    x = _to_float(img)
    if x.ndim == 2:
        x = x[:, :, None]
    h, w = x.shape[:2]

    noise = _value_noise(h, w, rng, octaves=4)
    depth = 0.5 + 0.5 * noise
    t = np.exp(-intensity * 3.0 * (1.0 - depth))
    t = t[:, :, None].astype(np.float32)

    out = x * t + airlight * (1.0 - t)
    if out.shape[2] == 1:
        out = out[:, :, 0]
    return _to_uint8(out)


def _value_noise(h: int, w: int, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    """Smooth [0,1] noise field built by upsampling coarse random grids."""
    acc = np.zeros((h, w), dtype=np.float32)
    total = 0.0
    for o in range(octaves):
        gh = max(2, h // (2 ** (octaves - o + 2)) + 2)
        gw = max(2, w // (2 ** (octaves - o + 2)) + 2)
        grid = rng.random((gh, gw)).astype(np.float32)
        acc += _bilinear_resize(grid, h, w) * (0.5 ** o)
        total += 0.5 ** o
    acc /= total
    lo, hi = float(acc.min()), float(acc.max())
    return (acc - lo) / (hi - lo) if hi > lo else np.zeros_like(acc)


def _bilinear_resize(src: np.ndarray, h: int, w: int) -> np.ndarray:
    sh, sw = src.shape
    ys = np.linspace(0, sh - 1, h, dtype=np.float32)
    xs = np.linspace(0, sw - 1, w, dtype=np.float32)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, sh - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, sw - 1)
    wy = (ys - y0)[:, None]; wx = (xs - x0)[None, :]
    top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
    bot = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


REGISTRY = {
    "fog": fog,
    "brightness": brightness,
    "motion_blur": motion_blur,
    "gaussian_noise": gaussian_noise,
}


def apply_corruption(img: np.ndarray, name: str, severity: int, seed: int = 0) -> np.ndarray:
    """Deterministic given (name, severity, seed). Returns a new array.

    Seeds via sha256 rather than Python's hash(): string hashing is randomized
    per process, so hash() would silently produce different corruptions on
    every run and make the condition irreproducible.
    """
    if name not in REGISTRY:
        raise ValueError(f"unknown corruption {name!r}; known: {sorted(REGISTRY)}")
    digest = hash_obj({"corruption": name, "severity": severity, "seed": seed})
    rng = np.random.default_rng(int(digest[:16], 16))
    return REGISTRY[name](np.asarray(img), severity=severity, rng=rng)
