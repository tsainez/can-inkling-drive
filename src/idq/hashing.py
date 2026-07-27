"""Canonical hashing.

Every hash in the harness goes through here so that two records that should
collide do collide, and two that should not, do not.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_obj(obj: Any) -> str:
    return sha256_str(canonical_json(obj))


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Hash a file without loading it into memory. Opens read-only."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def short(digest: str, n: int = 12) -> str:
    return digest[:n]
