"""Shared vector similarity helpers for benchmark adapters."""

from __future__ import annotations

import math
from collections.abc import Sequence


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity with a small-epsilon guard against zero vectors.

    Formula-identical to the per-adapter copies it replaces (``strict=False``
    zip, ``1e-9`` denominator guard) — pure refactor, benchmark results are
    unchanged.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)
