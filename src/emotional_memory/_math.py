"""Shared mathematical utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 if either is zero or contains NaN."""
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    va: NDArray[np.float64] = np.asarray(a, dtype=np.float64)
    vb: NDArray[np.float64] = np.asarray(b, dtype=np.float64)
    if bool(np.isnan(va).any()) or bool(np.isnan(vb).any()):
        return 0.0
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))
