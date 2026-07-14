"""Shared helpers for sync and async engine facades."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

DEFAULT_QUERY_AFFECT_GATE_TAU = 0.2

_SEMANTIC_ONLY_WEIGHTS: NDArray[np.float64] = np.array(
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
)


def semantic_only_weights() -> NDArray[np.float64]:
    """Pure-semantic retrieval weights (naive-cosine arm in Addendum Y)."""
    return _SEMANTIC_ONLY_WEIGHTS


def is_query_affect_neutral(valence: float, tau: float) -> bool:
    """True when the appraised query is routed to the cosine arm (strict ``<``)."""
    return abs(valence) < tau


def check_content_length(content: str, max_length: int | None, *, label: str = "content") -> None:
    """Raise ``ValueError`` when *content* exceeds the configured bound."""
    if max_length is not None and len(content) > max_length:
        raise ValueError(f"{label} length {len(content)} exceeds max_content_length={max_length}")


def validate_prune_threshold(threshold: float) -> None:
    """Raise ``ValueError`` when *threshold* is outside [0, 1]."""
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"prune threshold must be in [0.0, 1.0], got {threshold}")
