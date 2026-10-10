"""Unit tests for SQLiteStore vec-dimension gating (no sqlite-vec required)."""

from __future__ import annotations

import pytest

from emotional_memory.stores.sqlite import _MAX_VEC_DIM, _require_vec_dim


@pytest.mark.parametrize("dim", [1, 2, 384, 4096, _MAX_VEC_DIM])
def test_require_vec_dim_accepts_bounded_positive_ints(dim: int) -> None:
    assert _require_vec_dim(dim) == dim


@pytest.mark.parametrize("dim", [0, -1, _MAX_VEC_DIM + 1, 10**6])
def test_require_vec_dim_rejects_empty_or_oversized(dim: int) -> None:
    with pytest.raises(ValueError, match="embedding dimension"):
        _require_vec_dim(dim)
