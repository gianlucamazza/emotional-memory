"""Smoke tests for the top-level package exports."""

from __future__ import annotations

import emotional_memory


def test_all_exports_importable() -> None:
    for name in emotional_memory.__all__:
        assert hasattr(emotional_memory, name), f"{name!r} listed in __all__ but not importable"


def test_version_is_string() -> None:
    assert isinstance(emotional_memory.__version__, str)
    assert emotional_memory.__version__  # non-empty


def test_explainable_retrieval_exports_are_stable() -> None:
    assert "RetrievalSignals" in emotional_memory.__all__
    assert "RetrievalBreakdown" in emotional_memory.__all__
    assert "RetrievalExplanation" in emotional_memory.__all__
    assert "build_retrieval_plan" not in emotional_memory.__all__
