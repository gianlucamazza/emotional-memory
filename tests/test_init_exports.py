"""Smoke tests for the top-level package exports."""

from __future__ import annotations

import emotional_memory

# Names in __all__ that are only available when optional extras are installed.
_OPTIONAL_EXPORTS = frozenset(
    {
        "ChromaStore",
        "QdrantStore",
        "RedisAffectiveStateStore",
        "SentenceTransformerEmbedder",
        "EmotionalMemoryMem0Backend",
        "EmotionalMemoryChatHistory",
        "recommended_conversation_policy",
        "store_all_messages",
    }
)


def test_all_exports_importable() -> None:
    for name in emotional_memory.__all__:
        if name in _OPTIONAL_EXPORTS:
            continue  # skip extras not guaranteed to be installed
        assert hasattr(emotional_memory, name), f"{name!r} listed in __all__ but not importable"


def test_version_is_string() -> None:
    assert isinstance(emotional_memory.__version__, str)
    assert emotional_memory.__version__  # non-empty


def test_explainable_retrieval_exports_are_stable() -> None:
    assert "RetrievalSignals" in emotional_memory.__all__
    assert "RetrievalBreakdown" in emotional_memory.__all__
    assert "RetrievalExplanation" in emotional_memory.__all__
    assert "build_retrieval_plan" not in emotional_memory.__all__


def test_state_store_exports_are_stable() -> None:
    assert "AffectiveStateStore" in emotional_memory.__all__
    assert "InMemoryAffectiveStateStore" in emotional_memory.__all__
    assert hasattr(emotional_memory, "SQLiteAffectiveStateStore")
    assert hasattr(emotional_memory, "RedisAffectiveStateStore")


def test_vector_store_exports_are_stable() -> None:
    assert hasattr(emotional_memory, "InMemoryStore")
    assert hasattr(emotional_memory, "SQLiteStore")
    # optional extras — only assert if installed
    if emotional_memory._qdrant_available:  # type: ignore[attr-defined]
        assert hasattr(emotional_memory, "QdrantStore")
    if emotional_memory._chroma_available:  # type: ignore[attr-defined]
        assert hasattr(emotional_memory, "ChromaStore")
