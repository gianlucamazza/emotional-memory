"""Smoke tests for the top-level package exports."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

import emotional_memory

# Names that are gated behind optional extras; listed here for documentation.
# With PEP 562 __getattr__, these only appear in __all__ when the extra is installed.
_OPTIONAL_NAMES = frozenset(
    {
        "ChromaStore",
        "EmotionalMemoryChatHistory",
        "EmotionalMemoryMem0Backend",
        "QdrantStore",
        "RedisAffectiveStateStore",
        "SQLiteAffectiveStateStore",
        "SQLiteStore",
        "SentenceTransformerEmbedder",
        "messages_to_content",
        "recommended_conversation_policy",
        "store_all_messages",
    }
)


def test_all_exports_importable() -> None:
    """Every name in __all__ must be accessible as an attribute (no AttributeError)."""
    for name in emotional_memory.__all__:
        assert hasattr(emotional_memory, name), f"{name!r} in __all__ but not accessible"


def test_star_import_never_raises() -> None:
    """from emotional_memory import * must not raise even without any optional extras."""
    g: dict = {}
    exec("from emotional_memory import *", g)  # noqa: S102


def test_optional_export_missing_raises_import_error() -> None:
    """Accessing an optional export when the extra is absent raises ImportError with a hint."""
    import importlib

    # Simulate QdrantStore being absent by patching out the module.
    original = emotional_memory.__dict__.pop("QdrantStore", None)
    try:
        with (
            patch.dict(sys.modules, {"emotional_memory.stores.qdrant": None}),
            pytest.raises(ImportError, match="qdrant"),
        ):
            _ = emotional_memory.QdrantStore
    finally:
        if original is not None:
            emotional_memory.__dict__["QdrantStore"] = original
        else:
            emotional_memory.__dict__.pop("QdrantStore", None)
        # Restore module cache
        sys.modules.pop("emotional_memory.stores.qdrant", None)
        importlib.invalidate_caches()


def test_dir_includes_all_optional_names() -> None:
    """dir(emotional_memory) must include optional names for REPL/IDE discovery."""
    d = set(dir(emotional_memory))
    for name in _OPTIONAL_NAMES:
        assert name in d, f"{name!r} missing from dir(emotional_memory)"


def test_version_is_string() -> None:
    assert isinstance(emotional_memory.__version__, str)
    assert emotional_memory.__version__


def test_explainable_retrieval_exports_are_stable() -> None:
    assert "RetrievalSignals" in emotional_memory.__all__
    assert "RetrievalBreakdown" in emotional_memory.__all__
    assert "RetrievalExplanation" in emotional_memory.__all__
    assert "build_retrieval_plan" not in emotional_memory.__all__


def test_state_store_exports_are_stable() -> None:
    assert "AffectiveStateStore" in emotional_memory.__all__
    assert "InMemoryAffectiveStateStore" in emotional_memory.__all__
    # Optional backends — present only when their extra is installed.
    if "SQLiteAffectiveStateStore" in emotional_memory.__all__:
        assert hasattr(emotional_memory, "SQLiteAffectiveStateStore")
    if "RedisAffectiveStateStore" in emotional_memory.__all__:
        assert hasattr(emotional_memory, "RedisAffectiveStateStore")


def test_vector_store_exports_are_stable() -> None:
    assert hasattr(emotional_memory, "InMemoryStore")
    # Optional backends — check via __all__ membership (set dynamically).
    if "SQLiteStore" in emotional_memory.__all__:
        assert hasattr(emotional_memory, "SQLiteStore")
    if "QdrantStore" in emotional_memory.__all__:
        assert hasattr(emotional_memory, "QdrantStore")
    if "ChromaStore" in emotional_memory.__all__:
        assert hasattr(emotional_memory, "ChromaStore")
