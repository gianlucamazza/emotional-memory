"""Tests for SentenceTransformerEmbedder (requires [sentence-transformers] extra).

Uses unittest.mock to avoid downloading a real model in CI.
Skipped automatically when sentence-transformers is not installed.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip entire module when sentence-transformers is not installed
# ---------------------------------------------------------------------------

pytest.importorskip("sentence_transformers")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_st(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch SentenceTransformer at the source so no model is downloaded."""
    fake_model = MagicMock()
    fake_model.encode.side_effect = _fake_encode
    mock_cls = MagicMock(return_value=fake_model)
    # Patch the source module — SentenceTransformer is imported inside __init__(),
    # not at module level, so we must mock it where it lives.
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", mock_cls)
    return mock_cls


def _fake_encode(input: Any, **_: Any) -> np.ndarray:
    """Return 384-dim float32 vectors (same dim as all-MiniLM-L6-v2)."""
    if isinstance(input, list):
        return np.random.default_rng(0).random((len(input), 384)).astype("float32")
    return np.random.default_rng(0).random(384).astype("float32")


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


def test_importable_from_embedders_subpackage() -> None:
    from emotional_memory.embedders import SentenceTransformerEmbedder  # noqa: F401


def test_importable_from_top_level() -> None:
    import emotional_memory as em

    assert "SentenceTransformerEmbedder" in em.__all__


# ---------------------------------------------------------------------------
# Unit tests (mocked model)
# ---------------------------------------------------------------------------


def test_embed_returns_list_of_float(mock_st: MagicMock) -> None:
    from emotional_memory.embedders.sentence_transformers import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder("test-model")
    result = embedder.embed("hello world")

    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(v, float) for v in result)


def test_embed_batch_returns_list_of_lists(mock_st: MagicMock) -> None:
    from emotional_memory.embedders.sentence_transformers import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    texts = ["foo", "bar", "baz"]
    results = embedder.embed_batch(texts)

    assert isinstance(results, list)
    assert len(results) == len(texts)
    assert all(isinstance(row, list) for row in results)
    assert all(len(row) == 384 for row in results)


def test_default_model_name(mock_st: MagicMock) -> None:
    from emotional_memory.embedders.sentence_transformers import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    assert embedder._model_name == "all-MiniLM-L6-v2"
    mock_st.assert_called_once_with("all-MiniLM-L6-v2")


def test_custom_model_name(mock_st: MagicMock) -> None:
    from emotional_memory.embedders.sentence_transformers import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    assert embedder._model_name == "BAAI/bge-small-en-v1.5"
    mock_st.assert_called_once_with("BAAI/bge-small-en-v1.5")


def test_repr(mock_st: MagicMock) -> None:
    from emotional_memory.embedders.sentence_transformers import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder("my-model")
    assert "SentenceTransformerEmbedder" in repr(embedder)
    assert "my-model" in repr(embedder)


def test_import_error_without_package() -> None:
    """ImportError is raised at instantiation when sentence-transformers is absent."""
    from emotional_memory.embedders.sentence_transformers import SentenceTransformerEmbedder

    # sys.modules[key] = None makes `from key import ...` raise ImportError
    with (
        patch.dict(sys.modules, {"sentence_transformers": None}),
        pytest.raises(ImportError, match="sentence-transformers"),
    ):  # type: ignore[dict-item]
        SentenceTransformerEmbedder()


# ---------------------------------------------------------------------------
# Integration with EmotionalMemory
# ---------------------------------------------------------------------------


def test_works_as_emotional_memory_embedder(mock_st: MagicMock) -> None:
    from emotional_memory import CoreAffect, EmotionalMemory, InMemoryStore
    from emotional_memory.embedders.sentence_transformers import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    em = EmotionalMemory(store=InMemoryStore(), embedder=embedder)

    em.set_affect(CoreAffect(valence=0.5, arousal=0.5))
    mem = em.encode("test memory for embedder integration")
    assert mem.embedding is not None
    assert len(mem.embedding) == 384

    results = em.retrieve("test memory", top_k=1)
    assert len(results) == 1
    assert results[0].id == mem.id
