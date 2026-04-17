"""SentenceTransformerEmbedder — production-ready Embedder backed by sentence-transformers.

Requires the ``sentence-transformers`` optional dependency::

    pip install emotional-memory[sentence-transformers]

Usage::

    from emotional_memory import EmotionalMemory, InMemoryStore
    from emotional_memory.embedders import SentenceTransformerEmbedder

    em = EmotionalMemory(
        store=InMemoryStore(),
        embedder=SentenceTransformerEmbedder(),  # defaults to all-MiniLM-L6-v2
    )
"""

from __future__ import annotations

from emotional_memory.interfaces import SequentialEmbedder


class SentenceTransformerEmbedder(SequentialEmbedder):
    """Wraps a sentence-transformers model as an :class:`Embedder`.

    Overrides :meth:`embed_batch` to use the model's native batching (one
    forward pass for all texts) rather than the sequential fallback.

    Parameters
    ----------
    model_name:
        Any model id accepted by :class:`sentence_transformers.SentenceTransformer`.
        Default is ``"all-MiniLM-L6-v2"`` — 384 dims, fast, multilingual-friendly
        enough for demos. Pick a larger model for production retrieval quality.
    """

    __slots__ = ("_model", "_model_name")

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder.\n"
                "Install with: pip install 'emotional-memory[sentence-transformers]'"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()  # type: ignore[no-any-return]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model_name!r})"
