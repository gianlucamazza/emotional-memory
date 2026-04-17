import contextlib

__all__ = ["SentenceTransformerEmbedder"]

with contextlib.suppress(ImportError):
    from emotional_memory.embedders.sentence_transformers import (
        SentenceTransformerEmbedder as SentenceTransformerEmbedder,
    )
