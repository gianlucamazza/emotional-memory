"""Naive cosine-similarity baseline for the DailyDialog benchmark.

Stores turn embeddings in a flat list; retrieves by cosine similarity only,
with no affective signals.  This isolates the contribution of AFT's emotional
signals vs. pure semantic embedding quality.
"""

from __future__ import annotations

from benchmarks.common.similarity import cosine
from benchmarks.dailydialog.adapters.base import DailyDialogAdapter
from benchmarks.dailydialog.dataset import PersonaSession
from emotional_memory.embedders import SentenceTransformerEmbedder


class NaiveCosineDailyDialogAdapter(DailyDialogAdapter):
    """Pure semantic retrieval baseline — cosine similarity, no affect."""

    name = "naive_cosine"

    def __init__(self, *, embedder_name: str = "multilingual-e5-small") -> None:
        if embedder_name == "multilingual-e5-small":
            self._embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
        else:
            self._embedder = SentenceTransformerEmbedder.make_bge_small()
        # Each entry: (session_id, text, embedding)
        self._store: list[tuple[str, str, list[float]]] = []

    def reset(self) -> None:
        self._store = []

    def ingest_session(self, session: PersonaSession) -> None:
        for turn in session.turns:
            vec = self._embedder.embed(turn.text)
            self._store.append((session.session_id, turn.text, vec))

    def retrieve(self, query_text: str, *, top_k: int) -> list[str]:
        if not self._store:
            return []
        qvec = self._embedder.embed(query_text)
        scored = sorted(
            self._store,
            key=lambda entry: cosine(qvec, entry[2]),
            reverse=True,
        )
        return [entry[0] for entry in scored[:top_k]]
