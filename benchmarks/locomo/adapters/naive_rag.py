"""Naive RAG baseline for LoCoMo — bge-small + cosine, no affect."""

from __future__ import annotations

import math

from benchmarks.locomo.adapters.base import _ANSWER_SYSTEM, LoCoMoAdapter, call_llm
from benchmarks.locomo.dataset import Conversation, QAPair, Session
from emotional_memory.embedders import SentenceTransformerEmbedder

_TOP_K = 8


class NaiveRAGLoCoMoAdapter(LoCoMoAdapter):
    """Pure semantic retrieval baseline.

    Stores turn embeddings in-memory, retrieves by cosine similarity,
    then calls the same LLM as AFT for answer generation. This isolates
    the contribution of AFT's affective signals vs. raw embedding quality.
    """

    name = "naive_rag"

    def __init__(self, *, top_k: int = _TOP_K) -> None:
        self._top_k = top_k
        self._embedder = SentenceTransformerEmbedder.make_bge_small()
        self._store: list[tuple[str, list[float]]] = []

    def reset(self) -> None:
        self._store = []

    def ingest_session(self, session: Session, conversation: Conversation) -> None:
        for turn in session.turns:
            content = f"{turn.speaker}: {turn.text}"
            vec = self._embedder.embed(content)
            self._store.append((content, vec))

    def answer(self, qa: QAPair, conversation: Conversation) -> str:
        qvec = self._embedder.embed(qa.question)
        scored = sorted(
            ((text, _cosine(qvec, vec)) for text, vec in self._store),
            key=lambda x: x[1],
            reverse=True,
        )
        top = scored[: self._top_k]
        context = "\n".join(f"- {text}" for text, _ in top)
        prompt = f"Conversation excerpts:\n{context}\n\nQuestion: {qa.question}\n\nAnswer:"
        return call_llm(prompt, system=_ANSWER_SYSTEM)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)
