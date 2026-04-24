"""AFT adapter for the LoCoMo benchmark."""

from __future__ import annotations

import contextlib

from benchmarks.locomo.adapters.base import _ANSWER_SYSTEM, LoCoMoAdapter, call_llm
from benchmarks.locomo.dataset import Conversation, QAPair, Session
from emotional_memory import EmotionalMemory, EmotionalMemoryConfig, InMemoryStore
from emotional_memory.appraisal_llm import KeywordAppraisalEngine
from emotional_memory.embedders import SentenceTransformerEmbedder

_TOP_K = 8


class AFTLoCoMoAdapter(LoCoMoAdapter):
    """AFT-based memory adapter.

    Encodes each conversation turn as a memory (speaker + text).  Uses
    ``KeywordAppraisalEngine`` to infer affect from content without requiring
    a separate LLM key at encode time.  Retrieval uses the full 6-signal
    AFT scoring with bge-small-en-v1.5 embeddings.
    """

    name = "aft"

    def __init__(
        self,
        *,
        config: EmotionalMemoryConfig | None = None,
        top_k: int = _TOP_K,
    ) -> None:
        self._config = config
        self._top_k = top_k
        self._embedder = SentenceTransformerEmbedder.make_bge_small()
        self._engine: EmotionalMemory | None = None

    def reset(self) -> None:
        if self._engine is not None:
            self._engine.close()
        self._engine = EmotionalMemory(
            store=InMemoryStore(),
            embedder=self._embedder,
            appraisal_engine=KeywordAppraisalEngine(),
            config=self._config,
        )

    def ingest_session(self, session: Session, conversation: Conversation) -> None:
        engine = self._require_engine()
        for turn in session.turns:
            content = f"{turn.speaker}: {turn.text}"
            engine.encode(
                content,
                metadata={
                    "dia_id": turn.dia_id,
                    "speaker": turn.speaker,
                    "session": session.session_num,
                    "date": session.date_time,
                },
            )

    def answer(self, qa: QAPair, conversation: Conversation) -> str:
        engine = self._require_engine()
        retrieved = engine.retrieve(qa.question, top_k=self._top_k)
        context = "\n".join(f"- {mem.content}" for mem in retrieved)
        prompt = f"Conversation excerpts:\n{context}\n\nQuestion: {qa.question}\n\nAnswer:"
        return call_llm(prompt, system=_ANSWER_SYSTEM)

    def _require_engine(self) -> EmotionalMemory:
        if self._engine is None:
            raise RuntimeError("Call reset() before ingest_session().")
        return self._engine

    def __del__(self) -> None:
        if self._engine is not None:
            with contextlib.suppress(Exception):
                self._engine.close()
