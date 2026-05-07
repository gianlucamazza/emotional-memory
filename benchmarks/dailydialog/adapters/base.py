"""Shared adapter contract for the DailyDialog benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

from benchmarks.dailydialog.dataset import Persona, PersonaSession


class PersonaRunResult(TypedDict):
    query_id: str
    persona_id: str
    query_type: str
    query_text: str
    target_session_id: str
    retrieved_session_ids: list[str]
    top_k: int


class DailyDialogAdapter(ABC):
    """Session-aware adapter for the DailyDialog retrieval benchmark.

    Each persona is processed by calling ``reset()``, then ``ingest_session()``
    for every session in order, then ``retrieve()`` for each query.
    """

    name: str = "unnamed"

    @abstractmethod
    def reset(self) -> None:
        """Clear all state. Called once per persona."""

    @abstractmethod
    def ingest_session(self, session: PersonaSession) -> None:
        """Encode one session's turns into the adapter's memory store."""

    @abstractmethod
    def retrieve(self, query_text: str, *, top_k: int) -> list[str]:
        """Retrieve and return the session_ids of the top-k most relevant memories.

        The returned list must have length ≤ top_k.  Each element is the
        ``session_id`` of the source session for that retrieved memory.
        """

    def run_persona(self, persona: Persona) -> list[PersonaRunResult]:
        """Convenience: ingest all sessions then evaluate all queries.

        Returns one result dict per query.
        """
        self.reset()
        for session in persona.sessions:
            self.ingest_session(session)

        results: list[PersonaRunResult] = []
        for query in persona.queries:
            retrieved_ids = self.retrieve(query.text, top_k=query.top_k)
            results.append(
                PersonaRunResult(
                    query_id=query.query_id,
                    persona_id=persona.persona_id,
                    query_type=query.query_type,
                    query_text=query.text,
                    target_session_id=query.target_session_id,
                    retrieved_session_ids=retrieved_ids,
                    top_k=query.top_k,
                )
            )
        return results
