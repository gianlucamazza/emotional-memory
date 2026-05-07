"""AFT adapter for the DailyDialog benchmark.

Encodes each turn as a memory with the session's dominant-emotion PAD state
injected via ``engine.set_affect()``.  This allows AFT's 6-signal retrieval
scorer to use valence, arousal, momentum, and mood-congruence signals when
ranking memories for affect-conditioned queries.
"""

from __future__ import annotations

import contextlib

from benchmarks.dailydialog.adapters.base import DailyDialogAdapter
from benchmarks.dailydialog.dataset import PersonaSession
from emotional_memory import EmotionalMemory, EmotionalMemoryConfig, InMemoryStore
from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal_llm import KeywordAppraisalEngine
from emotional_memory.embedders import SentenceTransformerEmbedder


class AFTDailyDialogAdapter(DailyDialogAdapter):
    """AFT memory adapter for DailyDialog.

    Before encoding each session, ``set_affect()`` injects the session's
    PAD values so that encoded memories carry the correct affective context
    (rather than relying on KeywordAppraisalEngine to infer it from short
    dialogue turns which may not contain explicit emotion keywords).
    """

    name = "aft"

    def __init__(
        self,
        *,
        config: EmotionalMemoryConfig | None = None,
        embedder_name: str = "multilingual-e5-small",
    ) -> None:
        self._config = config
        if embedder_name == "multilingual-e5-small":
            self._embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
        else:
            self._embedder = SentenceTransformerEmbedder.make_bge_small()
        self._engine: EmotionalMemory | None = None
        # Map memory_id → session_id for retrieval scoring
        self._memory_session_map: dict[str, str] = {}

    def reset(self) -> None:
        if self._engine is not None:
            with contextlib.suppress(Exception):
                self._engine.close()
        self._engine = EmotionalMemory(
            store=InMemoryStore(),
            embedder=self._embedder,
            appraisal_engine=KeywordAppraisalEngine(),
            config=self._config,
        )
        self._memory_session_map = {}

    def ingest_session(self, session: PersonaSession) -> None:
        engine = self._require_engine()
        # Inject the session's dominant-emotion PAD values before encoding
        engine.set_affect(
            CoreAffect(
                valence=session.valence,
                arousal=session.arousal,
                dominance=session.dominance,
            )
        )
        for turn in session.turns:
            memory = engine.encode(
                turn.text,
                metadata={
                    "session_id": session.session_id,
                    "dialog_id": session.dialog_id,
                    "emotion": turn.emotion,
                },
            )
            self._memory_session_map[memory.id] = session.session_id

    def retrieve(self, query_text: str, *, top_k: int) -> list[str]:
        engine = self._require_engine()
        memories = engine.retrieve(query_text, top_k=top_k)
        return [self._memory_session_map.get(m.id, "") for m in memories]

    def _require_engine(self) -> EmotionalMemory:
        if self._engine is None:
            raise RuntimeError("Call reset() before ingest_session().")
        return self._engine

    def __del__(self) -> None:
        if self._engine is not None:
            with contextlib.suppress(Exception):
                self._engine.close()
