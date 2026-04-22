"""AFT adapter for replayable realistic benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from emotional_memory import (
    CoreAffect,
    EmotionalMemory,
    InMemoryStore,
    SQLiteAffectiveStateStore,
)

from .base import (
    ReplayAdapter,
    ReplayRetrievedItem,
    ReplaySessionEnd,
    ReplaySessionStart,
    TokenHashEmbedder,
)


class AFTReplayAdapter(ReplayAdapter):
    name = "aft"
    supports_explanations = True
    supports_persisted_state = True

    def __init__(self, workdir: Path) -> None:
        self._workdir = workdir
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._embedder = TokenHashEmbedder()
        self._state_path = workdir / "aft_state.sqlite"
        self._memories_path = workdir / "aft_memories.json"
        self._engine: EmotionalMemory | None = None

    def reset(self) -> None:
        self.close()
        if self._state_path.exists():
            self._state_path.unlink()
        if self._memories_path.exists():
            self._memories_path.unlink()

    def begin_session(self, session_id: str) -> ReplaySessionStart:
        state_store = SQLiteAffectiveStateStore(self._state_path)
        state_loaded = state_store.load() is not None
        engine = EmotionalMemory(
            store=InMemoryStore(),
            embedder=self._embedder,
            state_store=state_store,
        )
        if self._memories_path.exists():
            payload = json.loads(self._memories_path.read_text(encoding="utf-8"))
            engine.import_memories(payload, overwrite=True)
        self._engine = engine
        return ReplaySessionStart(
            state_loaded=state_loaded,
            memory_count_start=len(engine),
            mood_start=engine.get_state().mood.model_dump(mode="json"),
        )

    def encode(
        self,
        *,
        memory_alias: str,
        content: str,
        valence: float,
        arousal: float,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        engine = self._require_engine()
        event_metadata = {
            **(metadata or {}),
            "scenario_memory_id": memory_alias,
        }
        engine.set_affect(CoreAffect(valence=valence, arousal=arousal))
        memory = engine.encode(content, metadata=event_metadata)
        return memory.id

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        valence: float | None = None,
        arousal: float | None = None,
    ) -> list[ReplayRetrievedItem]:
        engine = self._require_engine()
        baseline_state = engine.save_state()
        if valence is not None and arousal is not None:
            engine.set_affect(CoreAffect(valence=valence, arousal=arousal))
        explanations = engine.retrieve_with_explanations(query, top_k=top_k)
        engine.load_state(baseline_state)
        return [
            ReplayRetrievedItem(
                id=item.memory.id,
                text=item.memory.content,
                score=item.score,
                metadata={
                    "explanation": {
                        "activation_level": item.activation_level,
                        "pass1_rank": item.pass1_rank,
                        "pass2_rank": item.pass2_rank,
                        "selected_as_seed": item.selected_as_seed,
                        "raw_signals": item.breakdown.raw_signals.model_dump(mode="json"),
                        "weighted_signals": item.breakdown.weighted_signals.model_dump(
                            mode="json"
                        ),
                    }
                },
            )
            for item in explanations
        ]

    def end_session(self) -> ReplaySessionEnd:
        engine = self._require_engine()
        self._memories_path.write_text(
            json.dumps(engine.export_memories(), indent=2),
            encoding="utf-8",
        )
        report = ReplaySessionEnd(
            memory_count_end=len(engine),
            mood_end=engine.get_state().mood.model_dump(mode="json"),
        )
        engine.close()
        self._engine = None
        return report

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def _require_engine(self) -> EmotionalMemory:
        if self._engine is None:
            raise RuntimeError("Session not started. Call begin_session() first.")
        return self._engine
