"""Step 10: EmotionalMemory — the main facade.

Orchestrates the full AFT pipeline:

  encode():
    event → appraisal → core_affect update → AffectiveState update
    → EmotionalTag (snapshot of all 5 layers) → embed → store → resonance links

  retrieve():
    query → embed → multi-signal score per candidate
    → top-k → reconsolidation check on each result

  get_state() / set_affect():
    read and write the runtime affective state
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal import AppraisalEngine, AppraisalVector, consolidation_strength
from emotional_memory.decay import DecayConfig
from emotional_memory.interfaces import Embedder, MemoryStore
from emotional_memory.models import Memory, make_emotional_tag
from emotional_memory.resonance import ResonanceConfig, build_resonance_links
from emotional_memory.retrieval import (
    RetrievalConfig,
    affective_prediction_error,
    reconsolidate,
    retrieval_score,
)
from emotional_memory.state import AffectiveState


class EmotionalMemoryConfig(BaseModel):
    decay: DecayConfig = DecayConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    resonance: ResonanceConfig = ResonanceConfig()
    stimmung_alpha: float = 0.1


class EmotionalMemory:
    """Emotional memory system for LLMs based on Affective Field Theory.

    Usage::

        store = InMemoryStore()
        embedder = MyEmbedder()          # implements Embedder protocol
        engine = EmotionalMemory(store, embedder)

        # Encode with automatic appraisal (if engine provided) or manual affect
        engine.set_affect(CoreAffect(valence=0.8, arousal=0.6))
        memory = engine.encode("Just completed a difficult project successfully!")

        # Retrieve — emotionally weighted by current state
        results = engine.retrieve("challenging work accomplishment")
    """

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        appraisal_engine: AppraisalEngine | None = None,
        config: EmotionalMemoryConfig | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._appraisal_engine = appraisal_engine
        self._config = config or EmotionalMemoryConfig()
        self._state = AffectiveState.initial()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        content: str,
        appraisal: AppraisalVector | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Encode content into emotional memory.

        Pipeline:
          1. Compute/use appraisal → derive CoreAffect
          2. Update AffectiveState (core_affect, momentum, stimmung)
          3. Build EmotionalTag (snapshot of all 5 layers)
          4. Embed content
          5. Store memory
          6. Build resonance links against existing memories
          7. Update stored memory with resonance links
        """
        now = datetime.now(tz=timezone.utc)

        # Step 1: resolve affect
        if appraisal is None and self._appraisal_engine is not None:
            appraisal = self._appraisal_engine.appraise(content)

        if appraisal is not None:
            new_affect = appraisal.to_core_affect()
        else:
            new_affect = self._state.core_affect

        # Step 2: update affective state
        self._state = self._state.update(
            new_affect, now=now, stimmung_alpha=self._config.stimmung_alpha
        )

        # Step 3: build EmotionalTag
        cs = consolidation_strength(new_affect.arousal, self._state.stimmung.arousal)
        tag = make_emotional_tag(
            core_affect=self._state.core_affect,
            momentum=self._state.momentum,
            stimmung=self._state.stimmung,
            consolidation_strength=cs,
            appraisal=appraisal,
        )

        # Step 4: embed
        embedding = self._embedder.embed(content)

        # Step 5: create and store memory (without resonance links yet)
        memory = Memory.create(content=content, tag=tag, embedding=embedding, metadata=metadata)
        self._store.save(memory)

        # Step 6: build resonance links against all existing memories
        candidates = self._store.list_all()
        links = build_resonance_links(memory, candidates, self._config.resonance)

        if links:
            updated_tag = tag.model_copy(update={"resonance_links": links})
            memory = memory.model_copy(update={"tag": updated_tag})
            self._store.update(memory)

        return memory

    def retrieve(self, query: str, top_k: int = 5) -> list[Memory]:
        """Retrieve the top-k most relevant memories for the query.

        Scoring uses all 6 AFT signals with Stimmung-adaptive weights.
        Each retrieved memory undergoes a reconsolidation check: if the
        Affective Prediction Error exceeds the threshold, the tag's
        core_affect is updated.
        """
        now = datetime.now(tz=timezone.utc)
        query_embedding = self._embedder.embed(query)

        candidates = self._store.list_all()
        if not candidates:
            return []

        scored = []
        for mem in candidates:
            score = retrieval_score(
                query_embedding=query_embedding,
                query_affect=self._state.core_affect,
                current_stimmung=self._state.stimmung,
                current_momentum=self._state.momentum,
                memory=mem,
                active_memory_ids=[],
                now=now,
                decay_config=self._config.decay,
                retrieval_config=self._config.retrieval,
            )
            scored.append((score, mem))

        scored.sort(key=lambda t: t[0], reverse=True)
        top = [mem for _, mem in scored[:top_k]]

        # Reconsolidation check and retrieval count update
        result = []
        for mem in top:
            updated = mem.model_copy(
                update={
                    "tag": mem.tag.model_copy(
                        update={
                            "last_retrieved": now,
                            "retrieval_count": mem.tag.retrieval_count + 1,
                        }
                    )
                }
            )
            ape = affective_prediction_error(mem.tag.core_affect, self._state.core_affect)
            if ape > self._config.retrieval.ape_threshold:
                updated = updated.model_copy(
                    update={
                        "tag": reconsolidate(
                            updated.tag,
                            self._state.core_affect,
                            ape,
                            self._config.retrieval.reconsolidation_learning_rate,
                        )
                    }
                )
            self._store.update(updated)
            result.append(updated)

        return result

    def get_state(self) -> AffectiveState:
        """Return a copy of the current affective state."""
        return self._state.model_copy()

    def set_affect(self, core_affect: CoreAffect) -> None:
        """Manually inject a CoreAffect (e.g. from external appraisal).

        Updates core_affect, momentum, and stimmung.
        """
        self._state = self._state.update(
            core_affect,
            stimmung_alpha=self._config.stimmung_alpha,
        )
