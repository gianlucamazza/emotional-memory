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

from datetime import UTC, datetime
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
from emotional_memory.stimmung import StimmungDecayConfig, StimmungField


class EmotionalMemoryConfig(BaseModel):
    decay: DecayConfig = DecayConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    resonance: ResonanceConfig = ResonanceConfig()
    stimmung_alpha: float = 0.1
    stimmung_decay: StimmungDecayConfig | None = None
    """Time-based regression of Stimmung toward baseline. None = disabled."""


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
        now = datetime.now(tz=UTC)

        # Step 1: resolve affect
        if appraisal is None and self._appraisal_engine is not None:
            appraisal = self._appraisal_engine.appraise(content, context=metadata)

        if appraisal is not None:
            new_affect = appraisal.to_core_affect()
        else:
            new_affect = self._state.core_affect

        # Step 2: update affective state
        self._state = self._state.update(
            new_affect,
            now=now,
            stimmung_alpha=self._config.stimmung_alpha,
            stimmung_decay=self._config.stimmung_decay,
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

        # Step 6: build resonance links — pre-filter when store is large (G2)
        resonance_limit = (
            self._config.resonance.max_links * self._config.resonance.candidate_multiplier
        )
        if len(self._store) > resonance_limit and memory.embedding is not None:
            candidates = self._store.search_by_embedding(memory.embedding, resonance_limit)
        else:
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

        Two-pass strategy for spreading activation (G1):
          Pass 1 — score all candidates with empty active_ids to get an initial
                   top-k ranking; their IDs become the active set.
          Pass 2 — re-score with the active set, allowing resonance links to
                   boost memories associated with the initially top-ranked ones.

        Pre-filter (G2): when the store is large, use search_by_embedding to
        narrow candidates to top_k * candidate_multiplier before full scoring.

        Reconsolidation (D1): only triggered if the memory was last retrieved
        within the reconsolidation_window_seconds lability window.
        """
        now = datetime.now(tz=UTC)
        query_embedding = self._embedder.embed(query)

        # G2 — pre-filter candidates via embedding search when store is large
        rc = self._config.retrieval
        candidate_limit = top_k * rc.candidate_multiplier
        if len(self._store) > candidate_limit:
            candidates = self._store.search_by_embedding(query_embedding, candidate_limit)
        else:
            candidates = self._store.list_all()

        if not candidates:
            return []

        def _score_all(active_ids: list[str]) -> list[tuple[float, Memory]]:
            scored = []
            for mem in candidates:
                score = retrieval_score(
                    query_embedding=query_embedding,
                    query_affect=self._state.core_affect,
                    current_stimmung=self._state.stimmung,
                    current_momentum=self._state.momentum,
                    memory=mem,
                    active_memory_ids=active_ids,
                    now=now,
                    decay_config=self._config.decay,
                    retrieval_config=rc,
                )
                scored.append((score, mem))
            scored.sort(key=lambda t: t[0], reverse=True)
            return scored

        # G1 — two-pass: first pass seeds the active set for spreading activation
        pass1 = _score_all([])
        active_ids = [mem.id for _, mem in pass1[:top_k]]
        pass2 = _score_all(active_ids)

        top = [mem for _, mem in pass2[:top_k]]

        # Reconsolidation check and retrieval count update
        window = rc.reconsolidation_window_seconds
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
            # D1 — only reconsolidate within the lability window
            last = mem.tag.last_retrieved
            in_window = last is not None and (now - last).total_seconds() <= window
            if ape > rc.ape_threshold and in_window:
                updated = updated.model_copy(
                    update={
                        "tag": reconsolidate(
                            updated.tag,
                            self._state.core_affect,
                            ape,
                            rc.reconsolidation_learning_rate,
                        )
                    }
                )
            self._store.update(updated)
            result.append(updated)

        return result

    def encode_batch(
        self,
        contents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[Memory]:
        """Encode multiple contents using a single embed_batch call.

        Uses embed_batch() for efficient batched embedding (G6), then encodes
        each item sequentially so that AffectiveState evolves naturally.
        Resonance links are built against the store state at each step, meaning
        later items in the batch can link to earlier ones.
        """
        embeddings = self._embedder.embed_batch(contents)
        results = []
        for i, (content, embedding) in enumerate(zip(contents, embeddings, strict=True)):
            meta = metadata[i] if metadata else None
            now = datetime.now(tz=UTC)

            # Resolve appraisal per item (mirrors encode())
            appraisal: AppraisalVector | None = None
            if self._appraisal_engine is not None:
                appraisal = self._appraisal_engine.appraise(content, context=meta)

            if appraisal is not None:
                new_affect = appraisal.to_core_affect()
            else:
                new_affect = self._state.core_affect

            self._state = self._state.update(
                new_affect,
                now=now,
                stimmung_alpha=self._config.stimmung_alpha,
                stimmung_decay=self._config.stimmung_decay,
            )

            cs = consolidation_strength(new_affect.arousal, self._state.stimmung.arousal)
            tag = make_emotional_tag(
                core_affect=self._state.core_affect,
                momentum=self._state.momentum,
                stimmung=self._state.stimmung,
                consolidation_strength=cs,
                appraisal=appraisal,
            )

            memory = Memory.create(content=content, tag=tag, embedding=embedding, metadata=meta)
            self._store.save(memory)

            resonance_limit = (
                self._config.resonance.max_links * self._config.resonance.candidate_multiplier
            )
            if len(self._store) > resonance_limit and memory.embedding is not None:
                candidates = self._store.search_by_embedding(memory.embedding, resonance_limit)
            else:
                candidates = self._store.list_all()
            links = build_resonance_links(memory, candidates, self._config.resonance)
            if links:
                updated_tag = tag.model_copy(update={"resonance_links": links})
                memory = memory.model_copy(update={"tag": updated_tag})
                self._store.update(memory)

            results.append(memory)
        return results

    def delete(self, memory_id: str) -> None:
        """Remove a memory from the store."""
        self._store.delete(memory_id)

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
            stimmung_decay=self._config.stimmung_decay,
        )

    def save_state(self) -> dict[str, Any]:
        """Serialise the current affective state for persistence.

        Returns a JSON-serialisable dict that can be stored alongside the
        memory store and later restored with ``load_state()``.  The private
        momentum history is included so that momentum computation continues
        correctly after a restart.
        """
        return self._state.snapshot()

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore a previously saved affective state.

        Args:
            data: A dict produced by a previous ``save_state()`` call.
        """
        self._state = AffectiveState.restore(data)

    def get_current_stimmung(self, now: datetime | None = None) -> StimmungField:
        """Return the Stimmung regressed to ``now`` without modifying state.

        Useful for read-only mood inspection between encode/retrieve calls.
        If ``stimmung_decay`` is not configured, returns the frozen Stimmung.
        """

        if now is None:
            now = datetime.now(tz=UTC)
        if self._config.stimmung_decay is not None:
            return self._state.stimmung.regress(now, self._config.stimmung_decay)
        return self._state.stimmung
