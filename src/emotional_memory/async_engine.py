"""AsyncEmotionalMemory — async-native facade for the AFT pipeline.

Mirrors ``EmotionalMemory`` exactly but awaits all I/O boundaries
(embed, store operations, appraise).  The CPU-bound scoring logic
(adaptive_weights, retrieval_score, build_resonance_links, decay) runs
synchronously inline — no threading overhead for pure computation.

Design follows the httpx / SQLAlchemy pattern: separate ``EmotionalMemory``
(sync) and ``AsyncEmotionalMemory`` (async) classes with no shared base class,
so each API surface is unambiguous and type-checker friendly.

Typical usage::

    from emotional_memory.async_engine import AsyncEmotionalMemory
    from emotional_memory.stores.in_memory import InMemoryStore
    from emotional_memory.async_adapters import SyncToAsyncStore, SyncToAsyncEmbedder

    engine = AsyncEmotionalMemory(
        store=SyncToAsyncStore(InMemoryStore()),
        embedder=SyncToAsyncEmbedder(my_sync_embedder),
    )

    memory = await engine.encode("Something happened")
    results = await engine.retrieve("what happened?")

Or wrap an existing sync engine::

    from emotional_memory.async_adapters import as_async
    async_engine = as_async(sync_engine)
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import UTC, datetime
from typing import Any

import numpy as np

from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal import AppraisalVector, consolidation_strength
from emotional_memory.engine import EmotionalMemoryConfig
from emotional_memory.interfaces_async import AsyncAppraisalEngine, AsyncEmbedder, AsyncMemoryStore
from emotional_memory.models import Memory, make_emotional_tag
from emotional_memory.resonance import build_resonance_links
from emotional_memory.retrieval import (
    adaptive_weights,
    affective_prediction_error,
    reconsolidate,
    retrieval_score,
)
from emotional_memory.state import AffectiveState
from emotional_memory.stimmung import StimmungField

logger = logging.getLogger(__name__)


class AsyncEmotionalMemory:
    """Async-native EmotionalMemory using the AFT pipeline.

    All public methods are coroutines.  The internal affective state machine
    (AffectiveState, StimmungField, AffectiveMomentum) updates synchronously
    between awaits — state is not shared across concurrent calls, so callers
    should serialise encode/retrieve calls that must observe a consistent state.
    """

    __slots__ = ("_appraisal_engine", "_config", "_embedder", "_state", "_store")

    def __init__(
        self,
        store: AsyncMemoryStore,
        embedder: AsyncEmbedder,
        appraisal_engine: AsyncAppraisalEngine | None = None,
        config: EmotionalMemoryConfig | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._appraisal_engine = appraisal_engine
        self._config = config or EmotionalMemoryConfig()
        self._state = AffectiveState.initial()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(store={self._store!r})"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def encode(
        self,
        content: str,
        appraisal: AppraisalVector | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Encode content into emotional memory (async).

        Mirrors ``EmotionalMemory.encode`` with awaited I/O calls.
        """
        now = datetime.now(tz=UTC)
        logger.debug("encode start: content_len=%d", len(content))

        # Step 1: resolve affect
        if appraisal is None and self._appraisal_engine is not None:
            appraisal = await self._appraisal_engine.appraise(content, context=metadata)

        new_affect = (
            appraisal.to_core_affect() if appraisal is not None else self._state.core_affect
        )

        # Step 2: update affective state (sync — pure computation)
        self._state = self._state.update(
            new_affect,
            now=now,
            stimmung_alpha=self._config.stimmung_alpha,
            stimmung_decay=self._config.stimmung_decay,
        )

        # Step 3: build EmotionalTag (sync)
        cs = consolidation_strength(new_affect.arousal, self._state.stimmung.arousal)
        tag = make_emotional_tag(
            core_affect=self._state.core_affect,
            momentum=self._state.momentum,
            stimmung=self._state.stimmung,
            consolidation_strength=cs,
            appraisal=appraisal,
        )

        # Step 4: embed (async I/O)
        embedding = await self._embedder.embed(content)
        if embedding and bool(np.isnan(np.asarray(embedding)).any()):
            warnings.warn(
                f"Embedder returned NaN values for content (len={len(content)}). "
                "Semantic retrieval will be degraded for this memory.",
                stacklevel=2,
            )

        # Step 5: store (async I/O)
        memory = Memory.create(content=content, tag=tag, embedding=embedding, metadata=metadata)
        await self._store.save(memory)

        # Step 6: resonance links — pre-filter when store is large (async I/O)
        resonance_limit = (
            self._config.resonance.max_links * self._config.resonance.candidate_multiplier
        )
        store_size = await self._store.count()
        if store_size > resonance_limit and memory.embedding is not None:
            candidates = await self._store.search_by_embedding(memory.embedding, resonance_limit)
        else:
            candidates = await self._store.list_all()

        links = build_resonance_links(memory, candidates, self._config.resonance)
        if links:
            updated_tag = tag.model_copy(update={"resonance_links": links})
            memory = memory.model_copy(update={"tag": updated_tag})
            await self._store.update(memory)
            logger.debug("encode resonance: id=%s links=%d", memory.id, len(links))

        logger.debug(
            "encode stored: id=%s valence=%.3f arousal=%.3f cs=%.3f",
            memory.id,
            tag.core_affect.valence,
            tag.core_affect.arousal,
            cs,
        )
        return memory

    async def retrieve(self, query: str, top_k: int = 5) -> list[Memory]:
        """Retrieve the top-k most relevant memories for the query (async).

        Mirrors ``EmotionalMemory.retrieve`` with awaited I/O calls.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        now = datetime.now(tz=UTC)
        store_count = await self._store.count()
        logger.debug(
            "retrieve start: query_len=%d top_k=%d store=%d", len(query), top_k, store_count
        )
        query_embedding = await self._embedder.embed(query)

        # G2 pre-filter (async)
        rc = self._config.retrieval
        candidate_limit = top_k * rc.candidate_multiplier
        store_size = await self._store.count()
        if store_size > candidate_limit:
            candidates = await self._store.search_by_embedding(query_embedding, candidate_limit)
        else:
            candidates = await self._store.list_all()

        if not candidates:
            return []

        # G1 two-pass scoring (sync — pure computation)
        # Compute adaptive weights once — invariant across all candidates
        weights = adaptive_weights(
            self._state.stimmung, rc.base_weights, rc.adaptive_weights_config
        )

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
                    precomputed_weights=weights,
                )
                scored.append((score, mem))
            scored.sort(key=lambda t: t[0], reverse=True)
            return scored

        pass1 = _score_all([])
        active_ids = [mem.id for _, mem in pass1[:top_k]]

        # Skip Pass 2 when no candidate has resonance links targeting the active set
        active_set = set(active_ids)
        needs_pass2 = any(
            any(link.target_id in active_set for link in mem.tag.resonance_links)
            for _, mem in pass1
        )
        pass2 = _score_all(active_ids) if needs_pass2 else pass1
        top = [mem for _, mem in pass2[:top_k]]

        # Reconsolidation + retrieval count update (async store writes)
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
                logger.debug("reconsolidate: id=%s ape=%.3f", mem.id, ape)
            await self._store.update(updated)
            result.append(updated)

        logger.debug("retrieve done: returned=%d candidates=%d", len(result), len(candidates))
        return result

    async def encode_batch(
        self,
        contents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[Memory]:
        """Encode multiple contents using a single embed_batch call (async)."""
        if metadata is not None and len(metadata) != len(contents):
            raise ValueError(
                f"metadata length ({len(metadata)}) must match contents length ({len(contents)})"
            )
        embeddings = await self._embedder.embed_batch(contents)
        results = []
        for i, (content, embedding) in enumerate(zip(contents, embeddings, strict=True)):
            meta = metadata[i] if metadata else None
            now = datetime.now(tz=UTC)

            appraisal: AppraisalVector | None = None
            if self._appraisal_engine is not None:
                appraisal = await self._appraisal_engine.appraise(content, context=meta)

            new_affect = (
                appraisal.to_core_affect() if appraisal is not None else self._state.core_affect
            )
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

            if embedding and bool(np.isnan(np.asarray(embedding)).any()):
                warnings.warn(
                    f"Embedder returned NaN values for content[{i}] (len={len(content)}). "
                    "Semantic retrieval will be degraded for this memory.",
                    stacklevel=2,
                )
            memory = Memory.create(content=content, tag=tag, embedding=embedding, metadata=meta)
            await self._store.save(memory)

            resonance_limit = (
                self._config.resonance.max_links * self._config.resonance.candidate_multiplier
            )
            store_size = await self._store.count()
            if store_size > resonance_limit and memory.embedding is not None:
                candidates = await self._store.search_by_embedding(
                    memory.embedding, resonance_limit
                )
            else:
                candidates = await self._store.list_all()

            links = build_resonance_links(memory, candidates, self._config.resonance)
            if links:
                updated_tag = tag.model_copy(update={"resonance_links": links})
                memory = memory.model_copy(update={"tag": updated_tag})
                await self._store.update(memory)

            results.append(memory)
        return results

    async def delete(self, memory_id: str) -> None:
        """Remove a memory from the store."""
        await self._store.delete(memory_id)

    async def get(self, memory_id: str) -> Memory | None:
        """Look up a single memory by ID, or None if not found."""
        return await self._store.get(memory_id)

    async def list_all(self) -> list[Memory]:
        """Return all memories in the store."""
        return await self._store.list_all()

    async def count(self) -> int:
        """Return the number of memories in the store."""
        return await self._store.count()

    def get_state(self) -> AffectiveState:
        """Return a copy of the current affective state."""
        return self._state.model_copy()

    def set_affect(self, core_affect: CoreAffect) -> None:
        """Manually inject a CoreAffect, updating state synchronously."""
        self._state = self._state.update(
            core_affect,
            stimmung_alpha=self._config.stimmung_alpha,
            stimmung_decay=self._config.stimmung_decay,
        )

    def save_state(self) -> dict[str, Any]:
        """Serialise the current affective state for persistence."""
        return self._state.snapshot()

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore a previously saved affective state."""
        self._state = AffectiveState.restore(data)

    async def prune(self, threshold: float = 0.05) -> int:
        """Remove memories whose effective strength has fallen below *threshold* (async).

        Args:
            threshold: Minimum acceptable strength in [0, 1].  Defaults to 0.05 (5%).

        Returns:
            Number of memories removed.
        """
        from emotional_memory.decay import compute_effective_strength

        now = datetime.now(tz=UTC)
        removed = 0
        for memory in await self._store.list_all():
            if compute_effective_strength(memory.tag, now, self._config.decay) < threshold:
                await self._store.delete(memory.id)
                removed += 1
        return removed

    async def export_memories(self) -> list[dict[str, Any]]:
        """Export all memories as a list of JSON-serialisable dicts (async)."""
        memories = await self._store.list_all()
        return [json.loads(m.model_dump_json()) for m in memories]

    async def import_memories(self, data: list[dict[str, Any]], *, overwrite: bool = False) -> int:
        """Import memories from a list of dicts produced by ``export_memories()`` (async).

        Args:
            data:      List of memory dicts (as returned by ``export_memories()``).
            overwrite: When True, existing memories with the same ID are replaced.
                       When False (default), duplicates are skipped silently.

        Returns:
            Number of memories actually written.
        """
        written = 0
        for item in data:
            memory = Memory.model_validate(item)
            existing = await self._store.get(memory.id)
            if not overwrite and existing is not None:
                continue
            await self._store.save(memory)
            written += 1
        return written

    def get_current_stimmung(self, now: datetime | None = None) -> StimmungField:
        """Return Stimmung regressed to ``now`` without modifying state."""
        if now is None:
            now = datetime.now(tz=UTC)
        if self._config.stimmung_decay is not None:
            return self._state.stimmung.regress(now, self._config.stimmung_decay)
        return self._state.stimmung

    async def close(self) -> None:
        """Release resources held by the underlying store, if supported.

        Awaits ``store.close()`` when available; falls back to sync ``close()``
        via ``asyncio.to_thread``; silently skips stores without cleanup.
        """
        close = getattr(self._store, "close", None)
        if close is None:
            return
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(close):
            await close()
        else:
            await asyncio.to_thread(close)

    async def __aenter__(self) -> AsyncEmotionalMemory:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
