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
from emotional_memory.categorize import label_tag
from emotional_memory.engine import EmotionalMemoryConfig
from emotional_memory.interfaces_async import AsyncAppraisalEngine, AsyncEmbedder, AsyncMemoryStore
from emotional_memory.models import Memory, ResonanceLink, make_emotional_tag
from emotional_memory.mood import MoodField
from emotional_memory.resonance import (
    build_resonance_links,
    hebbian_strengthen,
    spreading_activation,
)
from emotional_memory.retrieval import (
    adaptive_weights,
    compute_ape,
    reconsolidate,
    retrieval_score,
    update_prediction,
)
from emotional_memory.state import AffectiveState

logger = logging.getLogger(__name__)


class AsyncEmotionalMemory:
    """Async-native EmotionalMemory using the AFT pipeline.

    All public methods are coroutines.  The internal affective state machine
    (AffectiveState, MoodField, AffectiveMomentum) updates synchronously
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
        # Dual-path (LeDoux 1996): when dual_path_encoding is True and an explicit
        # appraisal was NOT supplied, skip appraisal and mark tag as pending.
        use_fast_path = (
            self._config.dual_path_encoding
            and appraisal is None
            and self._appraisal_engine is not None
        )

        if not use_fast_path and appraisal is None and self._appraisal_engine is not None:
            appraisal = await self._appraisal_engine.appraise(content, context=metadata)

        new_affect = (
            appraisal.to_core_affect() if appraisal is not None else self._state.core_affect
        )

        # Step 2: update affective state (sync — pure computation)
        self._state = self._state.update(
            new_affect,
            now=now,
            mood_alpha=self._config.mood_alpha,
            mood_decay=self._config.mood_decay,
        )

        # Step 3: build EmotionalTag (sync)
        cs = consolidation_strength(new_affect.arousal, self._state.mood.arousal)
        tag = make_emotional_tag(
            core_affect=self._state.core_affect,
            momentum=self._state.momentum,
            mood=self._state.mood,
            consolidation_strength=cs,
            appraisal=appraisal,
        )

        # Mark pending_appraisal for fast-path memories
        if use_fast_path:
            tag = tag.model_copy(update={"pending_appraisal": True})

        # Auto-categorize: attach Plutchik EmotionLabel
        if self._config.auto_categorize:
            tag = label_tag(tag)

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

            # Bidirectional links: add backward links on each target memory
            for link in links:
                target_mem = await self._store.get(link.target_id)
                if target_mem is None:
                    continue
                backward = ResonanceLink(
                    source_id=link.target_id,
                    target_id=memory.id,
                    strength=link.strength,
                    link_type=link.link_type,
                )
                existing = list(target_mem.tag.resonance_links)
                max_links = self._config.resonance.max_links
                if len(existing) >= max_links:
                    weakest_idx = min(range(len(existing)), key=lambda i: existing[i].strength)
                    if backward.strength <= existing[weakest_idx].strength:
                        continue
                    existing[weakest_idx] = backward
                else:
                    existing.append(backward)
                updated_target = target_mem.model_copy(
                    update={"tag": target_mem.tag.model_copy(update={"resonance_links": existing})}
                )
                await self._store.update(updated_target)

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

        # G1 — spreading activation (Collins & Loftus, 1975)
        # Compute adaptive weights once — invariant across all candidates
        weights = adaptive_weights(self._state.mood, rc.base_weights, rc.adaptive_weights_config)

        def _score_all(activation_map: dict[str, float]) -> list[tuple[float, Memory]]:
            scored = []
            for mem in candidates:
                score = retrieval_score(
                    query_embedding=query_embedding,
                    query_affect=self._state.core_affect,
                    current_mood=self._state.mood,
                    current_momentum=self._state.momentum,
                    memory=mem,
                    activation_map=activation_map,
                    now=now,
                    decay_config=self._config.decay,
                    retrieval_config=rc,
                    precomputed_weights=weights,
                )
                scored.append((score, mem))
            scored.sort(key=lambda t: t[0], reverse=True)
            return scored

        # Pass 1: score without resonance to identify seed memories
        pass1 = _score_all({})
        seed_ids = {mem.id for _, mem in pass1[:top_k]}

        # Compute multi-hop activation map from seeds through the link graph
        act_map = spreading_activation(
            seed_ids, candidates, self._config.resonance.propagation_hops
        )

        # Pass 2: re-score with activation map (skip when no activation spread)
        pass2 = _score_all(act_map) if act_map else pass1
        top = [mem for _, mem in pass2[:top_k]]

        # Reconsolidation + retrieval count update (async store writes)
        cfg = rc
        result = []
        for mem in top:
            tag = mem.tag.model_copy(
                update={
                    "last_retrieved": now,
                    "retrieval_count": mem.tag.retrieval_count + 1,
                }
            )
            ape = compute_ape(tag, self._state.core_affect)

            # APE-gated reconsolidation window (Nader & Schiller, 2000)
            # HIGH APE: open window + reconsolidate
            if ape > cfg.ape_threshold and tag.window_opened_at is None:
                tag = reconsolidate(
                    tag, self._state.core_affect, ape, cfg.reconsolidation_learning_rate
                )
                tag = tag.model_copy(update={"window_opened_at": now})
                logger.debug("reconsolidate: id=%s ape=%.3f (window opened)", mem.id, ape)

            # WITHIN OPEN WINDOW: reconsolidate regardless of APE
            elif tag.window_opened_at is not None:
                elapsed = (now - tag.window_opened_at).total_seconds()
                if elapsed <= cfg.reconsolidation_window_seconds:
                    tag = reconsolidate(
                        tag, self._state.core_affect, ape, cfg.reconsolidation_learning_rate
                    )
                    logger.debug("reconsolidate: id=%s ape=%.3f (within window)", mem.id, ape)
                else:
                    # Window expired — close it
                    tag = tag.model_copy(update={"window_opened_at": None})

            # Pearce-Hall predictive learning: always update prediction
            tag = update_prediction(tag, self._state.core_affect, ape)

            updated = mem.model_copy(update={"tag": tag})
            await self._store.update(updated)
            result.append(updated)

        # Hebbian co-retrieval strengthening (Hebb, 1949)
        increment = self._config.resonance.hebbian_increment
        if increment > 0.0 and len(result) > 1:
            co_ids = {m.id for m in result}
            for i, mem in enumerate(result):
                new_links = hebbian_strengthen(mem, co_ids - {mem.id}, increment)
                if new_links != list(mem.tag.resonance_links):
                    strengthened = mem.model_copy(
                        update={"tag": mem.tag.model_copy(update={"resonance_links": new_links})}
                    )
                    await self._store.update(strengthened)
                    result[i] = strengthened
                    logger.debug("hebbian: id=%s links_strengthened=%d", mem.id, len(new_links))

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

            # Resolve appraisal per item (mirrors encode())
            # Dual-path: skip appraisal on fast path
            use_fast_path = self._config.dual_path_encoding and self._appraisal_engine is not None
            appraisal: AppraisalVector | None = None
            if not use_fast_path and self._appraisal_engine is not None:
                appraisal = await self._appraisal_engine.appraise(content, context=meta)

            new_affect = (
                appraisal.to_core_affect() if appraisal is not None else self._state.core_affect
            )
            self._state = self._state.update(
                new_affect,
                now=now,
                mood_alpha=self._config.mood_alpha,
                mood_decay=self._config.mood_decay,
            )

            cs = consolidation_strength(new_affect.arousal, self._state.mood.arousal)
            tag = make_emotional_tag(
                core_affect=self._state.core_affect,
                momentum=self._state.momentum,
                mood=self._state.mood,
                consolidation_strength=cs,
                appraisal=appraisal,
            )

            # Mark pending_appraisal for fast-path memories
            if use_fast_path:
                tag = tag.model_copy(update={"pending_appraisal": True})

            # Auto-categorize: attach Plutchik EmotionLabel
            if self._config.auto_categorize:
                tag = label_tag(tag)

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

                # Bidirectional links: add backward links on each target memory
                for link in links:
                    target_mem = await self._store.get(link.target_id)
                    if target_mem is None:
                        continue
                    backward = ResonanceLink(
                        source_id=link.target_id,
                        target_id=memory.id,
                        strength=link.strength,
                        link_type=link.link_type,
                    )
                    existing = list(target_mem.tag.resonance_links)
                    max_links = self._config.resonance.max_links
                    if len(existing) >= max_links:
                        weakest_idx = min(range(len(existing)), key=lambda i: existing[i].strength)
                        if backward.strength <= existing[weakest_idx].strength:
                            continue
                        existing[weakest_idx] = backward
                    else:
                        existing.append(backward)
                    updated_target = target_mem.model_copy(
                        update={
                            "tag": target_mem.tag.model_copy(update={"resonance_links": existing})
                        }
                    )
                    await self._store.update(updated_target)

            results.append(memory)
        return results

    async def elaborate(self, memory_id: str) -> Memory | None:
        """Run full appraisal on a fast-path (pending) memory and blend core_affect.

        Async version of ``EmotionalMemory.elaborate()``.

        Args:
            memory_id: ID of the memory to elaborate.

        Returns:
            The updated Memory, or None if the memory does not exist,
            is not pending appraisal, or no appraisal engine is configured.
        """
        mem = await self._store.get(memory_id)
        if mem is None:
            return None
        if not mem.tag.pending_appraisal:
            return None
        if self._appraisal_engine is None:
            return None

        appraisal = await self._appraisal_engine.appraise(mem.content)
        appraised_affect = appraisal.to_core_affect()

        # Blend: elaboration_learning_rate controls how much the appraised
        # affect replaces the raw fast-path affect (30% raw, 70% appraised).
        lr = self._config.elaboration_learning_rate
        blended_affect = mem.tag.core_affect.lerp(appraised_affect, lr)

        # Use appraised arousal for consolidation_strength
        new_cs = consolidation_strength(appraised_affect.arousal, mem.tag.mood_snapshot.arousal)

        now = datetime.now(tz=UTC)
        updated_tag = mem.tag.model_copy(
            update={
                "core_affect": blended_affect,
                "appraisal": appraisal,
                "consolidation_strength": new_cs,
                "pending_appraisal": False,
                "window_opened_at": now,
            }
        )
        updated_mem = mem.model_copy(update={"tag": updated_tag})
        await self._store.update(updated_mem)
        logger.debug(
            "elaborate: id=%s blended_valence=%.3f blended_arousal=%.3f",
            memory_id,
            blended_affect.valence,
            blended_affect.arousal,
        )
        return updated_mem

    async def elaborate_pending(self) -> list[Memory]:
        """Elaborate all memories with pending_appraisal=True (async).

        Returns:
            List of updated Memory objects (only those that were elaborated).
        """
        results = []
        for mem in await self._store.list_all():
            if mem.tag.pending_appraisal:
                updated = await self.elaborate(mem.id)
                if updated is not None:
                    results.append(updated)
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
            mood_alpha=self._config.mood_alpha,
            mood_decay=self._config.mood_decay,
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

    def get_current_mood(self, now: datetime | None = None) -> MoodField:
        """Return the mood field regressed to ``now`` without modifying state."""
        if now is None:
            now = datetime.now(tz=UTC)
        if self._config.mood_decay is not None:
            return self._state.mood.regress(now, self._config.mood_decay)
        return self._state.mood

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
