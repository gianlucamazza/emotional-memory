"""Adapters for the ES-MemEval retrieval benchmark (Addendum X2).

Both arms share the same embedder and the same document text (the session
transcript only — third-party emotion/topic/summary labels never enter the
semantic channel; see pre-registration §Protocol). Ranking is restricted to the
query's 50-candidate pool (upstream design, pools shared verbatim by both arms).

The AFT arm ingests the full 401-session bank once (encode-time direct-VAD
appraisal), then serves each query from a fresh pool-scoped engine built via the
public ``export_memories()``/``import_memories()`` API: embeddings and appraised
tags are reused (no re-appraisal, no re-embedding), resonance links pointing
outside the pool are ignored by construction (``spreading_activation`` builds
adjacency over pool candidates only), and production side effects (Hebbian
strengthening, APE-gated reconsolidation) stay active within the query without
leaking across queries.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from benchmarks.common.similarity import cosine
from benchmarks.esmemeval.dataset import EsmemSession
from emotional_memory import (
    DIRECT_VAD_SCHEMA,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
)
from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal import AppraisalEngine
from emotional_memory.appraisal_llm import (
    KeywordAppraisalEngine,
    LLMAppraisalConfig,
    LLMAppraisalEngine,
)
from emotional_memory.decay import DecayConfig
from emotional_memory.embedders import SentenceTransformerEmbedder
from emotional_memory.llm_http import OpenAICompatibleLLMConfig, make_httpx_llm

# Amendment A1 pattern (Addendum X): time-invariant stored strength; the
# arousal-gated consolidation strength (affect channel) stays active.
_TIME_INVARIANT_CONFIG = EmotionalMemoryConfig(
    decay=DecayConfig(base_decay=0.0, arousal_modulation=0.0, retrieval_boost=0.0),
)


class EsmemAdapter(ABC):
    """Ingest the full session bank once, then rank within per-query pools."""

    name: str = "unnamed"

    @abstractmethod
    def ingest(self, sessions: list[EsmemSession]) -> None: ...

    @abstractmethod
    def retrieve(self, query_text: str, pool_keys: Sequence[str], *, top_k: int) -> list[str]:
        """Return ranked session keys from ``pool_keys`` (length <= top_k)."""

    def close(self) -> None:
        """Release adapter resources; no-op by default."""
        return None


def _make_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder.make_bge_small()


class NaiveCosineEsmemAdapter(EsmemAdapter):
    """Pure semantic baseline: cosine over session-transcript embeddings."""

    name = "naive_cosine"

    def __init__(self, *, embedder: SentenceTransformerEmbedder | None = None) -> None:
        self._embedder = embedder if embedder is not None else _make_embedder()
        self._vectors: dict[str, list[float]] = {}

    def ingest(self, sessions: list[EsmemSession]) -> None:
        self._vectors = {s.key: self._embedder.embed(s.text) for s in sessions}

    def retrieve(self, query_text: str, pool_keys: Sequence[str], *, top_k: int) -> list[str]:
        qvec = self._embedder.embed(query_text)
        ranked = sorted(pool_keys, key=lambda k: cosine(qvec, self._vectors[k]), reverse=True)
        return list(ranked[:top_k])


class AFTQueryAppraisedEsmemAdapter(EsmemAdapter):
    """Primary arm: AFT, oracle-free end to end.

    Encode-time affect: direct-VAD appraisal of the session transcript.
    Retrieve-time affect: direct-VAD appraisal of the query text, passed via the
    public ``query_affect`` API (s3 override, no state mutation) — the
    production-reachable mechanism of Addendum T. Ranking within the query's
    candidate pool via a fresh pool-scoped engine (module docstring).
    """

    name = "aft_query_appraised"

    def __init__(
        self,
        *,
        dry_run: bool = False,
        embedder: SentenceTransformerEmbedder | None = None,
    ) -> None:
        self._appraiser = _make_appraiser(dry_run=dry_run)
        self._embedder = embedder if embedder is not None else _make_embedder()
        # key -> exported Memory dict (embedding + appraised tag + links).
        self._exported: dict[str, dict[str, Any]] = {}
        self._id_to_key: dict[str, str] = {}
        # key -> appraised (valence, arousal), for diagnostics D1/D2.
        self.encoded_affect: dict[str, tuple[float, float]] = {}
        # query text -> appraised affect, for the closure record.
        self.appraised_query_affect: dict[str, CoreAffect] = {}

    def ingest(self, sessions: list[EsmemSession]) -> None:
        engine = EmotionalMemory(
            store=InMemoryStore(),
            embedder=self._embedder,
            appraisal_engine=self._appraiser,
            config=_TIME_INVARIANT_CONFIG,
        )
        try:
            for s in sessions:
                stored = engine.encode(s.text, metadata={"key": s.key})
                ca = stored.tag.core_affect
                self.encoded_affect[s.key] = (ca.valence, ca.arousal)
            for record in engine.export_memories():
                key = str(record["metadata"]["key"])
                self._exported[key] = record
                self._id_to_key[str(record["id"])] = key
        finally:
            with contextlib.suppress(Exception):
                engine.close()

    def retrieve(self, query_text: str, pool_keys: Sequence[str], *, top_k: int) -> list[str]:
        # Fresh pool-scoped engine per query: no cross-query state, resonance
        # adjacency restricted to the pool, production side effects active
        # within the query (pre-registration §Protocol).
        engine = EmotionalMemory(
            store=InMemoryStore(),
            embedder=self._embedder,
            config=_TIME_INVARIANT_CONFIG,
        )
        try:
            engine.import_memories([self._exported[k] for k in pool_keys])
            engine.reset_state()
            ca = self._appraiser.appraise(query_text).to_core_affect()
            self.appraised_query_affect[query_text] = ca
            memories = engine.retrieve(query_text, top_k=top_k, query_affect=ca)
            return [self._id_to_key[m.id] for m in memories if m.id in self._id_to_key]
        finally:
            with contextlib.suppress(Exception):
                engine.close()


def _make_appraiser(*, dry_run: bool) -> AppraisalEngine:
    if dry_run:
        # Smoke mode: no LLM, no API key — keyword appraisal stands in so the
        # pipeline is exercised end to end. Never used for scored runs.
        return KeywordAppraisalEngine()
    cfg = OpenAICompatibleLLMConfig.from_env()
    if cfg is None:
        raise RuntimeError(
            "EMOTIONAL_MEMORY_LLM_API_KEY not set — required for the scored run "
            "(use --dry-run for the no-LLM smoke test)."
        )
    return LLMAppraisalEngine(
        llm=make_httpx_llm(cfg),
        config=LLMAppraisalConfig(
            cache_size=4096, fallback_on_error=True, appraisal_schema=DIRECT_VAD_SCHEMA
        ),
    )
