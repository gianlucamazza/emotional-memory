"""Opt-in, offline longitudinal replay using the existing realistic dataset/runner.

The scoped clock substitution is benchmark-only and must run in a dedicated,
single-threaded process. Production engine time and retrieval semantics are unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

if TYPE_CHECKING:
    from emotional_memory import Embedder

from benchmarks.realistic.adapters.base import (
    ReplayAdapter,
    ReplayRetrievedItem,
    ReplaySessionEnd,
    ReplaySessionStart,
    TokenHashEmbedder,
)
from benchmarks.realistic.evaluation import reproducibility_manifest, state_divergence
from emotional_memory import CoreAffect, EmotionalMemory, EmotionalMemoryConfig, InMemoryStore
from emotional_memory.engine_shared import is_query_affect_neutral, semantic_only_weights

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
ARMS = ("cosine", "gated", "always_on")
DEFAULT_SHARED_HISTORY_EMBEDDER = "hash"
DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HASH_EMBEDDER_VERSION = "sha256-token-hash"
SEMANTIC_EMBEDDER_ALIASES: dict[str, str] = {
    "sbert-mini": DEFAULT_SEMANTIC_MODEL,
    "sentence-transformers": DEFAULT_SEMANTIC_MODEL,
    "e5-small-v2": "intfloat/e5-small-v2",
    "sbert-bge": "BAAI/bge-small-en-v1.5",
    "multilingual-e5-small": "intfloat/multilingual-e5-small",
}


def resolve_hash_salts(
    hash_salt: Sequence[str] | None = None,
    hash_salts: int | None = None,
) -> list[str]:
    """Resolve CLI salt flags to the list of salts used for one invocation.

    No flags → a single empty salt (byte-identical to the unsalted hash embedder).
    ``--hash-salts K`` → ``["0", "1", ..., "K-1"]``.
    ``--hash-salt`` may be repeated; it cannot be combined with ``--hash-salts``.
    """
    if hash_salt and hash_salts is not None:
        raise ValueError("Pass --hash-salt or --hash-salts, not both")
    if hash_salts is not None:
        if hash_salts < 1:
            raise ValueError("--hash-salts must be >= 1")
        return [str(index) for index in range(hash_salts)]
    if hash_salt:
        return list(hash_salt)
    return [""]


def semantic_model_for(embedder_name: str, embedder_model: str | None = None) -> str:
    """Return the Hugging Face model id for a semantic shared-history embedder."""
    if embedder_model:
        return embedder_model
    if embedder_name in SEMANTIC_EMBEDDER_ALIASES:
        return SEMANTIC_EMBEDDER_ALIASES[embedder_name]
    raise ValueError(
        f"Unknown shared-history embedder {embedder_name!r}. "
        "Use 'hash' or a sentence-transformers alias "
        f"({', '.join(sorted(SEMANTIC_EMBEDDER_ALIASES))}) "
        "with optional --embedder-model."
    )


def build_shared_history_embedder(
    embedder_name: str = DEFAULT_SHARED_HISTORY_EMBEDDER,
    *,
    embedder_model: str | None = None,
    hash_salt: str = "",
) -> Any:
    """Build the embedder for one shared-history run. Semantic models are CPU-only."""
    if embedder_name == "hash":
        return TokenHashEmbedder(salt=hash_salt)
    model_name = semantic_model_for(embedder_name, embedder_model)
    try:
        from emotional_memory.embedders.sentence_transformers import (
            SentenceTransformerEmbedder,
        )
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for semantic shared-history embedders.\n"
            "Install with: pip install 'emotional-memory[sentence-transformers]'"
        ) from exc
    return SentenceTransformerEmbedder(model_name, device="cpu")


def describe_embedder(
    embedder: Any,
    *,
    embedder_name: str,
    embedder_model: str | None = None,
    hash_salts: Sequence[str],
) -> dict[str, Any]:
    """Receipt metadata for the embedder actually used."""
    info: dict[str, Any] = {
        "name": embedder_name,
        "class": type(embedder).__name__,
        "hash_salts": list(hash_salts),
    }
    if isinstance(embedder, TokenHashEmbedder):
        info["dimension"] = embedder._dim
        info["version"] = HASH_EMBEDDER_VERSION
        info["model"] = None
        info["device"] = None
        return info
    model_name = getattr(embedder, "_model_name", None)
    info["model"] = embedder_model or model_name
    info["device"] = getattr(embedder, "_device", "cpu")
    try:
        info["version"] = version("sentence-transformers")
    except PackageNotFoundError:
        info["version"] = None
    return info


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@contextmanager
def controlled_runtime(now: datetime, memory_id: str) -> Iterator[None]:
    """Freeze library clocks and IDs only for one synchronous benchmark operation."""

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.replace(tzinfo=None) if tz is None else now.astimezone(tz)

    with ExitStack() as stack:
        for module in ("engine", "models", "mood", "state"):
            stack.enter_context(patch(f"emotional_memory.{module}.datetime", FrozenDateTime))
        stack.enter_context(
            patch("emotional_memory.models.uuid", SimpleNamespace(uuid4=lambda: memory_id))
        )
        yield


class SharedHistoryAdapter(ReplayAdapter):
    """Same empty checkpoint and external events; all retrieval writes stay local."""

    supports_persisted_state = True
    supports_explanations = False

    def __init__(
        self,
        arm: str,
        scenario_id: str,
        seed: int,
        config: EmotionalMemoryConfig | None = None,
        *,
        read_only_retrieves: bool = False,
        embedder: Embedder | None = None,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"Unsupported arm: {arm}")
        self.name = arm
        self.scenario_id = scenario_id
        self.seed = seed
        self.read_only_retrieves = read_only_retrieves
        self.embedder = embedder if embedder is not None else TokenHashEmbedder()
        self.config = (config or EmotionalMemoryConfig()).model_copy(deep=True)
        self.reset()

    def reset(self) -> None:
        self.index = 0
        self.sessions = 0
        self.trace: list[dict[str, Any]] = []
        with controlled_runtime(EPOCH, "unused"):
            self.engine = EmotionalMemory(
                InMemoryStore(), self.embedder, config=self.config.model_copy(deep=True)
            )
        self.checkpoint = self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                {"memories": self.engine.export_memories(), "state": self.engine.save_state()}
            )
        )

    @contextmanager
    def operation(self, kind: str, payload: dict[str, Any]) -> Iterator[None]:
        self.index += 1
        now = EPOCH + timedelta(seconds=self.index)
        memory_id = str(uuid5(NAMESPACE_URL, f"{self.seed}:{self.scenario_id}:{self.index}"))
        with controlled_runtime(now, memory_id):
            yield
        snapshot = self.snapshot()
        self.trace.append(
            {
                "index": self.index,
                "kind": kind,
                "timestamp": now.isoformat(),
                "input": deepcopy(payload),
                "input_sha256": digest(payload),
                "snapshot": snapshot,
                "state_sha256": digest(snapshot),
            }
        )

    def begin_session(self, session_id: str) -> ReplaySessionStart:
        with self.operation("begin_session", {"session_id": session_id}):
            # Exercise the public persistence round trip at each boundary.
            self.engine.load_state(self.engine.save_state())
            result = ReplaySessionStart(
                state_loaded=self.sessions > 0,
                memory_count_start=len(self.engine),
                mood_start=self.engine.get_state().mood.model_dump(mode="json"),
            )
            self.sessions += 1
        return result

    def encode(
        self,
        *,
        memory_alias: str,
        content: str,
        valence: float,
        arousal: float,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "memory_alias": memory_alias,
            "content": content,
            "valence": valence,
            "arousal": arousal,
            "metadata": metadata or {},
        }
        with self.operation("encode", payload):
            self.engine.set_affect(CoreAffect(valence=valence, arousal=arousal))
            memory = self.engine.encode(
                content, metadata={**deepcopy(metadata or {}), "scenario_memory_id": memory_alias}
            )
        return memory.id

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        valence: float | None = None,
        arousal: float | None = None,
    ) -> list[ReplayRetrievedItem]:
        if valence is None or arousal is None:
            raise ValueError("Shared-history queries require explicit recorded query affect")
        payload = {"query": query, "top_k": top_k, "valence": valence, "arousal": arousal}
        with self.operation("retrieve", payload):
            checkpoint = self.snapshot() if self.read_only_retrieves else None
            query_affect = CoreAffect(valence=valence, arousal=arousal)
            use_cosine = self.name == "cosine" or (
                self.name == "gated"
                and is_query_affect_neutral(valence, self.config.query_affect_gate_tau)
            )
            memories = self.engine.retrieve_with_explanations(
                query,
                top_k=top_k,
                query_affect=query_affect,
                precomputed_weights=semantic_only_weights() if use_cosine else None,
            )
            if checkpoint is not None:
                self.engine.import_memories(checkpoint["memories"], overwrite=True)
                self.engine.load_state(checkpoint["state"])
                if self.snapshot() != checkpoint:
                    raise RuntimeError("Read-only retrieve left store or affective state changed")
        return [
            ReplayRetrievedItem(id=m.memory.id, text=m.memory.content, score=m.score)
            for m in memories
        ]

    def end_session(self) -> ReplaySessionEnd:
        with self.operation("end_session", {}):
            result = ReplaySessionEnd(
                memory_count_end=len(self.engine),
                mood_end=self.engine.get_state().mood.model_dump(mode="json"),
            )
        return result

    def close(self) -> None:
        self.engine.close()


def _pairwise_inputs(systems: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy reports so repeated query ids across scenarios stay distinct."""
    copied = deepcopy(systems)
    for system in copied:
        for scenario in system["scenarios"]:
            scenario_id = scenario["scenario_id"]
            for session in scenario["sessions"]:
                session_id = session["session_id"]
                for position, query in enumerate(session["queries"]):
                    query["query_id"] = (
                        f"{scenario_id}:{session_id}:{position}:{query['query_id']}"
                    )
    return copied


def _run_shared_history_once(
    dataset,
    *,
    seed: int,
    n_bootstrap: int,
    top_k: int | None,
    config: EmotionalMemoryConfig | None,
    dataset_path: Path | None,
    read_only_retrieves: bool,
    embedder: Embedder,
    embedder_name: str,
    embedder_model: str | None,
    hash_salts: Sequence[str],
) -> dict[str, Any]:
    """Run isolated scenarios with a common empty checkpoint, no state resets at queries."""
    from benchmarks.realistic.runner import (
        DATASET,
        _aggregate_by_challenge_type,
        _aggregate_query_metrics,
        _build_pairwise_comparisons,
        build_protocol_metadata,
        dataset_declares_interleave,
        resolve_query_affect_gate_tau,
        run_system_on_scenario,
        validate_dataset_difficulty,
    )

    effective_config = config or EmotionalMemoryConfig()
    resolve_query_affect_gate_tau(dataset, effective_config)
    effective_k = dataset.default_top_k if top_k is None else top_k
    difficulty = validate_dataset_difficulty(dataset, top_k=effective_k)
    for scenario in dataset.scenarios:
        for session in scenario.sessions:
            if any(query.state is None for query in session.queries):
                raise ValueError("Shared-history queries require explicit recorded query affect")
    systems = []
    for arm in ARMS:
        scenarios = []
        for scenario in dataset.scenarios:
            adapter = SharedHistoryAdapter(
                arm,
                scenario.scenario_id,
                seed,
                config,
                read_only_retrieves=read_only_retrieves,
                embedder=embedder,
            )
            try:
                report = run_system_on_scenario(
                    adapter,
                    scenario,
                    default_top_k=effective_k,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    honor_session_order=True,
                )
                report.update(checkpoint=adapter.checkpoint, trace=adapter.trace)
                scenarios.append(report)
            finally:
                adapter.close()
        queries = [q for sc in scenarios for se in sc["sessions"] for q in se["queries"]]
        sessions = [se for sc in scenarios for se in sc["sessions"]]
        systems.append(
            {
                "system": arm,
                "supports_explanations": False,
                "supports_persisted_state": True,
                "scenarios": scenarios,
                "aggregate_metrics": {
                    **_aggregate_query_metrics(queries, n_bootstrap=n_bootstrap, seed=seed),
                    "stateful_session_rate": sum(s["state_loaded_from_store"] for s in sessions)
                    / max(1, len(sessions)),
                },
                "challenge_type_metrics": _aggregate_by_challenge_type(
                    queries, n_bootstrap=n_bootstrap, seed=seed
                ),
            }
        )
    protocol = build_protocol_metadata(
        dataset,
        system_names=list(ARMS),
        difficulty_profile=difficulty,
        dataset_path=DATASET if dataset_path is None else dataset_path,
    )
    interleaved = dataset_declares_interleave(dataset)
    protocol.update(
        mode="shared_history_timeline" if interleaved else "shared_history_v1",
        system_capabilities={
            arm: {"persisted_state": True, "explanations": False} for arm in ARMS
        },
        interpretation_guardrails=[
            "Offline scripted replay, not production or human validation.",
            "Recorded query affect is an oracle input, not evaluated appraisal quality.",
            "Cosine uses the same engine and write-back policy; only ranking differs.",
            "All arms retain retrieval updates; affective state may remain equal "
            "while tags diverge.",
        ],
    )
    if interleaved:
        protocol["event_order"] = "dataset_timeline"
    protocol["read_only_retrieves"] = read_only_retrieves
    embedder_info = describe_embedder(
        embedder,
        embedder_name=embedder_name,
        embedder_model=embedder_model,
        hash_salts=hash_salts,
    )
    protocol["embedder"] = embedder_info
    protocol["hash_salts"] = list(hash_salts)
    manifest = reproducibility_manifest(
        dataset,
        effective_config,
        seed=seed,
        top_k=effective_k,
        embedder=embedder_info,
    )
    manifest["query_affect_gate_tau"] = effective_config.query_affect_gate_tau
    protocol["manifest"] = manifest
    return {
        "manifest": manifest,
        "state_divergence": state_divergence(systems),
        "benchmark": dataset.name,
        "version": dataset.version,
        "top_k": effective_k,
        "difficulty_profile": difficulty,
        "protocol": protocol,
        "statistics": {
            "seed": seed,
            "n_bootstrap": n_bootstrap,
            "confidence": 0.95,
            "ci_method": "bootstrap_percentile",
        },
        "systems": systems,
        "pairwise_comparisons": _build_pairwise_comparisons(
            _pairwise_inputs(systems), baseline="cosine", seed=seed, n_bootstrap=n_bootstrap
        ),
    }


def run_shared_history(
    dataset,
    *,
    seed: int = 0,
    n_bootstrap: int = 1000,
    top_k: int | None = None,
    config=None,
    dataset_path: Path | None = None,
    read_only_retrieves: bool = False,
    embedder: Embedder | None = None,
    embedder_name: str = DEFAULT_SHARED_HISTORY_EMBEDDER,
    embedder_model: str | None = None,
    hash_salts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run isolated scenarios with a common empty checkpoint, no state resets at queries.

    ``hash_salts`` defaults to a single empty salt so the stock hash embedder is
    byte-identical to the unsalted path. Multiple salts produce one full result
    per salt under ``salt_runs``; the first salt is also expanded at top level
    so existing report writers keep working.
    """
    salts = list(hash_salts) if hash_salts is not None else [""]
    if not salts:
        salts = [""]
    if embedder is not None and len(salts) > 1:
        raise ValueError("An explicit embedder cannot be combined with multiple hash salts")

    runs: list[dict[str, Any]] = []
    for salt in salts:
        run_embedder = embedder
        if run_embedder is None:
            run_embedder = build_shared_history_embedder(
                embedder_name, embedder_model=embedder_model, hash_salt=salt
            )
        runs.append(
            _run_shared_history_once(
                dataset,
                seed=seed,
                n_bootstrap=n_bootstrap,
                top_k=top_k,
                config=config,
                dataset_path=dataset_path,
                read_only_retrieves=read_only_retrieves,
                embedder=run_embedder,
                embedder_name=embedder_name,
                embedder_model=embedder_model,
                hash_salts=[salt],
            )
        )
    result = deepcopy(runs[0]) if len(runs) > 1 else runs[0]
    result["protocol"]["hash_salts"] = list(salts)
    result["protocol"]["embedder"]["hash_salts"] = list(salts)
    result["manifest"]["embedder"]["hash_salts"] = list(salts)
    if len(runs) > 1:
        result["salt_runs"] = runs
    return result
