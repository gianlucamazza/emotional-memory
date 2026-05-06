"""Addendum G — Dual-path LLM appraisal study on affect-free dataset (Hg1).

Pre-registration: benchmarks/preregistration_addendum_g.md
Dataset:          benchmarks/datasets/realistic_recall_v3_noAF.json
Output:           benchmarks/appraisal_confound/results.hg1.{json,md,protocol.json}
Seed:             0  (frozen per pre-reg)

Systems
-------
aft_llm_dual  (confirmatory): AFT + LLMAppraisalEngine + dual_path_encoding=True.
               Affect inferred by LLM only — no preset valence/arousal injection.
naive_cosine  (control):      Semantic cosine, no affective state.
aft_neutral   (Hg2 expl.):   AFT with neutral CoreAffect(0,0.5,0.5), no LLM.
aft_llm_sync  (Hg3 expl.):   AFT + LLMAppraisalEngine, synchronous (not deferred).

Usage::

    uv run python -m benchmarks.appraisal_confound.runner_hg1
    uv run python -m benchmarks.appraisal_confound.runner_hg1 --embedder sbert-bge
    uv run python -m benchmarks.appraisal_confound.runner_hg1 --systems aft_llm_dual,naive_cosine
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from tqdm import tqdm

from benchmarks.common.statistics import (
    bootstrap_ci,
    ci_payload,
    cohens_d_paired,
    format_point_ci,
    paired_bootstrap_diff,
)
from benchmarks.realistic.adapters.base import (
    ReplayRetrievedItem,
    ReplaySessionEnd,
    ReplaySessionStart,
    TokenHashEmbedder,
    cosine_similarity,
)
from emotional_memory import (
    CoreAffect,
    Embedder,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    SQLiteAffectiveStateStore,
)

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
DEFAULT_DATASET = ROOT / "benchmarks" / "datasets" / "realistic_recall_v3_noAF.json"
DEFAULT_OUT_JSON = _HERE / "results.hg1.json"
DEFAULT_OUT_MD = _HERE / "results.hg1.md"
DEFAULT_OUT_PROTOCOL = _HERE / "results.hg1.protocol.json"

_PREREG_SEED = 0
_PREREG_N_BOOTSTRAP = 10_000
_HG1_ALPHA = 0.05
_HG1_EFFECT_THRESHOLD = 0.05

CHALLENGE_TYPES = frozenset(
    {
        "recency_confound",
        "semantic_confound",
        "affective_arc",
        "same_topic_distractor",
        "momentum_alignment",
    }
)

# ---------------------------------------------------------------------------
# Dataset models — no preset affect fields
# ---------------------------------------------------------------------------


class _NoAFEvent(BaseModel):
    memory_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class _NoAFQuery(BaseModel):
    query_id: str
    query: str
    expected_memory_ids: list[str]
    challenge_type: str
    top_k: int | None = None


class _NoAFSession(BaseModel):
    session_id: str
    description: str
    events: list[_NoAFEvent] = Field(default_factory=list)
    queries: list[_NoAFQuery] = Field(default_factory=list)


class _NoAFScenario(BaseModel):
    scenario_id: str
    description: str
    sessions: list[_NoAFSession]


class _NoAFDataset(BaseModel):
    name: str
    version: str
    description: str
    default_top_k: int = 3
    scenarios: list[_NoAFScenario]


def _load_dataset(path: Path) -> _NoAFDataset:
    return _NoAFDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _validate_dataset(dataset: _NoAFDataset, top_k: int) -> None:
    errors: list[str] = []
    for scenario in dataset.scenarios:
        encoded_ids: list[str] = []
        for session in scenario.sessions:
            encoded_ids.extend(ev.memory_id for ev in session.events)
            for q in session.queries:
                missing = [m for m in q.expected_memory_ids if m not in encoded_ids]
                if missing:
                    errors.append(f"{q.query_id}: expected memories not yet encoded: {missing}")
                if q.challenge_type not in CHALLENGE_TYPES:
                    errors.append(f"{q.query_id}: unknown challenge_type {q.challenge_type!r}")
                candidate_count = len(encoded_ids)
                if candidate_count <= top_k:
                    errors.append(
                        f"{q.query_id}: trivial candidate window "
                        f"({candidate_count} ≤ top_k={top_k})"
                    )
    has_semantic = any(
        q.challenge_type == "semantic_confound"
        for sc in dataset.scenarios
        for se in sc.sessions
        for q in se.queries
    )
    if not has_semantic:
        errors.append("Dataset must contain at least one semantic_confound query.")
    if errors:
        raise ValueError("Invalid noAF dataset:\n- " + "\n- ".join(errors))


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


def _require_llm() -> Any:
    from emotional_memory.llm_http import make_httpx_llm_from_env

    llm = make_httpx_llm_from_env()
    if llm is None:
        raise RuntimeError(
            "EMOTIONAL_MEMORY_LLM_API_KEY is not set. "
            "Run `make llm-config` to check the resolved LLM config, "
            "or set the variable in .env / environment before running this benchmark."
        )
    return llm


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _build_embedder(name: str | None) -> Embedder | None:
    if name is None or name == "hash":
        return None
    try:
        from emotional_memory.embedders import SentenceTransformerEmbedder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers required: pip install 'emotional-memory[sentence-transformers]'"
        ) from exc
    if name == "sbert-bge":
        return SentenceTransformerEmbedder.make_bge_small()
    if name == "multilingual-e5-small":
        return SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
    raise ValueError(f"Unknown embedder: {name!r}. Choices: hash, sbert-bge")


# ---------------------------------------------------------------------------
# Adapters — no preset affect injection
# ---------------------------------------------------------------------------


class _AFTNoAFBase:
    """Shared lifecycle for AFT noAF adapters (no valence/arousal inject)."""

    name: str = "aft_noAF_base"

    def __init__(self, workdir: Path, *, embedder: Embedder | None = None) -> None:
        self._workdir = workdir
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._embedder: Embedder = embedder if embedder is not None else TokenHashEmbedder()
        self._state_path = workdir / "state.sqlite"
        self._memories_path = workdir / "memories.json"
        self._engine: EmotionalMemory | None = None

    def reset(self) -> None:
        self.close()
        if self._state_path.exists():
            self._state_path.unlink()
        if self._memories_path.exists():
            self._memories_path.unlink()

    def _make_engine(self, state_store: SQLiteAffectiveStateStore) -> EmotionalMemory:
        return EmotionalMemory(
            store=InMemoryStore(),
            embedder=self._embedder,
            state_store=state_store,
        )

    def begin_session(self, session_id: str) -> ReplaySessionStart:
        state_store = SQLiteAffectiveStateStore(self._state_path)
        state_loaded = state_store.load() is not None
        engine = self._make_engine(state_store)
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
        self, *, memory_alias: str, content: str, metadata: dict[str, Any]
    ) -> str:  # overridden by subclasses
        raise NotImplementedError

    def retrieve(self, query: str, *, top_k: int) -> list[ReplayRetrievedItem]:
        engine = self._require_engine()
        explanations = engine.retrieve_with_explanations(query, top_k=top_k)
        return [
            ReplayRetrievedItem(
                id=item.memory.id,
                text=item.memory.content,
                score=item.score,
            )
            for item in explanations
        ]

    def end_session(self) -> ReplaySessionEnd:
        engine = self._require_engine()
        self._memories_path.write_text(
            json.dumps(engine.export_memories(), indent=2), encoding="utf-8"
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


class AFTLLMDualAdapter(_AFTNoAFBase):
    """AFT + LLMAppraisalEngine + dual_path_encoding=True. Hg1 primary system."""

    name = "aft_llm_dual"

    def __init__(self, workdir: Path, *, embedder: Embedder | None = None) -> None:
        super().__init__(workdir, embedder=embedder)
        self._llm = _require_llm()

    def _make_engine(self, state_store: SQLiteAffectiveStateStore) -> EmotionalMemory:
        from emotional_memory import LLMAppraisalEngine

        engine = EmotionalMemory(
            store=InMemoryStore(),
            embedder=self._embedder,
            state_store=state_store,
            config=EmotionalMemoryConfig(dual_path_encoding=True),
        )
        engine._appraisal_engine = LLMAppraisalEngine(self._llm)
        return engine

    def encode(self, *, memory_alias: str, content: str, metadata: dict[str, Any]) -> str:
        engine = self._require_engine()
        memory = engine.encode(content, metadata={**metadata, "scenario_memory_id": memory_alias})
        engine.elaborate(memory.id)
        return memory.id


class AFTNeutralAdapter(_AFTNoAFBase):
    """AFT with neutral CoreAffect(0,0.5,0.5) at every encode. Hg2 exploratory control."""

    name = "aft_neutral"
    _NEUTRAL = CoreAffect(valence=0.0, arousal=0.5, dominance=0.5)

    def encode(self, *, memory_alias: str, content: str, metadata: dict[str, Any]) -> str:
        engine = self._require_engine()
        engine.set_affect(self._NEUTRAL)
        memory = engine.encode(content, metadata={**metadata, "scenario_memory_id": memory_alias})
        return memory.id


class AFTLLMSyncAdapter(_AFTNoAFBase):
    """AFT + LLMAppraisalEngine synchronous (not deferred). Hg3 exploratory."""

    name = "aft_llm_sync"

    def __init__(self, workdir: Path, *, embedder: Embedder | None = None) -> None:
        super().__init__(workdir, embedder=embedder)
        self._llm = _require_llm()

    def _make_engine(self, state_store: SQLiteAffectiveStateStore) -> EmotionalMemory:
        from emotional_memory import LLMAppraisalEngine

        engine = EmotionalMemory(
            store=InMemoryStore(),
            embedder=self._embedder,
            state_store=state_store,
            config=EmotionalMemoryConfig(dual_path_encoding=False),
        )
        engine._appraisal_engine = LLMAppraisalEngine(self._llm)
        return engine

    def encode(self, *, memory_alias: str, content: str, metadata: dict[str, Any]) -> str:
        engine = self._require_engine()
        memory = engine.encode(content, metadata={**metadata, "scenario_memory_id": memory_alias})
        return memory.id


class NaiveCosineNoAFAdapter:
    """Semantic-only cosine baseline. No affective state. Control for Hg1."""

    name = "naive_cosine"

    def __init__(self, *, embedder: Embedder | None = None) -> None:
        self._embedder: Embedder = embedder if embedder is not None else TokenHashEmbedder()
        self._store: list[tuple[str, str, list[float]]] = []

    def reset(self) -> None:
        self._store = []

    def begin_session(self, session_id: str) -> ReplaySessionStart:
        return ReplaySessionStart(state_loaded=False, memory_count_start=len(self._store))

    def encode(self, *, memory_alias: str, content: str, metadata: dict[str, Any]) -> str:
        item_id = str(uuid.uuid4())
        self._store.append((item_id, content, self._embedder.embed(content)))
        return item_id

    def retrieve(self, query: str, *, top_k: int) -> list[ReplayRetrievedItem]:
        qvec = self._embedder.embed(query)
        scored = sorted(
            ((iid, txt, cosine_similarity(qvec, emb)) for iid, txt, emb in self._store),
            key=lambda x: x[2],
            reverse=True,
        )
        return [ReplayRetrievedItem(id=iid, text=txt, score=sc) for iid, txt, sc in scored[:top_k]]

    def end_session(self) -> ReplaySessionEnd:
        return ReplaySessionEnd(memory_count_end=len(self._store))

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Scenario runner (noAF: no affect injection)
# ---------------------------------------------------------------------------


def _run_scenario(
    adapter: _AFTNoAFBase | NaiveCosineNoAFAdapter,
    scenario: _NoAFScenario,
    *,
    default_top_k: int,
) -> dict[str, Any]:
    alias_to_actual: dict[str, str] = {}
    actual_to_alias: dict[str, str] = {}
    actual_id_order: list[str] = []
    session_reports: list[dict[str, Any]] = []

    for session in scenario.sessions:
        session_start = adapter.begin_session(session.session_id)
        session_report: dict[str, Any] = {
            "session_id": session.session_id,
            "state_loaded_from_store": session_start.state_loaded,
            "memory_count_start": session_start.memory_count_start,
            "queries": [],
        }

        for event in session.events:
            actual_id = adapter.encode(
                memory_alias=event.memory_id,
                content=event.content,
                metadata=event.metadata,
            )
            alias_to_actual[event.memory_id] = actual_id
            actual_to_alias[actual_id] = event.memory_id
            actual_id_order.append(actual_id)

        for query in session.queries:
            effective_top_k = query.top_k or default_top_k
            retrieved = adapter.retrieve(query.query, top_k=effective_top_k)
            expected_actual_ids = {
                alias_to_actual[m] for m in query.expected_memory_ids if m in alias_to_actual
            }
            retrieved_ids = [item.id for item in retrieved]
            top1_hit = bool(retrieved_ids) and retrieved_ids[0] in expected_actual_ids
            hit = any(rid in expected_actual_ids for rid in retrieved_ids)
            recency_order = list(reversed(actual_id_order))
            expected_recency_ranks = {
                actual_to_alias[mid]: recency_order.index(mid) + 1
                for mid in expected_actual_ids
                if mid in recency_order
            }
            session_report["queries"].append(
                {
                    "query_id": query.query_id,
                    "challenge_type": query.challenge_type,
                    "top_k": effective_top_k,
                    "candidate_count": len(actual_id_order),
                    "top1_hit": top1_hit,
                    "hit": hit,
                    "recency_would_hit_at_k": any(
                        rank <= effective_top_k for rank in expected_recency_ranks.values()
                    ),
                }
            )

        session_end = adapter.end_session()
        session_report["memory_count_end"] = session_end.memory_count_end
        session_reports.append(session_report)

    return {"scenario_id": scenario.scenario_id, "sessions": session_reports}


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

_ALL_SYSTEMS = ["aft_llm_dual", "naive_cosine", "aft_neutral", "aft_llm_sync"]
_LLM_SYSTEMS = {"aft_llm_dual", "aft_llm_sync"}


def _make_adapter(
    system_name: str,
    *,
    workdir: Path,
    embedder: Embedder | None = None,
) -> _AFTNoAFBase | NaiveCosineNoAFAdapter:
    if system_name == "aft_llm_dual":
        return AFTLLMDualAdapter(workdir / system_name, embedder=embedder)
    if system_name == "naive_cosine":
        return NaiveCosineNoAFAdapter(embedder=embedder)
    if system_name == "aft_neutral":
        return AFTNeutralAdapter(workdir / system_name, embedder=embedder)
    if system_name == "aft_llm_sync":
        return AFTLLMSyncAdapter(workdir / system_name, embedder=embedder)
    raise ValueError(f"Unknown system: {system_name!r}")


def _collect_top1_flags(scenario_reports: list[dict[str, Any]]) -> list[float]:
    return [
        1.0 if qr["top1_hit"] else 0.0
        for sr in scenario_reports
        for session in sr["sessions"]
        for qr in session["queries"]
    ]


def run_study(
    dataset_path: Path = DEFAULT_DATASET,
    *,
    systems: list[str] | None = None,
    embedder_name: str | None = None,
    n_bootstrap: int = _PREREG_N_BOOTSTRAP,
    seed: int = _PREREG_SEED,
) -> dict[str, Any]:
    _seed_everything(seed)
    selected = systems if systems is not None else _ALL_SYSTEMS
    dataset = _load_dataset(dataset_path)
    _validate_dataset(dataset, top_k=dataset.default_top_k)

    embedder = _build_embedder(embedder_name)
    system_scenario_reports: dict[str, list[dict[str, Any]]] = {s: [] for s in selected}

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        for system_name in tqdm(selected, desc="systems", unit="system"):
            adapter = _make_adapter(system_name, workdir=workdir, embedder=embedder)
            adapter.reset()
            for scenario in tqdm(
                dataset.scenarios, desc=system_name, unit="scenario", leave=False
            ):
                report = _run_scenario(adapter, scenario, default_top_k=dataset.default_top_k)
                system_scenario_reports[system_name].append(report)
            adapter.close()

    # per-system aggregate
    flags_by_system: dict[str, list[float]] = {
        s: _collect_top1_flags(rpts) for s, rpts in system_scenario_reports.items()
    }
    system_results: dict[str, dict[str, Any]] = {}
    for system_name, flags in flags_by_system.items():
        mean, lo, hi = bootstrap_ci(flags, n_bootstrap=n_bootstrap, seed=seed)
        system_results[system_name] = {
            "n_queries": len(flags),
            "top1_accuracy": round(mean, 4),
            "ci_95": ci_payload(mean, lo, hi, n_bootstrap=n_bootstrap),
        }

    hypotheses: dict[str, dict[str, Any]] = {}

    # Hg1 (confirmatory): aft_llm_dual > naive_cosine
    if "aft_llm_dual" in flags_by_system and "naive_cosine" in flags_by_system:
        a = flags_by_system["aft_llm_dual"]
        b = flags_by_system["naive_cosine"]
        delta, lo, hi, p = paired_bootstrap_diff(a, b, n_bootstrap=n_bootstrap, seed=seed)
        d = cohens_d_paired(a, b)
        pass_ = (p / 2.0) < _HG1_ALPHA and delta > _HG1_EFFECT_THRESHOLD
        hypotheses["Hg1"] = {
            "description": (
                f"aft_llm_dual.top1 > naive_cosine.top1 "
                f"(Δ > {_HG1_EFFECT_THRESHOLD}, one-tailed alpha={_HG1_ALPHA})"
            ),
            "result": "PASS" if pass_ else "FAIL",
            "confirmatory": True,
            "delta": round(delta, 4),
            "ci_95": ci_payload(delta, lo, hi, n_bootstrap=n_bootstrap),
            "p_one_sided": round(p / 2.0, 4),
            "p_two_sided": round(p, 4),
            "cohens_d": round(d, 4),
            "interpretation": (
                "LLM appraisal in dual-path mode provides a net positive over naive cosine "
                "on affect-free scenarios. Oracle-affect limitation (Add. D) partially resolved."
                if pass_
                else "No significant advantage of aft_llm_dual over naive cosine at Δ>0.05. "
                "LLM appraisal does not provide net positive on this dataset at this effect size."
            ),
        }

    # Hg2 (exploratory): aft_llm_dual > aft_neutral
    if "aft_llm_dual" in flags_by_system and "aft_neutral" in flags_by_system:
        a = flags_by_system["aft_llm_dual"]
        b = flags_by_system["aft_neutral"]
        delta, lo, hi, p = paired_bootstrap_diff(a, b, n_bootstrap=n_bootstrap, seed=seed)
        d = cohens_d_paired(a, b)
        hypotheses["Hg2"] = {
            "description": "aft_llm_dual.top1 > aft_neutral.top1 (exploratory)",
            "result": "PASS" if (p / 2.0 < _HG1_ALPHA and delta > 0) else "FAIL",
            "confirmatory": False,
            "delta": round(delta, 4),
            "ci_95": ci_payload(delta, lo, hi, n_bootstrap=n_bootstrap),
            "p_one_sided": round(p / 2.0, 4),
            "p_two_sided": round(p, 4),
            "cohens_d": round(d, 4),
            "interpretation": (
                "LLM-inferred affect outperforms neutral baseline — appraisal adds signal."
                if (p / 2.0 < _HG1_ALPHA and delta > 0)
                else "LLM appraisal does not reliably outperform neutral affect baseline."
            ),
        }

    # Hg3 (exploratory): aft_llm_dual > aft_llm_sync
    if "aft_llm_dual" in flags_by_system and "aft_llm_sync" in flags_by_system:
        a = flags_by_system["aft_llm_dual"]
        b = flags_by_system["aft_llm_sync"]
        delta, lo, hi, p = paired_bootstrap_diff(a, b, n_bootstrap=n_bootstrap, seed=seed)
        d = cohens_d_paired(a, b)
        hypotheses["Hg3"] = {
            "description": "aft_llm_dual.top1 > aft_llm_sync.top1 (exploratory)",
            "result": "PASS" if (p / 2.0 < _HG1_ALPHA and delta > 0) else "FAIL",
            "confirmatory": False,
            "delta": round(delta, 4),
            "ci_95": ci_payload(delta, lo, hi, n_bootstrap=n_bootstrap),
            "p_one_sided": round(p / 2.0, 4),
            "p_two_sided": round(p, 4),
            "cohens_d": round(d, 4),
            "interpretation": (
                "Dual-path (deferred) LLM appraisal outperforms synchronous LLM appraisal."
                if (p / 2.0 < _HG1_ALPHA and delta > 0)
                else "Deferred vs. synchronous LLM appraisal shows no reliable top1 difference."
            ),
        }

    return {
        "study": "addendum_g_hg1",
        "dataset": dataset.name,
        "dataset_version": dataset.version,
        "embedder": embedder_name or "hash",
        "n_scenarios": len(dataset.scenarios),
        "n_queries": len(next(iter(flags_by_system.values()))) if flags_by_system else 0,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "systems": system_results,
        "hypotheses": hypotheses,
    }


# ---------------------------------------------------------------------------
# Protocol + Markdown rendering
# ---------------------------------------------------------------------------


def _build_protocol(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "study": results["study"],
        "pre_registration": "benchmarks/preregistration_addendum_g.md",
        "dataset": results["dataset"],
        "dataset_version": results["dataset_version"],
        "dataset_note": (
            "realistic_recall_v3_noAF — derived from v2 with valence/arousal fields "
            "removed from events and state removed from queries. "
            "Affect must be inferred entirely by the appraisal engine."
        ),
        "embedder": results["embedder"],
        "systems": list(results["systems"].keys()),
        "n_bootstrap": results["n_bootstrap"],
        "seed": results["seed"],
        "hypotheses": {
            k: {
                "description": v["description"],
                "result": v["result"],
                "confirmatory": v["confirmatory"],
            }
            for k, v in results["hypotheses"].items()
        },
        "interpretation_notes": [
            "Hg1 is confirmatory; result reported regardless of outcome (pre-reg rule 1).",
            "Hg2 and Hg3 are exploratory; labeled as such (pre-reg rule 2).",
            "seed=0 is frozen; no re-seeding after results are known (pre-reg rule 3).",
            "One-tailed p = p_two_sided / 2 (bootstrap symmetric, direction fixed at pre-reg).",
            "No Holm correction: Hg1 is the sole confirmatory hypothesis (no family).",
            "If Hg1 FAIL: LLM appraisal adds no net positive over naive cosine on N=200 "
            "affect-free scenarios at Δ>0.05. See §7 Limitations (oracle-affect paragraph).",
            "If Hg1 PASS: oracle-affect limitation (Add. D) is partially resolved for "
            "the dual-path LLM configuration.",
        ],
    }


def _render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Addendum G — Dual-Path LLM Appraisal (Hg1)",
        "",
        f"Dataset: `{results['dataset']}` v{results['dataset_version']}  "
        f"({results['n_scenarios']} scenarios, {results['n_queries']} queries)  ",
        f"Embedder: `{results['embedder']}`  "
        f"n_bootstrap: {results['n_bootstrap']}  seed: {results['seed']}",
        "",
        "> **Note:** All memory events were encoded without preset valence/arousal. "
        "Affect signal comes exclusively from the LLM appraisal engine.",
        "",
        "## System Results",
        "",
        "| System | N | top1_acc | 95% CI |",
        "|---|---:|---:|---|",
    ]
    for sys_name, sys in results["systems"].items():
        ci = sys["ci_95"]
        lines.append(
            f"| `{sys_name}` | {sys['n_queries']} "
            f"| {sys['top1_accuracy']:.3f} "
            f"| [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}] |"
        )
    lines += ["", "## Hypothesis Tests", ""]
    for hyp_name, hyp in results["hypotheses"].items():
        verdict = "✓ PASS" if hyp["result"] == "PASS" else "✗ FAIL"
        tag = "(confirmatory)" if hyp.get("confirmatory") else "(exploratory)"
        ci = hyp["ci_95"]
        lines += [
            f"### {hyp_name} {tag} — {verdict}",
            "",
            f"**{hyp['description']}**",
            "",
            f"Δ = {format_point_ci(hyp['delta'], ci['ci_lower'], ci['ci_upper'])}  "
            f"p_one_sided = {hyp['p_one_sided']:.4f}  Cohen's d = {hyp['cohens_d']:.3f}",
            "",
            f"*{hyp['interpretation']}*",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Addendum G: dual-path LLM appraisal study on affect-free dataset (Hg1)."
    )
    parser.add_argument(
        "--embedder",
        default="sbert-bge",
        choices=["hash", "sbert-bge"],
        help="Embedder backend (default: sbert-bge, paper-canonical).",
    )
    parser.add_argument(
        "--systems",
        type=lambda v: [s.strip() for s in v.split(",") if s.strip()],
        default=None,
        help=(
            f"Comma-separated system list. Default: all ({', '.join(_ALL_SYSTEMS)}). "
            f"LLM systems ({', '.join(sorted(_LLM_SYSTEMS))}) "
            "require EMOTIONAL_MEMORY_LLM_API_KEY."
        ),
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=_PREREG_N_BOOTSTRAP,
        help=f"Bootstrap samples (pre-reg default: {_PREREG_N_BOOTSTRAP}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_PREREG_SEED,
        help=f"RNG seed (pre-reg default: {_PREREG_SEED}; frozen after pre-reg).",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-protocol", type=Path, default=DEFAULT_OUT_PROTOCOL)
    args = parser.parse_args()

    print(
        f"Running Addendum G study "
        f"(embedder={args.embedder}, n_bootstrap={args.n_bootstrap}, seed={args.seed}) …"
    )
    if args.systems:
        print(f"Systems: {', '.join(args.systems)}")

    results = run_study(
        args.dataset,
        systems=args.systems,
        embedder_name=args.embedder,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.out_md.write_text(_render_markdown(results), encoding="utf-8")
    args.out_protocol.write_text(json.dumps(_build_protocol(results), indent=2), encoding="utf-8")

    print(f"\nResults written to {args.out_json}")
    print("\n=== Hypothesis Summary ===")
    for hyp_name, hyp in results["hypotheses"].items():
        tag = "[C]" if hyp.get("confirmatory") else "[E]"
        print(
            f"  {tag} {hyp_name}: {hyp['result']}  "
            f"Δ={hyp['delta']:.3f}  p_one={hyp['p_one_sided']:.4f}  d={hyp['cohens_d']:.3f}"
        )


if __name__ == "__main__":
    main()
