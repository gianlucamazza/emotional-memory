"""Addendum Q — Affect-aware gating study on the mixed-gate dataset (Hq1-Hq3).

Pre-registration: benchmarks/preregistration_addendum_q_affect_gating.md
                  (incl. Amendment 1: gating as adapter-level front-router)
Dataset:          benchmarks/datasets/realistic_recall_v5_gate.json
Output:           benchmarks/appraisal_confound/results.hq.{json,md,protocol.json}
Seed:             0  (frozen per pre-reg)

Systems
-------
naive_cosine     (control):      Semantic cosine, no affective state.
aft_llm_dual     (Hq1):          AFT + shared LLMAppraisalEngine, dual-path, affect always on.
aft_gated_oracle (Hq4 expl.):    Front-router gated by ground-truth gate_label.
aft_gated_llm    (Hq2/Hq3):      Front-router gated by the LLM gate classifier.

Front-router (Amendment 1): gated adapters encode every event into the AFT
engine (identical to aft_llm_dual) AND into a local cosine index identical to
naive_cosine's. At retrieve time the gate routes: affect_free -> local cosine
index; affective -> engine.retrieve.

Usage::

    uv run python -m benchmarks.appraisal_confound.runner_hq --dry-run-oracle
    uv run python -m benchmarks.appraisal_confound.runner_hq --embedder sbert-bge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from pydantic import BaseModel, Field
from tqdm import tqdm

from benchmarks.appraisal_confound.runner_hg1 import (
    AFTLLMDualAdapter,
    NaiveCosineNoAFAdapter,
    _AFTNoAFBase,
    _build_embedder,
    _require_llm,
    _seed_everything,
)
from benchmarks.common.statistics import (
    bootstrap_ci,
    ci_payload,
    cohens_d_paired,
    format_point_ci,
    holm_bonferroni,
    paired_bootstrap_diff,
)
from benchmarks.realistic.adapters.base import ReplayRetrievedItem
from emotional_memory import Embedder, EmotionalMemory, SQLiteAffectiveStateStore
from emotional_memory.appraisal import AppraisalEngine

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
DEFAULT_DATASET = ROOT / "benchmarks" / "datasets" / "realistic_recall_v5_gate.json"
DEFAULT_OUT_JSON = _HERE / "results.hq.json"
DEFAULT_OUT_MD = _HERE / "results.hq.md"
DEFAULT_OUT_PROTOCOL = _HERE / "results.hq.protocol.json"

_PREREG_SEED = 0
_PREREG_N_BOOTSTRAP = 10_000
_HQ_ALPHA = 0.05
_HQ_EFFECT_THRESHOLD = 0.05
_APPRAISAL_CACHE_SIZE = 1024

GATE_LABELS = ("affective", "affect_free")

AFFECT_FREE_CHALLENGES = frozenset(
    {"semantic_confound", "recency_confound", "same_topic_distractor"}
)
AFFECTIVE_CHALLENGES = frozenset({"affect_congruent_tiebreak", "affective_arc_blind"})
CHALLENGE_TYPES_HQ = AFFECT_FREE_CHALLENGES | AFFECTIVE_CHALLENGES

# ---------------------------------------------------------------------------
# Dataset models — noAF schema extended with gate_label
# ---------------------------------------------------------------------------


class _GateEvent(BaseModel):
    memory_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class _GateQuery(BaseModel):
    query_id: str
    query: str
    expected_memory_ids: list[str]
    challenge_type: str
    gate_label: Literal["affective", "affect_free"]
    top_k: int | None = None


class _GateSession(BaseModel):
    session_id: str
    description: str
    events: list[_GateEvent] = Field(default_factory=list)
    queries: list[_GateQuery] = Field(default_factory=list)


class _GateScenario(BaseModel):
    scenario_id: str
    description: str
    sessions: list[_GateSession]


class _GateDataset(BaseModel):
    name: str
    version: str
    description: str
    default_top_k: int = 2
    scenarios: list[_GateScenario]


def _load_dataset(path: Path) -> _GateDataset:
    return _GateDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _validate_dataset(dataset: _GateDataset, top_k: int) -> None:
    """Structural acceptance gates (pre-reg §Dataset acceptance gates, item 1)."""
    errors: list[str] = []
    for scenario in dataset.scenarios:
        encoded_ids: list[str] = []
        labels: list[str] = []
        for session in scenario.sessions:
            encoded_ids.extend(ev.memory_id for ev in session.events)
            for q in session.queries:
                labels.append(q.gate_label)
                missing = [m for m in q.expected_memory_ids if m not in encoded_ids]
                if missing:
                    errors.append(f"{q.query_id}: expected memories not yet encoded: {missing}")
                if q.challenge_type not in CHALLENGE_TYPES_HQ:
                    errors.append(f"{q.query_id}: unknown challenge_type {q.challenge_type!r}")
                expected_label = (
                    "affective" if q.challenge_type in AFFECTIVE_CHALLENGES else "affect_free"
                )
                if q.gate_label != expected_label:
                    errors.append(
                        f"{q.query_id}: gate_label {q.gate_label!r} inconsistent with "
                        f"challenge_type {q.challenge_type!r}"
                    )
                candidate_count = len(encoded_ids)
                if candidate_count <= top_k:
                    errors.append(
                        f"{q.query_id}: trivial candidate window "
                        f"({candidate_count} <= top_k={top_k})"
                    )
        if labels.count("affective") != 2 or labels.count("affect_free") != 2:
            errors.append(
                f"{scenario.scenario_id}: gate-label balance must be 2/2, got "
                f"{labels.count('affective')}/{labels.count('affect_free')}"
            )
    if errors:
        raise ValueError("Invalid v5_gate dataset:\n- " + "\n- ".join(errors))


# ---------------------------------------------------------------------------
# Gate classifiers
# ---------------------------------------------------------------------------


class _OracleGateClassifier:
    """Ground-truth gate: exact query-text -> gate_label mapping from the dataset."""

    __slots__ = ("_mapping", "log")

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping
        self.log: list[tuple[str, str]] = []

    def classify(self, query: str) -> str:
        label = self._mapping.get(query, "affective")
        self.log.append((query, label))
        return label

    def __repr__(self) -> str:
        return f"_OracleGateClassifier(n={len(self._mapping)})"


_GATE_PROMPT = """\
Classify the retrieval query into exactly one category:
- affective: answering it requires tracking emotions, feelings, or the \
emotional trajectory of events (e.g. moments of relief, dread, turning \
points in how things felt)
- affect_free: it is a factual/semantic lookup answerable from event \
content alone

Return ONLY a JSON object {"gate_label": "<affective|affect_free>"}. No explanation.\
"""

_GATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"gate_label": {"type": "string", "enum": list(GATE_LABELS)}},
    "required": ["gate_label"],
    "additionalProperties": False,
}


class _LLMGateClassifier:
    """Binary LLM gate (frozen prompt). Cached; fallback label = 'affective'."""

    __slots__ = ("_cache", "_llm", "log")

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self._cache: dict[str, str] = {}
        self.log: list[tuple[str, str]] = []

    def classify(self, query: str) -> str:
        key = hashlib.sha256(query.encode()).hexdigest()
        if key in self._cache:
            label = self._cache[key]
            self.log.append((query, label))
            return label
        try:
            raw = self._llm(f'{_GATE_PROMPT}\n\nQuery: "{query}"', _GATE_SCHEMA)
            cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            match = re.search(r"\{[^{}]*\}", cleaned)
            if match is None:
                raise ValueError(f"No JSON in LLM gate response: {raw!r}")
            label = str(json.loads(match.group()).get("gate_label", "affective"))
            if label not in GATE_LABELS:
                label = "affective"
        except Exception:
            label = "affective"  # pre-reg fallback: preserve full-AFT behaviour
        self._cache[key] = label
        self.log.append((query, label))
        return label

    def __repr__(self) -> str:
        return f"_LLMGateClassifier(cached={len(self._cache)})"


def _gate_records(log: list[tuple[str, str]], queries: dict[str, str]) -> list[dict[str, str]]:
    """Hq6 records — match by query text (Hl3 lesson), never by index."""
    predicted_by_query = dict(log)
    return [
        {
            "query": q,
            "predicted_label": predicted_by_query.get(q, "unknown"),
            "ground_truth_label": gt,
        }
        for q, gt in queries.items()
    ]


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class AFTDualSharedAdapter(AFTLLMDualAdapter):
    """aft_llm_dual with an injected (shared) appraisal engine.

    Identical pipeline to Addendum P's AFTLLMDualAdapter; the appraisal engine
    instance is shared across the AFT arms so all three see identical inferred
    affect (pre-reg §Shared appraisal engine).
    """

    name = "aft_llm_dual"

    def __init__(
        self,
        workdir: Path,
        *,
        appraisal_engine: AppraisalEngine,
        embedder: Embedder | None = None,
    ) -> None:
        # Bypass AFTLLMDualAdapter.__init__ (which requires the LLM env):
        # the appraisal engine is injected instead.
        _AFTNoAFBase.__init__(self, workdir, embedder=embedder)
        self._appraisal = appraisal_engine

    def _make_engine(self, state_store: SQLiteAffectiveStateStore) -> EmotionalMemory:
        from emotional_memory import EmotionalMemoryConfig, InMemoryStore

        engine = EmotionalMemory(
            store=InMemoryStore(),
            embedder=self._embedder,
            state_store=state_store,
            config=EmotionalMemoryConfig(dual_path_encoding=True),
        )
        engine._appraisal_engine = self._appraisal
        return engine


class AFTGatedAdapter(AFTDualSharedAdapter):
    """Front-router (Amendment 1): gate routes each query to cosine or AFT.

    Every event is encoded into the AFT engine exactly as aft_llm_dual AND
    into a local cosine index identical to naive_cosine's. The gate decides
    per query: affect_free -> local cosine index; affective -> engine.
    """

    name = "aft_gated"

    def __init__(
        self,
        workdir: Path,
        *,
        name: str,
        gate: _OracleGateClassifier | _LLMGateClassifier,
        appraisal_engine: AppraisalEngine,
        embedder: Embedder | None = None,
    ) -> None:
        super().__init__(workdir, appraisal_engine=appraisal_engine, embedder=embedder)
        self.name = name
        self._gate = gate
        self._cosine = NaiveCosineNoAFAdapter(embedder=self._embedder)
        self._cosine_to_engine: dict[str, str] = {}

    def reset(self) -> None:
        super().reset()
        self._cosine.reset()
        self._cosine_to_engine.clear()

    def encode(self, *, memory_alias: str, content: str, metadata: dict[str, Any]) -> str:
        engine_id = super().encode(memory_alias=memory_alias, content=content, metadata=metadata)
        cosine_id = self._cosine.encode(
            memory_alias=memory_alias, content=content, metadata=metadata
        )
        self._cosine_to_engine[cosine_id] = engine_id
        return engine_id

    def retrieve(self, query: str, *, top_k: int) -> list[ReplayRetrievedItem]:
        if self._gate.classify(query) == "affect_free":
            items = self._cosine.retrieve(query, top_k=top_k)
            # Translate to engine ids so scoring sees one id namespace.
            return [
                ReplayRetrievedItem(
                    id=self._cosine_to_engine.get(item.id, item.id),
                    text=item.text,
                    score=item.score,
                )
                for item in items
            ]
        return super().retrieve(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Scenario runner — mirrors runner_hg1._run_scenario + gate_label bookkeeping
# ---------------------------------------------------------------------------


def _run_scenario(
    adapter: _AFTNoAFBase | NaiveCosineNoAFAdapter,
    scenario: _GateScenario,
    *,
    default_top_k: int,
) -> dict[str, Any]:
    alias_to_actual: dict[str, str] = {}
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
            actual_id_order.append(actual_id)

        for query in session.queries:
            effective_top_k = query.top_k or default_top_k
            retrieved = adapter.retrieve(query.query, top_k=effective_top_k)
            expected_actual_ids = {
                alias_to_actual[m] for m in query.expected_memory_ids if m in alias_to_actual
            }
            retrieved_ids = [item.id for item in retrieved]
            session_report["queries"].append(
                {
                    "query_id": query.query_id,
                    "challenge_type": query.challenge_type,
                    "gate_label": query.gate_label,
                    "top_k": effective_top_k,
                    "candidate_count": len(actual_id_order),
                    "top1_hit": bool(retrieved_ids) and retrieved_ids[0] in expected_actual_ids,
                    "hit": any(rid in expected_actual_ids for rid in retrieved_ids),
                }
            )

        session_end = adapter.end_session()
        session_report["memory_count_end"] = session_end.memory_count_end
        session_reports.append(session_report)

    return {"scenario_id": scenario.scenario_id, "sessions": session_reports}


def _query_records(scenario_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        qr for sr in scenario_reports for session in sr["sessions"] for qr in session["queries"]
    ]


def _flags(records: list[dict[str, Any]], *, gate_label: str | None = None) -> list[float]:
    return [
        1.0 if qr["top1_hit"] else 0.0
        for qr in records
        if gate_label is None or qr["gate_label"] == gate_label
    ]


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

_ALL_SYSTEMS = ["naive_cosine", "aft_llm_dual", "aft_gated_oracle", "aft_gated_llm"]


def _oracle_mapping(dataset: _GateDataset) -> dict[str, str]:
    return {
        q.query: q.gate_label for sc in dataset.scenarios for se in sc.sessions for q in se.queries
    }


def _hypothesis_payload(
    a: list[float],
    b: list[float],
    *,
    description: str,
    confirmatory: bool,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    delta, lo, hi, p_two = paired_bootstrap_diff(a, b, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "description": description,
        "confirmatory": confirmatory,
        "n": len(a),
        "delta": round(delta, 4),
        "ci_95": ci_payload(delta, lo, hi, n_bootstrap=n_bootstrap),
        "p_one_sided": round(p_two / 2.0, 4),
        "p_two_sided": round(p_two, 4),
        "cohens_d": round(cohens_d_paired(a, b), 4),
    }


def run_study(
    dataset_path: Path = DEFAULT_DATASET,
    *,
    systems: list[str] | None = None,
    embedder_name: str | None = None,
    n_bootstrap: int = _PREREG_N_BOOTSTRAP,
    seed: int = _PREREG_SEED,
    workdir: Path,
) -> dict[str, Any]:
    _seed_everything(seed)
    selected = systems if systems is not None else _ALL_SYSTEMS
    dataset = _load_dataset(dataset_path)
    _validate_dataset(dataset, top_k=dataset.default_top_k)

    embedder = _build_embedder(embedder_name)
    needs_llm = any(s != "naive_cosine" for s in selected)
    shared_appraisal: AppraisalEngine | None = None
    llm = None
    if needs_llm:
        from emotional_memory import LLMAppraisalConfig, LLMAppraisalEngine

        llm = _require_llm()
        shared_appraisal = LLMAppraisalEngine(
            llm, config=LLMAppraisalConfig(cache_size=_APPRAISAL_CACHE_SIZE)
        )

    gate_logs: dict[str, list[tuple[str, str]]] = {}

    def _make_adapter(name: str) -> _AFTNoAFBase | NaiveCosineNoAFAdapter:
        if name == "naive_cosine":
            return NaiveCosineNoAFAdapter(embedder=embedder)
        assert shared_appraisal is not None
        if name == "aft_llm_dual":
            return AFTDualSharedAdapter(
                workdir / name, appraisal_engine=shared_appraisal, embedder=embedder
            )
        if name == "aft_gated_oracle":
            gate: _OracleGateClassifier | _LLMGateClassifier = _OracleGateClassifier(
                _oracle_mapping(dataset)
            )
            gate_logs[name] = gate.log
            return AFTGatedAdapter(
                workdir / name,
                name=name,
                gate=gate,
                appraisal_engine=shared_appraisal,
                embedder=embedder,
            )
        if name == "aft_gated_llm":
            assert llm is not None
            llm_gate = _LLMGateClassifier(llm)
            gate_logs[name] = llm_gate.log
            return AFTGatedAdapter(
                workdir / name,
                name=name,
                gate=llm_gate,
                appraisal_engine=shared_appraisal,
                embedder=embedder,
            )
        raise ValueError(f"Unknown system: {name!r}")

    records_by_system: dict[str, list[dict[str, Any]]] = {}
    for system_name in tqdm(selected, desc="systems", unit="system"):
        adapter = _make_adapter(system_name)
        adapter.reset()
        reports = [
            _run_scenario(adapter, scenario, default_top_k=dataset.default_top_k)
            for scenario in tqdm(dataset.scenarios, desc=system_name, unit="scenario", leave=False)
        ]
        adapter.close()
        records_by_system[system_name] = _query_records(reports)

    # Per-system aggregates (full + per gate half + per challenge type)
    system_results: dict[str, dict[str, Any]] = {}
    for system_name, records in records_by_system.items():
        entry: dict[str, Any] = {}
        for scope, flags in (
            ("full", _flags(records)),
            ("affective", _flags(records, gate_label="affective")),
            ("affect_free", _flags(records, gate_label="affect_free")),
        ):
            mean, lo, hi = bootstrap_ci(flags, n_bootstrap=n_bootstrap, seed=seed)
            entry[scope] = {
                "n_queries": len(flags),
                "top1_accuracy": round(mean, 4),
                "ci_95": ci_payload(mean, lo, hi, n_bootstrap=n_bootstrap),
            }
        by_challenge: dict[str, dict[str, Any]] = {}
        for ctype in sorted(CHALLENGE_TYPES_HQ):
            cflags = [
                1.0 if r["top1_hit"] else 0.0 for r in records if r["challenge_type"] == ctype
            ]
            if cflags:
                by_challenge[ctype] = {
                    "n": len(cflags),
                    "top1_accuracy": round(sum(cflags) / len(cflags), 4),
                }
        entry["by_challenge_type"] = by_challenge
        system_results[system_name] = entry

    # Confirmatory family Hq1-Hq3 (Holm m=3) + exploratory Hq4-Hq5
    hypotheses: dict[str, dict[str, Any]] = {}

    def _have(*names: str) -> bool:
        return all(n in records_by_system for n in names)

    if _have("aft_llm_dual", "naive_cosine"):
        hypotheses["Hq1"] = _hypothesis_payload(
            _flags(records_by_system["aft_llm_dual"], gate_label="affective"),
            _flags(records_by_system["naive_cosine"], gate_label="affective"),
            description=(
                "aft_llm_dual.top1 > naive_cosine.top1 on the affective subset "
                f"(Δ > {_HQ_EFFECT_THRESHOLD}, one-tailed, Holm m=3)"
            ),
            confirmatory=True,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    if _have("aft_gated_llm", "aft_llm_dual"):
        hypotheses["Hq2"] = _hypothesis_payload(
            _flags(records_by_system["aft_gated_llm"]),
            _flags(records_by_system["aft_llm_dual"]),
            description=(
                "aft_gated_llm.top1 > aft_llm_dual.top1 on the full corpus "
                f"(Δ > {_HQ_EFFECT_THRESHOLD}, one-tailed, Holm m=3)"
            ),
            confirmatory=True,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    if _have("aft_gated_llm", "naive_cosine"):
        hypotheses["Hq3"] = _hypothesis_payload(
            _flags(records_by_system["aft_gated_llm"]),
            _flags(records_by_system["naive_cosine"]),
            description=(
                "aft_gated_llm.top1 > naive_cosine.top1 on the full corpus "
                f"(Δ > {_HQ_EFFECT_THRESHOLD}, one-tailed, Holm m=3)"
            ),
            confirmatory=True,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

    confirmatory_ids = [h for h in ("Hq1", "Hq2", "Hq3") if h in hypotheses]
    if confirmatory_ids:
        adjusted = holm_bonferroni([hypotheses[h]["p_one_sided"] for h in confirmatory_ids])
        for hyp_id, p_adj in zip(confirmatory_ids, adjusted, strict=True):
            hyp = hypotheses[hyp_id]
            hyp["p_holm"] = round(p_adj, 4)
            hyp["result"] = (
                "PASS" if (p_adj < _HQ_ALPHA and hyp["delta"] > _HQ_EFFECT_THRESHOLD) else "FAIL"
            )

    if _have("aft_gated_oracle", "naive_cosine"):
        hyp = _hypothesis_payload(
            _flags(records_by_system["aft_gated_oracle"]),
            _flags(records_by_system["naive_cosine"]),
            description="aft_gated_oracle vs naive_cosine, full corpus (exploratory upper bound)",
            confirmatory=False,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        hyp["result"] = "PASS" if (hyp["p_one_sided"] < _HQ_ALPHA and hyp["delta"] > 0) else "FAIL"
        hypotheses["Hq4"] = hyp
    if _have("aft_gated_llm", "naive_cosine"):
        hyp = _hypothesis_payload(
            _flags(records_by_system["aft_gated_llm"], gate_label="affect_free"),
            _flags(records_by_system["naive_cosine"], gate_label="affect_free"),
            description=(
                "aft_gated_llm vs naive_cosine on the affect-free subset "
                "(exploratory: gate false-positive cost; expected ≈ 0)"
            ),
            confirmatory=False,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        hyp["result"] = "REPORTED"
        hypotheses["Hq5"] = hyp

    # Hq6 — gate classifier accuracy (mandatory report; Hl3 lesson)
    gate_reports: dict[str, dict[str, Any]] = {}
    query_truth = _oracle_mapping(dataset)
    for system_name, log in gate_logs.items():
        recs = _gate_records(log, query_truth)
        n = len(recs)
        correct = sum(1 for r in recs if r["predicted_label"] == r["ground_truth_label"])
        confusion: dict[str, dict[str, int]] = {}
        for r in recs:
            confusion.setdefault(r["ground_truth_label"], {}).setdefault(r["predicted_label"], 0)
            confusion[r["ground_truth_label"]][r["predicted_label"]] += 1
        gate_reports[system_name] = {
            "n": n,
            "accuracy": round(correct / n, 4) if n else None,
            "confusion": confusion,
            "records": recs,
        }

    return {
        "study": "addendum_q_hq",
        "dataset": dataset.name,
        "dataset_version": dataset.version,
        "embedder": embedder_name or "hash",
        "n_scenarios": len(dataset.scenarios),
        "n_queries": len(next(iter(records_by_system.values()))) if records_by_system else 0,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "systems": system_results,
        "hypotheses": hypotheses,
        "gate_classifier": gate_reports,
        "per_query": {
            name: [
                {k: qr[k] for k in ("query_id", "challenge_type", "gate_label", "top1_hit", "hit")}
                for qr in records
            ]
            for name, records in records_by_system.items()
        },
    }


# ---------------------------------------------------------------------------
# Dry-run oracle equivalence check (pre-reg prerequisite, no LLM)
# ---------------------------------------------------------------------------


def dry_run_oracle(dataset_path: Path, *, workdir: Path) -> None:
    """Assert aft_gated_oracle.top1 == naive_cosine.top1 on every affect-free query.

    Runs with the hash embedder and the keyword appraisal engine — no LLM
    calls. The affective half is NOT asserted (affect routing differs by
    design); only the gate-off path equivalence is checked.
    """
    from emotional_memory.appraisal_llm import KeywordAppraisalEngine

    dataset = _load_dataset(dataset_path)
    _validate_dataset(dataset, top_k=dataset.default_top_k)

    cosine = NaiveCosineNoAFAdapter()
    gated = AFTGatedAdapter(
        workdir / "dry_oracle",
        name="aft_gated_oracle",
        gate=_OracleGateClassifier(_oracle_mapping(dataset)),
        appraisal_engine=KeywordAppraisalEngine(),
    )
    results: dict[str, list[dict[str, Any]]] = {}
    for name, adapter in (("naive_cosine", cosine), ("aft_gated_oracle", gated)):
        adapter.reset()
        reports = [
            _run_scenario(adapter, sc, default_top_k=dataset.default_top_k)
            for sc in dataset.scenarios
        ]
        adapter.close()
        results[name] = _query_records(reports)

    mismatches = [
        (a["query_id"], a["top1_hit"], b["top1_hit"])
        for a, b in zip(results["naive_cosine"], results["aft_gated_oracle"], strict=True)
        if a["gate_label"] == "affect_free" and a["top1_hit"] != b["top1_hit"]
    ]
    n_af = sum(1 for r in results["naive_cosine"] if r["gate_label"] == "affect_free")
    if mismatches:
        raise AssertionError(
            f"Gate-off equivalence violated on {len(mismatches)}/{n_af} affect-free queries: "
            f"{mismatches[:10]}"
        )
    print(f"[dry-run] gate-off equivalence holds on all {n_af} affect-free queries.")


# ---------------------------------------------------------------------------
# Protocol + Markdown rendering
# ---------------------------------------------------------------------------


def _build_protocol(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "study": results["study"],
        "pre_registration": "benchmarks/preregistration_addendum_q_affect_gating.md",
        "amendments": ["Amendment 1 (pre-execution): gating as adapter-level front-router"],
        "dataset": results["dataset"],
        "dataset_version": results["dataset_version"],
        "embedder": results["embedder"],
        "systems": list(results["systems"].keys()),
        "n_bootstrap": results["n_bootstrap"],
        "seed": results["seed"],
        "hypotheses": {
            k: {
                "description": v["description"],
                "result": v.get("result", "REPORTED"),
                "confirmatory": v["confirmatory"],
            }
            for k, v in results["hypotheses"].items()
        },
        "interpretation_notes": [
            "Hq1-Hq3 are the confirmatory family; Holm-Bonferroni m=3 on one-sided p-values.",
            "PASS rule per hypothesis: delta > 0.05 AND Holm-adjusted one-sided p < 0.05.",
            "Hq4 (oracle upper bound) and Hq5 (gate false-positive cost) are exploratory.",
            "Hq6 (gate classifier accuracy + confusion) is a mandatory report (Hl3 lesson).",
            "seed=0 frozen; no re-seeding after results are known.",
            "Shared appraisal engine across AFT arms: identical inferred affect by design.",
            "Branches A/B/C interpretations are pre-specified in the pre-registration.",
        ],
    }


def _render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Addendum Q — Affect-Aware Gating (Hq1-Hq3)",
        "",
        f"Dataset: `{results['dataset']}` v{results['dataset_version']}  "
        f"({results['n_scenarios']} scenarios, {results['n_queries']} queries)  ",
        f"Embedder: `{results['embedder']}`  "
        f"n_bootstrap: {results['n_bootstrap']}  seed: {results['seed']}",
        "",
        "## System Results (top1 accuracy)",
        "",
        "| System | full | affective half | affect-free half |",
        "|---|---:|---:|---:|",
    ]
    for sys_name, sys in results["systems"].items():
        cells = []
        for scope in ("full", "affective", "affect_free"):
            s = sys[scope]
            ci = s["ci_95"]
            cells.append(f"{s['top1_accuracy']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
        lines.append(f"| `{sys_name}` | " + " | ".join(cells) + " |")
    lines += ["", "## Hypothesis Tests", ""]
    for hyp_name, hyp in results["hypotheses"].items():
        verdict = hyp.get("result", "REPORTED")
        tag = "(confirmatory)" if hyp.get("confirmatory") else "(exploratory)"
        ci = hyp["ci_95"]
        p_line = f"p_one_sided = {hyp['p_one_sided']:.4f}"
        if "p_holm" in hyp:
            p_line += f"  p_holm = {hyp['p_holm']:.4f}"
        lines += [
            f"### {hyp_name} {tag} — {verdict}",
            "",
            f"**{hyp['description']}**  (N = {hyp['n']})",
            "",
            f"Δ = {format_point_ci(hyp['delta'], ci['ci_lower'], ci['ci_upper'])}  "
            f"{p_line}  Cohen's d = {hyp['cohens_d']:.3f}",
            "",
        ]
    if results.get("gate_classifier"):
        lines += ["## Gate Classifier (Hq6)", ""]
        for sys_name, rep in results["gate_classifier"].items():
            lines += [
                f"### `{sys_name}` — accuracy {rep['accuracy']}",
                "",
                f"Confusion (gt -> predicted): `{json.dumps(rep['confusion'], sort_keys=True)}`",
                "",
            ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Addendum Q: affect-aware gating study (Hq1-Hq3)."
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
        help=f"Comma-separated system list. Default: all ({', '.join(_ALL_SYSTEMS)}).",
    )
    parser.add_argument("--n-bootstrap", type=int, default=_PREREG_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=_PREREG_SEED)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-protocol", type=Path, default=DEFAULT_OUT_PROTOCOL)
    parser.add_argument(
        "--dry-run-oracle",
        action="store_true",
        help="No LLM: assert gate-off equivalence with naive_cosine (hash embedder).",
    )
    args = parser.parse_args()

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.dry_run_oracle:
            dry_run_oracle(args.dataset, workdir=Path(tmpdir))
            return

        print(
            f"Running Addendum Q study "
            f"(embedder={args.embedder}, n_bootstrap={args.n_bootstrap}, seed={args.seed}) …"
        )
        results = run_study(
            args.dataset,
            systems=args.systems,
            embedder_name=args.embedder,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            workdir=Path(tmpdir),
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.out_md.write_text(_render_markdown(results), encoding="utf-8")
    args.out_protocol.write_text(json.dumps(_build_protocol(results), indent=2), encoding="utf-8")

    print(f"\nResults written to {args.out_json}")
    print("\n=== Hypothesis Summary ===")
    for hyp_name, hyp in results["hypotheses"].items():
        tag = "[C]" if hyp.get("confirmatory") else "[E]"
        extra = f"  p_holm={hyp['p_holm']:.4f}" if "p_holm" in hyp else ""
        print(
            f"  {tag} {hyp_name}: {hyp.get('result', 'REPORTED')}  "
            f"Δ={hyp['delta']:.3f}  p_one={hyp['p_one_sided']:.4f}{extra}  "
            f"d={hyp['cohens_d']:.3f}"
        )
    for sys_name, rep in results.get("gate_classifier", {}).items():
        print(f"  [G] {sys_name}: gate accuracy={rep['accuracy']}")


if __name__ == "__main__":
    main()
