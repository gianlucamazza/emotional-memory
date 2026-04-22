"""Comparative replay benchmark for realistic affect-aware retrieval."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from benchmarks.realistic.adapters import (
    AFTReplayAdapter,
    NaiveCosineReplayAdapter,
    RecencyReplayAdapter,
    ReplayAdapter,
    ReplayRetrievedItem,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "benchmarks" / "datasets" / "realistic_recall_v1.json"
DEFAULT_OUT_JSON = ROOT / "benchmarks" / "realistic" / "results.json"
DEFAULT_OUT_MD = ROOT / "benchmarks" / "realistic" / "results.md"
DEFAULT_PROTOCOL = ROOT / "benchmarks" / "realistic" / "results.protocol.json"
DEFAULT_SYSTEMS = ["aft", "naive_cosine", "recency"]
CHALLENGE_TYPES = {
    "recency_confound",
    "semantic_confound",
    "affective_arc",
    "same_topic_distractor",
}


class QueryState(BaseModel):
    valence: float
    arousal: float


class ReplayEvent(BaseModel):
    memory_id: str
    content: str
    valence: float
    arousal: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplayQuery(BaseModel):
    query_id: str
    query: str
    expected_memory_ids: list[str]
    challenge_type: str
    top_k: int | None = None
    state: QueryState | None = None


class ReplaySession(BaseModel):
    session_id: str
    description: str
    events: list[ReplayEvent] = Field(default_factory=list)
    queries: list[ReplayQuery] = Field(default_factory=list)


class ReplayScenario(BaseModel):
    scenario_id: str
    description: str
    sessions: list[ReplaySession]


class ReplayDataset(BaseModel):
    name: str
    version: str
    description: str
    default_top_k: int = 3
    scenarios: list[ReplayScenario]


def load_dataset(path: Path = DATASET) -> ReplayDataset:
    return ReplayDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


def analyze_dataset_difficulty(
    dataset: ReplayDataset,
    *,
    top_k: int,
) -> dict[str, Any]:
    query_reports: list[dict[str, Any]] = []
    scenario_reports: list[dict[str, Any]] = []

    for scenario in dataset.scenarios:
        scenario_queries: list[dict[str, Any]] = []
        encoded_aliases: list[str] = []
        for session in scenario.sessions:
            encoded_aliases.extend(event.memory_id for event in session.events)
            recency_order = list(reversed(encoded_aliases))
            for query in session.queries:
                effective_top_k = query.top_k or top_k
                expected_recency_ranks = [
                    recency_order.index(memory_alias) + 1
                    for memory_alias in query.expected_memory_ids
                    if memory_alias in recency_order
                ]
                nontrivial = bool(expected_recency_ranks) and all(
                    rank > effective_top_k for rank in expected_recency_ranks
                )
                report = {
                    "scenario_id": scenario.scenario_id,
                    "session_id": session.session_id,
                    "query_id": query.query_id,
                    "challenge_type": query.challenge_type,
                    "candidate_count": len(encoded_aliases),
                    "top_k": effective_top_k,
                    "expected_memory_ids": query.expected_memory_ids,
                    "expected_recency_ranks": expected_recency_ranks,
                    "trivial_candidate_window": len(encoded_aliases) <= effective_top_k,
                    "recency_would_hit_at_k": any(
                        rank <= effective_top_k for rank in expected_recency_ranks
                    ),
                    "nontrivial_for_recency": nontrivial,
                }
                scenario_queries.append(report)
                query_reports.append(report)
        scenario_reports.append(
            {
                "scenario_id": scenario.scenario_id,
                "query_count": len(scenario_queries),
                "nontrivial_query_count": sum(
                    1 for query in scenario_queries if query["nontrivial_for_recency"]
                ),
            }
        )

    total_queries = len(query_reports)
    nontrivial_query_count = sum(1 for query in query_reports if query["nontrivial_for_recency"])
    return {
        "top_k": top_k,
        "total_queries": total_queries,
        "minimum_candidate_count": min(
            (query["candidate_count"] for query in query_reports),
            default=0,
        ),
        "nontrivial_query_count": nontrivial_query_count,
        "nontrivial_query_rate": round(
            nontrivial_query_count / max(total_queries, 1),
            4,
        ),
        "trivial_candidate_window_count": sum(
            1 for query in query_reports if query["trivial_candidate_window"]
        ),
        "challenge_type_counts": {
            challenge_type: sum(
                1 for query in query_reports if query["challenge_type"] == challenge_type
            )
            for challenge_type in sorted(CHALLENGE_TYPES)
        },
        "scenario_reports": scenario_reports,
        "queries": query_reports,
    }


def validate_dataset_difficulty(
    dataset: ReplayDataset,
    *,
    top_k: int,
) -> dict[str, Any]:
    profile = analyze_dataset_difficulty(dataset, top_k=top_k)
    errors: list[str] = []

    missing_expected = [
        query["query_id"] for query in profile["queries"] if not query["expected_recency_ranks"]
    ]
    if missing_expected:
        errors.append(
            "Queries reference expected memories that are not encoded before retrieval: "
            + ", ".join(missing_expected)
        )

    invalid_challenge_types = [
        query["query_id"]
        for query in profile["queries"]
        if query["challenge_type"] not in CHALLENGE_TYPES
    ]
    if invalid_challenge_types:
        errors.append(
            "Queries use unsupported challenge_type values: " + ", ".join(invalid_challenge_types)
        )

    trivial_candidate_queries = [
        query["query_id"] for query in profile["queries"] if query["trivial_candidate_window"]
    ]
    if trivial_candidate_queries:
        errors.append(
            "Queries have candidate_count <= top_k and would make hit@k trivial: "
            + ", ".join(trivial_candidate_queries)
        )

    weak_scenarios = [
        report["scenario_id"]
        for report in profile["scenario_reports"]
        if report["nontrivial_query_count"] == 0
    ]
    if weak_scenarios:
        errors.append(
            "Each scenario must contain at least one query that recency alone would miss: "
            + ", ".join(weak_scenarios)
        )

    if profile["nontrivial_query_rate"] < 0.5:
        errors.append(
            "At least half of the realistic replay queries must be non-trivial for recency."
        )

    if profile["challenge_type_counts"].get("semantic_confound", 0) == 0:
        errors.append("The dataset must declare at least one semantic_confound query.")

    if errors:
        raise ValueError("Invalid realistic benchmark dataset:\n- " + "\n- ".join(errors))

    return profile


def build_protocol_metadata(
    dataset: ReplayDataset,
    *,
    system_names: list[str],
    difficulty_profile: dict[str, Any],
    dataset_path: Path = DATASET,
) -> dict[str, Any]:
    scenario_count = len(dataset.scenarios)
    query_count = sum(
        len(session.queries) for scenario in dataset.scenarios for session in scenario.sessions
    )
    session_count = sum(len(scenario.sessions) for scenario in dataset.scenarios)
    dataset_ref = (
        str(dataset_path.relative_to(ROOT))
        if dataset_path.is_relative_to(ROOT)
        else str(dataset_path)
    )
    return {
        "benchmark": "realistic_recall_v1",
        "question": (
            "Do persisted affective state and AFT-style retrieval improve recall "
            "on scripted multi-session scenarios compared with simpler baselines?"
        ),
        "dataset": {
            "name": dataset.name,
            "version": dataset.version,
            "path": dataset_ref,
            "scenario_count": scenario_count,
            "session_count": session_count,
            "query_count": query_count,
            "type": "scripted multi-session replay benchmark",
        },
        "difficulty_profile": {
            "top_k": difficulty_profile["top_k"],
            "minimum_candidate_count": difficulty_profile["minimum_candidate_count"],
            "nontrivial_query_count": difficulty_profile["nontrivial_query_count"],
            "nontrivial_query_rate": difficulty_profile["nontrivial_query_rate"],
            "trivial_candidate_window_count": difficulty_profile["trivial_candidate_window_count"],
            "challenge_type_counts": difficulty_profile["challenge_type_counts"],
        },
        "systems": system_names,
        "primary_metrics": ["top1_accuracy", "hit@k"],
        "secondary_metrics": [
            "stateful_session_rate",
            "memory_count_growth",
            "candidate_count",
            "recency_triviality",
            "challenge_type",
        ],
        "system_capabilities": {
            "aft": {
                "persisted_state": True,
                "explanations": True,
                "notes": "Uses EmotionalMemory + SQLiteAffectiveStateStore across sessions.",
            },
            "naive_cosine": {
                "persisted_state": False,
                "explanations": False,
                "notes": "Semantic-only cosine retrieval with the same deterministic embedder.",
            },
            "recency": {
                "persisted_state": False,
                "explanations": False,
                "notes": "Recency-only control with no semantic or affective ranking.",
            },
        },
        "interpretation_guardrails": [
            "This is a controlled replay benchmark, not a production agent evaluation.",
            "top1_accuracy is the headline metric; hit@k is secondary support.",
            "Queries with candidate_count <= top_k are rejected as trivial by dataset validation.",
            (
                "Challenge-type aggregates are reported so localized AFT gains "
                "or failures are visible."
            ),
            "Only AFT models persisted affective state in this protocol.",
            "Baselines remain useful controls for memory carry-over without affective state.",
            "Results do not establish general downstream superiority or human-like memory.",
        ],
    }


def _aggregate_query_metrics(query_reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(query_reports),
        "top1_accuracy": round(
            sum(1 for query in query_reports if query["top1_hit"]) / max(len(query_reports), 1),
            4,
        ),
        "hit_at_k": round(
            sum(1 for query in query_reports if query["hit"]) / max(len(query_reports), 1),
            4,
        ),
        "minimum_candidate_count": min(
            (query["candidate_count"] for query in query_reports),
            default=0,
        ),
        "nontrivial_query_rate": round(
            sum(1 for query in query_reports if not query["recency_would_hit_at_k"])
            / max(len(query_reports), 1),
            4,
        ),
    }


def _aggregate_by_challenge_type(query_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for challenge_type in sorted(CHALLENGE_TYPES):
        typed_queries = [
            query for query in query_reports if query["challenge_type"] == challenge_type
        ]
        if not typed_queries:
            continue
        aggregates.append(
            {
                "challenge_type": challenge_type,
                **_aggregate_query_metrics(typed_queries),
            }
        )
    return aggregates


def _make_adapter(system_name: str, *, workdir: Path) -> ReplayAdapter:
    if system_name == "aft":
        return AFTReplayAdapter(workdir / system_name)
    if system_name == "naive_cosine":
        return NaiveCosineReplayAdapter()
    if system_name == "recency":
        return RecencyReplayAdapter()
    raise ValueError(f"Unknown system: {system_name}")


def _serialize_result(
    item: ReplayRetrievedItem,
    *,
    alias_by_actual: dict[str, str],
) -> dict[str, Any]:
    payload = {
        "memory_id": item.id,
        "memory_alias": alias_by_actual.get(item.id),
        "content": item.text,
        "score": item.score,
    }
    explanation = item.metadata.get("explanation")
    if isinstance(explanation, dict):
        payload["explanation"] = explanation
    return payload


def run_system_on_scenario(
    adapter: ReplayAdapter,
    scenario: ReplayScenario,
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
            "description": session.description,
            "state_loaded_from_store": session_start.state_loaded,
            "memory_count_start": session_start.memory_count_start,
            "mood_start": session_start.mood_start,
            "queries": [],
        }

        for event in session.events:
            actual_id = adapter.encode(
                memory_alias=event.memory_id,
                content=event.content,
                valence=event.valence,
                arousal=event.arousal,
                metadata=event.metadata,
            )
            alias_to_actual[event.memory_id] = actual_id
            actual_to_alias[actual_id] = event.memory_id
            actual_id_order.append(actual_id)

        for query in session.queries:
            effective_top_k = query.top_k or default_top_k
            retrieved = adapter.retrieve(
                query.query,
                top_k=effective_top_k,
                valence=None if query.state is None else query.state.valence,
                arousal=None if query.state is None else query.state.arousal,
            )
            expected_actual_ids = {
                alias_to_actual[memory_alias]
                for memory_alias in query.expected_memory_ids
                if memory_alias in alias_to_actual
            }
            recency_order = list(reversed(actual_id_order))
            expected_recency_ranks = {
                actual_to_alias[memory_id]: recency_order.index(memory_id) + 1
                for memory_id in expected_actual_ids
                if memory_id in recency_order
            }
            retrieved_ids = [item.id for item in retrieved]
            hit = any(memory_id in expected_actual_ids for memory_id in retrieved_ids)
            top1_hit = bool(retrieved_ids) and retrieved_ids[0] in expected_actual_ids
            session_report["queries"].append(
                {
                    "query_id": query.query_id,
                    "query": query.query,
                    "challenge_type": query.challenge_type,
                    "top_k": effective_top_k,
                    "candidate_count": len(actual_id_order),
                    "expected_memory_ids": query.expected_memory_ids,
                    "expected_recency_ranks": expected_recency_ranks,
                    "trivial_candidate_window": len(actual_id_order) <= effective_top_k,
                    "recency_would_hit_at_k": any(
                        rank <= effective_top_k for rank in expected_recency_ranks.values()
                    ),
                    "retrieved_memory_ids": retrieved_ids,
                    "retrieved_memory_aliases": [
                        actual_to_alias.get(memory_id) for memory_id in retrieved_ids
                    ],
                    "hit": hit,
                    "top1_hit": top1_hit,
                    "results": [
                        _serialize_result(item, alias_by_actual=actual_to_alias)
                        for item in retrieved
                    ],
                }
            )

        session_end = adapter.end_session()
        session_report["memory_count_end"] = session_end.memory_count_end
        session_report["mood_end"] = session_end.mood_end
        session_reports.append(session_report)

    query_reports = [query for session in session_reports for query in session["queries"]]
    memory_counts = [session["memory_count_end"] for session in session_reports]
    return {
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "sessions": session_reports,
        "metrics": {
            **_aggregate_query_metrics(query_reports),
            "stateful_session_rate": round(
                sum(1 for session in session_reports if session["state_loaded_from_store"])
                / max(len(session_reports), 1),
                4,
            ),
            "memory_count_growth": memory_counts[-1] - memory_counts[0] if memory_counts else 0,
        },
        "challenge_type_metrics": _aggregate_by_challenge_type(query_reports),
    }


def run_benchmark(
    dataset: ReplayDataset,
    *,
    systems: list[str] | None = None,
    top_k: int | None = None,
    dataset_path: Path = DATASET,
) -> dict[str, Any]:
    selected_systems = DEFAULT_SYSTEMS if systems is None else systems
    effective_top_k = dataset.default_top_k if top_k is None else top_k
    difficulty_profile = validate_dataset_difficulty(dataset, top_k=effective_top_k)
    system_results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="emotional-memory-realistic-") as tmp_dir:
        base_workdir = Path(tmp_dir)
        for system_name in selected_systems:
            adapter = _make_adapter(system_name, workdir=base_workdir)
            adapter.reset()
            try:
                scenario_results = [
                    run_system_on_scenario(adapter, scenario, default_top_k=effective_top_k)
                    for scenario in dataset.scenarios
                ]
            finally:
                adapter.close()
            all_queries = [
                query
                for scenario in scenario_results
                for session in scenario["sessions"]
                for query in session["queries"]
            ]
            all_sessions = [
                session for scenario in scenario_results for session in scenario["sessions"]
            ]
            system_results.append(
                {
                    "system": system_name,
                    "supports_explanations": adapter.supports_explanations,
                    "supports_persisted_state": adapter.supports_persisted_state,
                    "scenarios": scenario_results,
                    "aggregate_metrics": {
                        **_aggregate_query_metrics(all_queries),
                        "stateful_session_rate": round(
                            sum(
                                1 for session in all_sessions if session["state_loaded_from_store"]
                            )
                            / max(len(all_sessions), 1),
                            4,
                        ),
                    },
                    "challenge_type_metrics": _aggregate_by_challenge_type(all_queries),
                }
            )

    return {
        "benchmark": dataset.name,
        "version": dataset.version,
        "top_k": effective_top_k,
        "difficulty_profile": difficulty_profile,
        "protocol": build_protocol_metadata(
            dataset,
            system_names=selected_systems,
            difficulty_profile=difficulty_profile,
            dataset_path=dataset_path,
        ),
        "systems": system_results,
    }


def _render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Realistic Replay Benchmark",
        "",
        "This report summarizes the comparative multi-session benchmark that uses",
        "persisted affective state and memory carry-over across scripted sessions.",
        "",
        "Headline metric: `top1_accuracy`. `hit@k` remains secondary support.",
        "",
        (
            "| System | Queries | top1 | hit@k | Min candidates | "
            "Non-trivial queries | Stateful sessions |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system in results["systems"]:
        metrics = system["aggregate_metrics"]
        lines.append(
            f"| `{system['system']}` | {metrics['query_count']} | "
            f"{metrics['top1_accuracy']:.2f} | {metrics['hit_at_k']:.2f} | "
            f"{metrics['minimum_candidate_count']} | {metrics['nontrivial_query_rate']:.2f} | "
            f"{metrics['stateful_session_rate']:.2f} |"
        )
    lines.extend(["", "## Per Scenario", ""])
    for system in results["systems"]:
        lines.append(f"### `{system['system']}`")
        lines.append("")
        lines.append(
            "| Scenario | Queries | top1 | hit@k | Min candidates | "
            "Non-trivial queries | Stateful sessions |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for scenario in system["scenarios"]:
            metrics = scenario["metrics"]
            lines.append(
                f"| `{scenario['scenario_id']}` | {metrics['query_count']} | "
                f"{metrics['top1_accuracy']:.2f} | {metrics['hit_at_k']:.2f} | "
                f"{metrics['minimum_candidate_count']} | {metrics['nontrivial_query_rate']:.2f} | "
                f"{metrics['stateful_session_rate']:.2f} |"
            )
        lines.append("")
    lines.extend(["## By Challenge Type", ""])
    for system in results["systems"]:
        lines.append(f"### `{system['system']}`")
        lines.append("")
        lines.append(
            "| Challenge Type | Queries | top1 | hit@k | Min candidates | Non-trivial queries |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        lines.extend(
            [
                f"| `{metrics['challenge_type']}` | {metrics['query_count']} | "
                f"{metrics['top1_accuracy']:.2f} | {metrics['hit_at_k']:.2f} | "
                f"{metrics['minimum_candidate_count']} | {metrics['nontrivial_query_rate']:.2f} |"
                for metrics in system["challenge_type_metrics"]
            ]
        )
        lines.append("")
    return "\n".join(lines)


def write_results(
    results: dict[str, Any],
    *,
    out_json: Path,
    out_md: Path,
    out_protocol: Path,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_protocol.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(results), encoding="utf-8")
    out_protocol.write_text(json.dumps(results["protocol"], indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the realistic replay benchmark.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--systems",
        type=lambda value: [item.strip() for item in value.split(",") if item.strip()],
        default=DEFAULT_SYSTEMS,
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    results = run_benchmark(
        dataset,
        systems=args.systems,
        top_k=args.top_k,
        dataset_path=args.dataset,
    )
    write_results(
        results,
        out_json=args.out_json,
        out_md=args.out_md,
        out_protocol=args.out_protocol,
    )

    aggregates = ", ".join(
        f"{system['system']}: top1={system['aggregate_metrics']['top1_accuracy']:.2f}, "
        f"hit@k={system['aggregate_metrics']['hit_at_k']:.2f}"
        for system in results["systems"]
    )
    print("realistic benchmark complete:", aggregates)


if __name__ == "__main__":
    main()
