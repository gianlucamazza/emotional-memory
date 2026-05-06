"""Integration tests for the ablation study runner."""

from __future__ import annotations

import pytest

from benchmarks.ablation.runner import (
    _aggregate_link_stats,
    _collect_link_stats,
    run_ablation_study,
)
from benchmarks.realistic.runner import CHALLENGE_TYPES, DATASET, load_dataset


def test_ablation_study_structure() -> None:
    dataset = load_dataset()
    results = run_ablation_study(dataset, n_bootstrap=100, seed=0)

    # Benchmark id must be derived from dataset.name — not a hardcoded literal.
    assert results["benchmark"] == f"ablation_{dataset.name}"
    assert results["benchmark"] == "ablation_realistic_recall_v1"
    assert len(results["variants"]) == 8

    variant_names = [v["variant"] for v in results["variants"]]
    assert variant_names == [
        "full",
        "no_appraisal",
        "no_mood",
        "no_momentum",
        "no_resonance",
        "no_reconsolidation",
        "dual_path",
        "aft_keyword_synchronous",
    ]

    for v in results["variants"]:
        assert "aggregate_metrics" in v
        assert "challenge_type_metrics" in v
        assert isinstance(v["aggregate_metrics"]["top1_accuracy"], float)

    assert len(results["pairwise_vs_full"]) == 7
    pairwise_names = [r["variant"] for r in results["pairwise_vs_full"]]
    assert "no_appraisal" in pairwise_names
    assert "no_resonance" in pairwise_names
    assert "no_reconsolidation" in pairwise_names
    assert "dual_path" in pairwise_names
    assert "aft_keyword_synchronous" in pairwise_names

    for row in results["pairwise_vs_full"]:
        assert row["baseline"] == "full"
        for metric in ("top1", "hit_at_k"):
            m = row[metric]
            assert "diff" in m
            assert "ci_lower" in m
            assert "ci_upper" in m
            assert "p_bootstrap" in m
            assert "p_mcnemar" in m
            assert 0.0 <= m["p_bootstrap"] <= 1.0
            assert 0.0 <= m["p_mcnemar"] <= 1.0

    assert results["statistics"]["n_bootstrap"] == 100
    assert results["statistics"]["ci_method"] == "bootstrap_percentile"


@pytest.mark.parametrize(
    "path,expected_suffix",
    [
        (DATASET, "realistic_recall_v1"),
    ],
)
def test_ablation_benchmark_id_derived_from_dataset(path: object, expected_suffix: str) -> None:
    from pathlib import Path

    dataset = load_dataset(Path(str(path)))
    results = run_ablation_study(dataset, n_bootstrap=20, seed=0)
    assert results["benchmark"] == f"ablation_{expected_suffix}"
    assert results["base_benchmark"] == expected_suffix


def test_ablation_study_emits_link_set_stats() -> None:
    dataset = load_dataset()
    results = run_ablation_study(dataset, n_bootstrap=20, seed=0)
    stats = results["link_set_stats"]
    assert "sessions" in stats
    assert "n_memories_total" in stats
    assert stats["sessions"] > 0
    assert stats["n_memories_total"] > 0
    lpm = stats["links_per_memory"]
    assert lpm["mean"] <= lpm["max"]
    assert lpm["max"] <= 5  # ResonanceConfig.max_links default
    assert lpm["mean"] >= 0.0


def test_collect_link_stats_invariants() -> None:
    from benchmarks.realistic.adapters.base import TokenHashEmbedder
    from emotional_memory import EmotionalMemory
    from emotional_memory.stores.in_memory import InMemoryStore

    em = EmotionalMemory(store=InMemoryStore(), embedder=TokenHashEmbedder())
    for i in range(6):
        em.encode(f"memory content number {i} about topic {i % 3}", metadata={})

    stats = _collect_link_stats(em)
    em.close()

    # structural invariants
    assert stats["n_memories"] == 6
    assert len(stats["per_memory_counts"]) == 6
    assert stats["links_per_memory"]["mean"] <= stats["links_per_memory"]["max"]
    assert stats["links_per_memory"]["max"] <= 5

    # link_types counts must equal len(link_strength_distribution)
    total_links = sum(stats["link_types"].values())
    assert total_links == len(stats["link_strength_distribution"])

    # all strengths in [0, 1]
    assert all(0.0 <= s <= 1.0 for s in stats["link_strength_distribution"])


def test_aggregate_link_stats_invariants() -> None:
    session_a = {
        "n_memories": 3,
        "per_memory_counts": [2, 3, 1],
        "link_types": {"semantic": 4, "temporal": 2},
        "link_strength_distribution": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    }
    session_b = {
        "n_memories": 2,
        "per_memory_counts": [5, 0],
        "link_types": {"semantic": 3, "emotional": 2},
        "link_strength_distribution": [1.0, 0.95, 0.85, 0.75, 0.65],
    }
    agg = _aggregate_link_stats([session_a, session_b])

    assert agg["sessions"] == 2
    assert agg["n_memories_total"] == 5  # 3 + 2
    assert agg["links_per_memory"]["max"] == 5
    assert agg["link_types"]["semantic"] == 7  # 4 + 3
    assert agg["link_types"]["temporal"] == 2
    assert agg["link_types"]["emotional"] == 2
    total = sum(agg["link_types"].values())
    assert total == len(agg["link_strength_distribution"])
    # sorted descending
    strengths = agg["link_strength_distribution"]
    assert strengths == sorted(strengths, reverse=True)


def test_aggregate_link_stats_empty() -> None:
    agg = _aggregate_link_stats([])
    assert agg["sessions"] == 0
    assert agg["n_memories_total"] == 0


# ---------------------------------------------------------------------------
# per_query_records tests (Step 3a — Hi3 prerequisite)
# ---------------------------------------------------------------------------

VARIANT_NAMES = [
    "full",
    "no_appraisal",
    "no_mood",
    "no_momentum",
    "no_resonance",
    "no_reconsolidation",
    "dual_path",
    "aft_keyword_synchronous",
]


def test_per_query_records_absent_by_default() -> None:
    dataset = load_dataset()
    results = run_ablation_study(dataset, n_bootstrap=20, seed=0)
    assert "per_query_records" not in results


def test_per_query_records_emitted_when_requested() -> None:
    dataset = load_dataset()
    results = run_ablation_study(dataset, n_bootstrap=20, seed=0, emit_per_query=True)
    assert "per_query_records" in results
    pqr = results["per_query_records"]
    assert set(pqr.keys()) == set(VARIANT_NAMES)
    # All variants must have the same number of records
    counts = {variant: len(records) for variant, records in pqr.items()}
    assert len(set(counts.values())) == 1, f"Inconsistent per-query counts: {counts}"


def test_per_query_records_alignment() -> None:
    dataset = load_dataset()
    results = run_ablation_study(dataset, n_bootstrap=20, seed=0, emit_per_query=True)
    pqr = results["per_query_records"]

    # Each variant's records must be sorted by query_id
    for variant, records in pqr.items():
        ids = [r["query_id"] for r in records]
        assert ids == sorted(ids), f"Records not sorted for variant {variant!r}"

    # All variants must share the same query_id set
    id_sets = [frozenset(r["query_id"] for r in records) for records in pqr.values()]
    assert len(set(id_sets)) == 1, "Variants have different query_id sets"


def test_per_query_records_field_types() -> None:
    dataset = load_dataset()
    results = run_ablation_study(dataset, n_bootstrap=20, seed=0, emit_per_query=True)
    for records in results["per_query_records"].values():
        for rec in records:
            assert isinstance(rec["query_id"], str)
            assert isinstance(rec["scenario_id"], str)
            assert isinstance(rec["challenge_type"], str)
            assert rec["challenge_type"] in CHALLENGE_TYPES
            assert isinstance(rec["top1_hit"], bool)
            assert isinstance(rec["hit"], bool)


def test_v2_results_top_level_keys_unchanged() -> None:
    """Regression: emit_per_query=False produces the same top-level key set as v2."""
    expected_keys = {
        "benchmark",
        "base_benchmark",
        "variants",
        "pairwise_vs_full",
        "hf1_pairwise",
        "link_set_stats",
        "statistics",
    }
    dataset = load_dataset()
    results = run_ablation_study(dataset, n_bootstrap=20, seed=0)
    assert set(results.keys()) == expected_keys
