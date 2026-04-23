"""Integration tests for the ablation study runner."""

from __future__ import annotations

from benchmarks.ablation.runner import run_ablation_study
from benchmarks.realistic.runner import load_dataset


def test_ablation_study_structure() -> None:
    dataset = load_dataset()
    results = run_ablation_study(dataset, n_bootstrap=100, seed=0)

    assert results["benchmark"] == "ablation_realistic_v1"
    assert len(results["variants"]) == 5

    variant_names = [v["variant"] for v in results["variants"]]
    assert variant_names == ["full", "no_appraisal", "no_mood", "no_momentum", "no_resonance"]

    for v in results["variants"]:
        assert "aggregate_metrics" in v
        assert "challenge_type_metrics" in v
        assert isinstance(v["aggregate_metrics"]["top1_accuracy"], float)

    assert len(results["pairwise_vs_full"]) == 4
    pairwise_names = [r["variant"] for r in results["pairwise_vs_full"]]
    assert "no_appraisal" in pairwise_names
    assert "no_resonance" in pairwise_names

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
