"""Deterministic tests for the Gate 2 paired condition-significance test.

Lightweight (imports only `benchmarks.human_eval.pipeline`, which needs numpy — a
core dep — but no bench/LLM extras), so it runs in the default test job rather than
the slow benchmark job.
"""

from __future__ import annotations

from typing import Any

from benchmarks.human_eval.pipeline import summarize_ratings

DIMS = ("affective_coherence", "usefulness", "continuity", "plausibility")


def _ratings(
    aft_base: int, cosine_base: int, *, scenarios: int = 10, raters: int = 3
) -> list[dict[str, Any]]:
    """Completed ratings where aft is rated `aft_base` and cosine `cosine_base` on every dim."""
    rows: list[dict[str, Any]] = []
    for s in range(scenarios):
        for r in range(raters):
            for cond, base in (("aft", aft_base), ("naive_cosine", cosine_base)):
                rows.append(
                    {
                        "scenario_id": f"scn_{s}",
                        "condition": cond,
                        "rater_id": f"r{r}",
                        "ratings": dict.fromkeys(DIMS, base),
                        "note": "",
                    }
                )
    return rows


def test_significance_detects_aft_advantage() -> None:
    summary = summarize_ratings(_ratings(aft_base=4, cosine_base=2))
    sig = summary["condition_significance_by_dimension"]
    assert set(sig) == set(DIMS)
    for dim in DIMS:
        s = sig[dim]
        assert s["contrast"] == "aft - naive_cosine"
        assert s["n_pairs"] == 30  # 10 scenarios x 3 raters
        assert s["mean_diff"] == 2.0
        assert s["p"] < 0.05
        assert s["significant_favoring_treatment"] is True


def test_significance_null_when_conditions_tie() -> None:
    summary = summarize_ratings(_ratings(aft_base=3, cosine_base=3))
    for s in summary["condition_significance_by_dimension"].values():
        assert s["mean_diff"] == 0.0
        assert s["significant_favoring_treatment"] is False


def test_significance_handles_unpaired_conditions() -> None:
    # Only the aft condition present → no pairs → graceful null, no crash.
    rows = [r for r in _ratings(aft_base=4, cosine_base=2) if r["condition"] == "aft"]
    summary = summarize_ratings(rows)
    for s in summary["condition_significance_by_dimension"].values():
        assert s["n_pairs"] == 0
        assert s["mean_diff"] is None
        assert s["significant_favoring_treatment"] is False


def test_significance_is_deterministic() -> None:
    rows = _ratings(aft_base=4, cosine_base=2)
    assert (
        summarize_ratings(rows)["condition_significance_by_dimension"]
        == summarize_ratings(rows)["condition_significance_by_dimension"]
    )
