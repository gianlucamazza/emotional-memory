"""Smoke tests for benchmarks/dailydialog/t2a_runner.py (Addendum T2A).

Keyless and model-free: exercises the runner's pure logic (Pearson, verdict,
output writers, adapter factory). The full 3-arm benchmark needs an LLM key for
the aft_query_appraised arm and is covered by `make bench-t2a-dailydialog` in CI.

Run via:  uv run python -m pytest tests/test_t2a_runner.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PERSONAS = _ROOT / "benchmarks" / "datasets" / "dailydialog_personas_v1.json"


def test_personas_dataset_exists() -> None:
    assert _PERSONAS.exists(), f"Persona dataset not found: {_PERSONAS}"


def test_pearson_matches_numpy() -> None:
    from benchmarks.dailydialog.t2a_runner import _pearson

    xs = [0.1, 0.4, 0.2, 0.9, 0.5]
    ys = [0.2, 0.3, 0.25, 0.8, 0.55]
    assert _pearson(xs, ys) == pytest.approx(0.94, abs=0.05)
    # Degenerate input returns NaN, not an exception.
    assert math.isnan(_pearson([1.0], [1.0]))


def test_make_adapter_rejects_unknown() -> None:
    from benchmarks.dailydialog.t2a_runner import _make_adapter

    with pytest.raises(ValueError, match="Unknown system"):
        _make_adapter("nope", embedder_name="multilingual-e5-small")


def test_verdict_pass_and_fail() -> None:
    from benchmarks.dailydialog.t2a_runner import _verdict

    passing = {"delta": 0.12, "ci_lower": 0.05, "ci_upper": 0.2, "p_holm": 0.01, "pass_holm": True}
    failing = {"delta": -0.008, "ci_lower": -0.056, "ci_upper": 0.04, "p_holm": 1.0}
    assert _verdict(passing).startswith("PASS")
    assert _verdict(failing).startswith("FAIL")


def _synthetic_results() -> dict:
    """A results dict shaped like run_benchmark() output, with the T2A FAIL numbers."""

    def stats(delta: float, lo: float, hi: float, p: float, *, holm: float, ok: bool) -> dict:
        return {
            "delta": delta,
            "ci_lower": lo,
            "ci_upper": hi,
            "p_bootstrap_onetail": p,
            "p_holm": holm,
            "pass_holm": ok,
            "cohens_d": 0.0,
            "n": 396,
        }

    types = ["emotion_state_recall", "affect_conditioned_content", "affective_trajectory"]
    primary_stats = {"aggregate": stats(-0.008, -0.056, 0.04, 0.602, holm=1.0, ok=False)}
    for t in types:
        primary_stats[t] = stats(0.0, -0.09, 0.09, 0.5, holm=1.0, ok=False)

    return {
        "benchmark": "t2a_dailydialog_query_appraisal",
        "pre_registration": (
            "benchmarks/preregistration_addendum_t2a_naturalistic_query_appraisal.md"
        ),
        "dataset_version": "1",
        "n_personas": 120,
        "embedder": "multilingual-e5-small",
        "n_bootstrap": 10_000,
        "seed": 0,
        "systems": [
            {
                "system": s,
                "n_queries": 396,
                "top1_accuracy": acc,
                "hit_at_k": acc + 0.17,
                "per_type": [{"query_type": t, "n": 120, "top1_accuracy": acc} for t in types],
            }
            for s, acc in (("naive_cosine", 0.220), ("aft", 0.202), ("aft_query_appraised", 0.212))
        ],
        "pairwise_comparisons": [
            {"system": "aft_query_appraised", "baseline": "naive_cosine", "stats": primary_stats},
            {
                "system": "aft_query_appraised",
                "baseline": "aft",
                "stats": {
                    "aggregate": stats(0.010, -0.013, 0.033, 0.224, holm=float("nan"), ok=False)
                },
            },
            {
                "system": "aft",
                "baseline": "naive_cosine",
                "stats": {
                    "aggregate": stats(-0.018, -0.066, 0.03, 0.753, holm=float("nan"), ok=False)
                },
            },
        ],
        "diagnostic": {"n": 396, "valence_r": 0.69, "arousal_r": 0.74},
        "ht2a_pass": False,
        "n_directional_types_pass": 0,
    }


def test_write_results_round_trip(tmp_path: Path) -> None:
    from benchmarks.dailydialog.t2a_runner import write_results

    out_json = tmp_path / "t2a.json"
    out_md = tmp_path / "t2a.md"
    out_protocol = tmp_path / "t2a.protocol.json"
    write_results(
        _synthetic_results(), out_json=out_json, out_md=out_md, out_protocol=out_protocol
    )

    assert out_json.exists() and out_md.exists() and out_protocol.exists()

    md = out_md.read_text(encoding="utf-8")
    assert "Addendum T2A" in md
    assert "**Ht2a verdict: FAIL**" in md
    assert "valence r=0.690" in md  # diagnostic rendered

    protocol = json.loads(out_protocol.read_text(encoding="utf-8"))
    assert protocol["primary_contrast"] == "aft_query_appraised vs naive_cosine"
    assert protocol["ht2a_pass"] is False

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["ht2a_pass"] is False
    assert len(data["systems"]) == 3
