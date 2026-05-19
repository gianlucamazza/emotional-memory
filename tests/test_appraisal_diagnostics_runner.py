"""Smoke tests for benchmarks/appraisal_diagnostics/runner.py.

Run via:  uv run python -m pytest tests/test_appraisal_diagnostics_runner.py -v
All tests use --dry-run mode (no LLM API key required).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DATASET = _ROOT / "benchmarks" / "datasets" / "realistic_recall_v3.json"


def test_dataset_exists() -> None:
    assert _DATASET.exists(), f"Dataset not found: {_DATASET}"


def test_dry_run_completes(tmp_path: pytest.TempPathFactory) -> None:
    from benchmarks.appraisal_diagnostics.runner import run

    result = run(dry_run=True, n=5, seed=42, n_bootstrap=100)

    assert result["n_events"] == 5
    assert result["dry_run"] is True
    assert "decision" in result
    assert result["decision"].startswith("P1")


def test_dry_run_output_shape(tmp_path: pytest.TempPathFactory) -> None:
    from benchmarks.appraisal_diagnostics.runner import run

    result = run(dry_run=True, n=10, seed=0, n_bootstrap=50)

    # Valence residuals
    vr = result["valence_residuals"]
    assert vr["n"] == 10
    assert "bias_mean" in vr
    assert "bias_ci_lo" in vr
    assert "bias_ci_hi" in vr
    assert "std" in vr
    assert "mae" in vr
    assert "pearson_r_with_oracle" in vr

    # SEC descriptive — all 5 dimensions present
    sec = result["sec_descriptive"]
    for dim in (
        "novelty",
        "goal_relevance",
        "coping_potential",
        "norm_congruence",
        "self_relevance",
    ):
        assert dim in sec, f"Missing SEC dimension: {dim}"
        assert "mean" in sec[dim]

    # Confusion matrix
    conf = result["valence_sign_confusion"]
    assert conf["TP"] + conf["FP"] + conf["TN"] + conf["FN"] == 10

    # Latency
    assert result["latency"]["mean_ms"] >= 0.0


def test_dry_run_decision_is_string() -> None:
    from benchmarks.appraisal_diagnostics.runner import run

    result = run(dry_run=True, n=3, seed=1, n_bootstrap=20)
    assert isinstance(result["decision"], str)
    assert len(result["decision"]) > 0


def test_render_md_produces_output() -> None:
    from benchmarks.appraisal_diagnostics.runner import render_md, run

    result = run(dry_run=True, n=5, seed=42, n_bootstrap=50)
    md = render_md(result)

    assert "# Appraisal Diagnostics" in md
    assert "Residuals" in md
    assert "SEC Dimension" in md
    assert "Valence Sign Confusion" in md
    assert "Decision" in md


def test_cli_dry_run_writes_files(tmp_path: Path) -> None:
    from benchmarks.appraisal_diagnostics.runner import main

    out_json = tmp_path / "diag.json"
    out_md = tmp_path / "diag.md"
    main(
        [
            "--dry-run",
            "--n",
            "5",
            "--n-bootstrap",
            "20",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert out_json.exists()
    assert out_md.exists()
    data = json.loads(out_json.read_text())
    assert data["dry_run"] is True
    assert data["n_events"] == 5
