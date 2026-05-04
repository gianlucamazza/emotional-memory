"""Tests for scripts/generate_research_figures.py."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")


def test_generate_research_figures_writes_expected_outputs(tmp_path: object) -> None:
    from pathlib import Path

    import scripts.generate_research_figures as grf

    tmp = Path(str(tmp_path))
    png_dir = tmp / "png"
    pdf_dir = tmp / "pdf"

    grf.generate(png_dir, pdf_dir)

    expected = {
        "research_realistic_v2_overview",
        "research_realistic_v2_challenges",
        "research_ablation_s3",
        "research_multilingual_it",
        "research_locomo_negative",
    }
    assert {p.stem for p in png_dir.glob("*.png")} == expected
    assert {p.stem for p in pdf_dir.glob("*.pdf")} == expected


def test_load_json_reports_missing_required_artifact(tmp_path: object) -> None:
    from pathlib import Path

    import scripts.generate_research_figures as grf

    with pytest.raises(FileNotFoundError, match="Required benchmark artefact"):
        grf._load_json(Path(str(tmp_path)) / "missing.json")
