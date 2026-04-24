"""Tests for scripts/reproduce_paper.py resolver logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def _tmp_comp_dir(tmp_path: Path) -> Path:
    """Create a temporary comparative benchmark directory."""
    d = tmp_path / "benchmarks" / "comparative"
    d.mkdir(parents=True)
    return d


def _call_resolver(comp_dir: Path) -> tuple[Path, str]:
    """Call _resolve_comparative_csv with patched module-level paths."""
    import scripts.reproduce_paper as rp

    orig_sbert = rp.COMPARATIVE_CSV_SBERT
    orig_hash = rp.COMPARATIVE_CSV
    rp.COMPARATIVE_CSV_SBERT = comp_dir / "results.sbert.csv"
    rp.COMPARATIVE_CSV = comp_dir / "results.csv"
    try:
        return rp._resolve_comparative_csv()
    finally:
        rp.COMPARATIVE_CSV_SBERT = orig_sbert
        rp.COMPARATIVE_CSV = orig_hash


def test_resolve_prefers_sbert_when_both_exist(_tmp_comp_dir: Path) -> None:
    (_tmp_comp_dir / "results.sbert.csv").write_text("header\n")
    (_tmp_comp_dir / "results.csv").write_text("header\n")
    path, label = _call_resolver(_tmp_comp_dir)
    assert path.name == "results.sbert.csv"
    assert label == "sbert"


def test_resolve_falls_back_to_hash_when_sbert_missing(_tmp_comp_dir: Path) -> None:
    (_tmp_comp_dir / "results.csv").write_text("header\n")
    path, label = _call_resolver(_tmp_comp_dir)
    assert path.name == "results.csv"
    assert label == "hash"


def test_resolve_falls_back_to_hash_when_neither_exists(_tmp_comp_dir: Path) -> None:
    """When neither file exists the resolver falls back to hash path (caller handles missing)."""
    path, label = _call_resolver(_tmp_comp_dir)
    assert path.name == "results.csv"
    assert label == "hash"
