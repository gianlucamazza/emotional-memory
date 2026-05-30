"""Tests for scripts/check_doi_freshness.py (release-DOI anti-recurrence gate).

These exercise the pure verdict layer directly (no subprocess, no git, no real
tags), so they are deterministic and fast. The git I/O helpers are thin wrappers
validated end-to-end by the release workflow itself.
"""

from __future__ import annotations

import check_doi_freshness as gate

_TOML = '[release]\nconcept_doi = "10.5281/zenodo.1"\nversion_doi = "{doi}"\n'


def test_parse_version_doi_reads_value() -> None:
    assert gate.parse_version_doi(_TOML.format(doi="10.5281/zenodo.42")) == "10.5281/zenodo.42"


def test_parse_version_doi_missing_returns_none() -> None:
    assert gate.parse_version_doi('[release]\nconcept_doi = "x"\n') is None


def test_fails_when_version_doi_unchanged() -> None:
    """Reusing the previous release's DOI (the v0.11.2/v0.11.3 bug) must fail."""
    verdict = gate.evaluate("10.5281/zenodo.20440996", "10.5281/zenodo.20440996")
    assert verdict.code == 1
    assert "version_doi" in verdict.message


def test_passes_when_version_doi_changed() -> None:
    verdict = gate.evaluate("10.5281/zenodo.20440996", "10.5281/zenodo.20070143")
    assert verdict.code == 0
    assert "changed" in verdict.message


def test_passes_when_no_previous_doi() -> None:
    """First release / unreadable previous toml → skip (pass)."""
    verdict = gate.evaluate("10.5281/zenodo.1", None)
    assert verdict.code == 0
    assert "skipped" in verdict.message


def test_fails_when_current_doi_missing() -> None:
    verdict = gate.evaluate(None, "10.5281/zenodo.1")
    assert verdict.code == 1
    assert "could not read" in verdict.message
