"""Tests for scripts/check_metadata_ssot.py (SSOT + ReDoS-safe CFF parse)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    path = ROOT / "scripts" / "check_metadata_ssot.py"
    spec = importlib.util.spec_from_file_location("tests.scripts.check_metadata_ssot", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cff_first_author_reads_current_citation() -> None:
    module = _load()
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    author = module._cff_first_author(text)
    assert author["family-names"] == "Mazza"
    assert author["given-names"] == "Gianluca"
    assert author["email"] == "info@gianlucamazza.it"


def test_cff_first_author_strips_quotes_and_stops_at_second_author() -> None:
    module = _load()
    text = (
        "title: example\n"
        "authors:\n"
        "  - family-names: 'Doe'\n"
        '    given-names: "Jane Q"\n'
        "    email: jane@example.com\n"
        "    orcid: https://orcid.org/0000-0000-0000-0000\n"
        "  - family-names: Other\n"
        "    given-names: Person\n"
        "license: MIT\n"
    )
    assert module._cff_first_author(text) == {
        "family-names": "Doe",
        "given-names": "Jane Q",
        "email": "jane@example.com",
    }


def test_cff_first_author_returns_empty_without_both_name_fields() -> None:
    module = _load()
    assert module._cff_first_author("license: MIT\n") == {}
    assert module._cff_first_author("authors:\n  - family-names: Only\n") == {}


def test_cff_first_author_is_linear_on_adversarial_input() -> None:
    """Former ``(?:  -[\\s\\S]*?)+?`` matcher was polynomial ReDoS."""
    module = _load()
    blob = "authors:\n" + ("  - " * 20000) + "family-names: X\n"
    assert module._cff_first_author(blob) == {}


def test_read_cff_field_license_from_current_citation() -> None:
    module = _load()
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert module._read_cff_field(text, "license") == "MIT"


def test_check_metadata_ssot_passes_for_current_repo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load()
    assert module.main() == 0
    assert "OK: metadata SSOT" in capsys.readouterr().out
