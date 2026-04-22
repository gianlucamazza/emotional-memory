from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tests.scripts.{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _current_citation_doi() -> str:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r'^doi:\s*"([^"]+)"$', citation, re.MULTILINE)
    assert match is not None
    return match.group(1)


def _current_readme_concept_doi() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"\[!\[DOI\].*?\]\(https://doi\.org/([^)]+)\)", readme)
    assert match is not None
    return match.group(1)


def _current_version() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(pyproject["project"]["version"])


def test_check_release_metadata_passes_for_current_repo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module("check_release_metadata")

    assert module.main([]) == 0

    out = capsys.readouterr().out
    assert "release metadata OK for" in out


def test_sync_release_metadata_is_idempotent_for_current_repo() -> None:
    module = _load_script_module("sync_release_metadata")

    changed = module.sync_release_metadata(
        _current_citation_doi(),
        _current_readme_concept_doi(),
        dry_run=True,
    )

    assert changed == []


def test_verify_pypi_release_succeeds_when_version_is_visible(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script_module("verify_pypi_release")
    version = _current_version()

    monkeypatch.setattr(
        module,
        "_fetch_release_payload",
        lambda: {
            "info": {"version": "999.0.0"},
            "releases": {
                version: [{"filename": f"emotional_memory-{version}-py3-none-any.whl"}],
            },
        },
    )
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    assert module.main([version, "--timeout-seconds", "1", "--interval-seconds", "0"]) == 0

    out = capsys.readouterr().out
    assert f"https://pypi.org/project/emotional-memory/{version}/" in out
