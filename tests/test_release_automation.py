from __future__ import annotations

import importlib.util
import json
import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


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


def test_sync_readme_doi_badge_migrates_legacy_zenodo_badge() -> None:
    module = _load_script_module("sync_release_metadata")
    legacy = (
        "[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1.svg)]"
        "(https://doi.org/10.5281/zenodo.1)"
    )

    updated = module._sync_readme_doi_badge(legacy, "10.5281/zenodo.2")

    assert updated == (
        "[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.2-blue?logo=doi)]"
        "(https://doi.org/10.5281/zenodo.2)"
    )


def test_sync_demo_readme_updates_version_doi_and_cite_note() -> None:
    module = _load_script_module("sync_release_metadata")
    demo_path = ROOT / "demo" / "README.md"
    original = demo_path.read_text(encoding="utf-8")
    version = "9.9.9"
    version_doi = "10.5281/zenodo.99"
    concept_doi = "10.5281/zenodo.9"

    updated = module._replace(
        r"- \*\*Zenodo Concept DOI\*\*( \(all versions\))?: \[[^\]]+\]\(https://doi\.org/[^\)]+\)",
        rf"- **Zenodo Concept DOI**\g<1>: [{concept_doi}](https://doi.org/{concept_doi})",
        original,
        "demo README concept DOI",
    )
    updated = module._replace(
        r"- \*\*Zenodo Version DOI\*\* \([^)]+\): \[[^\]]+\]\(https://doi\.org/[^\)]+\)",
        f"- **Zenodo Version DOI** ({version}): [{version_doi}](https://doi.org/{version_doi})",
        updated,
        "demo README version DOI",
    )
    updated = module._replace(
        r"^To cite all versions rather than [^,]+, use the concept DOI\n\[[^\]]+\]\(https://doi\.org/[^\)]+\)\.$",
        f"To cite all versions rather than {version}, use the concept DOI\n[{concept_doi}](https://doi.org/{concept_doi}).",
        updated,
        "demo README cite-all-versions note",
    )

    assert (
        "- **Zenodo Version DOI** (9.9.9): [10.5281/zenodo.99](https://doi.org/10.5281/zenodo.99)"
    ) in updated
    assert (
        "To cite all versions rather than 9.9.9, use the concept DOI\n"
        "[10.5281/zenodo.9](https://doi.org/10.5281/zenodo.9)."
    ) in updated
    assert (
        "- **Zenodo Concept DOI** (all versions): "
        "[10.5281/zenodo.9](https://doi.org/10.5281/zenodo.9)"
    ) in updated
    assert demo_path.read_text(encoding="utf-8") == original


def test_sync_readme_doi_badge_updates_icon_badge() -> None:
    module = _load_script_module("sync_release_metadata")
    current = (
        "[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.1-blue?logo=doi)]"
        "(https://doi.org/10.5281/zenodo.1)"
    )

    updated = module._sync_readme_doi_badge(current, "10.5281/zenodo.2")

    assert updated == (
        "[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.2-blue?logo=doi)]"
        "(https://doi.org/10.5281/zenodo.2)"
    )


def test_codemeta_related_link_pins_concept_doi() -> None:
    # Regression: a stale per-version Zenodo DOI sat in relatedLink for 8
    # releases before sync_release_metadata learned to manage the field.
    import json

    data = json.loads((ROOT / "codemeta.json").read_text(encoding="utf-8"))
    zenodo_links = [
        link for link in data["relatedLink"] if link.startswith("https://doi.org/10.5281/zenodo.")
    ]
    assert zenodo_links == [f"https://doi.org/{_current_readme_concept_doi()}"]


def test_require_zenodo_base_url_accepts_official_hosts() -> None:
    module = _load_script_module("sync_release_metadata")
    assert module._require_zenodo_base_url("https://zenodo.org/") == "https://zenodo.org"
    assert (
        module._require_zenodo_base_url("https://sandbox.zenodo.org")
        == "https://sandbox.zenodo.org"
    )


def test_require_zenodo_base_url_rejects_lookalike_hosts() -> None:
    module = _load_script_module("sync_release_metadata")
    lookalikes = (
        "https://evil.example/zenodo.org",
        "https://notzenodo.org",
        "https://zenodo.org.evil.example",
        "https://zenodo.org@evil.example",
        "https://example.com/?next=https://zenodo.org",
    )
    for url in lookalikes:
        with pytest.raises(SystemExit, match="Refusing non-Zenodo"):
            module._require_zenodo_base_url(url)


def test_is_arxiv_identifier_accepts_arxiv_hosts() -> None:
    module = _load_script_module("sync_release_metadata")
    accepted = (
        "https://arxiv.org/abs/2601.00001",
        "http://arxiv.org/pdf/2601.00001",
        "https://export.arxiv.org/pdf/2601.00001.pdf",
        "https://www.arxiv.org/abs/2601.00001",
    )
    for url in accepted:
        assert module._is_arxiv_identifier(url), url


def test_is_arxiv_identifier_rejects_lookalike_hosts() -> None:
    module = _load_script_module("sync_release_metadata")
    lookalikes = (
        "https://evil.example/arxiv.org",
        "https://notarxiv.org/abs/2601.00001",
        "https://arxiv.org.evil.example/abs/2601.00001",
        "https://arxiv.org@evil.example/abs/2601.00001",
        "https://example.com/?next=https://arxiv.org",
        "https://arxiv.org.attacker.test",
        "not-a-url",
        "",
    )
    for url in lookalikes:
        assert not module._is_arxiv_identifier(url), url
    assert not module._is_arxiv_identifier(None)


_SYNC_FIXTURE_FILES = (
    "pyproject.toml",
    "README.md",
    "CITATION.cff",
    ".zenodo.json",
    "codemeta.json",
    "demo/README.md",
    "demo/app.py",
    "demo/requirements.txt",
    "paper/SUBMISSION.md",
    "paper/main.tex",
)

_LOOKALIKE_ARXIV_IDENTIFIERS = (
    "https://arxiv.org.evil.com/abs/1111.11111",
    "https://evilarxiv.org/abs/1111.11111",
    "https://evil.example/arxiv.org",
)


def _prepare_sync_root(tmp_path: Path) -> Path:
    """Copy only the files ``sync_release_metadata`` reads or writes."""
    for rel in _SYNC_FIXTURE_FILES:
        src = ROOT / rel
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
    return tmp_path


def _write_related_identifiers(root: Path, identifiers: list[str]) -> None:
    path = root / ".zenodo.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["related_identifiers"] = [
        {"identifier": value, "relation": "isSupplementTo", "scheme": "url"}
        for value in identifiers
    ]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _related_identifier_values(root: Path) -> list[str]:
    data = json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    return [str(entry["identifier"]) for entry in data["related_identifiers"]]


def _run_arxiv_related_identifier_sync(root: Path, arxiv_id: str) -> list[str]:
    module = _load_script_module("sync_release_metadata")
    module.ROOT = root
    module.sync_release_metadata(
        _current_citation_doi(),
        _current_readme_concept_doi(),
        dry_run=False,
        arxiv_id=arxiv_id,
    )
    return _related_identifier_values(root)


def test_sync_release_metadata_preserves_lookalike_arxiv_identifiers(
    tmp_path: Path,
) -> None:
    """Sync must not treat lookalike hosts as arXiv related identifiers.

    The helper tests call ``_is_arxiv_identifier`` directly. This exercises
    the ``.zenodo.json`` update branch with a bare ``arxiv_id`` so a
    regression to ``"arxiv.org" in identifier`` would fail.
    """
    github = "https://github.com/gianlucamazza/emotional-memory"
    arxiv_id = "2601.00001"
    expected = f"https://arxiv.org/abs/{arxiv_id}"

    append_root = _prepare_sync_root(tmp_path / "append")
    _write_related_identifiers(append_root, [github, *_LOOKALIKE_ARXIV_IDENTIFIERS])
    append_values = _run_arxiv_related_identifier_sync(append_root, arxiv_id)
    assert append_values == [github, *_LOOKALIKE_ARXIV_IDENTIFIERS, expected]

    stale_arxiv = "https://arxiv.org/abs/1111.11111"
    stale_export = "https://export.arxiv.org/pdf/1111.11111"
    update_root = _prepare_sync_root(tmp_path / "update")
    _write_related_identifiers(
        update_root,
        [github, *_LOOKALIKE_ARXIV_IDENTIFIERS, stale_arxiv, stale_export],
    )
    update_values = _run_arxiv_related_identifier_sync(update_root, arxiv_id)
    assert update_values == [github, *_LOOKALIKE_ARXIV_IDENTIFIERS, expected, expected]


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
