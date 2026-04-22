"""Regression checks for repository-level configuration contracts."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_ruff_configuration_lives_in_pyproject() -> None:
    root = _repo_root()
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    tool_config = pyproject["tool"]
    assert "ruff" in tool_config
    assert not (root / "ruff.toml").exists()


def test_demo_extra_declares_gradio_runtime() -> None:
    pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))

    optional_deps = pyproject["project"]["optional-dependencies"]
    assert "demo" in optional_deps
    assert any(dep.startswith("gradio>=") for dep in optional_deps["demo"])


def test_redis_extra_declared_for_shared_state_backend() -> None:
    pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))

    optional_deps = pyproject["project"]["optional-dependencies"]
    assert "redis" in optional_deps
    assert any(dep.startswith("redis>=") for dep in optional_deps["redis"])


def test_demo_requirements_pin_current_package_version() -> None:
    root = _repo_root()
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]

    requirements = (root / "demo" / "requirements.txt").read_text(encoding="utf-8").splitlines()
    effective_lines = [line for line in requirements if line and not line.startswith("#")]

    assert f"emotional-memory=={version}" == effective_lines[0]
    assert "gradio>=6.13,<7" in effective_lines


def test_env_example_lists_demo_ssr_flag() -> None:
    env_example = (_repo_root() / ".env.example").read_text(encoding="utf-8")

    assert "EMOTIONAL_MEMORY_DEMO_SSR=" in env_example


def test_citation_version_matches_pyproject() -> None:
    root = _repo_root()
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")

    assert f'version: "{version}"' in citation


def test_changelog_contains_current_release_heading() -> None:
    root = _repo_root()
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    pattern = rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}"

    assert "## [Unreleased]" in changelog
    assert re.search(pattern, changelog, re.MULTILINE)


def test_makefile_exposes_release_metadata_targets() -> None:
    makefile = (_repo_root() / "Makefile").read_text(encoding="utf-8")

    assert "meta-check:" in makefile
    assert "meta-check-local:" in makefile
    assert "verify-pypi-release:" in makefile
    assert "sync-release-metadata:" in makefile
    assert "bench-realistic:" in makefile
    assert "human-eval-packets:" in makefile
    assert "human-eval-summary:" in makefile


def test_ci_workflow_checks_release_metadata() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "meta-integrity:" in workflow
    assert "scripts/check_release_metadata.py" in workflow
    assert "scripts/preflight.py --fast --ci" in workflow


def test_release_workflow_verifies_pypi_and_uploads_artefacts() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "scripts/preflight.py --fast --ci" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "scripts/verify_pypi_release.py" in workflow


def test_public_docs_frame_validation_status_conservatively() -> None:
    root = _repo_root()
    readme = (root / "README.md").read_text(encoding="utf-8")
    state_of_art = (root / "docs" / "research" / "04_state_of_art.md").read_text(encoding="utf-8")
    benchmark_readme = (root / "benchmarks" / "comparative" / "README.md").read_text(
        encoding="utf-8"
    )
    evidence_page = (root / "docs" / "research" / "09_current_evidence.md").read_text(
        encoding="utf-8"
    )
    limitations = (root / "docs" / "research" / "08_limitations.md").read_text(encoding="utf-8")
    human_eval_readme = (root / "benchmarks" / "human_eval" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "## Current validation status" in readme
    assert "only system with an explicit multi-layer emotional model" not in readme
    assert "Siamo gli unici" not in state_of_art
    assert "controlled synthetic benchmark" in benchmark_readme
    assert "does **not establish general downstream superiority**" in benchmark_readme
    assert "## Study ladder" in evidence_page
    assert "## Claim matrix" in evidence_page
    assert "replayable" in evidence_page
    assert "RedisAffectiveStateStore" in limitations
    assert "No checked-in `summary.json` / `summary.md` artifacts" in human_eval_readme
