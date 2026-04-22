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
