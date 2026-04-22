"""Smoke tests for the examples/ scripts."""

import importlib.util
import runpy
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

_sqlite_available = importlib.util.find_spec("sqlite_vec") is not None
_matplotlib_available = importlib.util.find_spec("matplotlib") is not None


def test_basic_usage_runs() -> None:
    runpy.run_path(str(EXAMPLES_DIR / "basic_usage.py"), run_name="__main__")


def test_advanced_config_runs() -> None:
    runpy.run_path(str(EXAMPLES_DIR / "advanced_config.py"), run_name="__main__")


def test_appraisal_engines_runs() -> None:
    runpy.run_path(str(EXAMPLES_DIR / "appraisal_engines.py"), run_name="__main__")


def test_reconsolidation_runs() -> None:
    runpy.run_path(str(EXAMPLES_DIR / "reconsolidation.py"), run_name="__main__")


def test_async_usage_runs() -> None:
    runpy.run_path(str(EXAMPLES_DIR / "async_usage.py"), run_name="__main__")


@pytest.mark.skipif(not _matplotlib_available, reason="requires emotional-memory[viz]")
def test_retrieval_signals_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    runpy.run_path(str(EXAMPLES_DIR / "retrieval_signals.py"), run_name="__main__")


def test_retrieval_signals_uses_public_explainable_api() -> None:
    source = (EXAMPLES_DIR / "retrieval_signals.py").read_text()

    assert "retrieve_with_explanations" in source
    for private_name in (
        "_cosine",
        "_mood_congruence",
        "_affect_proximity",
        "_momentum_alignment",
        "_resonance_boost",
    ):
        assert private_name not in source


@pytest.mark.skipif(not _sqlite_available, reason="requires emotional-memory[sqlite]")
def test_persistence_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(EXAMPLES_DIR / "persistence.py"), run_name="__main__")


@pytest.mark.skipif(not _sqlite_available, reason="requires emotional-memory[sqlite]")
def test_emotional_journal_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(EXAMPLES_DIR / "emotional_journal.py"), run_name="__main__")
