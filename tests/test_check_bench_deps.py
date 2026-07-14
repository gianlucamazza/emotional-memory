"""Tests for scripts/check_bench_deps.py."""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("httpx")
pytest.importorskip("dotenv")
pytest.importorskip("sentence_transformers")

from scripts.check_bench_deps import main as check_main


def test_check_bench_deps_strict_passes_when_installed() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["check_bench_deps.py", "--strict"])
        assert check_main() == 0


def test_check_bench_deps_skip_embedder_passes_without_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sentence_transformers":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setattr(sys, "argv", ["check_bench_deps.py", "--strict", "--skip-embedder"])
    assert check_main() == 0
