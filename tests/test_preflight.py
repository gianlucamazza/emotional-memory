"""Tests for scripts/preflight.py URL host checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    path = ROOT / "scripts" / "preflight.py"
    spec = importlib.util.spec_from_file_location("tests.scripts.preflight", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_is_zenodo_webhook_url_accepts_official_hosts() -> None:
    module = _load()
    assert module._is_zenodo_webhook_url("https://zenodo.org/api/hooks/github")
    assert module._is_zenodo_webhook_url("https://sandbox.zenodo.org/api/hooks/github")


def test_is_zenodo_webhook_url_rejects_lookalikes() -> None:
    module = _load()
    lookalikes = (
        "https://evil.example/zenodo.org",
        "https://notzenodo.org/hooks",
        "https://zenodo.org.evil.example/hooks",
        "https://zenodo.org@evil.example/hooks",
        "https://example.com/?next=https://zenodo.org",
        "not-a-url",
        "",
    )
    for url in lookalikes:
        assert not module._is_zenodo_webhook_url(url), url
    assert not module._is_zenodo_webhook_url(None)
