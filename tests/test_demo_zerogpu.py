"""CPU-only checks for the ZeroGPU Space source in ``demo/zerogpu/``."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
ZEROGPU_DIR = ROOT / "demo" / "zerogpu"
APP_PATH = ZEROGPU_DIR / "app.py"

_TELEMETRY_KEYS = (
    "GRADIO_ANALYTICS_ENABLED",
    "HF_HUB_DISABLE_TELEMETRY",
    "DO_NOT_TRACK",
)


def _isolate_telemetry_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore Gradio/hub telemetry env mutations after the test."""
    for key in _TELEMETRY_KEYS:
        current = os.environ.get(key)
        if current is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, current)


def _recording_spaces_module() -> types.ModuleType:
    stub = types.ModuleType("spaces")

    def GPU(fn: object | None = None, **kwargs: object) -> object:
        duration = kwargs.get("duration")

        def decorator(func: object) -> object:
            recorded: Any = func
            recorded._zerogpu_duration = duration
            return func

        if callable(fn):
            return decorator(fn)
        return decorator

    stub.GPU = GPU  # type: ignore[attr-defined]
    return stub


def _ensure_spaces_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a no-op ``spaces`` module for this test only, if needed."""
    try:
        import spaces  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "spaces", _recording_spaces_module())


def _load_zerogpu_module(monkeypatch: pytest.MonkeyPatch):
    _isolate_telemetry_env(monkeypatch)
    _ensure_spaces_stub(monkeypatch)
    spec = importlib.util.spec_from_file_location(
        f"emotional_memory_zerogpu_demo_app_{id(monkeypatch)}",
        APP_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _force_offline_factories(monkeypatch: pytest.MonkeyPatch, demo_app: object) -> None:
    embedder = demo_app._HashEmbedder()  # type: ignore[attr-defined]
    appraisal = demo_app.KeywordAppraisalEngine()  # type: ignore[attr-defined]
    monkeypatch.setattr(demo_app, "_raw_embedder", lambda: embedder)
    monkeypatch.setattr(demo_app, "_make_embedder", lambda: embedder)
    monkeypatch.setattr(
        demo_app,
        "_make_appraisal_engine",
        lambda: (appraisal, "📝 Keyword fallback active"),
    )


def test_zerogpu_duration_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("gradio")
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")

    monkeypatch.setenv("EMOTIONAL_MEMORY_ZEROGPU_DURATION", "12")
    demo_app = _load_zerogpu_module(monkeypatch)
    assert demo_app._ZEROGPU_DURATION_SECONDS == 12
    assert demo_app._zerogpu_duration_seconds() == 12

    monkeypatch.setenv("EMOTIONAL_MEMORY_ZEROGPU_DURATION", "not-an-int")
    assert demo_app._zerogpu_duration_seconds() == demo_app._DEFAULT_ZEROGPU_DURATION_SECONDS


def test_zerogpu_app_imports_and_builds_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("gradio")
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")

    _isolate_telemetry_env(monkeypatch)
    demo_app = _load_zerogpu_module(monkeypatch)
    _force_offline_factories(monkeypatch, demo_app)

    demo = demo_app.build_demo()
    close = getattr(demo, "close", None)
    try:
        import gradio as gr

        assert isinstance(demo, gr.Blocks)
        assert callable(demo_app._embed_on_gpu)
        assert callable(demo_app._embed_batch_on_gpu)
    finally:
        if callable(close):
            close()


def test_zerogpu_gpu_scheduled_embedder_uses_decorated_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("gradio")
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")

    monkeypatch.delenv("EMOTIONAL_MEMORY_ZEROGPU_DURATION", raising=False)
    monkeypatch.setitem(sys.modules, "spaces", _recording_spaces_module())
    demo_app = _load_zerogpu_module(monkeypatch)

    class _FakeSentenceTransformerEmbedder:
        """Stand-in for the library embedder: no weights, no network."""

        def embed(self, text: str) -> list[float]:
            return [0.25, 0.75]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.25, 0.75] for _ in texts]

    fake = _FakeSentenceTransformerEmbedder()
    monkeypatch.setattr(demo_app, "_raw_embedder", lambda: fake)
    demo_app._make_embedder.cache_clear()

    adapter = demo_app._make_embedder()
    assert isinstance(adapter, demo_app._GpuScheduledEmbedder)

    seen: list[str] = []
    original_embed = demo_app._embed_on_gpu
    original_batch = demo_app._embed_batch_on_gpu

    def _embed_probe(text: str) -> list[float]:
        seen.append("embed")
        return original_embed(text)

    def _batch_probe(texts: list[str]) -> list[list[float]]:
        seen.append("batch")
        return original_batch(texts)

    monkeypatch.setattr(demo_app, "_embed_on_gpu", _embed_probe)
    monkeypatch.setattr(demo_app, "_embed_batch_on_gpu", _batch_probe)

    assert adapter.embed("hello") == [0.25, 0.75]
    assert adapter.embed_batch(["a", "b"]) == [[0.25, 0.75], [0.25, 0.75]]
    assert seen == ["embed", "batch"]

    assert demo_app._ZEROGPU_DURATION_SECONDS == 30
    assert demo_app._DEFAULT_ZEROGPU_DURATION_SECONDS == 30
    assert original_embed._zerogpu_duration == 30
    assert original_batch._zerogpu_duration == 30


def test_zerogpu_gpu_decorator_wraps_embed_functions() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    decorated = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "spaces"
                and decorator.func.attr == "GPU"
            )
            or (
                isinstance(decorator, ast.Attribute)
                and isinstance(decorator.value, ast.Name)
                and decorator.value.id == "spaces"
                and decorator.attr == "GPU"
            )
            for decorator in node.decorator_list
        )
    }
    assert decorated == {"_embed_on_gpu", "_embed_batch_on_gpu"}


def test_zerogpu_api_key_comment_describes_env_or_space_secret() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert "process environment or as a Hugging Face Space secret" in source
    assert "never hardcoded" in source
    assert "never logged" in source
    assert "sk-" not in source
    assert "hf_" not in source


def test_zerogpu_requirements_match_public_demo_plus_spaces() -> None:
    public = (ROOT / "demo" / "requirements.txt").read_text(encoding="utf-8")
    zerogpu = (ZEROGPU_DIR / "requirements.txt").read_text(encoding="utf-8")

    def _pins(text: str) -> set[str]:
        pins: set[str] = set()
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            pins.add(stripped.split()[0])
        return pins

    public_pins = _pins(public)
    zerogpu_pins = _pins(zerogpu)
    assert any(pin.startswith("brotli>=1.2.1") for pin in zerogpu_pins)
    assert "spaces" in zerogpu_pins
    assert "transformers==5.20.2" in zerogpu_pins
    assert public_pins <= zerogpu_pins
    assert zerogpu_pins - public_pins == {"spaces", "transformers==5.20.2"}


_NO_PUBLIC_SPACE = re.compile(
    r"huggingface\.co/spaces/|hf\.space",
    re.IGNORECASE,
)


def test_zerogpu_readme_is_source_folder_docs() -> None:
    readme = (ZEROGPU_DIR / "README.md").read_text(encoding="utf-8")
    readme_flat = re.sub(r"\s+", " ", readme.lower())
    assert "synced **verbatim**" in readme or "synced verbatim" in readme.lower()
    assert "is to be synced" in readme
    assert "merge SHA" in readme
    assert "uv run python demo/zerogpu/app.py" in readme
    assert "not yet public" in readme
    assert "CHANGED" not in readme
    assert "private draft" not in readme.lower()
    assert "pinned: false" in readme
    assert "suggested_hardware: zero-a10g" in readme
    assert "gianlucamazza/emotional-memory-benchmarks" in readme
    assert "latency" not in readme.lower()
    assert "MiniLM" in readme
    assert "@spaces.GPU" in readme
    assert "OpenAI-compatible" in readme
    assert "keyword fallback" in readme_flat
    assert "no in-process open-model appraisal" in readme_flat
    _assert_zerogpu_source_has_no_public_space()


def test_zerogpu_source_does_not_name_a_public_space() -> None:
    _assert_zerogpu_source_has_no_public_space()
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = changelog.split("## [", 2)[1]
    assert "demo/zerogpu/" in unreleased
    assert "a private ZeroGPU Space (not yet public)" in unreleased


def _assert_zerogpu_source_has_no_public_space() -> None:
    for name in ("README.md", "requirements.txt", "app.py"):
        text = (ZEROGPU_DIR / name).read_text(encoding="utf-8")
        assert _NO_PUBLIC_SPACE.search(text) is None, name
