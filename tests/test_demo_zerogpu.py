"""CPU-only checks for the ZeroGPU Space source in ``demo/zerogpu/``."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ZEROGPU_DIR = ROOT / "demo" / "zerogpu"
APP_PATH = ZEROGPU_DIR / "app.py"


def _ensure_spaces_stub() -> None:
    """Install a no-op ``spaces`` module when the package is not installed."""
    try:
        import spaces  # noqa: F401
    except ImportError:
        stub = types.ModuleType("spaces")

        def GPU(fn: object | None = None, **_kwargs: object) -> object:
            if callable(fn):
                return fn

            def decorator(func: object) -> object:
                return func

            return decorator

        stub.GPU = GPU  # type: ignore[attr-defined]
        sys.modules["spaces"] = stub


def _load_zerogpu_module():
    _ensure_spaces_stub()
    spec = importlib.util.spec_from_file_location("emotional_memory_zerogpu_demo_app", APP_PATH)
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
    demo_app = _load_zerogpu_module()
    assert demo_app._ZEROGPU_DURATION_SECONDS == 12
    assert demo_app._zerogpu_duration_seconds() == 12

    monkeypatch.setenv("EMOTIONAL_MEMORY_ZEROGPU_DURATION", "not-an-int")
    assert demo_app._zerogpu_duration_seconds() == demo_app._DEFAULT_ZEROGPU_DURATION_SECONDS


def test_zerogpu_app_imports_and_builds_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("gradio")
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")

    demo_app = _load_zerogpu_module()
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


def test_zerogpu_gpu_decorator_wraps_embed_functions() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    decorated = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Name)
            and decorator.func.value.id == "spaces"
            and decorator.func.attr == "GPU"
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
    assert public_pins <= zerogpu_pins
    assert zerogpu_pins - public_pins == {"spaces"}


def test_zerogpu_readme_is_source_folder_docs() -> None:
    readme = (ZEROGPU_DIR / "README.md").read_text(encoding="utf-8")
    assert "synced **verbatim**" in readme or "synced verbatim" in readme.lower()
    assert "merge SHA" in readme
    assert "uv run python demo/zerogpu/app.py" in readme
    assert "CHANGED" not in readme
    assert "private draft" not in readme.lower()
    assert "suggested_hardware" not in readme
    assert "latency" not in readme.lower()
