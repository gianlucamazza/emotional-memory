"""Compatibility checks for the Gradio demo wiring."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class _FakeContext:
    def __enter__(self) -> _FakeContext:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeBlocks:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self) -> _FakeBlocks:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def load(self, **kwargs) -> None:
        self.load_kwargs = kwargs

    def launch(self, **kwargs) -> None:
        self.launch_kwargs = kwargs


class _FakeEventComponent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def click(self, **kwargs) -> None:
        self.click_kwargs = kwargs

    def submit(self, **kwargs) -> None:
        self.submit_kwargs = kwargs


def _make_fake_gradio(*, launch_accepts_theme: bool) -> ModuleType:
    class _FakeBlocksWithTheme(_FakeBlocks):
        def launch(self, *, show_error=False, theme=None) -> None:
            self.launch_kwargs = {"show_error": show_error, "theme": theme}

    class _FakeBlocksWithoutTheme(_FakeBlocks):
        def launch(self, *, show_error=False) -> None:
            self.launch_kwargs = {"show_error": show_error}

    gradio = ModuleType("gradio")
    gradio.Blocks = _FakeBlocksWithTheme if launch_accepts_theme else _FakeBlocksWithoutTheme
    gradio.Row = lambda *args, **kwargs: _FakeContext()
    gradio.Column = lambda *args, **kwargs: _FakeContext()
    gradio.State = lambda *args, **kwargs: SimpleNamespace(value=args[0] if args else None)
    gradio.Markdown = lambda *args, **kwargs: _FakeEventComponent(**kwargs)
    gradio.Chatbot = lambda *args, **kwargs: _FakeEventComponent(**kwargs)
    gradio.Textbox = lambda *args, **kwargs: _FakeEventComponent(**kwargs)
    gradio.Button = lambda *args, **kwargs: _FakeEventComponent(**kwargs)
    gradio.Examples = lambda *args, **kwargs: None
    gradio.Image = lambda *args, **kwargs: _FakeEventComponent(**kwargs)
    gradio.Info = lambda *args, **kwargs: None
    gradio.themes = SimpleNamespace(Soft=lambda: "soft-theme")
    return gradio


def _make_fake_matplotlib() -> tuple[ModuleType, ModuleType]:
    matplotlib = ModuleType("matplotlib")
    matplotlib.use = lambda *args, **kwargs: None
    pyplot = ModuleType("matplotlib.pyplot")
    pyplot.subplots = lambda *args, **kwargs: (SimpleNamespace(), [SimpleNamespace()] * 3)
    pyplot.tight_layout = lambda *args, **kwargs: None
    pyplot.close = lambda *args, **kwargs: None
    return matplotlib, pyplot


def _make_fake_pil() -> tuple[ModuleType, ModuleType]:
    pil = ModuleType("PIL")
    image_module = ModuleType("PIL.Image")
    image_module.Image = object
    image_module.open = lambda *args, **kwargs: object()
    pil.Image = image_module
    return pil, image_module


def _load_demo_module(*, launch_accepts_theme: bool):
    app_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    spec = importlib.util.spec_from_file_location(
        f"emotional_memory_demo_ui_{'new' if launch_accepts_theme else 'old'}", app_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    fake_gradio = _make_fake_gradio(launch_accepts_theme=launch_accepts_theme)
    fake_matplotlib, fake_pyplot = _make_fake_matplotlib()
    fake_pil, fake_image = _make_fake_pil()
    with patch.dict(
        sys.modules,
        {
            "gradio": fake_gradio,
            "matplotlib": fake_matplotlib,
            "matplotlib.pyplot": fake_pyplot,
            "PIL": fake_pil,
            "PIL.Image": fake_image,
        },
    ):
        spec.loader.exec_module(module)
    return module


def _keyword_map(call: ast.Call) -> dict[str, ast.expr]:
    return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}


def test_theme_goes_to_launch_when_runtime_supports_it() -> None:
    module = _load_demo_module(launch_accepts_theme=True)

    assert module._LAUNCH_SUPPORTS_THEME is True
    assert module._BLOCKS_KWARGS == {"title": "Emotional Memory Demo"}
    assert "theme" in inspect.signature(module.gr.Blocks.launch).parameters


def test_theme_stays_on_blocks_when_launch_does_not_support_it() -> None:
    module = _load_demo_module(launch_accepts_theme=False)

    assert module._LAUNCH_SUPPORTS_THEME is False
    assert module._BLOCKS_KWARGS["title"] == "Emotional Memory Demo"
    assert module._BLOCKS_KWARGS["theme"] == module._DEMO_THEME
    assert "theme" not in inspect.signature(module.gr.Blocks.launch).parameters


def test_chatbot_explicitly_disables_tags() -> None:
    app_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))

    chatbot_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Chatbot"
    ]

    assert chatbot_calls
    chatbot_keywords = _keyword_map(chatbot_calls[0])
    assert isinstance(chatbot_keywords.get("allow_tags"), ast.Constant)
    assert chatbot_keywords["allow_tags"].value is False
