"""Regression tests for the Hugging Face / Gradio demo."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("gradio")
pytest.importorskip("matplotlib")
pytest.importorskip("PIL")


def _load_demo_module():
    app_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    spec = importlib.util.spec_from_file_location("emotional_memory_demo_app", app_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recall_does_not_store_query_or_assistant_reply() -> None:
    demo_app = _load_demo_module()

    chatbot, em_state, pad_history, _plot, _mode, msg_count = demo_app.reset_session()

    chatbot, em_state, pad_history, _plot, _msg_box, msg_count = demo_app.chat(
        "I got promoted and everyone celebrated with me.",
        chatbot,
        em_state,
        pad_history,
        msg_count,
    )

    chatbot, em_state, pad_history, _plot, _msg_box, _msg_count = demo_app.chat(
        "recall happy moments",
        chatbot,
        em_state,
        pad_history,
        msg_count,
    )

    stored_contents = [memory.content for memory in em_state.list_all()]

    assert stored_contents == ["I got promoted and everyone celebrated with me."]
    assert "recall happy moments" not in stored_contents
    assert all(
        "Top memories matching your current mood" not in content for content in stored_contents
    )
    assert "I got promoted and everyone celebrated with me." in chatbot[-1]["content"]


def test_normal_chat_stores_only_user_messages() -> None:
    demo_app = _load_demo_module()

    _chatbot, em_state, pad_history, _plot, _mode, msg_count = demo_app.reset_session()
    baseline = em_state.get_state()

    _chatbot, em_state, pad_history, _plot, _msg_box, _msg_count = demo_app.chat(
        "I feel relieved and proud after finishing this project.",
        [],
        em_state,
        pad_history,
        msg_count,
    )

    stored_contents = [memory.content for memory in em_state.list_all()]

    assert stored_contents == ["I feel relieved and proud after finishing this project."]
    assert em_state.get_state() != baseline


def test_reset_session_returns_demo_runtime_state_without_langchain_adapter() -> None:
    demo_app = _load_demo_module()

    session = demo_app.reset_session()

    assert len(session) == 6


def test_launch_kwargs_disable_ssr_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMOTIONAL_MEMORY_DEMO_SSR", raising=False)
    monkeypatch.delenv("GRADIO_SSR_MODE", raising=False)

    demo_app = _load_demo_module()

    kwargs = demo_app._launch_kwargs()

    assert kwargs["show_error"] is True
    assert kwargs["theme"] is demo_app._DEMO_THEME
    assert kwargs["ssr_mode"] is False


def test_launch_kwargs_allow_explicit_ssr_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMOTIONAL_MEMORY_DEMO_SSR", "1")
    monkeypatch.delenv("GRADIO_SSR_MODE", raising=False)

    demo_app = _load_demo_module()

    assert demo_app._launch_kwargs()["ssr_mode"] is True


def test_launch_kwargs_ignore_gradio_ssr_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMOTIONAL_MEMORY_DEMO_SSR", raising=False)
    monkeypatch.setenv("GRADIO_SSR_MODE", "true")

    demo_app = _load_demo_module()

    assert demo_app._launch_kwargs()["ssr_mode"] is False


def test_event_loop_cleanup_patch_is_import_safe() -> None:
    demo_app = _load_demo_module()

    demo_app._patch_event_loop_cleanup()
