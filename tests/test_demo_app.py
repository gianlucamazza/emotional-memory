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
        "Top memories matching your query and current mood" not in content
        and "Top memories matching your current mood" not in content
        for content in stored_contents
    )
    assert "I got promoted and everyone celebrated with me." in chatbot[-1]["content"]


def test_parse_recall_query_extracts_payload_and_handles_empty_command() -> None:
    demo_app = _load_demo_module()

    assert demo_app._parse_recall_query("recall project success") == "project success"
    assert demo_app._parse_recall_query(" remember surprising news ") == "surprising news"
    assert demo_app._parse_recall_query("recall") == ""
    assert demo_app._parse_recall_query("tell me more") is None


def test_recall_uses_only_payload_for_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    demo_app = _load_demo_module()

    chatbot, em_state, pad_history, _plot, _mode, msg_count = demo_app.reset_session()
    chatbot, em_state, pad_history, _plot, _msg_box, msg_count = demo_app.chat(
        "I got promoted and everyone celebrated with me.",
        chatbot,
        em_state,
        pad_history,
        msg_count,
    )

    seen: dict[str, str | int] = {}

    def _fake_retrieve(self, query: str, top_k: int = 5):  # type: ignore[no-untyped-def]
        seen["query"] = query
        seen["top_k"] = top_k
        return self.list_all()[:top_k]

    monkeypatch.setattr(demo_app.EmotionalMemory, "retrieve", _fake_retrieve)

    chatbot, em_state, pad_history, _plot, _msg_box, _msg_count = demo_app.chat(
        "recall project success",
        chatbot,
        em_state,
        pad_history,
        msg_count,
    )

    assert seen == {"query": "project success", "top_k": 3}
    assert "Top memories matching your query and current mood" in chatbot[-1]["content"]


def test_bare_recall_keeps_generic_mood_congruent_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_app = _load_demo_module()

    chatbot, em_state, pad_history, _plot, _mode, msg_count = demo_app.reset_session()
    chatbot, em_state, pad_history, _plot, _msg_box, msg_count = demo_app.chat(
        "I got promoted and everyone celebrated with me.",
        chatbot,
        em_state,
        pad_history,
        msg_count,
    )

    seen: dict[str, str | int] = {}

    def _fake_retrieve(self, query: str, top_k: int = 5):  # type: ignore[no-untyped-def]
        seen["query"] = query
        seen["top_k"] = top_k
        return self.list_all()[:top_k]

    monkeypatch.setattr(demo_app.EmotionalMemory, "retrieve", _fake_retrieve)

    chatbot, em_state, pad_history, _plot, _msg_box, _msg_count = demo_app.chat(
        "recall",
        chatbot,
        em_state,
        pad_history,
        msg_count,
    )

    stored_contents = [memory.content for memory in em_state.list_all()]

    assert seen == {"query": "recall", "top_k": 3}
    assert stored_contents == ["I got promoted and everyone celebrated with me."]
    assert "Top memories matching your current mood" in chatbot[-1]["content"]


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


def test_initial_em_state_factory_returns_runtime() -> None:
    demo_app = _load_demo_module()

    em_state = demo_app._initial_em_state()

    assert em_state is not None
    assert em_state.get_state() is not None


def test_runtime_badge_mentions_retrieval_mode() -> None:
    demo_app = _load_demo_module()

    badge = demo_app._runtime_mode_badge()

    assert "retrieval active" in badge
    assert "LLM appraisal active" in badge or "Keyword fallback active" in badge


def test_chat_accepts_seeded_initial_state() -> None:
    demo_app = _load_demo_module()

    chatbot, _em_state, pad_history, _plot, _mode, msg_count = demo_app.reset_session()

    chatbot, _em_state, pad_history, _plot, _msg_box, msg_count = demo_app.chat(
        "I feel calm and ready to start.",
        demo_app._INITIAL_CHAT_HISTORY.copy(),
        demo_app._initial_em_state(),
        list(demo_app._INITIAL_PAD_HISTORY),
        demo_app._INITIAL_MSG_COUNT,
    )

    assert msg_count == 1
    assert len(chatbot) == 2
    assert len(pad_history) >= 1


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
