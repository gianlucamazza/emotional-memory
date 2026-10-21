"""Regression tests for the Hugging Face / Gradio demo."""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("gradio")
pytest.importorskip("matplotlib")
pytest.importorskip("PIL")


@lru_cache(maxsize=1)
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


def _seed_chat_with_memory(demo_app):  # type: ignore[no-untyped-def]
    chatbot, em_state, pad_history, _plot, _mode, msg_count = demo_app.reset_session()
    chatbot, em_state, pad_history, _plot, _msg_box, msg_count = demo_app.chat(
        "I got promoted and everyone celebrated with me.",
        chatbot,
        em_state,
        pad_history,
        msg_count,
    )
    return chatbot, em_state, pad_history, msg_count


def test_retrieval_mode_dispatch_defaults_gated_and_bare_recall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_app = _load_demo_module()
    chatbot, em_state, pad_history, msg_count = _seed_chat_with_memory(demo_app)

    seen: list[dict[str, object]] = []
    gated_calls = 0

    def _fake_retrieve(self, query: str, top_k: int = 5, **kwargs):  # type: ignore[no-untyped-def]
        seen.append({"query": query, "top_k": top_k, **kwargs})
        return self.list_all()[:top_k]

    def _fake_gated(self, query: str, top_k: int = 5, *, tau=None):  # type: ignore[no-untyped-def]
        nonlocal gated_calls
        gated_calls += 1
        return self.list_all()[:top_k]

    monkeypatch.setattr(demo_app.EmotionalMemory, "retrieve", _fake_retrieve)
    monkeypatch.setattr(demo_app.EmotionalMemory, "retrieve_query_gated", _fake_gated)

    chatbot, em_state, pad_history, _plot, _msg_box, msg_count = demo_app.chat(
        "recall project success",
        chatbot,
        em_state,
        pad_history,
        msg_count,
    )
    assert gated_calls == 0
    assert seen == [{"query": "project success", "top_k": 3}]
    assert "Top memories matching your query and current mood" in chatbot[-1]["content"]

    seen.clear()
    chatbot, em_state, pad_history, _plot, _msg_box, msg_count = demo_app.chat(
        "recall project success",
        chatbot,
        em_state,
        pad_history,
        msg_count,
        demo_app.RETRIEVAL_MODE_QUERY_GATED,
    )
    assert gated_calls == 0
    assert len(seen) == 1
    assert seen[0]["query"] == "project success"
    assert seen[0]["top_k"] == 3
    assert "query_affect" in seen[0] or "precomputed_weights" in seen[0]
    reply = chatbot[-1]["content"]
    assert "query-gated" in reply.lower()
    assert "cosine only" in reply.lower() or "affect-aware" in reply.lower()

    seen.clear()
    chatbot, em_state, pad_history, _plot, _msg_box, _msg_count = demo_app.chat(
        "recall",
        chatbot,
        em_state,
        pad_history,
        msg_count,
        demo_app.RETRIEVAL_MODE_QUERY_GATED,
    )
    assert gated_calls == 0
    assert seen == [{"query": "recall", "top_k": 3}]
    assert "Top memories matching your current mood" in chatbot[-1]["content"]
    assert "mood-state" in chatbot[-1]["content"].lower()


def test_gated_path_matches_library_order_and_label() -> None:
    demo_app = _load_demo_module()
    embedder = demo_app._HashEmbedder()
    appraisal = demo_app.KeywordAppraisalEngine()
    texts = [
        "I succeeded! My project is complete and it's a great personal victory.",
        "I failed and made a terrible mistake — everything is broken.",
        "The meeting is at three o'clock in the conference room.",
    ]

    def _paired_engines() -> tuple[object, object]:
        em_demo = demo_app.EmotionalMemory(
            store=demo_app.InMemoryStore(),
            embedder=embedder,
            appraisal_engine=appraisal,
        )
        for text in texts:
            em_demo.encode(text)
        em_lib = demo_app.EmotionalMemory(
            store=demo_app.InMemoryStore(),
            embedder=embedder,
            appraisal_engine=appraisal,
        )
        em_lib.import_memories(em_demo.export_memories())
        em_lib.load_state(em_demo.save_state())
        return em_demo, em_lib

    cases = (
        ("the meeting time", "cosine only"),
        ("project success", "affect-aware"),
    )
    for query, expected_branch in cases:
        em_demo, em_lib = _paired_engines()
        results, branch, _keyword = demo_app._retrieve_query_gated(
            em_demo,
            query,
            top_k=3,
            appraisal_engine=appraisal,
            appraisal_mode="📝 Keyword fallback active",
        )
        library = em_lib.retrieve_query_gated(query, top_k=3)
        assert [memory.content for memory in results] == [memory.content for memory in library]
        assert expected_branch in branch.lower()


def test_gated_recall_notes_keyword_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    demo_app = _load_demo_module()
    chatbot, em_state, pad_history, msg_count = _seed_chat_with_memory(demo_app)

    monkeypatch.setattr(
        demo_app,
        "_make_appraisal_engine",
        lambda: (demo_app.KeywordAppraisalEngine(), "📝 Keyword fallback active"),
    )

    chatbot, _em_state, _pad_history, _plot, _msg_box, _msg_count = demo_app.chat(
        "recall project success",
        chatbot,
        em_state,
        pad_history,
        msg_count,
        demo_app.RETRIEVAL_MODE_QUERY_GATED,
    )
    assert "keyword" in chatbot[-1]["content"].lower()


def test_gated_fallback_hides_exception_text(monkeypatch: pytest.MonkeyPatch) -> None:
    demo_app = _load_demo_module()
    chatbot, em_state, pad_history, msg_count = _seed_chat_with_memory(demo_app)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("secret internals")

    monkeypatch.setattr(demo_app, "_retrieve_query_gated", _boom)

    chatbot, _em_state, _pad_history, _plot, _msg_box, _msg_count = demo_app.chat(
        "recall project success",
        chatbot,
        em_state,
        pad_history,
        msg_count,
        demo_app.RETRIEVAL_MODE_QUERY_GATED,
    )
    content = chatbot[-1]["content"]
    assert "query-gated recall unavailable, used mood-state recall" in content
    assert "secret internals" not in content


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


def test_demo_disables_gradio_and_hub_telemetry() -> None:
    _load_demo_module()

    assert os.environ.get("GRADIO_ANALYTICS_ENABLED") == "False"
    assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"
    assert os.environ.get("DO_NOT_TRACK") == "1"


class _CountingAppraisal:
    """Wraps KeywordAppraisalEngine and counts ``appraise`` calls."""

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.calls = 0
        self.texts: list[str] = []

    def appraise(self, event_text: str, context: dict[str, object] | None = None) -> object:
        self.calls += 1
        self.texts.append(event_text)
        return self.inner.appraise(event_text, context=context)


def _offline_session(demo_app, appraisal: object | None = None):  # type: ignore[no-untyped-def]
    embedder = demo_app._HashEmbedder()
    appraisal = appraisal or demo_app.KeywordAppraisalEngine()
    return demo_app.DemoSession(
        mood=demo_app.EmotionalMemory(
            store=demo_app.InMemoryStore(),
            embedder=embedder,
            appraisal_engine=appraisal,
        ),
        gated=demo_app.EmotionalMemory(
            store=demo_app.InMemoryStore(),
            embedder=embedder,
            appraisal_engine=appraisal,
        ),
        embedder=embedder,
        appraisal_engine=appraisal,
        appraisal_mode="📝 Keyword fallback active",
    )


def _force_offline_factories(
    monkeypatch: pytest.MonkeyPatch,
    demo_app,  # type: ignore[no-untyped-def]
    appraisal: object | None = None,
) -> object:
    embedder = demo_app._HashEmbedder()
    appraisal = appraisal or demo_app.KeywordAppraisalEngine()
    monkeypatch.setattr(demo_app, "_make_embedder", lambda: embedder)
    monkeypatch.setattr(
        demo_app,
        "_make_appraisal_engine",
        lambda: (appraisal, "📝 Keyword fallback active"),
    )
    return appraisal


def _snapshot(engine: object) -> tuple[list[object], object]:
    return engine.export_memories(), engine.save_state()


def test_recall_in_one_mode_leaves_the_other_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_app = _load_demo_module()
    appraisal = demo_app.KeywordAppraisalEngine()
    _force_offline_factories(monkeypatch, demo_app, appraisal)
    session = _offline_session(demo_app, appraisal)

    chatbot: list[dict[str, str]] = []
    pad_history = list(demo_app._INITIAL_PAD_HISTORY)
    msg_count = 0
    for text in (
        "I succeeded! My project is complete and it's a great personal victory.",
        "I failed and made a terrible mistake — everything is broken.",
        "The meeting is at three o'clock in the conference room.",
    ):
        chatbot, session, pad_history, _plot, _box, msg_count = demo_app.chat(
            text, chatbot, session, pad_history, msg_count
        )

    gated_before = _snapshot(session.gated)
    mood_before = _snapshot(session.mood)
    assert gated_before == mood_before

    chatbot, session, pad_history, _plot, _box, msg_count = demo_app.chat(
        "recall project success",
        chatbot,
        session,
        pad_history,
        msg_count,
        demo_app.RETRIEVAL_MODE_MOOD_STATE,
    )
    assert _snapshot(session.gated) == gated_before
    assert _snapshot(session.mood) != mood_before

    mood_after_own_recall = _snapshot(session.mood)
    chatbot, session, pad_history, _plot, _box, _msg_count = demo_app.chat(
        "recall project success",
        chatbot,
        session,
        pad_history,
        msg_count,
        demo_app.RETRIEVAL_MODE_QUERY_GATED,
    )
    assert _snapshot(session.mood) == mood_after_own_recall
    assert _snapshot(session.gated) != gated_before


def test_compare_both_leaves_both_sessions_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_app = _load_demo_module()
    appraisal = demo_app.KeywordAppraisalEngine()
    _force_offline_factories(monkeypatch, demo_app, appraisal)
    session = _offline_session(demo_app, appraisal)

    chatbot: list[dict[str, str]] = []
    pad_history = list(demo_app._INITIAL_PAD_HISTORY)
    msg_count = 0
    chatbot, session, pad_history, _plot, _box, msg_count = demo_app.chat(
        "I succeeded! My project is complete and it's a great personal victory.",
        chatbot,
        session,
        pad_history,
        msg_count,
    )
    chatbot, session, pad_history, _plot, _box, msg_count = demo_app.chat(
        "The meeting is at three o'clock in the conference room.",
        chatbot,
        session,
        pad_history,
        msg_count,
    )

    before_mood = _snapshot(session.mood)
    before_gated = _snapshot(session.gated)
    rendered = demo_app.compare_both_modes("recall project success", session)

    assert _snapshot(session.mood) == before_mood
    assert _snapshot(session.gated) == before_gated
    assert demo_app.RETRIEVAL_MODE_MOOD_STATE in rendered
    assert demo_app.RETRIEVAL_MODE_QUERY_GATED in rendered
    assert "affect-aware" in rendered
    assert "snapshot" in rendered.lower()


def test_non_recall_replicates_with_single_appraisal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_app = _load_demo_module()
    counting = _CountingAppraisal(demo_app.KeywordAppraisalEngine())
    _force_offline_factories(monkeypatch, demo_app, counting)
    session = _offline_session(demo_app, counting)

    demo_app._encode_user_both(
        session,
        "I feel relieved and proud after finishing this project.",
        {"role": "user"},
    )
    assert counting.calls == 1
    assert session.mood.export_memories() == session.gated.export_memories()
    assert session.mood.save_state() == session.gated.save_state()

    counting.calls = 0
    session = _offline_session(demo_app, counting)
    _chatbot, session, _pad, _plot, _box, _count = demo_app.chat(
        "I feel relieved and proud after finishing this project.",
        [],
        session,
        list(demo_app._INITIAL_PAD_HISTORY),
        0,
    )
    # One appraisal for the user encode, one for observe(reply); both modes share them.
    assert counting.calls == 2
    assert session.mood.export_memories() == session.gated.export_memories()
    assert session.mood.save_state() == session.gated.save_state()


def test_chat_guard_rebuilds_uninitialised_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_app = _load_demo_module()
    _force_offline_factories(monkeypatch, demo_app)

    chatbot, session, _pad_history, _plot, _box, msg_count = demo_app.chat(
        "recall happy moments",
        [],
        demo_app._initial_em_state,
        list(demo_app._INITIAL_PAD_HISTORY),
        demo_app._INITIAL_MSG_COUNT,
    )
    assert isinstance(session, demo_app.DemoSession)
    assert msg_count == 1
    assert chatbot[-1]["role"] == "assistant"

    chatbot, session, _pad_history, _plot, _box, msg_count = demo_app.chat(
        "I feel calm and ready to start.",
        [],
        demo_app._initial_em_state,
        list(demo_app._INITIAL_PAD_HISTORY),
        demo_app._INITIAL_MSG_COUNT,
    )
    assert isinstance(session, demo_app.DemoSession)
    stored = [memory.content for memory in session.mood.list_all()]
    assert stored == ["I feel calm and ready to start."]
    assert [memory.content for memory in session.gated.list_all()] == stored

    reset = demo_app.reset_session(demo_app._initial_em_state)
    assert len(reset) == 6
    assert isinstance(reset[1], demo_app.DemoSession)
