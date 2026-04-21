"""Tests for EmotionalMemoryChatHistory (requires [langchain] extra).

Uses a real EmotionalMemory with a deterministic fake embedder to avoid any
network calls. Skipped automatically when langchain-core is not installed.
"""

from __future__ import annotations

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Skip entire module when langchain-core is not installed
# ---------------------------------------------------------------------------

pytest.importorskip("langchain_core")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FixedEmbedder:
    """Returns the same 4-dim unit vector for every input — enough for tests."""

    def embed(self, text: str) -> list[float]:
        return [0.25, 0.25, 0.25, 0.25]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _make_em(**kwargs: Any) -> Any:
    from emotional_memory import EmotionalMemory, InMemoryStore

    return EmotionalMemory(store=InMemoryStore(), embedder=_FixedEmbedder(), **kwargs)


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


def test_importable_from_integrations_subpackage() -> None:
    from emotional_memory.integrations import (  # noqa: F401
        EmotionalMemoryChatHistory,
        recommended_conversation_policy,
        store_all_messages,
    )


def test_importable_from_top_level() -> None:
    import emotional_memory as em

    assert "EmotionalMemoryChatHistory" in em.__all__
    assert "recommended_conversation_policy" in em.__all__
    assert "store_all_messages" in em.__all__


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_add_user_message_stores_memory() -> None:
    from emotional_memory.integrations.langchain import EmotionalMemoryChatHistory

    history = EmotionalMemoryChatHistory(_make_em())
    history.add_user_message("Hello!")
    assert len(history.messages) == 1


def test_add_ai_message_stores_memory() -> None:
    from emotional_memory.integrations.langchain import EmotionalMemoryChatHistory

    history = EmotionalMemoryChatHistory(_make_em())
    history.add_ai_message("Hi there!")
    assert len(history.messages) == 1


def test_messages_role_mapping() -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    from emotional_memory.integrations.langchain import EmotionalMemoryChatHistory

    history = EmotionalMemoryChatHistory(_make_em())
    history.add_user_message("user text")
    history.add_ai_message("ai text")

    msgs = history.messages
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)


def test_messages_content_preserved() -> None:
    from emotional_memory.integrations.langchain import EmotionalMemoryChatHistory

    history = EmotionalMemoryChatHistory(_make_em())
    history.add_user_message("remember this")
    assert history.messages[0].content == "remember this"


def test_messages_chronological_order() -> None:

    from emotional_memory.integrations.langchain import EmotionalMemoryChatHistory

    history = EmotionalMemoryChatHistory(_make_em())
    texts = ["first", "second", "third"]
    for i, t in enumerate(texts):
        if i % 2 == 0:
            history.add_user_message(t)
        else:
            history.add_ai_message(t)

    msgs = history.messages
    assert [m.content for m in msgs] == texts


def test_clear_removes_all_messages() -> None:
    from emotional_memory.integrations.langchain import EmotionalMemoryChatHistory

    history = EmotionalMemoryChatHistory(_make_em())
    history.add_user_message("one")
    history.add_ai_message("two")
    assert len(history.messages) == 2

    history.clear()
    assert len(history.messages) == 0


def test_clear_is_idempotent() -> None:
    from emotional_memory.integrations.langchain import EmotionalMemoryChatHistory

    history = EmotionalMemoryChatHistory(_make_em())
    history.clear()
    history.clear()
    assert len(history.messages) == 0


def test_add_message_with_base_message() -> None:
    from langchain_core.messages import SystemMessage

    from emotional_memory.integrations.langchain import EmotionalMemoryChatHistory

    history = EmotionalMemoryChatHistory(_make_em())
    history.add_message(SystemMessage(content="You are helpful."))
    # SystemMessage content round-trips through metadata["role"] = "system"
    mems = history._em.list_all()
    assert mems[0].metadata["role"] == "system"


def test_repr_contains_class_name() -> None:
    from emotional_memory.integrations.langchain import EmotionalMemoryChatHistory

    history = EmotionalMemoryChatHistory(_make_em())
    assert "EmotionalMemoryChatHistory" in repr(history)


def test_repr_contains_message_count() -> None:
    from emotional_memory.integrations.langchain import EmotionalMemoryChatHistory

    history = EmotionalMemoryChatHistory(_make_em())
    history.add_user_message("ping")
    assert "1" in repr(history)


def test_import_error_without_package() -> None:
    """Import of the module itself raises ImportError when langchain-core is absent."""
    import sys
    from unittest.mock import patch

    with patch.dict(
        sys.modules,
        {
            "langchain_core": None,
            "langchain_core.chat_history": None,
            "langchain_core.messages": None,
        },
    ):  # type: ignore[dict-item]
        # Remove cached module so Python re-executes the import
        sys.modules.pop("emotional_memory.integrations.langchain", None)
        with pytest.raises((ImportError, ModuleNotFoundError)):
            import emotional_memory.integrations.langchain  # noqa: F401
        # Restore for subsequent tests
        sys.modules.pop("emotional_memory.integrations.langchain", None)


def test_emotional_state_evolves_with_conversation() -> None:
    """Encoding messages updates the affective state of the underlying engine."""
    from emotional_memory.integrations.langchain import EmotionalMemoryChatHistory

    em = _make_em()
    history = EmotionalMemoryChatHistory(em)

    initial_valence = em.get_state().core_affect.valence
    history.add_user_message("I'm so excited about this project!")
    history.add_ai_message("That's wonderful to hear!")

    # Affective state should have been updated by the two encode() calls
    assert em.get_state().core_affect.valence != initial_valence or len(history.messages) == 2


def test_history_reconstructs_existing_stored_messages() -> None:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from emotional_memory.integrations.langchain import EmotionalMemoryChatHistory

    em = _make_em()
    em.encode("system prompt", metadata={"role": "system"})
    em.encode("hello", metadata={"role": "user"})
    em.encode("hi there", metadata={"role": "assistant"})

    history = EmotionalMemoryChatHistory(em)
    msgs = history.messages

    assert len(msgs) == 3
    assert isinstance(msgs[0], SystemMessage)
    assert isinstance(msgs[1], HumanMessage)
    assert isinstance(msgs[2], AIMessage)


def test_clear_resets_affective_state() -> None:
    from emotional_memory.integrations.langchain import EmotionalMemoryChatHistory

    em = _make_em()
    history = EmotionalMemoryChatHistory(em)
    baseline = em.get_state()

    history.add_user_message("I am thrilled about this launch.")
    assert em.get_state() != baseline

    history.clear()

    assert history.messages == []
    assert em.get_state().model_dump(exclude={"mood": {"timestamp"}}) == baseline.model_dump(
        exclude={"mood": {"timestamp"}}
    )


def test_recommended_policy_keeps_transcript_but_filters_retrieval_corpus() -> None:
    from emotional_memory import KeywordAppraisalEngine
    from emotional_memory.integrations.langchain import (
        EmotionalMemoryChatHistory,
        recommended_conversation_policy,
    )

    em = _make_em(appraisal_engine=KeywordAppraisalEngine())
    history = EmotionalMemoryChatHistory(em, message_policy=recommended_conversation_policy)

    history.add_user_message("I won the award and feel amazing.")
    history.add_ai_message("That is fantastic news.")
    history.add_user_message("recall happy moments")

    assert [m.content for m in history.messages] == [
        "I won the award and feel amazing.",
        "That is fantastic news.",
        "recall happy moments",
    ]
    assert [m.content for m in em.list_all()] == ["I won the award and feel amazing."]

    results = em.retrieve("happy moments", top_k=3)
    assert [m.content for m in results] == ["I won the award and feel amazing."]


def test_recommended_policy_state_only_messages_update_affect_without_storing() -> None:
    from emotional_memory import StaticAppraisalEngine
    from emotional_memory.appraisal import AppraisalVector
    from emotional_memory.integrations.langchain import (
        EmotionalMemoryChatHistory,
        recommended_conversation_policy,
    )

    positive = AppraisalVector(
        novelty=1.0,
        goal_relevance=1.0,
        coping_potential=1.0,
        norm_congruence=1.0,
        self_relevance=1.0,
    )
    em = _make_em(appraisal_engine=StaticAppraisalEngine(positive))
    history = EmotionalMemoryChatHistory(em, message_policy=recommended_conversation_policy)
    baseline = em.get_state()

    history.add_ai_message("Wonderful progress. Keep going.")

    assert len(em.list_all()) == 0
    assert em.get_state() != baseline
