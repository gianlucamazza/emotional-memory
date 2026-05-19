"""Tests for EmotionalMemoryMem0Backend.

Uses a real EmotionalMemory with a deterministic fake embedder to avoid any
network calls.  The mem0ai package is NOT required to run these tests — the
adapter has no runtime mem0 dependency.
"""

from __future__ import annotations

from typing import Any

from emotional_memory import EmotionalMemory, InMemoryStore
from emotional_memory.integrations.mem0 import EmotionalMemoryMem0Backend, messages_to_content

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FixedEmbedder:
    """Returns the same 4-dim unit vector for every input."""

    def embed(self, text: str) -> list[float]:
        return [0.25, 0.25, 0.25, 0.25]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _make_em(**kwargs: Any) -> EmotionalMemory:
    return EmotionalMemory(store=InMemoryStore(), embedder=_FixedEmbedder(), **kwargs)


def _make_backend(**kwargs: Any) -> EmotionalMemoryMem0Backend:
    return EmotionalMemoryMem0Backend(_make_em(), **kwargs)


# ---------------------------------------------------------------------------
# messages_to_content
# ---------------------------------------------------------------------------


class TestMessagesToContent:
    def test_str_passthrough(self) -> None:
        assert messages_to_content("hello world") == "hello world"

    def test_empty_string(self) -> None:
        assert messages_to_content("") == ""

    def test_list_single_message_with_role(self) -> None:
        result = messages_to_content([{"role": "user", "content": "hello"}])
        assert result == "user: hello"

    def test_list_single_message_no_role(self) -> None:
        result = messages_to_content([{"content": "hello"}])
        assert result == "hello"

    def test_list_multiple_messages(self) -> None:
        msgs = [
            {"role": "user", "content": "I feel anxious today."},
            {"role": "assistant", "content": "Tell me more."},
        ]
        result = messages_to_content(msgs)
        assert "user: I feel anxious today." in result
        assert "assistant: Tell me more." in result

    def test_list_skips_empty_content(self) -> None:
        msgs = [{"role": "user", "content": ""}, {"role": "user", "content": "hi"}]
        result = messages_to_content(msgs)
        assert result == "user: hi"

    def test_list_skips_missing_content(self) -> None:
        msgs: list[dict[str, Any]] = [{"role": "user"}, {"content": "hi"}]
        result = messages_to_content(msgs)
        assert result == "hi"

    def test_empty_list(self) -> None:
        assert messages_to_content([]) == ""


# ---------------------------------------------------------------------------
# Import surface
# ---------------------------------------------------------------------------


class TestImportSurface:
    def test_importable_from_integrations_subpackage(self) -> None:
        from emotional_memory.integrations import (  # noqa: F401
            EmotionalMemoryMem0Backend,
            messages_to_content,
        )

    def test_importable_from_top_level(self) -> None:
        import emotional_memory as em

        assert "EmotionalMemoryMem0Backend" in em.__all__
        assert "messages_to_content" in em.__all__


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_string(self) -> None:
        backend = _make_backend()
        result = backend.add("I had a wonderful day.")
        assert result["results"][0]["event"] == "ADD"
        assert result["results"][0]["memory"] == "I had a wonderful day."
        assert isinstance(result["results"][0]["id"], str)

    def test_add_messages_list(self) -> None:
        backend = _make_backend()
        msgs = [{"role": "user", "content": "Feeling great today!"}]
        result = backend.add(msgs, user_id="alice")
        assert result["results"][0]["event"] == "ADD"
        assert "Feeling great today!" in result["results"][0]["memory"]

    def test_add_stores_user_id(self) -> None:
        backend = _make_backend()
        backend.add("Important memory.", user_id="alice")
        all_mems = backend._em.list_all()
        assert all_mems[0].metadata["user_id"] == "alice"

    def test_add_uses_default_user_id(self) -> None:
        backend = _make_backend(default_user_id="bob")
        backend.add("Default user memory.")
        all_mems = backend._em.list_all()
        assert all_mems[0].metadata["user_id"] == "bob"

    def test_add_stores_role(self) -> None:
        backend = _make_backend()
        backend.add([{"role": "user", "content": "Hello!"}])
        all_mems = backend._em.list_all()
        assert all_mems[0].metadata.get("role") == "user"

    def test_add_stores_extra_metadata(self) -> None:
        backend = _make_backend()
        backend.add("Test memory.", metadata={"source": "chat", "session": "1"})
        all_mems = backend._em.list_all()
        assert all_mems[0].metadata["source"] == "chat"
        assert all_mems[0].metadata["session"] == "1"

    def test_add_returns_single_result(self) -> None:
        backend = _make_backend()
        result = backend.add("Another memory.")
        assert len(result["results"]) == 1

    def test_add_increments_memory_count(self) -> None:
        backend = _make_backend()
        for i in range(3):
            backend.add(f"Memory {i}")
        assert len(backend._em.list_all()) == 3


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_returns_results_key(self) -> None:
        backend = _make_backend()
        backend.add("I felt joyful at the concert.", user_id="alice")
        result = backend.search("joy concert", user_id="alice")
        assert "results" in result

    def test_search_filters_by_user_id(self) -> None:
        backend = _make_backend()
        backend.add("Alice memory.", user_id="alice")
        backend.add("Bob memory.", user_id="bob")
        result = backend.search("memory", user_id="alice")
        for item in result["results"]:
            assert item["metadata"]["user_id"] == "alice"

    def test_search_result_shape(self) -> None:
        backend = _make_backend()
        backend.add("Happy thoughts.", user_id="u1")
        result = backend.search("happy", user_id="u1")
        if result["results"]:
            item = result["results"][0]
            assert "id" in item
            assert "memory" in item
            assert "score" in item
            assert "metadata" in item

    def test_search_respects_limit(self) -> None:
        backend = _make_backend()
        for i in range(10):
            backend.add(f"Memory {i}", user_id="u1")
        result = backend.search("memory", user_id="u1", limit=3)
        assert len(result["results"]) <= 3

    def test_search_empty_store(self) -> None:
        backend = _make_backend()
        result = backend.search("anything", user_id="u1")
        assert result == {"results": []}

    def test_search_no_matching_user_returns_empty(self) -> None:
        backend = _make_backend()
        backend.add("Only for alice.", user_id="alice")
        result = backend.search("alice", user_id="bob")
        assert result["results"] == []

    def test_search_uses_default_user_id(self) -> None:
        backend = _make_backend(default_user_id="dave")
        backend.add("Dave's memory.")
        result = backend.search("memory")
        assert len(result["results"]) >= 1


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_known_id(self) -> None:
        backend = _make_backend()
        add_result = backend.add("Specific memory.", user_id="u1")
        mid = add_result["results"][0]["id"]
        got = backend.get(mid)
        assert got is not None
        assert got["id"] == mid
        assert got["memory"] == "Specific memory."

    def test_get_unknown_id_returns_none(self) -> None:
        backend = _make_backend()
        assert backend.get("nonexistent-id") is None

    def test_get_result_shape(self) -> None:
        backend = _make_backend()
        add_result = backend.add("Shape test.")
        mid = add_result["results"][0]["id"]
        got = backend.get(mid)
        assert got is not None
        assert set(got.keys()) >= {"id", "memory", "metadata"}


# ---------------------------------------------------------------------------
# get_all()
# ---------------------------------------------------------------------------


class TestGetAll:
    def test_get_all_returns_results_key(self) -> None:
        backend = _make_backend()
        result = backend.get_all(user_id="u1")
        assert "results" in result

    def test_get_all_empty(self) -> None:
        backend = _make_backend()
        result = backend.get_all(user_id="u1")
        assert result["results"] == []

    def test_get_all_filters_by_user(self) -> None:
        backend = _make_backend()
        for i in range(3):
            backend.add(f"Alice {i}", user_id="alice")
        for i in range(2):
            backend.add(f"Bob {i}", user_id="bob")
        result = backend.get_all(user_id="alice")
        assert len(result["results"]) == 3
        for item in result["results"]:
            assert item["metadata"]["user_id"] == "alice"

    def test_get_all_respects_limit(self) -> None:
        backend = _make_backend()
        for i in range(10):
            backend.add(f"Item {i}", user_id="u1")
        result = backend.get_all(user_id="u1", limit=4)
        assert len(result["results"]) <= 4

    def test_get_all_uses_default_user_id(self) -> None:
        backend = _make_backend(default_user_id="eve")
        backend.add("Eve's memory.")
        result = backend.get_all()
        assert len(result["results"]) >= 1


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_removes_memory(self) -> None:
        backend = _make_backend()
        add_result = backend.add("To be deleted.", user_id="u1")
        mid = add_result["results"][0]["id"]
        backend.delete(mid)
        assert backend.get(mid) is None

    def test_delete_returns_success_message(self) -> None:
        backend = _make_backend()
        add_result = backend.add("Delete me.")
        mid = add_result["results"][0]["id"]
        result = backend.delete(mid)
        assert "message" in result
        assert "deleted" in result["message"].lower()

    def test_delete_reduces_count(self) -> None:
        backend = _make_backend()
        r1 = backend.add("One", user_id="u1")
        backend.add("Two", user_id="u1")
        backend.delete(r1["results"][0]["id"])
        all_mems = backend.get_all(user_id="u1")
        assert len(all_mems["results"]) == 1


# ---------------------------------------------------------------------------
# delete_all()
# ---------------------------------------------------------------------------


class TestDeleteAll:
    def test_delete_all_clears_user_memories(self) -> None:
        backend = _make_backend()
        for i in range(3):
            backend.add(f"Alice {i}", user_id="alice")
        backend.add("Bob stays.", user_id="bob")
        backend.delete_all(user_id="alice")
        assert backend.get_all(user_id="alice")["results"] == []
        assert len(backend.get_all(user_id="bob")["results"]) == 1

    def test_delete_all_returns_success_message(self) -> None:
        backend = _make_backend()
        backend.add("Temp.", user_id="u1")
        result = backend.delete_all(user_id="u1")
        assert "message" in result

    def test_delete_all_uses_default_user_id(self) -> None:
        backend = _make_backend(default_user_id="frank")
        backend.add("Frank's memory.")
        backend.delete_all()
        assert backend.get_all()["results"] == []


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all_memories(self) -> None:
        backend = _make_backend()
        backend.add("Alice mem.", user_id="alice")
        backend.add("Bob mem.", user_id="bob")
        backend.reset()
        assert backend._em.list_all() == []

    def test_reset_resets_affective_state(self) -> None:
        backend = _make_backend()
        backend.add("Positive memory.", user_id="u1")
        backend.reset()
        # After reset the mood should be near baseline (valence close to 0)
        state = backend._em.get_current_mood()
        assert abs(state.valence) < 0.5


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_does_not_raise(self) -> None:
        backend = _make_backend()
        backend.add("A memory.")
        backend.close()  # should not raise

    def test_context_manager_via_engine(self) -> None:
        em = _make_em()
        backend = EmotionalMemoryMem0Backend(em)
        with em:
            backend.add("Inside context.")
        # engine closed; further ops would raise, but we just verify no error here


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_class_name(self) -> None:
        backend = _make_backend()
        assert "EmotionalMemoryMem0Backend" in repr(backend)

    def test_repr_contains_default_user_id(self) -> None:
        backend = _make_backend(default_user_id="zara")
        assert "zara" in repr(backend)

    def test_repr_contains_memory_count(self) -> None:
        backend = _make_backend()
        backend.add("One.")
        backend.add("Two.")
        r = repr(backend)
        assert "memories=2" in r
