"""Mem0-compatible facade backed by EmotionalMemory.

Exposes the mem0 ``Memory`` API (``add`` / ``search`` / ``get`` / ``get_all`` /
``delete`` / ``delete_all`` / ``reset`` / ``close``) without importing mem0 at
runtime.  Use it as a drop-in replacement when you want affect-aware retrieval
instead of mem0's LLM-based fact extraction::

    from emotional_memory import EmotionalMemory, InMemoryStore
    from emotional_memory.integrations import EmotionalMemoryMem0Backend

    em = EmotionalMemory(store=InMemoryStore(), embedder=MyEmbedder())
    backend = EmotionalMemoryMem0Backend(em)

    backend.add([{"role": "user", "content": "I had a great day today."}])
    results = backend.search("positive experiences")
    print(results["results"][0]["memory"])

Install::

    uv pip install "emotional-memory[mem0]"

The ``[mem0]`` extra declares ``mem0ai>=2.0`` for users who want to chain the
real mem0 fact-extraction pipeline in front of this backend; the adapter itself
has no runtime mem0 dependency.

Trade-off note: this adapter exposes mem0's API surface but does *not* replicate
mem0's LLM-based fact extraction.  Memories are stored verbatim.  For fact-aware
extraction, chain a real ``mem0.Memory`` instance as a pre-processor and store
its extracted facts here.
"""

from __future__ import annotations

from typing import Any

from emotional_memory.engine import EmotionalMemory


def messages_to_content(messages: list[dict[str, Any]] | str) -> str:
    """Coerce a mem0-style messages list or a plain string to a single string.

    Parameters
    ----------
    messages:
        Either a ``str`` (returned as-is) or a list of dicts with optional
        ``"role"`` and ``"content"`` keys.  Dicts that lack a non-empty
        ``"content"`` value are skipped.
    """
    if isinstance(messages, str):
        return messages
    parts: list[str] = []
    for msg in messages:
        raw = msg.get("content", "")
        content = str(raw) if raw else ""
        if not content:
            continue
        raw_role = msg.get("role", "")
        role = str(raw_role) if raw_role else ""
        parts.append(f"{role}: {content}" if role else content)
    return "\n".join(parts)


class EmotionalMemoryMem0Backend:
    """mem0 ``Memory``-shaped facade backed by :class:`EmotionalMemory`.

    Parameters
    ----------
    em:
        A fully constructed :class:`EmotionalMemory` instance (store +
        embedder already wired).  Pass it from outside so callers retain
        control over the store backend and embedder choice.
    default_user_id:
        Fallback user ID when ``user_id`` is not supplied at call time.
    """

    __slots__ = ("_default_user_id", "_em")

    def __init__(
        self,
        em: EmotionalMemory,
        *,
        default_user_id: str = "default",
    ) -> None:
        self._em = em
        self._default_user_id = default_user_id

    # ------------------------------------------------------------------
    # mem0 API surface
    # ------------------------------------------------------------------

    def add(
        self,
        messages: list[dict[str, Any]] | str,
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Encode *messages* into emotional memory.

        Parameters
        ----------
        messages:
            A plain ``str`` or a list of ``{"role": ..., "content": ...}``
            dicts (mem0 convention).  Role information is prepended to the
            content string when present.
        user_id:
            Stored in ``memory.metadata["user_id"]`` so :meth:`search` and
            :meth:`get_all` can filter by user.
        metadata:
            Extra key/value pairs merged into ``memory.metadata``.
            ``user_id`` takes precedence over any conflicting key here.

        Returns
        -------
        dict
            ``{"results": [{"id": <memory_id>, "memory": <content>, "event": "ADD"}]}``
        """
        uid = user_id if user_id is not None else self._default_user_id
        content = messages_to_content(messages)

        combined: dict[str, Any] = {}
        if metadata:
            combined.update(metadata)
        combined["user_id"] = uid

        if isinstance(messages, list) and messages:
            raw_role = messages[0].get("role", "")
            if raw_role:
                combined.setdefault("role", str(raw_role))

        memory = self._em.encode(content, metadata=combined)
        return {"results": [{"id": memory.id, "memory": memory.content, "event": "ADD"}]}

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieve memories relevant to *query*.

        Parameters
        ----------
        query:
            Free-text search string passed through the full AFT retrieval
            pipeline (semantic similarity + mood congruence + decay + resonance).
        user_id:
            If provided, only memories whose stored metadata contains
            ``"user_id": user_id`` are returned.
        limit:
            Maximum number of results.
        filters:
            Ignored — present for API surface compatibility only.

        Returns
        -------
        dict
            ``{"results": [{"id": ..., "memory": ..., "score": ..., "metadata": ...}]}``
        """
        uid = user_id if user_id is not None else self._default_user_id
        # Over-retrieve to allow post-filtering by user_id without extra passes.
        fetch_k = max(limit * 4, 40)
        explanations = self._em.retrieve_with_explanations(query, top_k=fetch_k)
        results: list[dict[str, Any]] = []
        for exp in explanations:
            mem = exp.memory
            if mem.metadata.get("user_id") != uid:
                continue
            results.append(
                {
                    "id": mem.id,
                    "memory": mem.content,
                    "score": exp.score,
                    "metadata": dict(mem.metadata),
                }
            )
            if len(results) >= limit:
                break
        return {"results": results}

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Return a single memory by ID, or ``None`` if not found.

        Returns
        -------
        dict or None
            ``{"id": ..., "memory": ..., "metadata": ...}`` or ``None``.
        """
        for mem in self._em.list_all():
            if mem.id == memory_id:
                return {"id": mem.id, "memory": mem.content, "metadata": dict(mem.metadata)}
        return None

    def get_all(
        self,
        *,
        user_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Return all stored memories for *user_id*.

        Parameters
        ----------
        user_id:
            If provided, only memories whose metadata matches are returned.
        limit:
            Maximum items to return.

        Returns
        -------
        dict
            ``{"results": [{"id": ..., "memory": ..., "metadata": ...}]}``
        """
        uid = user_id if user_id is not None else self._default_user_id
        results: list[dict[str, Any]] = []
        for mem in self._em.list_all():
            if mem.metadata.get("user_id") != uid:
                continue
            results.append({"id": mem.id, "memory": mem.content, "metadata": dict(mem.metadata)})
            if len(results) >= limit:
                break
        return {"results": results}

    def delete(self, memory_id: str) -> dict[str, Any]:
        """Delete a single memory by ID.

        Returns
        -------
        dict
            ``{"message": "Memory deleted successfully!"}``
        """
        self._em.delete(memory_id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, *, user_id: str | None = None) -> dict[str, Any]:
        """Delete all memories belonging to *user_id*.

        Returns
        -------
        dict
            ``{"message": "Memory deleted successfully!"}``
        """
        uid = user_id if user_id is not None else self._default_user_id
        to_delete = [mem.id for mem in self._em.list_all() if mem.metadata.get("user_id") == uid]
        for mid in to_delete:
            self._em.delete(mid)
        return {"message": "Memory deleted successfully!"}

    def reset(self) -> None:
        """Delete all memories across all users and reset affective state."""
        all_ids = [mem.id for mem in self._em.list_all()]
        for mid in all_ids:
            self._em.delete(mid)
        self._em.reset_state()

    def close(self) -> None:
        """Release underlying engine resources."""
        self._em.close()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"memories={len(self._em.list_all())}, "
            f"default_user_id={self._default_user_id!r})"
        )
