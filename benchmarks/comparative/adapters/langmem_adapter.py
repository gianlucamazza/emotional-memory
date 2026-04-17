"""LangMem adapter — wraps langmem>=0.0.30 + langgraph InMemoryStore.

Requires OPENAI_API_KEY for the embedding index and pip install 'langmem>=0.0.30'.
"""

from __future__ import annotations

import contextlib
import os
import uuid

from .base import MemoryAdapter, RetrievedItem

_MISSING_KEY_REASON = "OPENAI_API_KEY not set (required by LangMem embedding index)"
_NOT_INSTALLED_REASON = (
    "langmem or langgraph not installed (pip install 'langmem>=0.0.30' 'langgraph>=0.3')"
)


class LangMemAdapter(MemoryAdapter):
    """LangMem adapter (requires: pip install 'langmem>=0.0.30' and OPENAI_API_KEY)."""

    name = "langmem"

    def __init__(self) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            self._available = False
            self._reason = _MISSING_KEY_REASON
            self._manage = None
            self._search = None
            return

        try:
            from langgraph.store.memory import InMemoryStore  # type: ignore[import-untyped]
            from langmem import (  # type: ignore[import-untyped]
                create_manage_memory_tool,
                create_search_memory_tool,
            )

            self._store = InMemoryStore(
                index={
                    "dims": 1536,
                    "embed": "openai:text-embedding-3-small",
                }
            )
            self._namespace = ("bench",)
            self._manage = create_manage_memory_tool(namespace=self._namespace, store=self._store)
            self._search = create_search_memory_tool(namespace=self._namespace, store=self._store)
            self._available = True
            self._reason = ""
        except ImportError:
            self._available = False
            self._reason = _NOT_INSTALLED_REASON
            self._manage = None
            self._search = None
        except Exception as exc:
            self._available = False
            self._reason = f"LangMem init failed: {exc}"
            self._manage = None
            self._search = None

    def encode(self, text: str, valence: float = 0.0, arousal: float = 0.5) -> str:
        if not self._available or self._manage is None:
            return str(uuid.uuid4())
        with contextlib.suppress(Exception):
            self._manage.invoke({"action": "create", "content": text})
        return str(uuid.uuid4())

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        valence: float = 0.0,
        arousal: float = 0.5,
    ) -> list[RetrievedItem]:
        if not self._available or self._search is None:
            return []
        try:
            hits = self._search.invoke({"query": query, "limit": top_k})
            if not isinstance(hits, list):
                hits = list(hits) if hits else []
            return [
                RetrievedItem(
                    id=str(getattr(h, "id", uuid.uuid4())),
                    text=str(getattr(h, "content", getattr(h, "page_content", str(h)))),
                    score=float(getattr(h, "score", 1.0 / (i + 1))),
                )
                for i, h in enumerate(hits[:top_k])
            ]
        except Exception:
            return []

    def reset(self) -> None:
        if not self._available:
            return
        with contextlib.suppress(Exception):
            from langgraph.store.memory import InMemoryStore  # type: ignore[import-untyped]
            from langmem import (  # type: ignore[import-untyped]
                create_manage_memory_tool,
                create_search_memory_tool,
            )

            self._store = InMemoryStore(
                index={
                    "dims": 1536,
                    "embed": "openai:text-embedding-3-small",
                }
            )
            self._manage = create_manage_memory_tool(namespace=self._namespace, store=self._store)
            self._search = create_search_memory_tool(namespace=self._namespace, store=self._store)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def not_available_reason(self) -> str:
        return self._reason
