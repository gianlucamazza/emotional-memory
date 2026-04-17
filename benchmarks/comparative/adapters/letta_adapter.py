"""Letta adapter — wraps letta-client if installed and LETTA_API_KEY is set.

Letta is a cloud-only, agent-based memory service (pre-1.0). Each encode() call
creates an agent memory entry via the hosted API, which requires LLM overhead.
Without LETTA_API_KEY the adapter self-reports as not_evaluated.
"""

from __future__ import annotations

import contextlib
import os
import uuid

from .base import MemoryAdapter, RetrievedItem

_MISSING_KEY_REASON = "LETTA_API_KEY not set (cloud-only service — see https://letta.ai)"
_NOT_INSTALLED_REASON = "letta-client not installed (pip install letta-client)"


class LettaAdapter(MemoryAdapter):
    """Letta adapter (requires: pip install letta-client and LETTA_API_KEY)."""

    name = "letta"

    def __init__(self) -> None:
        self._api_key = os.environ.get("LETTA_API_KEY", "")
        if not self._api_key:
            self._available = False
            self._reason = _MISSING_KEY_REASON
            self._client = None
            self._agent_id: str | None = None
            return

        try:
            from letta_client import Letta  # type: ignore[import-untyped]

            self._client = Letta(token=self._api_key)
            self._agent_id = self._create_agent()
            self._available = True
            self._reason = ""
        except ImportError:
            self._available = False
            self._reason = _NOT_INSTALLED_REASON
            self._client = None
            self._agent_id = None
        except Exception as exc:
            self._available = False
            self._reason = f"Letta init failed: {exc}"
            self._client = None
            self._agent_id = None

    def _create_agent(self) -> str:
        assert self._client is not None
        agent = self._client.agents.create(
            name="benchmark_agent",
            memory_blocks=[
                {"label": "human", "value": "benchmark user"},
                {"label": "persona", "value": "benchmark assistant"},
            ],
            model="letta-free",
            embedding="letta-free",
        )
        return str(agent.id)

    def encode(self, text: str, valence: float = 0.0, arousal: float = 0.5) -> str:
        if not self._available or self._client is None or self._agent_id is None:
            return str(uuid.uuid4())
        with contextlib.suppress(Exception):
            self._client.agents.messages.create(
                agent_id=self._agent_id,
                messages=[{"role": "user", "content": f"[STORE] {text}"}],
            )
        return str(uuid.uuid4())

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        valence: float = 0.0,
        arousal: float = 0.5,
    ) -> list[RetrievedItem]:
        if not self._available or self._client is None or self._agent_id is None:
            return []
        try:
            response = self._client.agents.messages.create(
                agent_id=self._agent_id,
                messages=[{"role": "user", "content": f"[RECALL] {query}"}],
            )
            texts = [
                m.content
                for m in (response.messages or [])
                if hasattr(m, "role") and m.role == "assistant" and m.content
            ]
            return [
                RetrievedItem(id=str(uuid.uuid4()), text=t, score=1.0 / (i + 1))
                for i, t in enumerate(texts[:top_k])
            ]
        except Exception:
            return []

    def reset(self) -> None:
        if not self._available or self._client is None or self._agent_id is None:
            return
        with contextlib.suppress(Exception):
            self._client.agents.delete(agent_id=self._agent_id)
        with contextlib.suppress(Exception):
            self._agent_id = self._create_agent()

    @property
    def available(self) -> bool:
        return self._available

    @property
    def not_available_reason(self) -> str:
        return self._reason
