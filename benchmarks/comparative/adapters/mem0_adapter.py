"""Mem0 adapter — wraps mem0ai>=2.0 if installed, otherwise marks as not_evaluated."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import uuid

from .base import MemoryAdapter, RetrievedItem

_MISSING_KEY_REASON = "OPENAI_API_KEY not set (required by mem0ai)"
_NOT_INSTALLED_REASON = "mem0ai not installed (pip install 'mem0ai>=2.0')"


class Mem0Adapter(MemoryAdapter):
    """mem0 adapter (requires: pip install 'mem0ai>=2.0' and OPENAI_API_KEY)."""

    name = "mem0"

    def __init__(self) -> None:
        self._qdrant_dir: str | None = None

        if not os.environ.get("OPENAI_API_KEY"):
            self._available = False
            self._reason = _MISSING_KEY_REASON
            self._mem = None
            return

        try:
            from mem0 import Memory  # type: ignore[import-untyped]

            self._qdrant_dir = tempfile.mkdtemp(prefix="qdrant_bench_")
            self._mem = Memory.from_config(
                {
                    "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
                    "embedder": {"provider": "openai"},
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "path": self._qdrant_dir,
                            "collection_name": "bench",
                        },
                    },
                }
            )
            self._available = True
            self._reason = ""
            self._user_id = "benchmark"
        except ImportError:
            self._available = False
            self._reason = _NOT_INSTALLED_REASON
            self._mem = None
        except Exception as exc:
            self._available = False
            self._reason = f"mem0 init failed: {exc}"
            self._mem = None

    def encode(self, text: str, valence: float = 0.0, arousal: float = 0.5) -> str:
        if not self._available or self._mem is None:
            return str(uuid.uuid4())
        with contextlib.suppress(Exception):
            result = self._mem.add(
                [{"role": "user", "content": text}],
                user_id=self._user_id,
            )
            results = result.get("results", []) if isinstance(result, dict) else []
            if results and isinstance(results[0], dict):
                return str(results[0].get("id", uuid.uuid4()))
        return str(uuid.uuid4())

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        valence: float = 0.0,
        arousal: float = 0.5,
    ) -> list[RetrievedItem]:
        if not self._available or self._mem is None:
            return []
        try:
            result = self._mem.search(
                query,
                filters={"user_id": self._user_id},
                limit=top_k,
            )
            items = result.get("results", []) if isinstance(result, dict) else result
            return [
                RetrievedItem(
                    id=str(r.get("id", i)),
                    text=r.get("memory", ""),
                    score=float(r.get("score", 0.0)),
                )
                for i, r in enumerate(items)
            ]
        except Exception:
            return []

    def reset(self) -> None:
        if self._available and self._mem is not None:
            with contextlib.suppress(Exception):
                self._mem.delete_all(user_id=self._user_id)
        # Clean up and recreate the qdrant temp directory between runs
        if self._qdrant_dir:
            with contextlib.suppress(Exception):
                shutil.rmtree(self._qdrant_dir)
            with contextlib.suppress(Exception):
                self._qdrant_dir = tempfile.mkdtemp(prefix="qdrant_bench_")

    @property
    def available(self) -> bool:
        return self._available

    @property
    def not_available_reason(self) -> str:
        return self._reason
