"""Shared LLM client and base class for LoCoMo adapters."""

from __future__ import annotations

import logging
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any

from benchmarks.locomo.dataset import Conversation, QAPair, Session

# ---------------------------------------------------------------------------
# Thin LLM client (reuses env-var convention from the main library)
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_TIMEOUT = 120

# Transient HTTP status codes that warrant an automatic retry.
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 520, 522, 524, 529})
_MAX_RETRIES = 5
_BACKOFF_BASE = 1.0  # seconds; wait = base * 2^attempt + uniform(0, 0.5) jitter

logger = logging.getLogger(__name__)


def _get_llm_config() -> dict[str, str]:
    return {
        "api_key": os.environ.get("EMOTIONAL_MEMORY_LLM_API_KEY", ""),
        "base_url": os.environ.get("EMOTIONAL_MEMORY_LLM_BASE_URL", _DEFAULT_BASE_URL),
        "model": os.environ.get("EMOTIONAL_MEMORY_LLM_MODEL", _DEFAULT_MODEL),
        "reasoning_effort": os.environ.get("EMOTIONAL_MEMORY_LLM_REASONING_EFFORT", ""),
    }


def call_llm(prompt: str, *, system: str = "", temperature: float = 0.0) -> str:
    """Call an OpenAI-compatible LLM and return the text content.

    Retries on transient HTTP 429/5xx with exponential back-off.
    Raises ``RuntimeError`` if no API key is set.
    """
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required for LoCoMo LLM calls.\n"
            "Install with: pip install 'emotional-memory[llm-test]'"
        ) from exc

    cfg = _get_llm_config()
    if not cfg["api_key"]:
        raise RuntimeError(
            "EMOTIONAL_MEMORY_LLM_API_KEY is not set. "
            "Export it before running the LoCoMo benchmark."
        )

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}
    timeout = float(os.environ.get("EMOTIONAL_MEMORY_LLM_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT))

    payload: dict[str, object] = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": temperature,
    }
    if cfg["reasoning_effort"]:
        payload["reasoning_effort"] = cfg["reasoning_effort"]

    response = httpx.post(url, headers=headers, json=payload, timeout=timeout)

    # Some OpenAI reasoning models (e.g. gpt-5-mini) reject custom temperature
    # and/or unrecognized parameters. Strip the offending key and retry once.
    # Payload mutations are idempotent — later retry iterations reuse the fixed payload.
    while response.status_code == 400:
        body = response.json()
        err = body.get("error", {})
        bad_param = err.get("param")
        err_msg = err.get("message", "")
        removed = False
        if bad_param and bad_param in payload:
            payload.pop(bad_param, None)
            removed = True
        elif "temperature" in err_msg and "temperature" in payload:
            payload.pop("temperature", None)
            removed = True
        elif "reasoning_effort" in err_msg and "reasoning_effort" in payload:
            payload.pop("reasoning_effort", None)
            removed = True
        if not removed:
            break
        response = httpx.post(url, headers=headers, json=payload, timeout=timeout)

    # Retry on transient server errors and rate limits with exponential back-off.
    for _attempt in range(_MAX_RETRIES):
        if response.status_code not in _RETRYABLE_STATUS:
            break
        wait = _BACKOFF_BASE * (2**_attempt) + random.uniform(0.0, 0.5)
        logger.warning(
            "LLM API %s (attempt %d/%d) — retrying in %.1f s",
            response.status_code,
            _attempt + 1,
            _MAX_RETRIES,
            wait,
        )
        time.sleep(wait)
        response = httpx.post(url, headers=headers, json=payload, timeout=timeout)

    response.raise_for_status()
    return str(response.json()["choices"][0]["message"]["content"])


# ---------------------------------------------------------------------------
# Adapter contract
# ---------------------------------------------------------------------------

_ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question based only on the provided "
    "conversation excerpts. Be concise — one sentence or a short phrase is enough. "
    "If the information is not in the excerpts, say 'Not mentioned in the conversation'."
)


class LoCoMoAdapter(ABC):
    """Session-aware adapter for the LoCoMo benchmark."""

    name: str = "unnamed"

    @abstractmethod
    def reset(self) -> None:
        """Clear all state. Called once per conversation."""

    @abstractmethod
    def ingest_session(self, session: Session, conversation: Conversation) -> None:
        """Encode one session worth of turns into the adapter's memory."""

    @abstractmethod
    def answer(self, qa: QAPair, conversation: Conversation) -> str:
        """Retrieve relevant context and generate an answer string."""

    def run_conversation(self, conv: Conversation) -> list[dict[str, Any]]:
        """Convenience: ingest all sessions then answer all QA pairs.

        Returns a list of prediction dicts (one per QA pair).
        """
        self.reset()
        for session in conv.sessions:
            self.ingest_session(session, conv)

        results: list[dict[str, Any]] = []
        for qa in conv.qa_pairs:
            prediction = self.answer(qa, conv)
            results.append(
                {
                    "sample_id": conv.sample_id,
                    "question": qa.question,
                    "gold": qa.answer,
                    "prediction": prediction,
                    "category": qa.category,
                    "category_name": qa.category_name,
                    "is_adversarial": qa.is_adversarial,
                }
            )
        return results
