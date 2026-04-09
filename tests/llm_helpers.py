"""Helpers for real-LLM tests.

Creates an LLMCallable from environment variables so integration tests and
quality benchmarks can run against any OpenAI-compatible endpoint.

Environment variables:
    EMOTIONAL_MEMORY_LLM_API_KEY   Required. API key for the provider.
    EMOTIONAL_MEMORY_LLM_BASE_URL  Optional. Default: https://api.openai.com/v1
    EMOTIONAL_MEMORY_LLM_MODEL     Optional. Default: gpt-4o-mini

Requires the ``llm-test`` optional dependency group:
    pip install -e ".[dev,llm-test]"
"""

from __future__ import annotations

import os
from typing import Any

from emotional_memory.appraisal_llm import LLMCallable

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o-mini"


def make_openai_compatible_llm() -> LLMCallable | None:
    """Return an LLMCallable backed by an OpenAI-compatible API, or None if unconfigured."""
    api_key = os.environ.get("EMOTIONAL_MEMORY_LLM_API_KEY")
    if not api_key:
        return None

    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "httpx is required for real-LLM tests. Install with: pip install -e '.[dev,llm-test]'"
        ) from exc

    base_url = os.environ.get("EMOTIONAL_MEMORY_LLM_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("EMOTIONAL_MEMORY_LLM_MODEL", _DEFAULT_MODEL)

    def _call(prompt: str, json_schema: dict[str, Any]) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    assert isinstance(_call, LLMCallable)
    return _call


def make_llm_or_skip() -> LLMCallable:
    """Return an LLMCallable or raise pytest.skip if EMOTIONAL_MEMORY_LLM_API_KEY is not set."""
    import pytest

    llm = make_openai_compatible_llm()
    if llm is None:
        pytest.skip("EMOTIONAL_MEMORY_LLM_API_KEY not set")
    return llm
