"""Helpers for real-LLM tests.

Creates an LLMCallable from environment variables so integration tests and
quality benchmarks can run against any OpenAI-compatible endpoint.

Environment variables:
    EMOTIONAL_MEMORY_LLM_API_KEY   Required. API key for the provider.
    EMOTIONAL_MEMORY_LLM_BASE_URL  Optional. Default: https://api.openai.com/v1
    EMOTIONAL_MEMORY_LLM_MODEL     Optional. Default: gpt-5-mini
    EMOTIONAL_MEMORY_LLM_OUTPUT_MODE Optional. Default: plain
    EMOTIONAL_MEMORY_LLM_TIMEOUT_SECONDS Optional. Default: 30

Requires the ``llm-test`` optional dependency group:
    pip install -e ".[dev,llm-test]"
"""

from __future__ import annotations

import os

from emotional_memory.appraisal_llm import LLMCallable
from emotional_memory.llm_http import (
    OpenAICompatibleLLMConfig,
    make_httpx_llm,
    probe_openai_compatible_llm,
)


def make_openai_compatible_llm() -> LLMCallable | None:
    """Return an LLMCallable backed by an OpenAI-compatible API, or None if unconfigured."""
    config = OpenAICompatibleLLMConfig.from_env(os.environ)
    if config is None:
        return None
    probe_openai_compatible_llm(config)
    llm = make_httpx_llm(config)
    assert isinstance(llm, LLMCallable)
    return llm


def make_llm_or_skip() -> LLMCallable:
    """Return an LLMCallable or raise pytest.skip if EMOTIONAL_MEMORY_LLM_API_KEY is not set."""
    import pytest

    llm = make_openai_compatible_llm()
    if llm is None:
        pytest.skip("EMOTIONAL_MEMORY_LLM_API_KEY not set")
    return llm
