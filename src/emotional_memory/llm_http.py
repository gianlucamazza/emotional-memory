"""Shared OpenAI-compatible HTTP configuration and request helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from emotional_memory.appraisal_llm import LLMCallable

DEFAULT_OPENAI_COMPAT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_COMPAT_MODEL = "gpt-5-mini"
DEFAULT_OPENAI_COMPAT_TIMEOUT_SECONDS = 30.0
KNOWN_INVALID_PROJECT_MODELS = frozenset({"gpt-5.2-mini"})

LLMOutputMode: TypeAlias = Literal["plain", "json_object"]
DEFAULT_OPENAI_COMPAT_OUTPUT_MODE: LLMOutputMode = "plain"


def _parse_output_mode(raw_value: str) -> LLMOutputMode:
    normalized = raw_value.strip().lower()
    if normalized == "plain":
        return "plain"
    if normalized == "json_object":
        return "json_object"
    raise ValueError("EMOTIONAL_MEMORY_LLM_OUTPUT_MODE must be one of: plain, json_object")


@dataclass(frozen=True, slots=True)
class OpenAICompatibleLLMConfig:
    """Resolved configuration for an OpenAI-compatible HTTP endpoint."""

    api_key: str
    base_url: str = DEFAULT_OPENAI_COMPAT_BASE_URL
    model: str = DEFAULT_OPENAI_COMPAT_MODEL
    output_mode: LLMOutputMode = DEFAULT_OPENAI_COMPAT_OUTPUT_MODE
    timeout_seconds: float = DEFAULT_OPENAI_COMPAT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> OpenAICompatibleLLMConfig | None:
        data = env or os.environ
        api_key = data.get("EMOTIONAL_MEMORY_LLM_API_KEY", "").strip()
        if not api_key:
            return None

        base_url = data.get(
            "EMOTIONAL_MEMORY_LLM_BASE_URL",
            DEFAULT_OPENAI_COMPAT_BASE_URL,
        ).rstrip("/")
        model = data.get("EMOTIONAL_MEMORY_LLM_MODEL", DEFAULT_OPENAI_COMPAT_MODEL).strip()
        if not model:
            model = DEFAULT_OPENAI_COMPAT_MODEL

        output_mode = _parse_output_mode(
            data.get(
                "EMOTIONAL_MEMORY_LLM_OUTPUT_MODE",
                DEFAULT_OPENAI_COMPAT_OUTPUT_MODE,
            )
        )

        timeout_raw = data.get("EMOTIONAL_MEMORY_LLM_TIMEOUT_SECONDS")
        timeout_seconds = DEFAULT_OPENAI_COMPAT_TIMEOUT_SECONDS
        if timeout_raw:
            timeout_seconds = float(timeout_raw)
        if timeout_seconds <= 0:
            raise ValueError("EMOTIONAL_MEMORY_LLM_TIMEOUT_SECONDS must be > 0")

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            output_mode=output_mode,
            timeout_seconds=timeout_seconds,
        )

    def public_summary(self) -> dict[str, object]:
        """Safe diagnostic summary without leaking secrets."""
        return {
            "api_key_set": bool(self.api_key),
            "base_url": self.base_url,
            "model": self.model,
            "output_mode": self.output_mode,
            "timeout_seconds": self.timeout_seconds,
        }


def project_config_issues(config: OpenAICompatibleLLMConfig) -> list[str]:
    """Return project-level compatibility issues for a resolved config."""
    issues: list[str] = []
    if config.model in KNOWN_INVALID_PROJECT_MODELS:
        issues.append(
            f"model {config.model!r} is not a supported project default; "
            f"use {DEFAULT_OPENAI_COMPAT_MODEL!r} or another accessible provider model"
        )
    return issues


def build_openai_compatible_payload(
    prompt: str,
    json_schema: dict[str, Any],
    config: OpenAICompatibleLLMConfig,
) -> dict[str, Any]:
    """Build a provider-generic chat-completions payload."""
    del json_schema
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if config.output_mode == "json_object":
        payload["response_format"] = {"type": "json_object"}
    return payload


def make_httpx_llm(config: OpenAICompatibleLLMConfig) -> LLMCallable:
    """Build an LLMCallable using raw httpx."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "httpx is required for OpenAI-compatible HTTP LLM calls. "
            "Install with: pip install -e '.[dev,llm-test]'"
        ) from exc

    def _call(prompt: str, json_schema: dict[str, Any]) -> str:
        payload = build_openai_compatible_payload(prompt, json_schema, config)
        try:
            response = httpx.post(
                f"{config.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = getattr(exc.response, "text", "").strip().replace("\n", " ")
            detail = detail[:300]
            raise RuntimeError(
                "LLM request failed "
                f"(model={config.model!r}, base_url={config.base_url!r}, "
                f"status={exc.response.status_code}): {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "LLM transport failed "
                f"(model={config.model!r}, base_url={config.base_url!r}): {exc}"
            ) from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            preview = str(data)[:300]
            raise RuntimeError(
                "LLM response did not contain choices[0].message.content "
                f"(model={config.model!r}): {preview}"
            ) from exc
        if not isinstance(content, str):
            raise RuntimeError(
                "LLM response content was not a string "
                f"(model={config.model!r}, type={type(content).__name__})"
            )
        return content

    return _call


def make_httpx_llm_from_env(env: Mapping[str, str] | None = None) -> LLMCallable | None:
    """Build an HTTP-backed LLMCallable from environment variables."""
    config = OpenAICompatibleLLMConfig.from_env(env)
    if config is None:
        return None
    return make_httpx_llm(config)


def probe_openai_compatible_llm(config: OpenAICompatibleLLMConfig) -> None:
    """Fail fast on auth/model/request-shape problems without leaking secrets."""
    llm = make_httpx_llm(config)
    llm("Return only a JSON object: {}", {"type": "object", "properties": {}})
