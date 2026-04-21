"""Tests for shared OpenAI-compatible HTTP LLM helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from emotional_memory.llm_http import (
    DEFAULT_OPENAI_COMPAT_BASE_URL,
    DEFAULT_OPENAI_COMPAT_MODEL,
    DEFAULT_OPENAI_COMPAT_OUTPUT_MODE,
    DEFAULT_OPENAI_COMPAT_TIMEOUT_SECONDS,
    KNOWN_INVALID_PROJECT_MODELS,
    OpenAICompatibleLLMConfig,
    build_openai_compatible_payload,
    make_httpx_llm,
    probe_openai_compatible_llm,
    project_config_issues,
)


def test_config_from_env_uses_defaults() -> None:
    config = OpenAICompatibleLLMConfig.from_env({"EMOTIONAL_MEMORY_LLM_API_KEY": "secret"})

    assert config is not None
    assert config.base_url == DEFAULT_OPENAI_COMPAT_BASE_URL
    assert config.model == DEFAULT_OPENAI_COMPAT_MODEL
    assert config.output_mode == DEFAULT_OPENAI_COMPAT_OUTPUT_MODE
    assert config.timeout_seconds == DEFAULT_OPENAI_COMPAT_TIMEOUT_SECONDS


def test_config_from_env_reads_all_overrides() -> None:
    config = OpenAICompatibleLLMConfig.from_env(
        {
            "EMOTIONAL_MEMORY_LLM_API_KEY": "secret",
            "EMOTIONAL_MEMORY_LLM_BASE_URL": "https://example.invalid/v1/",
            "EMOTIONAL_MEMORY_LLM_MODEL": "custom-model",
            "EMOTIONAL_MEMORY_LLM_OUTPUT_MODE": "json_object",
            "EMOTIONAL_MEMORY_LLM_TIMEOUT_SECONDS": "12.5",
        }
    )

    assert config is not None
    assert config.base_url == "https://example.invalid/v1"
    assert config.model == "custom-model"
    assert config.output_mode == "json_object"
    assert config.timeout_seconds == 12.5


def test_invalid_output_mode_raises() -> None:
    with pytest.raises(ValueError, match="OUTPUT_MODE"):
        OpenAICompatibleLLMConfig.from_env(
            {
                "EMOTIONAL_MEMORY_LLM_API_KEY": "secret",
                "EMOTIONAL_MEMORY_LLM_OUTPUT_MODE": "schema",
            }
        )


def test_project_config_issues_flags_known_invalid_model() -> None:
    config = OpenAICompatibleLLMConfig(
        api_key="secret",
        model=next(iter(KNOWN_INVALID_PROJECT_MODELS)),
    )

    issues = project_config_issues(config)

    assert issues
    assert config.model in issues[0]


def test_project_config_issues_accepts_default_model() -> None:
    config = OpenAICompatibleLLMConfig(api_key="secret")

    assert project_config_issues(config) == []


def test_public_summary_does_not_leak_secret() -> None:
    config = OpenAICompatibleLLMConfig(api_key="super-secret")

    summary = config.public_summary()

    assert summary["api_key_set"] is True
    assert "super-secret" not in str(summary)


def test_build_payload_plain_omits_response_format() -> None:
    config = OpenAICompatibleLLMConfig(api_key="secret", output_mode="plain")

    payload = build_openai_compatible_payload("prompt", {}, config)

    assert payload["model"] == config.model
    assert "response_format" not in payload


def test_build_payload_json_object_sets_response_format() -> None:
    config = OpenAICompatibleLLMConfig(api_key="secret", output_mode="json_object")

    payload = build_openai_compatible_payload("prompt", {}, config)

    assert payload["response_format"] == {"type": "json_object"}


def test_make_httpx_llm_uses_timeout_and_output_mode() -> None:
    seen: dict[str, object] = {}

    class FakeHTTPError(Exception):
        pass

    class FakeHTTPStatusError(FakeHTTPError):
        def __init__(self, response: object) -> None:
            super().__init__("status error")
            self.response = response

    class FakeResponse:
        status_code = 200
        text = '{"choices":[{"message":{"content":"ok"}}]}'

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "{}"}}]}

    def fake_post(
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> FakeResponse:
        seen["url"] = url
        seen["headers"] = headers
        seen["json"] = json
        seen["timeout"] = timeout
        return FakeResponse()

    fake_httpx = SimpleNamespace(
        HTTPError=FakeHTTPError,
        HTTPStatusError=FakeHTTPStatusError,
        post=fake_post,
    )

    config = OpenAICompatibleLLMConfig(
        api_key="secret",
        base_url="https://example.invalid/v1",
        model="model-x",
        output_mode="json_object",
        timeout_seconds=9.5,
    )

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        llm = make_httpx_llm(config)
        assert llm("Return JSON please", {}) == "{}"

    assert seen["url"] == "https://example.invalid/v1/chat/completions"
    assert seen["timeout"] == 9.5
    headers = seen["headers"]
    assert isinstance(headers, dict)
    assert "Authorization" in headers
    payload = seen["json"]
    assert isinstance(payload, dict)
    assert payload["response_format"] == {"type": "json_object"}


def test_probe_raises_clear_error_on_http_failure() -> None:
    class FakeHTTPError(Exception):
        pass

    class FakeHTTPStatusError(FakeHTTPError):
        def __init__(self, response: object) -> None:
            super().__init__("status error")
            self.response = response

    class FakeResponse:
        status_code = 404
        text = '{"error":{"message":"model_not_found"}}'

        def raise_for_status(self) -> None:
            raise FakeHTTPStatusError(self)

        def json(self) -> dict[str, object]:
            return {"error": {"message": "model_not_found"}}

    fake_httpx = SimpleNamespace(
        HTTPError=FakeHTTPError,
        HTTPStatusError=FakeHTTPStatusError,
        post=lambda *args, **kwargs: FakeResponse(),
    )

    config = OpenAICompatibleLLMConfig(
        api_key="secret",
        base_url="https://example.invalid/v1",
        model="missing-model",
    )

    with (
        patch.dict(sys.modules, {"httpx": fake_httpx}),
        pytest.raises(RuntimeError, match="missing-model"),
    ):
        probe_openai_compatible_llm(config)
