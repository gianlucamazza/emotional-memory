"""Tests for optional integration exports when LangChain is unavailable."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


def _reload_integrations_module():
    sys.modules.pop("emotional_memory.integrations", None)
    sys.modules.pop("emotional_memory.integrations.langchain", None)
    return importlib.import_module("emotional_memory.integrations")


def test_integrations_subpackage_hides_unavailable_langchain_exports() -> None:
    with patch.dict(
        sys.modules,
        {
            "langchain_core": None,
            "langchain_core.chat_history": None,
            "langchain_core.messages": None,
        },
    ):
        integrations = _reload_integrations_module()
        assert integrations.__all__ == []


def test_integrations_subpackage_raises_actionable_error_for_optional_exports() -> None:
    with patch.dict(
        sys.modules,
        {
            "langchain_core": None,
            "langchain_core.chat_history": None,
            "langchain_core.messages": None,
        },
    ):
        integrations = _reload_integrations_module()

        with pytest.raises(ImportError, match="emotional-memory\\[langchain\\]"):
            _ = integrations.recommended_conversation_policy

    sys.modules.pop("emotional_memory.integrations", None)
