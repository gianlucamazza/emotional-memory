"""Optional integration exports.

This subpackage currently exposes the LangChain adapter when the
``langchain`` extra is installed.
"""

from __future__ import annotations

from typing import Any

_LANGCHAIN_EXPORTS = (
    "EmotionalMemoryChatHistory",
    "recommended_conversation_policy",
    "store_all_messages",
)

__all__: list[str] = []
_langchain_import_error: ImportError | None = None

try:
    from emotional_memory.integrations import langchain as _langchain
except ImportError as exc:
    _langchain_import_error = exc
else:
    EmotionalMemoryChatHistory = _langchain.EmotionalMemoryChatHistory
    recommended_conversation_policy = _langchain.recommended_conversation_policy
    store_all_messages = _langchain.store_all_messages
    __all__ = list(_LANGCHAIN_EXPORTS)


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    if name in _LANGCHAIN_EXPORTS and _langchain_import_error is not None:
        raise ImportError(
            "LangChain integration requires the optional 'langchain' extra. "
            "Install with: pip install 'emotional-memory[langchain]'"
        ) from _langchain_import_error
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
