"""Optional integration exports.

This subpackage exposes:

- **LangChain adapter** (``EmotionalMemoryChatHistory``, ``recommended_conversation_policy``,
  ``store_all_messages``) — available when the ``langchain`` extra is installed.
- **mem0-compatible facade** (``EmotionalMemoryMem0Backend``, ``messages_to_content``) —
  always available; no runtime ``mem0ai`` dependency required.
"""

from __future__ import annotations

from typing import Any

from emotional_memory.integrations.mem0 import (
    EmotionalMemoryMem0Backend,
    messages_to_content,
)

_LANGCHAIN_EXPORTS = (
    "EmotionalMemoryChatHistory",
    "recommended_conversation_policy",
    "store_all_messages",
)

_MEM0_EXPORTS = (
    "EmotionalMemoryMem0Backend",
    "messages_to_content",
)

__all__: list[str] = ["EmotionalMemoryMem0Backend", "messages_to_content"]
_langchain_import_error: ImportError | None = None

try:
    from emotional_memory.integrations import langchain as _langchain
except ImportError as exc:
    _langchain_import_error = exc
else:
    EmotionalMemoryChatHistory = _langchain.EmotionalMemoryChatHistory
    recommended_conversation_policy = _langchain.recommended_conversation_policy
    store_all_messages = _langchain.store_all_messages
    __all__ = [
        "EmotionalMemoryChatHistory",
        "EmotionalMemoryMem0Backend",
        "messages_to_content",
        "recommended_conversation_policy",
        "store_all_messages",
    ]


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
