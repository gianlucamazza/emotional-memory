import contextlib

__all__ = ["EmotionalMemoryChatHistory"]

with contextlib.suppress(ImportError):
    from emotional_memory.integrations.langchain import (
        EmotionalMemoryChatHistory as EmotionalMemoryChatHistory,
    )
