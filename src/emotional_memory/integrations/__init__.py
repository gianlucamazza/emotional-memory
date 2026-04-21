import contextlib

__all__ = [
    "EmotionalMemoryChatHistory",
    "recommended_conversation_policy",
    "store_all_messages",
]

with contextlib.suppress(ImportError):
    from emotional_memory.integrations.langchain import (
        EmotionalMemoryChatHistory,
        recommended_conversation_policy,
        store_all_messages,
    )
