"""LangChain adapter — EmotionalMemoryChatHistory.

Extends ``BaseChatMessageHistory`` so any LangChain chain or agent that
accepts a chat-history object can be backed by an :class:`EmotionalMemory`
instance::

    from emotional_memory import EmotionalMemory, InMemoryStore
    from emotional_memory.integrations import EmotionalMemoryChatHistory
    from my_embedder import MyEmbedder

    em = EmotionalMemory(store=InMemoryStore(), embedder=MyEmbedder())
    history = EmotionalMemoryChatHistory(em)

    # Drop-in replacement for ConversationBufferMemory patterns:
    history.add_user_message("How are you today?")
    history.add_ai_message("I feel great, thanks for asking!")
    print(history.messages)   # [HumanMessage(...), AIMessage(...)]

Requires the ``langchain`` extra::

    pip install emotional-memory[langchain]

The adapter is intentionally thin:

- ``add_message()`` calls :meth:`EmotionalMemory.encode` with role metadata
  so the affective state evolves naturally with the conversation.
- ``messages`` reconstructs the conversation in timestamp order from the store.
- ``clear()`` removes all memories, resetting both the store and the
  affective state trajectory.

Multimodal content blocks (``List[Union[str, Dict]]``) are coerced to a
plain string via ``str()`` before encoding, which is the safest default when
the underlying embedder is text-only.
"""

from __future__ import annotations

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from emotional_memory.engine import EmotionalMemory


class EmotionalMemoryChatHistory(BaseChatMessageHistory):
    """LangChain ``BaseChatMessageHistory`` backed by :class:`EmotionalMemory`.

    Parameters
    ----------
    em:
        A fully constructed :class:`EmotionalMemory` instance (store +
        embedder already wired).  Pass it from outside so callers retain
        control over the store backend and embedder choice.
    """

    __slots__ = ("_em",)

    def __init__(self, em: EmotionalMemory) -> None:
        self._em = em

    # ------------------------------------------------------------------
    # BaseChatMessageHistory protocol
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[BaseMessage]:
        """All stored messages in chronological order."""
        mems = self._em.list_all()
        mems.sort(key=lambda m: m.tag.timestamp)
        result: list[BaseMessage] = []
        for mem in mems:
            role = mem.metadata.get("role", "user")
            text = mem.content
            if role == "assistant":
                result.append(AIMessage(content=text))
            else:
                result.append(HumanMessage(content=text))
        return result

    @messages.setter
    def messages(self, value: list[BaseMessage]) -> None:
        """Replace all stored messages (clears then re-encodes each one)."""
        self.clear()
        for msg in value:
            self.add_message(msg)

    def add_message(self, message: BaseMessage) -> None:
        """Encode *message* into emotional memory with its role preserved."""
        if isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            role = message.type  # "system", "tool", etc.
        # Coerce multimodal content blocks to plain text
        content = message.content if isinstance(message.content, str) else str(message.content)
        self._em.encode(content, metadata={"role": role})

    def clear(self) -> None:
        """Remove all memories from the store."""
        for mem in self._em.list_all():
            self._em.delete(mem.id)

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(messages={len(self._em.list_all())})"
