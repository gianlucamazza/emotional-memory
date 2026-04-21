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

- ``add_message()`` appends to an in-memory transcript and applies a caller-
  supplied message policy.
- Policies can choose ``store`` (persist retrievable memory), ``state_only``
  (update affective state without storing), or ``ignore``.
- ``clear()`` removes the transcript, all stored memories, and resets the
  affective state trajectory.

Multimodal content blocks (``List[Union[str, Dict]]``) are coerced to a
plain string via ``str()`` before encoding, which is the safest default when
the underlying embedder is text-only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeAlias

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from emotional_memory.engine import EmotionalMemory

if TYPE_CHECKING:

    class BaseChatMessageHistory:
        """Typed subset of LangChain's chat history base used by this adapter."""

        @property
        def messages(self) -> list[BaseMessage]: ...

        @messages.setter
        def messages(self, value: list[BaseMessage]) -> None: ...

        def add_message(self, message: BaseMessage) -> None: ...

        def add_user_message(self, message: str) -> None: ...

        def add_ai_message(self, message: str) -> None: ...

        def clear(self) -> None: ...

else:
    from langchain_core.chat_history import BaseChatMessageHistory

MessageHandlingMode: TypeAlias = Literal["store", "state_only", "ignore"]
MessagePolicy: TypeAlias = Callable[[BaseMessage], MessageHandlingMode]


def store_all_messages(_: BaseMessage) -> MessageHandlingMode:
    """Backwards-compatible policy: every message becomes retrievable memory."""
    return "store"


def _coerce_content(message: BaseMessage) -> str:
    """Coerce multimodal LangChain content blocks to plain text."""
    return message.content if isinstance(message.content, str) else str(message.content)


def _is_control_message(content: str) -> bool:
    """Return True when *content* is an operational recall command."""
    normalized = content.strip().lower()
    return (
        normalized == "recall"
        or normalized.startswith("recall ")
        or normalized == "remember"
        or normalized.startswith("remember ")
    )


def recommended_conversation_policy(message: BaseMessage) -> MessageHandlingMode:
    """Recommended chat policy for episodic memory + separate transcript.

    - user messages become retrievable memories
    - assistant messages only shape the affective state
    - recall/remember commands and non-conversational roles are ignored
    """
    if isinstance(message, AIMessage):
        return "state_only"
    if isinstance(message, HumanMessage):
        return "ignore" if _is_control_message(_coerce_content(message)) else "store"
    return "ignore"


class EmotionalMemoryChatHistory(BaseChatMessageHistory):
    """LangChain ``BaseChatMessageHistory`` backed by :class:`EmotionalMemory`.

    Parameters
    ----------
    em:
        A fully constructed :class:`EmotionalMemory` instance (store +
        embedder already wired).  Pass it from outside so callers retain
        control over the store backend and embedder choice.
    """

    __slots__ = ("_em", "_message_policy", "_messages")

    def __init__(
        self,
        em: EmotionalMemory,
        *,
        message_policy: MessagePolicy | None = None,
    ) -> None:
        self._em = em
        self._message_policy = message_policy or store_all_messages
        self._messages = self._load_messages()

    # ------------------------------------------------------------------
    # BaseChatMessageHistory protocol
    # ------------------------------------------------------------------

    def _load_messages(self) -> list[BaseMessage]:
        """Reconstruct any previously stored transcript from the memory store."""
        mems = self._em.list_all()
        mems.sort(key=lambda m: m.tag.timestamp)
        result: list[BaseMessage] = []
        for mem in mems:
            role = mem.metadata.get("role", "user")
            if role == "assistant":
                result.append(AIMessage(content=mem.content))
            elif role == "system":
                result.append(SystemMessage(content=mem.content))
            else:
                result.append(HumanMessage(content=mem.content))
        return result

    @property
    def messages(self) -> list[BaseMessage]:
        """All transcript messages in chronological order."""
        return list(self._messages)

    @messages.setter
    def messages(self, value: list[BaseMessage]) -> None:
        """Replace all stored messages (clears then re-encodes each one)."""
        self.clear()
        for msg in value:
            self.add_message(msg)

    def add_message(self, message: BaseMessage) -> None:
        """Record *message* in the transcript and apply the configured policy."""
        if isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            role = message.type  # "system", "tool", etc.
        content = _coerce_content(message)

        self._messages.append(message)
        mode = self._message_policy(message)
        if mode == "store":
            self._em.encode(content, metadata={"role": role})
        elif mode == "state_only":
            self._em.observe(content, metadata={"role": role})
        elif mode != "ignore":
            raise ValueError(f"unknown message handling mode: {mode}")

    def add_user_message(self, message: str) -> None:
        """Append a user-authored message to the transcript."""
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Append an assistant-authored message to the transcript."""
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        """Remove transcript messages, stored memories, and affective state."""
        self._messages = []
        for mem in self._em.list_all():
            self._em.delete(mem.id)
        self._em.reset_state()

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(messages={len(self._messages)})"
