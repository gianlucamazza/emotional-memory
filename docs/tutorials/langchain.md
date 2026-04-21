# LangChain integration

`emotional_memory` ships a drop-in `BaseChatMessageHistory` implementation
that backs any LangChain chain or agent with the full AFT pipeline.  The
adapter keeps a transcript for LangChain while letting you choose whether each
message becomes retrievable memory, only updates affective state, or is ignored.

## Installation

```bash
pip install emotional-memory[langchain]
```

This pulls in `langchain-core>=0.3`.  You will also need your preferred
LangChain provider package (e.g. `langchain-openai`, `langchain-anthropic`).

## Basic usage

```python
from emotional_memory import EmotionalMemory, InMemoryStore
from emotional_memory.integrations import (
    EmotionalMemoryChatHistory,
    recommended_conversation_policy,
)

em = EmotionalMemory(store=InMemoryStore(), embedder=MyEmbedder())
history = EmotionalMemoryChatHistory(em, message_policy=recommended_conversation_policy)

history.add_user_message("Explain the concept of affective computing.")
history.add_ai_message("Affective computing is the study of systems that can recognize, "
                       "interpret, process, and simulate human emotions…")

print(history.messages)   # [HumanMessage(...), AIMessage(...)]
print(repr(history))      # EmotionalMemoryChatHistory(messages=2)
```

With `recommended_conversation_policy`, normal user messages become retrievable
memories, assistant replies call `em.observe()` so they still shape mood and
momentum, and recall/control commands such as `recall happy moments` are ignored.

## Using with `RunnableWithMessageHistory`

LangChain's `RunnableWithMessageHistory` accepts any `BaseChatMessageHistory`
factory.  Swap in `EmotionalMemoryChatHistory` to make your chain
emotion-aware:

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Build or load your LLM chain
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
llm = ChatOpenAI(model="gpt-5-mini")
chain = prompt | llm

# One EmotionalMemory per session — wire into a factory
sessions: dict[str, EmotionalMemoryChatHistory] = {}

def get_session_history(session_id: str) -> EmotionalMemoryChatHistory:
    if session_id not in sessions:
        em = EmotionalMemory(store=InMemoryStore(), embedder=MyEmbedder())
        sessions[session_id] = EmotionalMemoryChatHistory(
            em,
            message_policy=recommended_conversation_policy,
        )
    return sessions[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Invoke as normal — history is automatically loaded and saved
response = chain_with_history.invoke(
    {"input": "How are you doing today?"},
    config={"configurable": {"session_id": "user-42"}},
)
print(response.content)
```

## Persistent sessions with SQLiteStore

Replace `InMemoryStore` with `SQLiteStore` to survive process restarts.
Pair it with `save_state` / `load_state` for mood continuity:

```python
import json, pathlib
from emotional_memory import SQLiteStore

STATE_FILE = pathlib.Path("session_state.json")

def get_session_history(session_id: str) -> EmotionalMemoryChatHistory:
    db_path = f"sessions/{session_id}.db"
    em = EmotionalMemory(
        store=SQLiteStore(db_path),
        embedder=MyEmbedder(),
    )
    if STATE_FILE.exists():
        em.load_state(json.loads(STATE_FILE.read_text()))
    history = EmotionalMemoryChatHistory(
        em,
        message_policy=recommended_conversation_policy,
    )
    return history
```

!!! note
    Call `em.save_state()` and persist the result when shutting down the
    session.  `SQLiteStore` persists memories automatically; `save_state()`
    additionally saves the mood trajectory (valence, arousal, momentum
    history).

## Mood-aware retrieval from history

`EmotionalMemoryChatHistory.messages` returns the in-memory transcript kept by
the adapter.  Stored memories remain queryable through the underlying engine
(e.g. "what distressed the user most?"):

```python
em = history._em   # access the underlying engine
results = em.retrieve("frustration anger conflict", top_k=3)
for mem in results:
    print(mem.content, mem.tag.core_affect)
```

## API summary

| Method | Description |
|---|---|
| `add_message(msg)` | Append to transcript and apply the configured message policy |
| `add_user_message(text)` | Shortcut — wraps in `HumanMessage` |
| `add_ai_message(text)` | Shortcut — wraps in `AIMessage` |
| `messages` (property) | Transcript messages in timestamp order |
| `messages` (setter) | Clear + re-encode from a list |
| `clear()` | Clear transcript, delete stored memories, reset affective state |

## Choosing a message policy

Two helpers are exported from `emotional_memory.integrations`:

- `store_all_messages` — backwards-compatible mode; every message is stored
- `recommended_conversation_policy` — user memories only, assistant `state_only`,
  recall/control commands ignored

Use `store_all_messages` when you intentionally want conversation-as-memory.
Use `recommended_conversation_policy` when you want episodic retrieval without
assistant replies or operational commands polluting the corpus.

## See also

- [`EmotionalMemoryChatHistory` source](https://github.com/gianlucamazza/emotional-memory/blob/main/src/emotional_memory/integrations/langchain.py)
- [Persistence tutorial](persistence.md)
- [Async tutorial](async.md)
- [LangChain `BaseChatMessageHistory` docs](https://python.langchain.com/docs/concepts/chat_history/)
