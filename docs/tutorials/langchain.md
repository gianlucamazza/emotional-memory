# LangChain integration

`emotional_memory` ships a drop-in `BaseChatMessageHistory` implementation
that backs any LangChain chain or agent with the full AFT pipeline.  Every
message added to the history is **encoded into emotional memory**, so the
agent's mood and momentum evolve naturally with the conversation.

## Installation

```bash
pip install emotional-memory[langchain]
```

This pulls in `langchain-core>=0.3`.  You will also need your preferred
LangChain provider package (e.g. `langchain-openai`, `langchain-anthropic`).

## Basic usage

```python
from emotional_memory import EmotionalMemory, InMemoryStore
from emotional_memory.integrations import EmotionalMemoryChatHistory

em = EmotionalMemory(store=InMemoryStore(), embedder=MyEmbedder())
history = EmotionalMemoryChatHistory(em)

history.add_user_message("Explain the concept of affective computing.")
history.add_ai_message("Affective computing is the study of systems that can recognize, "
                       "interpret, process, and simulate human emotions…")

print(history.messages)   # [HumanMessage(...), AIMessage(...)]
print(repr(history))      # EmotionalMemoryChatHistory(messages=2)
```

Every call to `add_message()` internally calls `em.encode()`, which updates
valence, arousal, momentum, and mood field in real time.

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
llm = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | llm

# One EmotionalMemory per session — wire into a factory
sessions: dict[str, EmotionalMemoryChatHistory] = {}

def get_session_history(session_id: str) -> EmotionalMemoryChatHistory:
    if session_id not in sessions:
        em = EmotionalMemory(store=InMemoryStore(), embedder=MyEmbedder())
        sessions[session_id] = EmotionalMemoryChatHistory(em)
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
    history = EmotionalMemoryChatHistory(em)
    return history
```

!!! note
    Call `em.save_state()` and persist the result when shutting down the
    session.  `SQLiteStore` persists memories automatically; `save_state()`
    additionally saves the mood trajectory (valence, arousal, momentum
    history).

## Mood-aware retrieval from history

`EmotionalMemoryChatHistory.messages` reconstructs the conversation in
chronological order from `em.list_all()`.  To query the emotional history
directly (e.g. "what distressed the user most?"), use `em.retrieve()`:

```python
em = history._em   # access the underlying engine
results = em.retrieve("frustration anger conflict", top_k=3)
for mem in results:
    print(mem.content, mem.tag.core_affect)
```

## API summary

| Method | Description |
|---|---|
| `add_message(msg)` | Encode any `BaseMessage` into emotional memory |
| `add_user_message(text)` | Shortcut — wraps in `HumanMessage` |
| `add_ai_message(text)` | Shortcut — wraps in `AIMessage` |
| `messages` (property) | All messages in timestamp order |
| `messages` (setter) | Clear + re-encode from a list |
| `clear()` | Delete all memories and reset affective state |

## See also

- [`EmotionalMemoryChatHistory` source](https://github.com/gianlucamazza/emotional-memory/blob/main/src/emotional_memory/integrations/langchain.py)
- [Persistence tutorial](persistence.md)
- [Async tutorial](async.md)
- [LangChain `BaseChatMessageHistory` docs](https://python.langchain.com/docs/concepts/chat_history/)
