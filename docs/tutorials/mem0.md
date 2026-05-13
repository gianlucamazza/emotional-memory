# mem0 integration

`emotional_memory` ships an `EmotionalMemoryMem0Backend` that exposes the
[mem0](https://github.com/mem0ai/mem0) `Memory` API — `add`, `search`, `get`,
`get_all`, `delete`, `delete_all`, `reset`, `close` — backed by the full AFT
retrieval pipeline (semantic similarity + mood congruence + decay + resonance).

Use it as a drop-in replacement when you want affect-aware retrieval instead
of mem0's LLM-based fact extraction.

## Installation

No runtime `mem0ai` dependency is required.  The backend is always available:

```bash
uv pip install "emotional-memory[sentence-transformers]"
```

If you also want the real mem0 for the chain pattern (see below), add the extra:

```bash
uv pip install "emotional-memory[mem0,sentence-transformers]"
```

## Basic usage

```python
from emotional_memory import EmotionalMemory, InMemoryStore
from emotional_memory.embedders import SentenceTransformerEmbedder
from emotional_memory.integrations import EmotionalMemoryMem0Backend

em = EmotionalMemory(store=InMemoryStore(), embedder=SentenceTransformerEmbedder())
backend = EmotionalMemoryMem0Backend(em, default_user_id="alice")

# Add memories — accepts a string or a list of {"role": ..., "content": ...} dicts
backend.add("I had a wonderful day at the park.")
backend.add([{"role": "user", "content": "Feeling anxious about the deadline."}])

# Retrieve with affect-aware scoring
results = backend.search("outdoors positive moments")
for item in results["results"]:
    print(item["memory"], "  score:", round(item["score"], 3))
```

## Multi-user sessions

Pass `user_id` to `add` and `search` to keep per-user memory spaces:

```python
backend.add("Alice loves hiking.", user_id="alice")
backend.add("Bob prefers museums.", user_id="bob")

alice_memories = backend.search("weekend activities", user_id="alice")
# → returns only Alice's memories
```

`default_user_id` on the constructor sets the fallback when `user_id` is
omitted from a call:

```python
backend = EmotionalMemoryMem0Backend(em, default_user_id="alice")
backend.add("Quick note.")   # stored under user_id="alice"
```

## API summary

| Method | Description |
|---|---|
| `add(messages, *, user_id, metadata)` | Encode and store; returns `{"results": [{"id", "memory", "event"}]}` |
| `search(query, *, user_id, limit, filters)` | Affect-aware retrieval; returns `{"results": [{"id", "memory", "score", "metadata"}]}` |
| `get(memory_id)` | Fetch one memory by ID; returns dict or `None` |
| `get_all(*, user_id, limit)` | List all memories for a user |
| `delete(memory_id)` | Remove one memory |
| `delete_all(*, user_id)` | Remove all memories for a user |
| `reset()` | Clear all memories and reset affective state |
| `close()` | Release engine resources |

`messages` in `add()` accepts either a plain `str` or a list of mem0-style
`{"role": "user", "content": "..."}` dicts.  The helper
`messages_to_content(messages)` performs the coercion and is exported from
`emotional_memory.integrations` for custom pre-processing.

## Mood-aware search

The retrieval pipeline weighs semantic similarity alongside the current mood
trajectory.  After encoding several memories the affective state shifts, so
a query for "frustration" will rank high-arousal negative memories above
neutral ones sharing the same keywords:

```python
from emotional_memory import EmotionalMemory, InMemoryStore
from emotional_memory.embedders import SentenceTransformerEmbedder
from emotional_memory.integrations import EmotionalMemoryMem0Backend

em = EmotionalMemory(store=InMemoryStore(), embedder=SentenceTransformerEmbedder())
backend = EmotionalMemoryMem0Backend(em)

backend.add("Missed the train and arrived late to the meeting.")
backend.add("The project review went poorly; the client seemed unhappy.")
backend.add("Grabbed coffee with a friend; felt better afterwards.")

results = backend.search("stressful work moments", limit=2)
for item in results["results"]:
    print(item["memory"])
```

Access the underlying engine directly for full retrieval-with-explanations:

```python
em = backend._em
explanations = em.retrieve_with_explanations("stress", top_k=3)
for exp in explanations:
    print(exp.memory.content)
    print("  semantic:", round(exp.breakdown.raw_signals.semantic, 3))
    print("  mood_congruence:", round(exp.breakdown.raw_signals.mood_congruence, 3))
```

## Chain pattern: mem0 fact-extraction + AFT retrieval

The adapter stores memories verbatim.  If you want LLM-based fact extraction
first, chain a real `mem0.Memory` instance as a pre-processor and route its
extracted facts into the backend:

```python
from mem0 import Memory  # requires pip install mem0ai

mem = Memory()           # uses its own LLM + vector store
backend = EmotionalMemoryMem0Backend(em)

# Extract facts with mem0, then store in AFT backend
raw_text = "I finished the marathon in 4h12m and felt incredibly proud."
extracted = mem.add([{"role": "user", "content": raw_text}], user_id="alice")

for fact in extracted.get("results", []):
    backend.add(fact["memory"], user_id="alice")

# Retrieval uses AFT scoring
results = backend.search("athletic achievements", user_id="alice")
```

## Persistent sessions with SQLiteStore

Replace `InMemoryStore` with `SQLiteStore` to survive process restarts:

```python
from emotional_memory import SQLiteStore

em = EmotionalMemory(
    store=SQLiteStore("user_memories.db"),
    embedder=SentenceTransformerEmbedder(),
)
backend = EmotionalMemoryMem0Backend(em, default_user_id="alice")
```

## See also

- [`EmotionalMemoryMem0Backend` source](https://github.com/gianlucamazza/emotional-memory/blob/main/src/emotional_memory/integrations/mem0.py)
- [LangChain integration](langchain.md)
- [Persistence tutorial](persistence.md)
- [mem0 documentation](https://docs.mem0.ai/)
