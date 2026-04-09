# emotional_memory

Emotional memory for LLMs based on **Affective Field Theory (AFT)** — a 5-layer model that encodes not just *what* happened, but *how it felt*, *how that feeling was moving*, and *what mood colored the moment*.

## Installation

```bash
pip install emotional_memory
```

For development:

```bash
git clone https://github.com/gianlucamazza/emotional-memory
cd emotional-memory
pip install -e ".[dev]"
```

## Quickstart

```python
from emotional_memory import (
    EmotionalMemory, EmotionalMemoryConfig,
    InMemoryStore, CoreAffect, AppraisalVector,
)

# Bring your own embedder — anything with .embed(text) -> list[float]
class MyEmbedder:
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

em = EmotionalMemory(store=InMemoryStore(), embedder=MyEmbedder())

# Set current emotional state
em.set_affect(CoreAffect(valence=0.8, arousal=0.6))

# Encode memories — each one captures the full affective context
em.encode("Just shipped the feature after three hard weeks.")
em.encode("Team celebration in the office.", metadata={"source": "slack"})

# Retrieve — ranked by semantic relevance AND emotional congruence
results = em.retrieve("difficult project success", top_k=3)
for mem in results:
    print(mem.content, mem.tag.core_affect)
```

## Affective Field Theory

AFT models emotion as a **field** — distributed, dynamic, multi-layer — rather than a discrete label or a single coordinate. Five layers are captured at encoding time:

| Layer | Model | What it captures |
|---|---|---|
| **CoreAffect** | Barrett/Russell circumplex | Continuous `(valence, arousal)` — the emotional substrate |
| **AffectiveMomentum** | Spinoza — affect as transition | Velocity and acceleration of affect change |
| **StimmungField** | Heidegger — *Stimmung* as attunement | Slow-moving global mood with inertia (EMA) |
| **AppraisalVector** | Scherer/Lazarus/Stoics | Emotion derived from evaluation: novelty, goal-relevance, coping, norm-congruence, self-relevance |
| **ResonanceLinks** | Aristotle/Hume/Bower | Associative graph: semantic, emotional, temporal, causal, contrastive links |

Full theoretical foundations: [`docs/research/`](docs/research/)

## API Overview

### `EmotionalMemory`

```python
em = EmotionalMemory(
    store: MemoryStore,
    embedder: Embedder,
    appraisal_engine: AppraisalEngine | None = None,  # optional: auto-appraise via LLM
    config: EmotionalMemoryConfig | None = None,
)
```

| Method | Description |
|---|---|
| `encode(content, appraisal=None, metadata=None) -> Memory` | Encode content with full AFT pipeline |
| `retrieve(query, top_k=5) -> list[Memory]` | Emotionally-weighted retrieval + reconsolidation |
| `get_state() -> AffectiveState` | Current affective state (read-only copy) |
| `set_affect(core_affect)` | Manually inject a CoreAffect |

### Key config classes

- `EmotionalMemoryConfig` — top-level config (decay, retrieval, resonance, stimmung alpha)
- `RetrievalConfig` — weights, APE threshold, reconsolidation learning rate
- `ResonanceConfig` — similarity threshold, max links, semantic/emotional/temporal weights
- `DecayConfig` — power-law decay parameters, arousal modulation, floor values

### Interfaces (bring your own)

```python
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

class MemoryStore(Protocol):
    def save(self, memory: Memory) -> None: ...
    def get(self, memory_id: str) -> Memory | None: ...
    def update(self, memory: Memory) -> None: ...
    def delete(self, memory_id: str) -> None: ...
    def list_all(self) -> list[Memory]: ...
    def search_by_embedding(self, embedding: list[float], top_k: int) -> list[Memory]: ...
```

`InMemoryStore` is included as a reference implementation (dict-backed, brute-force cosine search).

## Development

```bash
pytest                        # run tests
ruff check .                  # lint
ruff format .                 # format
mypy src/emotional_memory/    # type check
```

## License

MIT — see [LICENSE](LICENSE)
