# emotional_memory

[![CI](https://github.com/gianlucamazza/emotional-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/gianlucamazza/emotional-memory/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/emotional_memory)](https://pypi.org/project/emotional_memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gianlucamazza/emotional-memory/blob/main/LICENSE)

Emotional memory for LLMs based on **Affective Field Theory (AFT)** — a 5-layer model that
encodes not just *what* happened, but *how it felt*, *how that feeling was moving*, and
*what mood colored the moment*.

## Installation

```bash
pip install emotional-memory
pip install emotional-memory[sqlite]   # SQLite persistence
pip install emotional-memory[viz]      # matplotlib visualization
pip install emotional-memory[docs]     # docs generation (dev)
```

## Quickstart

```python
from emotional_memory import (
    EmotionalMemory, EmotionalMemoryConfig,
    InMemoryStore, CoreAffect, SequentialEmbedder,
)

class MyEmbedder(SequentialEmbedder):
    def embed(self, text: str) -> list[float]:
        return my_model.encode(text).tolist()
    # embed_batch() provided automatically

with EmotionalMemory(store=InMemoryStore(), embedder=MyEmbedder()) as em:
    em.set_affect(CoreAffect(valence=0.8, arousal=0.6))
    em.encode("Just shipped the feature after three hard weeks.")
    results = em.retrieve("difficult project success", top_k=3)
    for mem in results:
        print(mem.content, mem.tag.core_affect)
```

## The 5 Layers

| Layer | Class | Theory |
|---|---|---|
| **CoreAffect** | `CoreAffect` | Barrett/Russell circumplex — continuous (valence, arousal) |
| **AffectiveMomentum** | `AffectiveMomentum` | Spinoza — affect as transition (velocity + acceleration) |
| **MoodField** | `MoodField` | Heidegger §29 — slow-moving global mood with inertia |
| **AppraisalVector** | `AppraisalVector` | Scherer CPM — emotion from evaluation (5 dimensions) |
| **ResonanceLinks** | `ResonanceLink` | Aristotle/Bower/Collins & Loftus/Hebb — bidirectional associative graph (5 link types), multi-hop spreading activation + Hebbian co-retrieval strengthening |

See [Research](research/index.md) for full theoretical foundations.
