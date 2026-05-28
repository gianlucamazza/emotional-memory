# Embedders

Embedders convert text into dense vectors used by the retrieval pipeline.
They implement the [`Embedder`](interfaces.md) protocol — duck-typed, no
inheritance required — but production backends typically subclass
`SequentialEmbedder` (in `interfaces.py`) and override `embed_batch()` to
use the backend's native batching.

See [Interfaces](interfaces.md) for the protocol definition and
`AsyncEmbedder` for the async variant.

## SentenceTransformerEmbedder

!!! note
    Requires the `sentence-transformers` extra:
    `uv pip install "emotional-memory[sentence-transformers]"`

Importable from the top-level package once the extra is installed:

```python
from emotional_memory import EmotionalMemory, InMemoryStore
from emotional_memory.embedders import SentenceTransformerEmbedder

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=SentenceTransformerEmbedder.make_bge_small(),
)
```

::: emotional_memory.embedders.sentence_transformers.SentenceTransformerEmbedder
