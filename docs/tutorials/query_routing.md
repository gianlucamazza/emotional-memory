# Query routing

The query classifier routes each `retrieve()` call to a per-type weight profile,
applying different retrieval strategies depending on whether the query targets an
emotion state, an affect-conditioned topic, or a temporal narrative trajectory.

This tutorial covers:

1. Enabling the heuristic classifier (zero config, fast)
2. Plugging in the Addendum L routing table (`LOCOMO_ROUTING`)
3. Writing a custom classifier
4. (Advanced) LLM-backed classification

## 1. Enable heuristic routing

```python
from emotional_memory import EmotionalMemory, InMemoryStore, EmotionalMemoryConfig
from emotional_memory.retrieval import RetrievalConfig, QueryClassifierConfig
from emotional_memory.query_classifier import HeuristicQueryClassifier, LOCOMO_ROUTING

config = EmotionalMemoryConfig(
    retrieval=RetrievalConfig(
        query_classifier=QueryClassifierConfig(
            routed_weights=LOCOMO_ROUTING,
        )
    )
)

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=my_embedder,
    config=config,
    query_classifier=HeuristicQueryClassifier(),
)
```

The engine now classifies every query and applies the matching weight vector.
Unrecognised types fall back to `default_type` (default: `"default"`).

## 2. The LOCOMO routing table

`LOCOMO_ROUTING` maps each LoCoMo question type to a 6-element weight vector
`[semantic, mood_congruence, affect_proximity, momentum, recency, resonance_boost]`:

| Type | Weights | Strategy |
|---|---|---|
| `single_hop` | `[0.70, 0.10, 0.05, 0.05, 0.05, 0.05]` | Suppress affect, maximise semantic for literal-fact questions |
| `multi_hop` | `[0.50, 0.30, 0.10, 0.05, 0.05, 0.00]` | Moderate mood, no resonance |
| `open_domain` | `[0.40, 0.20, 0.15, 0.10, 0.10, 0.05]` | Balanced semantic + mood |
| `temporal` | `[0.50, 0.30, 0.10, 0.05, 0.05, 0.00]` | Same as multi_hop |
| `default` | `[0.35, 0.25, 0.15, 0.10, 0.10, 0.05]` | Standard AFT baseline |

> **Empirical note**: These weights are derived from Addendum J closure (oracle Pareto sweep).
> Addendum L (confirmatory, N=1540) tested whether routing improves over W2 on LoCoMo — result
> pending. The routing table is pre-registered; use `LOCOMO_ROUTING` as-is for reproducibility.

## 3. Custom classifier

Implement the `QueryClassifier` protocol (a single `classify(query: str) -> str` method):

```python
from emotional_memory.query_classifier import QueryClassifier, QUERY_TYPES

class EmotionKeywordClassifier:
    """Routes queries mentioning an emotion word to open_domain."""

    _EMOTION_WORDS = frozenset(["happy", "sad", "angry", "afraid", "surprised"])

    def classify(self, query: str) -> str:
        if any(w in query.lower() for w in self._EMOTION_WORDS):
            return "open_domain"
        return "default"
```

Pass it to `EmotionalMemory`:

```python
em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=my_embedder,
    config=config,
    query_classifier=EmotionKeywordClassifier(),
)
```

Unknown labels are silently mapped to `default_type` — no exception.

## 4. LLM-backed classifier (advanced)

`LLMQueryClassifier` uses an OpenAI-compatible endpoint to classify queries.
It mirrors the `LLMAppraisalEngine` pattern (thread-safe, LRU cache):

```python
from emotional_memory.query_classifier import LLMQueryClassifier
from emotional_memory.llm_http import make_httpx_llm_from_env

classifier = LLMQueryClassifier(
    llm=make_httpx_llm_from_env(),
    valid_types=frozenset(LOCOMO_ROUTING.keys()),
    default_type="default",
    cache_size=512,
)
```

Requires `EMOTIONAL_MEMORY_LLM_API_KEY` in the environment. See
[LLM configuration](../index.md) for environment variables.

## API reference

[Query Classifier API](../api/query_classifier.md)
