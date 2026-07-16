# Configuration Guide

`EmotionalMemoryConfig` nests behaviour across decay, retrieval, resonance, and
top-level flags. Use this decision tree for common deployments.

## Quick start (no LLM)

```python
from emotional_memory import EmotionalMemory, EmotionalMemoryConfig, InMemoryStore

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=my_embedder,
    config=EmotionalMemoryConfig(
        auto_categorize=True,
    ),
)
```

Use `KeywordAppraisalEngine` or manual `set_affect()` when appraisal is needed
without an LLM.

## Production with LLM appraisal

```python
from emotional_memory import DIRECT_VAD_SCHEMA, LLMAppraisalConfig, LLMAppraisalEngine
from emotional_memory.llm_http import OpenAICompatibleLLMConfig, make_httpx_llm

appraisal = LLMAppraisalEngine(
    llm=make_httpx_llm(OpenAICompatibleLLMConfig.from_env()),
    config=LLMAppraisalConfig(appraisal_schema=DIRECT_VAD_SCHEMA),
)
```

Set `appraisal_max_concurrency` (default 8) to bound parallel `encode_batch` LLM
calls. Use `1` if your LLM client is not thread-safe.

LLM appraisal is usually the dominant encode latency (hundreds of ms–seconds).
For a lower hot-path cost set `dual_path_encoding=True` and call `elaborate()` /
`elaborate_pending()` off the critical path — details and measured retrieve
breakdowns in [Performance & Scaling](../guides/performance_scaling.md).

## Oracle-free retrieve-time affect

```python
# Appraise query, score with query affect (Addendum T)
results = em.retrieve_with_query_appraisal("I'm worried about tomorrow")

# Safe wrapper: neutral queries → semantic-only (Addendum Y)
results = em.retrieve_query_gated("What is the capital of France?")
```

Tune `query_affect_gate_tau` (default `0.2`) on `EmotionalMemoryConfig`.

## Query-type routing (Addendum L)

Enable per-query weight routing when dialogue mixes factual and narrative queries:

```python
from emotional_memory import EmotionalMemoryConfig
from emotional_memory.query_classifier import LOCOMO_ROUTING
from emotional_memory.retrieval import QueryClassifierConfig, RetrievalConfig

config = EmotionalMemoryConfig(
    retrieval=RetrievalConfig(
        query_classifier=QueryClassifierConfig(
            mode="heuristic",
            routed_weights=LOCOMO_ROUTING,
        )
    )
)
```

## Persistence & scale

| Scale | Store | State |
|-------|-------|-------|
| Dev / tests | `InMemoryStore` | `InMemoryAffectiveStateStore` |
| Single process | `SQLiteStore` | `SQLiteAffectiveStateStore` |
| Multi-worker | `QdrantStore` / `ChromaStore` | Redis per worker (`strict=True`) |

## Safety bounds

- `max_content_length` — reject oversized `encode` / `retrieve` strings (optional).
- `prune(threshold)` — accepts `threshold` in `[0, 1]` only.

## Ablation flags

Set `enable_mood_signal`, `enable_momentum`, `enable_resonance`, or
`enable_reconsolidation` to `False` to isolate layers during experiments.

## See also

- [Performance & Scaling](../guides/performance_scaling.md)
- [Troubleshooting](../troubleshooting.md)
- [LLM environment](../contributing/llm-environment.md)
- [Async tutorial](async.md)
