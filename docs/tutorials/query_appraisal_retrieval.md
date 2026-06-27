# Retrieve-time query appraisal

By default the affect-proximity signal (s3) in retrieval scores each memory against
the engine's **current runtime affective state**. When the query itself carries
affect that the runtime state does not reflect, you can score against the **query's
own affect** instead — without mutating the runtime state.

This is the production-reachable mechanism studied in Addendum T: appraising the query
at retrieve-time recovers most of the oracle-affect advantage on the curated
`realistic_recall_v2` benchmark, with no oracle. See the scope caveat at the end.

This tutorial covers three usage patterns:

1. Pass an explicit `query_affect`
2. Auto-appraise the query (`retrieve_with_query_appraisal`)
3. Pair with `DIRECT_VAD_SCHEMA` (recommended)

## 1. Pass an explicit `query_affect`

`retrieve()` and `retrieve_with_explanations()` accept a keyword-only
`query_affect: CoreAffect | None`. When provided, it feeds the affect-proximity
signal instead of the runtime state — the runtime state is left untouched.

```python
from emotional_memory import EmotionalMemory, InMemoryStore
from emotional_memory.affect import CoreAffect

em = EmotionalMemory(store=InMemoryStore(), embedder=my_embedder)
# ... encode some memories ...

# Score against the query's affect (e.g. an anxious, negative-valence query),
# regardless of the engine's current mood.
results = em.retrieve(
    "I'm worried I forgot something important before the trip",
    top_k=5,
    query_affect=CoreAffect(valence=-0.6, arousal=0.7),
)
```

`query_affect=None` (the default) preserves the previous behaviour (uses the runtime
state), so this is fully backward-compatible.

## 2. Auto-appraise the query

If you have an appraisal engine configured, `retrieve_with_query_appraisal()`
appraises the query text for you and uses the result as `query_affect` — no oracle,
no state mutation:

```python
results = em.retrieve_with_query_appraisal(
    "I'm worried I forgot something important before the trip",
    top_k=5,
)
```

It raises `RuntimeError` if no appraisal engine is configured. Both methods exist on
`AsyncEmotionalMemory` as well (awaitable).

## 3. Pair with `DIRECT_VAD_SCHEMA` (recommended)

Addendum V found that having the LLM rate valence/arousal/dominance **directly**
estimates query affect more faithfully than the default Scherer SEC→projection. Pair
the convenience method with an LLM appraisal engine using `DIRECT_VAD_SCHEMA`:

```python
from emotional_memory import DIRECT_VAD_SCHEMA
from emotional_memory.appraisal_llm import LLMAppraisalConfig, LLMAppraisalEngine
from emotional_memory.llm_http import OpenAICompatibleLLMConfig, make_httpx_llm

cfg = OpenAICompatibleLLMConfig.from_env()
appraisal_engine = LLMAppraisalEngine(
    llm=make_httpx_llm(cfg),
    config=LLMAppraisalConfig(appraisal_schema=DIRECT_VAD_SCHEMA, cache_size=4096),
)

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=my_embedder,
    appraisal_engine=appraisal_engine,
)
results = em.retrieve_with_query_appraisal("...", top_k=5)
```

The cache means each distinct query text is appraised once.

## Scope and honest caveat

Retrieve-time query appraisal helps **in the affect-discriminative regime** — where
the correct memory is distinguished by affect, not just lexical/semantic overlap. On
the curated `realistic_recall_v2` benchmark it beats cosine (Addendum T, Ht1 PASS).

A pre-registered naturalistic re-test (Addendum T2A) found it does **not** beat cosine
on naturalistic, affect-sparse dialogue (DailyDialog): the query appraisal there is
faithful (valence r=0.69, arousal r=0.74) yet the affect signal does not discriminate
when semantic overlap already carries it. Treat query appraisal as a targeted booster
for affect-conditioned retrieval, not a universal win.
