# Mental Model: How emotional-memory Works

This page is for developers who want to understand the library in 5 minutes before reading the API docs or the research papers.

---

## The core idea

Standard LLM memory stores text and retrieves it by semantic similarity. emotional-memory stores text *plus its emotional context* and retrieves by semantic similarity **and** emotional congruence.

This means: if you were anxious when you encoded a memory, the library will surface that memory more readily when you are anxious again — even if the query is semantically distant. This is [mood-congruent recall](https://doi.org/10.1037/0022-3514.40.6.905) (Bower 1981), a well-replicated psychological phenomenon.

---

## What happens on `encode`

```
text + current emotional state
           │
           ▼
    ┌─────────────┐
    │ AppraisalVector │  ← "Is this novel? Relevant to my goals? Threatening?"
    │  (Scherer CPM)  │    5 dimensions → mapped to CoreAffect delta
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  CoreAffect  │  ← (valence, arousal) — the emotional substrate of this moment
    └──────┬──────┘
           │
           ▼
    ┌────────────────┐
    │ AffectiveState │  ← updates momentum (velocity of affect change)
    │    .update()   │    updates MoodField (slow-moving background mood)
    └──────┬─────────┘
           │
           ▼
    ┌──────────────┐
    │ EmotionalTag  │  ← snapshot: core_affect, momentum, mood_snapshot,
    │               │    consolidation_strength, appraisal_vector, ...
    └──────┬────────┘
           │
           ▼
    embed(text) → vector
           │
           ▼
    store.save(Memory(content, embedding, tag))
           │
           ▼
    build ResonanceLinks  ← bidirectional graph: semantic / emotional /
                            temporal / causal / contrastive links
                            Hebbian strengthening on co-activation
```

The result is a `Memory` object: text + embedding vector + a rich `EmotionalTag` capturing the full affective context at encoding time.

**Dual-path encoding** (optional, `dual_path_encoding=True`): mirrors LeDoux's two-route model. The fast path skips appraisal and encodes immediately with raw affect (`pending_appraisal=True`). The slow path runs full appraisal later via `elaborate(memory_id)` and blends the result back in (70% appraised / 30% raw).

---

## What happens on `retrieve`

```
query text
    │
    ▼
embed(query) → query_vector
    │
    ▼
Pass 1 — score all candidates (6 signals, no spreading):
    s1  semantic similarity     (cosine of embeddings)
    s2  mood congruence         (how close is memory's mood to current mood?)
    s3  core affect proximity   (distance in valence-arousal space)
    s4  momentum alignment      (do affect trajectories point the same way?)
    s5  recency / decay         (ACT-R power-law: strength decays with time)
    s6  resonance boost         (placeholder; set to 0 in Pass 1)

Weights are adaptive: the MoodField shifts w1-w6 based on current arousal.
High arousal → semantic signal dominates. Low arousal → emotional signals dominate.
    │
    ▼
Top-k seeds
    │
    ▼
spreading_activation(seeds)  ← multi-hop BFS on ResonanceGraph
    │                           returns activation_map: memory_id → score
    ▼
Pass 2 — re-score seeds with resonance boost (s6 = activation_map[id])
    │
    ▼
Per memory: compute_ape() — Affective Prediction Error
    │         "How surprising is this memory given the current context?"
    ▼
APE-gated reconsolidation:
    high APE → open lability window → memory becomes reconsolidatable
    retrieval within open window → update EmotionalTag with current state
    │
    ▼
hebbian_strengthen() on co-retrieved ResonanceLinks
    │
    ▼
return top-k Memory objects
```

---

## The 5 layers at a glance

| Layer | What it is | Why it matters for retrieval |
|---|---|---|
| **CoreAffect** | (valence, arousal) point in the circumplex | Mood-congruent matching: memories from similar emotional states rank higher |
| **AffectiveMomentum** | Velocity + acceleration of affect change | Captures the *direction* of emotion, not just its position |
| **MoodField** | Slow-moving background mood (EMA) | Provides the "ambient" emotional filter that shifts retrieval weights |
| **AppraisalVector** | 5 Scherer evaluation checks → CoreAffect delta | Explains *why* something felt the way it did |
| **ResonanceLinks** | Bidirectional associative graph | Surfaces memories linked by association, not just by text similarity |

---

## What makes this different from plain vector memory

| | Plain vector memory | emotional-memory |
|---|---|---|
| Stored per memory | text + embedding | text + embedding + full emotional tag |
| Retrieval signal | cosine similarity | 6-signal weighted score (mood, affect, momentum, decay, resonance) |
| Memory strength | flat (or TTL) | ACT-R power-law decay modulated by arousal at encoding |
| Memory update | none / overwrite | APE-gated reconsolidation: high-surprise retrieval rewrites the tag |
| Association graph | none | Bidirectional typed graph with Hebbian strengthening |
| Global state | none | MoodField (persists across encodes/retrieves) |

---

## Key objects

```python
from emotional_memory import (
    EmotionalMemory,     # main entry point
    EmotionalMemoryConfig,
    InMemoryStore,       # reference store (swap for SQLiteStore or custom)
    CoreAffect,          # (valence: float, arousal: float) — both frozen Pydantic
    EmotionalTag,        # snapshot attached to each Memory at encoding time
    Memory,              # content + embedding + tag
    AffectiveState,      # mutable engine state (mood, momentum, history)
)
```

`EmotionalMemory` is the façade. It orchestrates all 5 layers. You interact with it via three verbs: `set_affect`, `encode`, `retrieve`.

---

## Minimal working example

```python
from emotional_memory import EmotionalMemory, InMemoryStore, CoreAffect
from emotional_memory.embedders import SentenceTransformerEmbedder

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=SentenceTransformerEmbedder(),
)

# Encode with a positive, high-arousal state (e.g. after shipping a feature)
em.set_affect(CoreAffect(valence=0.8, arousal=0.7))
em.encode("Finally shipped the new recommendation engine after three hard weeks.")
em.encode("Team celebrated with pizza — everyone relieved and proud.")

# Encode with a negative state (e.g. an incident)
em.set_affect(CoreAffect(valence=-0.7, arousal=0.9))
em.encode("Database outage at 3am — on-call for two hours.")

# Retrieve from a positive state → mood-congruent memories rank first
em.set_affect(CoreAffect(valence=0.6, arousal=0.5))
results = em.retrieve("project work", top_k=2)

# results[0] will be one of the positive memories, not the outage
for m in results:
    print(m.content, "| valence:", m.tag.core_affect.valence)
```

---

## Further reading

- Full theoretical foundations: [`docs/research/`](research/)
- Known limitations: [`docs/research/08_limitations.md`](research/08_limitations.md)
- API reference: generated from docstrings (run `make docs-serve`)
- Source of truth for retrieval scoring: [`src/emotional_memory/retrieval.py`](https://github.com/gianlucamazza/emotional-memory/blob/main/src/emotional_memory/retrieval.py)
