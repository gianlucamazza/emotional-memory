---
date: 2026-04-17
authors:
  - gianlucamazza
categories:
  - research
  - open-source
tags:
  - affective-computing
  - llm-memory
  - emotional-memory
  - affective-field-theory
---

# Why I Built emotional-memory: An Affective Field Theory for LLM Agents

Every memory system for language model agents faces the same fundamental question:
*what makes one memory more likely to surface than another?*

Most answers are pragmatic: semantic similarity, recency, importance score.
These work well for factual recall. But human memory is not primarily factual —
it is **affective**. We remember what moved us.

`emotional-memory` is my attempt to build a memory system that takes this seriously.

---

## The problem with semantic-only retrieval

Consider a user who says: *"I'm terrified about the presentation tomorrow."*

A system that retrieves memories purely by cosine similarity will surface other
mentions of "presentation" — regardless of emotional tone. It might return a
cheerful memory of a successful talk, a neutral reminder about slide formatting,
or a colleague's enthusiastic feedback.

What the user actually needs is mood-congruent retrieval: memories that resonate
with their current emotional state — past anxiety, moments of similar vulnerability,
or coping strategies from comparable situations.

Mood-congruent memory is not speculative. Gordon Bower demonstrated it empirically
in 1981: people are significantly more likely to recall material that was encoded
in a matching emotional state. Cognitive systems that ignore this leave real retrieval
quality on the table.

## Affective Field Theory — five layers

I designed **Affective Field Theory (AFT)** as a five-layer model, each grounded
in established psychology:

| Layer | Component | Basis |
|-------|-----------|-------|
| 1 | **CoreAffect** — valence × arousal | Russell 1980 circumplex |
| 2 | **AffectiveMomentum** — velocity + acceleration | Spinozist affect dynamics |
| 3 | **MoodField** — slow PAD background | Heidegger *Stimmung*, EMA |
| 4 | **AppraisalVector** — Scherer's 5 SECs | Component Process Model |
| 5 | **ResonanceLink** — associative graph | Collins & Loftus 1975 |

These are not independent modules — they interact. A high-arousal memory encodes
with a stronger consolidation weight (McGaugh 2004), decaying more slowly. When
retrieved, a high prediction-error opens a reconsolidation window (Nader & Schiller
2000) that can update the memory's affective fingerprint. Mood shifts the adaptive
weights in retrieval, making emotionally coherent memories more accessible.

## What it looks like in code

```python
from emotional_memory import EmotionalMemory, InMemoryStore, CoreAffect
from emotional_memory.embedders import SentenceTransformerEmbedder

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=SentenceTransformerEmbedder(),
)

# Set the current affective state
em.set_affect(CoreAffect(valence=-0.7, arousal=0.8))  # anxious

# Encode creates a full EmotionalTag snapshot
em.encode("I'm terrified about the presentation tomorrow.")

# Retrieval is filtered by current mood
results = em.retrieve("something difficult is coming up", top_k=3)
```

The retrieved memories are ranked by a six-signal composite score: semantic
similarity, mood congruence, affect proximity, momentum alignment, ACT-R
power-law decay, and spreading activation through the resonance graph.

## How it differs from existing systems

Systems like Mem0, Letta, and Zep solve the *persistence* problem brilliantly —
they keep large stores searchable across long sessions. What they don't provide is
an explicit emotional model. There is no notion of valence or arousal influencing
retrieval, no mood state that drifts with the conversation, no reconsolidation
triggered by affective prediction error.

`emotional-memory` is not a replacement for those systems — it is a complementary
layer. You could use `SQLiteStore` as the backend and connect it to an existing
LangChain pipeline via `EmotionalMemoryChatHistory`, adding emotional depth without
rearchitecting your stack.

## Current state

The library is at v0.5.1:

- **534 passing tests** including 126 parametrized fidelity cases validating
  20 psychological phenomena
- **mypy strict** — fully typed, zero `# type: ignore` in source code
- **Extras**: `[sentence-transformers]`, `[sqlite]`, `[langchain]`, `[viz]`
- **MIT license**, Python 3.11–3.14

A technical paper is in preparation, with a comparative benchmark against
semantic-only baselines and a literature review of 29 related systems.

## What's next

- **v0.6.0** — Qdrant/Chroma store adapters, async LangChain support
- **v0.7.0** — OpenTelemetry hooks for production observability
- **Paper** — arXiv submission with fidelity validation tables and comparative results

The repository is at [github.com/gianlucamazza/emotional-memory](https://github.com/gianlucamazza/emotional-memory).
If you build something with it, or find a psychological claim that doesn't hold up,
open an issue — I'd genuinely like to know.

---

*Gianluca Mazza — April 2026*
