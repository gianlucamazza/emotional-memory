# Comparison with Existing Systems

How `emotional_memory` positions itself against general-purpose LLM memory
systems. For a synthetic one-line summary see the [Home page](index.md); for
the underlying numbers see [Current Evidence](research/09_current_evidence.md).

| | **emotional-memory (AFT)** | **Mem0** | **Letta** | **Zep** | **LangChain Memory** |
|---|---|---|---|---|---|
| **License** | MIT | Apache 2.0 | Apache 2.0 | Apache 2.0 | MIT |
| **Persistence** | InMemory / SQLite / Qdrant / Chroma | Qdrant, Chroma, Pinecone, PG, MongoDB | PostgreSQL / SQLite | Neo4j (self-hosted) / Cloud | In-memory / custom |
| **BYO embedder** | ✅ any `Embedder` protocol | ✅ (OpenAI default) | ⚠️ partial | ⚠️ partial | ✅ |
| **Emotion model** | ✅ 5-layer AFT (valence, arousal, dominance, mood, appraisal, resonance) | ❌ | ❌ | ❌ | ❌ |
| **Reconsolidation** | ✅ APE-gated lability window | ✅ auto update/remove | ✅ tool-call edit | ✅ edge invalidation | ❌ |
| **Persistent mood state** | ✅ MoodField (Heidegger EMA) | ❌ | ❌ | ❌ | ❌ |
| **LLM-agnostic** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **LangChain integration** | ✅ `EmotionalMemoryChatHistory` | ✅ official | ✅ tools interop | ✅ ZepVectorStore | ✅ native |
| **Internal fidelity tests** | ✅ 127 cases, 20 phenomena ([bench-fidelity](https://github.com/gianlucamazza/emotional-memory/tree/main/benchmarks/fidelity)) | — | — | — | — |
| **External benchmark** | ✅ LoCoMo (FAIL: F1 0.168 vs 0.271; Pareto Hj1 FAIL) | ✅ LoCoMo, LongMemEval, BEAM | ✅ LoCoMo, DMR | ✅ DMR, LongMemEval | ❌ |
| **Codebase size** | ~4.8k LOC (src/) | >50k LOC | >50k LOC | >50k LOC | ~5k LOC |

**Key differentiator**: emotional-memory makes affect a first-class, multi-layer part
of encoding and retrieval. Compared with the general-purpose memory systems in this
table, it emphasizes mood-congruent retrieval, appraisal-conditioned tagging, and
APE-gated reconsolidation rather than generic conversational recall alone.

**Benchmark caveat**: AFT fidelity tests validate psychological invariants
(intra-theory). The comparative benchmark in this repo is a controlled synthetic
retrieval probe, not a general downstream evaluation of production memory systems.

## See also

- [Current Evidence](research/09_current_evidence.md) — the study ladder and claim-to-evidence matrix
- [Benchmarks](benchmarks.md) — fidelity, performance, and appraisal-quality suites
- [Related Work](research/07_related_work.md) — full 29-system comparative positioning
