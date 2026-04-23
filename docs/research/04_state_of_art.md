# State of the Art: LLM Memory Systems and Affective Computing

> Analysis of existing systems, focusing on gaps in emotional memory implementation. This document
> motivates the positioning of our library within the current landscape.

---

## 1. LLM Memory Systems

### 1.1 MemGPT / Letta (Packer et al., 2023)

**Paper**: "MemGPT: Towards LLMs as Operating Systems" (arXiv:2310.08560)
**GitHub**: https://github.com/cpacker/MemGPT → became https://research.memgpt.ai/

**Architecture — Three OS-inspired tiers**:
- **Core Memory** (always in-context, like RAM): system persona, key user information
- **Recall Memory** (searchable conversational history): vector database of past conversations
- **Archival Memory** (long-term storage, like disk): unboundedly expandable memory

**Key innovation**: The LLM itself decides what to store, summarize, and forget via tool calls —
self-management of memory.

**Emotional gap**: No emotional representation. The importance of a memory is determined by the
LLM in an opaque way, without a structured affective dimension. There is no differentiated decay
based on emotional features.

---

### 1.2 Mem0 (2024)

**Paper**: arXiv:2504.19413
**GitHub**: https://github.com/mem0ai/mem0 (~48K stars, $24M funding)

**Approach**: Universal memory layer for AI agents. Dynamically extracts salient information from
conversations, consolidates it, and retrieves it.

**Metrics**: 26% higher accuracy, 91% lower latency, 90% token savings compared to full-context
approaches.

**Emotional gap**: Extracts facts and preferences, not emotional states. Does not distinguish
between "the user prefers coffee" and "the user was distressed when talking about their work."

---

### 1.3 Zep (2024)

**GitHub**: https://github.com/getzep/zep

**Approach**: Temporal knowledge graph that connects past interactions, structured datasets, and
context changes. The Graphiti engine combines episodic memory (chat), semantic memory (entities),
and group-level subgraphs.

**Emotional gap**: The knowledge graph captures semantic relationships but not affective
dimensions. There is no node or edge representing the emotional state associated with an
interaction.

---

### 1.4 Generative Agents (Park et al., 2023)

**Paper**: "Generative Agents: Interactive Simulacra of Human Behavior" (arXiv:2304.03442)
**Citations**: ~3000+ — foundational paper for the field

**Retrieval formula**:
```
score = recency(t) + importance(m) + relevance(q, m)
```
- `recency`: exponential decay from the time of last access
- `importance`: LLM evaluation on a 1–10 scale ("how important is this memory to the character?")
- `relevance`: cosine similarity between query embedding and memory embedding

**Emotional gap**: `importance` is an opaque proxy that may capture emotional relevance, but:
1. It is unstructured — does not distinguish valence, arousal, emotion type
2. It does not modulate decay — a very important memory decays at the same rate as a trivial one
3. There is no mood-congruent retrieval — the character's mood does not weight retrieval
4. There is no reconsolidation

This paper defined the de facto standard that most subsequent systems reproduce with minor
variations.

---

### 1.5 OpenMemory / Cavira (2025)

**GitHub**: https://github.com/CaviraOSS/OpenMemory
**Docs**: https://openmemory.cavira.app/

**Architecture — 5 sectors with differentiated decay**:
1. **Semantic** (facts, concepts): standard decay
2. **Episodic** (specific events): moderate decay
3. **Procedural** (procedures, skills): slow decay (habits persist)
4. **Emotional** (emotional states, affective experiences): slow decay with minimum floor
5. **Reflective** (self-assessments, insights): very slow decay

**Key innovation**: The first system to include an **explicit emotional sector** with
differentiated decay. "Reinforcement pulses" raise critical memories above the retention
threshold.

**Gap**: Emotion as a separate sector, not as a pervasive dimension. The internal emotional
representation is not documented in detail. No appraisal engine.

---

### 1.6 Memory Bear AI (2025–2026)

**Paper**: arXiv:2512.20651 (Dec 2025) + Technical Report arXiv:2603.22306 (Mar 2026)

**Innovation**: **Emotion Memory Units (EMUs)** — the first system to treat emotion as a
**native dimension** of memory, not as a separate sector or post-hoc label.

Each EMU combines:
- Semantic content
- Emotional state at encoding time
- Subsequent emotional impact traces

**Relevance**: The EMU approach is the closest to our vision. But it remains limited — it lacks
the multi-layer structure (momentum, mood field, vectorial appraisal) and the multi-signal
retrieval with adaptive weights.

---

### 1.7 Emotional RAG (Huang et al., 2024)

**Paper**: "Emotional RAG: Enhancing Role-Playing Agents through Emotional Retrieval"
(arXiv:2410.23041)

**Theoretical foundation**: Bower's Mood-Dependent Memory Theory (1981)

**Two retrieval strategies**:
1. **Combination**: joint weighting of semantic similarity + emotional state
2. **Sequential**: filtering by emotion then semantic ranking (or vice versa)

Outperforms standard RAG on InCharacter, CharacterEval, Character-LLM datasets.

**Gap**: Tested only in role-playing scenarios. No long-term memory management. Emotion is a
single signal, not a multi-layer field.

---

### 1.8 A-Mem / Agentic Memory (2025)

**Paper**: "A-MEM: Agentic Memory for LLM Agents" (NeurIPS 2025, arXiv:2502.12110)
**GitHub**: https://github.com/WujiangXu/A-mem

**Approach**: The LLM agent autonomously manages its own memory operations (what to remember,
how to structure it, when to forget). Memory as a self-organizing adaptive system.

**Emotional gap**: Focus on self-management, not affective dimensions.

---

### 1.9 MemOS — Memory Operating System (2025)

**GitHub**: https://github.com/MemTensor/MemOS

**Approach**: Operating system for LLM memory with a knowledge base, memory feedback,
multi-modal memory, and memory tools for agent planning.

**Emotional gap**: Systematic structure without an affective component.

---

## 2. MemEmo Benchmark: The Definitive Gap (Feb 2026)

**Paper**: "MemEmo: A Benchmark for Evaluating Emotional Memory in LLM Agents" (arXiv:2602.23944)

**Key result**: **No current memory system correctly handles emotion** across all phases:
extraction, updating, and question-answering.

**Dataset**: HLME (Human-Like Memory with Emotion) — conversations with explicit and implicit
emotional states, state changes, and queries requiring emotional reasoning over memory.

**Evaluated tasks**:
1. **Emotion Extraction**: Identify the emotional state in a conversation
2. **Emotion Update**: Update memory when emotion changes
3. **Emotion QA**: Answer questions requiring reasoning over emotional memory

**Systems tested**: All major systems (Mem0, Zep, MemGPT/Letta, others) fail systematically,
especially on emotional memory updating and reasoning.

This paper provides strong motivation for a library that treats emotion as a structural part of
memory, but does not by itself constitute a validation of our implementation.

---

## 3. Affective Computing: Contemporary Landscape

### Multimodal Fusion

The dominant trend (2022–2026) combines:
- Facial expression
- Speech prosody
- Text
- Physiological signals (EDA, heart rate, EEG)
- Eye tracking

Cross-modal transformers (e.g. Joint-Dimension-Aware Transformer) dominate. Accuracy > 90% in
some paradigms.

**Relevance**: Our Layer 4 (Appraisal) can potentially accept multimodal input beyond text.

### Emotional Support Conversation (ESC)

LLM-based systems for emotional support: they detect the user's emotional state and generate
empathetic dialogue with validation, normalization, and reframing.

**Gap**: These systems detect and respond to emotion in the moment, but do not store it
structurally to inform future interactions.

### EmoLLMs (Liu et al., 2024)

**Paper**: arXiv:2401.08508
First series of open-source LLMs for comprehensive affective analysis. Includes AAID (234K
samples) and AEB benchmark (14 tasks). Outperforms ChatGPT/GPT-4 on most tasks.

### EmotionPrompt & NegativePrompt

**EmotionPrompt** (Li et al., 2023, arXiv:2307.11760): Emotional stimuli in prompts produce
+8% on Instruction Induction, +115% on BIG-Bench.

**NegativePrompt** (Wang et al., 2024, arXiv:2405.02814): Negative emotional stimuli: +12.89%
and +46.25% respectively.

**Implication**: Emotional context significantly modifies LLM behavior. A library that maintains
and injects emotional context into system memory can systematically improve response quality.

### Latent Emotional Representations in LLMs (Zhang & Zhong, 2025)

**Paper**: "Decoding Emotion in the Deep" (arXiv:2510.04064)

**Key findings**:
- LLMs develop a **well-defined internal geometry** of emotion (structured latent representations)
- The emotional signal **peaks at mid-network layers**
- The initial emotional tone **persists for hundreds of tokens**

**Revolutionary implication**: LLMs already represent emotion internally in a structured way. An
emotional memory library can hook into these latent representations rather than building everything
from the outside.

---

## 4. Gap Analysis and Positioning

### Gaps identified in the current landscape

| Gap | Impact | Our response |
|-----|--------|--------------|
| Emotion as a separate sector, not a pervasive dimension | Every system misses the affective impact on encoding/retrieval | Pervasive emotional tagging on every memory |
| Static representation (point in space) | Does not capture dynamics — where you are going | Layer 2: Affective Momentum (derivatives) |
| No global mood field | Current mood does not weight retrieval in a structured way | Layer 3: MoodField |
| No integrated appraisal | Emotion is assumed as input, not generated | Layer 4: Appraisal Vector |
| Single-signal retrieval | Semantics only, or semantics + one static emotional signal | Multi-signal retrieval with 6 components and adaptive weights |
| No reconsolidation | Retrieval cannot update the emotional tag | Post-retrieval reconsolidation window |
| Uniform decay | Emotionally intense memories are not systematically privileged | Non-linear decay modulated by arousal |

### Our positioning

```
                    Breadth of emotional coverage
                    (how many aspects of emotion are covered)
                              Low        High
                         ┌─────────────────────┐
                    High  │              ★ Our  │
Depth of                 │              library │
integration              │  Memory Bear AI      │
(how pervasive emotion   │  OpenMemory          │
is within the system)    │                      │
                    Low   │ MemGPT, Mem0, Zep   │
                         └─────────────────────┘
```

In the landscape analyzed here, AFT occupies a relatively rare position: **high depth of affective
integration** (emotion weights both encoding and retrieval) and **multi-layer coverage** (core
affect + momentum + mood field + appraisal + resonance). This map is interpretive, not proof of
exclusivity: systems such as Memory Bear AI and OpenMemory cover important subsets of the same
design space with different abstractions.

---

## Bibliographic Notes

- Packer, C. et al. (2023). "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560.
- Mem0 paper. arXiv:2504.19413.
- Park, J.S. et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior."
  arXiv:2304.03442.
- OpenMemory / Cavira: https://github.com/CaviraOSS/OpenMemory
- Memory Bear AI: arXiv:2512.20651, arXiv:2603.22306.
- Huang et al. (2024). "Emotional RAG." arXiv:2410.23041.
- Xu, W. et al. (2025). "A-MEM: Agentic Memory for LLM Agents." NeurIPS 2025. arXiv:2502.12110.
- MemEmo (2026). arXiv:2602.23944.
- Liu, Z. et al. (2024). "EmoLLMs: A Series of Emotional Large Language Models."
  arXiv:2401.08508.
- Li, C. et al. (2023). "EmotionPrompt: Elevating LLM Performance Through Emotional
  Intelligence." arXiv:2307.11760.
- Wang et al. (2024). "NegativePrompt." arXiv:2405.02814.
- Zhang & Zhong (2025). "Decoding Emotion in the Deep." arXiv:2510.04064.
- Dynamic Affective Memory Management (2025). arXiv:2510.27418.
