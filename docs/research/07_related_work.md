# Related Work: Emotional Memory for LLM Agents

## Scope and Methodology

This review surveys work from 2022 to 2026 relevant to Affective Field Theory (AFT) and its implementation in the `emotional-memory` library. The survey covers four clusters: (1) general LLM memory architectures that provide the structural substrate in which AFT operates; (2) emotion- and affect-aware retrieval systems that overlap most directly with AFT's mood-congruent retrieval and resonance scoring; (3) foundational affective computing results that ground the psychological claims underlying AFT's layers; and (4) memory reconsolidation and dynamic updating mechanisms that parallel AFT's APE-gated reconsolidation engine. Papers were discovered via Hugging Face Paper Search, targeted web queries on arXiv, and ACL/AAAI/CHI proceedings. Priority was given to 2023–2026 papers; classical foundations (Russell 1980, Bower 1981, Scherer 1984, etc.) are cited in companion documents `01_foundations.md` and `06_bibliography.md` and not repeated here. The 29 papers below constitute a systematic but non-exhaustive survey; the field is growing rapidly and searches were time-boxed.

---

## 1. LLM Memory Architectures

These papers establish the general landscape of memory systems in which AFT situates itself. None implement affect-weighted encoding or mood-congruent retrieval, making them natural baselines against which AFT's additional layers can be evaluated.

### 1.1 Foundational Systems

**[P1] Generative Agents: Interactive Simulacra of Human Behavior**
Park, J.S., O'Brien, J.C., Cai, C.J., Morris, M.R., Liang, P., Bernstein, M.S. (2023). *UIST 2023 / arXiv:2304.03442*.

Twenty-five LLM-powered agents inhabit a sandbox world. Each agent maintains a *memory stream* of natural-language observations retrieved by a composite score of recency (exponential decay), importance (LLM self-rating 1–10), and semantic relevance (cosine similarity). Higher-level *reflections* are synthesized periodically. This is the closest structural predecessor to AFT: AFT replaces the scalar importance score with a six-signal retrieval vector (valence, arousal, mood congruence, momentum alignment, ACT-R decay, resonance) and adds five affective layers absent here. Park et al. provide no mechanism for reconsolidation or Hebbian link strengthening.

**[P2] MemGPT: Towards LLMs as Operating Systems**
Packer, C., Fang, V., Patil, S.G., Zhang, K., Wooders, S., Gonzalez, J.E. (2023). *arXiv:2310.08560*.

Treats the LLM context window as virtual RAM and external storage as disk. An interrupt-driven architecture moves information between tiers, enabling multi-session dialogue and large document analysis. Defines the scaffolding inside which AFT's `EmotionalMemory` engine could run as the affective scoring layer. No emotional or affective signals; no decay beyond recency.

**[P3] MemoryBank: Enhancing Large Language Models with Long-Term Memory**
Zhong, W., Guo, L., Gao, Q., Wang, Y. (2023). *AAAI 2024 / arXiv:2305.10250*.

Introduces SiliconFriend, an emotional support companion backed by MemoryBank. Memory strength is updated with an Ebbinghaus-inspired decay schedule; user personality summaries are maintained. This is the closest prior system to AFT's use of psychological forgetting models, but the Ebbinghaus schedule is additive and heuristic rather than the ACT-R power-law with arousal modulation used in AFT. No valence-arousal circumplex or appraisal vector.

**[P4] A Survey on the Memory Mechanism of Large Language Model-based Agents**
Zhang, Z., Bo, X., Ma, C., Li, R., Chen, X., Dai, Q., Zhu, J., Dong, Z. et al. (2024). *arXiv:2404.13501*.

Taxonomizes memory mechanisms (sensory, short-term, long-term, hybrid) and retrieval strategies across 50+ LLM agent systems. Useful as a reference frame: of the systems surveyed, none combine valence-arousal encoding with reconsolidation or Hebbian strengthening.

**[P5] A-MEM: Agentic Memory for LLM Agents**
Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., Zhang, Y. (2025). *arXiv:2502.12110*.

Applies the Zettelkasten note-linking method to LLM memory: each memory note carries contextual descriptions, keywords, tags, and bidirectional links that evolve over time. The link evolution is structurally analogous to AFT's `ResonanceLink` Hebbian strengthening, but the weighting is semantic (embedding similarity) rather than affective (co-retrieval count × arousal modulation).

**[P6] Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory**
Chhikara, P. et al. (2025). *arXiv:2504.19413*.

A production memory layer that dynamically extracts, consolidates, and retrieves salient information using a hybrid vector + graph store. Graph memory achieves ~2 % improvement over vector-only. Emphasizes latency and token cost over psychological fidelity. No affect model.

**[P7] On the Structural Memory of LLM Agents**
Zeng, R., Fang, J., Liu, S., Meng, Z. (2024). *arXiv:2412.15266*.

Empirical ablation over memory structure (chunks, knowledge triples, atomic facts, summaries) × retrieval strategy (single-step, rerank, iterative). Finds no single best structure; iterative retrieval consistently wins. Provides a methodology AFT could adopt to evaluate whether affective weighting modulates the structure-retrieval interaction.

**[P8] Human-inspired Perspectives: A Survey on AI Long-term Memory**
He, Z., Lin, W., Zheng, H., Zhang, F., Jones, M., Aitchison, L., Xu, X., Liu, M. et al. (2024). *arXiv:2411.00489*.

Maps human long-term memory taxonomy (episodic, semantic, procedural, emotional) onto AI systems and proposes the SALM (Self-Adaptive Long-term Memory) framework. Explicitly calls for emotional memory as a missing component in current AI agents—an observation that AFT directly addresses.

**[P9] Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers**
Du, P. (2026). *arXiv:2603.07670*.

A 2026 survey covering write–manage–read loops, consolidation, reflective improvement, and multi-session benchmarks. Notes memory consolidation as an "emerging frontier" that remains understudied. AFT's APE-gated reconsolidation directly targets this gap.

### 1.2 Biologically-Inspired and Graph Memory

**[P10] Human-like Episodic Memory for Infinite Context LLMs (EM-LLM)**
Fountas, Z., Benfeghoul, M.A., Oomerjee, A., Christopoulou, F., Lampouras, G., Bou-Ammar, H., Wang, J. (2024). *arXiv:2407.09450*.

Segments token sequences into episodic events using Bayesian surprise + graph-theoretic boundary refinement, then retrieves via similarity + temporal-contiguity. Outperforms InfLLM and RAG on LongBench. Event boundaries correlate with human-perceived event boundaries, providing a computational bridge to event cognition theory. AFT's MoodField layer could be integrated as an affective salience signal for event segmentation; AFT currently uses explicit `encode()` calls rather than continuous streaming.

**[P11] SYNAPSE: LLM Agents with Episodic-Semantic Memory via Spreading Activation**
Jiang, H., Chen, J., Pan, Y., Chen, L., You, W., Zhou, Y., Zhang, R., Zhao, L. et al. (2026). *arXiv:2601.02744*.

Builds a dynamic graph where spreading activation (Collins & Loftus 1975) propagates activation through memory nodes, with lateral inhibition and temporal decay. Triple hybrid retrieval combines embedding, keyword, and graph traversal. This is the closest architectural parallel to AFT's `ResonanceLink` spreading activation; differences are that SYNAPSE does not weight edges by affective valence/arousal, applies no Hebbian strengthening on retrieval, and has no appraisal layer.

**[P12] CraniMem: Cranial Inspired Gated and Bounded Memory for Agentic Systems**
Mody, P., Panchal, M., Kar, R., Bhowmick, K., Karani, R. (2026). *arXiv:2603.15642*.

Neurocognitive multi-stage memory with episodic buffer, knowledge graph, and consolidation loop inspired by hippocampal-neocortical consolidation. Bounded memory prevents unbounded growth. The consolidation loop is structural but not probabilistic; no prediction-error gating analogous to AFT's APE.

---

## 2. Emotion-Aware and Affect-Aware Retrieval

These papers are the most direct competitors and collaborators of AFT's mood-congruent retrieval, resonance scoring, and affective encoding pipeline.

**[P13] Emotional RAG: Enhancing Role-Playing Agents through Emotional Retrieval**
Huang, L., Lan, H., Sun, Z., Shi, C., Bai, T. (2024). *arXiv:2410.23041*.

Implements "Mood-Dependent Memory" theory in a RAG pipeline for role-playing agents: memories are retrieved based on the agent's current emotional state as well as semantic similarity. Two integration strategies (combination, sequential) are tested. This is the nearest published analog to AFT's mood-congruent retrieval (Bower 1981 signal), but lacks valence-arousal circumplex encoding, momentum, appraisal, and reconsolidation.

**[P14] LUFY: Enhancing Long-term RAG Chatbots with Psychological Models of Memory Importance and Forgetting**
Sumida, R., Inoue, K., Kawahara, T. (2024). *arXiv:2409.12524 / Dialogue & Discourse*.

Evaluates six memory importance signals (emotional arousal, surprise, LLM importance, retrieval-induced forgetting, etc.) in a RAG chatbot. Learned weights balance prioritization and forgetting; <10 % of conversation is retained. Multi-session user study (four 30-min sessions per participant) is the most rigorous human evaluation in this space. Key finding: emotional arousal is the strongest predictor of useful retention—directly validating AFT's design choice to modulate ACT-R decay by encoding arousal (McGaugh 2004).

**[P15] Dynamic Affective Memory Management for Personalized LLM Agents (DAM-LLM)**
Authors not disclosed in search results (2025). *arXiv:2510.27418*.

Bayesian-inspired probabilistic memory update treats each memory unit as a confidence distribution, minimizing global entropy as new observations arrive. Introduces DABench for evaluating emotional expression and emotional change tracking. Addresses memory redundancy and staleness via dynamic pruning—problems AFT solves via `prune(threshold)` + ACT-R decay. DAM-LLM uses Bayesian updates; AFT uses deterministic reconsolidation gated by prediction error. Neither approach has been compared empirically.

**[P16] MemEmo: Evaluating Emotion in Memory Systems of Agents**
(2026). *arXiv:2602.23944*.

Introduces HLME (Human-Like Memory Emotion) benchmark with three dimensions: emotional information extraction, emotional memory updating, and emotional memory question answering. Evaluates MemOS, memobase, and mem0—all fail to achieve robust performance across all three tasks. Directly establishes the benchmark gap that AFT is designed to fill: none of the evaluated systems combine valence-arousal encoding, appraisal, and reconsolidation.

**[P17] AnnaAgent: Dynamic Evolution Agent System with Multi-Session Memory for Realistic Seeker Simulation**
Wang, M., Wang, P., Wu, L., Yang, X., Wang, D., Feng, S., Chen, Y., Wang, B. et al. (2025). *arXiv:2506.00551*.

Counseling simulation agent with tertiary memory (short-term, long-term, emotional modulator) and complaint elicitor for tracking affect across sessions. The emotional modulator maintains a running affective state—structurally analogous to AFT's `MoodField` PAD EMA—but without explicit valence/arousal/dominance decomposition or Heidegger-inspired Stimmung theory.

**[P18] Self-evolving Agents with reflective and memory-augmented abilities**
Liang, X., He, Y., Xia, Y., Song, X., Wang, J., Tao, M., Sun, L., Yuan, X. et al. (2024). *arXiv:2409.00872*.

Integrates iterative feedback, reflective mechanisms, and memory optimization based on the Ebbinghaus forgetting curve. Closest to AFT among non-affective systems in using a psychologically grounded decay model, but the forgetting curve is applied uniformly (no arousal modulation) and there is no associative resonance network.

---

## 3. Affective Computing Foundations (2022–2026)

These papers provide or challenge the psychological theories that underpin AFT's five layers.

**[P19] Affective Computing in the Era of Large Language Models: A Survey from the NLP Perspective**
Zhang, Y., Yang, X., Xu, X., Gao, Z., Huang, Y., Mu, S., Feng, S., Wang, D. et al. (2024). *arXiv:2408.04638*.

Comprehensive survey of LLM applications to affective understanding (sentiment analysis, emotion recognition, emotion cause analysis) and affective generation (empathetic dialogue, emotional support). Maps the transition from PLMs to LLMs. Establishes that while LLMs excel at emotion *classification*, they lack grounded internal affective states that persist across interactions—the core motivation for AFT.

**[P20] Human-like Affective Cognition in Foundation Models**
Gandhi, K., Lynch, Z., Fränken, J.-P., Patterson, K., Wambu, S., Gerstenberg, T., Ong, D.C., Goodman, N.D. (2024). *arXiv:2409.11733*.

Shows that GPT-4, Claude-3, and Gemini-1.5-Pro match or exceed human performance on appraisal-theory tasks (Scherer SECs) in zero-shot settings. This supports AFT's `LLMAppraisalEngine` design choice: using an off-the-shelf LLM to compute the five SEC scores is psychologically valid. The paper does not address how appraisals should be stored and updated over time.

**[P21] An Appraisal-Based Chain-Of-Emotion Architecture for Affective Language Model Game Agents**
Croissant, M., Frister, M., Schofield, G., McCall, C. (2024). *PLOS One / arXiv:2309.05076*.

Implements a chain-of-thought appraisal step before each response: a separate LLM call evaluates emotional relevance, novelty, and goal congruence, then appends the inferred emotion to the generation context. User study in a conversational game shows significant improvement over control architectures. This is the closest existing appraisal-based system to AFT's `AppraisalVector`, but: (a) appraisals are ephemeral (not stored in memory), (b) no mood background layer, (c) no ACT-R decay or reconsolidation.

**[P22] Artificial Emotion: A Survey of Theories and Debates on Realising Emotion in Artificial Intelligence**
Li, Y., Sun, Q., Schlicher, M., Lim, Y.W., Schuller, B.W. (2025). *arXiv:2508.10286*.

Reviews 80+ years of debate on whether AI systems can have genuine emotion vs. simulation. Distinguishes emotion recognition, synthesis, and "inner emotion" (persistent internal states). AFT's valence-arousal + momentum + mood layers are positioned in the "inner emotion" paradigm; this survey provides theoretical grounding for that design choice.

**[P23] Using Large Language Models to Estimate Features of Multi-Word Expressions: Concreteness, Valence, Arousal**
Martínez, G., Molero, J.D., González, S., Conde, J., Brysbaert, M., Reviriego, P. (2024). *arXiv:2408.16012*.

Demonstrates that GPT-4o produces reliable dimensional affect ratings (valence, arousal) for multi-word expressions, outperforming prior computational methods. Provides empirical support for AFT's `KeywordAppraisalEngine` and `LLMAppraisalEngine` approaches to estimating CoreAffect from text.

**[P24] EmoLLMs: A Series of Emotional Large Language Models and Annotation Tools for Comprehensive Affective Analysis**
Liu, Z., Yang, K., Zhang, T., Xie, Q., Yu, Z., Ananiadou, S. (2024). *arXiv:2401.08508*.

Fine-tunes LLMs for multi-task affective analysis, surpassing specialized models on sentiment classification, emotion detection, and affective regression. Evaluates on AAID and AEB benchmarks. EmoLLMs could serve as drop-in `LLMAppraisalEngine` backends; AFT's architecture is model-agnostic by design.

**[P25] EICAP: Deep Dive in Assessment and Enhancement of LLMs in Emotional Intelligence through Multi-Turn Conversations**
Nazar, N., Asgari, E. (2025). *arXiv:2508.06196*.

Benchmark for LLM emotional intelligence with four layers: emotional tracking, cause inference, appraisal, and emotionally appropriate response. Fine-tuning on UltraChat improves appraisal. Finding that fine-tuning improves only the appraisal layer—not emotional tracking or cause inference—supports AFT's decision to separate the appraisal engine from the memory encoding pipeline.

---

## 4. Reconsolidation and Memory Updating

These papers address the dynamic nature of stored memories—the mechanism most differentiated in AFT via APE-gated reconsolidation.

**[P26] FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory**
Wei, L., Peng, X., Dong, X., Xie, N., Wang, B. (2026). *arXiv:2601.18642*.

Dual-layer memory with adaptive exponential decay rates that factor in semantic relevance, access frequency, and temporal patterns. Conflict-aware fusion prevents contradictory memories from coexisting. Closest published system to AFT's `DecayEngine`, but: (a) decay is exponential rather than ACT-R power-law, (b) no arousal-floor for high-salience memories, (c) no prediction-error trigger for reconsolidation windows.

**[P27] "My agent understands me better": Integrating Dynamic Human-like Memory Recall and Consolidation in LLM-Based Agents**
(2024). *arXiv:2404.00573*.

Proposes a consolidation loop where new experiences retroactively refine existing memory graph nodes—enabling associative learning. Structurally analogous to AFT's reconsolidation pathway but driven by semantic overlap rather than prediction error (APE). No affective signals.

**[P28] Human-Like Remembering and Forgetting in LLM Agents: An ACT-R-Inspired Memory Architecture**
(2025). *ACM HAI 2025 / dl.acm.org/doi/10.1145/3765766.3765803*.

Integrates the ACT-R activation formula directly into the LLM generation process, making memory recall and forgetting transparent to the generation itself rather than treating ACT-R as a downstream component. This is the most rigorous existing ACT-R implementation for LLM agents. AFT differs in adding affective modulation (arousal-based decay floor, McGaugh 2004) and coupling decay to a Hebbian resonance network.

**[P29] In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents**
Tan, Z., Yan, J., Hsu, I.-H., Han, R., Wang, Z., Le, L.T., Song, Y., Chen, Y. et al. (2025). *arXiv:2503.08026*.

Prospective reflection (forward-looking) and retrospective reflection (backward-looking) update stored memories using online RL on LongMemEval. Achieves better temporal consistency than retrieval-only approaches. The reflection-as-update mechanism is functionally analogous to AFT's `elaborate()` dual-path elaboration (slow-path appraisal blending), though the trigger is conversation-turn-based rather than prediction-error-gated.

---

## 5. Summary Comparison Table

| Author(s) | Year | Title (abbreviated) | Venue | Memory Model | Affect Model | Reconsolidation | Open Source | Delta vs AFT |
|---|---|---|---|---|---|---|---|---|
| Park et al. | 2023 | Generative Agents | UIST / arXiv:2304.03442 | Stream + reflection + planning | None | No | Yes | No valence-arousal, no appraisal, no mood field, no Hebbian links |
| Packer et al. | 2023 | MemGPT | arXiv:2310.08560 | OS-style tiered context | None | No | Yes | Pure architectural; no emotional signals |
| Zhong et al. | 2023 | MemoryBank / SiliconFriend | AAAI 2024 / arXiv:2305.10250 | Ebbinghaus decay, personality summary | Empathy prompts only | No | Yes | Heuristic decay; no circumplex, no appraisal, no reconsolidation |
| Zhang et al. | 2024 | Survey on Memory Mechanisms | arXiv:2404.13501 | Taxonomy (sensory/STM/LTM) | None | No | N/A | Survey; no affective system |
| Xu et al. | 2025 | A-MEM | arXiv:2502.12110 | Zettelkasten note-graph | None | No | Yes | Semantic links only; no affective weighting |
| Chhikara et al. | 2025 | Mem0 | arXiv:2504.19413 | Vector + graph hybrid | None | No | Yes | Production focus; no affect model |
| Zeng et al. | 2024 | Structural Memory of LLM Agents | arXiv:2412.15266 | Chunks / triples / facts / summaries | None | No | No | Structural ablation; no emotional signals |
| He et al. | 2024 | Survey: Human-Inspired AI LTM | arXiv:2411.00489 | SALM taxonomy | Calls for emotional memory | No | N/A | Identifies gap AFT fills |
| Du | 2026 | Memory for Autonomous LLM Agents | arXiv:2603.07670 | Write-manage-read survey | None | Notes gap | N/A | Survey; reconsolidation identified as frontier |
| Fountas et al. | 2024 | EM-LLM (Episodic Memory) | arXiv:2407.09450 | Event-segmented episodic memory | None (Bayesian surprise as salience) | No | Yes | No affective dimensions; segmentation vs encode |
| Jiang et al. | 2026 | SYNAPSE (Spreading Activation) | arXiv:2601.02744 | Dynamic graph + spreading activation | None | No | No | Closest to ResonanceLink; no affective edge weights |
| Mody et al. | 2026 | CraniMem | arXiv:2603.15642 | Gated multi-stage + KG consolidation | None | Structural only | No | No prediction-error gating; no affect |
| Huang et al. | 2024 | Emotional RAG | arXiv:2410.23041 | Emotion-filtered RAG | Emotion state label | No | No | Nearest mood-congruent retrieval; no circumplex, no appraisal, no decay |
| Sumida et al. | 2024 | LUFY | arXiv:2409.12524 | Arousal-prioritized RAG | Arousal + surprise scoring | No | Yes | Validates arousal-decay coupling; no full circumplex or reconsolidation |
| (Anon.) | 2025 | DAM-LLM | arXiv:2510.27418 | Bayesian probabilistic memory | Affective confidence distributions | Bayesian update | No | Probabilistic vs. deterministic APE; no appraisal vector |
| (Anon.) | 2026 | MemEmo | arXiv:2602.23944 | Evaluates existing systems | HLME benchmark | No | Partial | Benchmark shows gap AFT addresses; no new architecture |
| Wang et al. | 2025 | AnnaAgent | arXiv:2506.00551 | Tertiary memory + emotional modulator | Running affective state | No | No | PAD-like modulator but no theory grounding; no reconsolidation |
| Liang et al. | 2024 | Self-evolving Reflective Agents | arXiv:2409.00872 | Ebbinghaus + reflection | None | No | No | Uniform decay; no arousal modulation |
| Zhang et al. | 2024 | Affective Computing + LLMs Survey | arXiv:2408.04638 | Classification / generation survey | Emotion recognition/generation | No | N/A | Survey; establishes lack of persistent internal affect states |
| Gandhi et al. | 2024 | Human-like Affective Cognition | arXiv:2409.11733 | N/A | Appraisal evaluation (Scherer SECs) | No | No | Validates LLM appraisal quality; no memory integration |
| Croissant et al. | 2024 | Chain-of-Emotion Architecture | PLOS One / arXiv:2309.05076 | Conversation history only | Appraisal-before-response (ephemeral) | No | No | Ephemeral appraisal; AFT persists appraisals in memory |
| Li et al. | 2025 | Artificial Emotion Survey | arXiv:2508.10286 | N/A | Inner emotion theory review | No | N/A | Theoretical grounding for AFT's inner-affect design |
| Martínez et al. | 2024 | LLMs Estimate Valence, Arousal | arXiv:2408.16012 | N/A | Dimensional affect ratings | No | No | Validates LLM-based CoreAffect estimation |
| Liu et al. | 2024 | EmoLLMs | arXiv:2401.08508 | N/A | Multi-task affective fine-tuning | No | Yes | Backend for AppraisalEngine; no memory |
| Nazar & Asgari | 2025 | EICAP Benchmark | arXiv:2508.06196 | N/A | 4-layer EI benchmark | No | Partial | Fine-tuning improves appraisal only; supports AFT separation |
| Wei et al. | 2026 | FadeMem | arXiv:2601.18642 | Dual-layer exponential decay | None | No | No | Exponential decay vs. ACT-R power-law; no reconsolidation gating |
| (Anon.) | 2024 | Dynamic Consolidation in LLM Agents | arXiv:2404.00573 | Semantic consolidation loop | None | Semantic-trigger | No | No prediction error; no affective signals |
| (Anon.) | 2025 | ACT-R-Inspired LLM Memory (HAI) | ACM HAI 2025 | ACT-R activation inline | None | No | No | Most rigorous ACT-R; AFT adds arousal modulation + resonance |
| Tan et al. | 2025 | Prospective/Retrospective Reflection | arXiv:2503.08026 | Reflective online RL | None | Turn-triggered reflection | No | No prediction-error gate; functionally similar to elaborate() |

---

## 6. Key Observations

**What AFT adds beyond the state of the art:**

1. **Five-layer architecture**: No existing system combines all of (a) valence-arousal CoreAffect, (b) AffectiveMomentum (velocity + acceleration), (c) Heidegger-PAD MoodField, (d) Scherer appraisal vector, and (e) Hebbian resonance graph in a single memory engine.

2. **Mood-congruent retrieval with circumplex geometry**: Emotional RAG [P13] and LUFY [P14] implement emotion-state-weighted retrieval, but neither uses the Russell (1980) valence-arousal circumplex as the encoding space. AFT's mood congruence signal is computed as the cosine distance in circumplex space, providing a theoretically grounded metric.

3. **APE-gated reconsolidation**: No published LLM memory system implements prediction-error-gated reconsolidation windows (Nader 2000; Pearce-Hall 1980). FadeMem [P26] and DAM-LLM [P15] update memories dynamically but without a biologically motivated trigger. The ACT-R system [P28] uses activation thresholds, not prediction error.

4. **Arousal-modulated decay floor**: LUFY [P14] demonstrates empirically that arousal is the strongest predictor of long-term memory retention; AFT translates this into a formal ACT-R decay floor based on encoding arousal, with support from McGaugh (2004). No other system reviewed implements this mechanism.

5. **Dual-path LeDoux encoding**: The fast/slow dual-path model (LeDoux 1996)—where immediate encoding bypasses appraisal and elaboration occurs asynchronously—is not present in any reviewed system. This enables real-time encoding under latency constraints while maintaining psychological fidelity.

**Limitations compared to reviewed work:**

- AFT does not yet have a formal benchmark comparable to HLME [P16] or LUFY's user study [P14]. A comparative evaluation against Emotional RAG, LUFY, and DAM-LLM on affective memory tasks would strengthen empirical claims.
- AFT's `SQLiteStore` uses sqlite-vec ANN search; recent work such as EM-LLM [P10] and SYNAPSE [P11] show that graph-based traversal adds recall precision for multi-hop associative queries.
- The async architecture in `AsyncEmotionalMemory` has not been benchmarked at the scale of Mem0 [P6], which reports 91% latency reduction; production-scale validation remains future work.

---

*Generated: 2026-04-17. Sources include arXiv, ACL Anthology, AAAI proceedings, PLOS One, ACM DL, and Hugging Face Paper Search.*
