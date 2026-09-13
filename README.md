# emotional_memory

[![CI](https://github.com/gianlucamazza/emotional-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/gianlucamazza/emotional-memory/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/gianlucamazza/emotional-memory/graph/badge.svg)](https://codecov.io/gh/gianlucamazza/emotional-memory)
[![PyPI](https://img.shields.io/pypi/v/emotional_memory)](https://pypi.org/project/emotional_memory/)
[![Last commit](https://img.shields.io/github/last-commit/gianlucamazza/emotional-memory)](https://github.com/gianlucamazza/emotional-memory/commits/main)
[![Python](https://img.shields.io/pypi/pyversions/emotional_memory)](https://pypi.org/project/emotional_memory/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19972258-blue?logo=doi)](https://doi.org/10.5281/zenodo.19972258)
[![Benchmarks](https://img.shields.io/badge/benchmarks-tracked-blue)](https://gianlucamazza.github.io/emotional-memory/dev/bench/)
[![SLSA 3](https://slsa.dev/images/gh-badge-level3.svg)](https://github.com/gianlucamazza/emotional-memory/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Emotional memory for LLMs based on **Affective Field Theory (AFT)** — a 5-layer model that encodes not just _what_ happened, but _how it felt_, _how that feeling was moving_, and _what mood colored the moment_.

Pre-registered evaluation on `realistic_recall_v2`: English (N=200, SBERT Δ=+0.21, d=0.49) and French (N=120, me5, Δ=+0.18, p<0.0001, Hedges g=0.42 — Addendum M Branch A PASS). Italian/Spanish me5 at declared power (N=120) FAIL; English-SBERT and SBERT-Spanish (N=80) hold. External-QA evaluation (LoCoMo), naturalistic dialogue (DailyDialog), and both released third-party emotional corpora (MADial-Bench, Addendum X — counter-congruent supportive recall; ES-MemEval, Addendum X2 — affect-orthogonal QA gold) FAIL — the AFT advantage is regime-specific to affect-discriminative, mood-congruent recall, not general superiority. Full [claim-validation matrix](https://github.com/gianlucamazza/emotional-memory/blob/main/docs/research/claim_validation_matrix.json).

<!-- ssot:positioning-start -->

## Why emotional_memory?

Most LLM memory libraries treat retrieval as semantic-only: vector similarity over text. Real human recall is driven by more:

- **Affective congruence** — we remember things that feel like how we feel now (Bower 1981)
- **Arousal-modulated consolidation** — emotionally-charged events consolidate more strongly (Cahill & McGaugh 1995; ACT-R power-law with arousal floor, McGaugh 2004)
- **Reconsolidation** — retrieved memories become labile and update with prediction error (Nader & Schiller 2000; APE-gated lability window)
- **Dual-path encoding** — fast affective signal precedes slow appraisal (LeDoux 1996)
- **3D affect** — perceived control (dominance) discriminates fear from anger (Mehrabian & Russell 1974; PAD)

`emotional_memory` operationalizes these as a single retrieval pipeline. Validated against 20 published psychological phenomena (127 fidelity tests) and 25+ pre-registered confirmatory studies — including [committed negative results](https://github.com/gianlucamazza/emotional-memory/blob/main/docs/research/claim_validation_matrix.json).

### How it compares

| Library                         | Memory model                                                           | Affective retrieval           | Reconsolidation          | Decay model                          | Psychological fidelity tests |
| ------------------------------- | ---------------------------------------------------------------------- | ----------------------------- | ------------------------ | ------------------------------------ | ---------------------------- |
| **emotional_memory**            | 5-layer AFT (semantic + valence/arousal + momentum + mood + appraisal) | ✅ mood-congruent + APE-gated | ✅ Nader & Schiller 2000 | ACT-R power-law + arousal modulation | 127 tests, 20 phenomena      |
| MemGPT / Letta                  | Hierarchical context (working + archival)                              | ❌                            | ❌                       | None                                 | —                            |
| mem0                            | Fact extraction + vector store                                         | ❌                            | ❌                       | None                                 | —                            |
| A-MEM                           | Atomic notes + dynamic links                                           | ❌                            | ❌                       | None                                 | —                            |
| LangMem                         | Hot/cold memory tiers                                                  | ❌                            | ❌                       | Time-based eviction                  | —                            |
| Generative Agents (Park et al.) | Importance + recency + relevance                                       | Partial (importance only)     | ❌                       | Exponential                          | —                            |
