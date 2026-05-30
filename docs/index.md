# emotional_memory

[![CI](https://github.com/gianlucamazza/emotional-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/gianlucamazza/emotional-memory/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/emotional_memory)](https://pypi.org/project/emotional_memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gianlucamazza/emotional-memory/blob/main/LICENSE)

Emotional memory for LLMs based on **Affective Field Theory (AFT)** — a 5-layer model that
encodes not just *what* happened, but *how it felt*, *how that feeling was moving*, and
*what mood colored the moment*.

<!--
Positioning, comparison table, and 30-second example are sourced from the
top-level README.md (SSOT). To edit, change the README between the
`ssot:positioning-start` / `ssot:positioning-end` markers; the docs site
will pick the change up on next build.
-->
{%
  include-markdown "../README.md"
  start="<!-- ssot:positioning-start -->"
  end="<!-- ssot:positioning-end -->"
  rewrite-relative-urls=true
%}

## Installation & Quickstart

See **[Getting Started](getting-started.md)** for install recipes (extras, dev setup) and a
runnable quickstart — both sourced from the README so they stay in sync with PyPI.

## The 5 Layers

| Layer | Class | Theory |
|---|---|---|
| **CoreAffect** | `CoreAffect` | Russell-Mehrabian PAD model — continuous (valence, arousal, dominance) |
| **AffectiveMomentum** | `AffectiveMomentum` | Spinoza — affect as transition (velocity + acceleration) |
| **MoodField** | `MoodField` | Heidegger §29 — slow-moving global mood with inertia |
| **AppraisalVector** | `AppraisalVector` | Scherer CPM — emotion from evaluation (5 dimensions) |
| **ResonanceLinks** | `ResonanceLink` | Aristotle/Bower/Collins & Loftus/Hebb — bidirectional associative graph (5 link types), multi-hop spreading activation + Hebbian co-retrieval strengthening |

See [Research](research/index.md) for full theoretical foundations.

## Tutorials

- [Async usage](tutorials/async.md) — `AsyncEmotionalMemory`, `as_async()`, `encode_batch()`
- [Persistence](tutorials/persistence.md) — `SQLiteStore`, `save_state`, `export_memories`, `prune()`
- [LangChain integration](tutorials/langchain.md) — `EmotionalMemoryChatHistory`, `RunnableWithMessageHistory`

## Research status

- [Current Evidence](research/09_current_evidence.md) — what is validated today,
  what is only early controlled evidence, and what still needs stronger studies
- [Limitations](research/08_limitations.md) — current empirical and architectural gaps
