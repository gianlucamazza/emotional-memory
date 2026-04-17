# Roadmap

This document describes planned work for upcoming releases. Items are ordered by priority within each milestone. Dates are targets, not commitments.

For already-shipped features see [CHANGELOG.md](CHANGELOG.md).

---

## v0.5.x — Stabilisation (current)

Patch releases fixing regressions and improving developer experience. No new APIs.

- [x] Fix SQLiteStore cross-thread safety (`threading.RLock`)
- [x] `SentenceTransformerEmbedder` — first-class embedder, `[sentence-transformers]` extra
- [x] README quickstart: `pip install emotional-memory[sentence-transformers]` works out-of-the-box
- [x] `CITATION.cff` — Zenodo-ready, GitHub "Cite this repository" button
- [x] Fidelity benchmark table links to source test files
- [x] `docs/research/08_limitations.md` — documented known limits
- [ ] Publish to PyPI (blocked: GitHub billing + Trusted Publisher setup)
- [ ] Zenodo DOI (blocked: depends on PyPI publish)
- [ ] Merge 9 pending Dependabot PRs (blocked: GitHub billing)

---

## v0.6.0 — Discovery & Integration (target: 2026 Q2)

Goal: make the library discoverable and integrable into existing LLM workflows.

### LangChain adapter
- `EmotionalMemoryChatHistory` in `src/emotional_memory/integrations/langchain.py`
- Implements `BaseChatMessageHistory` — drop-in replacement in LangChain chains
- `[langchain]` optional extra
- CI job for the new extra

### Comparative benchmark
- Dataset: 200-500 affect-labeled examples in `benchmarks/datasets/affect_reference_v1.jsonl`
- Harness in `benchmarks/comparative/` benchmarking AFT vs Mem0 / Letta / Zep on mood-congruent retrieval recall@k
- `make bench-comparative` and `make reproduce-paper` targets

### arXiv technical report
- 10-12 page paper describing AFT, fidelity validation, and comparative results
- Submitted to arXiv cs.AI (software track or workshop)
- `paper/` directory with LaTeX source

### Docs site
- mkdocs-material site deployed on GitHub Pages (`gianlucamazza.github.io/emotional-memory`)
- API reference auto-generated via mkdocstrings
- Tutorials: basic usage, async, LangChain integration, persistence

### Classifier bump
- `Development Status :: 4 - Beta` in `pyproject.toml`

---

## v0.7.0 — Production Readiness (target: 2026 Q3)

Goal: make the library production-grade for teams running agents at scale.

### Enterprise vector stores
- `QdrantStore` — adapter for Qdrant (distributed, production-proven)
- `ChromaStore` — adapter for Chroma (local and managed)
- `[qdrant]` and `[chroma]` optional extras

### Observability
- Optional OpenTelemetry spans on `encode`, `retrieve`, `elaborate`
- Structured logging on all pipeline events (already uses `logging.DEBUG`)
- `[otel]` optional extra

### BYO appraisal schema
- Parameterize the Scherer CPM prompt so custom appraisal taxonomies (OCC, GRID) can be injected without forking
- `AppraisalSchema` config class

### HuggingFace Spaces demo
- Interactive Gradio demo: chat with an LLM that remembers your mood
- PAD visualization panel in real-time
- Hosted at `homen3/emotional-memory-demo` (or similar)

---

## v0.8.0 — Scientific Validation (target: 2026 Q4)

Goal: close the gap between intra-theory validation (current) and ecological validation.

### Human evaluation
- 5-10 conversational scenarios, 20-30 annotators (Prolific/MTurk)
- Metric: "Is this retrieved memory relevant?" (binary + free text)
- Results published alongside the arXiv paper or as a follow-up

### Distributed affective state
- `AffectiveStateBackend` protocol for shared state across agent instances
- Reference implementation: `RedisAffectiveStateBackend`
- Configurable conflict resolution (last-write-wins vs. merge)

### PAD dominance as primary dimension
- Promote `dominance` from optional appraisal output to first-class `CoreAffect` field
- Migration path for serialized states

### Multilingual validation
- Fidelity benchmark coverage for at least one non-English language (Spanish or French)
- Document language-specific limitations

---

## Contributing

Want to work on something on this roadmap? Open an issue first to discuss scope and approach. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions.

Items not on this roadmap but worth discussing:
- Persistent memory compression / summarisation
- Cross-agent emotional resonance (shared mood fields)
- Integration with more LLM frameworks (LlamaIndex, CrewAI, AutoGen)
- Real-time streaming encode (partial affective updates)
