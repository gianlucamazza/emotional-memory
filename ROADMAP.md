# Roadmap

This document describes planned work for upcoming releases. Items are ordered by priority within each milestone. Dates are targets, not commitments.

For already-shipped features see [CHANGELOG.md](CHANGELOG.md).

---

## v0.5.x — Stabilisation (shipped ✓)

Patch releases fixing regressions and improving developer experience. No new APIs.

- [x] Fix SQLiteStore cross-thread safety (`threading.RLock`)
- [x] `SentenceTransformerEmbedder` — first-class embedder, `[sentence-transformers]` extra
- [x] README quickstart: `pip install emotional-memory[sentence-transformers]` works out-of-the-box
- [x] `CITATION.cff` — Zenodo-ready, GitHub "Cite this repository" button
- [x] Fidelity benchmark table links to source test files
- [x] `docs/research/08_limitations.md` — documented known limits
- [x] Published to PyPI as `emotional-memory==0.5.2`
- [x] Zenodo DOI `10.5281/zenodo.19636356`
- [x] arXiv-style paper 10p (`paper/main.tex`) — 4 figures, comparative + perf tables

---

## v0.6.0 — Discovery & Integration (current, target: 2026 Q2)

Goal: make the library discoverable and integrable into existing LLM workflows.

### LangChain adapter ✓
- [x] `EmotionalMemoryChatHistory` in `src/emotional_memory/integrations/langchain.py`
- [x] Implements `BaseChatMessageHistory` — drop-in replacement in LangChain chains
- [x] `[langchain]` optional extra + CI job

### Comparative benchmark ✓
- [x] Dataset: 258 affect-labeled examples in `benchmarks/datasets/affect_reference_v1.jsonl`
- [x] Harness in `benchmarks/comparative/` — AFT vs `naive_cosine` vs `recency` baselines
- [x] `make bench-comparative` and `make reproduce-paper` targets
- [ ] Additional baselines: Mem0, Letta, LangMem adapters

### arXiv technical report ✓
- [x] 10-page paper (`paper/main.tex`) — AFT, fidelity validation, comparative benchmark
- [x] Zenodo DOI `10.5281/zenodo.19636356`, PyPI `emotional-memory==0.5.2`
- [x] `paper/arxiv-submission.tar.gz` target (`make paper-arxiv`) + `paper/SUBMISSION.md` checklist
- [ ] arXiv submission (cs.AI / cs.LG) — bundle ready, pending endorsement choice

### Docs site
- [x] mkdocs-material source in `docs/`
- [x] Tutorials: async (`docs/tutorials/async.md`), LangChain (`docs/tutorials/langchain.md`), persistence (`docs/tutorials/persistence.md`)
- [ ] Deploy to GitHub Pages (`gianlucamazza.github.io/emotional-memory`) — blocked: GitHub billing

### Classifier bump ✓
- [x] `Development Status :: 4 - Beta` in `pyproject.toml`

### HuggingFace Spaces demo
- [x] `demo/app.py` Gradio app ready
- [x] `demo/README.md` HF Space metadata complete (`python_version: "3.11"` pinned)
- [ ] Deploy to `homen3/emotional-memory-demo` (requires HF token + `huggingface-cli` push)

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
