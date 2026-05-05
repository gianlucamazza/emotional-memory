# Roadmap

This document describes planned work for upcoming releases. Items are ordered by priority within each milestone. Dates are targets, not commitments.

For already-shipped features see [CHANGELOG.md](CHANGELOG.md).

---

## v0.5.x — Stabilisation (shipped ✓)

Patch releases fixing regressions and improving developer experience. No new APIs.

- [x] Fix SQLiteStore cross-thread safety (`threading.RLock`)
- [x] `SentenceTransformerEmbedder` — first-class embedder, `[sentence-transformers]` extra
- [x] README quickstart with `pip install emotional-memory[sentence-transformers]`
- [x] `CITATION.cff` — Zenodo-ready, GitHub "Cite this repository" button
- [x] Fidelity benchmark table links to source test files
- [x] `docs/research/08_limitations.md` — documented known limits
- [x] Published to PyPI as `emotional-memory==0.5.2`
- [x] Zenodo DOI `10.5281/zenodo.19636356`
- [x] arXiv-style paper 10p (`paper/main.tex`) — 4 figures, comparative + perf tables

---

## v0.6.0 — Discovery & Integration (shipped ✓)

- [x] LangChain adapter (`EmotionalMemoryChatHistory`, `[langchain]` extra + CI job)
- [x] Comparative benchmark vs `naive_cosine`, `recency`, Mem0, LangMem (Letta cloud-only, availability-guarded)
- [x] Dataset `affect_reference_v1` (258 examples)
- [x] arXiv-style paper updated (Zenodo DOI `10.5281/zenodo.19640250`, PyPI `0.6.0`)
- [x] mkdocs-material site source (`docs/`) with async, LangChain, persistence tutorials
- [x] HuggingFace Space `homen3/emotional-memory-demo` live
- [x] `Development Status :: 4 - Beta`

---

## v0.7.0 — Scientific Evidence Push (shipped ✓ 2026-05-02)

The originally-scoped "production readiness" items (Qdrant, Chroma, OpenTelemetry, BYO appraisal) **did not ship in v0.7.0** — they have been moved to v0.9.0. v0.7.0 instead consolidated empirical evidence and architecture attribution.

### Pre-registered evidence programme
- [x] **Gate 1 (external benchmark, LoCoMo)** — runner, hypothesis tests, full N=1986 QA results. **Honest negative**: H1/H2 FAIL (`benchmarks/locomo/`). Claim `locomo_external_qa_negative` added to claim matrix.
- [x] **Gate 3 (architecture attribution)** — Hd1 PASS (Addendum D, seed=1): `aft_noAppraisal` Δ=+0.23 vs `naive_cosine`. Closes the appraisal-confound objection.
- [x] **G4/G5 cross-embedder at N=200** — `realistic_recall_v2` (50 scenarios × 4 challenge types). SBERT Δ=+0.205 (d=0.49), e5-small-v2 Δ=+0.155 (d=0.31). CLOSED.
- [x] **G6 multilingual (Italian)** — `realistic_recall_v2_it.json` on SBERT and `multilingual-e5-small`. hit@k significant on both (p=0.0005 / p=0.001). EN-centric SBERT confirmed as IT accuracy bottleneck.
- [x] **G9 ablations** — `no_reconsolidation` (He2: null), `dual_path` (He1: destructive — keyword-driven), `aft_keyword_synchronous` (Hf1: deferral mitigates synchronous override).
- [x] **`EmotionalMemoryConfig.enable_reconsolidation: bool = True`** — new public flag (sync + async engines).
- [x] **Pre-registration corpus** — addenda v2 (B), v3 (D/E), F (Hf1), H (G6 cross-embedder).
- [x] **`docs/research/audit_2026-04.md`** — running tracker; gates table, gap inventory, claim coherence.

### Paper / artefacts
- [x] arXiv submission bundle ready (`paper/arxiv-submission.tar.gz`, cs.LG)
- [x] Zenodo DOI `10.5281/zenodo.19972285`
- [x] PyPI `emotional-memory==0.7.0`
- [x] Six project-scoped Claude Code skills (`.claude/skills/`)

### Production-readiness items NOT in v0.7.0
Moved to **v0.9.0** below. The dot-release window remains open for paper polish only.

---

## v0.7.x — Paper polish (open)

Dot release(s) for the paper bundle, no API changes:

- [x] Footnote linking Addendum H from §Limitations of `paper/main.tex`
- [ ] arXiv submission executed (cs.LG, no endorsement) — bundle is shipped, upload pending
- [ ] Post-acceptance: update `CITATION.cff` with arXiv URL, refresh README badges

---

## v0.8.0 — Closing the open gates (target: 2026 Q3)

Goal: lift the public-claim ceiling by closing Gate 2 and adding the missing intra-theory dimension.

### Gate 2 — Human evaluation (kit shipped, execution pending)
- [ ] Distribute the existing `benchmarks/human_eval/` packets to 20–30 raters (Prolific or MTurk)
- [ ] Collect `ratings.jsonl`, run pre-registered analysis pipeline (`benchmarks/human_eval/pipeline.py`)
- [ ] Update `claim_validation_matrix.json` and audit doc

### S3 ablation @ N=200 + Hd2 generalisation — CLOSED 2026-05-04
- [x] Re-point `benchmarks/ablation/runner.py` at `realistic_recall_v2` — `--dataset` flag added; `bench-s3-sbert` / `bench-s3-e5` targets added to Makefile
- [x] Execute confirmatory ablation at full power (N=200, seed=0, n_bootstrap=2000, both SBERT and e5)
  - Ha (no_mood): FAIL on both embedders (SBERT Δ=-0.02 p=0.264; e5 Δ=-0.005 p=0.915)
  - Hb (no_resonance): FAIL on both (SBERT Δ=+0.02 p=0.203; e5 Δ=+0.085 p<0.001 — opposite direction)
  - Hc (no_appraisal invariant): PASS on both — benchmark correctly inert
  - Hd2 / Hd2_IT: PASS (EN Δ=0.125, IT Δ=0.163 — architecture advantage generalizes)
- [x] Closure docs: `benchmarks/preregistration_addendum_s3_closure.md`, `benchmarks/preregistration_addendum_hd2_closure.md`, S3 footer in `preregistration.md`, Hd2 footer in `preregistration_addendum_v3.md`

### G7 — PAD dominance as first-class CoreAffect
- [ ] Promote `dominance` from `MoodField`-only to `CoreAffect` 3D field
- [ ] Migration path for serialised `AffectiveState` (back-compat read for v0.7-era snapshots)
- [ ] Re-enable `benchmarks/fidelity/test_dominance_retrieval_gap.py` (currently `xfail strict`)
- [ ] Design note `docs/research/11_dominance_design.md` already shipped — implements that design

### Multilingual breadth (beyond Italian)
- [ ] One additional non-English slice (Spanish or French) using the existing `make_multilingual()` factory
- [ ] Extends G6 cross-embedder claim to ≥3 language families

---

## v0.9.0 — Production Readiness (target: 2026 Q4)

Goal: make the library production-grade for teams running agents at scale. (These are the items originally scoped for v0.7.0; deferred while the evidence programme took priority.)

### Enterprise vector stores
- [x] `QdrantStore` adapter + `[qdrant]` optional extra
- [x] `ChromaStore` adapter + `[chroma]` optional extra
- [ ] Both implement the `MemoryStore` protocol; ANN behaviour parity with `SQLiteStore` documented

### Observability
- [x] Optional OpenTelemetry spans on `encode`, `retrieve`, `encode_batch`, `elaborate`, `observe`, `prune`
- [x] `[otel]` optional extra; spans no-op when extra is absent
- [x] (Structured `logging.DEBUG` events on pipeline operations already shipped in v0.2.0.)

### BYO appraisal schema
- [ ] `AppraisalSchema` config class — parameterise the Scherer CPM prompt so OCC, GRID, or custom taxonomies can be injected without forking
- [ ] Schema-validated `AppraisalVector` (Pydantic) for non-Scherer outputs

### LoCoMo per-task tuning (Pareto study)
- [ ] Pre-register a per-task-type weight search (Gate 1 follow-up to the negative result) — explicit confirmatory design before any tuning runs
- [ ] Sweep harness extending `benchmarks/locomo/runner.py`
- [ ] Pareto-frontier analysis: AFT-favourable categories vs naive-favourable categories

---

## v1.0.0 — Stability commitment (target: when above is closed)

- [ ] Public-API freeze; semver-stability commitment for `EmotionalMemory`, `AsyncEmotionalMemory`, all `interfaces.py` protocols, the `EmotionalMemoryConfig` tree, and the persistence formats (`AffectiveState.snapshot`, `Memory.model_dump`).
- [ ] Migration guide for v0.x → v1.0.

---

## Contributing

Want to work on something on this roadmap? Open an issue first to discuss scope and approach. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions.

Items not on this roadmap but worth discussing:
- Persistent memory compression / summarisation
- Cross-agent emotional resonance (shared mood fields)
- Integration with more LLM frameworks (LlamaIndex, CrewAI, AutoGen)
- Real-time streaming encode (partial affective updates)
