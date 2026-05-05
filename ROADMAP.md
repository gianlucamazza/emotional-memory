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
- [x] arXiv submission bundle ready (`paper/arxiv-submission.tar.gz`, `make check-arxiv-bundle` enforces freshness)
- [ ] arXiv submission executed (cs.LG, no endorsement) — upload pending (user action)
- [ ] Post-submission: update `release.toml: arxiv_id`, run `sync_release_metadata --from-toml`, refresh README badges

---

## v0.8.x — Evidence programme, closed (2026-04 → 2026-05) ✓

All items in this milestone shipped across v0.8.0–v0.8.3 dot releases.

### Gate evidence — all closed
- [x] **S3 ablation @ N=200** — Ha (no_mood): FAIL; Hb (no_resonance): FAIL (e5 opposite direction — sign-reversal open question, moved to v0.10); Hc: PASS; Hd2/Hd2_IT: PASS (EN Δ=+0.125, IT Δ=+0.163). Closure docs in `benchmarks/`.
- [x] **G7 PAD dominance design** — design note `docs/research/11_dominance_design.md` shipped. CoreAffect 3D promotion deferred to v0.10 (back-compat migration required).
- [x] **SSOT automation** — `make bump VERSION=X.Y.Z` (atomic 3-file edit + propagation + preflight), `make check-all`, `sync_release_metadata` covers demo/README drift.

### Gate 2 — Human evaluation (kit shipped, execution deferred to v0.10)
Kit is ready in `benchmarks/human_eval/`. Execution (Prolific/MTurk distribution) moved to v0.10.

### Multilingual breadth (ES/FR) — deferred to v0.10
Italian slice (G6) closed. Spanish/French extension moved to v0.10.

---

## v0.9.0 — Production Readiness (shipped, PR #24) ✓

Goal: make the library production-grade for teams running agents at scale.

### Enterprise vector stores
- [x] `QdrantStore` adapter + `[qdrant]` optional extra
- [x] `ChromaStore` adapter + `[chroma]` optional extra
- [x] Both implement the `MemoryStore` protocol; ANN behaviour documented in `docs/api/stores.md`

### Observability
- [x] Optional OpenTelemetry spans on `encode`, `retrieve`, `encode_batch`, `elaborate`, `observe`, `prune`
- [x] `[otel]` optional extra; spans no-op when extra is absent
- [x] (Structured `logging.DEBUG` events on pipeline operations already shipped in v0.2.0.)

### Deferred to v0.10
BYO appraisal schema and LoCoMo per-task tuning were descoped from v0.9 to keep the release focused. See v0.10 below.

---

## v0.10.0 — Evidence + parametricity (target: 2026)

Collecting items deferred from v0.7–v0.9 plus the open sign-reversal question.

### BYO appraisal schema (deferred from v0.9)
- [ ] `AppraisalSchema` config class — parameterise the Scherer CPM prompt so OCC, GRID, or custom taxonomies can be injected without forking
- [ ] Schema-validated `AppraisalVector` (Pydantic) for non-Scherer outputs

### LoCoMo per-task Pareto study (deferred from v0.9 — Gate 1 follow-up)
- [ ] Pre-register a per-task-type weight search before any tuning runs
- [ ] Sweep harness extending `benchmarks/locomo/runner.py`
- [ ] Pareto-frontier analysis: AFT-favourable vs naive-favourable task categories

### Gate 2 — Human evaluation execution (deferred from v0.8)
- [ ] Distribute `benchmarks/human_eval/` packets to 20–30 raters (Prolific or MTurk)
- [ ] Collect `ratings.jsonl`, run `benchmarks/human_eval/pipeline.py`
- [ ] Update `claim_validation_matrix.json` and audit doc

### G7 — PAD dominance as first-class CoreAffect (deferred from v0.8)
- [ ] Promote `dominance` from `MoodField`-only to `CoreAffect` 3D field
- [ ] Migration path for serialised `AffectiveState` (back-compat read for v0.7-era snapshots)
- [ ] Re-enable `benchmarks/fidelity/test_dominance_retrieval_gap.py` (currently `xfail strict`)
- [ ] Design note `docs/research/11_dominance_design.md` already shipped

### Sign-reversal experiment (open from v0.8 S3 closure)
- [ ] Investigate why `no_resonance` ablation shows `Δ=+0.085 p<0.001` with e5 (opposite direction to SBERT Δ=+0.02)
- [ ] Controlled experiment: e5 × resonance enabled/disabled; pre-register before running

### Multilingual breadth (deferred from v0.8)
- [ ] One additional non-English slice (Spanish or French) using the existing `make_multilingual()` factory
- [ ] Extends G6 cross-embedder claim to ≥3 language families

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
