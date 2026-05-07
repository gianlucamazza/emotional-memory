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
- [x] **S3 ablation @ N=200** — Ha (no_mood): FAIL; Hb (no_resonance): FAIL (e5 magnitude amplification — decomposed in Add. I, #29 closed); Hc: PASS; Hd2: PASS (EN Δ=+0.125). Closure docs in `benchmarks/`.
- [x] **P2-1 power top-up Hd2 IT/ES to N=120** — Branch C (FAIL-FAIL, 2026-05-07). me5: IT Δ=+0.058 p=0.276, ES Δ=0.000 p=1.00. Cross-language scoped to ES-SBERT N=80 exploratory positive. Closure: `benchmarks/preregistration_addendum_hd2_powertopup_closure.md`.
- [x] **G7 PAD dominance design** — design note `docs/research/11_dominance_design.md` shipped. CoreAffect 3D promotion deferred to v0.10 (back-compat migration required).
- [x] **SSOT automation** — `make bump VERSION=X.Y.Z` (atomic 3-file edit + propagation + preflight), `make check-all`, `sync_release_metadata` covers demo/README drift.

### Gate 2 — Human evaluation (kit shipped, execution deferred to v0.10)
Kit is ready in `benchmarks/human_eval/`. Execution (Prolific/MTurk distribution) moved to v0.10.

### Multilingual breadth — Italian + Spanish shipped; French deferred to v0.10
Italian (G6) and Spanish (Hd2_ES, v0.8.2) slices closed. French extension deferred to v0.10.

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

### Post-v0.9 follow-ups (under [Unreleased] in CHANGELOG, scoped to v0.10)
BYO appraisal schema (#25) shipped post-v0.9 (commit `57ef515`). LoCoMo per-task Pareto study (#26) executed post-v0.9 (Add. J, Hj1 FAIL). See v0.10 below.

---

## v0.10.0 — Evidence + parametricity (target: 2026)

Collecting items deferred from v0.7–v0.9.

### BYO appraisal schema (shipped, #25) ✓
- [x] `AppraisalSchema` config class — parameterise the Scherer CPM prompt so OCC, GRID, or custom taxonomies can be injected without forking (commit `57ef515`)
- [x] Schema-validated `AppraisalVector` (Pydantic) for non-Scherer outputs (commit `57ef515`)

### LoCoMo per-task Pareto study (Gate 1 follow-up) — CLOSED, #26
- [x] Pre-registration frozen — Addendum J (`benchmarks/preregistration_addendum_j.md`): 10-config weight grid, 200-QA stratified subsample, cost ~$1.
- [x] Implement `benchmarks/locomo/pareto_runner.py` (stratified sampling, 10-config sweep)
- [x] Execute sweep (200 QA × 11 configs including naive_rag baseline)
- [x] Pareto-frontier analysis: **Hj1 FAIL** — no AFT config ≥ naive_rag on any category. Best: W2 aggregate F1=0.1765 vs naive_rag=0.2092. Per-task `base_weights` tuning line closed. See `benchmarks/preregistration_addendum_j_closure.md`.

### Gate 2 — Human evaluation execution (deferred from v0.8)
- [ ] Distribute `benchmarks/human_eval/` packets to 20–30 raters (Prolific or MTurk)
- [ ] Collect `ratings.jsonl`, run `benchmarks/human_eval/pipeline.py`
- [ ] Update `claim_validation_matrix.json` and audit doc

### G7 — PAD dominance (CLOSED, #28 wontfix)
- [x] 3D `CoreAffect` shipped in v0.8.2 (`8b9ddbe`); fidelity test re-enabled. Design note: `docs/research/11_dominance_design.md`.
- Back-compat read for pre-v0.8.2 snapshots intentionally not implemented (see #28 closure comment).

### Sign-reversal experiment (CLOSED, #29)
- [x] Per-challenge decomposition: magnitude amplification, not sign reversal. `semantic_confound` is the dominant driver (e5 Δ=+0.125 vs SBERT Δ=+0.025). See `benchmarks/preregistration_addendum_i.md`.
- [x] **Hi3 confirmatory (CLOSED, 2026-05-06)**: N=500 confirms cross-embedder amplification on `semantic_confound` (Δ=+0.090 [0.030,0.160], d=0.257, Holm-adj p=0.0234, **PASS**). Secondary Hi3_recency PASS (Δ=+0.070); Hi3_arc FAIL (Δ=+0.010). Closure: `benchmarks/preregistration_addendum_i_closure.md`. Mechanism (Hi2: link-set density differentiation) remains exploratory — both embedders saturate top-5 link cap; link-type/strength instrumentation not yet measured.

### Multilingual breadth (deferred from v0.8)
- [x] One additional non-English slice — Spanish (`realistic_recall_v2_es.json`, shipped v0.8.2 as Hd2_ES). Closes #30.
- [x] Extends G6 cross-embedder claim to ≥3 language families — `_figure_multilingual` now shows IT + ES × SBERT + me5.

### P3-1 — DailyDialog ecological replication (CLOSED, Branch B, 2026-05-07)
- [x] Pre-registration: `benchmarks/preregistration_addendum_k_dailydialog.md` (Hk1).
- [x] DailyDialog benchmark infrastructure: persona builder, programmatic query generator (4 types), AFT + naive_cosine adapters, runner with paired bootstrap + Holm m=4.
- [x] **Hk1 FAIL (Branch B)**: N=120 personas, 396 queries, multilingual-e5-small. AFT top1=0.212 vs naive_cosine=0.220 (Δ=-0.008, p_holm=1.000, d=-0.015). Only `affective_trajectory` shows exploratory positive trend (Δ=+0.103, d=0.186, N=39; underpowered). Cross-domain ecological replication not established. Regime-specificity of AFT advantage confirmed consistent with LoCoMo FAIL. Closure: `benchmarks/preregistration_addendum_k_dailydialog_closure.md`.

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
