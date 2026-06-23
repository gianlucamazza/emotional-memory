# Problem Register & Correct Resolutions (June 2026)

A complete, prioritized register of the project's open problems — research /
evidence gaps, technical / security issues, and the review critiques that remain
materially unresolved — each with its **root cause** and its **correct
resolution**.

The resolution philosophy here is **honest re-scoping**: where a problem is not
solvable in the short term, the correct resolution is to *bound or correct the
claims* rather than fake a fix. Technical issues that are fixable are listed with
their real fix; issues with no upstream fix are documented and monitored.
Research-execution items that need external resources (human ratings, an LLM
judge for a downstream task) are recorded as explicitly-scoped **future work**,
not silently implied as done. The one item completable in-repo without external
resources — the multi-seed robustness sweep (A7) — has been implemented and run.

This document is a companion to [`review_response_2026-06.md`](review_response_2026-06.md)
(which maps the external review point-by-point) and defers to
[`claim_validation_matrix.json`](claim_validation_matrix.json) as the canonical
claim status. Where the two disagree, the matrix wins.

---

## 1. Severity snapshot

| ID | Problem | Category | Severity | Resolution mode |
|---|---|---|---|---|
| A1 | End-to-end LLM appraisal loses to cosine | Science | High | Already falsified; bound wording |
| A2 | Top results need oracle affect + state injection | Science | High | Bound every public citation |
| A3 | No downstream / agent-level benefit | Science | High | Scope to retrieval ranking; surface FAILs |
| A4 | No human / ecological validation | Science | High | Keep `not_established`; future work |
| A5 | Affect-signal construct validity unproven vs human gold | Science | Medium | Bound appraisal wording; future work |
| A6 | Cross-lingual generalization limited (IT/ES FAIL) | Science | Medium | Already scoped; keep current |
| A7 | Single-seed; cross-run variance uncharacterized | Method | Medium | **Resolved** — sweep added; retrieval verified deterministic |
| B1 | Response doc conflates *scoped* with *solved* | Meta | Medium | Add 3-state legend; relabel |
| B2 | "Five already addressed" snapshot overstates | Meta | Low | Re-word |
| C1 | README comparison ✅ implies general superiority | Public claim | Medium | Footnote on the table |
| C2 | Bare "Emotional memory" one-liners | Public claim | Low | Optional light suffix |
| C3 | No consolidated "When NOT to use" section | Public claim | High | Add section (issue #32) |
| D1 | chromadb CVE-2026-45829 (critical, optional/dev) | Security | Medium | No upstream fix; document + monitor |
| D2 | torch CVE-2025-3000 (low, dev-only) | Security | Low | No upstream fix; document + monitor |

The headline is that the **core scientific problems are already resolved in the
honest sense** — they are pre-registered FAILs with committed closures and the
correct `falsified` / `not_established` statuses in the claim matrix. The work
left is to make sure no *public* surface drifts ahead of that record, plus two
unpatched optional-dependency CVEs.

---

## 2. A — Core scientific problems

### A1 — End-to-end LLM appraisal does not beat cosine

**Problem.** With a real `LLMAppraisalEngine` (not oracle affect), AFT does not
outperform naive cosine on affect-free queries: Hg1 FAIL (Δ=−0.010, p=0.367),
and the leakage-free re-run with the *recalibrated* mapping is significantly
worse — Addendum P, `realistic_recall_v4_noAF`, N=160: Δ=−0.087 [−0.144, −0.031],
p=0.0018, d=−0.242 (`benchmarks/preregistration_addendum_p_hg1_rerun_closure.md`).

**Root cause.** Even a directionally-correct, recalibrated affect signal
(Addendum O lifted held-out valence/arousal bias to +0.072 / −0.023, Pearson
r≈0.79–0.83) acts as a *net distractor* in a regime where semantics already
discriminates the target.

**Correct resolution.** Already done at the canonical level —
`appraisal_llm_real_dual_path` is **`falsified`** in the matrix and the paper
abstract states it. Residual action: ensure no public surface implies an
automatic-appraisal benefit (see C1, C3). Note the one genuinely positive
contrast that must *not* be inflated: dual-path scheduling beats synchronous
appraisal (Hp3, d≈0.95) — that is a *within-AFT* result, not a win over cosine.

### A2 — The headline advantage requires oracle affect *and* query-time state injection

**Problem.** The strongest PASSes — realistic_recall_v2 (SBERT Δ=+0.205 [0.150,
0.265], d=0.49; e5 Δ=+0.155, d=0.31) and French (me5 Δ=+0.18 [0.11, 0.26],
g=0.424) — all inject preset valence/arousal at encode time *and* align the
query's affective state before retrieval. Addendum Q showed that real sessions do
not supply that query↔state alignment for free: gating recovers the always-on
penalty (Hq2 +0.080, p_holm=0.0009) but neither gated nor always-on AFT exceeds
cosine, and even on the affective subset `aft_llm_dual` loses (Hq1 Δ=−0.050).

**Root cause.** AFT retrieval signals are *state-based* and the query is never
appraised; the benchmark performs the alignment that production lacks. **The
oracle-affect boundary is also a state-injection boundary**
(`benchmarks/preregistration_addendum_q_affect_gating_closure.md`;
[`08_limitations.md`](08_limitations.md) §2.4).

**Correct resolution.** Every *public* citation of the +0.21 / +0.18 numbers must
carry the oracle-affect qualifier in-line, not 270 lines later. The matrix
already sets `requires_oracle_affect=true` on these claims; the paper discussion
re-scopes them. Residual action: README comparison-table footnote (C1).

### A3 — No downstream / agent-level benefit demonstrated

**Problem.** The only oracle-free, naturalistic evaluations FAIL: LoCoMo
conversational QA (AFT F1=0.168 vs naive_rag 0.271, Δ=−0.101, Gate 1 FAIL) and
DailyDialog short-turn dialogue (Δ=−0.008, p_holm=1.000). Two recovery attempts
also FAIL (Addendum J Pareto sweep; Addendum L query routing Δ=−0.017 / −0.081).

**Root cause.** The advantage is regime-specific to *affect-discriminative*
retrieval, where valence/arousal distinguishes the target from distractors — a
property of the curated v2/FR benchmarks, not of factual QA or natural dialogue.

**Correct resolution.** No "better memory for agents / downstream" wording. Scope
the claim to "affect-discriminative retrieval ranking." The `locomo_external_qa_negative`
claim is committed; the correct user-facing action is to **surface** these FAIL
regimes in the README so adopters meet them before integration (C3). An actual
downstream encode→retrieve→generate→judge task is **future work**.

### A4 — No human / ecological validation (Gate 2 OPEN)

**Problem.** The `benchmarks/human_eval/` kit (protocol, rater instructions,
packets, Krippendorff-α pipeline) has **zero collected ratings**. All validation
to date is intra-theoretical (127 fidelity tests), not ecological.

**Root cause.** Running raters requires people and time, not analysis.

**Correct resolution.** Keep `models_human_emotional_memory` =
**`not_established`** and state "no human validation" prominently (already true in
the matrix and `10_scientific_quality_bar.md`). Executing the pilot is the
single highest-value **future work** item; it is not faked here.

### A5 — Construct validity of the affect signal unproven vs human gold

**Problem.** Appraisal is *diagnosed* (Addendum N: valence Pearson r=0.883 vs
gold but +0.169 bias; prompt recalibration Hn1/Hn2 FAIL) and *partly fixed*
(Addendum O mapping recalibration PASS), but never validated against
**human-annotated** affect labels.

**Root cause.** No human-gold affect benchmark exists in-repo; the "gold" used so
far is itself LLM-derived.

**Correct resolution.** Scope appraisal claims to "directionally correct,
mis-calibration partially corrected, **not human-validated**." No "accurate
appraisal" wording anywhere (this corrects a phrase carried in the issue-#32
template — see C3). A human-gold comparison (e.g. EmoBank / DailyDialog affect
subset) is future work.

### A6 — Cross-lingual generalization is limited

**Problem.** At declared power (N=120, me5), Italian (Δ=+0.058, p=0.276) and
Spanish (Δ=0.000, p=1.000) both **FAIL**; only English-SBERT and French-me5
(plus exploratory Spanish-SBERT N=80) hold.

**Root cause.** Small per-language N and language-specific scenario authoring.

**Correct resolution.** Already honestly scoped in README line 15 and the matrix
(`cross_domain_affect_replication`). No change beyond keeping numbers current.

### A7 — Single-seed; cross-run variance uncharacterized — *Resolved*

**Problem.** Most runners pin one seed (ablation seed=0, Hi3 seed=1, Pareto
seed=42); there was no automated multi-seed sweep reporting cross-run variance,
and the reported CIs are bootstrap CIs *within* a single run, not across seeds.

**Root cause.** No multi-seed wrapper around the existing runners.

**Resolution (done).** Added `benchmarks/realistic/multiseed_runner.py`
(`make bench-multiseed`): it re-runs the realistic benchmark across seeds
`{0, 1, 7, 42, 123}`, each in an isolated subprocess invoking the canonical
runner, and reports cross-seed mean/stdev/min/max. Committed result
(`benchmarks/realistic/multiseed_results.md`, hash embedder on v2): **cross-seed
stdev = spread = 0.0000** — per-query top-1 outcomes are identical across seeds.
This *verifies* (rather than assumes) that retrieval is deterministic given a
fixed dataset + deterministic embedder; the only seed-sensitive quantity is the
bootstrap CI resampling. See [`08_limitations.md`](08_limitations.md) §2.9.
A by-product finding: running several full benchmarks in one process leaks global
state, so the harness isolates each seed in its own subprocess — the canonical
one-run-per-process model.

---

## 3. B — Response-doc self-corrections

### B1 — "Already addressed" conflates *scoped* with *solved*

**Problem.** The verdict table in [`review_response_2026-06.md`](review_response_2026-06.md)
§2 marks §3.1 (LoCoMo / downstream) and §3.3 (framing) as **"Already addressed."**
For §3.1 this is misleading: the *criticism* — no downstream win — is not solved;
it is a committed FAIL that has been honestly **scoped**. Documenting a failure is
not resolving it.

**Correct resolution.** Introduce a 3-state legend — **Resolved** /
**Honestly scoped (committed FAIL; underlying gap open)** / **Open** — and
relabel §3.1 to *Honestly scoped, not solved*. (§3.3 is legitimately resolved by
*under*-claiming, which is a real resolution; it keeps "Resolved" with a note.)

### B2 — Snapshot overstates

**Problem.** The snapshot sentence "five are already addressed" carries the same
conflation.

**Correct resolution.** Re-word to "five are resolved or honestly scoped, two are
partially open; the underlying downstream and construct-validity gaps remain
open."

---

## 4. C — Public-claim wording

### C1 — Comparison-table ✅ implies general superiority

**Problem.** `README.md` "How it compares" marks AFT ✅ on *Affective retrieval*
and *Reconsolidation* against all-❌ competitors. A binary checkmark reads as
general superiority; in fact the *measured* advantage is regime-specific and
oracle-gated (A2), and under automatic appraisal it reverses (A1).

**Correct resolution.** Add a footnote directly under the table: **✅ = feature
implemented & theory-faithful; it is not a head-to-head performance result. The
measured retrieval advantage is regime-specific to affect-discriminative recall
under oracle-affect labeling (see Validation / When NOT to use). ❌ = feature
absent, not a quality judgment.**

### C2 — Bare "Emotional memory" one-liners *(optional, low)*

**Problem.** `README.md` L13, the `pyproject.toml` description, and the
`src/emotional_memory/__init__.py` docstring all say "Emotional memory for LLMs"
without scope. This is a metaphor that *could* imply downstream benefit.

**Correct resolution.** Low priority — README L15 scopes it immediately. Optional:
a light suffix such as "…; affect-discriminative retrieval — see Validation for
scope." Not forced; recorded for completeness.

### C3 — No consolidated "When NOT to use" section (issue #32)

**Problem.** The four FAIL regimes (factual/open-domain QA, end-to-end LLM
appraisal, short-turn dialogue, query-type routing) are documented but scattered;
an adopter can miss them.

**Correct resolution.** Add a "When NOT to use" section after "How it compares",
listing the four regimes with their committed numbers and a "Recommended for"
counterpart — adapted from `.github/issue_templates/issue_1_readme_when_not_to_use.md`,
**with one correction**: the template's "requires oracle affect *or accurate
appraisal*" is itself an overclaim (Addendum P shows recalibrated appraisal still
loses). The honest wording is "requires oracle affect; the gain has **not**
transferred to automatic appraisal, calibrated or not."

---

## 5. D — Technical / security

Context: a recent `uv.lock` refresh resolved 8 of 10 Dependabot alerts. Two
remain, both without an upstream patch, both confined to optional/dev
dependencies. The published runtime wheel is unaffected — consistent with the
existing `SECURITY.md` *Scope* section (dev/optional/third-party deps are out of
the library's security scope).

### D1 — chromadb CVE-2026-45829 (critical, optional `[chroma]` / dev only)

**Problem.** chromadb ≤1.5.9 is vulnerable; no patched release is on PyPI.
chromadb is pulled only by the optional `[chroma]` extra and dev/test installs.

**Correct resolution.** No fix available. Document in `SECURITY.md` (Known
advisories) that it does not affect the runtime wheel; monitor PyPI and bump
`uv.lock` immediately when a patch ships. If it blocks `pip-audit` in CI, add a
justified, time-boxed ignore referencing this entry.

### D2 — torch CVE-2025-3000 (low, dev-only)

**Problem.** torch ≤2.12.0 affected; no patched version published; reachable only
via an optional dev dependency chain.

**Correct resolution.** Same as D1 — document and monitor; no runtime exposure.

---

## 6. E — No action required (positive signals)

Recorded so the register is complete and the "all problems" claim is honest:
minimal type/lint debt (1 `# noqa`, 0 `type: ignore`/`cast`); robust CI matrix
(Python 3.11–3.14, mypy strict + basedpyright, 80% branch-coverage floor,
`pip-audit`); SQLite store/state thread-safety resolved (`threading.RLock`); no
silent `xfail`/skip in tests (skips are env/feature-gated).

---

## 7. Future work (scoped, not faked)

In priority order, the items whose *correct* resolution is execution rather than
re-scoping — none claimed as done:

1. **Run the human-eval pilot** (A4) — the highest-value gap (Gate 2); needs raters.
2. **Human-gold appraisal comparison** (A5) — validate the affect signal against
   people, not LLM-derived gold; needs annotations.
3. **Minimal downstream task** (A3) — encode→retrieve→generate→judge, to test
   whether ranking gains convert to end-to-end value; needs an LLM judge.

(A7, the multi-seed robustness sweep, is **done** — see §A7 above.)
