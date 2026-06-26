# Project Validation Report — June 2026

> **Scope.** A full execution of the project's validation suite plus a point-by-point
> re-verification of every entry in
> [`problem_register_2026-06.md`](problem_register_2026-06.md), run on branch
> `claude/project-validation-issues-njkdxx` against `v0.11.4`. This is a *point-in-time
> validation snapshot*; where it disagrees with the register, the discrepancy is
> recorded here as a finding. The canonical claim status remains
> [`claim_validation_matrix.json`](claim_validation_matrix.json).

## Executive summary

| Area | Result |
|---|---|
| Lint / format (ruff) | 🟢 pass |
| Type checking (mypy strict) | 🟢 pass — 37 files, no issues |
| Release-metadata / claim-ref integrity | 🟢 pass — 122/122 refs resolve |
| Unit + integration tests | 🟢 992 passed, 3 skipped; **branch coverage 91.52%** (≥80%) |
| Fidelity benchmarks | 🟢 127 passed |
| Paper-table reproduction | 🟢 no diff (tables fresh) |
| Docs build (`mkdocs --strict`) | 🟡→🟢 **broken link fixed** (was failing; now passes) |
| Preflight (`--fast --ci`) | 🟢 7/7 gates |
| Performance benchmarks | 🟡 25/26 (one 10k-scale setup hits the 120 s pytest timeout) |
| Multi-seed robustness (A7) | 🔴 **determinism claim does not reproduce** — see Finding F2 |
| Security (pip-audit) | 🟢 runtime clean · 🟡 D1 chromadb CVE stands · D2 torch patch now available |
| LLM-dependent suites | ⚪ **Blocked** — `EMOTIONAL_MEMORY_LLM_API_KEY` not set |

**Headline:** the engineering quality gates are green and the core scientific FAILs
are honestly recorded in the claim matrix exactly as the register states. Three things
need attention: (F1) one docs link broke the strict build — **fixed in this pass**;
(F2) the A7 "retrieval is deterministic, cross-seed stdev = 0.0000" claim **fails to
reproduce** (a fresh sweep shows genuine cross-seed variance); (F3) the register is now
**stale** on C1/C3 (both already implemented in the README) and on D2 (a patched torch
is now published).

---

## 1. Gate-by-gate results

All commands run with `uv` in a Python 3.11.15 venv after `make install-all`.

| Gate | Command | Result | Notes |
|---|---|---|---|
| Lint | `make lint` | 🟢 | ruff check + `format --check` (224 files) |
| Typecheck | `make typecheck` | 🟢 | mypy strict, 37 files, 0 issues |
| Metadata | `make meta-check` | 🟢 | `release metadata OK for 0.11.4`; 122 claim refs exist |
| Coverage | `make cov` | 🟢 | 992 passed / 3 skipped / 41 deselected; **TOTAL 91.52%** |
| Fidelity | `make bench-fidelity` | 🟢 | 127 passed in 0.25 s |
| Paper repro | `make reproduce-paper-check` | 🟢 | `git diff --exit-code paper/tables/` clean |
| Docs | `make docs` (strict) | 🟢¹ | ¹passed **after** the F1 link fix |
| Preflight | `preflight.py --fast --ci` | 🟢 | G1–G3b, G7–G9 pass; git/build gates skipped by flags |
| Perf | `make bench-perf` | 🟡 | 25 passed, 1 timeout (F4) |
| Multiseed | `make bench-multiseed` | 🔴 | non-deterministic (F2) |
| Comparative | `make bench-comparative` (hash) | 🟢 | aft recall@5 0.50 > cosine 0.45 > recency 0.25 |
| Ablation | `make bench-ablation` (hash) | 🟢 | synchronous keyword appraisal harmful (Δ=−0.13, p_mc<0.001) — confirms A1/Hp3 |
| Security | `pip-audit` | 🟢/🟡 | runtime deps clean; D1 stands, D2 patch available (F3) |
| LLM tests | `make test-llm`, `make bench-appraisal` | ⚪ | Blocked — no API key |

### Coverage note (ordering dependency)

A bare `make cov` / `make test` on a fresh checkout fails exactly one test —
`tests/test_figure_inventory.py::test_outputs_exist_on_disk` — because the declared
`.pdf` figure outputs are git-ignored and generated on demand. CI compensates by
running `make figures` **before** pytest (`.github/workflows/ci.yml`). After
`make figures` locally, the full inventory suite (12 tests) passes. **Not a defect**,
but a local-vs-CI ordering trap worth a one-line note in the contributor docs.

---

## 2. Problem-register verification

Legend: **Resolved** · **Honestly scoped** (committed FAIL / open gap, correctly
bounded) · **Open** (future work) · **Stale** (register text no longer matches reality).

| ID | Register status | Verified status | Evidence |
|---|---|---|---|
| A1 | Falsified / bound | ✅ **Honestly scoped** | `appraisal_llm_real_dual_path` = `falsified` in matrix; ablation reproduces the synchronous-appraisal penalty (Δ=−0.13) |
| A2 | Bound every citation | ✅ **Honestly scoped** | 15 claims carry `requires_oracle_affect`; README oracle-affect boundary note present (L301–303) |
| A3 | Scope to ranking | ✅ **Honestly scoped** | `locomo_external_qa_negative` = `controlled_evidence` (committed negative); README documents F1 0.168 vs 0.271 |
| A4 | `not_established`; future work | ✅ **Open** | `models_human_emotional_memory` = `not_established`; tracked by **open issue #27** |
| A5 | Bound; future work | ✅ **Open** | Appraisal bounded as not human-validated; **untracked** (candidate issue) |
| A6 | Already scoped | ✅ **Honestly scoped** | `cross_domain_affect_replication` = `controlled_evidence`; README documents IT/ES FAIL |
| A7 | **Resolved — deterministic** | 🔴 **Does not reproduce** | Fresh sweep shows cross-seed variance — see **Finding F2** |
| B1 | Add 3-state legend | ✅ **Resolved** | Legend present in `review_response_2026-06.md` L38–39; §3.1 relabelled "Honestly scoped, not solved" |
| B2 | Re-word snapshot | ✅ **Resolved** | — |
| C1 | **Open** — needs footnote | ✅ **Done (register stale)** | Footnote already under "How it compares" (README L41) |
| C2 | Optional suffix | ⚪ **No action** | README L15 already scopes the one-liner; not forced |
| C3 | **Open** — add section | ✅ **Done (register stale)** | "When NOT to use" section present (README L45) with "Recommended for"; tracking issue #32 closed |
| D1 | Document + monitor | ✅ **Open / accurate** | `pip-audit` confirms chromadb 1.5.9 still carries CVE-2026-45829; optional `[chroma]` only, not in runtime wheel |
| D2 | No upstream fix | 🟡 **Patch now available** | torch **2.12.1** published and audits clean; `uv.lock` still pins 2.12.0 — see **Finding F3** |

Net: of 13 register entries, **9 are accurate**, **2 are stale-but-already-fixed**
(C1, C3 — good news), **1 needs a dependency bump** (D2), and **1 fails to reproduce**
(A7).

---

## 3. Findings

### F1 — Broken docs link broke the strict build *(fixed in this pass)*

`make docs` (mkdocs strict) aborted on:

```
WARNING - Doc file 'research/review_response_2026-06.md' contains a link
'comparison.md', but the target 'research/comparison.md' is not found …
```

The target lives at `docs/comparison.md`; the link from `docs/research/` must be
`../comparison.md`. **Fixed** (`review_response_2026-06.md` L163). `make docs` now
exits 0. Because `docs.yml` deploys on every push to `main`, this would have failed
the docs deploy.

### F2 — A7 "retrieval is deterministic" does not reproduce *(needs a decision)*

`make bench-multiseed` (hash embedder, seeds {0,1,7,42,123}, subprocess-isolated)
reports `retrieval_deterministic=False`. The committed artifact and prose claim the
opposite:

| Source | `aft` top1 mean | stdev | spread | verdict |
|---|---:|---:|---:|---|
| Committed `multiseed_results.md` | 0.1250 | **0.0000** | 0.0000 | "✅ identical across all seeds" |
| `problem_register` §A7 | — | **0.0000** | 0.0000 | "retrieval verified deterministic" |
| `08_limitations` §2.9 | — | **0.0000** | 0.0000 | "exactly 0.0000" |
| **This re-run** | 0.1210 | **0.0020** | 0.0050 | "⚠️ genuine cross-seed variance" |

The variance is **not** seed/RNG driven — it is the wall-clock timing effect the
register itself documents (the engine stamps encode/retrieve with `datetime.now`, and
ACT-R decay tracks `now − encoded_at`, so near-tie queries flip between sequential
subprocess runs). The scientific conclusion is unaffected: AFT still beats cosine by
+0.075–0.080 and the spread sits inside the bootstrap CIs. **But the specific claim —
"cross-seed stdev = exactly 0.0000, retrieval is deterministic" — is an overclaim that
a fresh run falsifies.** Honest fix: re-scope A7 from "Resolved (deterministic)" to
"near-deterministic; sub-CI timing-driven variance on near-ties," and regenerate the
committed artifact. This changes scientific framing across three files, so it is
flagged for the maintainer's decision rather than applied unilaterally.

### F3 — Register stale on C1/C3 and D2

- **C1 / C3** are listed "Open" in the register but are **already implemented** in the
  README (comparison-table footnote and the "When NOT to use" section). Good news; the
  register's §1 severity snapshot is simply behind reality. Tracking issue #32 is no
  longer open.
- **D2**: the register says "torch ≤2.12.0 affected; no patched version published."
  torch **2.12.1 is now on PyPI** and `pip-audit` reports it clean. `uv.lock` still
  pins 2.12.0. Recommend `uv lock --upgrade-package torch` (→ 2.12.1) and refresh
  `SECURITY.md` once confirmed.

### F4 — 10k-scale perf benchmark hits the global timeout *(environment)*

`bench_retrieve_top5[10000]` times out: the `populate_store` setup encodes 10 000
memories while building resonance links, exceeding the 120 s `pytest-timeout` ceiling
in this environment (it reached ~5–10k before the cut). Library logic is fine — the
setup cost at 10k scale is the issue. Options: raise the timeout for the perf suite,
or shrink the largest `store_size` parameter. Low priority.

---

## 4. Blocked / not run

| Item | Why | How to unblock |
|---|---|---|
| `make test-llm` | `EMOTIONAL_MEMORY_LLM_API_KEY` unset | export key (see `docs/contributing/llm-environment.md`), re-run |
| `make bench-appraisal` | same | same |
| Full LoCoMo re-run | dataset downloads on demand; the fetch came back incomplete through the sandbox proxy | run where the snap-research LoCoMo JSON is reachable; committed A3 FAIL (F1 0.168 vs 0.271) stands |
| Optional backends (qdrant/chroma/redis) | not pulled by `install-all`; chroma deliberately not installed (D1 CVE) | `make install-redis` / targeted extras, then per-backend tests |
| Full sbert/me5 comparative & multilingual reruns | long-running + model downloads | committed EN/FR/IT/ES numbers stand; qualitative story reproduced on the hash embedder |

---

## 5. Remediation applied in this pass

1. **F1** — `docs/research/review_response_2026-06.md`: `comparison.md` → `../comparison.md`
   (unblocks `mkdocs --strict`).
2. **Gap #4** — `README.md`: added a **Limitations** link
   (`docs/research/08_limitations.md`) to the "Validation & Benchmarks" list so the
   scoped-boundaries doc is discoverable from the hero.

Both verified: `make docs` (strict) and `make meta-check` pass after the edits; all
README doc links resolve (no dead links — gap #9 clean).

## 6. Recommended follow-ups (not applied — need maintainer decision)

- **F2 / A7**: re-scope the determinism claim and regenerate `multiseed_results.md`
  (scientific-framing change across 3 files).
- **F3 / D2**: bump `uv.lock` torch 2.12.0 → 2.12.1; update `SECURITY.md`.
- **Issues**: A4 and arXiv are already tracked (#27, #31). Untracked gaps worth an
  issue: **A3** minimal downstream task (encode→retrieve→generate→judge), **A5**
  human-gold appraisal comparison, and **F2** the A7 reproducibility finding.
- **F4**: raise the perf-suite timeout or cap the 10k `store_size` parameter.
