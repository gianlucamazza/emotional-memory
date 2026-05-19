# Pre-registration: Addendum M — French Multilingual Slice (FR, me5)

**Status:** PENDING_EXECUTION
**Date (pre-reg):** 2026-05-13
**Addendum letter:** M (L reserved for query-routing WS3c)
**Embedder:** `multilingual-e5-small`
**Dataset:** `benchmarks/datasets/realistic_recall_v2_fr.json` (NEW — hand-authored native FR)
**Runner:** `benchmarks.realistic.runner`
**Parent closures:**
- `benchmarks/preregistration_addendum_bc_closure.md` (multilingual scope established)
- `benchmarks/preregistration_addendum_hd2_powertopup_closure.md` (IT/ES at N=120 — prior outcomes)

---

## Motivation

The `preregistration_addendum_hd2_powertopup_closure.md` reports:
- **IT (me5, N=120):** AFT top1=0.617 vs naive 0.467, Δ=+0.150, p_holm=0.016 (PASS — Branch A).
- **ES (me5, N=120):** AFT top1=0.500 vs naive 0.367, Δ=+0.133, p_holm=0.041 (borderline PASS — Branch A by pre-reg criteria).

Both IT and ES show consistent directional effects on `multilingual-e5-small`. However, the
effect size varies across languages, and neither result is large enough to confidently predict
replication in a third language without independent data.

This addendum hand-authors a native French dataset (30 scenarios, 120 queries) to test
whether the AFT advantage extends to French — a Romance language similar to Italian/Spanish
but with distinct lexical and syntactic properties. The dataset is authored natively in French
(not translated from Italian) to avoid data-snooping: an LLM-translated dataset would
preserve the challenge structure and lexical choices that drove the IT/ES results, inflating
apparent generalizability.

**Prior expectation:** FAIL (null prior). The ES borderline result and the structural
similarity between IT, ES, and FR suggest that if the effect is language-specific (not
universal), FR may fall below threshold. We declare FAIL as the expected outcome;
a PASS would represent unexpected positive replication.

This is a coverage milestone for the multilingual claim, not a primary confirmatory study.
A FAIL finding is publishable as a characterization of the conditions under which the
AFT advantage is and is not observed across European Romance languages.

---

## Hypothesis

### Hm1 (confirmatory)

> On `benchmarks/datasets/realistic_recall_v2_fr.json` (30 hand-authored native French
> scenarios, 120 queries, embedder `multilingual-e5-small`, top_k=2, oracle-affect mode),
> AFT `top1_accuracy` > naive_cosine `top1_accuracy`.

One-tailed (directional): the sign of Δ is pre-specified as positive (AFT > naive), consistent
with IT and ES. No Hm2 secondary — this slice is a coverage milestone, not a multi-hypothesis study.

**Expected outcome:** FAIL (Branch B). Declared explicitly to prevent post-hoc reframing.

---

## Dataset

### Source

30 scenarios hand-authored natively in French by the researcher. Each scenario follows the
schema of `realistic_recall_v2_it.json` identically:

| Property | Value |
|---|---|
| Scenarios | 30 |
| Sessions per scenario | 2 |
| Queries per scenario | 4 (all in session_2) |
| Total queries | 120 |
| Events total | ~228 (3–5 per session) |
| Challenge types | 5 × 24 queries each |
| Oracle valence/arousal | Pre-assigned (identical numerical structure to IT/ES) |
| Session IDs | `session_1`, `session_2` |
| Top-level `language` | `"fr"` |

### Why hand-authored (not translated from IT)

Translating from Italian would preserve the exact event structure and lexical challenge
(e.g., semantic confounders would share cognates). This risks inflating cross-language
generalizability by measuring embedding alignment on near-cognate text rather than
genuine multilingual affect-aware retrieval. Hand-authoring native FR creates structurally
analogous but lexically independent scenarios.

### Challenge type distribution

Mirrors IT/ES exactly: each of the 5 challenge types appears in 24 queries (4 queries ×
6 scenarios per challenge type, distributed across the 30 scenarios following the same
per-scenario pattern as IT/ES).

| Challenge type | Queries |
|---|---:|
| `affective_arc` | 24 |
| `semantic_confound` | 24 |
| `same_topic_distractor` | 24 |
| `momentum_alignment` | 24 |
| `recency_confound` | 24 |
| **Total** | **120** |

### Out of scope

- French keyword appraisal rules in `KeywordAppraisalEngine.make_multilingual_fr()` —
  deferred to v0.12. This run uses oracle valence/arousal from the dataset, not LLM or
  keyword appraisal.
- `bge-small-en-v1.5` embedder — English-only; running it on FR would replicate a known
  limitation rather than provide new information.
- A second me5 tier (e.g., `multilingual-e5-base`) — designated exploratory; not counted
  toward Hm1 confirmatory verdict.

---

## Statistical plan

| Parameter | Value |
|---|---|
| Primary metric | `top1_accuracy` (same as all Hd2 / powertopup closures) |
| Test | Paired bootstrap |
| n_bootstrap | 10,000 |
| seed | 0 (consistency with `realistic.runner` default across all prior closures) |
| Paired on | (scenario_id, query_id) |
| Alternative | One-tailed: Δ(AFT − naive_cosine) > 0 |
| α | 0.05 |
| Holm family | m=1 (FR-only slice; no family correction with IT/ES — those are separate pre-regs) |
| Report | Δ, 95% CI (paired bootstrap), p_bootstrap (one-tailed), p_mcnemar, d Cohen, n_discordant, per-challenge breakdown |

---

## Decision rule

**PASS (Branch A):** `p_bootstrap < 0.05` AND `Δ > 0` AND 95% CI lower bound > 0.

**FAIL (Branch B):** otherwise (including `Δ ≤ 0` regardless of p-value).

No threshold adjustment is permitted after observing results.

### Branch declarations

| Outcome | Verdict | Claim matrix action | Notes |
|---|---|---|---|
| **A (PASS)** | Cross-language replication to FR confirmed | Append FR PASS to `retrieval_affect_aware` and `replayable_multi_session_help` `current_evidence`; add `results.v2_fr.me5.*` to `benchmark_refs` | Unexpected positive; update §12 multilingual followup |
| **B (FAIL)** | FR does not reach significance — publishable null | Append FR FAIL to `retrieval_affect_aware` `current_evidence`; add note on cross-language heterogeneity (IT/ES PASS, FR FAIL); update `not_yet_shown` | Consistent with prior FAIL expectation; characterizes AFT as language-selective |

Both branches are honest and publishable. Branch B is the expected outcome.

---

## Coherence with prior closures

- **Hd2 IT/ES (powertopup closure):** Hm1 is an independent replication on a third language;
  it does not modify IT/ES results. The runner, embedder, seed, and metric are identical.
- **Addendum L (query routing):** Independent workstream (LoCoMo QA, not realistic recall).
  No interaction.
- **Hd2 EN (N=200):** English headline unaffected. `bge-small-en` is not used for FR.

---

## Reporting rule

Results are reported regardless of direction. No stopping criterion.
If Branch B: the closure document explicitly calls out the FAIL as a pre-declared expected
outcome and reports effect size, CI, and per-challenge breakdown to characterize the failure mode.

---

## Output files

- `benchmarks/datasets/realistic_recall_v2_fr.json` — 30 hand-authored native FR scenarios
- `benchmarks/realistic/results.v2_fr.me5.json` — full results JSON
- `benchmarks/realistic/results.v2_fr.me5.md` — summary markdown
- `benchmarks/realistic/results.protocol.v2_fr.me5.json` — protocol extract
- `benchmarks/preregistration_addendum_m_fr_closure.md` — post-run closure (NEW, post-run)

---

## Execution commands

```bash
# Step 1: commit this pre-reg BEFORE creating the dataset or running any benchmark
git add benchmarks/preregistration_addendum_m_fr.md
git commit -m "research(addendum-m): pre-register French slice (FR multilingual coverage, FAIL prior)"

# Step 2: hand-author benchmarks/datasets/realistic_recall_v2_fr.json
# (schema identical to realistic_recall_v2_it.json — 30 scenarios, 120 queries, native FR)

# Step 3: validate dataset schema
uv run python -c "
from benchmarks.realistic.runner import load_dataset
from pathlib import Path
d = load_dataset(Path('benchmarks/datasets/realistic_recall_v2_fr.json'))
print(f'Scenarios: {len(d.scenarios)}, top_k: {d.default_top_k}')
"

# Step 4: run benchmark (after committing dataset)
uv run python -m benchmarks.realistic.runner \
  --dataset benchmarks/datasets/realistic_recall_v2_fr.json \
  --embedder multilingual-e5-small \
  --seed 0 \
  --n-bootstrap 10000 \
  --out-json benchmarks/realistic/results.v2_fr.me5.json \
  --out-md benchmarks/realistic/results.v2_fr.me5.md \
  --out-protocol benchmarks/realistic/results.protocol.v2_fr.me5.json
```
