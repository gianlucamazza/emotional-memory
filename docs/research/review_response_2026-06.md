# Response to External Review (June 2026)

A point-by-point reconciliation of an external critical review of AFT against the
**current** state of the repository. The review is methodologically sound and its
severity ordering is reasonable, but it was written against an earlier snapshot:
the large majority of its criticisms and roadmap items are **already
implemented and committed** — most as pre-registered studies with honest
PASS/FAIL closures. The purpose of this document is to (a) map each review point
to the artifact that already answers it, and (b) isolate the small residue that
is _genuinely_ open so it does not get lost.

This is a synthesis/citation document. It introduces no code, runs no new
experiments, and changes no claims; live claim status remains governed by
[`claim_validation_matrix.json`](claim_validation_matrix.json) and
[`09_current_evidence.md`](09_current_evidence.md).

---

## 1. Snapshot

The review's central worry — _"a Δ in cosine similarity is not better memory"_ —
is already the explicit organizing principle of the corpus. AFT's claim ceiling
is pre-committed in [`10_scientific_quality_bar.md`](10_scientific_quality_bar.md)
as **"strong theory-driven prototype with early controlled evidence + one honest
external negative,"** and the negative results the review fears a hostile reviewer
would raise (LoCoMo, DailyDialog, end-to-end appraisal) are all committed FAILs
with closures.

Of the seven review criticisms (§3.1–§3.7), five are **resolved or honestly
scoped** and two (§3.4 construct validity, §3.5 multi-seed) are **partially
open**. Importantly, "honestly scoped" is _not_ the same as "solved": §3.1's
underlying downstream gap is a committed FAIL that has been bounded, not closed.
The real residue is four items, tracked in §5 below and itemized in the companion
[`problem_register_2026-06.md`](problem_register_2026-06.md).

## 2. Verdict at a glance

**Status legend.** **Resolved** = the criticism no longer applies.
**Honestly scoped** = the underlying problem is a committed FAIL / open gap that is
documented and claim-bounded, _not_ fixed. **Partially open** = some of it is
addressed, some is open.

| Review point                             | Severity (as filed) | Repo status                                                  | Primary evidence                                                               |
| ---------------------------------------- | ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| §3.1 embedding→downstream / LoCoMo       | CRITICA             | **Honestly scoped, not solved** (downstream FAIL committed)  | `08_limitations.md`, `benchmarks/locomo/`, Addenda J/L/Q closures              |
| §3.2 ablation of the mechanism           | ALTA                | **Resolved**                                                 | `benchmarks/ablation/runner.py` (8 variants), `runner_hi3.py`                  |
| §3.3 "field" framing earns its place     | ALTA                | **Resolved** (by under-claiming: capped "prototype" ceiling) | `01_foundations.md`, `05_design_principles.md`, `10_scientific_quality_bar.md` |
| §3.4 construct validity of affect signal | MEDIA               | **Partially open**                                           | Addenda N/O/P; `human_eval/` built, never run                                  |
| §3.5 power / seeds / CIs                 | MEDIA               | **Partially open** (CIs done; multi-seed sweep absent)       | `benchmarks/common/statistics.py`                                              |
| §3.6 positioning vs prior art            | BASSA               | **Resolved**                                                 | `07_related_work.md` (29 systems), `comparison.md`                             |
| §3.7 bus-factor / sustainability         | BASSA               | **Acknowledged** (out of scope for evidence)                 | CI/supply-chain hardening as partial mitigation                                |

## 3. Point-by-point response

### §3.1 — embedding-space → downstream (the LoCoMo FAIL) — _Honestly scoped, not solved_

This is the review's CRITICA, and it is the axis the project has worked hardest
on. The gap is not undiscovered — it is **committed, externally validated, and
framed as regime specificity**, not swept aside:

- **S1 / LoCoMo Gate 1 (FAIL):** AFT F1 = 0.168 vs naive_rag F1 = 0.271
  (Δ = −0.101). Both pre-registered H1/H2 FAIL. See `benchmarks/locomo/` and
  [`08_limitations.md`](08_limitations.md).
- **Addendum J — Pareto weight sweep (Hj1 FAIL):** no fixed per-task weight
  configuration reaches naive_rag parity on any category (best W2 aggregate
  F1 = 0.1765 vs 0.2092). `preregistration_addendum_j_closure.md`.
- **Addendum L — closed-loop query routing (Hl1/Hl2 FAIL):** heuristic routing
  does not close the gap — Hl1 Δ = −0.017 (below the +0.02 practical threshold),
  Hl2 vs naive_rag Δ = −0.081. `preregistration_addendum_l_query_routing_closure.md`.
- **Addendum Q — affect-aware gating (Branch C FAIL):** the deepest finding —
  the oracle-affect boundary is also a **state-injection boundary**. Even on the
  affective subset, `aft_llm_dual` loses to cosine (Hq1 Δ = −0.050); gating only
  _recovers_ cosine on the affect-free half (Hq2 PASS, +0.080) rather than
  beating it. "No routing scheme can rescue a channel that loses in its own
  regime." `preregistration_addendum_q_affect_gating_closure.md`.

The review's "prudent path" (§4.1 — claim the embedding-space level explicitly,
frame LoCoMo as scope) is therefore **already the adopted position**. The
"ambitious path" (a downstream end-to-end task) remains genuinely open — see §5.

### §3.2 — ablation of the mechanism — _Resolved_

The review asks whether AFT adds anything over recency/salience weighting. The
ablation harness exists and isolates every affective layer:

- Config toggles `enable_appraisal`, `enable_mood_signal`, `enable_momentum`,
  `enable_resonance`, `enable_reconsolidation` are defined at
  `src/emotional_memory/engine.py` L85–97 and wired into retrieval via weight
  masking (L246–250) and conditional branches (encode/resonance L338).
- `benchmarks/ablation/runner.py` (L210+) runs **8 variants** —
  `full`, `no_appraisal`, `no_mood`, `no_momentum`, `no_resonance`,
  `no_reconsolidation`, `dual_path`, `aft_keyword_synchronous` — each scored vs
  `full` with paired bootstrap, McNemar exact, and Hedges-corrected Cohen's d.
- Confirmatory layer isolation: `runner_hi3.py` (N = 500, Holm-corrected, m = 3),
  resonance amplification on `semantic_confound` Δ = +0.090, d = 0.257,
  Holm p = 0.0234 (`preregistration_addendum_i_closure.md`).

### §3.3 — does "Affective Field Theory" earn the name? — _Resolved (by under-claiming)_

The framing decision the review asks for (§4.3: formalize the field _or_ demote
it to declared metaphor) has effectively been made on the side of **disciplined,
theory-faithful operationalization with a capped claim**:

- The 5 layers carry formal definitions and source citations in
  [`05_design_principles.md`](05_design_principles.md) and
  [`01_foundations.md`](01_foundations.md); uniqueness vs prior art is argued in
  [`07_related_work.md`](07_related_work.md) §6.
- Crucially, the claim is **not** inflated to "field theory makes novel
  predictions": the public ceiling in
  [`10_scientific_quality_bar.md`](10_scientific_quality_bar.md) is a
  "theory-driven prototype." The ambiguity the review warns against is closed by
  _under_-claiming, which is defensible.

### §3.4 — construct validity of the affective signal — _Partially open_

This is the review's most durable point. The chain "text → affect signal → weight
→ retrieval" has been probed at the first link, and the result is honest but
**not closed**:

- **Diagnosed, not blind:** Addendum N found LLM appraisal _mis-calibrated, not
  random_ — valence Pearson r = 0.883 vs gold, but +0.169 bias; prompt
  recalibration **FAILED** (Hn1/Hn2 FAIL — valence bias fell to +0.044 but
  arousal bias would not move). `preregistration_addendum_n_appraisal_calibration_closure.md`.
- **Mapping fix worked, retrieval still lost:** Addendum O recalibrated the
  SEC→affect projection (model M1 PASS — valence bias +0.200→+0.072, arousal
  −0.144→−0.023). But Addendum P then showed the better signal is a **net
  distractor** on affect-free queries: `aft_llm_dual` Δ = −0.087 vs cosine
  (p = 0.0018, d = −0.242). `preregistration_addendum_{o,p}_*_closure.md`.
- **The genuine gap:** there are still **no human gold affect ratings**. The
  human-eval kit (`benchmarks/human_eval/`: protocol, rater instructions,
  packets, Krippendorff-α pipeline) is built but has **zero collected ratings**
  (Gate 2 OPEN). All validation to date is intra-theoretical, not ecological
  ([`08_limitations.md`](08_limitations.md) §2.1, §2.4).

So the review is right that this link is the weakest — but the work has advanced
from "undocumented" to "diagnosed and quantified," and the remaining task is
human annotation, not analysis. Tracked in §5.

### §3.5 — power, seeds, confidence intervals — _Partially open_

The statistical machinery the review asks for already exists and is used
throughout:

- `benchmarks/common/statistics.py`: `bootstrap_ci`, `paired_bootstrap_diff`,
  `mcnemar_exact`, `holm_bonferroni`, `cohens_d_paired` (Hedges g). No scipy
  dependency.
- CIs and effect sizes are reported on the headline results, e.g. realistic
  recall v2 SBERT Δ = +0.205 [0.150, 0.265], d = 0.49; e5 Δ = +0.155, d = 0.31.
- The review's §4.5 ask for a **third, non-Romance-independent** language is
  partially met: French is a committed PASS (Addendum M — Δ = +0.18 [0.11, 0.26],
  Hedges g = 0.424), and the IT/ES powered runs are committed **FAILs**
  (Hd2-PowerTopUp: IT Δ = +0.058 p = 0.276; ES Δ = 0.000 p = 1.000) — i.e. the
  generalization limit is already mapped, not assumed.
- **Open edge:** most runs pin a single seed; there is no automated _multi-seed
  robustness sweep_ reporting cross-run variance. This is the one concrete piece
  of §3.5 still worth doing (§5).

### §3.6 — positioning vs prior art — _Resolved_

[`07_related_work.md`](07_related_work.md) is a 29-system comparative table
(MemGPT/Letta, Mem0, Generative Agents, A-MEM, SYNAPSE, Emotional RAG, LUFY,
DAM-LLM, …) on the exact axes the review requests (importance model, decay,
reconsolidation, affect layer); [`comparison.md`](../comparison.md) and
[`04_state_of_art.md`](04_state_of_art.md) cover the systems-level comparison.
The review's §4.6 "positioning table" already exists.

### §3.7 — bus-factor / sustainability — _Acknowledged_

A solo-author sustainability risk is real and outside the evidence scope of this
response. Partial mitigation already in place: SLSA provenance, CycloneDX SBOM,
CodeQL, pinned/`uv`-locked builds, and a DOI lower the cost of independent
pickup and reproduction.

## 4. Improvements (§4) and open decisions (§5) — mapping

| Review item                        | Status in repo                                        |
| ---------------------------------- | ----------------------------------------------------- |
| §4.1 downstream bridge (ambitious) | **Open** — no end-to-end task yet (see §5)            |
| §4.1 LoCoMo as scope (prudent)     | **Done** — adopted position (§3.1)                    |
| §4.2 affective ablation            | **Done** — `benchmarks/ablation/`                     |
| §4.3 formalize-or-demote the field | **Decided** — capped "prototype" claim                |
| §4.4 construct-validity section    | **Partial** — diagnosed (N/O/P); human gold open      |
| §4.5 CIs + seed count              | **Partial** — CIs done; multi-seed sweep open         |
| §4.5 third non-Romance language    | **Partial** — FR PASS; IT/ES FAIL committed           |
| §4.6 positioning table             | **Done** — `07_related_work.md`                       |
| §5.1 central claim                 | **Decided** — affect-discriminative retrieval, capped |
| §5.2 LoCoMo: scope or failure      | **Decided** — scope boundary, committed FAIL          |
| §5.3 field: theory or metaphor     | **Decided** — theory-faithful, under-claimed          |
| §5.4 venue                         | Author decision — out of scope here                   |

## 5. What is genuinely open (the real residue)

**Executed since this response was written (June 2026):** items 2, 3 and 4 below
have been carried out.

- ✅ **Construct validity vs human-gold (Addendum S, #62 closed).** The appraisal
  signal was validated against human-annotated affect (EmoBank, N=300): LLM valence
  is human-validated (r=0.70 [0.66, 0.75]); arousal/dominance are weak-positive; the
  +0.15 valence bias persists; the keyword engine is not human-validated.
- ✅ **Downstream end-to-end value (Addendum R, #61 closed).** The
  encode→retrieve→generate→judge task moved an end-to-end metric in the
  affect-discriminative (oracle-affect) regime: AFT judge-accuracy 0.595 vs cosine
  0.440 (Δ=+0.155, p<0.001). It does **not** reframe the LoCoMo narrative — that FAIL
  is the oracle-free regime, and the two coexist.
- ✅ **Multi-seed robustness sweep (#60 closed).** `make bench-multiseed` ships; the
  result is characterized as near-deterministic (sub-CI, timing-driven variance).

The single highest-value gap that remains genuinely open:

1. **Ecological / human validation (Gate 2, #27).** Execute the existing
   `benchmarks/human_eval/` kit to collect real ratings and report inter-rater
   agreement. (Note: this is human _perceived utility_ — distinct from Addendum S,
   which validates the appraisal _signal_ against human affect labels.)

## 6. Closing

The review's strongest advice — _let epistemic honesty be the protagonist_ — is
already the corpus's operating principle: pre-registration, committed FAILs
(LoCoMo, DailyDialog, end-to-end appraisal, affect-gating), an explicit claim
ceiling, and a machine-readable claim matrix. Measured against the current
repository, the review is best read not as a list of unmet criticisms but as
independent confirmation that the project anticipated them — with the residue
narrowed to the four ecological/downstream items in §5.
