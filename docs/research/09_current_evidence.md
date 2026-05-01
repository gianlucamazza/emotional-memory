# Current Evidence and Study Ladder

This page summarizes what the project currently supports, what is validated, and
which study steps come next. It is meant to keep implementation status, public
claims, and research ambition aligned.

---

## Study ladder

The project now uses a five-step evidence ladder:

1. **Theory fidelity**: does the implementation behave coherently with the
   psychological/computational theory it encodes?
2. **Controlled retrieval behavior**: does the retrieval engine change ranking
   in the intended affect-aware direction under a constrained protocol?
3. **Appraisal quality**: does the appraisal layer produce directionally useful
   outputs on natural-language inputs?
4. **Realistic memory tasks**: does the system help on multi-session, agent-like
   scenarios where memory must persist and update over time?
5. **Human / ecological validation**: do humans perceive better affective
   coherence or utility, and does the behavior track human-like emotional memory
   in realistic settings?

Current repo strength is concentrated in steps **1–4**. Step **4** now has
controlled evidence: the v2 realistic replay benchmark (N=200, 5 challenge
types × 40) shows a decisive advantage over `naive_cosine` on both SBERT and
e5-small-v2 embedders. Step **5** remains open research work.

---

## Claim matrix

Canonical source: [`claim_validation_matrix.json`](claim_validation_matrix.json).
This JSON file is the repo's versioned source of truth for claim wording,
evidence level, and still-open gaps.

**Status legend**

- `Implemented`: present in the codebase, but not externally validated as
  valuable.
- `Controlled evidence`: supported under a narrow, documented benchmark
  protocol.
- `Strong intra-theory evidence`: strongly supported inside the theory the
  implementation operationalizes.
- `Early controlled evidence`: initial task or appraisal evidence exists, but
  remains narrow.
- `Not established`: not supported by current public evidence.

| ID | Claim area | Status | Evidence level | Allowed public wording | Current evidence | Not yet shown | Next study |
|---|---|---|---|---|---|---|---|
| `aft_multilayer_engine` | Architecture | Implemented | `1_theory_fidelity` | AFT is implemented as a coherent multi-layer memory engine. | Public API, engine/retrieval/state modules, sync/async parity tests. | External validation of architectural value. | Keep API/docs aligned while external evaluations grow. |
| `retrieval_affect_aware` | Retrieval behavior | Controlled evidence | `2_controlled_retrieval` | Retrieval is affect-aware, not semantic-only. | 126 fidelity cases validate affect-aware ranking logic. The controlled quadrant benchmark (SBERT) ties AFT and naive_cosine at recall@5 = 0.80 (ceiling effect, N = 20 items), while clearly beating recency (Δ = −0.55, p < 0.001). The realistic replay benchmark (SBERT) shows AFT top1 = 0.70 vs naive_cosine = 0.50 (N = 100). | General downstream superiority over production memory systems. | Expand external and realistic retrieval evaluations. |
| `theory_faithful_operationalization` | Theory fidelity | Strong intra-theory evidence | `1_theory_fidelity` | The implementation is faithful to the theories it operationalizes. | 126 fidelity cases across 20 phenomena validate expected intra-theory behavior. | Ecological correspondence to human emotional memory. | Human and behavioral validation. |
| `appraisal_directionally_useful` | Appraisal quality | Early controlled evidence | `3_appraisal_quality` | Appraisal is directionally useful. | The appraisal-quality benchmark provides early controlled evidence on natural-language inputs. | Calibration across models, domains, and languages. | Appraisal robustness study across models, domains, and languages. |
| `replayable_multi_session_help` | Realistic tasks | Controlled evidence | `4_realistic_tasks` | AFT helps on replayable multi-session memory tasks (v2, N=200, SBERT: AFT top1=0.53 vs naive_cosine=0.33, Δ=+0.21 [0.15,0.27], p<0.001; e5-small-v2: AFT top1=0.50 vs 0.34, Δ=+0.16 [0.09,0.22], p<0.001). | Realistic replay v2 (50 scenarios, 200 queries). SBERT: AFT 0.53 vs naive 0.33, Δ=+0.21 p<0.001 d=0.49. e5-small-v2: AFT 0.50 vs naive 0.34, Δ=+0.16 p<0.001 d=0.31. | Architecture attribution (appraisal confound pending); multilingual; external open-domain QA (LoCoMo Gate 1 FAIL). | Run appraisal confound. Add Italian slice (PR-F). |
| `locomo_external_qa_negative` | External benchmarks | Controlled evidence | `4_realistic_tasks` | On the LoCoMo conversational QA benchmark (1986 QA pairs), AFT retrieval underperforms a naive RAG baseline (F1 0.168 vs 0.271; Gate 1 not met). | Pre-registered S1 run completed 2026-04-27. H1 and H2 both fail (Δ<0, p_one=1.0). | Whether task-specific retrieval weight tuning could close the gap. | Investigate per-task configuration; document as scope limitation in paper. |
| `models_human_emotional_memory` | Ecological validity | Not established | `5_human_ecological` | The system is theory-inspired, but does not yet have human or ecological validation. | Theory-inspired design only. | Human behavioral correspondence. | Pilot human evaluation with completed ratings and external benchmarks. |

---

## What is strongest today

- **Strongest evidence**: theory-fidelity benchmarks. These are the clearest
  proof that the code behaves as designed.
- **Second-best evidence**: realistic replay v2 benchmark (N=200, 5 challenge
  types × 40). SBERT bge-small: AFT top1=0.53 vs naive_cosine=0.33,
  Δ=+0.205 [0.150,0.265], p_bootstrap<0.001, d=0.49. e5-small-v2: AFT
  top1=0.50 vs 0.34, Δ=+0.155 [0.090,0.225], p_bootstrap<0.001, d=0.31.
  The advantage holds on both SBERT and e5-small-v2 (two distinct
  embedder classes), addressing G5. The controlled quadrant probe
  (`affect_reference_v1`) ties AFT and naive_cosine at SBERT ceiling (both 0.80)
  but confirms the benchmark discriminates clearly against the recency baseline.
- **Useful but narrow evidence**: appraisal-quality checks and demo-level
  product behavior.
- **Step-4 evidence upgraded to controlled**: v2 (50 scenarios, 200 queries)
  shows decisive aggregate advantage on both embedder classes. Per-challenge
  breakdown (SBERT): semantic_confound 0.72 vs 0.47, affective_arc 0.42 vs 0.15,
  momentum_alignment 0.60 vs 0.33, same_topic_distractor 0.75 vs 0.62,
  recency_confound 0.15 vs 0.05. Architecture attribution still pending
  (appraisal confound unresolved).
- **Negative external result (Gate 1, 2026-04-27)**: On the LoCoMo conversational QA benchmark (1986 QA pairs, 10 conversations), AFT retrieval underperforms a naive RAG baseline (F1 0.168 vs 0.271; judge_acc 0.279 vs 0.441). Gate 1 was not met. AFT's affective weighting does not help on factual open-domain QA. See `locomo_external_qa_negative` in `claim_validation_matrix.json`.
- **Study-readiness improvement**: the human-eval pilot is now operationally
  specified as a 10-scenario, 2-condition (`aft` vs `naive_cosine`) protocol,
  but still awaits real completed ratings.

---

## What should come next

The next recommended studies, in order:

1. **Protocol upgrade for comparative retrieval**
   Standardize metadata, assumptions, and reporting for the existing benchmark.
2. ~~**Expand the realistic replay benchmark**~~
   *Completed (v2.0): 50 scenarios / 200 queries; decisive advantage on SBERT
   and e5-small-v2 (both p_bootstrap<0.001). G4 + G5 addressed.*
3. **Execute the human-eval pilot with completed ratings**
   Collect ratings on coherence, usefulness, continuity, and plausibility from
   at least 3 raters (Krippendorff's alpha is wired and reported automatically
   when `ratings.jsonl` is filled).
4. ~~**Run LoCoMo end-to-end for external benchmark validation**~~
   *Completed 2026-04-27 (Gate 1 FAIL): `benchmarks/locomo/results.json` committed.
   AFT F1=0.168 vs naive_rag F1=0.271 on 1986 QA pairs; both H1 and H2 fail
   Holm correction. Negative result — affective weighting does not improve
   open-domain factual QA. Claim ceiling unchanged; see `locomo_external_qa_negative`
   in `claim_validation_matrix.json`.*

---

## Reading guide

- For the theoretical motivation: [Design Principles](05_design_principles.md)
- For neighboring systems: [State of the Art](04_state_of_art.md) and
  [Related Work](07_related_work.md)
- For known limitations: [Limitations](08_limitations.md)
