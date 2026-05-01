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

Current repo strength is concentrated in steps **1–3**. Step **4** now has
initial infrastructure in the repo (persistent affective state plus a replayable
multi-session benchmark), but the evidence there is still early. Step **5**
remains open research work.

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
| `replayable_multi_session_help` | Realistic tasks | Early controlled evidence | `4_realistic_tasks` | AFT helps on replayable multi-session memory tasks. | Realistic replay (SBERT, N=100): AFT top1=0.70 vs naive_cosine=0.50. Appraisal confound (Hd1 PASS, seed=1): aft_noAppraisal=0.78 > naive_cosine=0.55, Δ=+0.23 — advantage confirmed as architectural (Gate 3 CLOSED 2026-04-27). | Broad downstream or production superiority; cross-embedder generalization; N≥200 power. | realistic_recall_v2 (N≥200, 5 challenge types, cross-embedder slice). |
| `models_human_emotional_memory` | Ecological validity | Not established | `5_human_ecological` | The system is theory-inspired, but does not yet have human or ecological validation. | Theory-inspired design only. | Human behavioral correspondence. | Pilot human evaluation with completed ratings and external benchmarks. |

---

## What is strongest today

- **Strongest evidence**: theory-fidelity benchmarks. These are the clearest
  proof that the code behaves as designed.
- **Second-best evidence**: realistic replay benchmark (SBERT, N = 100):
  AFT top1 = 0.70 vs naive_cosine = 0.50. The controlled quadrant probe
  (`affect_reference_v1`) ties AFT and naive_cosine at SBERT ceiling (both 0.80)
  but confirms the benchmark discriminates clearly against the recency baseline.
- **Useful but narrow evidence**: appraisal-quality checks and demo-level
  product behavior.
- **Emerging step-4 evidence**: the realistic replay benchmark (v1.4, expanded
  to 50 scenarios / 100 queries) rules out trivial recency wins and edges
  `naive_cosine` under the default `sbert-bge` embedder (aggregate top1 0.70
  vs 0.50, N = 100). On the `semantic_confound` subset (N = 30), AFT top1
  reaches 0.73 vs naive 0.47, Δ = +0.27 [0.10, 0.43], p_adj = 0.006 after
  Holm correction — the first per-challenge result to survive correction. The
  earlier regression on this subset under the hash embedder (AFT 0.12 vs
  naive 0.25) is confirmed as a hash-embedder artefact in
  `benchmarks/realistic/challenge_subset_pairwise.json`.
- **Study-readiness improvement**: the human-eval pilot is now operationally
  specified as a 10-scenario, 2-condition (`aft` vs `naive_cosine`) protocol,
  but still awaits real completed ratings.

---

## What should come next

The next recommended studies, in order:

1. **Protocol upgrade for comparative retrieval**
   Standardize metadata, assumptions, and reporting for the existing benchmark.
2. ~~**Expand the realistic replay benchmark**~~
   *Completed (v1.4): 50 scenarios / 100 queries; `semantic_confound` N = 30
   now yields p_adj = 0.006 after Holm correction.*
3. **Execute the human-eval pilot with completed ratings**
   Collect ratings on coherence, usefulness, continuity, and plausibility from
   at least 3 raters (Krippendorff's alpha is wired and reported automatically
   when `ratings.jsonl` is filled).
4. **Run LoCoMo end-to-end for external benchmark validation**
   The adapter is ready (`make bench-locomo`); execution needs an LLM API key
   and a modest budget for the judge pass.

---

## Reading guide

- For the theoretical motivation: [Design Principles](05_design_principles.md)
- For neighboring systems: [State of the Art](04_state_of_art.md) and
  [Related Work](07_related_work.md)
- For known limitations: [Limitations](08_limitations.md)
