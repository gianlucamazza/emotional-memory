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

| Claim area | Status | Current evidence | Not yet shown | Next study |
|---|---|---|---|---|
| AFT is implemented as a coherent multi-layer memory engine | Implemented | Public API, engine/retrieval/state modules, sync/async parity tests | External validation of architectural value | Keep API/docs aligned |
| Retrieval is affect-aware, not semantic-only | Controlled evidence | Fidelity tests + comparative quadrant benchmark | General downstream superiority | More realistic retrieval dataset |
| The implementation is faithful to the theories it operationalizes | Strong intra-theory evidence | 126 fidelity cases across 20 phenomena | Ecological correspondence to human memory | Human / behavioral validation |
| Appraisal is directionally useful | Early controlled evidence | `benchmarks/appraisal_quality` | Calibration across models/domains/languages | Appraisal robustness study |
| AFT helps on replayable multi-session memory tasks | Early controlled evidence | `benchmarks/realistic` comparative replay benchmark with persisted state, non-trivial candidate pools, and challenge-typed queries | Broad downstream or production superiority; robustness beyond the current small replay set | Expand scenario coverage + run larger evaluation |
| The system models human emotional memory | Not established | Theory-inspired design only | Human behavioral correspondence | Pilot human evaluation |

---

## What is strongest today

- **Strongest evidence**: theory-fidelity benchmarks. These are the clearest
  proof that the code behaves as designed.
- **Second-best evidence**: controlled comparative retrieval results on
  `affect_reference_v1`, especially against semantic-only and recency baselines.
- **Useful but narrow evidence**: appraisal-quality checks and demo-level
  product behavior.
- **Emerging step-4 evidence**: the realistic replay benchmark now rules out
  trivial recency wins and edges `naive_cosine` under the default `sbert-bge`
  embedder (aggregate top1 0.85 vs 0.75, N = 20). With sbert-bge the
  `semantic_confound` subset no longer regresses — AFT ties `naive_cosine`
  on top1 (both 0.62) and leads on hit@k (0.88 vs 0.62). The earlier
  regression on this subset under the hash embedder (AFT 0.12 vs naive 0.25)
  is confirmed as a hash-embedder artefact in
  `benchmarks/realistic/challenge_subset_pairwise.json`. N = 8 on this subset
  is underpowered and no per-challenge result is individually significant
  after Holm correction, so the benchmark is still not decisive as a general
  realistic benchmark.
- **Study-readiness improvement**: the human-eval pilot is now operationally
  specified as a 10-scenario, 2-condition (`aft` vs `naive_cosine`) protocol,
  but still awaits real completed ratings.

---

## What should come next

The next recommended studies, in order:

1. **Protocol upgrade for comparative retrieval**
   Standardize metadata, assumptions, and reporting for the existing benchmark.
2. **Expand the realistic replay benchmark**
   Increase scenario diversity from 10 to at least 50 scenarios to tighten
   per-challenge confidence intervals (the current N = 8 on `semantic_confound`
   after the sbert-bge upgrade is underpowered for definitive conclusions).
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
