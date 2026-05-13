# Pre-registration Addendum L — Query-Type Routing Confirmatory Study

**Date written:** 2026-05-13
**Protocol version:** addendum_l_v1
**Parent pre-regs:** `benchmarks/preregistration.md` (S1 — LoCoMo primary),
`benchmarks/preregistration_addendum_j.md` (Pareto sweep),
`benchmarks/preregistration_addendum_j_closure.md` (Pareto closure — routing table derived)
**Hypotheses:** Hl1 (confirmatory), Hl2 (secondary), Hl3 (exploratory)
**Closes:** v0.11.0 WS3c

> **Epistemic status:** This is a **prospective pre-registration**. The routing
> table (§Routing table) was derived from Addendum J closure results (committed
> before this document is written) and is treated as a fixed input — it is not
> modified based on Addendum L execution. Addendum J closure is prior work;
> this study tests whether mechanically applying per-query-type weights to a
> classifier-routed retrieval pipeline yields aggregate improvement.
>
> **This document must not be modified after execution begins.** Hypotheses,
> systems, decision rule, and primary outcome metric are frozen at commit time.

---

## Background and motivation

Addendum J Pareto sweep (2026-05-07) identified per-query-type optimal
weight configurations:

| Query type | Best config | F1 vs naive_rag |
|---|---|---|
| single_hop | W7 `[0.70, 0.10, 0.05, 0.05, 0.05, 0.05]` | still below (Δ < 0) |
| multi_hop | W2 `[0.50, 0.30, 0.10, 0.05, 0.05, 0.00]` | approached |
| open_domain | W5 `[0.40, 0.20, 0.15, 0.10, 0.10, 0.05]` | approached |
| temporal | W2 `[0.50, 0.30, 0.10, 0.05, 0.05, 0.00]` | still below |

The Addendum J oracle was run with ground-truth query-type labels. This study
tests the closed-loop pipeline: a classifier (heuristic or LLM) predicts the
query type at retrieval time, selects the matching weights, and retrieves.

**Research question:** Does routing via a query-type classifier reduce the
aggregate AFT/naive_rag gap on LoCoMo QA, compared to the fixed-weight
baseline W2 (the single best Addendum J config)?

---

## Systems

Five systems are evaluated on the same LoCoMo QA subset used in S1:

| System ID | Description |
|---|---|
| `aft_routed_heuristic` | `EmotionalMemory` + `HeuristicQueryClassifier` + `LOCOMO_ROUTING` |
| `aft_routed_llm` | `EmotionalMemory` + `LLMQueryClassifier` + `LOCOMO_ROUTING` |
| `aft_W0` | Fixed default weights (S1 baseline: `[0.35, 0.25, 0.15, 0.10, 0.10, 0.05]`) |
| `aft_W2` | Fixed W2 (best Addendum J config: `[0.50, 0.30, 0.10, 0.05, 0.05, 0.00]`) |
| `naive_rag` | Semantic-only retrieval (S1 baseline, reused if checkpointed) |
| `aft_oracle_routed` | Ground-truth query-type labels applied as routing (upper bound) |

`aft_oracle_routed` is the Addendum J oracle run — included as an upper bound
to bound classifier accuracy loss. It is not confirmatory.

---

## Routing table (frozen)

Derived from Addendum J closure — not modified for this study:

```python
LOCOMO_ROUTING = {
    "single_hop": [0.70, 0.10, 0.05, 0.05, 0.05, 0.05],   # W7
    "multi_hop":  [0.50, 0.30, 0.10, 0.05, 0.05, 0.00],   # W2
    "open_domain": [0.40, 0.20, 0.15, 0.10, 0.10, 0.05],  # W5
    "temporal":   [0.50, 0.30, 0.10, 0.05, 0.05, 0.00],   # W2
    "default":    [0.35, 0.25, 0.15, 0.10, 0.10, 0.05],   # W0
}
```

---

## Hypotheses

### Hl1 — Confirmatory: routing reduces aggregate gap (primary)

**H₀:** Δ_aggregate(aft_routed_heuristic, aft_W2) ≤ 0
**H₁:** Δ_aggregate(aft_routed_heuristic, aft_W2) > 0

Where Δ_aggregate is the difference in sample-weighted mean F1 across all four
LoCoMo QA categories (single_hop, multi_hop, temporal, open_domain), weighted
by category n (same as S1).

**Effect size threshold:** Δ_aggregate > 0.02 (two percentage points F1).
A statistically significant but sub-threshold result is labelled
"statistically significant, not practically meaningful."

**Test:** one-tailed paired bootstrap (n=10,000, seed=42) on per-question F1
differences between `aft_routed_heuristic` and `aft_W2`. α = 0.05,
Holm-corrected for m=2 (Hl1 + Hl2).

**Branch decision rule:**
- **Branch A (Hl1 supported):** p_adj < 0.05 AND Δ > 0.02.
  Interpretation: heuristic routing yields meaningful improvement; ship WS3c
  as a recommended default, document Hl1 as `controlled_evidence` (positive).
- **Branch B (Hl1 not supported):** p_adj ≥ 0.05 OR Δ ≤ 0.02.
  Interpretation: classifier routing does not close the gap meaningfully;
  ship WS3c as an optional feature, document Hl1 as `controlled_evidence`
  (negative). `locomo_external_qa_negative` claim status remains `controlled_evidence`.

### Hl2 — Secondary: routing closes gap vs naive_rag (secondary)

**H₀:** Δ_aggregate(aft_routed_heuristic, naive_rag) ≤ 0
**H₁:** Δ_aggregate(aft_routed_heuristic, naive_rag) > 0

Same bootstrap protocol, α = 0.05, Holm-corrected jointly with Hl1.
This tests whether heuristic routing is sufficient to flip the S1 FAIL.

**Branch decision rule:**
- **Branch A:** p_adj < 0.05 AND Δ > 0 → `locomo_external_qa_negative` status
  changed to `controlled_evidence` (positive). arXiv note updated.
- **Branch B:** otherwise → `locomo_external_qa_negative` remains negative.

### Hl3 — Exploratory: classifier accuracy vs ground-truth (exploratory)

Measure per-query agreement between `HeuristicQueryClassifier` (and
`LLMQueryClassifier`) predicted query type and the LoCoMo ground-truth
query type label.

No pre-specified threshold. Reported as: accuracy, per-category recall, and
confusion matrix. Used to diagnose Hl1 Branch B (if routing fails, is it
because the classifier mislabels or because the routing table itself doesn't
help?).

---

## Sampling protocol

- **Dataset:** LoCoMo QA subset, same 1,540 questions used in S1
  (`benchmarks/locomo/dataset.py` + `benchmarks/locomo/runner.py`).
- **--subset 200qa option:** For a fast smoke-test, `--subset 200qa` samples
  200 questions stratified by category (proportional to S1 category counts).
  Full run is required for confirmatory claims.
- **Seed:** 42 (bootstrap + stratified sampling).
- **Embedder:** `bge-small-en-v1.5` (same as S1 and Addendum J).

---

## Outcome metrics

Primary: sample-weighted mean F1 (consistent with S1 and Addendum J).
Secondary: per-category F1, absolute and Δ vs naive_rag.
Exploratory: classifier accuracy (Hl3).

All results written to:
- `benchmarks/locomo/routing_results.json` — raw per-question predictions
- `benchmarks/locomo/routing_results.md` — formatted summary table

---

## Exclusions and deviations

Any deviation from this protocol must be documented in
`benchmarks/preregistration_addendum_l_query_routing_closure.md` with
justification before the results file is read.

---

## Relationship to existing claims

| Claim ID | Current status | Hl1 Branch A outcome | Hl2 Branch A outcome |
|---|---|---|---|
| `locomo_external_qa_negative` | `controlled_evidence` (negative) | No change (routing helps but EM still trails) | Changed to positive |

---

## Runner invocation (not a pre-registration — for reference)

```bash
python -m benchmarks.locomo.routing_runner \
    --embedder bge-small-en-v1.5 \
    --seed 42 \
    --output benchmarks/locomo/routing_results.json
```

Full run (all 1,540 questions, all 6 systems) is required for confirmatory
claims. The `--subset 200qa` flag is smoke-test only.
