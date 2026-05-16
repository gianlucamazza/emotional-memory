# Pre-registration Addendum L — Query-Type Routing Closure

**Status:** ⏳ PENDING (routing_results.json not yet produced)
**Date executed:** 2026-05-16
**Parent pre-reg:** `benchmarks/preregistration_addendum_l_query_routing.md`

> **Epistemic status:** [FILL once routing_results.json arrives].
> JSON canonical, MD interpretive.

---

## Motivation recap

Addendum J Pareto sweep (2026-05-07) identified per-query-type optimal weight
configurations. S1 (LoCoMo, Gate 1) FAIL — AFT F1=0.168 vs naive_rag=0.271
(Δ=−0.101). Addendum J Pareto confirmed no fixed-weight config closes the gap
(Hj1 FAIL). This study tests the closed-loop pipeline: heuristic or LLM
classifier predicts query type at retrieval time → per-type routing weights.

---

## Execution

```bash
PYTHONUNBUFFERED=1 uv run python -m benchmarks.locomo.routing_runner
# Outputs: benchmarks/locomo/routing_results.json
#           benchmarks/locomo/routing_results.md
```

Systems: `aft_routed_heuristic`, `aft_routed_llm`, `aft_W0`, `aft_W2`,
`naive_rag`, `aft_oracle_routed`. Bootstrap: n=10,000, seed=42, Holm m=2.

---

## Results — Addendum L routing (N=1540 QA, LoCoMo full set)

[FILL from routing_results.md once available]

| System | F1 | Judge accuracy |
|---|---|---|
| aft_routed_heuristic | [FILL] | [FILL] |
| aft_routed_llm | [FILL] | [FILL] |
| aft_W2 | [FILL] | [FILL] |
| aft_W0 | [FILL] | [FILL] |
| naive_rag | [FILL] | [FILL] |
| aft_oracle_routed | [FILL] | [FILL] |

### Hl1 pairwise: aft_routed_heuristic vs aft_W2

| Metric | Δ | 95% CI | p_bootstrap | p_holm | Practical (Δ>0.02) |
|---|---|---|---|---|---|
| F1 (sample-weighted) | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |

### Hl2 pairwise: aft_routed_heuristic vs naive_rag

| Metric | Δ | 95% CI | p_bootstrap | p_holm | Verdict |
|---|---|---|---|---|---|
| F1 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |

### Hl3 — Classifier accuracy (exploratory)

[FILL: heuristic vs LLM accuracy breakdown from _LoggingClassifier output]

---

## Branch decision (pre-registered)

> **Branch A (Hl1 PASS):** p_holm < 0.05 AND Δ_aggregate > 0.02
> → routing ships as recommended default; `locomo_external_qa_negative`
> `current_evidence` updated with routing note.
>
> **Branch B (Hl1 FAIL):** p_holm ≥ 0.05 OR Δ ≤ 0.02
> → routing ships as optional feature; claim status unchanged.

**[FILL: Branch X — Verdict: PASS / FAIL]**

---

## Configuration verification (no post-hoc deviation)

- Dataset: LoCoMo full set, same 1986 QA pairs as S1 (match pre-reg §Dataset)
- Bootstrap seed: 42 (match exact)
- n_bootstrap: 10,000 (match exact)
- Holm correction: m=2 (Hl1 + Hl2, match exact)
- Routing table: `LOCOMO_ROUTING` frozen at Addendum J closure (match exact)

---

## Interpretation

[FILL once results known]

---

## Cascade changes

[FILL based on Branch decision]

| File | Change |
|---|---|
| `docs/research/claim_validation_matrix.json` | `locomo_external_qa_negative`: append routing result to `current_evidence`; `benchmark_refs` + `protocol_refs` |
| `docs/research/09_current_evidence.md` | Append Hl1/Hl2/Hl3 to LoCoMo section |
| `ROADMAP.md` | WS3c `[ ]` → `[x]` with Hl1 verdict |
| `CHANGELOG.md` | Addendum L bullet under `### Research` |
| `paper/main.tex` (conditional) | If Hl2 PASS: update §External-benchmark scope (l.685-705) |

---

## Artefact index

| File | Description |
|---|---|
| `benchmarks/preregistration_addendum_l_query_routing.md` | Parent pre-registration (frozen) |
| `benchmarks/locomo/routing_results.json` | Raw bootstrap results (pending) |
| `benchmarks/locomo/routing_results.md` | Human-readable report (pending) |
| `benchmarks/locomo/routing_results.md` | Human-readable report (pending) |
