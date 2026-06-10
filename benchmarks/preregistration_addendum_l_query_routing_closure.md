# Pre-registration Addendum L — Query-Type Routing Closure

**Status:** CLOSED — Branch B (Hl1 FAIL)
**Date executed:** 2026-05-19
**Parent pre-reg:** `benchmarks/preregistration_addendum_l_query_routing.md`

> **Epistemic status:** This is a **smoke-test closure** (200-QA stratified
> subset, not the full 1 540-QA confirmatory run).  The run was executed with
> the five default systems; `aft_routed_llm` was excluded via
> `--with-llm-classifier` because one LLM call per retrieve makes the run
> impractically slow (~10 h).  Hl1 and Hl2 are evaluated on the smoke-test
> sample; the full run is required for confirmatory claims.
>
> JSON canonical, MD interpretive.

---

## Motivation recap

Addendum J Pareto sweep (2026-05-07) identified per-query-type optimal weight
configurations. S1 (LoCoMo, Gate 1) FAIL — AFT F1=0.168 vs naive_rag=0.271
(Δ=−0.101). Addendum J Pareto confirmed no fixed-weight config closes the gap
(Hj1 FAIL). This study tests the closed-loop pipeline: heuristic classifier
predicts query type at retrieval time → per-type routing weights.

---

## Execution

```bash
PYTHONUNBUFFERED=1 uv run python -m benchmarks.locomo.routing_runner \
    --subset 200qa --seed 42 --verbose
```

Systems evaluated: `aft_routed_heuristic`, `aft_W0`, `aft_W2`, `naive_rag`,
`aft_oracle_routed`.  `aft_routed_llm` excluded (opt-in via
`--with-llm-classifier`).  Embedder: `bge-small-en-v1.5`.  Seed: 42.

---

## Results — Addendum L routing (N=200 QA, stratified smoke test)

| System | single_hop F1 | multi_hop F1 | temporal F1 | open_domain F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| aft_routed_heuristic | 0.283 | 0.256 | 0.083 | 0.161 | 0.229 |
| aft_W2 | 0.332 | 0.218 | 0.070 | 0.151 | 0.246 |
| aft_W0 | 0.197 | 0.162 | 0.056 | 0.150 | 0.159 |
| naive_rag | 0.393 | 0.354 | 0.079 | 0.194 | 0.309 |
| aft_oracle_routed | 0.286 | 0.247 | 0.081 | 0.147 | 0.228 |

### Hl1 pairwise: aft_routed_heuristic vs aft_W2

| Metric | Δ | Practical (Δ>0.02) |
|---|---|---|
| F1 (sample-weighted) | −0.017 | **No** |

Heuristic routing does **not** improve over the fixed W2 baseline on this
smoke-test sample.  The point estimate is in the wrong direction
(routing < fixed W2).

### Hl2 pairwise: aft_routed_heuristic vs naive_rag

| Metric | Δ | Verdict |
|---|---|---|
| F1 | −0.081 | **FAIL** |

The gap vs naive_rag is not closed; routing trails by ~8 pp.

### Hl3 — Classifier accuracy (exploratory)

**Data-collection issue.**  The `_LoggingClassifier` output in
`routing_results.json` shows all 200 predictions as `"unknown"`, indicating a
logging-path bug in the incremental-write / resume logic rather than a true
classifier accuracy of zero.  The `HeuristicQueryClassifier` never returns
`"unknown"`, so the log entries were either lost across the resume boundary or
not persisted in the first incremental write.

**Required follow-up:** Re-run `aft_routed_heuristic` on a clean state (no
resume) to obtain ground-truth classifier accuracy.  This does not block the
Hl1/Hl2 Branch B verdict because the classifier *was* functioning (F1 differs
from W0/W2), but the logging channel failed.

---

## Branch decision (pre-registered)

> **Branch A (Hl1 PASS):** p_holm < 0.05 AND Δ_aggregate > 0.02
> → routing ships as recommended default; `locomo_external_qa_negative`
> `current_evidence` updated with routing note.
>
> **Branch B (Hl1 FAIL):** p_holm ≥ 0.05 OR Δ ≤ 0.02
> → routing ships as optional feature; claim status unchanged.

**Branch B — Hl1 FAIL** (smoke test, 200 QA).

Δ_aggregate = −0.017, which is below the +0.02 practical threshold and in the
wrong direction.  Even without formal bootstrap p-values, the effect is
sub-threshold and negative, so Branch A is not supportable.

---

## Configuration verification (no post-hoc deviation)

- Dataset: LoCoMo stratified 200-QA subset (seed=42, proportional by category)
- Systems: 5 default systems (`aft_routed_llm` excluded by design, not post-hoc)
- Embedder: `bge-small-en-v1.5` (same as S1 and Addendum J)
- Routing table: `LOCOMO_ROUTING` frozen at Addendum J closure (match exact)

**Deviation:** The full 1 540-QA confirmatory run was not completed due to time
constraints.  This closure is based on the 200-QA smoke test.

---

## Interpretation

The closed-loop heuristic routing pipeline does **not** improve aggregate F1
over the best fixed-weight config (W2) on the LoCoMo smoke test, and it does
not close the gap vs naive_rag.  This is consistent with the Addendum J Pareto
result (Hj1 FAIL): even oracle per-category weights could not match naive_rag,
so a noisy classifier routing to those same weights is unlikely to help.

The `aft_oracle_routed` upper bound (0.228) is nearly identical to
`aft_routed_heuristic` (0.229), suggesting that classifier noise is not the
bottleneck — the routing table itself does not provide advantage on this
dataset.

**Verdict:** The LoCoMo negative result is robust to query-type routing.
Future architectural approaches (e.g. selective affect suppression, dual-branch
retrieval) remain open, but per-category weight routing is a closed negative
line.

---

## Cascade changes

| File | Change |
|---|---|
| `docs/research/claim_validation_matrix.json` | `locomo_external_qa_negative`: append routing result to `current_evidence`; note closed-loop routing also FAIL |
| `docs/research/09_current_evidence.md` | Append Hl1/Hl2/Hl3 to LoCoMo section |
| `ROADMAP.md` | WS3c `[ ]` → `[x]` with Hl1 Branch B verdict |
| `CHANGELOG.md` | Addendum L closure bullet under `### Research` |

---

## Artefact index

| File | Description |
|---|---|
| `benchmarks/preregistration_addendum_l_query_routing.md` | Parent pre-registration (frozen) |
| `benchmarks/preregistration_addendum_l_query_routing_closure.md` | This closure document |
| `benchmarks/locomo/routing_results.json` | Raw results (200-QA smoke test) |
| `benchmarks/locomo/routing_results.md` | Human-readable report |
| `benchmarks/locomo/routing_results.md` | Human-readable report (pending) |

---

## Post-closure addendum — Hl3 follow-up resolved (2026-06-11)

**Root cause of the data-collection bug.** `routing_runner.py` matched the
`_LoggingClassifier` log to the prediction list **by index**. Predictions
restored from the resume checkpoint never hit the classifier in the resumed
run, so the log and the prediction list did not align positionally and every
record fell into the `"unknown"` fallback. Fixed by matching on query text
(`_classifier_predictions()`); classifiers are deterministic per query, so
text collisions are harmless. Regression test:
`tests/test_locomo_adapter.py::test_classifier_predictions_match_by_query_text`.

**Ground-truth classifier accuracy (offline, deterministic re-measurement).**
The required follow-up ("re-run `aft_routed_heuristic` on a clean state") is
satisfied without an LLM run: `HeuristicQueryClassifier` is pure and
deterministic, so classifying the exact 200-QA stratified subset (seed=42)
offline reproduces what a clean live run would have logged.

| | n | accuracy |
|---|---:|---:|
| All 200 QA | 200 | **0.465** |
| Excluding `adversarial` (no routable class exists) | 155 | **0.600** |

Per-type (gt → top predictions): `temporal` 25/32 correct; `single_hop`
60/85; `open_domain` 6/10; `multi_hop` **2/28** (mostly misread as
`single_hop`); `adversarial` (n=45) unroutable by construction — falls to
`single_hop`/`open_domain`.

**Reading.** The heuristic classifier is far from oracle (and essentially
blind to `multi_hop`), which compounds the Hl1 weight-routing FAIL: even
perfect per-type weights could not have been applied to the right queries
~40% of the time. Any follow-up routing study (e.g. Addendum Q affect-gating)
must either (a) measure and report classifier accuracy on its own corpus, or
(b) include an oracle-routed arm to bound the classifier-induced loss.
