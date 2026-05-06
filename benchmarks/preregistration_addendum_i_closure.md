# Pre-Registration Addendum I — Hi3 Closure (Resonance Amplification Confirmatory)

**Status:** PENDING_EXECUTION
**Template written:** 2026-05-06
**Date executed:** {{date_executed}}
**Protocol version:** addendum_i_closure_v1
**Parent pre-reg:** `benchmarks/preregistration_addendum_i.md` (Hi3 frozen design)

> **Epistemic status:** This is a pre-specified closure template for the Hi3
> confirmatory family. Sections marked `{{placeholder}}` must be filled verbatim
> from `results.hi3.{json,md}` after execution. The two interpretive branches
> (PASS / FAIL on the primary Hi3) are pre-registered and must not be modified
> after results are known. All numeric fields are placeholders — zero values
> committed at the time of this template.

---

## Background

Addendum S3 (closure 2026-05-04) reported `Hb` FAIL on `realistic_recall_v2`
(N=200) for both embedders, with e5 showing a +0.075 aggregate amplification
vs SBERT's +0.030 — a ~2.5× embedder-dependent amplification of resonance's
interference effect. Per-challenge decomposition (Hi1, post-hoc descriptive)
located the dominant amplification channel in `semantic_confound` (e5 −
SBERT = +0.100), with secondary channels in `recency_confound` and
`affective_arc` (each +0.050).

Hi3 (Add. I) was pre-registered as a deferred confirmatory study to test
whether the v2 per-challenge amplification on `semantic_confound` is stable
at N ≥ 500 (half the v2 estimate, +0.05, after regression-to-mean correction),
and to extend the test as a 3-hypothesis Holm-corrected family with
`recency_confound` and `affective_arc` as secondaries.

Infrastructure committed prior to execution:
- `realistic_recall_v3.json` (N=500, author-blind, commit `320cd23`)
- Ablation runner per-query records (`--per-query-records` flag, `d858be6`)
- `runner_hi3.py` analyzer (Holm m=3, paired bootstrap seed=1, `abe0263`)
- Makefile targets `bench-hi3-{sbert,e5,analyze}` (`c548dd9`/`abe0263`)

---

## Execution

```bash
make install-sentence-transformers   # one-time, ~130 MB
make bench-hi3-sbert                  # ~15-25 min CPU
make bench-hi3-e5                     # ~15-25 min CPU
make bench-hi3-analyze                # <1 min — produces results.hi3.{json,md,protocol.json}
```

Parameters (frozen in `runner_hi3.py`):
- seed: **1** (frozen at instrumentation commit; seed=0 reserved for v2 replication)
- n_bootstrap: 10,000
- α: 0.05 (one-tailed, directional)
- Δ threshold: 0.05 (5 pp)
- Holm family (m=3): {Hi3 (semantic_confound, primary), Hi3_recency, Hi3_arc}
- Statistic per hypothesis:
  `delta = mean(amp_e5) - mean(amp_sbert)`
  where `amp[i] = no_resonance_top1_hit[i] - full_top1_hit[i]`
  (positive = resonance hurts; double-difference = e5 interference amplification over SBERT;
  matches pre-reg §Hi3 sign convention: Δ = no_resonance − full)
- Dataset: `realistic_recall_v3.json` (N={{n_queries}} queries)

Output files:
- `benchmarks/ablation/results.v3.sbert.json`, `results.v3.e5.json`
- `benchmarks/ablation/results.hi3.{json,md,protocol.json}`

---

## Results

*To be filled verbatim from `results.hi3.md` after `make bench-hi3-analyze`.*

### Confirmatory family (Holm-corrected, m=3)

| Hypothesis | Challenge | Δ [95% CI] | p_one | p_adj (Holm) | Cohen's d | Verdict |
|---|---|---|---|---|---|---|
| **Hi3** (primary) | semantic_confound | {{hi3_delta}} [{{hi3_ci_lo}}, {{hi3_ci_hi}}] | {{hi3_p_one}} | {{hi3_p_adj}} | {{hi3_d}} | **{{hi3_verdict}}** |
| Hi3_recency | recency_confound | {{recency_delta}} [{{recency_ci_lo}}, {{recency_ci_hi}}] | {{recency_p_one}} | {{recency_p_adj}} | {{recency_d}} | {{recency_verdict}} |
| Hi3_arc | affective_arc | {{arc_delta}} [{{arc_ci_lo}}, {{arc_ci_hi}}] | {{arc_p_one}} | {{arc_p_adj}} | {{arc_d}} | {{arc_verdict}} |

### Aggregate per-embedder reference

| Embedder | full top1 | no_resonance top1 | aggregate amp |
|---|---|---|---|
| sbert-bge | {{sbert_full}} | {{sbert_no_res}} | {{sbert_agg_amp}} |
| e5-small-v2 | {{e5_full}} | {{e5_no_res}} | {{e5_agg_amp}} |

### Mechanism (exploratory, descriptive — no formal test)

`link_set_stats` from `results.v3.{sbert,e5}.json` (variant=full):

| Embedder | links/mem (mean) | links/mem (max) | dominant link types |
|---|---|---|---|
| sbert-bge | {{sbert_links_mean}} | {{sbert_links_max}} | {{sbert_link_types}} |
| e5-small-v2 | {{e5_links_mean}} | {{e5_links_max}} | {{e5_link_types}} |

---

## Interpretation

*Pre-specified branches — select the applicable branch for the **primary Hi3**
and delete the other. Secondary verdicts (Hi3_recency, Hi3_arc) are reported
descriptively within whichever branch is selected.*

### Branch A — If Hi3 PASS (Δ > 0.05 AND p_adj < 0.05)

**Hi3 PASS.** The per-challenge amplification on `semantic_confound` is
confirmed at N=500 with Δ = {{hi3_delta}} pp (Cohen's d = {{hi3_d}}, Holm-adj
p = {{hi3_p_adj}}). The v2 finding (post-hoc Δ = +0.100) is **not** a
sample-size artefact; embedder choice modulates resonance interference at a
detectable, reliable magnitude.

**Mechanism interpretation:** Hi2 (e5 forms tighter intra-topic clusters,
spreading-activation over-fires) is consistent with the link_set_stats
table above ({{e5_links_mean}} vs {{sbert_links_mean}} mean links/memory).
This is exploratory; the formal Hi3 test does not depend on the link-set
comparison.

**Secondary verdicts:**
- Hi3_recency: {{recency_verdict}} (Δ = {{recency_delta}}, p_adj = {{recency_p_adj}})
- Hi3_arc: {{arc_verdict}} (Δ = {{arc_delta}}, p_adj = {{arc_p_adj}})

If both secondaries PASS: amplification is multi-channel (consistent with v2
descriptive finding). If only the primary PASSES, the embedder gap is
concentrated in `semantic_confound` and not a generic phenomenon — meaningful
narrowing of the claim. Either secondary outcome is admissible under the
pre-reg.

**`claim_validation_matrix.json` updates:**
- `resonance_amplification_e5` → `status: PASS`, `delta: {{hi3_delta}}`,
  `p_adj: {{hi3_p_adj}}`, `d: {{hi3_d}}`
- Add rows for `Hi3_recency`, `Hi3_arc` with respective verdicts.

**§7 Limitations action:** Update the paragraph "resonance mechanism deferred
(Hi3)" — remove the deferral language, replace with a scoped statement
acknowledging the confirmed embedder-dependent amplification, with explicit
mention of N=500 and Cohen's d.

---

### Branch B — If Hi3 FAIL (Δ ≤ 0.05 OR p_adj ≥ 0.05)

**Hi3 FAIL.** The per-challenge amplification on `semantic_confound` does
not reach the pre-specified threshold (Δ > 0.05, p_adj < 0.05) at N=500.
Observed Δ = {{hi3_delta}} (Cohen's d = {{hi3_d}}, Holm-adj p = {{hi3_p_adj}}).

**Pre-specified interpretation:** the v2 per-challenge finding (post-hoc Δ =
+0.100 on semantic_confound) is downgraded from "embedder-dependent
amplification of resonance interference" to "**v2 sample-size artefact**".
Per Add. I §Hi3 frozen design: "if Hi3 FAIL, the v2 finding is downgraded
[...] Claim matrix updated accordingly."

**What is NOT implied:** This FAIL does not invalidate:
- Hb FAIL itself (S3 closure: aggregate-level no_resonance > full on e5,
  p<0.001 — that result is on different data and used seed=0).
- Hi1 (descriptive per-challenge decomposition of v2 data) — the descriptive
  table in Add. I remains valid as a record of v2 numerics.
- Hi2 (mechanistic hypothesis) — formally untested in either v2 or v3;
  the link_set_stats comparison is descriptive only.

**Secondary verdicts:**
- Hi3_recency: {{recency_verdict}}
- Hi3_arc: {{arc_verdict}}

If any secondary PASSES while the primary FAILS: report descriptively, but
do **not** promote a secondary to a confirmatory claim — this would violate
the pre-registered family hierarchy (Hi3 is the primary; m=3 Holm assumes
the primary is the focal test).

**`claim_validation_matrix.json` updates:**
- `resonance_amplification_e5` → `status: FAIL`, `delta: {{hi3_delta}}`,
  `p_adj: {{hi3_p_adj}}`, `d: {{hi3_d}}`, with note "v2 finding downgraded
  to sample-size artefact per Hi3 FAIL".
- Add rows for `Hi3_recency`, `Hi3_arc` with respective verdicts.

**§7 Limitations action:** Replace the "resonance mechanism deferred (Hi3)"
paragraph with a paragraph that states: (a) Hi3 was executed at N=500, (b)
the per-challenge amplification did not reach the pre-specified threshold,
(c) the v2 finding is therefore exploratory only, (d) the Hi1/Hi2 framing
remains valid as descriptive but cannot be used to claim an embedder-level
phenomenon.

---

## Coherence with prior closures

- **S3 closure (Hb FAIL)**: Hb FAIL is independent of Hi3 verdict — Hb is
  about the aggregate `full` vs `no_resonance` comparison on v2 (per
  embedder), while Hi3 is the cross-embedder amplification gap on a v3
  per-challenge slice. Both can be true simultaneously.
- **Hi1 (descriptive, post-hoc)**: Hi1 is exploratory by design and stands
  regardless of Hi3 verdict. PASS confirms the descriptive trend; FAIL
  contextualises Hi1 as an underpowered v2 observation.
- **Hd2 PASS**: Hd2 demonstrates the architecture advantage on v2 (Δ=+0.125,
  p<0.001). Hi3 verdict does not affect Hd2 — they test different claims
  (system advantage vs embedder-dependent resonance behaviour).
- **Hf1 PASS**: dual-path mitigation. Independent claim, unaffected.

---

## Update checklist (to complete at execution time)

- [ ] Replace all `{{placeholder}}` values with verbatim numbers from `results.hi3.json` / `.md`
- [ ] Select and delete the inapplicable interpretive branch (A or B)
- [ ] Set `Status: PASS` or `Status: FAIL` in the header
- [ ] Set `Date executed: <actual date>` in the header
- [ ] Update `docs/research/claim_validation_matrix.json` rows for Hi3, Hi3_recency, Hi3_arc
- [ ] Update `paper/main.tex §7 Limitations` per the selected branch action
- [ ] Run `make reproduce-paper-check` and confirm zero diff
- [ ] Run `make check` and confirm suite passes
- [ ] Commit atomically: closure doc + results.v3.* + results.hi3.* + claim matrix + §7 edit
