---
name: evidence-update
description: Update docs/research/09_current_evidence.md after a benchmark study completes. Use after any bench-locomo, bench-realistic, bench-ablation, or bench-appraisal run. Pulls numbers from results.md, enforces DISCOVERY vs CONFIRMATORY framing, cross-references the pre-reg.
disable-model-invocation: true
arguments: [study]
allowed-tools: Read Edit
---

# Evidence Matrix Update

Argument: `$study` — one of: `locomo`, `realistic`, `ablation`, `appraisal`.

Updates `docs/research/09_current_evidence.md` with the latest results from
the completed study, maintaining the DISCOVERY vs CONFIRMATORY framing that is
central to the paper's scientific credibility.

**Do not remove existing limitations or caveats unless the new data specifically
resolves them.**

## Results File Locations

| Argument | Results file |
|---|---|
| `locomo` | `benchmarks/locomo/results.md` |
| `realistic` | `benchmarks/realistic/results.md` (or `results.v2.md` if v2 run) |
| `ablation` | `benchmarks/ablation/results.md` (or `results.v2.md` if v2 run) |
| `appraisal` | `benchmarks/appraisal_quality/results.md` |

## Steps

### 1. Read the results

Open the results file for `$study`. Extract:
- Aggregate metric (F1 / top1_acc / judge_acc / pass_rate)
- Δ vs baseline system
- Bootstrap 95% CI (if present)
- Per-category / per-challenge breakdown
- Holm-corrected p-values (report only those that survived)

### 2. Determine framing

```
git log --oneline --format="%ci" benchmarks/preregistration.md | head -1
git log --oneline --format="%ci" benchmarks/<study>/results.json | head -1
```

- Pre-reg timestamp **before** results timestamp → **CONFIRMATORY**
  Report the H-label from `benchmarks/preregistration.md` (H1/H2 for S1,
  H3–H6 for S2, Ha–Hb for S3).
- Pre-reg timestamp **after** or no results committed → **DISCOVERY**
  Must include a note: "confirmatory evidence still pending".

### 3. Update `docs/research/09_current_evidence.md`

Find the section corresponding to `$study`. Update:
- Metric values and CI (replace stale numbers)
- Framing label (DISCOVERY / CONFIRMATORY)
- Run date (today's date)
- Whether H-labels were confirmed or refuted

Keep existing cells for other studies untouched.

### 4. Sanity check

After editing, confirm the updated section contains:
- At least one numeric value with CI
- The framing label (DISCOVERY or CONFIRMATORY) explicitly stated
- If DISCOVERY: the pending-confirmatory note
- If CONFIRMATORY: the H-label referenced from the pre-reg
