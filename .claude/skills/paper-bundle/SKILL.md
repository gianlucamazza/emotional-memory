---
name: paper-bundle
description: Build the paper + arXiv submission bundle. Use before arXiv upload or when preparing a workshop submission. Runs latexmk, produces paper/arxiv-submission.tar.gz, then validates all result tables against committed benchmarks/*/results.md.
disable-model-invocation: true
allowed-tools: Bash(make paper*) Bash(tar *) Bash(git status*) Bash(git log*) Read
---

# Paper + arXiv Bundle Builder

Builds `paper/arxiv-submission.tar.gz` from `paper/main.tex` and validates
that all numeric tables in the paper match committed results files.

Run `/prereg-guard` first. If it reports any FAIL, resolve before building.

## Steps

### 1. Build the paper

```bash
make paper-arxiv
```

Expected: exit code 0 and `paper/arxiv-submission.tar.gz` created.

On latexmk failure: read `paper/main.log` for the first error line.
Common causes: missing `.bbl` (run `pdflatex + bibtex + pdflatex × 2`
manually), undefined citation, or broken figure path.

### 2. Verify archive contents

```bash
tar -tzf paper/arxiv-submission.tar.gz
```

Must contain: `main.tex`, `main.bbl`, `refs.bib`, `figures/`, `tables/`.
Missing `main.bbl` → bibliography not compiled.

### 3. Cross-check tables vs results files

Read each file in `paper/tables/` and compare numeric values against:

| Table source | Results file |
|---|---|
| LoCoMo F1 / judge_acc | `benchmarks/locomo/results.md` |
| Realistic top1_acc + CI | `benchmarks/realistic/results.md` |
| Ablation Δ, d, p_adj | `benchmarks/ablation/results.md` |
| Appraisal pass rate | `benchmarks/appraisal_quality/results.md` |

Flag any mismatch between the table value and the current results file.
Stale numbers must be updated before submission.

### 4. Limitations section check

Read `paper/main.tex` §Limitations. Confirm it mentions:
- DISCOVERY vs CONFIRMATORY distinction for any unreplicated finding
- N count and 95% CI for all reported effect sizes
- The null ablation result (if applicable)
- Absence of distributed memory-store validation

### 5. Results committed check

```bash
git status --short benchmarks/
```

All `benchmarks/*/results.{json,md}` files referenced in the paper must be
committed. Uncommitted results → commit them before submitting to arXiv.

## Pre-submission report

Output a checklist:
```
✓ / ✗  Build: exit code 0, arxiv-submission.tar.gz created
✓ / ✗  Archive: main.tex, main.bbl, refs.bib, figures/, tables/ all present
✓ / ✗  Tables consistent with results files (list any mismatches)
✓ / ✗  §Limitations mentions DISCOVERY/CONFIRMATORY and null ablation
✓ / ✗  All referenced results files committed
```
