---
name: prereg-guard
description: Verify pre-registration integrity — no uncommitted edits to benchmarks/preregistration.md, and declared studies match produced result files. Use before paper building, release gates, or when resuming scientific work after a break.
disable-model-invocation: true
allowed-tools: Bash(git diff*) Bash(git log*) Bash(git status*) Read
---

# Pre-Registration Integrity Guard

Checks that `benchmarks/preregistration.md` has not been modified since its
last commit, and that result files exist for studies that have been run.

**Any FAIL = stop before paper building or publishing.**

The pre-registration file (`benchmarks/preregistration.md`) is the scientific
contract. It must never be edited after the first commit that introduced it.
Modifying it post-hoc would invalidate the confirmatory status of any study
whose hypothesis it declares.

## Checks

### 1. Pre-reg file unmodified in working tree

```bash
git diff benchmarks/preregistration.md
git diff --cached benchmarks/preregistration.md
```

Expected: empty output. Any diff is a pre-reg violation.

### 2. Pre-reg file not in staged or unstaged changes

```bash
git status --short benchmarks/preregistration.md
```

Expected: no output (file not listed).

### 3. Result files vs declared studies

Read `benchmarks/preregistration.md` to find declared studies. For each:

| Study | Result file to check |
|---|---|
| S1 (LoCoMo) | `benchmarks/locomo/results.json`, `benchmarks/locomo/results.md` |
| S2 (realistic v2) | `benchmarks/realistic/results.v2.md` |
| S3 (ablation v2) | `benchmarks/ablation/results.v2.md` |

If a result file is missing → WARN (study not yet run, not a violation).
If a result file exists but is not committed → WARN (uncommitted results).

### 4. Date ordering (CONFIRMATORY validity)

```bash
git log --oneline --format="%ci %s" benchmarks/preregistration.md | head -1
git log --oneline --format="%ci %s" benchmarks/locomo/results.json | head -1
```

Pre-reg commit timestamp must be strictly before each result commit timestamp
for the study to qualify as CONFIRMATORY.

## Output format

```
PASS ✓  preregistration.md unmodified (working tree clean)
PASS ✓  preregistration.md unmodified (staged area clean)
PASS ✓  S1 results present: benchmarks/locomo/results.json
PASS ✓  S1 results committed
PASS ✓  S1 pre-reg predates results → CONFIRMATORY claim valid
WARN ⚠  S2 results missing: benchmarks/realistic/results.v2.md (study not yet run)
WARN ⚠  S3 results missing: benchmarks/ablation/results.v2.md (study not yet run)
```

PASS on all checks → safe to proceed with paper building or release.
Any FAIL → stop and resolve before continuing.
