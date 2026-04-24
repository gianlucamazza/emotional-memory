---
name: bench-study
description: Run one of the project's comparative benchmark studies (realistic, ablation, appraisal, comparative). Use when executing Pre-reg S2/S3 or running appraisal quality. Validates env, runs the make target, summarizes results, and prompts for evidence update.
disable-model-invocation: true
arguments: [study]
allowed-tools: Bash(make bench-*) Bash(make llm-config*) Read
---

# Benchmark Study Runner

Argument: `$study` — one of: `realistic`, `ablation`, `appraisal`, `comparative`.

Stop immediately if the argument is not in the list above and ask the user to pick.

## Study → Make Target Mapping

| Argument | Make target | Pre-reg | LLM required |
|---|---|---|---|
| `realistic` | `make bench-realistic` | S2 | No |
| `ablation` | `make bench-ablation` | S3 | No |
| `appraisal` | `make bench-appraisal` | M1.3 | Yes (`EMOTIONAL_MEMORY_LLM_API_KEY`) |
| `comparative` | `make bench-comparative` | — | No |

## Steps

### 1. Validate config (LLM-requiring studies only)

For `appraisal`:

```bash
make llm-config-strict
```

### 2. Run the study

```bash
make bench-$study
```

For `realistic` and `ablation` the Makefile already uses `--embedder sbert-bge`.

### 3. Interpret results

Read `benchmarks/$study/results.md` (or console output for appraisal).

Report:
- Aggregate metric (top1_acc / F1 / pass rate)
- Per-category / per-challenge breakdown
- Bootstrap 95% CI if available
- Any result that survived Holm correction
- Comparison to prior numbers in `docs/research/09_current_evidence.md`

### 4. Determine DISCOVERY vs CONFIRMATORY framing

```bash
git log --oneline benchmarks/preregistration.md | head -1
git log --oneline benchmarks/$study/results.json 2>/dev/null | head -1
```

- Pre-reg commit predates results commit **and** the study is declared in pre-reg → **CONFIRMATORY**
- Otherwise → **DISCOVERY** (note that confirmatory evidence is still pending)

### 5. Prompt for evidence update

Remind the user to run `/evidence-update $study` to update
`docs/research/09_current_evidence.md` with the new numbers and framing.
