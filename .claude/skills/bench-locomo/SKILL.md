---
name: bench-locomo
description: Run the LoCoMo external benchmark (Pre-reg Study S1) end-to-end. Use when executing or re-running the AFT vs naive_rag comparison on LoCoMo10. Enforces dry-run-first, env-var check, background execution, and post-run evidence update.
disable-model-invocation: true
allowed-tools: Bash(make bench-locomo*) Bash(make llm-config*) Read Edit
---

# LoCoMo Benchmark — Study S1

Pre-registration: `benchmarks/preregistration.md` §Study S1.
Canonical execution: `make bench-locomo` (exports `.env`, sets `PYTHONUNBUFFERED=1`).

**CRITICAL:** `benchmarks/preregistration.md` is frozen. Never edit it.

## Steps

### 1. Validate LLM config

```bash
make llm-config-strict
```

Fail fast if `EMOTIONAL_MEMORY_LLM_API_KEY` is missing or config is unsupported.

### 2. Smoke test (2 conversations, 5 QA, no judge — <$0.01, ~2 min)

```bash
make bench-locomo-dry
```

Confirm exit code 0 and non-empty `benchmarks/locomo/results.md` before
committing to the full run.

### 3. Full run (~3h, ~$5–8 with gpt-5-mini + reasoning_effort=minimal)

Start the full run as a background task. Stream stdout monitor. Flag and stop on:
- HTTP 400/401/429 errors → check API key / rate limits
- `prediction` fields consistently empty → retrieval issue in adapter
- Python exception tracebacks → fix before restarting

### 4. Verify outputs

After the run completes:
- `benchmarks/locomo/results.json` — non-empty, n_conversations=10, n_qa_total=1986
- `benchmarks/locomo/results.md` — Aggregate Scores table populated with F1, BLEU-1, Judge Acc
- Report H1 (AFT F1 > naive_rag F1?) and H2 (AFT judge_acc > naive_rag judge_acc?)
- Check Δ and whether bootstrap CI excludes 0

### 5. Update evidence matrix

Run `/evidence-update locomo` to update `docs/research/09_current_evidence.md`.

Framing: this run is **CONFIRMATORY** (pre-reg S1 was committed before any LoCoMo
data — verify with `git log --oneline benchmarks/preregistration.md | head -1`).

## gpt-5-mini operational notes

- `temperature=0` is rejected by gpt-5-mini; `call_llm` strips it automatically on the
  first HTTP 400 and retries — no manual intervention needed.
- Set `EMOTIONAL_MEMORY_LLM_REASONING_EFFORT=minimal` in `.env` to reduce per-call
  latency from ~4.7 s → ~1.4 s (validated in local testing).
- Both answer-generation and LLM-judge steps use the same configured model.

## Cost and time estimate

| Mode | Cost | Wall-clock |
|---|---|---|
| Dry run (2 conv, 5 QA, no judge) | <$0.01 | ~2 min |
| Full run (10 conv, 1986 QA, with judge) | ~$5–8 | ~3 h |
