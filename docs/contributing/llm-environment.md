# LLM Environment Variables

This page is the **single source of truth** for the `EMOTIONAL_MEMORY_LLM_*`
environment variables that configure real-LLM integration tests
(`make test-llm`) and the appraisal-quality benchmarks (`make bench-appraisal`).
Copies elsewhere (README, `CLAUDE.md`, `CONTRIBUTING.md`) are mirrors — when you
change a default here, update them too.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `EMOTIONAL_MEMORY_LLM_API_KEY` | Yes | — | API key for the LLM provider |
| `EMOTIONAL_MEMORY_LLM_BASE_URL` | No | `https://api.openai.com/v1` | OpenAI-compatible endpoint (Ollama, vLLM, LiteLLM, …) |
| `EMOTIONAL_MEMORY_LLM_MODEL` | No | `gpt-5-mini` | Model name |
| `EMOTIONAL_MEMORY_LLM_OUTPUT_MODE` | No | `plain` | Response mode: `plain` or `json_object` |
| `EMOTIONAL_MEMORY_LLM_TIMEOUT_SECONDS` | No | `30` | HTTP timeout in seconds |
| `EMOTIONAL_MEMORY_LLM_REPEATS` | No | `3` | Repeats per phrase in quality benchmarks |
| `EMOTIONAL_MEMORY_LLM_REASONING_EFFORT` | No | `""` | `reasoning_effort` for o-series / gpt-5 models (`minimal` / `low` / `medium` / `high`); omitted when empty |

## Usage

`make` targets export `.env` automatically. To have `.env` auto-loaded when invoking a
benchmark module directly (e.g. `python -m benchmarks.appraisal_diagnostics.runner`), run
`make install-dotenv` (installs `python-dotenv`). Real-LLM tests need the HTTP client —
run `make install-llm-test` (installs `httpx`). Verify the resolved config any time with
`make llm-config` (prints values, no secrets).

```bash
EMOTIONAL_MEMORY_LLM_API_KEY=... make test-llm
EMOTIONAL_MEMORY_LLM_API_KEY=... make bench-appraisal
```

!!! note "Release secrets are separate"
    `PYPI_TOKEN`, `ZENODO_TOKEN`, and `ZENODO_BASE` are **release** secrets, not LLM
    configuration. They are documented in
    [CONTRIBUTING.md](https://github.com/gianlucamazza/emotional-memory/blob/main/CONTRIBUTING.md).

## See also

- [Benchmarks](../benchmarks.md) — the appraisal-quality suite that consumes these variables
- [SSOT Policy](ssot-policy.md) — why this page is canonical
