# LoCoMo Benchmark

Long-context conversational memory benchmark (Maharana et al. 2024, LoCoMo10).
This is **Study S1** of the pre-registered evidence programme for AFT.

Pre-registration: [`benchmarks/preregistration.md`](../preregistration.md) §Study S1.

## Execution

```bash
# Smoke test — 2 conversations, 5 QA each, no LLM judge (~2 min, <$0.01)
make bench-locomo-dry

# Full run — 10 conversations, 1 986 QA pairs, with LLM judge (~3 h, ~$5–8)
make bench-locomo
```

`make bench-locomo` exports `.env` and sets `PYTHONUNBUFFERED=1` for real-time
progress output. Prefer it over calling `python -m benchmarks.locomo.runner`
directly.

## Required environment variables

| Variable | Purpose |
|---|---|
| `EMOTIONAL_MEMORY_LLM_API_KEY` | API key (required for answer generation and LLM judge) |
| `EMOTIONAL_MEMORY_LLM_BASE_URL` | OpenAI-compatible endpoint (default: `https://api.openai.com/v1`) |
| `EMOTIONAL_MEMORY_LLM_MODEL` | Model (pre-reg pins `gpt-5-mini`; default in code: `gpt-4o-mini`) |
| `EMOTIONAL_MEMORY_LLM_REASONING_EFFORT` | Reasoning budget for o-series / gpt-5 models — `minimal` / `low` / `medium` / `high`; omit or leave empty to skip the param |

Set these in your `.env` file (loaded automatically by `make`).

## gpt-5-mini operational notes

- **Custom temperature not accepted.** gpt-5-mini rejects `temperature != 1`.
  `call_llm` strips the `temperature` key automatically on the first HTTP 400
  and retries, so no manual intervention is needed.
- **Reasoning overhead.** With the default reasoning effort, gpt-5-mini uses
  ~320 reasoning tokens per call (~4.7 s / call). Setting
  `EMOTIONAL_MEMORY_LLM_REASONING_EFFORT=minimal` drops this to 0 reasoning
  tokens (~1.4 s / call) at the cost of slightly less deliberate answers —
  acceptable for QA retrieval benchmarking.
  *Empirical, local measurement; not a benchmark claim.*

## Outputs

| File | Content |
|---|---|
| `benchmarks/locomo/results.json` | Slim aggregate + per-category scores (no raw predictions) |
| `benchmarks/locomo/results.md` | Human-readable Markdown table |

## Systems / adapters

| Adapter | Description |
|---|---|
| `AFTLoCoMoAdapter` | Full AFT pipeline (5-layer emotional model, 6-signal retrieval) |
| `NaiveRAGLoCoMoAdapter` | Pure cosine retrieval, no affective state — control baseline |

Both adapters use `bge-small-en-v1.5` (SentenceTransformer, 384 dims) as the
embedder, per the pre-registration §Systems. The embedder is shared; only the
retrieval and state logic differ.

## Dataset

LoCoMo10 is downloaded on first use via the `benchmarks/locomo/dataset.py`
loader (Hugging Face datasets or local cache). It contains 10 long multi-session
conversations with ~1 986 QA pairs across four categories:
`single_hop`, `multi_hop`, `open_domain`, `temporal`, and adversarial (cat-5).
