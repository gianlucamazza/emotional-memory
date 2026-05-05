---
title: Emotional Memory — AFT Demo
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "6.13.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
tags:
  - affective-computing
  - memory
  - llm
  - gradio
  - research
suggested_hardware: cpu-basic
short_description: Live demo of Affective Field Theory memory for LLM agents
---

# Emotional Memory — AFT Demo

Interactive demo of the [`emotional-memory`](https://github.com/gianlucamazza/emotional-memory)
library, which implements **Affective Field Theory** — a 5-layer emotional model for LLM memory.

## What it shows

- **Emotion-aware chat loop** — user messages become episodic memories, while
  assistant replies update affective state without polluting retrieval.
- **PAD state plot** — valence (positive/negative), arousal (calm/excited), and
  dominant mood update with each exchange.
- **Semantic + affect-aware retrieval** — type "recall X" to retrieve memories
  ranked by semantic similarity and the current emotional state.
- **Best-practice memory policy** — assistant replies and recall commands stay
  out of the retrievable corpus to avoid self-retrieval artifacts.
- **Example prompts** — click any pre-filled example to see the pipeline in action.

## Architecture

```
User message → encode() → EmotionalTag snapshot
                            ├── CoreAffect (valence × arousal)
                            ├── AffectiveMomentum (velocity + acceleration)
                            ├── MoodField (Heidegger EMA)
                            ├── AppraisalVector (Scherer SEC)
                            └── ResonanceLink (spreading activation)
```

## Appraisal modes

| Mode | When active | Languages |
|---|---|---|
| 🧠 LLM appraisal | `EMOTIONAL_MEMORY_LLM_API_KEY` secret is set | Multilingual |
| 📝 Keyword fallback | No API key | English only |

The Space is intended to run with real semantic embeddings via
`sentence-transformers`. If that dependency is unavailable, the demo falls back
to an internal hash embedder and marks retrieval as approximate in the UI.

To enable LLM appraisal when duplicating this Space, add a **Secret** named
`EMOTIONAL_MEMORY_LLM_API_KEY` in the Space Settings.  Optionally set Variables
`EMOTIONAL_MEMORY_LLM_MODEL` (default: `gpt-5-mini`) and
`EMOTIONAL_MEMORY_LLM_BASE_URL` (default: `https://api.openai.com/v1`).
For OpenAI-compatible endpoints you can also set
`EMOTIONAL_MEMORY_LLM_OUTPUT_MODE` (default: `plain`) and
`EMOTIONAL_MEMORY_LLM_TIMEOUT_SECONDS` (default: `30`).

## Local run

```bash
make install-demo
make demo-run
```

`demo/app.py` reads configuration from the process environment only. It does
not call `load_dotenv()` itself. For local `.env` convenience, `make demo-run`
is the recommended path because the project `Makefile` already includes and
exports `.env` when present.

`demo/requirements.txt` is reserved for the Hugging Face Space runtime overlay.
For contributor machines, `make install-demo` is the canonical local setup path.

For a manual shell launch, export the file first:

```bash
set -a
source .env
set +a
uv run python demo/app.py
```

Before validating the LLM-backed path locally, fail fast on config issues:

```bash
make llm-config-strict
make demo-check
make test-llm
```

By default the demo runs with `ssr_mode=False` for a more stable local startup
and shutdown path on Python 3.11. To opt into Gradio SSR explicitly:

```bash
EMOTIONAL_MEMORY_DEMO_SSR=1 make demo-run
```

The demo intentionally ignores platform-managed `GRADIO_SSR_MODE` values so the
Hugging Face Space keeps the stable non-SSR startup path unless we opt in.

For the deployed Hugging Face Space, use Space Secrets/Variables rather than a
local `.env` file.

The bootstrap also applies a narrow Python 3.11 event-loop cleanup workaround
for the Gradio runtime: it suppresses only the known
`ValueError: Invalid file descriptor: -1` traceback seen at shutdown and leaves
all other exceptions untouched.

## Links

- **PyPI**: [`emotional-memory 0.9.0`](https://pypi.org/project/emotional-memory/0.9.0/)
- **GitHub**: [gianlucamazza/emotional-memory](https://github.com/gianlucamazza/emotional-memory)
- **Zenodo Concept DOI**: [10.5281/zenodo.19972258](https://doi.org/10.5281/zenodo.19972258)

## Citation

```bibtex
@software{mazza_emotional_memory_2026,
  author  = {Mazza, Gianluca},
  title   = {emotional-memory: Affective Field Theory for LLM Memory},
  version = {0.9.0},
  year    = {2026},
  doi     = {10.5281/zenodo.20040352},
  url     = {https://github.com/gianlucamazza/emotional-memory}
}
```
