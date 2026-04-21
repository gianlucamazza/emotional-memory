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
- **Mood-congruent retrieval** — type "recall X" to retrieve memories filtered
  through the current emotional state.
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

To enable LLM appraisal when duplicating this Space, add a **Secret** named
`EMOTIONAL_MEMORY_LLM_API_KEY` in the Space Settings.  Optionally set Variables
`EMOTIONAL_MEMORY_LLM_MODEL` (default: `gpt-5-mini`) and
`EMOTIONAL_MEMORY_LLM_BASE_URL` (default: `https://api.openai.com/v1`).
For OpenAI-compatible endpoints you can also set
`EMOTIONAL_MEMORY_LLM_OUTPUT_MODE` (default: `plain`) and
`EMOTIONAL_MEMORY_LLM_TIMEOUT_SECONDS` (default: `30`).

## Local run

```bash
pip install emotional-memory httpx "gradio>=6.13,<7" matplotlib
# Optional: set EMOTIONAL_MEMORY_LLM_API_KEY for LLM appraisal
python demo/app.py
```

By default the demo runs with `ssr_mode=False` for a more stable local startup
and shutdown path on Python 3.11. To opt into Gradio SSR explicitly:

```bash
EMOTIONAL_MEMORY_DEMO_SSR=1 python demo/app.py
```

The demo intentionally ignores platform-managed `GRADIO_SSR_MODE` values so the
Hugging Face Space keeps the stable non-SSR startup path unless we opt in.

## Links

- **PyPI**: [`emotional-memory 0.6.1`](https://pypi.org/project/emotional-memory/0.6.1/)
- **GitHub**: [gianlucamazza/emotional-memory](https://github.com/gianlucamazza/emotional-memory)
- **Zenodo DOI**: [10.5281/zenodo.19686077](https://doi.org/10.5281/zenodo.19686077)

## Citation

```bibtex
@software{mazza_emotional_memory_2026,
  author  = {Mazza, Gianluca},
  title   = {emotional-memory: Affective Field Theory for LLM Memory},
  version = {0.6.1},
  year    = {2026},
  doi     = {10.5281/zenodo.19686078},
  url     = {https://github.com/gianlucamazza/emotional-memory}
}
```
