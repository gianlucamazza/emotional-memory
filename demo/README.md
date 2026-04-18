---
title: Emotional Memory — AFT Demo
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.50.0"
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

- **Chat history** backed by `EmotionalMemoryChatHistory` — every message shapes
  the agent's affective state in real time.
- **PAD state plot** — valence (positive/negative), arousal (calm/excited), and
  dominant mood update with each exchange.
- **Mood-congruent retrieval** — type "recall X" to retrieve memories filtered
  through the current emotional state.
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
`EMOTIONAL_MEMORY_LLM_MODEL` (default: `gpt-4o-mini`) and
`EMOTIONAL_MEMORY_LLM_BASE_URL` (default: `https://api.openai.com/v1`).

## Local run

```bash
pip install "emotional-memory[langchain]" httpx gradio matplotlib
# Optional: set EMOTIONAL_MEMORY_LLM_API_KEY for LLM appraisal
python demo/app.py
```

## Links

- **PyPI**: [`emotional-memory 0.6.0`](https://pypi.org/project/emotional-memory/0.6.0/)
- **GitHub**: [gianlucamazza/emotional-memory](https://github.com/gianlucamazza/emotional-memory)
- **Zenodo DOI**: [10.5281/zenodo.19636355](https://doi.org/10.5281/zenodo.19636355)

## Citation

```bibtex
@software{mazza_emotional_memory_2026,
  author  = {Mazza, Gianluca},
  title   = {emotional-memory: Affective Field Theory for LLM Memory},
  version = {0.6.0},
  year    = {2026},
  doi     = {10.5281/zenodo.19640250},
  url     = {https://github.com/gianlucamazza/emotional-memory}
}
```
