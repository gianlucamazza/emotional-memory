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
- **Mood-congruent retrieval** — the "Search memories" panel retrieves messages
  filtered through the current emotional state.

## Architecture

```
User message → encode() → EmotionalTag snapshot
                            ├── CoreAffect (valence × arousal)
                            ├── AffectiveMomentum (velocity + acceleration)
                            ├── MoodField (Heidegger EMA)
                            ├── AppraisalVector (Scherer SEC)
                            └── ResonanceLink (spreading activation)
```

## Local run

```bash
pip install "emotional-memory[sentence-transformers]" gradio matplotlib
python demo/app.py
```
