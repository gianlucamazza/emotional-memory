---
title: Emotional Memory — AFT Demo
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "6.13.0"
python_version: "3.11"
app_file: app.py
pinned: true
license: mit
tags:
  - affective-computing
  - memory
  - llm
  - gradio
  - research
  - agent-memory
  - emotion
  - retrieval
models:
  - sentence-transformers/all-MiniLM-L6-v2
suggested_hardware: cpu-basic
short_description: Live demo of Affective Field Theory memory for LLM agents
---

# Emotional Memory — AFT Demo

Interactive demo of the [`emotional-memory`](https://github.com/gianlucamazza/emotional-memory)
library. The library implements **Affective Field Theory** (AFT), a 5-layer emotional model for
LLM memory. It gives LLM agents affect-aware episodic memory. Each episode is stored with valence,
arousal, mood and appraisal. Recall is ranked by meaning plus that affective context.

## Who it is for

Researchers and engineers who need mood-congruent recall in multi-session agents (journaling,
long-horizon conversational memory). It is not a drop-in replacement for general-purpose RAG:
read **Scope and limits** below before using it.

## What it shows

- **Emotion-aware chat loop**: user messages become episodic memories. Assistant
  replies update the affective state but stay out of retrieval.
- **PAD state plot**: valence (positive/negative), arousal (calm/excited), and
  dominant mood update with each exchange.
- **Semantic + affect-aware retrieval**: type "recall X" to retrieve memories
  ranked by semantic similarity and the current emotional state.
- **Retrieval mode toggle**: the default is mood-state recall (the mechanism
  demo). Query-gated recall appraises the recall query and uses cosine-only
  ranking when that query is affectively neutral. A bare `recall` command
  always uses mood-state recall. Normal recall updates memory (retrieval
  count, reconsolidation, link strengthening) as part of the mechanism; each
  mode keeps its own memory so the comparison stays fair.
- **Memory policy**: assistant replies and recall commands stay out of the
  retrievable corpus, so the agent does not retrieve its own output.
- **Example prompts**: click any pre-filled example to see the pipeline run.

## Scope and limits

`emotional_memory` is a focused primitive for **affect-discriminative recall**, not a general-purpose
memory system. Its measured advantage holds only when relevance is itself affect-conditioned _and_
affect is supplied at encode time. Pre-registered studies show it does **not** help, and can hurt,
outside that regime:

- **Factual / open-domain QA**: on LoCoMo, AFT underperforms a naive RAG baseline (F1 0.168 vs 0.271).
- **Emotional-support recall (counter-congruent)**: on MADial-Bench (NAACL 2025), cosine is
  significantly better (nDCG@5 0.304 vs 0.221).
- **Content-determined QA over emotional history**: on ES-MemEval (WWW 2026), cosine is again
  significantly better (nDCG@4 0.284 vs 0.133).
- Also no advantage on short-turn naturalistic dialogue (DailyDialog) or with end-to-end automatic
  appraisal instead of oracle affect.

This demo shows how the mechanism works. It is not evidence that AFT is better in general. Full record:
[When NOT to use](https://github.com/gianlucamazza/emotional-memory#when-not-to-use) ·
[current evidence](https://github.com/gianlucamazza/emotional-memory/blob/main/docs/research/09_current_evidence.md).

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
`EMOTIONAL_MEMORY_LLM_API_KEY` in the Space Settings. Optionally set Variables
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
Hugging Face Space keeps the stable non-SSR startup path unless SSR is enabled explicitly.

For the deployed Hugging Face Space, use Space Secrets/Variables rather than a
local `.env` file.

The bootstrap also applies a narrow Python 3.11 event-loop cleanup workaround
for the Gradio runtime: it suppresses only the known
`ValueError: Invalid file descriptor: -1` traceback seen at shutdown and leaves
all other exceptions untouched.

## Links

- **PyPI**: [`emotional-memory 0.18.1`](https://pypi.org/project/emotional-memory/0.18.1/)
- **GitHub**: [gianlucamazza/emotional-memory](https://github.com/gianlucamazza/emotional-memory)
- **Zenodo Concept DOI** (all versions): [10.5281/zenodo.19972258](https://doi.org/10.5281/zenodo.19972258)
- **Zenodo Version DOI** (0.18.1): [10.5281/zenodo.22724258](https://doi.org/10.5281/zenodo.22724258)
- **Author**: [Gianluca Mazza](https://gianlucamazza.it). I take LLM systems into production: state, recovery, eval, cost.

## License

MIT, for both the library and this demo. See the
[LICENSE](https://github.com/gianlucamazza/emotional-memory/blob/main/LICENSE) file in the GitHub repo.

## Citation

```bibtex
@software{mazza_emotional_memory_2026,
  author  = {Mazza, Gianluca},
  title   = {emotional-memory: Affective Field Theory for LLM Memory},
  version = {0.18.1},
  year    = {2026},
  doi     = {10.5281/zenodo.22724258},
  url     = {https://github.com/gianlucamazza/emotional-memory}
}
```

To cite all versions rather than 0.18.1, use the concept DOI
[10.5281/zenodo.19972258](https://doi.org/10.5281/zenodo.19972258).
