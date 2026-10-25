---
title: Emotional Memory — AFT Demo (ZeroGPU)
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "6.13.0"
python_version: "3.12"
app_file: app.py
pinned: false
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
  - zerogpu
models:
  - sentence-transformers/all-MiniLM-L6-v2
datasets:
  - gianlucamazza/emotional-memory-benchmarks
suggested_hardware: zero-a10g
short_description: Source for a private ZeroGPU AFT memory demo
---

# `demo/zerogpu/`

This folder is the source tree for a private ZeroGPU Space (not yet public).
That Space is to be synced **verbatim** from these files at a merge SHA of this
repository: copy `app.py`, `requirements.txt`, and this `README.md` as they
appear in that commit, with no extra markers or generated files.

The app matches the public demo in `demo/app.py`. The only GPU work is the
MiniLM embedding calls (`all-MiniLM-L6-v2`) under `@spaces.GPU`. Appraisal
uses an OpenAI-compatible LLM only if `EMOTIONAL_MEMORY_LLM_API_KEY` is set;
otherwise it uses the English keyword fallback. There is no in-process
open-model appraisal.

## Local run (CPU)

Off ZeroGPU, `@spaces.GPU` is a no-op. If the `spaces` package is not
installed, `app.py` uses the same no-op shim. The Space overlay still lists
`spaces` in `requirements.txt`.

```bash
make install-demo
uv run python demo/zerogpu/app.py
```

Or, from a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r demo/zerogpu/requirements.txt
python demo/zerogpu/app.py
```

Configuration is read from the process environment only. This app does not
call `load_dotenv()`. For local `.env` convenience, `source` the file in your
shell before launching, or export the variables below.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `EMOTIONAL_MEMORY_LLM_API_KEY` | unset | LLM appraisal. Read from the environment or a Space secret. Never hardcoded or logged. |
| `EMOTIONAL_MEMORY_LLM_MODEL` | `gpt-5-mini` | OpenAI-compatible model id |
| `EMOTIONAL_MEMORY_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `EMOTIONAL_MEMORY_LLM_OUTPUT_MODE` | `plain` | `plain` or `json_object` |
| `EMOTIONAL_MEMORY_LLM_TIMEOUT_SECONDS` | `30` | HTTP timeout for appraisal calls |
| `EMOTIONAL_MEMORY_ZEROGPU_DURATION` | `30` | `@spaces.GPU` duration reservation, in seconds |
| `EMOTIONAL_MEMORY_DEMO_SSR` | unset (off) | Set to `1` / `true` / `yes` / `on` to enable Gradio SSR |

Without `EMOTIONAL_MEMORY_LLM_API_KEY`, the demo uses `KeywordAppraisalEngine`
(English only).
