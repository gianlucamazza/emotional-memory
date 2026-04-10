"""LLM appraisal — emotional tagging powered by an OpenAI-compatible API.

Demonstrates how to wire up LLMAppraisalEngine with any OpenAI-compatible
provider, including cache behaviour, error fallback, and the full encode
pipeline. Falls back to a mock LLM when no API key is available so the
script is always runnable.

Requires (for real LLM mode):
    pip install openai
    export EMOTIONAL_MEMORY_LLM_API_KEY=sk-...

Optional overrides:
    export EMOTIONAL_MEMORY_LLM_BASE_URL=https://api.openai.com/v1
    export EMOTIONAL_MEMORY_LLM_MODEL=gpt-4o-mini

Run with:
    python examples/llm_appraisal.py
"""

import json
import os
import time

from emotional_memory import (
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    LLMAppraisalConfig,
    LLMAppraisalEngine,
    RetrievalConfig,
)

# ---------------------------------------------------------------------------
# Minimal embedder — no ML dependencies required
# ---------------------------------------------------------------------------


class HashEmbedder:
    """Deterministic 8-dim embedder based on string hashing."""

    DIM = 8

    def embed(self, text: str) -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        vec = [((h >> i) & 0xFF) / 255.0 for i in range(self.DIM)]
        total = sum(vec) or 1.0
        return [v / total for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# LLMCallable — wrap any OpenAI-compatible SDK
# ---------------------------------------------------------------------------


def make_openai_llm(api_key: str, base_url: str, model: str):  # type: ignore[return]
    """Return an LLMCallable wrapping the openai SDK."""
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError:
        print("[!] 'openai' package not installed. Run: pip install openai")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)

    def llm_callable(prompt: str, json_schema: dict) -> str:  # type: ignore[type-arg]
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or "{}"

    return llm_callable


def make_mock_llm():
    """Return a deterministic mock LLM for offline use."""

    # Hardcoded responses simulating Scherer CPM output
    _responses = {
        "record": {
            "novelty": 0.5,
            "goal_relevance": 0.9,
            "coping_potential": 0.8,
            "norm_congruence": 0.7,
            "self_relevance": 0.8,
        },
        "outage": {
            "novelty": 0.2,
            "goal_relevance": -0.9,
            "coping_potential": 0.2,
            "norm_congruence": -0.6,
            "self_relevance": 0.9,
        },
        "default": {
            "novelty": 0.1,
            "goal_relevance": 0.1,
            "coping_potential": 0.5,
            "norm_congruence": 0.0,
            "self_relevance": 0.2,
        },
    }

    def mock_llm(prompt: str, json_schema: dict) -> str:  # type: ignore[type-arg]
        key = (
            "record"
            if "record" in prompt.lower()
            else "outage"
            if "outage" in prompt.lower()
            else "default"
        )
        return json.dumps(_responses[key])

    return mock_llm


# ---------------------------------------------------------------------------
# Select real or mock LLM
# ---------------------------------------------------------------------------

api_key = os.environ.get("EMOTIONAL_MEMORY_LLM_API_KEY", "")
base_url = os.environ.get("EMOTIONAL_MEMORY_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
model = os.environ.get("EMOTIONAL_MEMORY_LLM_MODEL", "gpt-4o-mini")

if api_key:
    llm = make_openai_llm(api_key, base_url, model)
    if llm is None:
        llm = make_mock_llm()
        mode = "mock (openai not installed)"
    else:
        mode = f"real ({model})"
else:
    llm = make_mock_llm()
    mode = "mock (no EMOTIONAL_MEMORY_LLM_API_KEY)"

print(f"LLM mode: {mode}\n")

# ---------------------------------------------------------------------------
# Build LLMAppraisalEngine
# ---------------------------------------------------------------------------

appraisal_engine = LLMAppraisalEngine(
    llm=llm,
    config=LLMAppraisalConfig(
        cache_size=128,  # LRU cache — identical texts return instantly
        fallback_on_error=True,  # return neutral vector instead of raising
    ),
)

# ---------------------------------------------------------------------------
# Appraise events directly — inspect the AppraisalVector output
# ---------------------------------------------------------------------------

print("=== Direct appraisal (no engine) ===\n")

events = [
    "We hit a record quarter — revenue up 60% year over year.",
    "Three-hour database outage took down the entire platform.",
    "Finished reading the quarterly engineering newsletter.",
]

for text in events:
    av = appraisal_engine.appraise(text)
    ca = av.to_core_affect()
    sign = "[+]" if ca.valence > 0.1 else "[-]" if ca.valence < -0.1 else "[~]"
    print(f"{sign} {text[:60]}")
    print(
        f"    novelty={av.novelty:+.2f}  goal_rel={av.goal_relevance:+.2f}"
        f"  coping={av.coping_potential:.2f}  norm={av.norm_congruence:+.2f}"
        f"  self_rel={av.self_relevance:.2f}"
    )
    print(f"    → affect: valence={ca.valence:+.3f}  arousal={ca.arousal:.3f}\n")

# ---------------------------------------------------------------------------
# Cache demonstration — same text returns instantly from LRU cache
# ---------------------------------------------------------------------------

print("=== Cache hit ===\n")

same_text = events[0]
t0 = time.perf_counter()
av1 = appraisal_engine.appraise(same_text)
t1 = time.perf_counter()
av2 = appraisal_engine.appraise(same_text)  # cache hit
t2 = time.perf_counter()

print(f"First call:  {(t1 - t0) * 1000:.2f} ms")
print(f"Second call: {(t2 - t1) * 1000:.2f} ms  (cache hit)")
print(f"Vectors identical: {av1 == av2}\n")

appraisal_engine.clear_cache()
print("Cache cleared.\n")

# ---------------------------------------------------------------------------
# Wired into the full EmotionalMemory pipeline
# ---------------------------------------------------------------------------

print("=== Full encode → retrieve pipeline with LLM appraisal ===\n")

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    appraisal_engine=appraisal_engine,
    config=EmotionalMemoryConfig(
        retrieval=RetrievalConfig(base_weights=[0.20, 0.30, 0.25, 0.10, 0.10, 0.05]),
    ),
)

more_events = [
    "We hit a record quarter — revenue up 60% year over year.",
    "Three-hour database outage took down the entire platform.",
    "Team retrospective went smoothly, morale is high.",
    "Deployed a hotfix at 1 AM to recover from the outage.",
]

for text in more_events:
    mem = em.encode(text)
    av = mem.tag.appraisal
    ca = mem.tag.core_affect
    sign = "[+]" if ca.valence > 0.1 else "[-]" if ca.valence < -0.1 else "[~]"
    print(f"{sign} {mem.content[:60]}")
    if av is not None:
        strength = mem.tag.consolidation_strength
        print(f"    valence={ca.valence:+.2f}  arousal={ca.arousal:.2f}  strength={strength:.2f}")
    print()

# Retrieval is now emotionally anchored by LLM-computed appraisal
results = em.retrieve("platform reliability incident", top_k=3)
print("Top-3 for 'platform reliability incident':")
for i, mem in enumerate(results, 1):
    ca = mem.tag.core_affect
    sign = "[+]" if ca.valence > 0.1 else "[-]"
    print(f"  {i}. {sign} {mem.content[:60]}")
    print(f"       valence={ca.valence:+.2f}  retrieval_count={mem.tag.retrieval_count}")

print("\nDone.")
