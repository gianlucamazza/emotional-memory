"""httpx LLM integration — SDK-agnostic appraisal, .env config, full API tour.

Demonstrates:
  - .env config loading via python-dotenv (optional)
  - LLMCallable implemented with raw httpx (no openai package required)
  - KeywordAppraisalEngine as rule-based fallback
  - AffectiveMomentum inspection (velocity + acceleration in affect-space)
  - Manual EmotionalTag construction with make_emotional_tag + consolidation_strength
  - ResonanceLink traversal (link_type, strength, target_id)
  - SyncToAsyncAppraisalEngine bridge for the async facade
  - __version__ banner

Requires (for .env loading):
    pip install emotional-memory[dotenv]

Requires (for real LLM calls):
    pip install httpx

Create a .env file (or copy .env.example) to enable real LLM mode:
    EMOTIONAL_MEMORY_LLM_API_KEY=sk-...
    EMOTIONAL_MEMORY_LLM_BASE_URL=https://api.openai.com/v1  # optional
    EMOTIONAL_MEMORY_LLM_MODEL=gpt-4o-mini                   # optional

Run with:
    python examples/httpx_llm_integration.py
"""

import asyncio
import os
from typing import Any

from emotional_memory import (
    AffectiveMomentum,
    AsyncEmotionalMemory,
    CoreAffect,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    KeywordAppraisalEngine,
    LLMAppraisalConfig,
    LLMAppraisalEngine,
    LLMCallable,
    ResonanceConfig,
    ResonanceLink,
    RetrievalConfig,
    SyncToAsyncAppraisalEngine,
    SyncToAsyncEmbedder,
    SyncToAsyncStore,
    __version__,
    consolidation_strength,
    make_emotional_tag,
)

# ---------------------------------------------------------------------------
# 1. Load .env (python-dotenv optional)
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv

    load_dotenv()
    _dotenv_loaded = True
except ImportError:
    _dotenv_loaded = False

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o-mini"

api_key = os.environ.get("EMOTIONAL_MEMORY_LLM_API_KEY", "")
base_url = os.environ.get("EMOTIONAL_MEMORY_LLM_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
model = os.environ.get("EMOTIONAL_MEMORY_LLM_MODEL", _DEFAULT_MODEL)

print(f"emotional_memory v{__version__}")
print(f"  dotenv loaded:  {_dotenv_loaded}")
print(f"  API key set:    {bool(api_key)}")
print(f"  base_url:       {base_url}")
print(f"  model:          {model}")
print()

# ---------------------------------------------------------------------------
# 2. Minimal embedder
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
# 3. httpx-based LLMCallable (SDK-agnostic)
# ---------------------------------------------------------------------------


def make_httpx_llm(key: str, endpoint: str, llm_model: str) -> LLMCallable | None:
    """Build an LLMCallable using raw httpx — no openai package required."""
    try:
        import httpx
    except ImportError:
        return None

    def _call(prompt: str, json_schema: dict[str, Any]) -> str:
        payload: dict[str, Any] = {
            "model": llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        response = httpx.post(
            f"{endpoint}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    return _call


# ---------------------------------------------------------------------------
# 4. Appraisal engine selection
# ---------------------------------------------------------------------------

_llm_callable: LLMCallable | None = make_httpx_llm(api_key, base_url, model) if api_key else None

if _llm_callable is not None:
    appraisal_engine = LLMAppraisalEngine(
        llm=_llm_callable,
        config=LLMAppraisalConfig(cache_size=64, fallback_on_error=True),
    )
    mode = f"LLMAppraisalEngine (httpx → {model})"
else:
    appraisal_engine = KeywordAppraisalEngine()
    mode = "KeywordAppraisalEngine (rule-based fallback)"

print(f"Appraisal mode: {mode}\n")

# ---------------------------------------------------------------------------
# 5. Build engine + encode 4 events
# ---------------------------------------------------------------------------

config = EmotionalMemoryConfig(
    resonance=ResonanceConfig(threshold=0.05, max_links=5),
    retrieval=RetrievalConfig(base_weights=[0.25, 0.25, 0.20, 0.10, 0.10, 0.10]),
)

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    appraisal_engine=appraisal_engine,
    config=config,
)

events = [
    (
        CoreAffect(valence=0.8, arousal=0.7),
        "Shipped a feature the team had been waiting months for.",
    ),
    (CoreAffect(valence=-0.7, arousal=0.8), "Production outage — all hands on deck at 2 AM."),
    (CoreAffect(valence=0.5, arousal=0.4), "Resolved the incident; wrote the post-mortem report."),
    (CoreAffect(valence=0.3, arousal=0.2), "Quiet Friday afternoon reading engineering blogs."),
]

print("=== Encoding 4 events ===\n")
for affect, text in events:
    em.set_affect(affect)
    em.encode(text)
    sign = "[+]" if affect.valence > 0 else "[-]"
    print(f"  {sign} {text[:60]}")

# ---------------------------------------------------------------------------
# 6. AffectiveMomentum inspection
# ---------------------------------------------------------------------------

print("\n=== AffectiveMomentum — velocity in affect-space ===\n")

state = em.get_state()
mom: AffectiveMomentum = state.momentum  # explicit AffectiveMomentum annotation

print(f"  d_valence  (velocity):      {mom.d_valence:+.4f}")
print(f"  d_arousal  (velocity):      {mom.d_arousal:+.4f}")
print(f"  dd_valence (acceleration):  {mom.dd_valence:+.4f}")
print(f"  dd_arousal (acceleration):  {mom.dd_arousal:+.4f}")
print(f"  magnitude  (speed):         {mom.magnitude():.4f}")

# ---------------------------------------------------------------------------
# 7. Manual tag: consolidation_strength + make_emotional_tag
# ---------------------------------------------------------------------------

print("\n=== Manual EmotionalTag construction ===\n")

# consolidation_strength: standalone function (not just the property)
high_arousal_strength = consolidation_strength(arousal=0.8, stimmung_arousal=0.5)
low_arousal_strength = consolidation_strength(arousal=0.1, stimmung_arousal=0.2)
print(f"  consolidation_strength(arousal=0.8, stimmung=0.5): {high_arousal_strength:.3f}")
print(f"  consolidation_strength(arousal=0.1, stimmung=0.2): {low_arousal_strength:.3f}")
print("  (Yerkes-Dodson inverted-U — high arousal → stronger consolidation)")

# make_emotional_tag: convenience constructor
manual_tag = make_emotional_tag(
    core_affect=CoreAffect(valence=0.6, arousal=0.8),
    momentum=mom,
    stimmung=state.stimmung,
    consolidation_strength=high_arousal_strength,
)
print(f"\n  make_emotional_tag → consolidation_strength={manual_tag.consolidation_strength:.3f}")

# Compare with engine-produced tag
encoded_mems = em.list_all()
engine_tag = encoded_mems[0].tag
print(f"  engine tag          → consolidation_strength={engine_tag.consolidation_strength:.3f}")

# ---------------------------------------------------------------------------
# 8. Retrieve + ResonanceLink traversal
# ---------------------------------------------------------------------------

print("\n=== ResonanceLink traversal ===\n")

em.set_affect(CoreAffect(valence=0.5, arousal=0.5))
results = em.retrieve("production incident response", top_k=3)

all_mems = {m.id: m for m in em.list_all()}

total_links = 0
for mem in results:
    links: list[ResonanceLink] = mem.tag.resonance_links  # explicit type annotation
    total_links += len(links)
    if links:
        top: ResonanceLink = max(links, key=lambda lnk: lnk.strength)
        target_snippet = all_mems[top.target_id].content[:30] if top.target_id in all_mems else "?"
        print(f"  '{mem.content[:40]}'")
        print(f"    → {top.link_type:12s}  strength={top.strength:.3f}  target='{target_snippet}'")
    else:
        print(f"  '{mem.content[:40]}'  — no resonance links")

print(f"\n  Total resonance links across top-3 results: {total_links}")

# ---------------------------------------------------------------------------
# 9. Async bridge via SyncToAsyncAppraisalEngine
# ---------------------------------------------------------------------------

print("\n=== Async bridge — SyncToAsyncAppraisalEngine ===\n")


async def run_async() -> None:
    async_engine = AsyncEmotionalMemory(
        store=SyncToAsyncStore(InMemoryStore()),
        embedder=SyncToAsyncEmbedder(HashEmbedder()),
        appraisal_engine=SyncToAsyncAppraisalEngine(appraisal_engine),
        config=config,
    )
    async with async_engine:
        async_engine.set_affect(CoreAffect(valence=0.7, arousal=0.6))
        mem = await async_engine.encode("Async encode: deployed the hotfix successfully.")
        print(f"  Encoded async: '{mem.content[:55]}'")
        print(
            f"    valence={mem.tag.core_affect.valence:+.2f}  "
            f"strength={mem.tag.consolidation_strength:.2f}"
        )

        results_async = await async_engine.retrieve("deployment success", top_k=1)
        print(f"  Retrieved:     '{results_async[0].content[:55]}'")

    print("  AsyncEmotionalMemory closed cleanly.")


asyncio.run(run_async())

print("\nDone.")
