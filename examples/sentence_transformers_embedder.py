"""Real embedder integration — sentence-transformers via SequentialEmbedder.

Shows how to wrap any embedding model using the SequentialEmbedder base class:
implement embed() and get embed_batch() for free, or override it for native
batching efficiency.

Falls back to HashEmbedder when sentence-transformers is not installed, so
the script is always runnable. With real embeddings the semantic signal (s1)
captures genuine meaning; with HashEmbedder it is effectively random noise.

Requires (for real embeddings):
    pip install sentence-transformers

Run with:
    python examples/sentence_transformers_embedder.py
"""

from emotional_memory import (
    CoreAffect,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    ResonanceConfig,
    RetrievalConfig,
    SequentialEmbedder,
)

# ---------------------------------------------------------------------------
# HashEmbedder — fallback when sentence-transformers is not installed
# ---------------------------------------------------------------------------


class HashEmbedder(SequentialEmbedder):
    """Deterministic 8-dim embedder based on string hashing.

    Subclasses SequentialEmbedder and only implements embed() —
    embed_batch() is inherited as a sequential loop over embed().
    """

    DIM = 8

    def embed(self, text: str) -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        vec = [((h >> i) & 0xFF) / 255.0 for i in range(self.DIM)]
        total = sum(vec) or 1.0
        return [v / total for v in vec]


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder — real embedder via SequentialEmbedder
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder(SequentialEmbedder):
    """Wraps sentence-transformers for use with EmotionalMemory.

    Overrides embed_batch() to use the model's native batching rather than
    the sequential fallback provided by SequentialEmbedder.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Import inside __init__ so ImportError is raised at instantiation,
        # not at module load time — enabling clean try/except at call site.
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()  # type: ignore[no-any-return]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Native batching: one forward pass for all texts (more efficient
        # than the SequentialEmbedder default of calling embed() N times).
        return self._model.encode(texts).tolist()  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model_name!r})"


# ---------------------------------------------------------------------------
# Select embedder — real or fallback
# ---------------------------------------------------------------------------

try:
    embedder = SentenceTransformerEmbedder()
    mode = f"sentence-transformers ({embedder._model_name})"
    dim = len(embedder.embed("test"))
except ImportError:
    embedder = HashEmbedder()
    mode = "HashEmbedder (fallback — install sentence-transformers for real embeddings)"
    dim = HashEmbedder.DIM

print(f"Embedder : {mode}")
print(f"Dimension: {dim}\n")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=embedder,
    config=EmotionalMemoryConfig(
        resonance=ResonanceConfig(threshold=0.10),
        retrieval=RetrievalConfig(base_weights=[0.40, 0.25, 0.15, 0.08, 0.08, 0.04]),
        mood_alpha=0.2,
    ),
)

# ---------------------------------------------------------------------------
# Encode memories — two semantic clusters + one outlier
# ---------------------------------------------------------------------------

print("=== Encoding memories ===\n")

memories = [
    # Cluster A: infrastructure / CI-CD
    (
        CoreAffect(valence=-0.6, arousal=0.7),
        "The deployment pipeline failed due to a misconfigured Docker image.",
    ),
    (
        CoreAffect(valence=0.5, arousal=0.5),
        "Fixed the CI/CD pipeline by pinning the base image to a stable tag.",
    ),
    (
        CoreAffect(valence=-0.7, arousal=0.8),
        "Critical bug in the authentication service caused a security incident.",
    ),
    (
        CoreAffect(valence=0.6, arousal=0.5),
        "Patched the auth service and added integration tests to prevent recurrence.",
    ),
    # Cluster B: team / social
    (
        CoreAffect(valence=0.8, arousal=0.6),
        "Had a wonderful team offsite — everyone left energised and aligned.",
    ),
    (
        CoreAffect(valence=0.7, arousal=0.4),
        "One-on-one with my manager — clear goals set for the next quarter.",
    ),
    # Outlier
    (
        CoreAffect(valence=0.3, arousal=0.2),
        "Quiet Saturday morning: coffee, a book, no notifications.",
    ),
]

for affect, text in memories:
    em.set_affect(affect)
    mem = em.encode(text)
    sign = "[+]" if affect.valence > 0 else "[-]"
    print(f"{sign} {text[:65]}")
    strength = mem.tag.consolidation_strength
    print(
        f"   valence={affect.valence:+.1f}  arousal={affect.arousal:.1f}  strength={strength:.2f}"
    )

# ---------------------------------------------------------------------------
# embed_batch — efficiency demonstration
# ---------------------------------------------------------------------------

print("\n=== embed_batch ===\n")

test_texts = [
    "infrastructure failure",
    "successful deployment",
    "team collaboration",
]
batch_embeddings = embedder.embed_batch(test_texts)
print(f"Batch-embedded {len(test_texts)} texts — dim={len(batch_embeddings[0])}")
for text, emb in zip(test_texts, batch_embeddings, strict=True):
    print(f"  '{text}' → [{emb[0]:.4f}, {emb[1]:.4f}, ...]")

# ---------------------------------------------------------------------------
# Retrieval quality comparison
# ---------------------------------------------------------------------------

print("\n=== Retrieval: 'fixing infrastructure problems' ===\n")

em.set_affect(CoreAffect(valence=0.0, arousal=0.5))
results = em.retrieve("fixing infrastructure problems", top_k=4)

for i, mem in enumerate(results, 1):
    ca = mem.tag.core_affect
    sign = "[+]" if ca.valence > 0 else "[-]"
    print(f"  {i}. {sign} {mem.content[:65]}")
    print(f"       valence={ca.valence:+.2f}  retrievals={mem.tag.retrieval_count}")

print()
if isinstance(embedder, SentenceTransformerEmbedder):
    print("With real embeddings: infrastructure/ops memories rank first because")
    print("the semantic signal (s1, weight=0.40) captures genuine meaning.")
else:
    print("With HashEmbedder: ranking is driven by emotional/Mood signals")
    print("(s2, s3) since the semantic signal is hash-based and effectively random.")
    print("Install sentence-transformers to see semantic retrieval in action.")

print("\n=== Retrieval: 'team alignment and goals' ===\n")

results2 = em.retrieve("team alignment and goals", top_k=3)
for i, mem in enumerate(results2, 1):
    ca = mem.tag.core_affect
    sign = "[+]" if ca.valence > 0 else "[-]"
    print(f"  {i}. {sign} {mem.content[:65]}")

print("\nDone.")
