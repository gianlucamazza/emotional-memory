"""Appraisal engines — automatic emotional tagging without manual set_affect().

Demonstrates three appraisal strategies that remove the need to call
set_affect() manually: KeywordAppraisalEngine (rule-based), a custom
KeywordRule list for a specific domain, and StaticAppraisalEngine for
testing and reproducibility.

Run with:
    python examples/appraisal_engines.py
"""

from emotional_memory import (
    AppraisalVector,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    KeywordAppraisalEngine,
    KeywordRule,
    RetrievalConfig,
    StaticAppraisalEngine,
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
# Default KeywordAppraisalEngine — built-in rules, no configuration needed
# ---------------------------------------------------------------------------

print("=== Default keyword appraisal ===\n")

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    appraisal_engine=KeywordAppraisalEngine(),  # uses 8 built-in rules
    config=EmotionalMemoryConfig(
        retrieval=RetrievalConfig(base_weights=[0.20, 0.30, 0.25, 0.10, 0.10, 0.05]),
    ),
)

events = [
    "The deployment succeeded — all health checks passed.",
    "The production database is down, users cannot log in.",
    "A new and unexpected edge case appeared in the parser.",
    "Finished migrating the old codebase, nothing surprising.",
]

for text in events:
    mem = em.encode(text)
    av = mem.tag.appraisal
    ca = mem.tag.core_affect
    sign = "[+]" if ca.valence > 0 else "[-]" if ca.valence < -0.1 else "[~]"
    print(f"{sign} {text[:60]}")
    if av is not None:
        print(
            f"    appraisal: novelty={av.novelty:+.2f}  goal_rel={av.goal_relevance:+.2f}"
            f"  coping={av.coping_potential:.2f}  self_rel={av.self_relevance:.2f}"
        )
    print(f"    affect:    valence={ca.valence:+.2f}  arousal={ca.arousal:.2f}")
    print(f"    strength:  {mem.tag.consolidation_strength:.2f}\n")

# ---------------------------------------------------------------------------
# Custom KeywordRule list — domain-specific vocabulary (customer support)
# ---------------------------------------------------------------------------

print("=== Custom rules: customer-support domain ===\n")

support_rules = [
    KeywordRule(
        r"\bresolved\b|\bclosed\b|\bfixed\b",
        goal_relevance=0.7,
        coping_potential=0.4,
        self_relevance=0.3,
    ),
    KeywordRule(
        r"\bescalated\b|\bescalation\b",
        goal_relevance=-0.5,
        coping_potential=-0.3,
        self_relevance=0.4,
    ),
    KeywordRule(
        r"\bSLA\s+breach\b|\boverdue\b",
        goal_relevance=-0.8,
        norm_congruence=-0.5,
        self_relevance=0.5,
    ),
    KeywordRule(r"\bfeedback\b|\bsurvey\b", novelty=0.2, goal_relevance=0.3, self_relevance=0.2),
    KeywordRule(
        r"\bcritical\b|\burgent\b|\bP1\b", novelty=0.4, goal_relevance=-0.6, coping_potential=-0.4
    ),
]

em_support = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    appraisal_engine=KeywordAppraisalEngine(rules=support_rules),
)

tickets = [
    "Ticket #4821 resolved after 2 hours — customer confirmed fix.",
    "Ticket #4830 escalated to Tier 2 due to repeated failures.",
    "SLA breach on #4815 — response exceeded 8-hour window.",
    "Received positive feedback survey from enterprise account.",
    "P1 critical outage reported — all hands on deck.",
]

for text in tickets:
    mem = em_support.encode(text)
    av = mem.tag.appraisal
    ca = mem.tag.core_affect
    sign = "[+]" if ca.valence > 0.1 else "[-]" if ca.valence < -0.1 else "[~]"
    print(f"{sign} {text[:65]}")
    if av is not None:
        strength = mem.tag.consolidation_strength
        print(f"    valence={ca.valence:+.2f}  arousal={ca.arousal:.2f}  strength={strength:.2f}")
    print()

# ---------------------------------------------------------------------------
# StaticAppraisalEngine — fixed vector for testing and reproducibility
# ---------------------------------------------------------------------------

print("=== StaticAppraisalEngine (fixed vector, useful for tests) ===\n")

fixed_av = AppraisalVector(
    novelty=0.6,
    goal_relevance=0.8,
    coping_potential=0.7,
    norm_congruence=0.5,
    self_relevance=0.9,
)
derived_ca = fixed_av.to_core_affect()
print(f"Fixed appraisal: novelty={fixed_av.novelty}  goal_rel={fixed_av.goal_relevance}")
print(f"Derived affect:  valence={derived_ca.valence:.3f}  arousal={derived_ca.arousal:.3f}\n")

em_static = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    appraisal_engine=StaticAppraisalEngine(vector=fixed_av),
)

# All memories get the same appraisal — useful for unit tests that need
# deterministic emotional state regardless of content.
for text in ["Alpha event.", "Beta event.", "Gamma event."]:
    mem = em_static.encode(text)
    ca = mem.tag.core_affect
    print(f"  '{text}' → valence={ca.valence:+.3f}  arousal={ca.arousal:.3f}")

print("\nAll three memories share identical affect (static engine).")

# ---------------------------------------------------------------------------
# Mood-congruent retrieval driven by automatic appraisal
# ---------------------------------------------------------------------------

print("\n=== Retrieval after automatic appraisal ===\n")

results = em.retrieve("production incident failure", top_k=3)
for i, mem in enumerate(results, 1):
    ca = mem.tag.core_affect
    sign = "[+]" if ca.valence > 0 else "[-]"
    print(f"  {i}. {sign} {mem.content[:60]}")
    print(f"       valence={ca.valence:+.2f}  retrieval_count={mem.tag.retrieval_count}")

print("\nDone.")
