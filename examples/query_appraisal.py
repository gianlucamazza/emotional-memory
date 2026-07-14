"""Retrieve-time query appraisal and gated retrieval (Addenda T + Y).

Run with:
    python examples/query_appraisal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _embedders import HashEmbedder

from emotional_memory import EmotionalMemory, InMemoryStore, KeywordAppraisalEngine


def main() -> None:
    em = EmotionalMemory(
        store=InMemoryStore(),
        embedder=HashEmbedder(),
        appraisal_engine=KeywordAppraisalEngine(),
    )
    em.encode("I felt anxious before the job interview.")
    em.encode("The weather was sunny and calm.")

    print("=== retrieve_with_query_appraisal ===")
    for mem in em.retrieve_with_query_appraisal("I'm nervous about tomorrow", top_k=2):
        print(f"  {mem.content!r}")

    print("=== retrieve_query_gated (neutral factual query) ===")
    for mem in em.retrieve_query_gated("What happened with the weather?", top_k=2):
        print(f"  {mem.content!r}")


if __name__ == "__main__":
    main()
