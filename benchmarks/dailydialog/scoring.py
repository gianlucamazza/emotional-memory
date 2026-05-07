"""Scoring functions for the DailyDialog affect-conditioned retrieval benchmark.

Evaluation is purely retrieval-based: top-k accuracy and hit@k.
A prediction is correct if any retrieved memory belongs to the target session.
"""

from __future__ import annotations

from benchmarks.dailydialog.dataset import PersonaQuery


def top1_correct(retrieved_session_ids: list[str], query: PersonaQuery) -> bool:
    """Return True if the top-1 retrieved memory is from the target session."""
    if not retrieved_session_ids:
        return False
    return retrieved_session_ids[0] == query.target_session_id


def hit_at_k(retrieved_session_ids: list[str], query: PersonaQuery) -> bool:
    """Return True if any of the top-k retrieved memories is from the target session."""
    return query.target_session_id in retrieved_session_ids[: query.top_k]
