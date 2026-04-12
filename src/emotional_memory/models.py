"""Core data models: ResonanceLink, EmotionalTag, Memory.

EmotionalTag is the full affective fingerprint of a memory — a snapshot
of all five AFT layers at encoding time plus consolidation and
reconsolidation metadata.

Every Memory carries an EmotionalTag. There are no affectively neutral
memories (Colombetti, 2014; Richter-Levin, 2004).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.appraisal import AppraisalVector

# EmotionLabel is imported directly (not under TYPE_CHECKING) because Pydantic
# needs the class at model-definition time for field type resolution when
# `from __future__ import annotations` is active.  categorize.py is a leaf
# module with no runtime imports from models.py, so there is no circular dep.
from emotional_memory.categorize import EmotionLabel
from emotional_memory.mood import MoodField


class ResonanceLink(BaseModel):
    """A directed affective-associative link between two memories.

    link_type encodes the dominant associative principle (Aristotle's
    laws of association + Bower's affective network theory):
      semantic    — similar content
      emotional   — similar core_affect at encoding
      temporal    — close in time
      causal      — perceived causal relationship
      contrastive — opposing affective valence (anxiety ↔ relief)
    """

    model_config = {"frozen": True}

    source_id: str
    target_id: str
    strength: float = Field(ge=0.0, le=1.0)
    link_type: Literal["semantic", "emotional", "temporal", "causal", "contrastive"]


class EmotionalTag(BaseModel):
    """Full affective fingerprint of a memory at encoding time.

    Captures all five AFT layers as snapshots:
      Layer 1: core_affect          — where the system was
      Layer 2: momentum             — where it was heading
      Layer 3: mood_snapshot    — global mood background
      Layer 4: appraisal            — why this emotion arose (if computed)
      Layer 5: resonance_links      — associative cluster membership

    consolidation_strength modulates decay rate (McGaugh, 2004):
      high arousal → strong consolidation → slow decay → high retrieval priority

    Reconsolidation (Nader & Schiller, 2000): every retrieval may update
    the tag. reconsolidation_count tracks how many times this occurred.
    reconsolidation_window is managed at runtime via last_retrieved +
    window_seconds — not stored as a boolean flag.

    Frozen like all other value objects — use model_copy(update=...) to derive modified versions.
    """

    model_config = ConfigDict(frozen=True)

    core_affect: CoreAffect
    momentum: AffectiveMomentum
    mood_snapshot: MoodField
    appraisal: AppraisalVector | None = None
    resonance_links: list[ResonanceLink] = Field(default_factory=list)
    timestamp: datetime
    consolidation_strength: float  # [0.0, 1.0]
    last_retrieved: datetime | None = None
    retrieval_count: int = 0
    reconsolidation_count: int = 0

    # Dual-path encoding (LeDoux, 1996)
    pending_appraisal: bool = False
    """True when fast-path encoding was used; appraisal is deferred to elaborate()."""

    # APE-gated reconsolidation window
    window_opened_at: datetime | None = None
    """Timestamp when the reconsolidation lability window was opened (APE-gated)."""

    # Pearce-Hall predictive learning
    expected_affect: CoreAffect | None = None
    """EMA prediction of core_affect, updated by update_prediction() on retrieval."""

    prediction_learning_rate: float = 0.2
    """Adaptive learning rate for expected_affect EMA. Clamped to [0.05, 0.80]."""

    # Plutchik discrete emotion label (auto_categorize)
    emotion_label: EmotionLabel | None = None
    """Discrete Plutchik emotion label, set when auto_categorize=True."""


class Memory(BaseModel):
    """A stored experience with its full affective fingerprint."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    embedding: list[float] | None = None
    tag: EmotionalTag
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        content: str,
        tag: EmotionalTag,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        return cls(
            content=content,
            tag=tag,
            embedding=embedding,
            metadata=metadata or {},
        )


def _now() -> datetime:
    return datetime.now(tz=UTC)


def make_emotional_tag(
    core_affect: CoreAffect,
    momentum: AffectiveMomentum,
    mood: MoodField,
    consolidation_strength: float,
    appraisal: AppraisalVector | None = None,
) -> EmotionalTag:
    """Convenience constructor for EmotionalTag with sensible defaults."""
    return EmotionalTag(
        core_affect=core_affect,
        momentum=momentum,
        mood_snapshot=mood,
        appraisal=appraisal,
        timestamp=_now(),
        consolidation_strength=max(0.0, min(1.0, consolidation_strength)),
    )
