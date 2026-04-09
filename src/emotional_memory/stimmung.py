"""Layer 3: StimmungField — the slow-moving global mood.

Heidegger's Stimmung (Being and Time §29) is not a coloring added to
neutral experience but a fundamental ontological disclosure: the system
is always already in a mood that orients every cognitive operation.

Unlike CoreAffect (which changes per-event), StimmungField changes slowly
through exponential moving average — it is the gravitational field through
which all memory encoding and retrieval is curved.

The PAD model (Mehrabian & Russell, 1974) is used here as the coordinate
system: Pleasure (valence), Arousal, Dominance.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from pydantic import BaseModel, field_validator

from emotional_memory.affect import CoreAffect


class StimmungField(BaseModel):
    """Global mood field — slow EMA of accumulated CoreAffect history.

    valence   : [-1.0, +1.0] negative ↔ positive
    arousal   : [ 0.0,  1.0] calm ↔ highly activated
    dominance : [ 0.0,  1.0] submissive ↔ dominant/in-control
    inertia   : [ 0.0,  1.0] 0=volatile, 1=completely stable
    """

    model_config = {"frozen": True}

    valence: float
    arousal: float
    dominance: float
    inertia: float
    timestamp: datetime

    @field_validator("valence")
    @classmethod
    def _clamp_valence(cls, v: float) -> float:
        return max(-1.0, min(1.0, v))

    @field_validator("arousal", "dominance", "inertia")
    @classmethod
    def _clamp_unit(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @classmethod
    def neutral(cls) -> StimmungField:
        return cls(
            valence=0.0,
            arousal=0.0,
            dominance=0.5,
            inertia=0.5,
            timestamp=datetime.now(tz=UTC),
        )

    def update(self, core_affect: CoreAffect, alpha: float = 0.1) -> StimmungField:
        """Return a new StimmungField updated via exponential moving average.

        Effective alpha is modulated by inertia:
            effective_alpha = alpha * (1 - inertia)

        High inertia → Stimmung barely moves per event.
        Low inertia  → Stimmung follows affect closely.
        """
        eff_alpha = alpha * (1.0 - self.inertia)
        new_valence = (1.0 - eff_alpha) * self.valence + eff_alpha * core_affect.valence
        new_arousal = (1.0 - eff_alpha) * self.arousal + eff_alpha * core_affect.arousal
        # Dominance tracks arousal direction as a rough proxy (no input yet)
        new_dominance = self.dominance
        return StimmungField(
            valence=new_valence,
            arousal=new_arousal,
            dominance=new_dominance,
            inertia=self.inertia,
            timestamp=datetime.now(tz=UTC),
        )

    def distance(self, other: StimmungField) -> float:
        """Euclidean distance in valence-arousal-dominance space."""
        return math.sqrt(
            (self.valence - other.valence) ** 2
            + (self.arousal - other.arousal) ** 2
            + (self.dominance - other.dominance) ** 2
        )
