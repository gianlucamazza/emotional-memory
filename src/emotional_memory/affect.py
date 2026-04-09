"""Layer 1 & 2: CoreAffect and AffectiveMomentum.

CoreAffect is the fundamental continuous substrate — a point in the
valence-arousal circumplex (Russell, 1980).

AffectiveMomentum captures the first and second time-derivatives of
CoreAffect, implementing Spinoza's insight that affect *is* transition,
not a static state ("Laetitia est transitio hominis a minore ad maiorem
perfectionem", Ethics III).
"""

from __future__ import annotations

import math

from pydantic import BaseModel, field_validator


class CoreAffect(BaseModel):
    """A point in the valence-arousal circumplex.

    valence: [-1.0, +1.0]  negative → positive
    arousal: [ 0.0,  1.0]  calm → highly activated
    """

    model_config = {"frozen": True}

    valence: float
    arousal: float

    @field_validator("valence")
    @classmethod
    def _clamp_valence(cls, v: float) -> float:
        return max(-1.0, min(1.0, v))

    @field_validator("arousal")
    @classmethod
    def _clamp_arousal(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @classmethod
    def neutral(cls) -> CoreAffect:
        return cls(valence=0.0, arousal=0.0)

    def distance(self, other: CoreAffect) -> float:
        """Euclidean distance in valence-arousal space."""
        return math.sqrt((self.valence - other.valence) ** 2 + (self.arousal - other.arousal) ** 2)

    def lerp(self, other: CoreAffect, alpha: float) -> CoreAffect:
        """Linear interpolation. alpha=0 → self, alpha=1 → other."""
        alpha = max(0.0, min(1.0, alpha))
        return CoreAffect(
            valence=self.valence + alpha * (other.valence - self.valence),
            arousal=self.arousal + alpha * (other.arousal - self.arousal),
        )


class AffectiveMomentum(BaseModel):
    """Time-derivatives of CoreAffect.

    Captures the direction and speed of affective change — the Spinozist
    "transition" that is constitutive of the emotion itself.

    d_valence / d_arousal  : first derivatives  (velocity)
    dd_valence / dd_arousal: second derivatives (acceleration)
    """

    model_config = {"frozen": True}

    d_valence: float = 0.0
    d_arousal: float = 0.0
    dd_valence: float = 0.0
    dd_arousal: float = 0.0

    @classmethod
    def zero(cls) -> AffectiveMomentum:
        return cls()

    def magnitude(self) -> float:
        """Speed in affect-space (norm of velocity vector)."""
        return math.sqrt(self.d_valence**2 + self.d_arousal**2)
