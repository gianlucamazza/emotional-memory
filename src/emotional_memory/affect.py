"""Layer 1 & 2: CoreAffect and AffectiveMomentum.

CoreAffect is the fundamental continuous substrate — a point in the
PAD (Pleasure-Arousal-Dominance) space (Mehrabian & Russell, 1974;
Russell, 1980).

AffectiveMomentum captures the first and second time-derivatives of
CoreAffect — velocity and acceleration over recent history. Inspired by
the idea that affect is inherently a transition rather than a static state
(cf. Spinoza, Ethics III: "Laetitia est transitio hominis a minore ad
maiorem perfectionem").
"""

from __future__ import annotations

import math

from pydantic import BaseModel, field_validator


class CoreAffect(BaseModel):
    """A point in the valence-arousal-dominance (PAD) space.

    valence   : [-1.0, +1.0]  negative → positive
    arousal   : [ 0.0,  1.0]  calm → highly activated
    dominance : [ 0.0,  1.0]  submissive → dominant/in-control

    References: Mehrabian & Russell (1974); Russell (1980).
    """

    model_config = {"frozen": True}

    valence: float
    arousal: float
    dominance: float = 0.5

    @field_validator("valence")
    @classmethod
    def _clamp_valence(cls, v: float) -> float:
        return max(-1.0, min(1.0, v))

    @field_validator("arousal", "dominance")
    @classmethod
    def _clamp_unit(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @classmethod
    def neutral(cls) -> CoreAffect:
        return cls(valence=0.0, arousal=0.0, dominance=0.5)

    def distance(self, other: CoreAffect) -> float:
        """Euclidean distance in valence-arousal-dominance space."""
        return math.sqrt(
            (self.valence - other.valence) ** 2
            + (self.arousal - other.arousal) ** 2
            + (self.dominance - other.dominance) ** 2
        )

    def lerp(self, other: CoreAffect, alpha: float) -> CoreAffect:
        """Linear interpolation. alpha=0 → self, alpha=1 → other."""
        alpha = max(0.0, min(1.0, alpha))
        return CoreAffect(
            valence=self.valence + alpha * (other.valence - self.valence),
            arousal=self.arousal + alpha * (other.arousal - self.arousal),
            dominance=self.dominance + alpha * (other.dominance - self.dominance),
        )


class AffectiveMomentum(BaseModel):
    """Time-derivatives of CoreAffect.

    Captures the direction and speed of affective change — velocity and
    acceleration in valence-arousal-dominance space (inspired by the view
    that affect is inherently a process of transition, cf. Spinoza).

    d_valence / d_arousal / d_dominance  : first derivatives  (velocity)
    dd_valence / dd_arousal / dd_dominance: second derivatives (acceleration)
    """

    model_config = {"frozen": True}

    d_valence: float = 0.0
    d_arousal: float = 0.0
    d_dominance: float = 0.0
    dd_valence: float = 0.0
    dd_arousal: float = 0.0
    dd_dominance: float = 0.0

    @classmethod
    def zero(cls) -> AffectiveMomentum:
        return cls()

    def magnitude(self) -> float:
        """Speed in affect-space (norm of 3D velocity vector)."""
        return math.sqrt(self.d_valence**2 + self.d_arousal**2 + self.d_dominance**2)
