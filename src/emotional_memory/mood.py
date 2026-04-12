"""Layer 3: MoodField — the slow-moving global mood background.

Inspired by Heidegger's notion of Stimmung (Being and Time §29) — mood as
something that is always already there, orienting every cognitive operation
rather than being a coloring added to neutral experience. The implementation
operationalises this as a slow EMA of accumulated CoreAffect history, drawing
more directly on Bower's (1981) mood-congruent recall and the PAD model
(Mehrabian & Russell, 1974).

Unlike CoreAffect (which changes per-event), MoodField changes slowly
through exponential moving average — it is the gravitational field through
which all memory encoding and retrieval is curved.

The PAD model is used as the coordinate system:
  Pleasure (valence), Arousal, Dominance.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from pydantic import BaseModel, field_validator, model_validator

from emotional_memory.affect import CoreAffect


class MoodDecayConfig(BaseModel):
    """Parameters for time-based regression of MoodField toward baseline.

    Implements exponential decay toward a resting baseline:
        regressed = baseline + (current - baseline) * exp(-t * ln2 / half_life)

    The effective half-life is modulated by inertia so that a rigid mood
    (high inertia) regresses more slowly than a volatile one.
    """

    base_half_life_seconds: float = 3600.0
    """Base half-life in seconds (default 1 hour, must be > 0). Modulated by inertia."""

    inertia_scale: float = 2.0
    """Multiplier on inertia: effective_half_life = base * (1 + inertia * scale)."""

    baseline_valence: float = 0.0
    """Resting valence attractor (neutral)."""

    baseline_arousal: float = 0.3
    """Resting arousal attractor (not zero — resting alertness, not inert)."""

    baseline_dominance: float = 0.5
    """Resting dominance attractor (balanced perceived control)."""

    @model_validator(mode="after")
    def _validate_half_life(self) -> MoodDecayConfig:
        if self.base_half_life_seconds <= 0:
            raise ValueError(
                f"base_half_life_seconds must be > 0, got {self.base_half_life_seconds}"
            )
        return self


class MoodField(BaseModel):
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
    def neutral(cls) -> MoodField:
        return cls(
            valence=0.0,
            arousal=0.0,
            dominance=0.5,
            inertia=0.5,
            timestamp=datetime.now(tz=UTC),
        )

    def regress(self, now: datetime, config: MoodDecayConfig) -> MoodField:
        """Return a new MoodField regressed toward baseline by elapsed time.

        Uses exponential decay:
            value(t) = baseline + (value - baseline) * exp(-t * ln2 / half_life)

        Effective half-life is modulated by inertia: high-inertia moods
        resist regression (as in trauma or persistent elation).
        """
        elapsed = max(0.0, (now - self.timestamp).total_seconds())
        eff_hl = config.base_half_life_seconds * (1.0 + self.inertia * config.inertia_scale)
        if eff_hl <= 0.0:
            return self
        factor = math.exp(-elapsed * math.log(2) / eff_hl)
        return MoodField(
            valence=config.baseline_valence + (self.valence - config.baseline_valence) * factor,
            arousal=config.baseline_arousal + (self.arousal - config.baseline_arousal) * factor,
            dominance=config.baseline_dominance
            + (self.dominance - config.baseline_dominance) * factor,
            inertia=self.inertia,
            timestamp=now,
        )

    def update(
        self,
        core_affect: CoreAffect,
        alpha: float = 0.1,
        now: datetime | None = None,
        decay_config: MoodDecayConfig | None = None,
    ) -> MoodField:
        """Return a new MoodField updated via exponential moving average.

        If ``decay_config`` is provided, temporal regression toward baseline is
        applied first (accounting for elapsed silence since last update), then
        the EMA step runs on the regressed state.

        Effective alpha is modulated by inertia:
            effective_alpha = alpha * (1 - inertia)

        High inertia → mood barely moves per event.
        Low inertia  → mood follows affect closely.

        Dominance is updated via the PAD model heuristic: positive valence +
        high arousal signals perceived control, negative valence + high arousal
        signals threat and loss of control.
        """
        if now is None:
            now = datetime.now(tz=UTC)

        # Step 1: regress toward baseline for elapsed silent time
        base = self.regress(now, decay_config) if decay_config is not None else self

        eff_alpha = alpha * (1.0 - base.inertia)
        new_valence = (1.0 - eff_alpha) * base.valence + eff_alpha * core_affect.valence
        new_arousal = (1.0 - eff_alpha) * base.arousal + eff_alpha * core_affect.arousal
        # Dominance: PAD heuristic — valence * arousal shifts perceived control.
        # Coefficient 0.5 gives full [-1,1] * [0,1] -> [0, 1] range coverage.
        dominance_signal = 0.5 + 0.5 * core_affect.valence * core_affect.arousal
        new_dominance = (1.0 - eff_alpha) * base.dominance + eff_alpha * dominance_signal
        return MoodField(
            valence=new_valence,
            arousal=new_arousal,
            dominance=new_dominance,
            inertia=base.inertia,
            timestamp=now,
        )

    def distance(self, other: MoodField) -> float:
        """Euclidean distance in valence-arousal-dominance space."""
        return math.sqrt(
            (self.valence - other.valence) ** 2
            + (self.arousal - other.arousal) ** 2
            + (self.dominance - other.dominance) ** 2
        )
