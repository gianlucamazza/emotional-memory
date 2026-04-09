"""Step 9: AffectiveState — the runtime emotional state machine.

Bundles CoreAffect, AffectiveMomentum, and StimmungField into a single
evolving object. On each update:
  1. Momentum is recalculated from the recent history of CoreAffect values
     via finite differences (velocity = current - previous,
     acceleration = current_velocity - previous_velocity).
  2. StimmungField is updated via EMA.
  3. History is maintained for momentum computation (last 3 points).

AffectiveState is immutable — update() returns a new instance.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, PrivateAttr

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.stimmung import StimmungField

# History entry: (iso_timestamp, valence, arousal)
_HistoryEntry = tuple[str, float, float]


class AffectiveState(BaseModel):
    """Snapshot of the system's complete affective state at a point in time."""

    core_affect: CoreAffect
    momentum: AffectiveMomentum
    stimmung: StimmungField

    # Private: not serialised, not part of the schema.
    # Stores the last 3 (timestamp, valence, arousal) points for momentum.
    _history: list[_HistoryEntry] = PrivateAttr(default_factory=list)

    @classmethod
    def initial(cls) -> AffectiveState:
        """Neutral starting state."""
        return cls(
            core_affect=CoreAffect.neutral(),
            momentum=AffectiveMomentum.zero(),
            stimmung=StimmungField.neutral(),
        )

    def update(
        self,
        new_affect: CoreAffect,
        now: datetime | None = None,
        stimmung_alpha: float = 0.1,
    ) -> AffectiveState:
        """Return a new AffectiveState reflecting the updated affect.

        Momentum is computed from finite differences over the last 3 history
        points (velocity = Δaffect, acceleration = Δvelocity).
        """
        if now is None:
            now = datetime.now(tz=UTC)

        history = [*self._history, (now.isoformat(), new_affect.valence, new_affect.arousal)]
        history = history[-3:]

        momentum = _compute_momentum(history)
        new_stimmung = self.stimmung.update(new_affect, alpha=stimmung_alpha)

        next_state = AffectiveState(
            core_affect=new_affect,
            momentum=momentum,
            stimmung=new_stimmung,
        )
        next_state._history = history
        return next_state


def _compute_momentum(history: list[_HistoryEntry]) -> AffectiveMomentum:
    """Finite differences over last ≤3 history points."""
    if len(history) < 2:
        return AffectiveMomentum.zero()

    _, v1, a1 = history[-2]
    _, v2, a2 = history[-1]
    d_v = v2 - v1
    d_a = a2 - a1

    if len(history) < 3:
        return AffectiveMomentum(d_valence=d_v, d_arousal=d_a)

    _, v0, a0 = history[-3]
    dd_v = d_v - (v1 - v0)
    dd_a = d_a - (a1 - a0)

    return AffectiveMomentum(d_valence=d_v, d_arousal=d_a, dd_valence=dd_v, dd_arousal=dd_a)
