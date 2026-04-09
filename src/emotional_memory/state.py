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
from typing import Any

from pydantic import BaseModel, PrivateAttr

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.stimmung import StimmungDecayConfig, StimmungField

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

    def snapshot(self) -> dict[str, Any]:
        """Serialise to a dict including the private momentum history.

        Suitable for JSON persistence and later restoration via
        ``AffectiveState.restore()``.  Pydantic's normal serialisation
        excludes ``_history``; this method adds it back.
        """
        data: dict[str, Any] = self.model_dump(mode="json")
        data["_history"] = list(self._history)
        return data

    @classmethod
    def restore(cls, data: dict[str, Any]) -> AffectiveState:
        """Reconstruct an AffectiveState from a snapshot dict.

        Restores the private momentum history that Pydantic excludes from
        normal serialisation, so that the next ``update()`` call produces
        correct momentum derivatives.
        """
        raw_history = data.get("_history", [])
        filtered = {k: v for k, v in data.items() if k != "_history"}
        state = cls.model_validate(filtered)
        state._history = [
            (str(entry[0]), float(entry[1]), float(entry[2]))
            for entry in raw_history
            if len(entry) >= 3
        ]
        return state

    def update(
        self,
        new_affect: CoreAffect,
        now: datetime | None = None,
        stimmung_alpha: float = 0.1,
        stimmung_decay: StimmungDecayConfig | None = None,
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
        new_stimmung = self.stimmung.update(
            new_affect, alpha=stimmung_alpha, now=now, decay_config=stimmung_decay
        )

        next_state = AffectiveState(
            core_affect=new_affect,
            momentum=momentum,
            stimmung=new_stimmung,
        )
        next_state._history = history
        return next_state


def _compute_momentum(history: list[_HistoryEntry]) -> AffectiveMomentum:
    """Time-normalised finite differences over last ≤3 history points.

    velocity  = Δaffect / Δt   (per-second rate of change)
    acceleration = Δvelocity    (second finite difference, not re-normalised
                                 since the two Δt intervals are typically equal)
    """
    if len(history) < 2:
        return AffectiveMomentum.zero()

    t1_str, v1, a1 = history[-2]
    t2_str, v2, a2 = history[-1]
    t1 = datetime.fromisoformat(t1_str)
    t2 = datetime.fromisoformat(t2_str)
    dt12 = max((t2 - t1).total_seconds(), 0.001)

    d_v = (v2 - v1) / dt12
    d_a = (a2 - a1) / dt12

    if len(history) < 3:
        return AffectiveMomentum(d_valence=d_v, d_arousal=d_a)

    t0_str, v0, a0 = history[-3]
    t0 = datetime.fromisoformat(t0_str)
    dt01 = max((t1 - t0).total_seconds(), 0.001)

    d_v0 = (v1 - v0) / dt01
    d_a0 = (a1 - a0) / dt01

    dd_v = d_v - d_v0
    dd_a = d_a - d_a0

    return AffectiveMomentum(d_valence=d_v, d_arousal=d_a, dd_valence=dd_v, dd_arousal=dd_a)
