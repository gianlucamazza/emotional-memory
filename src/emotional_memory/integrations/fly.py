"""affective-fly circuit as the CoreAffect source.

emotional-memory is the host: it owns time, ``mood_dt``, and the memory
store. Each tick asks affective-fly for valence / arousal / approach-avoid.

This is the inverse of ``python -m affective_fly run`` / ``study``, which
construct ``EmotionalMemory`` inside the fly package. Do not invert that
here. Fly Policy / LaunchGate stay at library defaults and are not the
product. Phase 6 hypothesis taus (300 / 60 / 180) are not retuned.

Requires `affective-fly`_ (not on PyPI)::

    pip install "affective-fly @ git+https://github.com/gianlucamazza/affective-fly.git"

.. _affective-fly: https://github.com/gianlucamazza/affective-fly
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

try:
    from affective_fly import (
        AffectiveLoop,
        FakeEmbedder,
        FlyAffectReadout,
        HostAdapter,
        HostFrame,
        LIFCircuit,
        MoodField,
        mood_dts_from_timestamps,
    )
except ImportError as exc:
    raise ImportError(
        "FlyAffectHost requires affective-fly (not on PyPI). "
        "Install with: pip install "
        "'affective-fly @ git+https://github.com/gianlucamazza/affective-fly.git'"
    ) from exc

from emotional_memory.engine import EmotionalMemory
from emotional_memory.interfaces import Embedder, MemoryStore
from emotional_memory.stores.in_memory import InMemoryStore


@dataclass(frozen=True)
class FlyAffect:
    """Affect this host received from the fly this tick.

    Policy actions stay on the fly side. The host product reads these
    three numbers (plus the ``mood_dt`` it supplied).
    """

    valence: float
    arousal: float
    approach: float
    mood_dt: float


class FlyAffectHost:
    """Thin host: owns memory and the clock; fly is a dependency.

    Follows ``examples/host_owns_loop.py`` and ``docs/HOST_INTEGRATION.md``
    in affective-fly. The host constructs ``EmotionalMemory`` and computes
    ``mood_dt`` (wall-clock or HostFrame timestamps). Each tick calls
    ``AffectiveLoop`` / ``HostAdapter`` and returns valence, arousal, and
    approach — not a fly Policy action.
    """

    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        embedder: Embedder | None = None,
        memory: EmotionalMemory | None = None,
        fly_circuit: FlyAffectReadout | None = None,
        mood_field: MoodField | None = None,
    ) -> None:
        if memory is None:
            self.store = store if store is not None else InMemoryStore()
            self.embedder = embedder if embedder is not None else FakeEmbedder()
            self.memory = EmotionalMemory(store=self.store, embedder=self.embedder)
        else:
            if store is None or embedder is None:
                raise ValueError(
                    "when passing memory=, also pass the same store= and embedder= "
                    "instances used to construct it (AffectiveLoop cannot read "
                    "EmotionalMemory internals)"
                )
            self.store = store
            self.embedder = embedder
            self.memory = memory
        # MoodField() is the Phase 6 hypothesis (300 / 60 / 180). Not lab 8/4/5.
        self.mood = mood_field if mood_field is not None else MoodField()
        circuit = fly_circuit or LIFCircuit(n_kc=200, n_dan=20, n_mbon=34, seed=42)
        self.loop = AffectiveLoop(
            fly_circuit=circuit,
            emotional_memory=self.memory,
            store=self.store,
            embedder=self.embedder,
            mood_field=self.mood,
        )
        self._prev_now: float | None = None

    def tick(self, frame: HostFrame, mood_dt: float) -> FlyAffect:
        """Host-owned tick: pass ``mood_dt`` in, read affect back."""
        decision = self.loop.step(frame.to_sensory_frame(), mood_dt=float(mood_dt))
        return FlyAffect(
            valence=decision.mood_valence,
            arousal=decision.mood_arousal,
            approach=decision.approach_tendency,
            mood_dt=float(mood_dt),
        )

    def tick_wall_clock(self, frame: HostFrame, now: float) -> FlyAffect:
        """Live tick: host computes ``mood_dt`` from its own clock.

        ``now`` is seconds from ``time.monotonic()`` (or a test double).
        The first tick is 0.0 — there is no previous sample to invent.
        """
        mood_dt = 0.0 if self._prev_now is None else max(0.0, float(now) - self._prev_now)
        self._prev_now = float(now)
        return self.tick(frame, mood_dt)

    def replay(self, frames: Sequence[HostFrame]) -> list[FlyAffect]:
        """Replay HostFrames with ``mood_dt`` from their timestamps."""
        dts = mood_dts_from_timestamps(frames)
        decisions = HostAdapter.replay(list(frames), self.loop, mood_dts=dts)
        return [
            FlyAffect(
                valence=decision.mood_valence,
                arousal=decision.mood_arousal,
                approach=decision.approach_tendency,
                mood_dt=float(mood_dt),
            )
            for decision, mood_dt in zip(decisions, dts, strict=True)
        ]
