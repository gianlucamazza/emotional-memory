"""emotional-memory is the host; affective-fly is the affect source.

This script is not a journal app or UI. It shows the ownership shape:

1. The host owns time and ``mood_dt`` (wall-clock or HostFrame timestamps).
2. The host constructs ``EmotionalMemory``.
3. Each tick the host calls the fly for valence, arousal, and approach/avoid.

Hypothesis MoodField taus (300 / 60 / 180) and Policy / LaunchGate
thresholds stay at library defaults. Do not retune them here.

Requires affective-fly (not on PyPI)::

    make install-fly
    python examples/fly_affect_source.py
"""

from __future__ import annotations

from affective_fly import HostFrame

from emotional_memory.integrations.fly import FlyAffectHost


def recorded_frames() -> list[HostFrame]:
    """HostFrame v1.0 samples from affective-fly ``docs/HOST_INTEGRATION.md``.

    ``visual_hash`` is chosen at creation so replay can regenerate the
    same visual. Outcomes are present only when the host has one.
    """
    return [
        HostFrame(
            timestamp="2026-09-12T10:00:00Z",
            visual_hash="exp-042-journal",
            context={
                "context": "journal",
                "note_id": "exp-042-replication",
                "query": "successful replication of experiment 042",
                "sentiment": 1.0,
            },
        ),
        HostFrame(
            timestamp="2026-09-12T14:30:00Z",
            visual_hash="exp-042-review",
            context={
                "context": "review",
                "note_id": "exp-042-replication",
                "query": "replication failed validation",
                "sentiment": -0.9,
                "outcome": -0.8,
            },
        ),
    ]


def main() -> None:
    host = FlyAffectHost()
    affects = host.replay(recorded_frames())

    print("host owns EmotionalMemory and mood_dt; fly returns affect")
    print(f"{'dt':>8}  {'V':>7}  {'A':>7}  {'App':>7}  memories")
    for affect in affects:
        print(
            f"{affect.mood_dt:8.1f}  {affect.valence:+7.3f}  "
            f"{affect.arousal:7.3f}  {affect.approach:+7.3f}  "
            f"{len(host.memory)}"
        )
    print(
        f"taus={host.mood.tau_valence:.0f}/{host.mood.tau_arousal:.0f}/"
        f"{host.mood.tau_approach:.0f} (hypothesis; not retuned)"
    )


if __name__ == "__main__":
    main()
