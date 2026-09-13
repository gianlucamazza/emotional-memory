"""Phase 6 measurement session owned by emotional-memory.

This is not ``python -m affective_fly study``. The host owns time and writes
``measure.jsonl``; the fly only supplies affect. Hypothesis MoodField taus
(300 / 60 / 180) and Policy / LaunchGate thresholds stay at library defaults.

Replay uses HostFrame timestamps so *mood time* spans tens of minutes even
when wall-clock runtime is seconds. A few 0.03 s live ticks is not a session.

Requires affective-fly (not on PyPI)::

    make install-fly
    python examples/fly_phase6_session.py --measure measure.jsonl
    python -m affective_fly calibrate measure.jsonl
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from affective_fly import HostFrame

from emotional_memory.integrations.fly import FlyAffectHost

_SCENES: tuple[dict[str, object], ...] = (
    {
        "context": "journal",
        "note_id": "exp-042",
        "sentiment": 0.95,
        "query": "successful replication of experiment 042",
    },
    {
        "context": "journal",
        "note_id": "exp-042",
        "sentiment": 0.55,
        "query": "notes look clean after first pass",
    },
    {
        "context": "review",
        "note_id": "exp-042",
        "sentiment": -0.75,
        "query": "reviewer pushback on methods",
    },
    {
        "context": "review",
        "note_id": "exp-042",
        "sentiment": -0.35,
        "query": "revision required",
        "outcome": -0.4,
    },
    {
        "context": "journal",
        "note_id": "exp-043",
        "sentiment": 0.15,
        "query": "new protocol drafted",
    },
    {
        "context": "journal",
        "note_id": "exp-043",
        "sentiment": 0.8,
        "query": "pilot run worked",
        "outcome": 0.5,
    },
    {
        "context": "review",
        "note_id": "exp-043",
        "sentiment": -0.2,
        "query": "minor clarification requested",
    },
    {
        "context": "journal",
        "note_id": "exp-044",
        "sentiment": -0.9,
        "query": "equipment failure overnight",
        "outcome": -0.7,
    },
)


def session_frames(
    *,
    ticks: int = 28,
    step_seconds: float = 45.0,
    start: datetime | None = None,
) -> list[HostFrame]:
    """HostFrames spanning ``(ticks-1) * step_seconds`` of mood time."""
    origin = start if start is not None else datetime(2026, 9, 13, 10, 0, tzinfo=UTC)
    frames: list[HostFrame] = []
    for i in range(ticks):
        scene = dict(_SCENES[i % len(_SCENES)])
        stamp = origin + timedelta(seconds=i * step_seconds)
        frames.append(
            HostFrame(
                timestamp=stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                visual_hash=f"phase6-{scene['note_id']}-{i}",
                context=scene,
            )
        )
    return frames


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measure",
        default="measure.jsonl",
        help="Host-owned Phase 6 JSONL (default: measure.jsonl)",
    )
    parser.add_argument("--ticks", type=int, default=28, help="Replay frames (default: 28)")
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=45.0,
        help="Mood-time seconds between HostFrame timestamps (default: 45)",
    )
    args = parser.parse_args(argv)

    frames = session_frames(ticks=args.ticks, step_seconds=args.step_seconds)
    host = FlyAffectHost(measure_path=args.measure)
    affects = host.replay(frames)
    span = sum(a.mood_dt for a in affects)

    print("FlyAffectHost Phase 6 session (host owns time; fly supplies affect)")
    print(f"  ticks={len(affects)}  mood_span_s={span:.1f}  measure={args.measure}")
    print(
        f"  taus={host.mood.tau_valence:.0f}/{host.mood.tau_arousal:.0f}/"
        f"{host.mood.tau_approach:.0f} (hypothesis; not retuned)"
    )
    print(f"  memories={len(host.memory)}  measure_bytes={Path(args.measure).stat().st_size}")
    print(f"{'dt':>8}  {'V':>7}  {'A':>7}  {'App':>7}")
    for affect in affects:
        print(
            f"{affect.mood_dt:8.1f}  {affect.valence:+7.3f}  "
            f"{affect.arousal:7.3f}  {affect.approach:+7.3f}"
        )


if __name__ == "__main__":
    main()
