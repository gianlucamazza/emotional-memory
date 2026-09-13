# affective-fly host

`emotional_memory` is the host. [affective-fly](https://github.com/gianlucamazza/affective-fly)
is the in-process affect source: a reduced *Drosophila* mushroom-body circuit that
returns valence, arousal, and approach/avoid.

This is the inverse of `python -m affective_fly run` / `study`, which construct
`EmotionalMemory` inside the fly package. Do not invert that. The fly's Policy
and LaunchGate stay at library defaults and are not the product. Phase 6
hypothesis mood taus (300 / 60 / 180 s) are not retuned here.

The contract is `HostFrame` v1.0 plus host-owned `mood_dt`, documented in
affective-fly's `docs/HOST_INTEGRATION.md` and `examples/host_owns_loop.py`.

## Installation

`affective-fly` is not on PyPI (and listing it as a locked extra would cycle:
fly already depends on this package).

```bash
make install-fly
# or:
uv pip install "affective-fly @ git+https://github.com/gianlucamazza/affective-fly.git"
```

## Ownership

1. This package constructs `EmotionalMemory` and the store.
2. This package owns time and computes `mood_dt` (wall-clock or HostFrame timestamps).
3. Each tick calls the fly (`AffectiveLoop` / `HostAdapter` / circuit) and reads
   valence / arousal / approach-avoid.

`FlyAffectHost` is the thin wrapper. Pass your own `store`, `embedder`, and
`memory` when you already have an engine; pass the **same** store and embedder
instances used to construct that memory (`AffectiveLoop` cannot read engine
internals).

## Live tick

```python
from affective_fly import HostFrame
from emotional_memory.integrations import FlyAffectHost

host = FlyAffectHost()
frame = HostFrame(
    visual_hash="note-1",  # chosen at creation; required for exact visual replay
    context={
        "context": "journal",
        "note_id": "n1",
        "query": "successful replication",
        "sentiment": 0.8,
    },
)
affect = host.tick_wall_clock(frame, now=10.0)
print(affect.valence, affect.arousal, affect.approach, affect.mood_dt)
```

`now` is seconds from `time.monotonic()` (or a test double). The first tick is
`mood_dt=0.0` — there is no previous sample to invent.

## Replay

Replay recorded `HostFrame` JSON with `mood_dt` from timestamps. Report an
outcome (`reward` / `outcome` / `pnl`) only when you have one — do not invent
rewards.

```python
from affective_fly import HostAdapter, HostFrame
from emotional_memory.integrations import FlyAffectHost

frames = [
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
host = FlyAffectHost()
affects = host.replay(frames)  # mood_dt: 0.0, then 16200.0
```

You can also load a JSONL journal with `HostAdapter.load_journal(...)`.

## Defaults that stay frozen

| Knob | Value | Notes |
| --- | --- | --- |
| Fly `MoodField` taus | 300 / 60 / 180 s | Phase 6 hypothesis; not lab 8 / 4 / 5 |
| `Policy.threshold_act` | 0.2 | Unchanged |
| `Policy.threshold_calm` | 0.0 | Unchanged |
| `LaunchGate.threshold_approach` | 0.2 | Unchanged |
| Default circuit | `LIFCircuit(n_kc=200, ...)` | Pass `MockFlyCircuit` only in unit tests |

This is not a UI or a token-launch product. Circuit `CoreAffect` is the fast
path; Scherer appraisal stays tag-only. Do not pass `appraisal=` to `encode()`.

## Run the example

```bash
make install-fly
python examples/fly_affect_source.py
uv run python -m pytest tests/test_fly_adapter.py
```
