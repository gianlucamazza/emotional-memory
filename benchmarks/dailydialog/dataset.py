"""DailyDialog benchmark data structures and persona loader.

Loads pre-built synthetic-persona JSON produced by ``persona_builder.py``.
The raw DailyDialog corpus is NOT redistributed; the pre-built personas JSON
is committed to the repo after one-time generation via ``persona_builder.py``.

Reference: Li et al. 2017, IJCNLP — "DailyDialog: A Manually Labelled Multi-turn
Dialogue Dataset"  (CC BY-NC-SA 4.0)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ekman label mappings
# ---------------------------------------------------------------------------

EKMAN_LABEL_NAMES: dict[int, str] = {
    0: "no_emotion",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise",
}

# Canonical Ekman → PAD mapping (valence, arousal, dominance).
# Integer codes match the raw DailyDialog emotion files (Li et al. 2017).
# Same valence/arousal conventions as appraisal_keyword.py in the main library.
EKMAN_PAD_MAP: dict[int, tuple[float, float, float]] = {
    0: (0.00, 0.20, 0.50),  # no_emotion — neutral baseline
    1: (-0.50, 0.75, 0.60),  # anger
    2: (-0.60, 0.40, 0.30),  # disgust
    3: (-0.55, 0.75, -0.60),  # fear
    4: (0.70, 0.50, 0.60),  # happiness
    5: (-0.60, 0.20, -0.40),  # sadness
    6: (0.05, 0.80, -0.20),  # surprise — valence ambiguous, high arousal
}

# ---------------------------------------------------------------------------
# Raw dialog type (used by persona_builder, not the runner)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DailyTurn:
    text: str
    emotion: int  # 0-6
    act: int  # 1-4


@dataclass(frozen=True)
class RawDialog:
    dialog_id: str
    turns: tuple[DailyTurn, ...]

    @property
    def emotion_density(self) -> float:
        if not self.turns:
            return 0.0
        return sum(1 for t in self.turns if t.emotion != 0) / len(self.turns)

    @property
    def dominant_emotion(self) -> int:
        """Most frequent non-zero emotion; 0 if all turns are no_emotion."""
        counts: dict[int, int] = {}
        for t in self.turns:
            counts[t.emotion] = counts.get(t.emotion, 0) + 1
        # Prefer non-zero emotions when tie-breaking
        best = max(counts.items(), key=lambda x: (x[1], x[0] != 0))
        return best[0]


# ---------------------------------------------------------------------------
# Persona data structures (JSON-serialisable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonaSession:
    session_id: str
    dialog_id: str
    turns: tuple[DailyTurn, ...]
    dominant_emotion: int
    dominant_emotion_name: str
    valence: float
    arousal: float
    dominance: float
    emotion_density: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "dialog_id": self.dialog_id,
            "turns": [{"text": t.text, "emotion": t.emotion, "act": t.act} for t in self.turns],
            "dominant_emotion": self.dominant_emotion,
            "dominant_emotion_name": self.dominant_emotion_name,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "emotion_density": self.emotion_density,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PersonaSession:
        return cls(
            session_id=str(d["session_id"]),
            dialog_id=str(d["dialog_id"]),
            turns=tuple(
                DailyTurn(
                    text=str(t["text"]),
                    emotion=int(t["emotion"]),
                    act=int(t["act"]),
                )
                for t in d["turns"]
            ),
            dominant_emotion=int(d["dominant_emotion"]),
            dominant_emotion_name=str(d["dominant_emotion_name"]),
            valence=float(d["valence"]),
            arousal=float(d["arousal"]),
            dominance=float(d["dominance"]),
            emotion_density=float(d["emotion_density"]),
        )


@dataclass(frozen=True)
class PersonaQuery:
    query_id: str
    query_type: str
    text: str
    target_session_id: str
    distractor_session_ids: tuple[str, ...]
    top_k: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_type": self.query_type,
            "text": self.text,
            "target_session_id": self.target_session_id,
            "distractor_session_ids": list(self.distractor_session_ids),
            "top_k": self.top_k,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PersonaQuery:
        return cls(
            query_id=str(d["query_id"]),
            query_type=str(d["query_type"]),
            text=str(d["text"]),
            target_session_id=str(d["target_session_id"]),
            distractor_session_ids=tuple(str(s) for s in d.get("distractor_session_ids", [])),
            top_k=int(d.get("top_k", 2)),
        )


@dataclass
class Persona:
    persona_id: str
    sessions: list[PersonaSession] = field(default_factory=list)
    queries: list[PersonaQuery] = field(default_factory=list)

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)

    @property
    def n_turns(self) -> int:
        return sum(len(s.turns) for s in self.sessions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "sessions": [s.to_dict() for s in self.sessions],
            "queries": [q.to_dict() for q in self.queries],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Persona:
        p = cls(persona_id=str(d["persona_id"]))
        p.sessions = [PersonaSession.from_dict(s) for s in d.get("sessions", [])]
        p.queries = [PersonaQuery.from_dict(q) for q in d.get("queries", [])]
        return p


@dataclass
class DailyDialogPersonaDataset:
    version: str
    n_personas: int
    seed: int
    emotion_density_threshold: float
    personas: list[Persona] = field(default_factory=list)

    @property
    def total_queries(self) -> int:
        return sum(len(p.queries) for p in self.personas)

    @property
    def total_turns(self) -> int:
        return sum(p.n_turns for p in self.personas)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "n_personas": self.n_personas,
            "seed": self.seed,
            "emotion_density_threshold": self.emotion_density_threshold,
            "personas": [p.to_dict() for p in self.personas],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DailyDialogPersonaDataset:
        ds = cls(
            version=str(d.get("version", "1")),
            n_personas=int(d.get("n_personas", 0)),
            seed=int(d.get("seed", 0)),
            emotion_density_threshold=float(d.get("emotion_density_threshold", 0.30)),
        )
        ds.personas = [Persona.from_dict(p) for p in d.get("personas", [])]
        return ds


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

DEFAULT_PERSONA_FILE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "datasets"
    / "dailydialog_personas_v1.json"
)


def load_personas(path: Path | None = None) -> DailyDialogPersonaDataset:
    """Load pre-built persona dataset from *path*.

    Parameters
    ----------
    path:
        Path to ``dailydialog_personas_v1.json``. Defaults to the canonical
        location in ``benchmarks/datasets/``.
    """
    dest = path if path is not None else DEFAULT_PERSONA_FILE
    if not dest.exists():
        raise FileNotFoundError(
            f"Persona file not found: {dest}\n"
            "Build it first with:\n"
            "  uv run python -m benchmarks.dailydialog.persona_builder --n 120 --seed 0"
        )
    raw: dict[str, Any] = json.loads(dest.read_text(encoding="utf-8"))
    return DailyDialogPersonaDataset.from_dict(raw)
