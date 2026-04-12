"""Discrete emotion categorization via Plutchik's wheel.

Maps continuous (valence, arousal) coordinates from the Russell circumplex
to Plutchik's 8 primary emotions with intensity tiers (Plutchik, 1980).

The mapping is angular: each of the 8 primaries occupies a 45° sector in
the centered valence-arousal plane (arousal shifted to [-0.5, 0.5]).
Radial distance from the origin determines intensity:
  low      r < 0.30  (serenity, acceptance, apprehension, …)
  moderate r ∈ [0.30, 0.70)  (joy, trust, fear, …)
  high     r ≥ 0.70  (ecstasy, admiration, terror, …)

When ``dominance`` is available (e.g. from MoodField), it disambiguates
the fear/anger region (negative valence + high arousal):
  dominance ≥ 0.5 → anger  (high perceived control)
  dominance < 0.5 → fear   (low perceived control)

References
----------
Plutchik, R. (1980). Emotion: A psychoevolutionary synthesis. Harper & Row.
Russell, J. A. (1980). A circumplex model of affect.
  Journal of Personality and Social Psychology, 39(6), 1161-1178.
Mehrabian, A., & Russell, J. A. (1974). An approach to environmental
  psychology. MIT Press.  (PAD dominance dimension)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from emotional_memory.affect import CoreAffect
    from emotional_memory.models import EmotionalTag

PrimaryEmotion = Literal[
    "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"
]
Intensity = Literal["low", "moderate", "high"]

# Intensity thresholds based on radial distance in centered affect space
# Max r ≈ sqrt(1² + 0.5²) ≈ 1.12; thresholds are relative to that range.
_INTENSITY_LOW_THRESHOLD: float = 0.30
_INTENSITY_HIGH_THRESHOLD: float = 0.70

# Plutchik's dyadic intensity tier names for each primary
_INTENSITY_NAMES: dict[PrimaryEmotion, dict[Intensity, str]] = {
    "joy": {"low": "serenity", "moderate": "joy", "high": "ecstasy"},
    "trust": {"low": "acceptance", "moderate": "trust", "high": "admiration"},
    "fear": {"low": "apprehension", "moderate": "fear", "high": "terror"},
    "surprise": {"low": "distraction", "moderate": "surprise", "high": "amazement"},
    "sadness": {"low": "pensiveness", "moderate": "sadness", "high": "grief"},
    "disgust": {"low": "boredom", "moderate": "disgust", "high": "loathing"},
    "anger": {"low": "annoyance", "moderate": "anger", "high": "rage"},
    "anticipation": {"low": "interest", "moderate": "anticipation", "high": "vigilance"},
}

# 8 sectors of 45° each, counterclockwise from east (0°).
# Sector i occupies [(i*45 - 22.5)deg, (i*45 + 22.5)deg) in normalised [0deg, 360deg).
# Positions are chosen to match Russell's (1980) empirical placements.
#
#  sector 0  [337.5deg,  22.5deg) -> joy           (+ valence, neutral-moderate arousal)
#  sector 1  [ 22.5deg,  67.5deg) -> anticipation  (+ valence, + arousal / eagerness)
#  sector 2  [ 67.5deg, 112.5deg) -> surprise      (neutral valence, high arousal)
#  sector 3  [112.5deg, 157.5deg) -> fear / anger  (- valence, high arousal) <- dominance
#  sector 4  [157.5deg, 202.5deg) -> disgust       (- valence, neutral arousal)
#  sector 5  [202.5deg, 247.5deg) -> sadness       (- valence, - arousal)
#  sector 6  [247.5deg, 292.5deg) -> sadness       (neutral valence, very - arousal) [boredom]
#  sector 7  [292.5deg, 337.5deg) -> trust         (+ valence, - arousal / calm)
_SECTOR_EMOTION: list[PrimaryEmotion] = [
    "joy",  # sector 0 — centre 0°   (happiness, contentment)
    "anticipation",  # sector 1 — centre 45°  (eagerness, excitement)
    "surprise",  # sector 2 — centre 90°  (arousal, shock)
    "fear",  # sector 3 — centre 135° (overridden to "anger" when dominance >= 0.5)
    "disgust",  # sector 4 — centre 180° (revulsion, loathing)
    "sadness",  # sector 5 — centre 225° (grief, depression)
    "sadness",  # sector 6 — centre 270° (boredom, lethargy)
    "trust",  # sector 7 — centre 315° (serenity, acceptance)
]
_FEAR_ANGER_SECTOR: int = 3
_DOMINANCE_THRESHOLD: float = 0.5


class EmotionLabel(BaseModel):
    """Discrete emotion label derived from continuous affect coordinates.

    primary    — one of Plutchik's 8 primary emotions
    intensity  — low / moderate / high (Plutchik intensity tiers)
    name       — canonical dyadic tier name (e.g. "serenity", "joy", "ecstasy")
    confidence — how close the point is to the sector centre [0, 1]
    """

    model_config = {"frozen": True}

    primary: PrimaryEmotion
    intensity: Intensity
    name: str
    confidence: float  # [0, 1]


def categorize_affect(
    core_affect: CoreAffect,
    dominance: float | None = None,
) -> EmotionLabel:
    """Map a CoreAffect point to a discrete Plutchik emotion label.

    Args:
        core_affect: The continuous valence-arousal point.
        dominance:   Optional PAD dominance value [0, 1].  When provided,
                     disambiguates fear vs anger in the high-arousal /
                     negative-valence region.

    Returns:
        An :class:`EmotionLabel` with the primary emotion, intensity tier,
        canonical tier name, and sector-centre confidence.

    Notes:
        The arousal axis is centred at 0.5 before computing angles so that
        the neutral point aligns with the origin of the circumplex.
    """
    v = core_affect.valence
    a_centered = core_affect.arousal - 0.5  # re-centre to [-0.5, 0.5]

    # Angular position in degrees, normalised to [0°, 360°)
    angle_deg = math.degrees(math.atan2(a_centered, v))
    angle_norm = angle_deg % 360.0  # [0, 360)

    # Determine sector index (0-7)
    sector_idx = int((angle_norm + 22.5) / 45.0) % 8

    # Resolve primary emotion
    primary: PrimaryEmotion = _SECTOR_EMOTION[sector_idx]
    if sector_idx == _FEAR_ANGER_SECTOR and dominance is not None:
        primary = "anger" if dominance >= _DOMINANCE_THRESHOLD else "fear"

    # Radial distance from origin → intensity
    r = math.sqrt(v**2 + a_centered**2)
    if r < _INTENSITY_LOW_THRESHOLD:
        intensity: Intensity = "low"
    elif r < _INTENSITY_HIGH_THRESHOLD:
        intensity = "moderate"
    else:
        intensity = "high"

    # Confidence: 1.0 at sector centre, 0.0 at sector edge (±22.5°)
    sector_center_deg = (sector_idx * 45.0) % 360.0
    # Angular distance, wrapped to [0°, 180°]
    delta = abs((angle_norm - sector_center_deg + 180.0) % 360.0 - 180.0)
    confidence = max(0.0, 1.0 - delta / 22.5)

    return EmotionLabel(
        primary=primary,
        intensity=intensity,
        name=_INTENSITY_NAMES[primary][intensity],
        confidence=confidence,
    )


def label_tag(
    tag: EmotionalTag,
    dominance: float | None = None,
) -> EmotionalTag:
    """Return a copy of *tag* with the ``emotion_label`` field populated.

    Args:
        tag:       The emotional tag to label.
        dominance: Optional dominance value for fear/anger disambiguation.
                   If ``None`` and the tag has a ``mood_snapshot``, the
                   mood's dominance is used automatically.

    Returns:
        A new frozen :class:`~emotional_memory.models.EmotionalTag` with
        ``emotion_label`` set.
    """
    # Auto-extract dominance from mood snapshot when not provided
    if dominance is None and tag.mood_snapshot is not None:
        dominance = tag.mood_snapshot.dominance
    label = categorize_affect(tag.core_affect, dominance=dominance)
    return tag.model_copy(update={"emotion_label": label})
