"""Unit tests for the discrete emotion categorization module (Plutchik, 1980)."""

import pytest
from pydantic import ValidationError

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.categorize import (
    EmotionLabel,
    categorize_affect,
    label_tag,
)
from emotional_memory.models import make_emotional_tag
from emotional_memory.mood import MoodField

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tag(valence: float = 0.0, arousal: float = 0.5):
    return make_emotional_tag(
        core_affect=CoreAffect(valence=valence, arousal=arousal),
        momentum=AffectiveMomentum.zero(),
        mood=MoodField.neutral(),
        consolidation_strength=0.5,
    )


# ---------------------------------------------------------------------------
# Primary emotion mapping
# ---------------------------------------------------------------------------


class TestPrimaryEmotionMapping:
    def test_joy_positive_valence_neutral_arousal(self):
        label = categorize_affect(CoreAffect(valence=0.8, arousal=0.6))
        assert label.primary == "joy"

    def test_anticipation_positive_valence_high_arousal(self):
        label = categorize_affect(CoreAffect(valence=0.5, arousal=0.9))
        assert label.primary == "anticipation"

    def test_surprise_neutral_valence_high_arousal(self):
        label = categorize_affect(CoreAffect(valence=0.0, arousal=1.0))
        assert label.primary == "surprise"

    def test_fear_negative_valence_high_arousal_low_dominance(self):
        label = categorize_affect(CoreAffect(valence=-0.5, arousal=0.9), dominance=0.2)
        assert label.primary == "fear"

    def test_anger_negative_valence_high_arousal_high_dominance(self):
        label = categorize_affect(CoreAffect(valence=-0.5, arousal=0.9), dominance=0.8)
        assert label.primary == "anger"

    def test_disgust_very_negative_valence_neutral_arousal(self):
        label = categorize_affect(CoreAffect(valence=-0.9, arousal=0.5))
        assert label.primary == "disgust"

    def test_sadness_negative_valence_low_arousal(self):
        label = categorize_affect(CoreAffect(valence=-0.7, arousal=0.2))
        assert label.primary == "sadness"

    def test_trust_positive_valence_low_arousal(self):
        label = categorize_affect(CoreAffect(valence=0.7, arousal=0.2))
        assert label.primary == "trust"

    def test_fear_without_dominance_defaults_to_fear(self):
        """No dominance provided → default to fear in the ambiguous sector."""
        label = categorize_affect(CoreAffect(valence=-0.6, arousal=0.85))
        assert label.primary == "fear"

    def test_dominance_threshold_boundary(self):
        ca = CoreAffect(valence=-0.5, arousal=0.9)
        at_threshold = categorize_affect(ca, dominance=0.5)
        below_threshold = categorize_affect(ca, dominance=0.49)
        assert at_threshold.primary == "anger"
        assert below_threshold.primary == "fear"


# ---------------------------------------------------------------------------
# Intensity tiers
# ---------------------------------------------------------------------------


class TestIntensityTiers:
    def test_low_intensity_near_origin(self):
        # Very small r < _INTENSITY_LOW_THRESHOLD
        ca = CoreAffect(valence=0.1, arousal=0.5)  # centered arousal ≈ 0
        label = categorize_affect(ca)
        assert label.intensity == "low"

    def test_high_intensity_far_from_origin(self):
        ca = CoreAffect(valence=1.0, arousal=1.0)
        label = categorize_affect(ca)
        assert label.intensity == "high"

    def test_moderate_intensity_mid_range(self):
        ca = CoreAffect(valence=0.5, arousal=0.7)  # r ≈ 0.53 → moderate
        label = categorize_affect(ca)
        assert label.intensity == "moderate"

    def test_joy_intensity_tier_names(self):
        serenity = categorize_affect(CoreAffect(valence=0.1, arousal=0.5))
        joy = categorize_affect(CoreAffect(valence=0.5, arousal=0.6))
        # (1.0, 0.55) → a_scaled=0.1 → angle ≈ 5.7° → joy sector, r ≈ 1.00 → high → ecstasy
        ecstasy = categorize_affect(CoreAffect(valence=1.0, arousal=0.55))
        assert serenity.name == "serenity"
        assert joy.name == "joy"
        assert ecstasy.name == "ecstasy"

    def test_fear_intensity_tier_names(self):
        apprehension = categorize_affect(CoreAffect(valence=-0.1, arousal=0.6), dominance=0.2)
        terror = categorize_affect(CoreAffect(valence=-0.8, arousal=1.0), dominance=0.2)
        assert apprehension.name == "apprehension"
        assert terror.name == "terror"


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_confidence_in_range(self):
        for v, a in [(0.8, 0.6), (-0.5, 0.9), (-0.7, 0.2), (0.0, 1.0)]:
            label = categorize_affect(CoreAffect(valence=v, arousal=a))
            assert 0.0 <= label.confidence <= 1.0

    def test_sector_centre_has_high_confidence(self):
        # Exactly at (1.0, 0.5) → angle 0° → centre of joy sector
        label = categorize_affect(CoreAffect(valence=1.0, arousal=0.5))
        assert label.confidence == pytest.approx(1.0, abs=1e-9)

    def test_sector_edge_has_lower_confidence(self):
        # A point right at the boundary between two sectors has low confidence
        centre_label = categorize_affect(CoreAffect(valence=1.0, arousal=0.5))
        # Move toward sector boundary (45° angle from sector 0 centre)
        edge_ca = CoreAffect(valence=0.7, arousal=0.8)
        edge_label = categorize_affect(edge_ca)
        assert edge_label.confidence < centre_label.confidence


# ---------------------------------------------------------------------------
# EmotionLabel model
# ---------------------------------------------------------------------------


class TestEmotionLabel:
    def test_is_frozen(self):
        label = categorize_affect(CoreAffect(valence=0.5, arousal=0.7))
        with pytest.raises(ValidationError):
            label.primary = "sadness"  # type: ignore[misc]

    def test_has_all_fields(self):
        label = categorize_affect(CoreAffect(valence=0.5, arousal=0.7))
        assert isinstance(label, EmotionLabel)
        assert label.primary in (
            "joy",
            "trust",
            "fear",
            "surprise",
            "sadness",
            "disgust",
            "anger",
            "anticipation",
        )
        assert label.intensity in ("low", "moderate", "high")
        assert isinstance(label.name, str)
        assert len(label.name) > 0


# ---------------------------------------------------------------------------
# label_tag helper
# ---------------------------------------------------------------------------


class TestLabelTag:
    def test_label_tag_sets_emotion_label(self):
        # (0.8, 0.5) → a_scaled=0.0 → angle=0° → joy sector
        tag = _make_tag(valence=0.8, arousal=0.5)
        assert tag.emotion_label is None
        labelled = label_tag(tag)
        assert labelled.emotion_label is not None
        assert labelled.emotion_label.primary == "joy"

    def test_label_tag_preserves_other_fields(self):
        tag = _make_tag(valence=0.8, arousal=0.7)
        labelled = label_tag(tag)
        assert labelled.core_affect == tag.core_affect
        assert labelled.momentum == tag.momentum
        assert labelled.consolidation_strength == tag.consolidation_strength

    def test_label_tag_uses_mood_dominance_when_no_dominance_arg(self):
        """Without explicit dominance, MoodField.dominance is used."""
        from datetime import UTC, datetime

        from emotional_memory.mood import MoodField

        high_dom_mood = MoodField(
            valence=-0.5, arousal=0.8, dominance=0.9, inertia=0.5, timestamp=datetime.now(tz=UTC)
        )
        tag = make_emotional_tag(
            core_affect=CoreAffect(valence=-0.5, arousal=0.9),
            momentum=AffectiveMomentum.zero(),
            mood=high_dom_mood,
            consolidation_strength=0.5,
        )
        labelled = label_tag(tag)
        assert labelled.emotion_label is not None
        assert labelled.emotion_label.primary == "anger"

    def test_label_tag_explicit_dominance_overrides_mood(self):
        """Explicit dominance arg overrides mood.dominance."""
        from datetime import UTC, datetime

        from emotional_memory.mood import MoodField

        high_dom_mood = MoodField(
            valence=-0.5, arousal=0.8, dominance=0.9, inertia=0.5, timestamp=datetime.now(tz=UTC)
        )
        tag = make_emotional_tag(
            core_affect=CoreAffect(valence=-0.5, arousal=0.9),
            momentum=AffectiveMomentum.zero(),
            mood=high_dom_mood,
            consolidation_strength=0.5,
        )
        labelled = label_tag(tag, dominance=0.1)  # override to low dominance → fear
        assert labelled.emotion_label is not None
        assert labelled.emotion_label.primary == "fear"

    def test_label_tag_returns_new_instance(self):
        tag = _make_tag(valence=0.8, arousal=0.7)
        labelled = label_tag(tag)
        assert labelled is not tag
        assert tag.emotion_label is None  # original unchanged
