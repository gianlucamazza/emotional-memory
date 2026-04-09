import pytest

from emotional_memory.appraisal import (
    AppraisalEngine,
    AppraisalVector,
    StaticAppraisalEngine,
    consolidation_strength,
)


class TestAppraisalVector:
    def test_neutral(self):
        v = AppraisalVector.neutral()
        assert v.novelty == 0.0
        assert v.goal_relevance == 0.0
        assert v.coping_potential == 0.5
        assert v.norm_congruence == 0.0
        assert v.self_relevance == 0.0

    def test_clamp_signed(self):
        v = AppraisalVector(
            novelty=2.0,
            goal_relevance=-3.0,
            coping_potential=0.5,
            norm_congruence=1.5,
            self_relevance=0.5,
        )
        assert v.novelty == 1.0
        assert v.goal_relevance == -1.0
        assert v.norm_congruence == 1.0

    def test_clamp_unit(self):
        v = AppraisalVector(
            novelty=0.0,
            goal_relevance=0.0,
            coping_potential=-0.5,
            norm_congruence=0.0,
            self_relevance=2.0,
        )
        assert v.coping_potential == 0.0
        assert v.self_relevance == 1.0

    def test_frozen(self):
        v = AppraisalVector.neutral()
        with pytest.raises(Exception):
            v.novelty = 0.5  # type: ignore[misc]

    def test_to_core_affect_produces_valid_affect(self):
        v = AppraisalVector(
            novelty=0.5,
            goal_relevance=0.8,
            coping_potential=0.7,
            norm_congruence=0.3,
            self_relevance=0.6,
        )
        ca = v.to_core_affect()
        assert -1.0 <= ca.valence <= 1.0
        assert 0.0 <= ca.arousal <= 1.0

    def test_positive_appraisal_yields_positive_valence(self):
        positive = AppraisalVector(
            novelty=0.0,
            goal_relevance=1.0,
            coping_potential=1.0,
            norm_congruence=1.0,
            self_relevance=0.5,
        )
        ca = positive.to_core_affect()
        assert ca.valence > 0.0

    def test_negative_appraisal_yields_negative_valence(self):
        negative = AppraisalVector(
            novelty=0.0,
            goal_relevance=-1.0,
            coping_potential=0.0,
            norm_congruence=-1.0,
            self_relevance=0.5,
        )
        ca = negative.to_core_affect()
        assert ca.valence < 0.0

    def test_high_novelty_raises_arousal(self):
        high_novelty = AppraisalVector(
            novelty=1.0,
            goal_relevance=0.0,
            coping_potential=0.5,
            norm_congruence=0.0,
            self_relevance=0.0,
        )
        low_novelty = AppraisalVector(
            novelty=0.0,
            goal_relevance=0.0,
            coping_potential=0.5,
            norm_congruence=0.0,
            self_relevance=0.0,
        )
        assert high_novelty.to_core_affect().arousal > low_novelty.to_core_affect().arousal

    def test_neutral_appraisal_neutral_affect(self):
        ca = AppraisalVector.neutral().to_core_affect()
        # coping=0.5 → coping_signed=0, all others=0 → valence=0, arousal=0.15 (from coping)
        assert ca.valence == 0.0
        assert ca.arousal >= 0.0


class TestConsolidationStrength:
    def test_peaks_at_effective_07(self):
        # effective_arousal = 0.7*arousal + 0.3*stimmung_arousal
        # With stimmung_arousal=0: peak is at arousal=1.0 (effective=0.7)
        # With stimmung_arousal=1: peak is at arousal=0.571 (effective≈0.7)
        # Test with equal stimmung so effective_arousal = raw arousal
        s_peak = consolidation_strength(0.7, 0.7)  # effective = 0.7 → peak
        s_low = consolidation_strength(0.0, 0.0)  # effective = 0.0 → low
        s_high = consolidation_strength(1.0, 1.0)  # effective = 1.0 → low
        assert s_peak > s_low
        assert s_peak > s_high
        assert s_peak == pytest.approx(1.0)

    def test_returns_unit_range(self):
        for arousal in [0.0, 0.25, 0.5, 0.7, 0.85, 1.0]:
            v = consolidation_strength(arousal, 0.0)
            assert 0.0 <= v <= 1.0

    def test_clamped_inputs(self):
        assert consolidation_strength(-1.0, -1.0) >= 0.0
        assert consolidation_strength(2.0, 2.0) <= 1.0

    def test_stimmung_arousal_blends_in(self):
        # With stimmung_arousal=1.0, effective arousal shifts higher
        # relative to stimmung_arousal=0.0
        v1 = consolidation_strength(0.0, 0.0)
        v2 = consolidation_strength(0.0, 1.0)
        # effective_arousal shifts from 0.0 to 0.3 → closer to peak → stronger
        assert v2 > v1


class TestStaticAppraisalEngine:
    def test_returns_fixed_vector(self):
        vec = AppraisalVector(
            novelty=0.5,
            goal_relevance=0.3,
            coping_potential=0.8,
            norm_congruence=0.1,
            self_relevance=0.4,
        )
        engine = StaticAppraisalEngine(vec)
        result = engine.appraise("anything")
        assert result == vec

    def test_default_is_neutral(self):
        engine = StaticAppraisalEngine()
        result = engine.appraise("test")
        assert result == AppraisalVector.neutral()

    def test_context_ignored(self):
        engine = StaticAppraisalEngine()
        r1 = engine.appraise("text", context={"key": "value"})
        r2 = engine.appraise("text", context=None)
        assert r1 == r2

    def test_satisfies_protocol(self):
        engine = StaticAppraisalEngine()
        assert isinstance(engine, AppraisalEngine)
