import math
from datetime import UTC, datetime

import pytest
from conftest import make_test_memory
from pydantic import ValidationError

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.decay import DecayConfig
from emotional_memory.models import make_emotional_tag
from emotional_memory.mood import MoodField
from emotional_memory.retrieval import (
    RetrievalConfig,
    adaptive_weights,
    affective_prediction_error,
    build_retrieval_plan,
    reconsolidate,
    retrieval_score,
)


def _neutral_mood():
    return MoodField.neutral()


def _mood(valence: float = 0.0, arousal: float = 0.0):
    return MoodField(
        valence=valence,
        arousal=arousal,
        dominance=0.5,
        inertia=0.5,
        timestamp=datetime.now(tz=UTC),
    )


class TestAdaptiveWeights:
    def test_sums_to_one_neutral(self):
        w = adaptive_weights(_neutral_mood(), [0.35, 0.25, 0.15, 0.10, 0.10, 0.05])
        assert math.isclose(w.sum(), 1.0, rel_tol=1e-9)

    def test_sums_to_one_negative(self):
        w = adaptive_weights(_mood(valence=-0.8), [0.35, 0.25, 0.15, 0.10, 0.10, 0.05])
        assert math.isclose(w.sum(), 1.0, rel_tol=1e-9)

    def test_sums_to_one_high_arousal(self):
        w = adaptive_weights(_mood(arousal=0.9), [0.35, 0.25, 0.15, 0.10, 0.10, 0.05])
        assert math.isclose(w.sum(), 1.0, rel_tol=1e-9)

    def test_sums_to_one_calm(self):
        w = adaptive_weights(_mood(valence=0.1, arousal=0.1), [0.35, 0.25, 0.15, 0.10, 0.10, 0.05])
        assert math.isclose(w.sum(), 1.0, rel_tol=1e-9)

    def test_negative_mood_increases_emotional_weight(self):
        base = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        w_neg = adaptive_weights(_mood(valence=-0.8), base)
        w_neu = adaptive_weights(_neutral_mood(), base)
        # w[1] = mood_congruence, w[2] = affect_proximity
        assert w_neg[1] > w_neu[1]

    def test_negative_mood_reduces_semantic_weight(self):
        base = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        w_neg = adaptive_weights(_mood(valence=-0.8), base)
        w_neu = adaptive_weights(_neutral_mood(), base)
        assert w_neg[0] < w_neu[0]

    def test_high_arousal_increases_momentum_weight(self):
        base = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        w_high = adaptive_weights(_mood(arousal=0.9), base)
        w_neu = adaptive_weights(_neutral_mood(), base)
        assert w_high[3] > w_neu[3]

    def test_calm_increases_semantic_weight(self):
        # Compare calm (qualifies) vs non-calm (does not qualify)
        base = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        w_calm = adaptive_weights(_mood(valence=0.1, arousal=0.1), base)
        w_active = adaptive_weights(_mood(valence=0.5, arousal=0.6), base)
        assert w_calm[0] > w_active[0]

    def test_no_negative_weights(self):
        base = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        for valence in [-1.0, -0.5, 0.0, 0.5]:
            for arousal in [0.0, 0.5, 1.0]:
                w = adaptive_weights(_mood(valence, arousal), base)
                assert all(wi >= 0.0 for wi in w)


class TestAdaptiveWeightsConfig:
    """Tests for smooth sigmoid/Gaussian weight modulation."""

    def test_custom_config_accepted(self):
        from emotional_memory.retrieval import AdaptiveWeightsConfig

        cfg = AdaptiveWeightsConfig(negative_mood_strength=0.20)
        base = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        w = adaptive_weights(_mood(valence=-0.9), base, config=cfg)
        assert math.isclose(w.sum(), 1.0, rel_tol=1e-9)

    def test_high_sharpness_approximates_hard_threshold(self):
        """Sharpness=50 should behave like the old if-threshold at the boundary."""
        from emotional_memory.retrieval import AdaptiveWeightsConfig

        cfg_sharp = AdaptiveWeightsConfig(negative_mood_sharpness=50.0)
        base = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        # Well below threshold — high gate activation
        w_far = adaptive_weights(_mood(valence=-0.9), base, config=cfg_sharp)
        # Well above threshold — near-zero gate activation
        w_near = adaptive_weights(_mood(valence=0.0), base, config=cfg_sharp)
        assert w_far[1] > w_near[1]

    def test_smooth_monotone_negative_mood(self):
        """Emotional weight should rise monotonically as valence decreases."""
        base = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        valences = [0.0, -0.2, -0.5, -0.8, -1.0]
        weights = [adaptive_weights(_mood(valence=v), base)[1] for v in valences]
        assert all(weights[i] <= weights[i + 1] for i in range(len(weights) - 1))

    def test_smooth_monotone_high_arousal(self):
        """Momentum weight should rise monotonically as arousal increases."""
        base = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        arousals = [0.0, 0.3, 0.6, 0.8, 1.0]
        weights = [adaptive_weights(_mood(arousal=a), base)[3] for a in arousals]
        assert all(weights[i] <= weights[i + 1] for i in range(len(weights) - 1))

    def test_continuity_at_old_threshold_valence(self):
        """No discontinuous jump at old hard threshold valence=-0.5."""
        base = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        w_just_above = adaptive_weights(_mood(valence=-0.49), base)
        w_just_below = adaptive_weights(_mood(valence=-0.51), base)
        # Difference should be small (continuous transition), not a step
        assert abs(float(w_just_below[1]) - float(w_just_above[1])) < 0.05

    def test_continuity_at_old_threshold_arousal(self):
        """No discontinuous jump at old hard threshold arousal=0.7."""
        base = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        w_just_below = adaptive_weights(_mood(arousal=0.69), base)
        w_just_above = adaptive_weights(_mood(arousal=0.71), base)
        assert abs(float(w_just_above[3]) - float(w_just_below[3])) < 0.05


class TestSmoothGate:
    def test_at_center_returns_half(self):
        from emotional_memory.retrieval import _smooth_gate

        assert _smooth_gate(0.0, 0.0, 5.0) == pytest.approx(0.5, abs=1e-6)

    def test_positive_sharpness_activates_above(self):
        from emotional_memory.retrieval import _smooth_gate

        assert _smooth_gate(1.0, 0.0, 5.0) > 0.5
        assert _smooth_gate(-1.0, 0.0, 5.0) < 0.5

    def test_negative_sharpness_activates_below(self):
        from emotional_memory.retrieval import _smooth_gate

        assert _smooth_gate(-1.0, 0.0, -5.0) > 0.5
        assert _smooth_gate(1.0, 0.0, -5.0) < 0.5

    def test_output_in_unit_interval(self):
        from emotional_memory.retrieval import _smooth_gate

        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            v = _smooth_gate(x, 0.0, 5.0)
            assert 0.0 <= v <= 1.0


class TestAffinePredictionError:
    def test_zero_for_identical(self):
        ca = CoreAffect(valence=0.3, arousal=0.5)
        assert affective_prediction_error(ca, ca) == 0.0

    def test_symmetric(self):
        a = CoreAffect(valence=0.0, arousal=0.0)
        b = CoreAffect(valence=1.0, arousal=1.0)
        assert math.isclose(
            affective_prediction_error(a, b),
            affective_prediction_error(b, a),
        )

    def test_positive(self):
        a = CoreAffect(valence=-0.5, arousal=0.2)
        b = CoreAffect(valence=0.5, arousal=0.8)
        assert affective_prediction_error(a, b) > 0.0


class TestReconsolidate:
    def test_modifies_only_core_affect_and_count(self):
        tag = make_emotional_tag(
            core_affect=CoreAffect(valence=-0.5, arousal=0.3),
            momentum=AffectiveMomentum.zero(),
            mood=_neutral_mood(),
            consolidation_strength=0.7,
        )
        new_affect = CoreAffect(valence=0.5, arousal=0.7)
        updated = reconsolidate(tag, new_affect, ape=0.5, learning_rate=0.2)

        # core_affect changed
        assert updated.core_affect != tag.core_affect
        # reconsolidation_count incremented
        assert updated.reconsolidation_count == tag.reconsolidation_count + 1
        # everything else unchanged
        assert updated.momentum == tag.momentum
        assert updated.consolidation_strength == tag.consolidation_strength
        assert updated.retrieval_count == tag.retrieval_count

    def test_shift_proportional_to_ape(self):
        tag = make_emotional_tag(
            core_affect=CoreAffect(valence=0.0, arousal=0.0),
            momentum=AffectiveMomentum.zero(),
            mood=_neutral_mood(),
            consolidation_strength=0.5,
        )
        target = CoreAffect(valence=1.0, arousal=1.0)
        small = reconsolidate(tag, target, ape=0.1, learning_rate=0.2)
        large = reconsolidate(tag, target, ape=1.0, learning_rate=0.2)
        assert large.core_affect.valence > small.core_affect.valence

    def test_max_shift_capped_at_50_percent(self):
        """alpha is capped at 0.5, so valence moves at most halfway."""
        tag = make_emotional_tag(
            core_affect=CoreAffect(valence=0.0, arousal=0.0),
            momentum=AffectiveMomentum.zero(),
            mood=_neutral_mood(),
            consolidation_strength=0.5,
        )
        target = CoreAffect(valence=1.0, arousal=1.0)
        updated = reconsolidate(tag, target, ape=100.0, learning_rate=1.0)
        # alpha capped at 0.5 → valence = 0 + 0.5*(1-0) = 0.5
        assert math.isclose(updated.core_affect.valence, 0.5)


class TestRetrievalScore:
    def test_returns_float_in_unit_range(self):
        config = RetrievalConfig()
        decay = DecayConfig()
        m = make_test_memory(embedding=[1.0, 0.0])
        score = retrieval_score(
            query_embedding=[1.0, 0.0],
            query_affect=CoreAffect.neutral(),
            current_mood=_neutral_mood(),
            current_momentum=AffectiveMomentum.zero(),
            memory=m,
            activation_map={},
            now=datetime.now(tz=UTC),
            decay_config=decay,
            retrieval_config=config,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_emotionally_congruent_scores_higher_under_negative_mood(self):
        """Under strong negative Mood, a negative memory should outscore a positive one."""
        neg_mood = _mood(valence=-0.9, arousal=0.2)
        config = RetrievalConfig()
        decay = DecayConfig(base_decay=0.01)  # minimal decay
        now = datetime.now(tz=UTC)

        neg_mem = make_test_memory(valence=-0.8)
        pos_mem = make_test_memory(valence=0.8)

        def score(m):
            return retrieval_score(
                query_embedding=[],
                query_affect=CoreAffect(valence=-0.8, arousal=0.3),
                current_mood=neg_mood,
                current_momentum=AffectiveMomentum.zero(),
                memory=m,
                activation_map={},
                now=now,
                decay_config=decay,
                retrieval_config=config,
            )

        assert score(neg_mem) > score(pos_mem)

    def test_activation_map_boosts_score(self):
        """A memory with a positive activation level scores higher than without."""
        config = RetrievalConfig()
        decay = DecayConfig(base_decay=0.01)
        now = datetime.now(tz=UTC)
        m = make_test_memory(embedding=[1.0, 0.0])

        score_no_boost = retrieval_score(
            query_embedding=[1.0, 0.0],
            query_affect=CoreAffect.neutral(),
            current_mood=_neutral_mood(),
            current_momentum=AffectiveMomentum.zero(),
            memory=m,
            activation_map={},
            now=now,
            decay_config=decay,
            retrieval_config=config,
        )
        score_boosted = retrieval_score(
            query_embedding=[1.0, 0.0],
            query_affect=CoreAffect.neutral(),
            current_mood=_neutral_mood(),
            current_momentum=AffectiveMomentum.zero(),
            memory=m,
            activation_map={m.id: 1.0},
            now=now,
            decay_config=decay,
            retrieval_config=config,
        )
        assert score_boosted > score_no_boost

    def test_activation_map_absent_memory_gets_zero_boost(self):
        """A memory not in the activation map gets 0.0 from s6."""
        from emotional_memory.retrieval import _resonance_boost

        assert _resonance_boost("some-id", {}) == 0.0
        assert _resonance_boost("some-id", {"other-id": 0.8}) == 0.0

    def test_activation_map_present_memory_returns_level(self):
        """A memory in the activation map returns its exact activation level."""
        from emotional_memory.retrieval import _resonance_boost

        assert _resonance_boost("abc", {"abc": 0.6}) == pytest.approx(0.6)


class TestRetrievalPlan:
    def test_build_retrieval_plan_separates_ranking_from_side_effects(self):
        primary = make_test_memory(content="primary", embedding=[1.0, 0.0], valence=0.8)
        secondary = make_test_memory(content="secondary", embedding=[0.0, 1.0], valence=-0.8)

        def fake_spread(seed_ids: set[str], candidates: list, hops: int) -> dict[str, float]:
            assert seed_ids == {primary.id}
            assert hops == 2
            assert len(candidates) == 2
            return {secondary.id: 0.4}

        plan = build_retrieval_plan(
            query_embedding=[1.0, 0.0],
            query_affect=CoreAffect(valence=0.8, arousal=0.5),
            current_mood=_neutral_mood(),
            current_momentum=AffectiveMomentum.zero(),
            candidates=[primary, secondary],
            top_k=1,
            now=datetime.now(tz=UTC),
            decay_config=DecayConfig(),
            retrieval_config=RetrievalConfig(),
            propagation_hops=2,
            spreading_activation_fn=fake_spread,
        )

        assert plan.candidate_count == 2
        assert plan.seed_ids == [primary.id]
        assert len(plan.pass1) == 2
        assert len(plan.pass2) == 2
        assert plan.weights.total() == pytest.approx(1.0)
        assert plan.pass1[0].memory.id == primary.id
        assert plan.activation_map[secondary.id] == pytest.approx(0.4)
        assert plan.pass2[1].breakdown.raw_signals.resonance == pytest.approx(0.4)


class TestRetrievalConfig:
    def test_default_weights_valid(self):
        cfg = RetrievalConfig()
        assert len(cfg.base_weights) == 6

    def test_wrong_weights_length_raises(self):
        with pytest.raises(ValidationError, match="exactly 6"):
            RetrievalConfig(base_weights=[0.5, 0.5])

    def test_five_weights_raises(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(base_weights=[0.2, 0.2, 0.2, 0.2, 0.2])

    def test_seven_weights_raises(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(base_weights=[0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1])
