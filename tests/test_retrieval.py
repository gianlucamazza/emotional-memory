import math
from datetime import datetime, timezone

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.decay import DecayConfig
from emotional_memory.models import Memory, make_emotional_tag
from emotional_memory.retrieval import (
    RetrievalConfig,
    adaptive_weights,
    affective_prediction_error,
    reconsolidate,
    retrieval_score,
)
from emotional_memory.stimmung import StimmungField


def _neutral_stimmung():
    return StimmungField.neutral()


def _stimmung(valence: float = 0.0, arousal: float = 0.0):
    return StimmungField(
        valence=valence,
        arousal=arousal,
        dominance=0.5,
        inertia=0.5,
        timestamp=datetime.now(tz=timezone.utc),
    )


def _memory(valence: float = 0.0, arousal: float = 0.5, embedding=None):
    tag = make_emotional_tag(
        core_affect=CoreAffect(valence=valence, arousal=arousal),
        momentum=AffectiveMomentum.zero(),
        stimmung=_neutral_stimmung(),
        consolidation_strength=0.7,
    )
    return Memory.create(content="test", tag=tag, embedding=embedding)


class TestAdaptiveWeights:
    def test_sums_to_one_neutral(self):
        w = adaptive_weights(_neutral_stimmung(), [0.35, 0.25, 0.15, 0.10, 0.10, 0.05])
        assert math.isclose(w.sum(), 1.0, rel_tol=1e-9)

    def test_sums_to_one_negative(self):
        w = adaptive_weights(_stimmung(valence=-0.8), [0.35, 0.25, 0.15, 0.10, 0.10, 0.05])
        assert math.isclose(w.sum(), 1.0, rel_tol=1e-9)

    def test_sums_to_one_high_arousal(self):
        w = adaptive_weights(_stimmung(arousal=0.9), [0.35, 0.25, 0.15, 0.10, 0.10, 0.05])
        assert math.isclose(w.sum(), 1.0, rel_tol=1e-9)

    def test_sums_to_one_calm(self):
        w = adaptive_weights(
            _stimmung(valence=0.1, arousal=0.1), [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        )
        assert math.isclose(w.sum(), 1.0, rel_tol=1e-9)

    def test_negative_stimmung_increases_emotional_weight(self):
        base = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        w_neg = adaptive_weights(_stimmung(valence=-0.8), base)
        w_neu = adaptive_weights(_neutral_stimmung(), base)
        # w[1] = stimmung_congruence, w[2] = affect_proximity
        assert w_neg[1] > w_neu[1]

    def test_negative_stimmung_reduces_semantic_weight(self):
        base = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        w_neg = adaptive_weights(_stimmung(valence=-0.8), base)
        w_neu = adaptive_weights(_neutral_stimmung(), base)
        assert w_neg[0] < w_neu[0]

    def test_high_arousal_increases_momentum_weight(self):
        base = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        w_high = adaptive_weights(_stimmung(arousal=0.9), base)
        w_neu = adaptive_weights(_neutral_stimmung(), base)
        assert w_high[3] > w_neu[3]

    def test_calm_increases_semantic_weight(self):
        # Compare calm (qualifies) vs non-calm (does not qualify)
        base = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        w_calm = adaptive_weights(_stimmung(valence=0.1, arousal=0.1), base)
        w_active = adaptive_weights(_stimmung(valence=0.5, arousal=0.6), base)
        assert w_calm[0] > w_active[0]

    def test_no_negative_weights(self):
        base = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        for valence in [-1.0, -0.5, 0.0, 0.5]:
            for arousal in [0.0, 0.5, 1.0]:
                w = adaptive_weights(_stimmung(valence, arousal), base)
                assert all(wi >= 0.0 for wi in w)


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
            stimmung=_neutral_stimmung(),
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
            stimmung=_neutral_stimmung(),
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
            stimmung=_neutral_stimmung(),
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
        m = _memory(embedding=[1.0, 0.0])
        score = retrieval_score(
            query_embedding=[1.0, 0.0],
            query_affect=CoreAffect.neutral(),
            current_stimmung=_neutral_stimmung(),
            current_momentum=AffectiveMomentum.zero(),
            memory=m,
            active_memory_ids=[],
            now=datetime.now(tz=timezone.utc),
            decay_config=decay,
            retrieval_config=config,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_emotionally_congruent_scores_higher_under_negative_stimmung(self):
        """Under strong negative Stimmung, a negative memory should outscore a positive one."""
        neg_stimmung = _stimmung(valence=-0.9, arousal=0.2)
        config = RetrievalConfig()
        decay = DecayConfig(base_decay=0.01)  # minimal decay
        now = datetime.now(tz=timezone.utc)

        neg_mem = _memory(valence=-0.8)
        pos_mem = _memory(valence=0.8)

        def score(m):
            return retrieval_score(
                query_embedding=[],
                query_affect=CoreAffect(valence=-0.8, arousal=0.3),
                current_stimmung=neg_stimmung,
                current_momentum=AffectiveMomentum.zero(),
                memory=m,
                active_memory_ids=[],
                now=now,
                decay_config=decay,
                retrieval_config=config,
            )

        assert score(neg_mem) > score(pos_mem)
