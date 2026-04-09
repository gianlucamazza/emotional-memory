import math

import pytest
from pydantic import ValidationError

from emotional_memory.affect import CoreAffect
from emotional_memory.stimmung import StimmungField


class TestStimmungField:
    def test_neutral_values(self):
        s = StimmungField.neutral()
        assert s.valence == 0.0
        assert s.arousal == 0.0
        assert s.dominance == 0.5
        assert s.inertia == 0.5

    def test_frozen(self):
        s = StimmungField.neutral()
        with pytest.raises(ValidationError):
            s.valence = 0.5  # type: ignore[misc]

    def test_clamp_valence(self):
        s = StimmungField(
            valence=2.0,
            arousal=0.5,
            dominance=0.5,
            inertia=0.5,
            timestamp=StimmungField.neutral().timestamp,
        )
        assert s.valence == 1.0

    def test_clamp_arousal(self):
        s = StimmungField(
            valence=0.0,
            arousal=-1.0,
            dominance=0.5,
            inertia=0.5,
            timestamp=StimmungField.neutral().timestamp,
        )
        assert s.arousal == 0.0

    def test_update_moves_toward_affect(self):
        s = StimmungField.neutral()
        positive = CoreAffect(valence=1.0, arousal=1.0)
        s2 = s.update(positive, alpha=0.1)
        # With inertia=0.5, effective_alpha = 0.1 * 0.5 = 0.05
        assert s2.valence > s.valence
        assert s2.arousal > s.arousal

    def test_update_returns_new_instance(self):
        s = StimmungField.neutral()
        s2 = s.update(CoreAffect(valence=1.0, arousal=1.0))
        assert s is not s2

    def test_high_inertia_resists_change(self):
        high = StimmungField(
            valence=0.0,
            arousal=0.0,
            dominance=0.5,
            inertia=0.9,
            timestamp=StimmungField.neutral().timestamp,
        )
        low = StimmungField(
            valence=0.0,
            arousal=0.0,
            dominance=0.5,
            inertia=0.1,
            timestamp=StimmungField.neutral().timestamp,
        )
        affect = CoreAffect(valence=1.0, arousal=1.0)
        assert high.update(affect).valence < low.update(affect).valence

    def test_repeated_positive_updates_drift_positive(self):
        s = StimmungField.neutral()
        affect = CoreAffect(valence=1.0, arousal=0.8)
        for _ in range(50):
            s = s.update(affect, alpha=0.2)
        assert s.valence > 0.5

    def test_distance_zero_for_equal(self):
        s = StimmungField.neutral()
        assert s.distance(s) == 0.0


class TestStimmungDecayConfig:
    def test_defaults_are_sensible(self):
        from emotional_memory.stimmung import StimmungDecayConfig

        cfg = StimmungDecayConfig()
        assert cfg.base_half_life_seconds == pytest.approx(3600.0)
        assert cfg.baseline_valence == pytest.approx(0.0)
        assert cfg.baseline_arousal == pytest.approx(0.3)
        assert cfg.baseline_dominance == pytest.approx(0.5)


class TestStimmungRegress:
    def test_zero_elapsed_returns_identical_values(self):
        from datetime import UTC, datetime

        from emotional_memory.stimmung import StimmungDecayConfig

        s = StimmungField(
            valence=0.8, arousal=0.9, dominance=0.7, inertia=0.5, timestamp=datetime.now(tz=UTC)
        )
        r = s.regress(s.timestamp, StimmungDecayConfig())
        assert r.valence == pytest.approx(s.valence, abs=1e-9)
        assert r.arousal == pytest.approx(s.arousal, abs=1e-9)

    def test_regression_moves_toward_baseline(self):
        from datetime import UTC, datetime, timedelta

        from emotional_memory.stimmung import StimmungDecayConfig

        cfg = StimmungDecayConfig(base_half_life_seconds=1.0, inertia_scale=0.0)
        s = StimmungField(
            valence=1.0, arousal=1.0, dominance=1.0, inertia=0.0, timestamp=datetime.now(tz=UTC)
        )
        future = s.timestamp + timedelta(hours=1)
        r = s.regress(future, cfg)
        assert r.valence < s.valence
        assert r.arousal < s.arousal

    def test_half_life_semantics(self):
        """After one half-life, deviation from baseline should halve."""
        from datetime import UTC, datetime, timedelta

        from emotional_memory.stimmung import StimmungDecayConfig

        half_life = 3600.0
        cfg = StimmungDecayConfig(base_half_life_seconds=half_life, inertia_scale=0.0)
        s = StimmungField(
            valence=1.0, arousal=0.5, dominance=0.5, inertia=0.0, timestamp=datetime.now(tz=UTC)
        )
        future = s.timestamp + timedelta(seconds=half_life)
        r = s.regress(future, cfg)
        # valence deviation halves: 1.0 → 0.5 (baseline 0.0)
        assert r.valence == pytest.approx(0.5, abs=0.01)

    def test_high_inertia_slows_regression(self):
        from datetime import UTC, datetime, timedelta

        from emotional_memory.stimmung import StimmungDecayConfig

        cfg = StimmungDecayConfig(base_half_life_seconds=1.0, inertia_scale=10.0)
        ts = datetime.now(tz=UTC)
        rigid = StimmungField(valence=1.0, arousal=0.5, dominance=0.5, inertia=0.9, timestamp=ts)
        fluid = StimmungField(valence=1.0, arousal=0.5, dominance=0.5, inertia=0.0, timestamp=ts)
        future = ts + timedelta(seconds=100)
        assert rigid.regress(future, cfg).valence > fluid.regress(future, cfg).valence

    def test_regress_returns_new_instance(self):
        from datetime import timedelta

        from emotional_memory.stimmung import StimmungDecayConfig

        s = StimmungField.neutral()
        future = s.timestamp + timedelta(seconds=60)
        r = s.regress(future, StimmungDecayConfig())
        assert r is not s

    def test_update_with_decay_config_regresses_first(self):
        """Stimmung after a long silence should reflect baseline pull."""
        from datetime import UTC, datetime, timedelta

        from emotional_memory.stimmung import StimmungDecayConfig

        cfg = StimmungDecayConfig(base_half_life_seconds=1.0, inertia_scale=0.0)
        ts = datetime.now(tz=UTC)
        s = StimmungField(valence=1.0, arousal=0.9, dominance=0.9, inertia=0.0, timestamp=ts)
        # After a very long time, a neutral update should produce near-baseline result
        future = ts + timedelta(hours=100)
        affect = CoreAffect(valence=0.0, arousal=0.3)
        updated = s.update(affect, alpha=0.01, now=future, decay_config=cfg)
        assert updated.valence < 0.1  # pulled strongly toward 0.0

    def test_distance_symmetric(self):
        a = StimmungField.neutral()
        b = StimmungField(
            valence=0.5, arousal=0.5, dominance=0.8, inertia=0.3, timestamp=a.timestamp
        )
        assert math.isclose(a.distance(b), b.distance(a))

    def test_zero_inertia_follows_affect_fully(self):
        # inertia=0 → effective_alpha = alpha * 1.0 = alpha
        s = StimmungField(
            valence=0.0,
            arousal=0.0,
            dominance=0.5,
            inertia=0.0,
            timestamp=StimmungField.neutral().timestamp,
        )
        affect = CoreAffect(valence=1.0, arousal=1.0)
        s2 = s.update(affect, alpha=0.5)
        assert math.isclose(s2.valence, 0.5)  # 0.5 * 0 + 0.5 * 1 = 0.5


class TestStimmungProperties:
    """Mathematical invariants for StimmungField."""

    @pytest.mark.parametrize(
        "valence,arousal",
        [(-5.0, 5.0), (0.8, 0.9), (-0.9, 0.1), (1.0, 0.0), (0.0, 1.0)],
    )
    def test_bounded_after_extreme_updates(self, valence, arousal):
        """After many extreme-valued updates, Stimmung stays in valid ranges."""
        s = StimmungField.neutral()
        affect = CoreAffect(valence=valence, arousal=arousal)
        for _ in range(100):
            s = s.update(affect, alpha=0.3)
        assert -1.0 <= s.valence <= 1.0, f"valence={s.valence} out of [-1,1]"
        assert 0.0 <= s.arousal <= 1.0, f"arousal={s.arousal} out of [0,1]"

    def test_convergence_toward_constant_affect(self):
        """Repeated constant-affect updates converge within 5% of the target."""
        s = StimmungField.neutral()
        target_valence = 0.8
        affect = CoreAffect(valence=target_valence, arousal=0.5)
        for _ in range(300):
            s = s.update(affect, alpha=0.1)
        assert abs(s.valence - target_valence) < 0.05, (
            f"Expected convergence to {target_valence}, got {s.valence:.4f}"
        )

    @pytest.mark.parametrize("inertia", [0.0, 0.25, 0.5, 0.75, 0.9])
    def test_single_update_bounded_by_inertia(self, inertia):
        """One impulse shifts valence by at most alpha*(1-inertia)*delta."""
        alpha = 0.3
        s = StimmungField(
            valence=0.0,
            arousal=0.0,
            dominance=0.5,
            inertia=inertia,
            timestamp=StimmungField.neutral().timestamp,
        )
        affect = CoreAffect(valence=1.0, arousal=0.5)
        s2 = s.update(affect, alpha=alpha)
        max_shift = alpha * (1.0 - inertia) * 1.0  # delta_valence = 1.0
        assert s2.valence <= max_shift + 1e-9, (
            f"inertia={inertia}: shift {s2.valence:.4f} exceeds bound {max_shift:.4f}"
        )

    @pytest.mark.parametrize("n_updates", [1, 5, 20, 100])
    def test_values_always_in_range(self, n_updates):
        """Stimmung stays in bounds regardless of update count."""
        s = StimmungField.neutral()
        for i in range(n_updates):
            v = 1.0 if i % 2 == 0 else -1.0
            s = s.update(CoreAffect(valence=v, arousal=float(i % 2)), alpha=0.2)
        assert -1.0 <= s.valence <= 1.0
        assert 0.0 <= s.arousal <= 1.0
