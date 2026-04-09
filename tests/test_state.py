from datetime import datetime, timedelta, timezone

from emotional_memory.affect import CoreAffect
from emotional_memory.state import AffectiveState


def _now():
    return datetime.now(tz=timezone.utc)


def _later(seconds: float):
    return _now() + timedelta(seconds=seconds)


class TestAffectiveState:
    def test_initial_is_neutral(self):
        s = AffectiveState.initial()
        assert s.core_affect.valence == 0.0
        assert s.core_affect.arousal == 0.0
        assert s.momentum.d_valence == 0.0
        assert s.momentum.d_arousal == 0.0
        assert s.stimmung.valence == 0.0

    def test_update_returns_new_instance(self):
        s = AffectiveState.initial()
        s2 = s.update(CoreAffect(valence=0.5, arousal=0.5))
        assert s is not s2

    def test_update_changes_core_affect(self):
        s = AffectiveState.initial()
        new_affect = CoreAffect(valence=0.7, arousal=0.6)
        s2 = s.update(new_affect)
        assert s2.core_affect.valence == 0.7
        assert s2.core_affect.arousal == 0.6

    def test_stimmung_drifts_positive_after_positive_updates(self):
        s = AffectiveState.initial()
        for _ in range(30):
            s = s.update(CoreAffect(valence=1.0, arousal=0.8), stimmung_alpha=0.2)
        assert s.stimmung.valence > 0.3

    def test_momentum_reflects_direction(self):
        s = AffectiveState.initial()
        # Positive affect → valence increasing
        s = s.update(CoreAffect(valence=0.0, arousal=0.0))
        s = s.update(CoreAffect(valence=0.5, arousal=0.0))
        assert s.momentum.d_valence > 0.0

    def test_momentum_negative_when_declining(self):
        s = AffectiveState.initial()
        s = s.update(CoreAffect(valence=0.8, arousal=0.0))
        s = s.update(CoreAffect(valence=0.2, arousal=0.0))
        assert s.momentum.d_valence < 0.0

    def test_acceleration_reflects_change_of_velocity(self):
        s = AffectiveState.initial()
        # Accelerating: steps +0.2, +0.4
        s = s.update(CoreAffect(valence=0.0, arousal=0.0))
        s = s.update(CoreAffect(valence=0.2, arousal=0.0))
        s = s.update(CoreAffect(valence=0.6, arousal=0.0))
        # dd_valence should be positive (velocity increased)
        assert s.momentum.dd_valence > 0.0

    def test_zero_momentum_with_single_update(self):
        s = AffectiveState.initial()
        s2 = s.update(CoreAffect(valence=0.5, arousal=0.5))
        # Only 1 history point → no diff possible → zero momentum
        assert s2.momentum.d_valence == 0.0
