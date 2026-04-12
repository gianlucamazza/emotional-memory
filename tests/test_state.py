from datetime import UTC, datetime, timedelta

import pytest

from emotional_memory.affect import CoreAffect
from emotional_memory.state import AffectiveState


def _now():
    return datetime.now(tz=UTC)


def _later(seconds: float):
    return _now() + timedelta(seconds=seconds)


class TestAffectiveState:
    def test_initial_is_neutral(self):
        s = AffectiveState.initial()
        assert s.core_affect.valence == 0.0
        assert s.core_affect.arousal == 0.0
        assert s.momentum.d_valence == 0.0
        assert s.momentum.d_arousal == 0.0
        assert s.mood.valence == 0.0

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

    def test_mood_drifts_positive_after_positive_updates(self):
        s = AffectiveState.initial()
        for _ in range(30):
            s = s.update(CoreAffect(valence=1.0, arousal=0.8), mood_alpha=0.2)
        assert s.mood.valence > 0.3

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


class TestAffectiveStateSnapshot:
    def test_snapshot_contains_history(self):
        s = AffectiveState.initial()
        s = s.update(CoreAffect(valence=0.5, arousal=0.5))
        snap = s.snapshot()
        assert "_history" in snap
        assert len(snap["_history"]) >= 1

    def test_restore_roundtrip_core_affect(self):
        s = AffectiveState.initial()
        s = s.update(CoreAffect(valence=0.7, arousal=0.4))
        snap = s.snapshot()
        restored = AffectiveState.restore(snap)
        assert restored.core_affect.valence == pytest.approx(0.7)
        assert restored.core_affect.arousal == pytest.approx(0.4)

    def test_restore_roundtrip_mood(self):
        s = AffectiveState.initial()
        for _ in range(20):
            s = s.update(CoreAffect(valence=1.0, arousal=0.8), mood_alpha=0.2)
        snap = s.snapshot()
        restored = AffectiveState.restore(snap)
        assert restored.mood.valence == pytest.approx(s.mood.valence, abs=1e-9)

    def test_restore_preserves_momentum_history(self):
        s = AffectiveState.initial()
        s = s.update(CoreAffect(valence=0.2, arousal=0.3))
        s = s.update(CoreAffect(valence=0.5, arousal=0.5))
        s = s.update(CoreAffect(valence=0.8, arousal=0.7))
        snap = s.snapshot()
        restored = AffectiveState.restore(snap)
        # Use identical `now` so dt is the same for both and momentum is deterministic
        test_now = datetime.now(tz=UTC)
        next_s = s.update(CoreAffect(valence=0.9, arousal=0.9), now=test_now)
        next_r = restored.update(CoreAffect(valence=0.9, arousal=0.9), now=test_now)
        assert next_s.momentum.d_valence == pytest.approx(next_r.momentum.d_valence, abs=1e-6)

    def test_restore_does_not_mutate_snapshot(self):
        s = AffectiveState.initial()
        s = s.update(CoreAffect(valence=0.5, arousal=0.5))
        snap = s.snapshot()
        keys_before = set(snap.keys())
        AffectiveState.restore(snap)
        assert set(snap.keys()) == keys_before

    def test_restore_handles_empty_history(self):
        s = AffectiveState.initial()
        snap = s.snapshot()
        snap["_history"] = []
        restored = AffectiveState.restore(snap)
        assert restored.momentum.d_valence == 0.0

    def test_restore_skips_malformed_history_entries(self):
        s = AffectiveState.initial()
        snap = s.snapshot()
        snap["_history"] = [["2024-01-01", 0.5, 0.5], ["bad"]]  # second entry malformed
        restored = AffectiveState.restore(snap)
        assert len(restored._history) == 1
