"""Fidelity benchmark: Appraisal-to-affect mapping (Scherer CPM).

Hypothesis: the five Scherer appraisal dimensions map systematically to
valence-arousal space:
  - High novelty → high arousal
  - High goal relevance + high coping → positive valence
  - Low goal relevance + low coping → negative valence
  - High self-relevance + low coping → high arousal

Reference: Scherer, K.R. (2009). The dynamic architecture of emotion:
Evidence for the component process model. Cognition and Emotion.
"""

import pytest

from emotional_memory.appraisal import AppraisalVector

pytestmark = pytest.mark.fidelity


def _appraise(
    novelty: float = 0.0,
    goal_relevance: float = 0.0,
    coping_potential: float = 0.5,
    norm_congruence: float = 0.0,
    self_relevance: float = 0.0,
):
    return AppraisalVector(
        novelty=novelty,
        goal_relevance=goal_relevance,
        coping_potential=coping_potential,
        norm_congruence=norm_congruence,
        self_relevance=self_relevance,
    ).to_core_affect()


@pytest.mark.parametrize("novelty", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_high_novelty_raises_arousal(novelty):
    """Higher novelty produces higher arousal, all else equal."""
    low = _appraise(novelty=0.0)
    high = _appraise(novelty=novelty)
    if novelty > 0.0:
        assert high.arousal > low.arousal, (
            f"novelty={novelty}: expected higher arousal, "
            f"got {high.arousal:.3f} vs {low.arousal:.3f}"
        )


def test_positive_goal_and_high_coping_yield_positive_valence():
    """Goal-furthering event + high coping → positive valence."""
    ca = _appraise(goal_relevance=1.0, coping_potential=1.0, norm_congruence=0.5)
    assert ca.valence > 0.0, f"Expected positive valence, got {ca.valence:.3f}"


def test_negative_goal_and_low_coping_yield_negative_valence():
    """Goal-obstructing event + helplessness → negative valence."""
    ca = _appraise(goal_relevance=-1.0, coping_potential=0.0, norm_congruence=-0.5)
    assert ca.valence < 0.0, f"Expected negative valence, got {ca.valence:.3f}"


def test_neutral_appraisal_zero_valence():
    """Completely neutral appraisal maps to zero valence."""
    ca = AppraisalVector.neutral().to_core_affect()
    assert ca.valence == 0.0, f"Neutral appraisal should have valence=0, got {ca.valence}"


def test_high_self_relevance_low_coping_raises_arousal():
    """High personal stakes + helplessness → heightened arousal."""
    low_stakes = _appraise(self_relevance=0.0, coping_potential=0.5)
    high_stakes = _appraise(self_relevance=1.0, coping_potential=0.0)
    assert high_stakes.arousal > low_stakes.arousal, (
        f"High self-relevance+low coping should raise arousal: "
        f"{high_stakes.arousal:.3f} vs {low_stakes.arousal:.3f}"
    )


def test_output_always_in_valid_range():
    """to_core_affect() always returns valid valence-arousal bounds."""
    test_cases = [
        dict(
            novelty=1.0,
            goal_relevance=1.0,
            coping_potential=1.0,
            norm_congruence=1.0,
            self_relevance=1.0,
        ),
        dict(
            novelty=-1.0,
            goal_relevance=-1.0,
            coping_potential=0.0,
            norm_congruence=-1.0,
            self_relevance=1.0,
        ),
        dict(
            novelty=0.0,
            goal_relevance=0.0,
            coping_potential=0.5,
            norm_congruence=0.0,
            self_relevance=0.0,
        ),
    ]
    for kwargs in test_cases:
        ca = _appraise(**kwargs)
        assert -1.0 <= ca.valence <= 1.0, f"valence out of range: {ca.valence}"
        assert 0.0 <= ca.arousal <= 1.0, f"arousal out of range: {ca.arousal}"


def test_norm_congruence_modulates_valence():
    """Norm-conforming events are more positively valenced than norm-violating ones."""
    conforming = _appraise(norm_congruence=1.0)
    violating = _appraise(norm_congruence=-1.0)
    assert conforming.valence > violating.valence, (
        f"Norm congruence should raise valence: "
        f"{conforming.valence:.3f} vs {violating.valence:.3f}"
    )
