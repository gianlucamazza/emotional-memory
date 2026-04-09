"""Gold-standard dataset for LLM appraisal quality benchmarks.

Each AppraisalCase pairs a natural-language phrase with directional assertions
over AppraisalVector dimensions. Assertions use wide bands (e.g. > 0.3, < -0.2)
to handle LLM non-determinism while still catching systematic prompt failures.

AppraisalVector fields (Scherer CPM):
    novelty         [-1, 1]  unexpected ↔ routine
    goal_relevance  [-1, 1]  obstructs ↔ furthers goals
    coping_potential [0, 1]  helpless ↔ full control
    norm_congruence [-1, 1]  violates norms ↔ conforms
    self_relevance   [0, 1]  irrelevant ↔ deeply personal
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DirectionalAssertion:
    """A single directional check on one dimension."""

    dimension: str  # AppraisalVector field name
    operator: str  # ">" or "<"
    threshold: float


@dataclass(frozen=True)
class AppraisalCase:
    """A phrase with expected directional appraisal outcomes."""

    label: str
    phrase: str
    assertions: tuple[DirectionalAssertion, ...]


def _gt(dim: str, threshold: float) -> DirectionalAssertion:
    return DirectionalAssertion(dim, ">", threshold)


def _lt(dim: str, threshold: float) -> DirectionalAssertion:
    return DirectionalAssertion(dim, "<", threshold)


APPRAISAL_DATASET: list[AppraisalCase] = [
    AppraisalCase(
        label="personal_loss_job",
        phrase="I just got fired from my job without any warning.",
        assertions=(
            _lt("goal_relevance", -0.2),
            _lt("coping_potential", 0.6),
            _gt("self_relevance", 0.3),
        ),
    ),
    AppraisalCase(
        label="achievement_promotion",
        phrase="I got promoted to senior engineer after two years of hard work.",
        assertions=(
            _gt("goal_relevance", 0.2),
            _gt("norm_congruence", 0.0),
            _gt("self_relevance", 0.3),
        ),
    ),
    AppraisalCase(
        label="surprise_lottery",
        phrase="I won the lottery completely out of the blue.",
        assertions=(
            _gt("novelty", 0.3),
            _gt("goal_relevance", 0.0),
        ),
    ),
    AppraisalCase(
        label="routine_lunch",
        phrase="I ate the same sandwich I always have for lunch.",
        assertions=(
            _lt("novelty", 0.2),
            _lt("self_relevance", 0.5),
        ),
    ),
    AppraisalCase(
        label="moral_violation_credit",
        phrase="My coworker lied and took credit for my project in front of the whole team.",
        assertions=(
            _lt("norm_congruence", -0.2),
            _lt("goal_relevance", 0.0),
            _gt("self_relevance", 0.3),
        ),
    ),
    AppraisalCase(
        label="physical_danger",
        phrase="A car ran a red light and nearly hit me while I was crossing the street.",
        assertions=(
            _gt("novelty", 0.0),
            _lt("coping_potential", 0.6),
            _gt("self_relevance", 0.3),
        ),
    ),
    AppraisalCase(
        label="kindness_stranger",
        phrase="A complete stranger helped me carry my heavy groceries to the car.",
        assertions=(
            _gt("norm_congruence", 0.0),
            _gt("goal_relevance", 0.0),
        ),
    ),
    AppraisalCase(
        label="grief_loss",
        phrase="My grandmother passed away last night after a long illness.",
        assertions=(
            _lt("goal_relevance", -0.2),
            _gt("self_relevance", 0.5),
            _lt("coping_potential", 0.6),
        ),
    ),
    AppraisalCase(
        label="relief_medical",
        phrase="The biopsy results came back completely negative — no cancer.",
        assertions=(
            _gt("goal_relevance", 0.0),
            _gt("coping_potential", 0.3),
        ),
    ),
    AppraisalCase(
        label="embarrassment_public",
        phrase="I tripped and fell flat on my face in front of a large crowd.",
        assertions=(
            _lt("norm_congruence", 0.0),
            _gt("self_relevance", 0.3),
        ),
    ),
    AppraisalCase(
        label="betrayal_friend",
        phrase="My best friend shared my deepest secrets with people I barely know.",
        assertions=(
            _lt("norm_congruence", -0.3),
            _gt("self_relevance", 0.4),
            _lt("goal_relevance", 0.0),
        ),
    ),
    AppraisalCase(
        label="accomplishment_skill",
        phrase="After months of daily practice, I finally played the piano piece perfectly.",
        assertions=(
            _gt("goal_relevance", 0.2),
            _gt("coping_potential", 0.4),
            _gt("self_relevance", 0.3),
        ),
    ),
    AppraisalCase(
        label="injustice_verdict",
        phrase="The court acquitted someone who was clearly guilty based on all the evidence.",
        assertions=(_lt("norm_congruence", -0.3),),
    ),
    AppraisalCase(
        label="neutral_report",
        phrase="The quarterly financial report was released on schedule as expected.",
        assertions=(
            _lt("novelty", 0.3),
            _lt("self_relevance", 0.5),
        ),
    ),
    AppraisalCase(
        label="existential_mortality",
        phrase="I suddenly realized how short life really is and how little time I have left.",
        assertions=(
            _gt("self_relevance", 0.4),
            _lt("coping_potential", 0.7),
        ),
    ),
]
