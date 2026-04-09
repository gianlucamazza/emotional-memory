"""Smoke tests for emotional_memory.visualization.

All tests are skipped automatically when matplotlib is not installed.
"""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures() -> object:
    """Close all figures after each test to prevent memory leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------


def _sample_memories() -> list[tuple[float, float, float]]:
    return [
        (-0.8, 0.9, 0.3),
        (0.5, 0.6, 0.8),
        (-0.2, 0.3, 0.5),
        (0.9, 0.2, 0.9),
        (-0.5, 0.7, 0.2),
    ]


def _sample_stimmung_history() -> list[tuple[float, float, float, float]]:
    return [
        (0.0, 0.0, 0.3, 0.5),
        (300.0, -0.3, 0.6, 0.4),
        (600.0, -0.5, 0.8, 0.3),
        (900.0, -0.2, 0.5, 0.4),
        (1200.0, 0.1, 0.35, 0.5),
    ]


def _sample_links() -> list[object]:
    from emotional_memory.models import ResonanceLink

    return [
        ResonanceLink(source_id="mem-a", target_id="mem-b", strength=0.7, link_type="semantic"),
        ResonanceLink(source_id="mem-b", target_id="mem-c", strength=0.5, link_type="emotional"),
        ResonanceLink(source_id="mem-a", target_id="mem-c", strength=0.3, link_type="temporal"),
    ]


# ---------------------------------------------------------------------------
# 1. Circumplex
# ---------------------------------------------------------------------------


def test_plot_circumplex_returns_figure() -> None:
    from emotional_memory.visualization import plot_circumplex

    fig = plot_circumplex(_sample_memories())
    assert fig is not None
    assert hasattr(fig, "axes")


def test_plot_circumplex_empty_input() -> None:
    from emotional_memory.visualization import plot_circumplex

    fig = plot_circumplex([])
    assert fig is not None


def test_plot_circumplex_reuses_ax() -> None:
    from emotional_memory.visualization import plot_circumplex

    _, existing_ax = plt.subplots()
    fig = plot_circumplex(_sample_memories(), ax=existing_ax)
    assert fig is existing_ax.get_figure()


# ---------------------------------------------------------------------------
# 2. Decay Curves
# ---------------------------------------------------------------------------


def test_plot_decay_curves_returns_figure() -> None:
    from emotional_memory.visualization import plot_decay_curves

    fig = plot_decay_curves()
    assert fig is not None


def test_plot_decay_curves_multiple_lines() -> None:
    from emotional_memory.visualization import plot_decay_curves

    fig = plot_decay_curves(arousal_values=[0.0, 0.5, 1.0], retrieval_counts=[0, 5])
    ax = fig.axes[0]
    # 3 arousal x 2 retrieval = 6 lines
    assert len(ax.lines) == 6


# ---------------------------------------------------------------------------
# 3. Yerkes-Dodson
# ---------------------------------------------------------------------------


def test_plot_yerkes_dodson_returns_figure() -> None:
    from emotional_memory.visualization import plot_yerkes_dodson

    fig = plot_yerkes_dodson()
    assert fig is not None


def test_plot_yerkes_dodson_has_one_line() -> None:
    from emotional_memory.visualization import plot_yerkes_dodson

    fig = plot_yerkes_dodson()
    ax = fig.axes[0]
    assert len(ax.lines) >= 1


# ---------------------------------------------------------------------------
# 4. Retrieval Radar
# ---------------------------------------------------------------------------


def test_plot_retrieval_radar_returns_figure() -> None:
    from emotional_memory.visualization import plot_retrieval_radar

    fig = plot_retrieval_radar([0.8, 0.6, 0.4, 0.3, 0.7, 0.2])
    assert fig is not None


def test_plot_retrieval_radar_six_scores() -> None:
    from emotional_memory.visualization import SIGNAL_LABELS, plot_retrieval_radar

    scores = [0.8, 0.6, 0.4, 0.3, 0.7, 0.2]
    fig = plot_retrieval_radar(scores, SIGNAL_LABELS)
    ax = fig.axes[0]
    assert len(ax.get_xticks()) == 6


# ---------------------------------------------------------------------------
# 5. Stimmung Evolution
# ---------------------------------------------------------------------------


def test_plot_stimmung_evolution_returns_figure() -> None:
    from emotional_memory.visualization import plot_stimmung_evolution

    fig = plot_stimmung_evolution(_sample_stimmung_history())
    assert fig is not None


def test_plot_stimmung_evolution_three_main_lines() -> None:
    from emotional_memory.visualization import plot_stimmung_evolution

    fig = plot_stimmung_evolution(_sample_stimmung_history())
    ax = fig.axes[0]
    # 3 data lines + 3 baseline lines = 6 total
    assert len(ax.lines) >= 3


def test_plot_stimmung_evolution_empty() -> None:
    from emotional_memory.visualization import plot_stimmung_evolution

    fig = plot_stimmung_evolution([])
    assert fig is not None


# ---------------------------------------------------------------------------
# 6. Adaptive Weights Heatmap
# ---------------------------------------------------------------------------


def test_plot_adaptive_weights_heatmap_returns_figure() -> None:
    from emotional_memory.visualization import plot_adaptive_weights_heatmap

    fig = plot_adaptive_weights_heatmap(resolution=5)
    assert fig is not None


def test_plot_adaptive_weights_heatmap_six_subplots() -> None:
    from emotional_memory.visualization import plot_adaptive_weights_heatmap

    fig = plot_adaptive_weights_heatmap(resolution=5)
    assert len(fig.axes) >= 6


# ---------------------------------------------------------------------------
# 7. Resonance Network
# ---------------------------------------------------------------------------


def test_plot_resonance_network_returns_figure() -> None:
    from emotional_memory.visualization import plot_resonance_network

    fig = plot_resonance_network(_sample_links())
    assert fig is not None


def test_plot_resonance_network_empty_links() -> None:
    from emotional_memory.visualization import plot_resonance_network

    fig = plot_resonance_network([])
    assert fig is not None


def test_plot_resonance_network_with_labels() -> None:
    from emotional_memory.visualization import plot_resonance_network

    labels = {"mem-a": "Joy", "mem-b": "Fear", "mem-c": "Calm"}
    fig = plot_resonance_network(_sample_links(), node_labels=labels)
    assert fig is not None


# ---------------------------------------------------------------------------
# 8. Appraisal Radar
# ---------------------------------------------------------------------------


def test_plot_appraisal_radar_returns_figure_from_model() -> None:
    from emotional_memory.appraisal import AppraisalVector
    from emotional_memory.visualization import plot_appraisal_radar

    av = AppraisalVector(
        novelty=0.8,
        goal_relevance=0.5,
        coping_potential=0.3,
        norm_congruence=-0.2,
        self_relevance=0.9,
    )
    fig = plot_appraisal_radar(av)
    assert fig is not None


def test_plot_appraisal_radar_returns_figure_from_dict() -> None:
    from emotional_memory.visualization import plot_appraisal_radar

    d = {
        "novelty": 0.5,
        "goal_relevance": -0.3,
        "coping_potential": 0.7,
        "norm_congruence": 0.1,
        "self_relevance": 0.4,
    }
    fig = plot_appraisal_radar(d)
    assert fig is not None


def test_plot_appraisal_radar_five_spokes() -> None:
    from emotional_memory.appraisal import AppraisalVector
    from emotional_memory.visualization import plot_appraisal_radar

    av = AppraisalVector.neutral()
    fig = plot_appraisal_radar(av)
    ax = fig.axes[0]
    assert len(ax.get_xticks()) == 5


def test_plot_appraisal_radar_invalid_type() -> None:
    from emotional_memory.visualization import plot_appraisal_radar

    with pytest.raises(TypeError):
        plot_appraisal_radar("invalid")
