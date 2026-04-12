"""Generate static PNG images for documentation.

Usage::

    uv run python scripts/generate_docs_images.py

Writes 8 PNG files to docs/images/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Headless rendering — must be set before any pyplot import
import matplotlib

matplotlib.use("Agg")

# Make sure the src layout is importable when run from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt

from emotional_memory.visualization import (
    plot_adaptive_weights_heatmap,
    plot_appraisal_radar,
    plot_circumplex,
    plot_decay_curves,
    plot_mood_evolution,
    plot_resonance_network,
    plot_retrieval_radar,
    plot_yerkes_dodson,
)

OUT_DIR = Path(__file__).parent.parent / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: object, name: str) -> None:
    import matplotlib.figure

    if not isinstance(fig, matplotlib.figure.Figure):
        raise TypeError(f"Expected Figure, got {type(fig)}")
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.relative_to(Path.cwd())}")


def main() -> None:
    print("Generating docs/images/ ...")

    # 1. Circumplex
    memories = [
        (-0.8, 0.9, 0.2),  # tense
        (0.6, 0.8, 0.9),  # excited
        (-0.4, 0.2, 0.4),  # sad
        (0.7, 0.15, 0.8),  # calm/content
        (0.0, 0.5, 0.5),  # neutral
        (-0.6, 0.7, 0.3),  # fearful
        (0.9, 0.6, 0.95),  # elated
        (0.2, 0.3, 0.6),  # relaxed
    ]
    _save(plot_circumplex(memories), "circumplex")

    # 2. Decay curves
    _save(
        plot_decay_curves(
            arousal_values=[0.0, 0.5, 1.0],
            retrieval_counts=[0, 5, 10],
        ),
        "decay_curves",
    )

    # 3. Yerkes-Dodson
    _save(plot_yerkes_dodson(mood_arousal=0.3), "yerkes_dodson")

    # 4. Retrieval radar
    _save(
        plot_retrieval_radar([0.82, 0.61, 0.45, 0.30, 0.72, 0.18]),
        "retrieval_radar",
    )

    # 5. Mood evolution — simulate 30 minutes of events
    history = [
        (0.0, 0.0, 0.30, 0.50),
        (120.0, -0.20, 0.55, 0.40),
        (300.0, -0.50, 0.80, 0.30),
        (480.0, -0.60, 0.85, 0.25),
        (600.0, -0.45, 0.70, 0.32),
        (900.0, -0.20, 0.50, 0.42),
        (1200.0, 0.05, 0.38, 0.48),
        (1500.0, 0.10, 0.32, 0.50),
        (1800.0, 0.02, 0.31, 0.50),
    ]
    _save(plot_mood_evolution(history), "mood_evolution")

    # 6. Adaptive weights heatmap
    _save(plot_adaptive_weights_heatmap(resolution=30), "adaptive_weights_heatmap")

    # 7. Resonance network
    from emotional_memory.models import ResonanceLink

    links = [
        ResonanceLink(
            source_id="mem-joy", target_id="mem-hope", strength=0.8, link_type="emotional"
        ),
        ResonanceLink(
            source_id="mem-joy", target_id="mem-play", strength=0.6, link_type="semantic"
        ),
        ResonanceLink(
            source_id="mem-fear", target_id="mem-relief", strength=0.7, link_type="contrastive"
        ),
        ResonanceLink(source_id="mem-fear", target_id="mem-run", strength=0.5, link_type="causal"),
        ResonanceLink(
            source_id="mem-hope", target_id="mem-relief", strength=0.4, link_type="temporal"
        ),
        ResonanceLink(
            source_id="mem-play", target_id="mem-joy", strength=0.3, link_type="temporal"
        ),
    ]
    node_labels = {
        "mem-joy": "Joy",
        "mem-hope": "Hope",
        "mem-play": "Play",
        "mem-fear": "Fear",
        "mem-relief": "Relief",
        "mem-run": "Run",
    }
    _save(plot_resonance_network(links, node_labels=node_labels), "resonance_network")

    # 8. Appraisal radar
    from emotional_memory.appraisal import AppraisalVector

    av = AppraisalVector(
        novelty=0.8,
        goal_relevance=0.6,
        coping_potential=0.2,
        norm_congruence=-0.4,
        self_relevance=0.9,
    )
    _save(plot_appraisal_radar(av), "appraisal_radar")

    print(f"\nDone — {len(list(OUT_DIR.glob('*.png')))} PNG files in {OUT_DIR}")


if __name__ == "__main__":
    main()
