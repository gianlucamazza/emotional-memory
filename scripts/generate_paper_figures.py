"""Generate PDF figures for paper/main.tex.

Usage::

    uv run python scripts/generate_paper_figures.py
    uv run python scripts/generate_paper_figures.py --out paper/figures

Produces four deterministic figures (seeded synthetic data):
  fig1_circumplex.pdf  — Russell circumplex with 12 sample memories
  fig2_decay_curves.pdf — ACT-R decay curves at three arousal levels
  fig3_mood_evolution.pdf — PAD trajectory over a simulated conversation
  fig4_resonance_network.pdf — Resonance graph snapshot
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    print(f"matplotlib not installed: {exc}\nRun: pip install emotional-memory[viz]")
    sys.exit(1)


def _make_circumplex(out_dir: Path) -> None:
    from emotional_memory.visualization import plot_circumplex

    rng = random.Random(42)
    memories = [
        (rng.uniform(-0.9, 0.9), rng.uniform(0.05, 0.95), rng.uniform(0.2, 1.0)) for _ in range(12)
    ]
    fig = plot_circumplex(memories, title="Sample memory distribution (Russell circumplex)")
    fig.savefig(out_dir / "fig1_circumplex.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig1_circumplex.pdf")


def _make_decay_curves(out_dir: Path) -> None:
    from emotional_memory.visualization import plot_decay_curves

    fig = plot_decay_curves(
        arousal_values=(0.1, 0.5, 0.9),
        retrieval_counts=(0, 5, 15),
        title="ACT-R power-law decay modulated by arousal (McGaugh 2004)",
    )
    fig.savefig(out_dir / "fig2_decay_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_decay_curves.pdf")


def _make_mood_evolution(out_dir: Path) -> None:
    from emotional_memory.visualization import plot_mood_evolution

    rng = random.Random(7)
    v, a, d = 0.0, 0.5, 0.0
    history = []
    for t in range(20):
        v = max(-1.0, min(1.0, v + rng.gauss(0, 0.12)))
        a = max(0.0, min(1.0, a + rng.gauss(0, 0.08)))
        d = max(-1.0, min(1.0, d + rng.gauss(0, 0.10)))
        # 180 s per conversational turn → x-axis shows realistic minutes (0-57)
        history.append((float(t * 180), v, a, d))

    fig = plot_mood_evolution(history, title="PAD mood-field trajectory (simulated conversation)")
    fig.savefig(out_dir / "fig3_mood_evolution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_mood_evolution.pdf")


def _make_resonance_network(out_dir: Path) -> None:
    from emotional_memory.models import ResonanceLink
    from emotional_memory.visualization import plot_resonance_network

    rng = random.Random(99)
    node_ids = [f"m{i}" for i in range(7)]
    # canonical types matching Literal in ResonanceLink + visualization color map
    link_types: list[str] = ["semantic", "emotional", "temporal", "causal", "contrastive"]
    links = [
        ResonanceLink(
            source_id=src,
            target_id=tgt,
            link_type=rng.choice(link_types),  # type: ignore[arg-type]
            strength=round(rng.uniform(0.3, 1.0), 4),
        )
        for i, src in enumerate(node_ids)
        for tgt in node_ids[i + 1 :]
        if rng.random() < 0.45
    ]

    labels = {f"m{i}": f"M{i}" for i in range(7)}
    fig = plot_resonance_network(links, labels, title="Resonance link graph snapshot")
    fig.savefig(out_dir / "fig4_resonance_network.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig4_resonance_network.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing figures to {out_dir}/")
    _make_circumplex(out_dir)
    _make_decay_curves(out_dir)
    _make_mood_evolution(out_dir)
    _make_resonance_network(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
