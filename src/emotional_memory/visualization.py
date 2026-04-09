"""Visualization utilities for Affective Field Theory components.

All functions follow the matplotlib ``ax`` pattern: pass an existing
``Axes`` to embed in a larger figure, or omit it to get a standalone figure.
Each function returns the ``Figure`` it drew on.

Requires the optional ``viz`` extra::

    pip install emotional_memory[viz]
    # or
    from emotional_memory.visualization import plot_circumplex

Theory references preserved in individual function docstrings.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Color / style constants
# ---------------------------------------------------------------------------

LINK_TYPE_COLORS: dict[str, str] = {
    "semantic": "#4C72B0",
    "emotional": "#DD5555",
    "temporal": "#55A868",
    "causal": "#C4A000",
    "contrastive": "#8172B3",
}

SIGNAL_LABELS: list[str] = [
    "Semantic",
    "Stimmung",
    "Affect",
    "Momentum",
    "Recency",
    "Resonance",
]

APPRAISAL_LABELS: list[str] = [
    "Novelty",
    "Goal Relevance",
    "Coping Potential",
    "Norm Congruence",
    "Self Relevance",
]

# Stimmung baselines (from StimmungDecayConfig defaults)
_STIMMUNG_BASELINES: dict[str, float] = {
    "Valence": 0.0,
    "Arousal": 0.3,
    "Dominance": 0.5,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_matplotlib() -> None:
    """Raise ImportError with install hint if matplotlib is not installed."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install 'emotional_memory[viz]'"
        ) from None


def _make_figure(
    ax: Axes | None,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> tuple[Figure, Axes]:
    """Return (fig, ax): create new figure if ``ax`` is None, reuse otherwise."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure as MplFigure

    if ax is None:
        fig, new_ax = plt.subplots(figsize=figsize)
        assert isinstance(fig, MplFigure)
        return fig, new_ax
    parent = ax.get_figure()
    assert isinstance(parent, MplFigure)
    return parent, ax


# ---------------------------------------------------------------------------
# 1. Valence-Arousal Circumplex (Russell 1980)
# ---------------------------------------------------------------------------


def plot_circumplex(
    memories: Sequence[tuple[float, float, float]],
    *,
    ax: Axes | None = None,
    title: str = "Valence-Arousal Circumplex",
    cmap: str = "RdYlGn",
) -> Figure:
    """Plot memories on Russell's (1980) valence-arousal circumplex.

    Args:
        memories: Sequence of ``(valence, arousal, consolidation_strength)``
            tuples. ``valence`` ∈ [-1, +1], ``arousal`` ∈ [0, 1],
            ``consolidation_strength`` ∈ [0, 1] (used for color).
        ax: Optional existing ``Axes`` to draw on.
        title: Plot title.
        cmap: Matplotlib colormap name for consolidation strength.

    Returns:
        The ``Figure`` containing the plot.
    """
    _require_matplotlib()

    fig, _ax = _make_figure(ax, figsize=(7.0, 6.0))

    # Light reference lines
    _ax.axvline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    _ax.axhline(0.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)

    if memories:
        vs = [m[0] for m in memories]
        aros = [m[1] for m in memories]
        cs = [m[2] for m in memories]
        sc = _ax.scatter(
            vs,
            aros,
            c=cs,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=80,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        cbar = fig.colorbar(sc, ax=_ax, shrink=0.8)
        cbar.set_label("Consolidation strength", fontsize=9)

    # Quadrant labels
    label_kw: dict[str, object] = dict(
        fontsize=8, color="gray", ha="center", va="center", style="italic"
    )
    _ax.text(-0.7, 0.85, "Tense / Distressed", **label_kw)  # type: ignore[arg-type]
    _ax.text(0.7, 0.85, "Excited / Elated", **label_kw)  # type: ignore[arg-type]
    _ax.text(-0.7, 0.15, "Sad / Depressed", **label_kw)  # type: ignore[arg-type]
    _ax.text(0.7, 0.15, "Calm / Relaxed", **label_kw)  # type: ignore[arg-type]

    _ax.set_xlim(-1.0, 1.0)
    _ax.set_ylim(0.0, 1.0)
    _ax.set_xlabel("Valence  (negative ← → positive)")
    _ax.set_ylabel("Arousal  (calm ← → activated)")
    _ax.set_title(title)

    return fig


# ---------------------------------------------------------------------------
# 2. Decay Curves — ACT-R power-law (Anderson 1983, McGaugh 2004)
# ---------------------------------------------------------------------------


def plot_decay_curves(
    *,
    base_decay: float = 0.5,
    arousal_modulation: float = 0.5,
    retrieval_boost: float = 0.1,
    floor_arousal_threshold: float = 0.7,
    floor_value: float = 0.1,
    arousal_values: Sequence[float] = (0.0, 0.5, 1.0),
    retrieval_counts: Sequence[int] = (0, 5, 10),
    initial_strength: float = 1.0,
    ax: Axes | None = None,
    title: str = "Memory Decay — ACT-R Power Law",
) -> Figure:
    """Plot families of ACT-R power-law decay curves.

    Shows the McGaugh (2004) emotional enhancement effect (high arousal
    slows decay) and the spacing effect (retrieval count slows decay).

    Formula::

        eff_decay = base_decay
                  * (1 - arousal_modulation * arousal)
                  * (1 / (1 + retrieval_boost * retrieval_count))
        strength(t) = initial * t ^ (-eff_decay)

    Args:
        base_decay: Base power-law exponent.
        arousal_modulation: How much arousal slows decay.
        retrieval_boost: Decay reduction per retrieval event.
        floor_arousal_threshold: Arousal above which floor applies.
        floor_value: Minimum strength for high-arousal memories.
        arousal_values: Arousal levels to plot (colors vary).
        retrieval_counts: Retrieval counts to plot (linestyles vary).
        initial_strength: Starting consolidation strength.
        ax: Optional existing ``Axes``.
        title: Plot title.

    Returns:
        The ``Figure``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    fig, _ax = _make_figure(ax, figsize=(8.0, 5.0))

    t = np.linspace(1, 86_400, 600)  # 1 second → 24 hours

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(arousal_values)))  # type: ignore[attr-defined]
    linestyles = ["-", "--", ":"][: len(retrieval_counts)]
    if len(retrieval_counts) > 3:
        linestyles += ["-."] * (len(retrieval_counts) - 3)

    for i, aro in enumerate(arousal_values):
        for j, rc in enumerate(retrieval_counts):
            eff = (
                base_decay
                * (1.0 - arousal_modulation * aro)
                * (1.0 / (1.0 + retrieval_boost * rc))
            )
            eff = max(eff, 0.0)
            strength = initial_strength * (t ** (-eff))
            strength = np.clip(strength, 0.0, 1.0)
            if aro >= floor_arousal_threshold:
                strength = np.maximum(strength, floor_value)

            label = f"arousal={aro:.1f}, retrievals={rc}"
            _ax.plot(
                t / 3600,
                strength,
                color=colors[i],
                linestyle=linestyles[j],
                linewidth=1.5,
                label=label,
            )

    _ax.set_xlabel("Elapsed time (hours)")
    _ax.set_ylabel("Consolidation strength")
    _ax.set_title(title)
    _ax.legend(fontsize=7, ncol=2)
    _ax.set_ylim(0.0, 1.05)
    _ax.set_xlim(left=0)

    return fig


# ---------------------------------------------------------------------------
# 3. Yerkes-Dodson Inverted-U (McGaugh 2004, Brown & Kulik 1977)
# ---------------------------------------------------------------------------


def plot_yerkes_dodson(
    *,
    stimmung_arousal: float = 0.3,
    ax: Axes | None = None,
    title: str = "Yerkes-Dodson: Consolidation vs Arousal",
) -> Figure:
    """Plot the Yerkes-Dodson inverted-U for memory consolidation strength.

    Uses ``consolidation_strength(arousal, stimmung_arousal)`` directly —
    the same function the engine calls at encoding time. Peak is near
    effective_arousal ≈ 0.7 (blend of encoding + Stimmung arousal).

    Args:
        stimmung_arousal: Background Stimmung arousal (default 0.3).
        ax: Optional existing ``Axes``.
        title: Plot title.

    Returns:
        The ``Figure``.
    """
    _require_matplotlib()
    import numpy as np

    from emotional_memory.appraisal import consolidation_strength

    fig, _ax = _make_figure(ax, figsize=(6.0, 4.5))

    aro = np.linspace(0.0, 1.0, 200)
    cs = np.array([consolidation_strength(float(a), stimmung_arousal) for a in aro])

    _ax.plot(aro, cs, color="#DD5555", linewidth=2.2)
    _ax.fill_between(aro, cs, alpha=0.15, color="#DD5555")

    # Mark peak
    peak_idx = int(cs.argmax())
    _ax.axvline(aro[peak_idx], color="gray", linewidth=0.8, linestyle="--")
    _ax.annotate(
        f"peak ≈ {aro[peak_idx]:.2f}",
        xy=(aro[peak_idx], cs[peak_idx]),
        xytext=(aro[peak_idx] + 0.05, cs[peak_idx] - 0.08),
        fontsize=8,
        color="gray",
    )

    _ax.set_xlabel("Encoding arousal")
    _ax.set_ylabel("Consolidation strength")
    _ax.set_title(title)
    _ax.set_xlim(0.0, 1.0)
    _ax.set_ylim(0.0, 1.05)
    _ax.text(
        0.02,
        0.03,
        f"Stimmung arousal = {stimmung_arousal:.2f}",
        transform=_ax.transAxes,
        fontsize=8,
        color="gray",
    )

    return fig


# ---------------------------------------------------------------------------
# 4. Retrieval Signal Radar (6 signals)
# ---------------------------------------------------------------------------


def plot_retrieval_radar(
    scores: Sequence[float],
    labels: Sequence[str] = SIGNAL_LABELS,
    *,
    ax: Axes | None = None,
    title: str = "Retrieval Signal Breakdown",
    color: str = "#4C72B0",
) -> Figure:
    """Radar chart showing the six retrieval signal scores.

    Args:
        scores: Six values in [0, 1] corresponding to the six signals:
            [semantic, stimmung, affect, momentum, recency, resonance].
        labels: Spoke labels (must match length of ``scores``).
        ax: Optional existing *polar* ``Axes``.
        title: Plot title.
        color: Fill color.

    Returns:
        The ``Figure``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(scores)
    angles = np.linspace(0.0, 2 * math.pi, n, endpoint=False).tolist()
    angles_closed = [*angles, angles[0]]
    values_closed = [*list(scores), scores[0]]

    if ax is None:
        fig, _ax = plt.subplots(figsize=(6.0, 6.0), subplot_kw={"polar": True})
    else:
        fig = ax.get_figure()  # type: ignore[assignment]
        _ax = ax

    _ax.plot(angles_closed, values_closed, color=color, linewidth=2.0)
    _ax.fill(angles_closed, values_closed, alpha=0.25, color=color)

    _ax.set_xticks(angles)
    _ax.set_xticklabels(list(labels), fontsize=9)
    _ax.set_ylim(0.0, 1.0)
    _ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    _ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    _ax.set_title(title, pad=15)

    return fig


# ---------------------------------------------------------------------------
# 5. Stimmung Evolution Time Series
# ---------------------------------------------------------------------------


def plot_stimmung_evolution(
    history: Sequence[tuple[float, float, float, float]],
    *,
    ax: Axes | None = None,
    title: str = "Stimmung Field Evolution",
) -> Figure:
    """Plot valence, arousal, and dominance over time.

    Args:
        history: Sequence of ``(t_seconds, valence, arousal, dominance)``
            tuples sorted chronologically.
        ax: Optional existing ``Axes``.
        title: Plot title.

    Returns:
        The ``Figure``.
    """
    _require_matplotlib()

    fig, _ax = _make_figure(ax, figsize=(9.0, 4.5))

    if not history:
        _ax.set_title(title)
        return fig

    ts = [h[0] / 60.0 for h in history]  # → minutes
    vals = [h[1] for h in history]
    aros = [h[2] for h in history]
    doms = [h[3] for h in history]

    _ax.plot(ts, vals, color="#4C72B0", linewidth=2.0, label="Valence")
    _ax.plot(ts, aros, color="#DD5555", linewidth=2.0, label="Arousal")
    _ax.plot(ts, doms, color="#55A868", linewidth=2.0, label="Dominance")

    # Baseline attractors
    baseline_colors = {"Valence": "#4C72B0", "Arousal": "#DD5555", "Dominance": "#55A868"}
    for name, bv in _STIMMUNG_BASELINES.items():
        _ax.axhline(
            bv,
            color=baseline_colors[name],
            linewidth=0.8,
            linestyle=":",
            alpha=0.6,
            label=f"{name} baseline",
        )

    _ax.set_xlabel("Time (minutes)")
    _ax.set_ylabel("PAD value")
    _ax.set_title(title)
    _ax.legend(fontsize=8, ncol=2)
    _ax.set_ylim(-1.05, 1.05)

    return fig


# ---------------------------------------------------------------------------
# 6. Adaptive Weights Heatmap
# ---------------------------------------------------------------------------


def plot_adaptive_weights_heatmap(
    *,
    base_weights: Sequence[float] = (0.35, 0.25, 0.15, 0.10, 0.10, 0.05),
    resolution: int = 20,
    ax: Axes | None = None,
    title: str = "Adaptive Retrieval Weights vs Stimmung",
) -> Figure:
    """Heatmap of how each retrieval weight shifts across Stimmung states.

    Sweeps valence ∈ [-1, +1] and arousal ∈ [0, 1] on a grid
    (dominance fixed at 0.5). For each grid cell calls
    ``adaptive_weights()`` and displays a 6-panel heatmap.

    Args:
        base_weights: Six baseline retrieval weights (must sum to 1.0).
        resolution: Grid resolution (``resolution x resolution`` cells).
        ax: Unused — this plot always creates its own figure with subplots.
        title: Figure suptitle.

    Returns:
        The ``Figure``.
    """
    _require_matplotlib()
    from datetime import UTC, datetime

    import matplotlib.pyplot as plt
    import numpy as np

    from emotional_memory.retrieval import AdaptiveWeightsConfig, adaptive_weights
    from emotional_memory.stimmung import StimmungField

    valences = np.linspace(-1.0, 1.0, resolution)
    arousals = np.linspace(0.0, 1.0, resolution)
    base = list(base_weights)
    cfg = AdaptiveWeightsConfig()

    # shape: (n_signals, n_arousal, n_valence)
    grid = np.zeros((6, resolution, resolution))
    _ts = datetime.now(tz=UTC)

    for i, aro in enumerate(arousals):
        for j, val in enumerate(valences):
            sf = StimmungField(
                valence=float(val),
                arousal=float(aro),
                dominance=0.5,
                inertia=0.3,
                timestamp=_ts,
            )
            w = adaptive_weights(sf, base, cfg)
            for k in range(6):
                grid[k, i, j] = float(w[k])

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0))
    fig.suptitle(title, fontsize=12)

    for k, label in enumerate(SIGNAL_LABELS):
        row, col = divmod(k, 3)
        _a = axes[row][col]
        im = _a.imshow(
            grid[k],
            origin="lower",
            extent=[-1.0, 1.0, 0.0, 1.0],
            aspect="auto",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=0.55,
        )
        _a.set_title(label, fontsize=10)
        _a.set_xlabel("Valence", fontsize=8)
        _a.set_ylabel("Arousal", fontsize=8)
        fig.colorbar(im, ax=_a, shrink=0.8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Resonance Network Graph
# ---------------------------------------------------------------------------


def plot_resonance_network(
    links: Sequence[object],
    node_labels: dict[str, str] | None = None,
    *,
    ax: Axes | None = None,
    title: str = "Resonance Network",
) -> Figure:
    """Draw the resonance graph with circular node layout.

    Does not require networkx — uses a simple circular layout.

    Args:
        links: Sequence of ``ResonanceLink`` model instances.
        node_labels: Optional mapping ``{memory_id: display_label}``.
        ax: Optional existing ``Axes``.
        title: Plot title.

    Returns:
        The ``Figure``.
    """
    _require_matplotlib()
    import matplotlib.patches as mpatches
    import numpy as np

    fig, _ax = _make_figure(ax, figsize=(8.0, 8.0))
    _ax.set_aspect("equal")
    _ax.axis("off")
    _ax.set_title(title)

    from emotional_memory.models import ResonanceLink as RL

    typed_links: list[RL] = []
    for lnk in links:
        if isinstance(lnk, RL):
            typed_links.append(lnk)

    if not typed_links:
        return fig

    # Collect unique node IDs
    node_ids: list[str] = []
    seen: set[str] = set()
    for lnk in typed_links:
        for nid in (lnk.source_id, lnk.target_id):
            if nid not in seen:
                node_ids.append(nid)
                seen.add(nid)

    n_nodes = len(node_ids)
    angles = np.linspace(0, 2 * math.pi, n_nodes, endpoint=False)
    positions = {nid: (math.cos(a), math.sin(a)) for nid, a in zip(node_ids, angles, strict=False)}

    # Draw edges
    for lnk in typed_links:
        sx, sy = positions[lnk.source_id]
        tx, ty = positions[lnk.target_id]
        color = LINK_TYPE_COLORS.get(lnk.link_type, "#888888")
        _ax.annotate(
            "",
            xy=(tx, ty),
            xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.0 + lnk.strength * 2.0,
                alpha=0.7,
                connectionstyle="arc3,rad=0.1",
            ),
        )

    # Draw nodes
    for nid, (nx_, ny_) in positions.items():
        _ax.plot(nx_, ny_, "o", color="#444444", markersize=10, zorder=4)
        label = (node_labels or {}).get(nid, nid[:6])
        _ax.text(nx_ * 1.15, ny_ * 1.15, label, ha="center", va="center", fontsize=8)

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=lt) for lt, c in LINK_TYPE_COLORS.items()]
    _ax.legend(handles=legend_patches, loc="lower right", fontsize=8, title="Link type")

    _ax.set_xlim(-1.5, 1.5)
    _ax.set_ylim(-1.5, 1.5)
    return fig


# ---------------------------------------------------------------------------
# 8. Appraisal Radar Chart (Scherer CPM, 5 dimensions)
# ---------------------------------------------------------------------------


def plot_appraisal_radar(
    appraisal: object,
    *,
    ax: Axes | None = None,
    title: str = "Appraisal Vector (Scherer CPM)",
    color: str = "#DD5555",
) -> Figure:
    """Spider chart for the five Scherer Stimulus Evaluation Check dimensions.

    Args:
        appraisal: An ``AppraisalVector`` instance *or* a plain dict with
            keys: ``novelty``, ``goal_relevance``, ``coping_potential``,
            ``norm_congruence``, ``self_relevance``.
        ax: Optional existing *polar* ``Axes``.
        title: Plot title.
        color: Fill / line color.

    Returns:
        The ``Figure``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    from emotional_memory.appraisal import AppraisalVector

    if isinstance(appraisal, AppraisalVector):
        raw: dict[str, float] = {
            "novelty": appraisal.novelty,
            "goal_relevance": appraisal.goal_relevance,
            "coping_potential": appraisal.coping_potential,
            "norm_congruence": appraisal.norm_congruence,
            "self_relevance": appraisal.self_relevance,
        }
    elif isinstance(appraisal, dict):
        raw = {str(k): float(v) for k, v in appraisal.items()}
    else:
        raise TypeError(f"appraisal must be AppraisalVector or dict, got {type(appraisal)}")

    # Normalise all dimensions to [0, 1] for the radar
    # novelty/goal_relevance/norm_congruence are [-1,+1] → shift by 0.5
    def _norm(key: str, val: float) -> float:
        if key in ("novelty", "goal_relevance", "norm_congruence"):
            return (val + 1.0) / 2.0
        return val  # already [0,1]

    keys = ["novelty", "goal_relevance", "coping_potential", "norm_congruence", "self_relevance"]
    values = [_norm(k, raw.get(k, 0.0)) for k in keys]

    n = len(values)
    angles = np.linspace(0.0, 2 * math.pi, n, endpoint=False).tolist()
    angles_closed = [*angles, angles[0]]
    values_closed = [*values, values[0]]

    if ax is None:
        fig, _ax = plt.subplots(figsize=(6.0, 6.0), subplot_kw={"polar": True})
    else:
        fig = ax.get_figure()  # type: ignore[assignment]
        _ax = ax

    _ax.plot(angles_closed, values_closed, color=color, linewidth=2.0)
    _ax.fill(angles_closed, values_closed, alpha=0.25, color=color)

    _ax.set_xticks(angles)
    _ax.set_xticklabels(APPRAISAL_LABELS, fontsize=9)
    _ax.set_ylim(0.0, 1.0)
    _ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    _ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    _ax.set_title(title, pad=15)

    return fig
