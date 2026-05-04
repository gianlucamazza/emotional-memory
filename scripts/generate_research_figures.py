"""Generate research-evidence figures from committed benchmark JSON files.

Usage::

    uv run python scripts/generate_research_figures.py
    uv run python scripts/generate_research_figures.py \
        --png-dir docs/images/research --pdf-dir paper/figures

The figures are intentionally data-driven: they read the benchmark artefacts
already committed under ``benchmarks/`` and do not rerun long studies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    print(f"matplotlib not installed: {exc}\nRun: pip install emotional-memory[viz]")
    sys.exit(1)

REALISTIC_SBERT = ROOT / "benchmarks" / "realistic" / "results.v2.sbert.json"
REALISTIC_IT_SBERT = ROOT / "benchmarks" / "realistic" / "results.v2_it.sbert.json"
REALISTIC_IT_ME5 = ROOT / "benchmarks" / "realistic" / "results.v2_it.me5.json"
ABLATION_SBERT = ROOT / "benchmarks" / "ablation" / "results.v2.sbert.json"
LOCOMO = ROOT / "benchmarks" / "locomo" / "results.json"

SYSTEM_COLORS = {
    "aft": "#4C72B0",
    "naive_cosine": "#DD5555",
    "recency": "#55A868",
    "naive_rag": "#DD5555",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        raise FileNotFoundError(f"Required benchmark artefact is missing: {rel}")
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected object in {path.relative_to(ROOT)}")
    return data


def _ci(metric: dict[str, Any], key: str) -> tuple[float, float, float]:
    point = float(metric[key])
    ci = metric["ci"][key]
    return point, float(ci["ci_lower"]), float(ci["ci_upper"])


def _errorbar_parts(
    rows: list[tuple[str, float, float, float]],
) -> tuple[list[float], list[float]]:
    lower = [point - lo for _, point, lo, _ in rows]
    upper = [hi - point for _, point, _, hi in rows]
    return lower, upper


def _save(fig: object, png_dir: Path, pdf_dir: Path, stem: str) -> None:
    import matplotlib.figure

    if not isinstance(fig, matplotlib.figure.Figure):
        raise TypeError(f"Expected Figure, got {type(fig)}")

    png_path = png_dir / f"{stem}.png"
    pdf_path = pdf_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    png_label = png_path.relative_to(ROOT) if png_path.is_relative_to(ROOT) else png_path
    pdf_label = pdf_path.relative_to(ROOT) if pdf_path.is_relative_to(ROOT) else pdf_path
    print(f"  {png_label}")
    print(f"  {pdf_label}")


def _style_axis(ax: Any, *, ylabel: str, title: str) -> None:
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _figure_realistic_overview(data: dict[str, Any]) -> object:
    rows: list[tuple[str, float, float, float]] = []
    for system in data["systems"]:
        name = str(system["system"])
        metric = system["aggregate_metrics"]
        rows.append((name, *_ci(metric, "top1_accuracy")))

    labels = [name.replace("_", "\n") for name, *_ in rows]
    values = [point for _, point, _, _ in rows]
    colors = [SYSTEM_COLORS.get(name, "#777777") for name, *_ in rows]
    lower, upper = _errorbar_parts(rows)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(labels, values, color=colors, alpha=0.88)
    ax.errorbar(labels, values, yerr=[lower, upper], fmt="none", ecolor="#222222", capsize=4)
    _style_axis(ax, ylabel="Top-1 accuracy", title="Realistic replay v2 (SBERT, N=200)")
    ax.text(0.01, -0.22, "Error bars: 95% bootstrap CI", transform=ax.transAxes, fontsize=8)
    return fig


def _figure_challenge_breakdown(data: dict[str, Any]) -> object:
    systems = {
        str(s["system"]): s for s in data["systems"] if s["system"] in {"aft", "naive_cosine"}
    }
    challenge_names = [m["challenge_type"] for m in systems["aft"]["challenge_type_metrics"]]
    x = np.arange(len(challenge_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    for offset, name in [(-width / 2, "aft"), (width / 2, "naive_cosine")]:
        metrics = systems[name]["challenge_type_metrics"]
        values = [float(m["top1_accuracy"]) for m in metrics]
        ax.bar(
            x + offset,
            values,
            width=width,
            label=name.replace("_", " "),
            color=SYSTEM_COLORS[name],
            alpha=0.88,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in challenge_names], fontsize=8)
    _style_axis(ax, ylabel="Top-1 accuracy", title="Realistic replay v2 by challenge type")
    ax.legend(frameon=False, ncol=2)
    return fig


def _figure_ablation_forest(data: dict[str, Any]) -> object:
    rows = [row for row in data["pairwise_vs_full"] if row["variant"] not in {"no_appraisal"}]
    labels = [str(row["variant"]).replace("_", "\n") for row in rows]
    diffs = [float(row["top1"]["diff"]) for row in rows]
    lower = [diff - float(row["top1"]["ci_lower"]) for diff, row in zip(diffs, rows, strict=False)]
    upper = [float(row["top1"]["ci_upper"]) - diff for diff, row in zip(diffs, rows, strict=False)]
    colors = ["#DD5555" if diff < 0 else "#4C72B0" for diff in diffs]

    fig, ax = plt.subplots(figsize=(9.0, 4.7))
    x = np.arange(len(rows))
    ax.bar(x, diffs, color=colors, alpha=0.86)
    ax.errorbar(x, diffs, yerr=[lower, upper], fmt="none", ecolor="#222222", capsize=4)
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Delta top-1 vs full AFT")
    ax.set_title("S3 ablation: individual layers are not isolatable")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.01,
        -0.24,
        "Negative delta means the ablation hurt performance",
        transform=ax.transAxes,
        fontsize=8,
    )
    return fig


def _figure_multilingual(sbert: dict[str, Any], me5: dict[str, Any]) -> object:
    datasets = [("SBERT EN-only", sbert), ("multilingual-e5", me5)]
    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for offset, system_name in [(-width / 2, "aft"), (width / 2, "naive_cosine")]:
        rows: list[tuple[str, float, float, float]] = []
        for label, data in datasets:
            system = next(s for s in data["systems"] if s["system"] == system_name)
            rows.append((label, *_ci(system["aggregate_metrics"], "hit_at_k")))
        values = [point for _, point, _, _ in rows]
        lower, upper = _errorbar_parts(rows)
        ax.bar(
            x + offset,
            values,
            width=width,
            label=system_name.replace("_", " "),
            color=SYSTEM_COLORS[system_name],
            alpha=0.88,
        )
        ax.errorbar(
            x + offset,
            values,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#222222",
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in datasets])
    _style_axis(
        ax,
        ylabel="Hit@k",
        title="Italian slice: signal survives multilingual embedder swap",
    )
    ax.legend(frameon=False, ncol=2)
    return fig


def _figure_locomo(data: dict[str, Any]) -> object:
    tests = data["hypothesis_tests"]
    rows = [
        ("F1", tests["H1"]["aft"], tests["H1"]["baseline"]),
        ("Judge accuracy", tests["H2"]["aft"], tests["H2"]["baseline"]),
    ]
    x = np.arange(len(rows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for offset, label, key in [(-width / 2, "AFT", "aft"), (width / 2, "naive RAG", "baseline")]:
        metric_idx = 1 if key == "aft" else 2
        values = [float(row[metric_idx]["point"]) for row in rows]
        lower = [
            v - float(row[metric_idx]["ci_lower"]) for v, row in zip(values, rows, strict=False)
        ]
        upper = [
            float(row[metric_idx]["ci_upper"]) - v for v, row in zip(values, rows, strict=False)
        ]
        ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=SYSTEM_COLORS["aft" if key == "aft" else "naive_rag"],
            alpha=0.88,
        )
        ax.errorbar(
            x + offset,
            values,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#222222",
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([row[0] for row in rows])
    _style_axis(ax, ylabel="Score", title="LoCoMo external QA: Gate 1 FAIL")
    ax.legend(frameon=False, ncol=2)
    ax.text(
        0.01,
        -0.22,
        "Affective weighting underperforms on factual open-domain QA",
        transform=ax.transAxes,
        fontsize=8,
    )
    return fig


def generate(png_dir: Path, pdf_dir: Path) -> None:
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    realistic = _load_json(REALISTIC_SBERT)
    realistic_it_sbert = _load_json(REALISTIC_IT_SBERT)
    realistic_it_me5 = _load_json(REALISTIC_IT_ME5)
    ablation = _load_json(ABLATION_SBERT)
    locomo = _load_json(LOCOMO)

    print("Generating research figures ...")
    _save(
        _figure_realistic_overview(realistic),
        png_dir,
        pdf_dir,
        "research_realistic_v2_overview",
    )
    _save(
        _figure_challenge_breakdown(realistic),
        png_dir,
        pdf_dir,
        "research_realistic_v2_challenges",
    )
    _save(_figure_ablation_forest(ablation), png_dir, pdf_dir, "research_ablation_s3")
    _save(
        _figure_multilingual(realistic_it_sbert, realistic_it_me5),
        png_dir,
        pdf_dir,
        "research_multilingual_it",
    )
    _save(_figure_locomo(locomo), png_dir, pdf_dir, "research_locomo_negative")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate research evidence figures.")
    parser.add_argument("--png-dir", default=str(ROOT / "docs" / "images" / "research"))
    parser.add_argument("--pdf-dir", default=str(ROOT / "paper" / "figures"))
    args = parser.parse_args()

    generate(Path(args.png_dir), Path(args.pdf_dir))


if __name__ == "__main__":
    main()
