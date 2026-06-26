"""A5 construct validity — appraisal vs human-gold affect (Addendum S, Hs1).

Runs ``LLMAppraisalEngine`` (and ``KeywordAppraisalEngine`` as a rule-based
baseline) over the human-annotated EmoBank subset and reports, per affective
dimension (valence / arousal / dominance), the Pearson correlation, bias (mean
residual) and MAE against the human VAD labels — each with a bootstrap CI.

Pre-registration: ``benchmarks/preregistration_addendum_s_human_gold_appraisal.md``.
Dataset: ``benchmarks/datasets/emobank_v1.json`` (EmoBank, CC-BY-SA 4.0).

Usage::

    make bench-human-gold                                          # full run (needs API key)
    uv run python -m benchmarks.human_gold_appraisal.runner --dry-run   # keyword-only, no LLM
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from tqdm import tqdm

from benchmarks.common.statistics import DEFAULT_N_BOOTSTRAP, bootstrap_ci
from emotional_memory.appraisal_llm import (
    KeywordAppraisalEngine,
    LLMAppraisalConfig,
    LLMAppraisalEngine,
)
from emotional_memory.llm_http import OpenAICompatibleLLMConfig, make_httpx_llm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "benchmarks" / "datasets" / "emobank_v1.json"
_HERE = Path(__file__).parent
DEFAULT_OUT_JSON = _HERE / "results.json"
DEFAULT_OUT_MD = _HERE / "results.md"

DIMENSIONS = ("valence", "arousal", "dominance")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(x, y)
    return float(c[0, 1])


def _pearson_ci(
    pred: list[float], human: list[float], *, n_bootstrap: int, seed: int
) -> tuple[float, float, float]:
    x = np.asarray(pred, dtype=np.float64)
    y = np.asarray(human, dtype=np.float64)
    point = _pearson(x, y)
    n = len(x)
    if n < 2:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boots = np.array([_pearson(x[i], y[i]) for i in idx], dtype=np.float64)
    boots = boots[~np.isnan(boots)]
    if boots.size == 0:
        return (point, float("nan"), float("nan"))
    return (point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))


def _dimension_stats(
    pred: list[float], human: list[float], *, n_bootstrap: int, seed: int
) -> dict[str, Any]:
    residuals = [p - h for p, h in zip(pred, human, strict=True)]
    r, r_lo, r_hi = _pearson_ci(pred, human, n_bootstrap=n_bootstrap, seed=seed)
    bias, b_lo, b_hi = bootstrap_ci(residuals, n_bootstrap=n_bootstrap, seed=seed)
    mae = float(np.mean(np.abs(residuals))) if residuals else float("nan")
    return {
        "n": len(pred),
        "pearson_r": round(r, 4),
        "pearson_ci": [round(r_lo, 4), round(r_hi, 4)],
        "bias": round(bias, 4),
        "bias_ci": [round(b_lo, 4), round(b_hi, 4)],
        "mae": round(mae, 4),
        "human_validated": bool(r_lo > 0),
    }


def _appraise_all(engine: Any, texts: list[str], *, label: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
    for text in tqdm(texts, desc=label, unit="text"):
        ca = engine.appraise(text).to_core_affect()
        out["valence"].append(ca.valence)
        out["arousal"].append(ca.arousal)
        out["dominance"].append(ca.dominance)
    return out


def run(
    dataset: dict[str, Any],
    *,
    n_bootstrap: int,
    seed: int,
    use_llm: bool,
) -> dict[str, Any]:
    items = dataset["items"]
    texts = [it["text"] for it in items]
    human = {d: [it["human"][d] for it in items] for d in DIMENSIONS}

    engines: dict[str, Any] = {"keyword": KeywordAppraisalEngine()}
    if use_llm:
        cfg = OpenAICompatibleLLMConfig.from_env()
        if cfg is None:
            raise RuntimeError("EMOTIONAL_MEMORY_LLM_API_KEY not set — cannot run the llm engine.")
        engines["llm"] = LLMAppraisalEngine(
            llm=make_httpx_llm(cfg),
            config=LLMAppraisalConfig(cache_size=0, fallback_on_error=True),
        )

    systems: dict[str, Any] = {}
    for name, engine in engines.items():
        pred = _appraise_all(engine, texts, label=name)
        systems[name] = {
            d: _dimension_stats(pred[d], human[d], n_bootstrap=n_bootstrap, seed=seed)
            for d in DIMENSIONS
        }

    return {
        "benchmark": "human_gold_appraisal_a5",
        "dataset": dataset["name"],
        "version": dataset["version"],
        "license": dataset.get("license"),
        "source": dataset.get("source"),
        "n": len(items),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "llm_enabled": use_llm,
        "systems": systems,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# A5 — Appraisal vs human-gold affect (EmoBank, Addendum S)",
        "",
        f"Dataset: `{report['dataset']}` v{report['version']} (N={report['n']}, "
        f"{report['license']}) · bootstrap n={report['n_bootstrap']} · seed={report['seed']} · "
        f"llm={'on' if report['llm_enabled'] else 'OFF (keyword-only dry-run)'}.",
        "",
        f"Source: {report['source']}",
        "",
    ]
    for name, dims in report["systems"].items():
        lines += [
            f"## `{name}` engine",
            "",
            "| Dimension | N | Pearson r [95% CI] | bias [95% CI] | MAE | human-validated |",
            "|---|---:|---|---|---:|:---:|",
        ]
        for d in DIMENSIONS:
            s = dims[d]
            rc = s["pearson_ci"]
            bc = s["bias_ci"]
            hv = "✅" if s["human_validated"] else "✗"
            lines.append(
                f"| {d} | {s['n']} | {s['pearson_r']:+.3f} [{rc[0]:+.3f}, {rc[1]:+.3f}] | "
                f"{s['bias']:+.3f} [{bc[0]:+.3f}, {bc[1]:+.3f}] | {s['mae']:.3f} | {hv} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="A5 appraisal vs human-gold (Addendum S).")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Cap rows (smoke test).")
    parser.add_argument(
        "--dry-run", action="store_true", help="Keyword engine only; no LLM calls."
    )
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.limit is not None:
        dataset = {**dataset, "items": dataset["items"][: args.limit], "n": args.limit}

    report = run(
        dataset,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        use_llm=not args.dry_run,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out_md.write_text(_render_markdown(report), encoding="utf-8")
    if "llm" in report["systems"]:
        v = report["systems"]["llm"]["valence"]
        print(f"A5 complete: llm valence r={v['pearson_r']} CI={v['pearson_ci']} bias={v['bias']}")
    else:
        print("A5 dry-run complete (keyword engine only).")


if __name__ == "__main__":
    main()
