"""Addendum V — direct-VAD appraisal vs the SEC->projection (Hv1/Hv2/Hv3).

Tests whether asking the LLM for valence/arousal/dominance *directly* beats the
production path (5 Scherer SECs -> Addendum-O-recalibrated linear projection) on
human-gold affect, paired per item against EmoBank.

Both methods are ``LLMAppraisalEngine`` with a different ``AppraisalSchema`` — no
library change. Reuses the EmoBank harness helpers from
``benchmarks.human_gold_appraisal.runner``.

Pre-registration: ``benchmarks/preregistration_addendum_v_direct_vad.md``.

Usage::

    make bench-appraisal-vad                                  # full run (needs API key)
    uv run python -m benchmarks.appraisal_vad.runner --limit 4   # quick check
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

from benchmarks.common.statistics import DEFAULT_N_BOOTSTRAP
from benchmarks.human_gold_appraisal.runner import DIMENSIONS, _dimension_stats, _pearson
from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal_llm import LLMAppraisalConfig, LLMAppraisalEngine
from emotional_memory.appraisal_schema import AppraisalDimension, AppraisalSchema
from emotional_memory.llm_http import OpenAICompatibleLLMConfig, make_httpx_llm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "benchmarks" / "datasets" / "emobank_v1.json"
_HERE = Path(__file__).parent
DEFAULT_OUT_JSON = _HERE / "results.json"
DEFAULT_OUT_MD = _HERE / "results.md"

_VAD_PROMPT = """\
You are an affect rating system. Given a text, rate the emotion it expresses on three
dimensions and return ONLY a JSON object (no explanation, no markdown):
- valence   [-1, 1]  unpleasant/negative -> pleasant/positive
- arousal   [0, 1]   calm/subdued -> excited/activated
- dominance [0, 1]   controlled/submissive -> in-control/dominant
Return ONLY valid JSON with these exact keys.\
"""

VAD_SCHEMA = AppraisalSchema(
    name="direct_vad",
    dimensions=(
        AppraisalDimension(
            name="valence", range=(-1.0, 1.0), neutral=0.0, description="unpleasant -> pleasant"
        ),
        AppraisalDimension(
            name="arousal", range=(0.0, 1.0), neutral=0.5, description="calm -> excited"
        ),
        AppraisalDimension(
            name="dominance", range=(0.0, 1.0), neutral=0.5, description="submissive -> in-control"
        ),
    ),
    system_prompt=_VAD_PROMPT,
    project_to_core_affect=lambda d: CoreAffect(
        valence=d["valence"], arousal=d["arousal"], dominance=d["dominance"]
    ),
)


def _appraise_all(engine: Any, texts: list[str], *, label: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
    for text in tqdm(texts, desc=label, unit="text"):
        ca = engine.appraise(text).to_core_affect()
        out["valence"].append(ca.valence)
        out["arousal"].append(ca.arousal)
        out["dominance"].append(ca.dominance)
    return out


def _delta_r(
    pred_a: list[float], pred_b: list[float], human: list[float], *, n_bootstrap: int, seed: int
) -> dict[str, Any]:
    """Paired bootstrap on r(a) - r(b) over shared item indices (a = direct_vad, b = scherer)."""
    a = np.asarray(pred_a, dtype=np.float64)
    b = np.asarray(pred_b, dtype=np.float64)
    y = np.asarray(human, dtype=np.float64)
    point = _pearson(a, y) - _pearson(b, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_bootstrap, len(y)))
    boots = np.array([_pearson(a[i], y[i]) - _pearson(b[i], y[i]) for i in idx], dtype=np.float64)
    boots = boots[~np.isnan(boots)]
    if boots.size == 0:
        return {"delta_r": round(point, 4), "ci": [float("nan"), float("nan")]}
    return {
        "delta_r": round(point, 4),
        "ci": [
            round(float(np.quantile(boots, 0.025)), 4),
            round(float(np.quantile(boots, 0.975)), 4),
        ],
    }


def _write_predictions_dump(
    path: Path,
    *,
    items: list[dict[str, Any]],
    human: dict[str, list[float]],
    pred_scherer: dict[str, list[float]],
    pred_direct: dict[str, list[float]],
    dataset: dict[str, Any],
    seed: int,
) -> None:
    """Persist paired per-item predictions for downstream offline analysis (Addendum W).

    The Addendum V run is the only LLM pass; this dump lets deterministic studies
    (e.g. affine arousal calibration) run with no further LLM calls.
    """
    records = [
        {
            "id": it.get("id"),
            "split": it.get("split"),
            "human": {d: human[d][i] for d in DIMENSIONS},
            "scherer_m1": {d: pred_scherer[d][i] for d in DIMENSIONS},
            "direct_vad": {d: pred_direct[d][i] for d in DIMENSIONS},
        }
        for i, it in enumerate(items)
    ]
    payload = {
        "source_benchmark": "appraisal_direct_vad_v",
        "dataset": dataset["name"],
        "version": dataset["version"],
        "n": len(records),
        "seed": seed,
        "items": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(
    dataset: dict[str, Any],
    *,
    n_bootstrap: int,
    seed: int,
    dump_predictions: Path | None = None,
) -> dict[str, Any]:
    items = dataset["items"]
    texts = [it["text"] for it in items]
    human = {d: [it["human"][d] for it in items] for d in DIMENSIONS}

    cfg = OpenAICompatibleLLMConfig.from_env()
    if cfg is None:
        raise RuntimeError("EMOTIONAL_MEMORY_LLM_API_KEY not set — cannot run.")
    llm = make_httpx_llm(cfg)
    scherer = LLMAppraisalEngine(
        llm=llm, config=LLMAppraisalConfig(cache_size=0, fallback_on_error=True)
    )
    direct = LLMAppraisalEngine(
        llm=llm,
        config=LLMAppraisalConfig(
            cache_size=0, fallback_on_error=True, appraisal_schema=VAD_SCHEMA
        ),
    )

    pred_scherer = _appraise_all(scherer, texts, label="scherer_m1")
    pred_direct = _appraise_all(direct, texts, label="direct_vad")

    if dump_predictions is not None:
        _write_predictions_dump(
            dump_predictions,
            items=items,
            human=human,
            pred_scherer=pred_scherer,
            pred_direct=pred_direct,
            dataset=dataset,
            seed=seed,
        )

    stats = {
        "scherer_m1": {
            d: _dimension_stats(pred_scherer[d], human[d], n_bootstrap=n_bootstrap, seed=seed)
            for d in DIMENSIONS
        },
        "direct_vad": {
            d: _dimension_stats(pred_direct[d], human[d], n_bootstrap=n_bootstrap, seed=seed)
            for d in DIMENSIONS
        },
    }
    delta_r = {
        d: _delta_r(pred_direct[d], pred_scherer[d], human[d], n_bootstrap=n_bootstrap, seed=seed)
        for d in DIMENSIONS
    }

    def _ci_excludes_zero_pos(d: str) -> bool:
        return delta_r[d]["delta_r"] > 0 and delta_r[d]["ci"][0] > 0

    hv1 = _ci_excludes_zero_pos("arousal")
    hv2 = _ci_excludes_zero_pos("dominance")
    s_val_bias = abs(stats["scherer_m1"]["valence"]["bias"])
    d_val_bias = abs(stats["direct_vad"]["valence"]["bias"])
    hv3 = d_val_bias < s_val_bias
    gv = delta_r["valence"]["ci"][0] > -0.05  # no material valence regression
    adopt = bool((hv1 or hv2) and gv)

    return {
        "benchmark": "appraisal_direct_vad_v",
        "dataset": dataset["name"],
        "version": dataset["version"],
        "n": len(items),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "stats": stats,
        "paired_delta_r_directvad_minus_scherer": delta_r,
        "verdicts": {
            "Hv1_arousal_r_improved": hv1,
            "Hv2_dominance_r_improved": hv2,
            "Hv3_valence_bias_reduced": hv3,
            "Gv_valence_not_regressed": gv,
            "adopt_direct_vad": adopt,
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Addendum V — direct-VAD appraisal vs SEC->projection",
        "",
        f"Dataset: `{report['dataset']}` v{report['version']} (N={report['n']}, EmoBank human "
        f"VAD) · bootstrap n={report['n_bootstrap']} · seed={report['seed']}.",
        "",
        "## Per-dimension r / bias / MAE",
        "",
        "| Method | Dimension | Pearson r [95% CI] | bias | MAE |",
        "|---|---|---|---:|---:|",
    ]
    for method in ("scherer_m1", "direct_vad"):
        for d in DIMENSIONS:
            s = report["stats"][method][d]
            rc = s["pearson_ci"]
            lines.append(
                f"| `{method}` | {d} | {s['pearson_r']:+.3f} [{rc[0]:+.3f}, {rc[1]:+.3f}] | "
                f"{s['bias']:+.3f} | {s['mae']:.3f} |"
            )
    lines += [
        "",
        "## Paired Δr (direct_vad minus scherer_m1)",
        "",
        "| Dimension | Δr [95% CI] | improves? |",
        "|---|---|:---:|",
    ]
    for d in DIMENSIONS:
        dr = report["paired_delta_r_directvad_minus_scherer"][d]
        improves = "✅" if (dr["delta_r"] > 0 and dr["ci"][0] > 0) else "—"
        lines.append(
            f"| {d} | {dr['delta_r']:+.3f} [{dr['ci'][0]:+.3f}, {dr['ci'][1]:+.3f}] | {improves} |"
        )
    v = report["verdicts"]

    def _yn(flag: bool, yes: str, no: str) -> str:
        return yes if flag else no

    pf = {k: _yn(v[k], "✅ PASS", "✗ FAIL") for k in v}
    decision = _yn(v["adopt_direct_vad"], "✅ YES", "✗ NO (SEC->projection stands)")
    gv = _yn(v["Gv_valence_not_regressed"], "✅ OK", "✗ regressed")
    lines += [
        "",
        "## Verdicts",
        "",
        f"- **Hv1** (arousal r improved): {pf['Hv1_arousal_r_improved']}",
        f"- **Hv2** (dominance r improved): {pf['Hv2_dominance_r_improved']}",
        f"- **Hv3** (valence |bias| reduced): {pf['Hv3_valence_bias_reduced']}",
        f"- **Gv** (valence not regressed): {gv}",
        f"- **Decision — adopt direct-VAD:** {decision}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Addendum V — direct-VAD vs SEC->projection.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Cap rows (quick check).")
    parser.add_argument(
        "--dump-predictions",
        type=Path,
        default=None,
        help="Also write paired per-item predictions to this path (for Addendum W).",
    )
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.limit is not None:
        dataset = {**dataset, "items": dataset["items"][: args.limit], "n": args.limit}

    report = run(
        dataset,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        dump_predictions=args.dump_predictions,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out_md.write_text(_render_markdown(report), encoding="utf-8")
    v = report["verdicts"]
    print(
        f"Addendum V complete: adopt_direct_vad={v['adopt_direct_vad']} "
        f"(Hv1={v['Hv1_arousal_r_improved']} Hv2={v['Hv2_dominance_r_improved']} "
        f"Hv3={v['Hv3_valence_bias_reduced']} Gv={v['Gv_valence_not_regressed']})"
    )


if __name__ == "__main__":
    main()
