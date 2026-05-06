"""Addendum I — Hi3 confirmatory analysis: resonance amplification e5 vs SBERT (Hi3).

Pre-registration: benchmarks/preregistration_addendum_i.md
Dataset:          benchmarks/datasets/realistic_recall_v3.json  (N=500, seed=1)
Inputs:           benchmarks/ablation/results.v3.{sbert,e5}.json
Output:           benchmarks/ablation/results.hi3.{json,md,protocol.json}

Hi3 family (m=3, Holm-Bonferroni):
  Hi3          — semantic_confound amplification: e5 > SBERT by +0.05 (primary)
  Hi3_recency  — recency_confound  amplification: e5 > SBERT by +0.05 (secondary)
  Hi3_arc      — affective_arc     amplification: e5 > SBERT by +0.05 (secondary)

Statistic for each challenge type:
  amp_e5[i]    = no_resonance_top1(e5)[i]   - full_top1(e5)[i]
  amp_sbert[i] = no_resonance_top1(sbert)[i] - full_top1(sbert)[i]
  delta        = mean(amp_e5) - mean(amp_sbert)   [paired bootstrap, n=10_000, seed=1]

Sign convention (matches pre-reg §Hi3):
  amp > 0 means resonance *hurts* that embedder on that query.
  delta > 0 means e5 is hurt more by resonance than SBERT (amplification of interference).

PASS iff delta > 0.05  AND  p_adj (Holm) < 0.05  (one-tailed, directional).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.common.statistics import (
    cohens_d_paired,
    format_point_ci,
    holm_bonferroni,
    paired_bootstrap_diff,
)

_HERE = Path(__file__).resolve().parent

DEFAULT_RESULTS_SBERT = _HERE / "results.v3.sbert.json"
DEFAULT_RESULTS_E5 = _HERE / "results.v3.e5.json"
DEFAULT_OUT_JSON = _HERE / "results.hi3.json"
DEFAULT_OUT_MD = _HERE / "results.hi3.md"
DEFAULT_OUT_PROTOCOL = _HERE / "results.hi3.protocol.json"

_PREREG_SEED = 1
_PREREG_N_BOOTSTRAP = 10_000
_HI3_ALPHA = 0.05
_HI3_EFFECT_THRESHOLD = 0.05

# Confirmatory family (m=3) — Holm correction applied over one-sided p-values
CHALLENGE_FAMILY: tuple[str, ...] = (
    "semantic_confound",
    "recency_confound",
    "affective_arc",
)
HYPOTHESIS_NAMES: dict[str, str] = {
    "semantic_confound": "Hi3",
    "recency_confound": "Hi3_recency",
    "affective_arc": "Hi3_arc",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_per_query_records(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Return data["per_query_records"] from a runner JSON output file.

    Raises RuntimeError with a diagnostic message if the field is missing
    (e.g., runner was invoked without --per-query-records).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if "per_query_records" not in data:
        raise RuntimeError(
            f"{path} does not contain 'per_query_records'.\n"
            "Re-run the ablation runner with --per-query-records flag."
        )
    raw: dict[str, Any] = data["per_query_records"]
    return raw


def _load_link_set_stats(path: Path) -> dict[str, Any] | None:
    """Return data["link_set_stats"] if present (for mechanism analysis)."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, Any] | None = data.get("link_set_stats")
    return result


# ---------------------------------------------------------------------------
# Paired amplitude vectors
# ---------------------------------------------------------------------------


def _build_amp_vector(
    records: dict[str, list[dict[str, Any]]],
    *,
    challenge_type: str,
) -> tuple[list[str], list[float]]:
    """Build (query_ids, amp) for one challenge type and one embedder run.

    amp[i] = no_resonance_top1_hit[i] - full_top1_hit[i]
    Positive when resonance hurts for that query (pre-reg sign convention).
    Records are keyed by variant name; each list is sorted by query_id.
    """
    full_by_id = {r["query_id"]: r for r in records.get("full", [])}
    no_res_by_id = {r["query_id"]: r for r in records.get("no_resonance", [])}

    shared_ids = sorted(
        qid
        for qid in full_by_id
        if qid in no_res_by_id and full_by_id[qid]["challenge_type"] == challenge_type
    )
    amp = [
        float(no_res_by_id[qid]["top1_hit"]) - float(full_by_id[qid]["top1_hit"])
        for qid in shared_ids
    ]
    return shared_ids, amp


def _align_amp_vectors(
    qids_e5: list[str],
    amp_e5: list[float],
    qids_sbert: list[str],
    amp_sbert: list[float],
) -> tuple[list[float], list[float]]:
    """Inner-join on query_id; raise on non-trivial mismatch.

    Returns (aligned_e5, aligned_sbert) — same length, same query order.
    Missing IDs (one-side-only) are silently dropped only if the overlap
    covers ≥90% of the larger side; otherwise raises to surface data issues.
    """
    set_e5 = set(qids_e5)
    set_sbert = set(qids_sbert)
    shared = set_e5 & set_sbert
    larger = max(len(set_e5), len(set_sbert))
    if larger > 0 and len(shared) / larger < 0.9:
        only_e5 = sorted(set_e5 - set_sbert)[:5]
        only_sbert = sorted(set_sbert - set_e5)[:5]
        raise RuntimeError(
            f"Cross-embedder query_id mismatch: {len(shared)}/{larger} overlap.\n"
            f"  Only in e5   (first 5): {only_e5}\n"
            f"  Only in sbert (first 5): {only_sbert}\n"
            "Check that both runs used the same dataset and --seed."
        )
    e5_map = dict(zip(qids_e5, amp_e5, strict=True))
    sbert_map = dict(zip(qids_sbert, amp_sbert, strict=True))
    common_sorted = sorted(shared)
    return (
        [e5_map[q] for q in common_sorted],
        [sbert_map[q] for q in common_sorted],
    )


# ---------------------------------------------------------------------------
# Per-hypothesis test
# ---------------------------------------------------------------------------


def _test_hypothesis(
    amp_e5: list[float],
    amp_sbert: list[float],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Run one paired bootstrap test: delta = mean(amp_e5) - mean(amp_sbert).

    Returns raw (pre-Holm) result dict.  Verdict and p_adj filled in later.
    """
    delta, lo, hi, p_two = paired_bootstrap_diff(
        amp_e5, amp_sbert, n_bootstrap=n_bootstrap, seed=seed
    )
    # One-sided p: directional (e5 > sbert interference amplification, delta > 0)
    p_one = p_two / 2.0 if delta > 0 else 1.0 - p_two / 2.0
    d = cohens_d_paired(amp_e5, amp_sbert)
    return {
        "n_pairs": len(amp_e5),
        "delta": round(delta, 4),
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
        "p_two_sided": round(p_two, 4),
        "p_one_sided": round(p_one, 4),
        "cohens_d": round(d, 4) if d == d else None,  # NaN guard
        "p_adj_holm": None,  # filled after Holm pass
        "verdict": None,  # filled after Holm pass
    }


# ---------------------------------------------------------------------------
# Mechanism (exploratory, no formal test)
# ---------------------------------------------------------------------------


def _mechanism_summary(
    stats_sbert: dict[str, Any] | None,
    stats_e5: dict[str, Any] | None,
) -> dict[str, Any]:
    """Descriptive comparison of link_set_stats between sbert and e5.

    No statistical tests — exploratory only.  Returns an empty dict if
    link_set_stats are absent (e.g. older runner outputs).
    """
    if not stats_sbert or not stats_e5:
        return {"note": "link_set_stats not available in one or both result files"}

    def _extract(s: dict[str, Any]) -> dict[str, Any]:
        lpm = s.get("links_per_memory", {})
        return {
            "links_per_memory_mean": lpm.get("mean"),
            "links_per_memory_max": lpm.get("max"),
            "n_memories_total": s.get("n_memories_total"),
            "link_types": s.get("link_types", {}),
        }

    return {
        "sbert": _extract(stats_sbert),
        "e5": _extract(stats_e5),
        "note": "Exploratory — no formal test. Differences in link density may explain "
        "differential resonance amplification between embedders.",
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_hi3(
    results_sbert: Path = DEFAULT_RESULTS_SBERT,
    results_e5: Path = DEFAULT_RESULTS_E5,
    *,
    n_bootstrap: int = _PREREG_N_BOOTSTRAP,
    seed: int = _PREREG_SEED,
) -> dict[str, Any]:
    """Run the Hi3 Holm-family analysis and return the full results dict."""
    records_sbert = _load_per_query_records(results_sbert)
    records_e5 = _load_per_query_records(results_e5)
    link_stats_sbert = _load_link_set_stats(results_sbert)
    link_stats_e5 = _load_link_set_stats(results_e5)

    family_raw: list[dict[str, Any]] = []
    for challenge in CHALLENGE_FAMILY:
        qids_e5, amp_e5 = _build_amp_vector(records_e5, challenge_type=challenge)
        qids_sbert, amp_sbert = _build_amp_vector(records_sbert, challenge_type=challenge)
        aligned_e5, aligned_sbert = _align_amp_vectors(qids_e5, amp_e5, qids_sbert, amp_sbert)
        raw = _test_hypothesis(aligned_e5, aligned_sbert, n_bootstrap=n_bootstrap, seed=seed)
        raw["challenge_type"] = challenge
        raw["hypothesis"] = HYPOTHESIS_NAMES[challenge]
        family_raw.append(raw)

    # Holm correction over one-sided p-values (directional family)
    p_ones = [r["p_one_sided"] for r in family_raw]
    p_adjs = holm_bonferroni(p_ones)
    for row, p_adj in zip(family_raw, p_adjs, strict=True):
        row["p_adj_holm"] = round(p_adj, 4)
        row["verdict"] = (
            "PASS" if row["delta"] > _HI3_EFFECT_THRESHOLD and p_adj < _HI3_ALPHA else "FAIL"
        )

    hypotheses = {r["hypothesis"]: r for r in family_raw}
    mechanism = _mechanism_summary(link_stats_sbert, link_stats_e5)

    return {
        "study": "addendum_i_hi3",
        "prereg": "benchmarks/preregistration_addendum_i.md",
        "dataset": "benchmarks/datasets/realistic_recall_v3.json",
        "inputs": {
            "sbert": str(results_sbert),
            "e5": str(results_e5),
        },
        "protocol": {
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "alpha": _HI3_ALPHA,
            "effect_threshold": _HI3_EFFECT_THRESHOLD,
            "family_m": len(CHALLENGE_FAMILY),
            "correction": "holm_bonferroni",
            "ci_method": "bootstrap_percentile",
            "p_direction": "one_sided_directional",
        },
        "hypotheses": hypotheses,
        "mechanism_exploratory": mechanism,
        "run_timestamp": datetime.now(tz=UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_markdown(results: dict[str, Any]) -> str:
    proto = results["protocol"]
    hyps = results["hypotheses"]
    lines = [
        "# Addendum I — Hi3 Confirmatory Analysis",
        "",
        "Resonance amplification on affect-rich queries: e5-small-v2 vs SBERT.",
        "",
        f"Pre-registration: `{results['prereg']}`  ",
        f"Dataset: `{results['dataset']}`  ",
        f"Bootstrap: n={proto['n_bootstrap']}, seed={proto['seed']}, "
        f"CI=95%, {proto['p_direction']}, Holm m={proto['family_m']}",
        "",
        "Statistic: `delta = mean(amp_e5) - mean(amp_sbert)` where "
        "`amp[i] = no_resonance_top1_hit[i] - full_top1_hit[i]` "
        "(positive = resonance hurts; pre-reg sign convention).  ",
        f"PASS iff `delta > {proto['effect_threshold']}` AND `p_adj < {proto['alpha']}`.",
        "",
        "## Confirmatory Results",
        "",
        "| Hypothesis | Challenge | Δ [95% CI] | p_one | p_adj (Holm) | Cohen's d | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for challenge in CHALLENGE_FAMILY:
        name = HYPOTHESIS_NAMES[challenge]
        h = hyps[name]
        role = "**primary**" if challenge == "semantic_confound" else "secondary"
        d_str = f"{h['cohens_d']:.3f}" if h["cohens_d"] is not None else "n/a"
        ci_str = format_point_ci(h["delta"], h["ci_lower"], h["ci_upper"], dp=3)
        verdict_str = f"**{h['verdict']}**" if h["verdict"] == "PASS" else h["verdict"]
        lines.append(
            f"| {name} ({role}) | {challenge} | {ci_str} | "
            f"{h['p_one_sided']:.4f} | {h['p_adj_holm']:.4f} | {d_str} | {verdict_str} |"
        )

    # Mechanism table (exploratory)
    mech = results.get("mechanism_exploratory", {})
    if "sbert" in mech and "e5" in mech:
        lines += [
            "",
            "## Mechanism Analysis (Exploratory — no formal test)",
            "",
            "| Embedder | links/memory (mean) | links/memory (max) | N memories |",
            "|---|---|---|---|",
        ]
        for emb_key, label in [("sbert", "SBERT (bge-small)"), ("e5", "e5-small-v2")]:
            m = mech[emb_key]
            lines.append(
                f"| {label} | {m['links_per_memory_mean']} | "
                f"{m['links_per_memory_max']} | {m['n_memories_total']} |"
            )
        lines += ["", mech.get("note", "")]

    lines.append(f"\n*Run: {results['run_timestamp']}*")
    return "\n".join(lines)


def _build_protocol(results: dict[str, Any]) -> dict[str, Any]:
    hyps = results["hypotheses"]
    verdicts = {name: h["verdict"] for name, h in hyps.items()}
    overall = "PASS" if verdicts.get("Hi3") == "PASS" else "FAIL"
    return {
        "study": results["study"],
        "prereg": results["prereg"],
        "protocol": results["protocol"],
        "run_timestamp": results["run_timestamp"],
        "input_hashes": {
            k: hashlib.sha256(Path(v).read_bytes()).hexdigest()[:16]
            for k, v in results["inputs"].items()
            if Path(v).exists()
        },
        "verdicts": verdicts,
        "overall_verdict": overall,
        "exploratory_followup": [
            "counterfactual_cosine_threshold_raised — deferred; requires re-run with "
            "modified ResonanceConfig.cosine_threshold"
        ],
    }


def write_results(
    results: dict[str, Any],
    *,
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
    out_protocol: Path = DEFAULT_OUT_PROTOCOL,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(results), encoding="utf-8")
    out_protocol.write_text(json.dumps(_build_protocol(results), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hi3 confirmatory analysis: resonance amplification e5 vs SBERT"
    )
    parser.add_argument(
        "--results-sbert",
        type=Path,
        default=DEFAULT_RESULTS_SBERT,
        help=f"SBERT ablation results JSON (default: {DEFAULT_RESULTS_SBERT})",
    )
    parser.add_argument(
        "--results-e5",
        type=Path,
        default=DEFAULT_RESULTS_E5,
        help=f"e5-small-v2 ablation results JSON (default: {DEFAULT_RESULTS_E5})",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-protocol", type=Path, default=DEFAULT_OUT_PROTOCOL)
    parser.add_argument("--n-bootstrap", type=int, default=_PREREG_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=_PREREG_SEED)
    args = parser.parse_args()

    print(f"Hi3 analysis: {args.results_sbert} + {args.results_e5}")
    results = run_hi3(
        args.results_sbert,
        args.results_e5,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    write_results(
        results, out_json=args.out_json, out_md=args.out_md, out_protocol=args.out_protocol
    )
    print(f"Written: {args.out_json.parent}")

    print("\n=== Hi3 Family Results ===")
    for challenge in CHALLENGE_FAMILY:
        name = HYPOTHESIS_NAMES[challenge]
        h = results["hypotheses"][name]
        ci_str = format_point_ci(h["delta"], h["ci_lower"], h["ci_upper"], dp=3)
        print(
            f"  {name:15s} Δ={ci_str}  p_one={h['p_one_sided']:.4f}"
            f"  p_adj={h['p_adj_holm']:.4f}  [{h['verdict']}]"
        )


if __name__ == "__main__":
    main()
