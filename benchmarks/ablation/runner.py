"""Ablation study runner for the AFT layers (Addenda E + F, pre-reg v3 + f).

Runs the realistic replay benchmark once per variant and compares each ablation
to the full configuration via paired bootstrap + exact McNemar test.

Ablations defined:
  full                   All layers active (baseline AFT)
  no_appraisal           enable_appraisal=False  — skip Scherer CPM call at encode time
  no_mood                enable_mood_signal=False — zero s1 (mood-congruence) at retrieval
  no_momentum            enable_momentum=False   — zero s3 (momentum-alignment) at retrieval
  no_resonance           enable_resonance=False  — skip link building + spreading activation
  no_reconsolidation     enable_reconsolidation=False — skip APE-gated reconsolidation (He2)
  dual_path              dual_path_encoding=True + KeywordAppraisalEngine (He1); see caveat
  aft_keyword_synchronous KeywordAppraisalEngine synchronous at encode time (Hf1 baseline)

Note: no_appraisal is a no-op on this benchmark because AFTReplayAdapter does
not configure an appraisal engine (it uses explicit valence/arousal injection).
The flag is still exercised to confirm correct hook-up.

Note on dual_path (He1): KeywordAppraisalEngine overrides preset affect on this
dataset (same finding as G3/Addendum A). He1 compares dual_path vs full_aft
(no appraisal), so He1 FAIL is expected. See also Addendum F (Hf1).

Note on aft_keyword_synchronous (Hf1, Addendum F): synchronous keyword appraisal
baseline. Hf1 tests whether deferral (dual_path) mitigates the destructive
override vs synchronous (aft_keyword_synchronous). Expected: Hf1 PASS
(dual_path=0.35 > aft_keyword_synchronous≈0.16 based on Addendum A G3 data).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tqdm import tqdm

from benchmarks.common.statistics import (
    DEFAULT_N_BOOTSTRAP,
    cohens_d_paired,
    format_point_ci,
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap_diff,
)
from benchmarks.realistic.adapters import AFTReplayAdapter
from benchmarks.realistic.runner import _build_embedder, load_dataset, run_benchmark
from emotional_memory import Embedder, EmotionalMemoryConfig


class AFTDualPathReplayAdapter(AFTReplayAdapter):
    """AFTReplayAdapter with KeywordAppraisalEngine + explicit slow-path elaboration (He1)."""

    name = "dual_path"

    def begin_session(self, session_id: str) -> Any:
        from emotional_memory.appraisal_llm import KeywordAppraisalEngine

        result = super().begin_session(session_id)
        if self._engine is not None:
            self._engine._appraisal_engine = KeywordAppraisalEngine()
        return result

    def encode(
        self,
        *,
        memory_alias: str,
        content: str,
        valence: float,
        arousal: float,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        memory_id = super().encode(
            memory_alias=memory_alias,
            content=content,
            valence=valence,
            arousal=arousal,
            metadata=metadata,
        )
        # Slow path: blend 70% keyword-appraised + 30% raw preset affect (LeDoux, 1996)
        engine = self._require_engine()
        engine.elaborate(memory_id)
        return memory_id


class AFTKeywordSynchronousReplayAdapter(AFTReplayAdapter):
    """AFTReplayAdapter with KeywordAppraisalEngine running synchronously at encode time (Hf1).

    Synchronous path: appraisal runs during engine.encode() via the standard pipeline.
    No elaborate() call. Compare against dual_path (deferred) to test whether
    deferral mitigates the destructive override (Addendum F).
    """

    name = "aft_keyword_synchronous"

    def begin_session(self, session_id: str) -> Any:
        from emotional_memory.appraisal_llm import KeywordAppraisalEngine

        result = super().begin_session(session_id)
        if self._engine is not None:
            self._engine._appraisal_engine = KeywordAppraisalEngine()
        return result


ABLATIONS: list[tuple[str, dict[str, bool]]] = [
    ("full", {}),
    ("no_appraisal", {"enable_appraisal": False}),
    ("no_mood", {"enable_mood_signal": False}),
    ("no_momentum", {"enable_momentum": False}),
    ("no_resonance", {"enable_resonance": False}),
    ("no_reconsolidation", {"enable_reconsolidation": False}),
    ("dual_path", {"dual_path_encoding": True}),
    ("aft_keyword_synchronous", {}),
]

# Variants requiring a custom adapter class (keyed by variant name)
_ADAPTER_OVERRIDES: dict[str, type[AFTReplayAdapter]] = {
    "dual_path": AFTDualPathReplayAdapter,
    "aft_keyword_synchronous": AFTKeywordSynchronousReplayAdapter,
}

_HERE = Path(__file__).parent
RESULTS_JSON = _HERE / "results.json"
RESULTS_MD = _HERE / "results.md"
RESULTS_PROTOCOL = _HERE / "results.protocol.json"


def _extract_query_flags(bench_result: dict[str, Any]) -> dict[str, dict[str, bool]]:
    """Return {query_id: {"top1_hit": bool, "hit": bool}} for the aft system."""
    aft = next(s for s in bench_result["systems"] if s["system"] == "aft")
    flags: dict[str, dict[str, bool]] = {}
    for scenario in aft["scenarios"]:
        for session in scenario["sessions"]:
            for q in session["queries"]:
                flags[q["query_id"]] = {
                    "top1_hit": bool(q["top1_hit"]),
                    "hit": bool(q["hit"]),
                }
    return flags


def run_ablation_study(
    dataset: Any | None = None,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
    top_k: int | None = None,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Run the full ablation study and return the results dict."""
    if dataset is None:
        dataset = load_dataset()

    variant_results: list[dict[str, Any]] = []
    query_flags_by_variant: dict[str, dict[str, dict[str, bool]]] = {}

    for variant_name, flag_kwargs in tqdm(ABLATIONS, desc="ablation variants", unit="variant"):
        cfg = EmotionalMemoryConfig(**flag_kwargs)
        adapter_cls = _ADAPTER_OVERRIDES.get(variant_name)
        bench = run_benchmark(
            dataset,
            systems=["aft"],
            top_k=top_k,
            n_bootstrap=n_bootstrap,
            seed=seed,
            aft_config=cfg,
            aft_adapter_cls=adapter_cls,
            embedder=embedder,
        )
        aft_sys = next(s for s in bench["systems"] if s["system"] == "aft")
        query_flags_by_variant[variant_name] = _extract_query_flags(bench)
        variant_results.append(
            {
                "variant": variant_name,
                "flags": flag_kwargs,
                "aggregate_metrics": aft_sys["aggregate_metrics"],
                "challenge_type_metrics": aft_sys["challenge_type_metrics"],
            }
        )

    # Pairwise vs full
    full_flags = query_flags_by_variant["full"]
    query_ids = sorted(full_flags.keys())
    full_top1 = [full_flags[qid]["top1_hit"] for qid in query_ids]
    full_hit = [full_flags[qid]["hit"] for qid in query_ids]

    # First pass: compute raw stats for each ablation
    raw_rows: list[dict[str, Any]] = []
    for variant_name, *_ in ABLATIONS:
        if variant_name == "full":
            continue
        abl_flags = query_flags_by_variant[variant_name]
        abl_top1 = [abl_flags.get(qid, {"top1_hit": False})["top1_hit"] for qid in query_ids]
        abl_hit = [abl_flags.get(qid, {"hit": False})["hit"] for qid in query_ids]

        diff_top1, lo_top1, hi_top1, p_boot_top1 = paired_bootstrap_diff(
            abl_top1, full_top1, n_bootstrap=n_bootstrap, seed=seed
        )
        only_abl_top1 = sum(a and not f for a, f in zip(abl_top1, full_top1, strict=True))
        only_full_top1 = sum(f and not a for a, f in zip(abl_top1, full_top1, strict=True))
        p_mc_top1 = mcnemar_exact(only_abl_top1, only_full_top1)
        d_top1 = cohens_d_paired(abl_top1, full_top1, hedges_correction=True)

        diff_hit, lo_hit, hi_hit, p_boot_hit = paired_bootstrap_diff(
            abl_hit, full_hit, n_bootstrap=n_bootstrap, seed=seed
        )
        only_abl_hit = sum(a and not f for a, f in zip(abl_hit, full_hit, strict=True))
        only_full_hit = sum(f and not a for a, f in zip(abl_hit, full_hit, strict=True))
        p_mc_hit = mcnemar_exact(only_abl_hit, only_full_hit)
        d_hit = cohens_d_paired(abl_hit, full_hit, hedges_correction=True)

        raw_rows.append(
            {
                "variant": variant_name,
                "baseline": "full",
                "top1": {
                    "diff": round(diff_top1, 4),
                    "ci_lower": round(lo_top1, 4),
                    "ci_upper": round(hi_top1, 4),
                    "p_bootstrap": round(p_boot_top1, 4),
                    "p_mcnemar": round(p_mc_top1, 4),
                    "effect_size_d": round(d_top1, 4) if not math.isnan(d_top1) else None,
                    "n_queries": len(query_ids),
                    "n_discordant": only_abl_top1 + only_full_top1,
                },
                "hit_at_k": {
                    "diff": round(diff_hit, 4),
                    "ci_lower": round(lo_hit, 4),
                    "ci_upper": round(hi_hit, 4),
                    "p_bootstrap": round(p_boot_hit, 4),
                    "p_mcnemar": round(p_mc_hit, 4),
                    "effect_size_d": round(d_hit, 4) if not math.isnan(d_hit) else None,
                    "n_queries": len(query_ids),
                    "n_discordant": only_abl_hit + only_full_hit,
                },
            }
        )

    # Second pass: apply Holm-Bonferroni on the 4 bootstrap p-values (top1)
    p_boots_top1 = [r["top1"]["p_bootstrap"] for r in raw_rows]
    p_adj_top1 = holm_bonferroni(p_boots_top1)
    p_boots_hit = [r["hit_at_k"]["p_bootstrap"] for r in raw_rows]
    p_adj_hit = holm_bonferroni(p_boots_hit)

    pairwise: list[dict[str, Any]] = []
    for row, p_adj_t, p_adj_h in zip(raw_rows, p_adj_top1, p_adj_hit, strict=True):
        row["top1"]["p_bootstrap_adj_holm"] = round(p_adj_t, 4)
        row["hit_at_k"]["p_bootstrap_adj_holm"] = round(p_adj_h, 4)
        pairwise.append(row)

    return {
        "benchmark": f"ablation_{dataset.name}",
        "base_benchmark": dataset.name,
        "variants": variant_results,
        "pairwise_vs_full": pairwise,
        "statistics": {
            "n_bootstrap": n_bootstrap,
            "confidence": 0.95,
            "ci_method": "bootstrap_percentile",
            "seed": seed,
            "multiple_comparisons_correction": "holm_bonferroni",
            "effect_size": "cohens_d_paired_hedges_corrected",
        },
    }


def _render_markdown(results: dict[str, Any]) -> str:
    stats = results["statistics"]
    ci_note = (
        f"95% CI via percentile bootstrap (n={stats['n_bootstrap']}, seed={stats['seed']}). "
        "Pairwise delta = ablated - full (negative = layer helps)."
    )
    lines = [
        "# AFT Layer Ablation Study",
        "",
        "Measures the isolated contribution of each AFT layer to `top1_accuracy` "
        "on the realistic replay benchmark.",
        "",
        f"{ci_note}",
        "",
        "**Note on `no_appraisal`**: the realistic benchmark injects affect directly via "
        "`set_affect()`, so no appraisal engine is configured. This ablation is a no-op "
        "on this benchmark and confirms correct flag hook-up only.",
        "",
        "**Note on `dual_path` (He1 pre-reg v3)**: uses `KeywordAppraisalEngine` + slow-path "
        "`elaborate()`. He1 compares dual_path vs `full_aft` (pure preset affect). "
        "KeywordAppraisalEngine degrades affect on this dataset (G3/Addendum A: "
        "aft_keyword=0.16 vs aft_noAppraisal=0.78), so He1 FAIL is the expected outcome. "
        "The discriminative Hf1 comparison (dual_path vs aft_keyword_synchronous) is below.",
        "",
        "**Note on `no_reconsolidation` (He2 pre-reg v3)**: disables the APE-gated "
        "reconsolidation window; predictive-learning (`update_prediction`) still runs.",
        "",
        "**Note on `aft_keyword_synchronous` (Hf1 pre-reg Addendum F)**: synchronous "
        "keyword appraisal baseline. Compare vs `dual_path` (deferred) to test whether "
        "deferral mitigates the destructive override observed in G3/Addendum A. "
        "Hf1 PASS expected: dual_path=0.35 > aft_keyword_synchronous≈0.16.",
        "",
    ]

    # Per-variant summary table
    lines += [
        "## Results by Variant",
        "",
        "| Variant | top1 [95% CI] | hit@k [95% CI] | N queries |",
        "| ------- | ------------- | -------------- | --------- |",
    ]
    for v in results["variants"]:
        agg = v["aggregate_metrics"]
        ci = agg.get("ci", {})
        top1_str = (
            format_point_ci(
                ci["top1_accuracy"]["point"],
                ci["top1_accuracy"]["ci_lower"],
                ci["top1_accuracy"]["ci_upper"],
            )
            if "top1_accuracy" in ci
            else f"{agg['top1_accuracy']:.2f}"
        )
        hit_str = (
            format_point_ci(
                ci["hit_at_k"]["point"],
                ci["hit_at_k"]["ci_lower"],
                ci["hit_at_k"]["ci_upper"],
            )
            if "hit_at_k" in ci
            else f"{agg['hit_at_k']:.2f}"
        )
        n = agg.get("query_count", "?")
        lines.append(f"| `{v['variant']}` | {top1_str} | {hit_str} | {n} |")

    lines += [""]

    # Pairwise table
    lines += [
        "## Pairwise vs Full (top1_accuracy)",
        "",
        "| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm)"
        " | p (McNemar) | d (Hedges g) | N | Discordant |",
        "| ------- | ---------- | ------------- | ------------"
        " | ----------- | ------------ | - | ---------- |",
    ]
    for row in results["pairwise_vs_full"]:
        t = row["top1"]
        delta_str = format_point_ci(t["diff"], t["ci_lower"], t["ci_upper"])
        d_str = f"{t['effect_size_d']:.3f}" if t.get("effect_size_d") is not None else "—"
        p_adj = t.get("p_bootstrap_adj_holm", t["p_bootstrap"])
        lines.append(
            f"| `{row['variant']}` | {delta_str} | {t['p_bootstrap']:.3f} "
            f"| {p_adj:.3f} | {t['p_mcnemar']:.3f} | {d_str} "
            f"| {t['n_queries']} | {t['n_discordant']} |"
        )
    lines += [""]

    lines += [
        "## Pairwise vs Full (hit@k)",
        "",
        "| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm)"
        " | p (McNemar) | d (Hedges g) | N | Discordant |",
        "| ------- | ---------- | ------------- | ------------"
        " | ----------- | ------------ | - | ---------- |",
    ]
    for row in results["pairwise_vs_full"]:
        h = row["hit_at_k"]
        delta_str = format_point_ci(h["diff"], h["ci_lower"], h["ci_upper"])
        d_str = f"{h['effect_size_d']:.3f}" if h.get("effect_size_d") is not None else "—"
        p_adj = h.get("p_bootstrap_adj_holm", h["p_bootstrap"])
        lines.append(
            f"| `{row['variant']}` | {delta_str} | {h['p_bootstrap']:.3f} "
            f"| {p_adj:.3f} | {h['p_mcnemar']:.3f} | {d_str} "
            f"| {h['n_queries']} | {h['n_discordant']} |"
        )
    lines += [""]

    lines += [
        "## Interpretation",
        "",
        "A variant with Δ significantly negative (CI entirely below 0, p_adj < 0.05 after "
        "Holm-Bonferroni correction) indicates that layer **contributes** to retrieval quality: "
        "removing it hurts performance. A variant with Δ ≈ 0 and small |d| indicates the layer "
        "has no measurable impact on this benchmark — either the signal is redundant or the "
        "dataset is too small to detect the effect (limited power at N=100 queries).",
        "",
    ]

    # Hf1 supplementary: dual_path vs aft_keyword_synchronous
    pw = {r["variant"]: r for r in results["pairwise_vs_full"]}
    if "dual_path" in pw and "aft_keyword_synchronous" in pw:
        lines += [
            "## Supplementary: Hf1 — dual_path vs aft_keyword_synchronous (Addendum F)",
            "",
            "Hf1 tests whether deferring keyword appraisal (slow-path `elaborate()`) partially "
            "mitigates the destructive override of synchronous keyword appraisal.",
            "",
        ]
        # Compute direct Hf1 comparison from the stored variant results
        dp_row = next((v for v in results["variants"] if v["variant"] == "dual_path"), None)
        ks_row = next(
            (v for v in results["variants"] if v["variant"] == "aft_keyword_synchronous"), None
        )
        if dp_row and ks_row:
            dp_top1 = dp_row["aggregate_metrics"].get("top1_accuracy", float("nan"))
            ks_top1 = ks_row["aggregate_metrics"].get("top1_accuracy", float("nan"))
            delta_hf1 = round(dp_top1 - ks_top1, 4)
            verdict = "**Hf1 PASS**" if delta_hf1 > 0 else "**Hf1 FAIL**"
            lines += [
                "| Metric | dual_path | aft_keyword_synchronous | Δ (Hf1) | Verdict |",
                "|--------|-----------|------------------------|---------|---------|",
                f"| top1   | {dp_top1:.3f} | {ks_top1:.3f} | {delta_hf1:+.4f} | {verdict} |",
                "",
                "Note: paired bootstrap and Holm-corrected p-values for each variant "
                "vs full_aft are in the pairwise tables above. Direct Hf1 statistical "
                "significance requires a separate pairwise bootstrap not computed here; "
                "the delta direction is the pre-registered criterion.",
                "",
            ]

    return "\n".join(lines)


def _build_protocol(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": results["benchmark"],
        "base_benchmark": results["base_benchmark"],
        "ablations": [{"variant": v["variant"], "flags": v["flags"]} for v in results["variants"]],
        "statistics": results["statistics"],
        "interpretation_notes": [
            "delta = ablated - full: negative means layer helps.",
            "no_appraisal is a no-op: no appraisal engine configured in realistic benchmark.",
            "dual_path uses KeywordAppraisalEngine + elaborate(); He1 FAIL expected (keyword "
            "appraisal degrades affect on this dataset, same as G3/Addendum A).",
            "no_reconsolidation disables APE-gated reconsolidation window; update_prediction "
            "(predictive learning) still runs.",
            "aft_keyword_synchronous: synchronous keyword appraisal (Addendum F baseline); "
            "compare vs dual_path for Hf1 — does deferral mitigate destructive override?",
            "Holm denominator = 7 (all non-full variants); adding aft_keyword_synchronous "
            "slightly increases p_adj for existing variants.",
            "N=100 queries (v1.4 expansion); directional trends are informative even if not "
            "all results survive correction.",
        ],
    }


def write_results(
    results: dict[str, Any],
    *,
    out_json: Path = RESULTS_JSON,
    out_md: Path = RESULTS_MD,
    out_protocol: Path = RESULTS_PROTOCOL,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(results), encoding="utf-8")
    out_protocol.write_text(json.dumps(_build_protocol(results), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="AFT layer ablation study")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_HERE,
        help="Directory for results files (default: benchmarks/ablation/)",
    )
    parser.add_argument("--out-json", type=Path, default=None, help="Override results JSON path.")
    parser.add_argument(
        "--out-md", type=Path, default=None, help="Override results Markdown path."
    )
    parser.add_argument(
        "--out-protocol", type=Path, default=None, help="Override results protocol JSON path."
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="sbert-bge",
        choices=["hash", "sbert-bge", "sbert-mini", "e5-small-v2", "multilingual-e5-small"],
        help="Embedder backend (hash = TokenHashEmbedder, sbert-bge = BAAI/bge-small-en-v1.5).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset JSON path (default: realistic_recall_v1.json). "
        "Use benchmarks/datasets/realistic_recall_v2.json for S3 powered ablation.",
    )
    args = parser.parse_args()

    emb = _build_embedder(args.embedder)
    dataset = load_dataset() if args.dataset is None else load_dataset(args.dataset)
    print(f"Running ablation study on {dataset.name} ({len(dataset.scenarios)} scenarios)...")
    results = run_ablation_study(
        dataset,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        top_k=args.top_k,
        embedder=emb,
    )

    out_json = args.out_json if args.out_json is not None else args.output_dir / "results.json"
    out_md = args.out_md if args.out_md is not None else args.output_dir / "results.md"
    out_protocol = (
        args.out_protocol
        if args.out_protocol is not None
        else args.output_dir / "results.protocol.json"
    )
    write_results(results, out_json=out_json, out_md=out_md, out_protocol=out_protocol)
    print(f"Results written to {out_json.parent}")

    # Print summary
    print("\n=== Ablation Results (top1_accuracy) ===")
    for v in results["variants"]:
        agg = v["aggregate_metrics"]
        ci = agg.get("ci", {})
        if "top1_accuracy" in ci:
            c = ci["top1_accuracy"]
            print(
                f"  {v['variant']:20s} {format_point_ci(c['point'], c['ci_lower'], c['ci_upper'])}"
            )
        else:
            print(f"  {v['variant']:20s} {agg['top1_accuracy']:.2f}")

    print("\n=== Pairwise vs full (top1_accuracy) ===")
    for row in results["pairwise_vs_full"]:
        t = row["top1"]
        delta_str = format_point_ci(t["diff"], t["ci_lower"], t["ci_upper"])
        print(
            f"  {row['variant']:20s} Δ={delta_str}  "
            f"p_boot={t['p_bootstrap']:.3f}  p_mc={t['p_mcnemar']:.3f}"
        )


if __name__ == "__main__":
    main()
