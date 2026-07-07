"""Addendum Z runner — held-out learned retrieval profile across corpora.

For each corpus we extract, per query, the 6 AFT retrieval signals over the full
candidate pool (via ``build_retrieval_plan`` with the fixed base weights, so the
raw signals are weight-independent), the binary gold, and the corpus metric. Then:

- ``naive_cosine`` = rank by signal s1 (semantic) alone,
- ``aft_fixed``    = rank by the engine's fixed ``base_weights``,
- ``aft_learned``  = held-out cross-fit linear learning-to-rank over the 6 signals.

Confirmatory family (Holm m=3, one-tailed): Hz1 = {MADial, ES-MemEval, DailyDialog}
``aft_learned`` (held-out) > ``naive_cosine``. Hz2 = curated non-inferiority vs
``aft_fixed``. See ``benchmarks/preregistration_addendum_z_learned_profile.md``.

This harness PR wires the two dry-capable third-party break corpora (MADial-Bench X,
ES-MemEval X2), validated end-to-end by ``--dry-run`` (keyword appraiser, no LLM).
DailyDialog and curated (both stateful / LLM-only) are added in the run+closure PR;
until the full pre-registered family is present the Holm verdict is not evaluable.

Usage::

    make bench-z-profile                                       # scored (needs API key)
    uv run python -m benchmarks.learned_profile.runner --dry-run   # smoke, no LLM
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from benchmarks.common.ltr import (
    AFT_FIXED_WEIGHTS,
    COSINE_WEIGHTS,
    QueryFeatures,
    cross_fit,
    score_fixed,
)
from benchmarks.common.statistics import cohens_d_paired, holm_bonferroni, paired_bootstrap_diff
from emotional_memory.affect import AffectiveMomentum
from emotional_memory.engine import EmotionalMemory
from emotional_memory.mood import MoodField
from emotional_memory.resonance import spreading_activation
from emotional_memory.retrieval import build_retrieval_plan
from emotional_memory.stores.in_memory import InMemoryStore

_HERE = Path(__file__).parent
DEFAULT_OUT_JSON = _HERE / "results.json"
DEFAULT_OUT_MD = _HERE / "results.md"

DEFAULT_N_BOOTSTRAP = 10_000
DEFAULT_K = 5
_PLAN_TOPK = 5  # seed-set size for the s6 (resonance) activation map
_DRY_QUERY_LIMIT = 20  # smoke: cap queries per corpus (not a scored verdict)

# Pre-registered confirmatory family.
_HZ1_BREAK = ("madialbench", "esmemeval", "dailydialog_t2a")
_HZ2_PRESERVE = "realistic_recall_v2"
_NONINFERIORITY_EPS = 0.02


def _pool_features(
    engine: EmotionalMemory,
    query_embedding: list[float],
    query_affect: Any,
    candidates: list[Any],
    *,
    now: datetime,
) -> np.ndarray:
    """Extract the (n_candidates, 6) raw-signal matrix aligned to ``candidates``.

    Uses ``build_retrieval_plan`` with the fixed base weights; the raw signals it
    records per candidate are weight-independent (s6 uses the base-weight seed set,
    a declared protocol choice). Retrieve-time context is neutral mood + zero
    momentum + the appraised query affect (pre-registration §Protocol).
    """
    cfg = engine._config
    plan = build_retrieval_plan(
        query_embedding=query_embedding,
        query_affect=query_affect,
        current_mood=MoodField.neutral(),
        current_momentum=AffectiveMomentum.zero(),
        candidates=candidates,
        top_k=_PLAN_TOPK,
        now=now,
        decay_config=cfg.decay,
        retrieval_config=cfg.retrieval,
        propagation_hops=cfg.resonance.propagation_hops,
        spreading_activation_fn=spreading_activation,
        precomputed_weights=np.asarray(AFT_FIXED_WEIGHTS, dtype=np.float64),
    )
    sig: dict[str, list[float]] = {}
    for rm in plan.pass2:
        r = rm.breakdown.raw_signals
        sig[rm.memory.id] = [
            r.semantic_similarity,
            r.mood_congruence,
            r.affect_proximity,
            r.momentum_alignment,
            r.recency,
            r.resonance,
        ]
    return np.asarray([sig[c.id] for c in candidates], dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-corpus query builders (dry-capable break corpora: MADial X, ES-MemEval X2)
# ---------------------------------------------------------------------------


def _madial_queries(*, dry_run: bool) -> list[QueryFeatures]:
    from benchmarks.madialbench.adapters import AFTQueryAppraisedMadialAdapter, _make_embedder
    from benchmarks.madialbench.dataset import load_dataset
    from benchmarks.madialbench.metrics import ndcg_at_k

    dataset = load_dataset()
    adapter = AFTQueryAppraisedMadialAdapter(dry_run=dry_run, embedder=_make_embedder())
    adapter.ingest(list(dataset.memories))
    engine = adapter._engine
    id_of = adapter._memory_id_map  # store_id -> madial int id
    candidates = engine._store.list_all()
    now = datetime.now(tz=UTC)

    queries = dataset.queries[:_DRY_QUERY_LIMIT] if dry_run else dataset.queries
    out: list[QueryFeatures] = []
    for q in queries:
        qa = adapter._appraiser.appraise(q.text).to_core_affect()
        emb = engine._embedder.embed(q.text)
        feats = _pool_features(engine, emb, qa, candidates, now=now)
        gold = np.asarray([1.0 if id_of[c.id] in q.gold_ids else 0.0 for c in candidates])
        cand_ids = [id_of[c.id] for c in candidates]
        out.append(_make_qf(feats, gold, cand_ids, q.gold_ids, ndcg_at_k, 5))
    adapter.close()
    return out


def _esmemeval_queries(*, dry_run: bool) -> list[QueryFeatures]:
    from benchmarks.esmemeval.adapters import (
        _TIME_INVARIANT_CONFIG,
        AFTQueryAppraisedEsmemAdapter,
        _make_embedder,
    )
    from benchmarks.esmemeval.dataset import build_pools, load_dataset
    from benchmarks.esmemeval.metrics import PRIMARY_K, upstream_ndcg_at_k

    dataset = load_dataset()
    pools = build_pools(dataset)
    embedder = _make_embedder()
    adapter = AFTQueryAppraisedEsmemAdapter(dry_run=dry_run, embedder=embedder)
    adapter.ingest(list(dataset.sessions))
    now = datetime.now(tz=UTC)

    queries = dataset.queries[:_DRY_QUERY_LIMIT] if dry_run else dataset.queries
    out: list[QueryFeatures] = []
    for q in queries:
        pool_keys = pools[q.query_id]
        pool_engine = EmotionalMemory(
            store=InMemoryStore(), embedder=embedder, config=_TIME_INVARIANT_CONFIG
        )
        try:
            pool_engine.import_memories([adapter._exported[k] for k in pool_keys])
            candidates = pool_engine._store.list_all()
            qa = adapter._appraiser.appraise(q.text).to_core_affect()
            emb = embedder.embed(q.text)
            feats = _pool_features(pool_engine, emb, qa, candidates, now=now)
            key_of = adapter._id_to_key
            gold = np.asarray(
                [1.0 if key_of.get(c.id) in q.gold_keys else 0.0 for c in candidates]
            )
            cand_keys = [key_of.get(c.id, "") for c in candidates]
            out.append(
                _make_qf(feats, gold, cand_keys, q.gold_keys, upstream_ndcg_at_k, PRIMARY_K)
            )
        finally:
            pool_engine.close()
    adapter.close()
    return out


def _make_qf(
    feats: np.ndarray,
    gold: np.ndarray,
    cand_ids: Sequence[Any],
    gold_set: Any,
    metric_at: Callable[[Any, Sequence[Any], int], float],
    k: int,
) -> QueryFeatures:
    """Bind a ranking->metric closure (captures cand_ids/gold_set by value)."""

    def metric(ranking: Sequence[int], _ids=tuple(cand_ids), _g=gold_set, _k=k) -> float:
        return metric_at(_g, [_ids[j] for j in ranking], _k)

    return QueryFeatures(features=feats, gold=gold, metric=metric)


# (key, builder, metric_label, role, dry_capable)
CORPORA: list[tuple[str, Callable[..., list[QueryFeatures]], str, str, bool]] = [
    ("madialbench", _madial_queries, "ndcg@5", "break", True),
    ("esmemeval", _esmemeval_queries, "u_ndcg@4", "break", True),
]


def _contrast(a: Sequence[float], b: Sequence[float], *, n_bootstrap: int, seed: int) -> dict:
    diff, lo, hi, p_two = paired_bootstrap_diff(a, b, n_bootstrap=n_bootstrap, seed=seed)
    p_one = p_two / 2.0 if diff >= 0 else 1.0 - p_two / 2.0
    return {
        "delta": diff,
        "ci_lower": lo,
        "ci_upper": hi,
        "p_onetail": p_one,
        "cohens_d": cohens_d_paired(a, b),
    }


def run_benchmark(
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    k: int = DEFAULT_K,
    seed: int = 0,
    dry_run: bool = False,
) -> dict[str, Any]:
    corpora = [
        (key, fn, m, role) for (key, fn, m, role, dry_ok) in CORPORA if dry_ok or not dry_run
    ]
    reports: dict[str, Any] = {}
    for key, fn, metric_label, role in corpora:
        print(f"[{key}] building queries + features …")
        queries = fn(dry_run=dry_run)
        heldout, w_mean, in_sample = cross_fit(queries, k=k)
        cosine = score_fixed(queries, COSINE_WEIGHTS)
        aft_fixed = score_fixed(queries, AFT_FIXED_WEIGHTS)
        reports[key] = {
            "metric": metric_label,
            "role": role,
            "n_queries": len(queries),
            "cosine_mean": float(np.mean(cosine)) if cosine else float("nan"),
            "aft_fixed_mean": float(np.mean(aft_fixed)) if aft_fixed else float("nan"),
            "aft_learned_mean": float(np.mean(heldout)) if heldout else float("nan"),
            "in_sample_mean": in_sample,
            "generalization_gap": (in_sample - float(np.mean(heldout)))
            if heldout
            else float("nan"),
            "learned_weights": [float(x) for x in w_mean],
            "s2_sign": "negative" if w_mean[1] < 0 else "non-negative",
            "learned_vs_cosine": _contrast(heldout, cosine, n_bootstrap=n_bootstrap, seed=seed),
            "learned_vs_aft_fixed": _contrast(
                heldout, aft_fixed, n_bootstrap=n_bootstrap, seed=seed
            ),
        }

    verdict = _verdict(reports)
    return {
        "benchmark": "addendum_z_learned_profile",
        "pre_registration": "benchmarks/preregistration_addendum_z_learned_profile.md",
        "dry_run": dry_run,
        "k": k,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "corpora": reports,
        "verdict": verdict,
    }


def _verdict(reports: dict[str, Any]) -> dict[str, Any]:
    """Holm m=3 over the Hz1 break family + Hz2 non-inferiority. Only evaluable
    when every pre-registered corpus is present (scored run + closure PR)."""
    have_break = [c for c in _HZ1_BREAK if c in reports]
    if len(have_break) < len(_HZ1_BREAK) or _HZ2_PRESERVE not in reports:
        return {
            "evaluable": False,
            "reason": "pending pre-registered corpora "
            f"(have break={have_break}, need {list(_HZ1_BREAK)} + {_HZ2_PRESERVE})",
        }
    ps = [reports[c]["learned_vs_cosine"]["p_onetail"] for c in _HZ1_BREAK]
    p_holm = holm_bonferroni(ps)
    hz1 = [
        {
            "corpus": c,
            "delta": reports[c]["learned_vs_cosine"]["delta"],
            "p_holm": ph,
            "pass": ph < 0.05 and reports[c]["learned_vs_cosine"]["delta"] > 0,
        }
        for c, ph in zip(_HZ1_BREAK, p_holm, strict=True)
    ]
    pres = reports[_HZ2_PRESERVE]
    hz2_delta = pres["aft_learned_mean"] - pres["aft_fixed_mean"]
    return {
        "evaluable": True,
        "family": "Holm m=3 (Hz1 break) + non-inferiority (Hz2 preserve)",
        "hz1_break": hz1,
        "hz1_any_pass": any(h["pass"] for h in hz1),
        "hz2_preserve": {"delta_vs_fixed": hz2_delta, "pass": hz2_delta >= -_NONINFERIORITY_EPS},
        "branch_a": any(h["pass"] for h in hz1),
    }


def write_results(results: dict[str, Any], *, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Addendum Z — Held-out learned retrieval profile",
        "",
        f"**k:** {results['k']}  **Bootstrap:** n={results['n_bootstrap']}, seed={results['seed']}"
        + ("  **[DRY RUN — not a scored result]**" if results["dry_run"] else ""),
        "",
        "| Corpus | metric | role | n | cosine | aft_fixed | aft_learned | gap | s2 "
        "| learned-cosine |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, r in results["corpora"].items():
        lc = r["learned_vs_cosine"]
        contrast = (
            f"Δ={lc['delta']:+.3f} [{lc['ci_lower']:+.3f},{lc['ci_upper']:+.3f}] "
            f"p={lc['p_onetail']:.4f}"
        )
        lines.append(
            f"| {key} | {r['metric']} | {r['role']} | {r['n_queries']} | "
            f"{r['cosine_mean']:.3f} | {r['aft_fixed_mean']:.3f} | {r['aft_learned_mean']:.3f} | "
            f"{r['generalization_gap']:+.3f} | {r['s2_sign']} | {contrast} |"
        )
    v = results["verdict"]
    lines += ["", "## Verdict", ""]
    if v.get("evaluable"):
        for h in v["hz1_break"]:
            lines.append(
                f"- Hz1 {h['corpus']}: Δ={h['delta']:+.3f} p_holm={h['p_holm']:.4f} → "
                f"{'PASS' if h['pass'] else 'FAIL'}"
            )
        lines.append(f"- Hz2 preserve: Δ_vs_fixed={v['hz2_preserve']['delta_vs_fixed']:+.3f}")
        lines.append(f"\n**Branch A (≥1 break PASS): {v['branch_a']}**")
    else:
        lines.append(f"_Not evaluable: {v.get('reason')}_")
    lines += ["", "Decision rule: `benchmarks/preregistration_addendum_z_learned_profile.md`."]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Addendum Z learned retrieval profile")
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="Smoke: dry-capable corpora, no LLM")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_benchmark(
        n_bootstrap=args.n_bootstrap, k=args.k, seed=args.seed, dry_run=args.dry_run
    )
    out_json, out_md = DEFAULT_OUT_JSON, DEFAULT_OUT_MD
    if args.dry_run:
        out_json = out_json.with_name("results.dry.json")
        out_md = out_md.with_name("results.dry.md")
    write_results(results, out_json=out_json, out_md=out_md)
    print(f"\nResults written to {out_json}")
    v = results["verdict"]
    print(
        f"Verdict: {'Branch A=' + str(v['branch_a']) if v.get('evaluable') else v.get('reason')}"
    )


if __name__ == "__main__":
    main()
