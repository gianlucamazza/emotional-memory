"""Exploratory wrong-person memory retrieval study on pinned ES-MemEval."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from benchmarks.common.similarity import cosine
from benchmarks.common.statistics import paired_bootstrap_diff
from benchmarks.esmemeval.dataset import (
    EsmemDataset,
    EsmemQuery,
    EsmemSession,
    build_pools,
    load_dataset,
)
from benchmarks.false_memory.appraisal_cache import CheckpointedAppraiser
from emotional_memory import (
    DIRECT_VAD_SCHEMA,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
)
from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal import AppraisalEngine
from emotional_memory.appraisal_llm import (
    KeywordAppraisalEngine,
    LLMAppraisalConfig,
    LLMAppraisalEngine,
)
from emotional_memory.decay import DecayConfig
from emotional_memory.embedders import SentenceTransformerEmbedder
from emotional_memory.interfaces import Embedder
from emotional_memory.llm_http import OpenAICompatibleLLMConfig, make_httpx_llm
from emotional_memory.mood import MoodField
from emotional_memory.retrieval import adaptive_weights

OUT = Path(__file__).parent
CAPABILITIES = (
    "information extraction",
    "user modeling",
    "temporal reasoning",
    "conflict detection",
)
ARMS = ("cosine", "aft", "no_mood", "no_proximity", "no_resonance")
N_PER_CAPABILITY = 50
SELECTION_SEED = 42
BOOTSTRAP_SEED = 0
N_BOOTSTRAP = 10_000
TOP_K = 4
RETRIEVE_K = 50
PROTOCOL = "benchmarks/preregistration_addendum_aa_false_memory.md"
APPRAISAL_CHECKPOINT = OUT / "appraisals.checkpoint.jsonl"
BASE_CONFIG = EmotionalMemoryConfig(
    decay=DecayConfig(base_decay=0.0, arousal_modulation=0.0, retrieval_boost=0.0),
    retrieval={"candidate_multiplier": 10},
    appraisal_max_concurrency=1,
)


class CachedEmbedder:
    """Share identical embedding vectors across all arms and pool conditions."""

    def __init__(self, model: SentenceTransformerEmbedder) -> None:
        self.model = model
        self.cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        if text not in self.cache:
            self.cache[text] = self.model.embed(text)
        return self.cache[text]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if missing:
            self.cache.update(zip(missing, self.model.embed_batch(missing), strict=True))
        return [self.cache[text] for text in texts]


@dataclass(frozen=True)
class DecoyCase:
    query: EsmemQuery
    donor: EsmemSession
    replaced_key: str
    decoy_key: str
    decoy_text: str
    clean_pool: tuple[str, ...]
    altered_pool: tuple[str, ...]


def select_queries(
    dataset: EsmemDataset, *, per_capability: int = N_PER_CAPABILITY
) -> list[EsmemQuery]:
    """Fixed, balanced sample; never condition selection on retrieval outcomes."""
    rng = Random(SELECTION_SEED)
    selected: list[EsmemQuery] = []
    for capability in CAPABILITIES:
        eligible = sorted(
            (q for q in dataset.queries if q.capability == capability), key=lambda q: q.query_id
        )
        if len(eligible) < per_capability:
            raise ValueError(f"{capability}: only {len(eligible)} eligible queries")
        selected.extend(rng.sample(eligible, per_capability))
    return sorted(selected, key=lambda q: q.query_id)


def seeker_name(session: EsmemSession) -> str:
    for line in session.text.splitlines():
        if ": " in line:
            speaker = line.split(": ", 1)[0]
            if speaker != "supporter":
                return speaker
    raise ValueError(f"session {session.key} has no seeker turn")


def transplant(donor: EsmemSession, target: EsmemSession) -> str:
    """Change only the speaker prefix, retaining every utterance byte-for-byte."""
    source_name = seeker_name(donor)
    target_name = seeker_name(target)
    return "\n".join(
        f"{target_name}: {line[len(source_name) + 2 :]}"
        if line.startswith(f"{source_name}: ")
        else line
        for line in donor.text.splitlines()
    )


def make_case(
    query: EsmemQuery,
    sessions: dict[str, EsmemSession],
    pool: tuple[str, ...],
    query_vector: list[float],
    vectors: dict[str, list[float]],
) -> DecoyCase:
    pool_keys = set(pool)
    donors = [
        s for s in sessions.values() if s.seeker_id != query.seeker_id and s.key not in pool_keys
    ]
    if not donors:
        raise ValueError(f"query {query.query_id}: no outside-pool donor")
    donor = sorted(donors, key=lambda s: (-cosine(query_vector, vectors[s.key]), s.key))[0]
    replaceable = [
        k for k in pool if sessions[k].seeker_id != query.seeker_id and k not in query.gold_keys
    ]
    if not replaceable:
        raise ValueError(f"query {query.query_id}: no cross-seeker candidate to replace")
    replaced = sorted(replaceable, key=lambda k: (cosine(query_vector, vectors[k]), k))[0]
    target = sessions[sorted(query.gold_keys)[0]]
    decoy_key = f"decoy:{query.query_id}"
    altered = tuple(decoy_key if k == replaced else k for k in pool)
    return DecoyCase(query, donor, replaced, decoy_key, transplant(donor, target), pool, altered)


def make_appraiser(*, dry_run: bool) -> AppraisalEngine:
    if dry_run:
        return KeywordAppraisalEngine()
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except ImportError:
        pass
    config = OpenAICompatibleLLMConfig.from_env()
    if config is None:
        raise RuntimeError("EMOTIONAL_MEMORY_LLM_API_KEY is required for a scored run")
    engine = LLMAppraisalEngine(
        llm=make_httpx_llm(config),
        config=LLMAppraisalConfig(
            cache_size=2048, fallback_on_error=False, appraisal_schema=DIRECT_VAD_SCHEMA
        ),
    )
    return CheckpointedAppraiser(engine, model=config.model, path=APPRAISAL_CHECKPOINT)


def ablation_weights() -> NDArray[np.float64]:
    rc = BASE_CONFIG.retrieval
    weights = adaptive_weights(MoodField.neutral(), rc.base_weights, rc.adaptive_weights_config)
    weights[2] = 0.0
    return cast("NDArray[np.float64]", weights / weights.sum())


def rank_aft(
    records: dict[str, dict[str, Any]],
    pool: tuple[str, ...],
    query: EsmemQuery,
    affect: CoreAffect,
    embedder: Embedder,
    arm: str,
) -> list[str]:
    config = BASE_CONFIG.model_copy(
        update={
            "enable_mood_signal": arm != "no_mood",
            "enable_resonance": arm != "no_resonance",
        }
    )
    engine = EmotionalMemory(store=InMemoryStore(), embedder=embedder, config=config)
    engine.import_memories([records[k] for k in pool])
    engine.reset_state()
    weights = ablation_weights() if arm == "no_proximity" else None
    ranked = engine.retrieve(
        query.text, top_k=RETRIEVE_K, query_affect=affect, precomputed_weights=weights
    )
    return [str(memory.metadata["key"]) for memory in ranked]


def rank_cosine(
    pool: tuple[str, ...], query_vector: list[float], vectors: dict[str, list[float]]
) -> list[str]:
    return sorted(pool, key=lambda k: (-cosine(query_vector, vectors[k]), k))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _interval(a: list[float], b: list[float]) -> dict[str, float]:
    delta, lower, upper, _ = paired_bootstrap_diff(
        a, b, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )
    return {"delta": delta, "ci_lower": lower, "ci_upper": upper}


def run(*, dry_run: bool = False, limit: int | None = None) -> dict[str, Any]:
    dataset = load_dataset()
    selected = select_queries(dataset)
    if dry_run:
        selected = selected[:2]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("empty query sample")
    sessions = {s.key: s for s in dataset.sessions}
    pools = build_pools(dataset)
    embedder = CachedEmbedder(SentenceTransformerEmbedder.make_bge_small())
    appraiser = make_appraiser(dry_run=dry_run)
    bank = EmotionalMemory(
        store=InMemoryStore(),
        embedder=embedder,
        appraisal_engine=appraiser,
        config=BASE_CONFIG,
    )
    try:
        print(f"Encoding {len(sessions)} original sessions", flush=True)
        bank.encode_batch(
            [session.text for session in dataset.sessions],
            metadata=[{"key": session.key} for session in dataset.sessions],
        )
        records = {str(record["metadata"]["key"]): record for record in bank.export_memories()}
        vectors = {key: list(record["embedding"]) for key, record in records.items()}
        query_vectors = embedder.embed_batch([query.text for query in selected])
        cases = [
            make_case(query, sessions, pools[query.query_id], vector, vectors)
            for query, vector in zip(selected, query_vectors, strict=True)
        ]
        embedder.embed_batch([case.decoy_text for case in cases])
        observations: list[dict[str, Any]] = []
        for index, (query, query_vector, case) in enumerate(
            zip(selected, query_vectors, cases, strict=True), 1
        ):
            if index == 1 or index % 20 == 0:
                print(f"Scoring query {index}/{len(selected)}", flush=True)
            decoy_engine = EmotionalMemory(
                store=InMemoryStore(),
                embedder=embedder,
                appraisal_engine=appraiser,
                config=BASE_CONFIG,
            )
            decoy_engine.encode(case.decoy_text, metadata={"key": case.decoy_key})
            decoy_record = decoy_engine.export_memories()[0]
            records[case.decoy_key] = decoy_record
            vectors[case.decoy_key] = list(decoy_record["embedding"])
            affect = appraiser.appraise(query.text).to_core_affect()
            ranks: dict[str, dict[str, Any]] = {}
            for arm in ARMS:
                ranks[arm] = {}
                for condition, pool in (
                    ("clean", case.clean_pool),
                    ("altered", case.altered_pool),
                ):
                    ranked = (
                        rank_cosine(pool, query_vector, vectors)
                        if arm == "cosine"
                        else rank_aft(records, pool, query, affect, embedder, arm)
                    )
                    ranks[arm][condition] = {
                        "decoy_rank": ranked.index(case.decoy_key) + 1
                        if case.decoy_key in ranked
                        else None,
                        "gold_recall_4": len(set(ranked[:TOP_K]) & query.gold_keys)
                        / len(query.gold_keys),
                        "top_4": ranked[:TOP_K],
                    }
            observations.append(
                {
                    "query_id": query.query_id,
                    "capability": query.capability,
                    "target_seeker": query.seeker_id,
                    "donor_session": case.donor.key,
                    "replaced_session": case.replaced_key,
                    "decoy_key": case.decoy_key,
                    "gold_keys": sorted(query.gold_keys),
                    "query_affect": affect.model_dump(),
                    "decoy_affect": decoy_record["tag"]["core_affect"],
                    "ranks": ranks,
                    "review": {
                        "query": query.text,
                        "donor_text": case.donor.text,
                        "decoy_text": case.decoy_text,
                    }
                    if index <= 20
                    else None,
                }
            )
            del records[case.decoy_key]
            del vectors[case.decoy_key]

        def decoy_hits(arm: str, k: int) -> list[float]:
            return [
                float((o["ranks"][arm]["altered"]["decoy_rank"] or 51) <= k) for o in observations
            ]

        def gold_recall(arm: str, condition: str) -> list[float]:
            return [float(o["ranks"][arm][condition]["gold_recall_4"]) for o in observations]

        comparisons = {"aft_vs_cosine": _interval(decoy_hits("aft", 4), decoy_hits("cosine", 4))}
        comparisons.update(
            {
                f"aft_vs_{arm}": _interval(decoy_hits("aft", 4), decoy_hits(arm, 4))
                for arm in ARMS[2:]
            }
        )
        rates = {
            arm: {
                "decoy_at_4": _mean(decoy_hits(arm, 4)),
                "decoy_at_1": _mean(decoy_hits(arm, 1)),
                "gold_recall_4_clean": _mean(gold_recall(arm, "clean")),
                "gold_recall_4_altered": _mean(gold_recall(arm, "altered")),
            }
            for arm in ARMS
        }
        by_capability: dict[str, dict[str, Any]] = defaultdict(dict)
        for capability in CAPABILITIES:
            subset = [o for o in observations if o["capability"] == capability]
            if subset:
                by_capability[capability] = {
                    "n": len(subset),
                    **{
                        arm: _mean(
                            [
                                float((o["ranks"][arm]["altered"]["decoy_rank"] or 51) <= 4)
                                for o in subset
                            ]
                        )
                        for arm in ARMS
                    },
                }
        return {
            "protocol": PROTOCOL,
            "dry_run": dry_run,
            "partial": len(selected) != 200,
            "n": len(selected),
            "query_ids": [q.query_id for q in selected],
            "selection_seed": SELECTION_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_bootstrap": N_BOOTSTRAP,
            "rates": rates,
            "paired_contrasts": comparisons,
            "by_capability": by_capability,
            "observations": observations,
        }
    finally:
        close = getattr(appraiser, "close", None)
        if callable(close):
            close()


def write_report(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Addendum AA — wrong-person memory retrieval",
        "",
        "Exploratory; no PASS/FAIL claim.",
        "",
        f"N={result['n']} · dry_run={result['dry_run']} · partial={result['partial']}",
        "",
        "| Arm | Decoy@4 | Decoy@1 | Gold recall@4 clean | Gold recall@4 altered |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm, rates in result["rates"].items():
        lines.append(
            f"| {arm} | {rates['decoy_at_4']:.3f} | {rates['decoy_at_1']:.3f} | "
            f"{rates['gold_recall_4_clean']:.3f} | {rates['gold_recall_4_altered']:.3f} |"
        )
    lines.extend(["", "## Paired decoy@4 contrasts", ""])
    for name, ci in result["paired_contrasts"].items():
        lines.append(
            f"- {name}: Δ={ci['delta']:+.3f} [{ci['ci_lower']:+.3f}, {ci['ci_upper']:+.3f}]"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=OUT / "results.json")
    args = parser.parse_args()
    if args.limit is not None and not args.dry_run:
        raise SystemExit("--limit is only for --dry-run; scored runs use all 200 queries")
    result = run(dry_run=args.dry_run, limit=args.limit)
    write_report(result, args.out)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
