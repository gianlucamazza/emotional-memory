"""LoCoMo query-routing benchmark runner (Addendum L).

Evaluates five systems by default:
  aft_routed_heuristic  — AFT + HeuristicQueryClassifier + LOCOMO_ROUTING
  aft_W0                — AFT fixed W0 (S1 baseline)
  aft_W2                — AFT fixed W2 (best Addendum J config)
  naive_rag             — semantic-only (S1 baseline)
  aft_oracle_routed     — ground-truth query-type routing (upper bound)

Optional (slow — one LLM call per retrieve):
  aft_routed_llm        — AFT + LLMQueryClassifier + LOCOMO_ROUTING

Usage::

    uv run python -m benchmarks.locomo.routing_runner
    uv run python -m benchmarks.locomo.routing_runner --subset 200qa --seed 42
    uv run python -m benchmarks.locomo.routing_runner --with-llm-classifier

Environment variables: same as benchmarks/locomo/runner.py.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

from benchmarks.locomo.adapters.base import LoCoMoAdapter, call_llm
from benchmarks.locomo.adapters.naive_rag import NaiveRAGLoCoMoAdapter
from benchmarks.locomo.dataset import Conversation, QAPair, Session, load_dataset
from benchmarks.locomo.scoring import (
    build_judge_prompt,
    is_adversarial_correct,
    parse_judge_response,
    score_predictions,
)
from emotional_memory import (
    LOCOMO_ROUTING,
    EmotionalMemory,
    EmotionalMemoryConfig,
    HeuristicQueryClassifier,
    InMemoryStore,
    LLMQueryClassifier,
    QueryClassifier,
)
from emotional_memory.appraisal_llm import KeywordAppraisalEngine
from emotional_memory.embedders import SentenceTransformerEmbedder
from emotional_memory.retrieval import QueryClassifierConfig, RetrievalConfig

_HERE = Path(__file__).parent
DEFAULT_OUT_JSON = _HERE / "routing_results.json"
DEFAULT_OUT_MD = _HERE / "routing_results.md"
DEFAULT_CHECKPOINT = _HERE / "routing_results.checkpoint.jsonl"

_TOP_K = 8

# Weight configs from Addendum J closure
_W0 = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]  # S1 baseline
_W2 = [0.50, 0.30, 0.10, 0.05, 0.05, 0.00]  # Best Addendum J config

# Key used in flat-list dicts for stratified sampling and prediction records.
# Populated from QAPair.category_name at run time (not a QAPair attribute).
_QA_TYPE_FIELD = "question_type"


def _load_checkpoint(
    checkpoint_path: Path,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Return {(system, sample_id): judged_predictions} from a JSONL checkpoint."""
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not checkpoint_path.exists():
        return result
    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entry: dict[str, Any] = json.loads(line)
            result[(entry["system"], entry["sample_id"])] = entry["predictions"]
    return result


def _append_checkpoint(
    checkpoint_path: Path,
    system: str,
    sample_id: str,
    predictions: list[dict[str, Any]],
) -> None:
    """Append one completed (system, conversation) record to the checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps({"system": system, "sample_id": sample_id, "predictions": predictions})
            + "\n"
        )


class _LoggingClassifier:
    """Wraps a QueryClassifier to record (query, predicted_type) pairs for Hl3."""

    __slots__ = ("_inner", "log")

    def __init__(self, inner: QueryClassifier) -> None:
        self._inner = inner
        self.log: list[tuple[str, str]] = []

    def classify(self, query: str) -> str:
        pred = self._inner.classify(query)
        self.log.append((query, pred))
        return pred


class _AFTRoutedAdapter(LoCoMoAdapter):
    """AFT adapter with optional query-type classifier routing."""

    def __init__(
        self,
        *,
        name: str,
        config: EmotionalMemoryConfig,
        query_classifier: QueryClassifier | None = None,
        top_k: int = _TOP_K,
    ) -> None:
        self.name = name
        self._config = config
        self._top_k = top_k
        self._embedder = SentenceTransformerEmbedder.make_bge_small()
        self._engine: EmotionalMemory | None = None
        # Wrap with logger to record (query, predicted_type) for Hl3 analysis.
        if query_classifier is not None:
            self._logging_clf: _LoggingClassifier | None = _LoggingClassifier(query_classifier)
            self._query_classifier: QueryClassifier | None = self._logging_clf
        else:
            self._logging_clf = None
            self._query_classifier = None

    def reset(self) -> None:
        if self._engine is not None:
            self._engine.close()
        self._engine = EmotionalMemory(
            store=InMemoryStore(),
            embedder=self._embedder,
            appraisal_engine=KeywordAppraisalEngine(),
            config=self._config,
            query_classifier=self._query_classifier,
        )

    def ingest_session(self, session: Session, conversation: Conversation) -> None:
        engine = self._require_engine()
        for turn in session.turns:
            content = f"{turn.speaker}: {turn.text}"
            engine.encode(
                content,
                metadata={
                    "dia_id": turn.dia_id,
                    "speaker": turn.speaker,
                    "session": session.session_num,
                    "date": session.date_time,
                },
            )

    def answer(self, qa: QAPair, conversation: Conversation) -> str:
        engine = self._require_engine()
        retrieved = engine.retrieve(qa.question, top_k=self._top_k)
        context = "\n".join(f"- {mem.content}" for mem in retrieved)
        prompt = f"Conversation excerpts:\n{context}\n\nQuestion: {qa.question}\n\nAnswer:"
        return call_llm(prompt)

    def _require_engine(self) -> EmotionalMemory:
        if self._engine is None:
            raise RuntimeError("Call reset() before ingest_session().")
        return self._engine

    def __del__(self) -> None:
        import contextlib

        if self._engine is not None:
            with contextlib.suppress(Exception):
                self._engine.close()


def _make_system(name: str) -> LoCoMoAdapter:
    if name == "aft_routed_heuristic":
        qcc = QueryClassifierConfig(mode="heuristic", routed_weights=LOCOMO_ROUTING)
        config = EmotionalMemoryConfig(retrieval=RetrievalConfig(query_classifier=qcc))
        return _AFTRoutedAdapter(
            name=name,
            config=config,
            query_classifier=HeuristicQueryClassifier(),
        )

    if name == "aft_routed_llm":
        # LLMQueryClassifier uses the same LLM client as the answer step.
        # Import make_httpx_llm lazily so the runner still works without LLM key
        # for smoke-testing non-LLM systems.
        from emotional_memory.llm_http import OpenAICompatibleLLMConfig, make_httpx_llm

        llm_config = OpenAICompatibleLLMConfig.from_env()
        if llm_config is None:
            raise RuntimeError("EMOTIONAL_MEMORY_LLM_API_KEY is required for aft_routed_llm")
        llm = make_httpx_llm(llm_config)
        clf = LLMQueryClassifier(llm=llm, cache_size=512)
        qcc = QueryClassifierConfig(mode="llm", routed_weights=LOCOMO_ROUTING)
        config = EmotionalMemoryConfig(retrieval=RetrievalConfig(query_classifier=qcc))
        return _AFTRoutedAdapter(name=name, config=config, query_classifier=clf)

    if name == "aft_W0":
        config = EmotionalMemoryConfig(
            retrieval=RetrievalConfig(base_weights=_W0),
        )
        return _AFTRoutedAdapter(name=name, config=config)

    if name == "aft_W2":
        config = EmotionalMemoryConfig(
            retrieval=RetrievalConfig(base_weights=_W2),
        )
        return _AFTRoutedAdapter(name=name, config=config)

    if name == "naive_rag":
        return NaiveRAGLoCoMoAdapter()

    if name == "aft_oracle_routed":
        # Oracle: routes by ground-truth question_type; injected at answer() time.
        # Implemented as a special adapter below.
        return _OracleRoutedAdapter()

    raise ValueError(f"Unknown system: {name!r}")


class _OracleRoutedAdapter(_AFTRoutedAdapter):
    """AFT adapter that selects weights from the ground-truth question type."""

    def __init__(self) -> None:
        qcc = QueryClassifierConfig(mode="heuristic", routed_weights=LOCOMO_ROUTING)
        config = EmotionalMemoryConfig(retrieval=RetrievalConfig(query_classifier=qcc))
        super().__init__(name="aft_oracle_routed", config=config)

    def answer(self, qa: QAPair, conversation: Conversation) -> str:
        # Inject ground-truth type into HeuristicQueryClassifier by monkey-patching
        # the classify call via a fixed-response classifier.
        gt_type: str = qa.category_name if qa.category_name in LOCOMO_ROUTING else "default"

        engine = self._require_engine()
        # Temporarily override with an oracle classifier
        orig_clf = engine._query_classifier  # type: ignore[attr-defined]

        class _OracleCLF:
            def classify(self, query: str) -> str:
                return gt_type

        engine._query_classifier = _OracleCLF()  # type: ignore[attr-defined]
        try:
            retrieved = engine.retrieve(qa.question, top_k=self._top_k)
        finally:
            engine._query_classifier = orig_clf  # type: ignore[attr-defined]

        context = "\n".join(f"- {mem.content}" for mem in retrieved)
        prompt = f"Conversation excerpts:\n{context}\n\nQuestion: {qa.question}\n\nAnswer:"
        return call_llm(prompt)


DEFAULT_SYSTEMS = [
    "aft_routed_heuristic",
    "aft_W0",
    "aft_W2",
    "naive_rag",
    "aft_oracle_routed",
]


def _stratified_subsample(all_qa: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Proportional stratified sample by question_type."""
    from collections import defaultdict

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in all_qa:
        by_type[item.get(_QA_TYPE_FIELD, "open_domain")].append(item)

    rng = random.Random(seed)
    total = len(all_qa)
    sampled: list[dict[str, Any]] = []
    for items in by_type.values():
        k = max(1, round(n * len(items) / total))
        sampled.extend(rng.sample(items, min(k, len(items))))
    rng.shuffle(sampled)
    return sampled[:n]


def _run_system(
    system: LoCoMoAdapter,
    dataset: Any,
    *,
    subset: list[dict[str, Any]] | None = None,
    verbose: bool = False,
    checkpoint: Path | None = None,
    done: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    subset_ids = {item["qa_id"] for item in subset} if subset is not None else None
    done = done or {}

    conversations = list(dataset.conversations)
    conv_iter = tqdm(
        conversations,
        desc=f"[{system.name}] conversations",
        unit="conv",
        disable=not verbose,
    )
    for conv in conv_iter:
        key = (system.name, conv.sample_id)
        if key in done:
            if verbose:
                print(f"  [{system.name}] {conv.sample_id} — skipped (checkpoint)")
            predictions.extend(done[key])
            continue

        system.reset()
        for session in conv.sessions:
            system.ingest_session(session, conv)

        conv_predictions: list[dict[str, Any]] = []
        qa_iter = tqdm(
            conv.qa_pairs,
            desc=f"[{system.name}] {conv.sample_id} QA",
            unit="qa",
            leave=False,
            disable=not verbose,
        )
        for qa_idx, qa in enumerate(qa_iter):
            qa_id = f"{conv.sample_id}_{qa_idx}"
            if subset_ids is not None and qa_id not in subset_ids:
                continue
            try:
                pred = system.answer(qa, conv)
            except Exception as exc:
                pred = ""
                if verbose:
                    print(f"  [{system.name}] error on {qa_id}: {exc}")
            conv_predictions.append(
                {
                    "qa_id": qa_id,
                    "question": qa.question,
                    "gold": qa.answer,
                    "prediction": pred,
                    "category": qa.category,
                    _QA_TYPE_FIELD: qa.category_name,
                    "is_adversarial": qa.is_adversarial,
                }
            )
        predictions.extend(conv_predictions)

        if checkpoint is not None:
            _append_checkpoint(checkpoint, system.name, conv.sample_id, conv_predictions)
    return predictions


def _judge_predictions(predictions: list[dict[str, Any]], *, verbose: bool = False) -> None:
    for i, pred in enumerate(tqdm(predictions, desc="Judging", unit="qa", disable=not verbose)):
        if pred.get("is_adversarial"):
            pred["judge_correct"] = is_adversarial_correct(pred["prediction"])
            continue
        prompt = build_judge_prompt(
            str(pred["question"]), str(pred["gold"]), str(pred["prediction"])
        )
        try:
            response = call_llm(prompt, temperature=0.0)
            pred["judge_correct"] = parse_judge_response(response)
        except Exception as exc:
            if verbose:
                print(f"  Judge error on item {i}: {exc}")
            pred["judge_correct"] = None


def _format_table(results: dict[str, Any]) -> str:
    lines = [
        "# Addendum L — Query-Routing Results",
        "",
        "| System | single_hop F1 | multi_hop F1 | temporal F1 | open_domain F1 | Weighted F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sys_name, sys_data in results.items():
        cats = sys_data.get("by_category", {})
        weighted = sys_data.get("aggregate_weighted_f1", 0.0)
        row = [
            sys_name,
            f"{cats.get('single_hop', {}).get('f1', 0.0):.3f}",
            f"{cats.get('multi_hop', {}).get('f1', 0.0):.3f}",
            f"{cats.get('temporal', {}).get('f1', 0.0):.3f}",
            f"{cats.get('open_domain', {}).get('f1', 0.0):.3f}",
            f"{weighted:.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Addendum L — query-routing runner")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=DEFAULT_SYSTEMS,
        help="Systems to evaluate",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="'200qa' for stratified 200-question smoke-test",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--with-llm-classifier",
        action="store_true",
        help="Include aft_routed_llm (slow — one LLM call per retrieve)",
    )
    args = parser.parse_args(argv)

    # Auto-load .env if python-dotenv is available (mirrors Makefile behaviour).
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    systems = list(args.systems)
    if args.with_llm_classifier:
        systems.append("aft_routed_llm")

    dataset = load_dataset()
    print(f"Dataset: {len(dataset.conversations)} conversations")

    # Build optional subset
    if args.subset == "200qa":
        all_qa_flat = [
            {"qa_id": f"{conv.sample_id}_{qa_idx}", _QA_TYPE_FIELD: qa.category_name}
            for conv in dataset.conversations
            for qa_idx, qa in enumerate(conv.qa_pairs)
        ]
        subset_qa = _stratified_subsample(all_qa_flat, 200, args.seed)
        print(f"Subset: {len(subset_qa)} questions (stratified, seed={args.seed})")
    else:
        subset_qa = None

    checkpoint = None if args.no_checkpoint else args.checkpoint
    done = _load_checkpoint(checkpoint) if checkpoint else {}
    if done and args.verbose:
        print(f"  Resuming: {len(done)} conversation(s) already in checkpoint.")

    # Resume partial results from previous incremental writes.
    all_results: dict[str, Any] = {}
    if args.out_json.exists():
        try:
            all_results = json.loads(args.out_json.read_text())
            if args.verbose:
                print(f"  Resuming results from {args.out_json} ({len(all_results)} systems)")
        except json.JSONDecodeError:
            all_results = {}

    for sys_name in systems:
        if sys_name in all_results:
            if args.verbose:
                print(f"\n=== {sys_name} ===  (skipped — already in {args.out_json})")
            continue

        print(f"\n=== {sys_name} ===")
        system = _make_system(sys_name)
        preds = _run_system(
            system,
            dataset,
            subset=subset_qa,
            verbose=args.verbose,
            checkpoint=checkpoint,
            done=done,
        )
        _judge_predictions(preds, verbose=args.verbose)
        scores = score_predictions(preds)

        # B1: compute sample-weighted F1 from per-category counts (adversarial excluded).
        cats = scores.get("by_category", {})
        total_n = sum(c.get("n", 0) for c in cats.values())
        scores["aggregate_weighted_f1"] = (
            sum(c.get("n", 0) * c.get("f1", 0.0) for c in cats.values()) / total_n
            if total_n > 0
            else 0.0
        )

        # B2: persist raw predictions for paired bootstrap.
        result_entry: dict[str, Any] = {**scores, "predictions": preds}

        # B3: persist classifier log for Hl3 accuracy measurement.
        logging_clf = getattr(system, "_logging_clf", None)
        if logging_clf is not None:
            clf_entries = logging_clf.log
            result_entry["classifier_predictions"] = [
                {
                    "qa_id": preds[i]["qa_id"],
                    "predicted_type": (clf_entries[i][1] if i < len(clf_entries) else "unknown"),
                    "ground_truth_type": preds[i].get(_QA_TYPE_FIELD, "unknown"),
                }
                for i in range(len(preds))
            ]

        all_results[sys_name] = result_entry
        print(f"  Weighted F1: {scores['aggregate_weighted_f1']:.3f}")

        # Incremental write after each system so a crash doesn't lose everything.
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(all_results, indent=2))
        if args.verbose:
            print(f"  -> incremental write to {args.out_json}")

    # Write final summary MD
    args.out_md.write_text(_format_table(all_results))
    print(f"\nResults -> {args.out_json}")
    print(f"Summary -> {args.out_md}")


if __name__ == "__main__":
    main()
