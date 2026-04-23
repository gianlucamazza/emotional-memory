"""Helpers for building and summarizing the human-evaluation pilot."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any, Literal

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PILOT = ROOT / "benchmarks" / "human_eval" / "pilot_v1.json"
DEFAULT_RESULTS = ROOT / "benchmarks" / "realistic" / "results.json"
DEFAULT_PACKETS = ROOT / "benchmarks" / "human_eval" / "packets.json"
DEFAULT_TEMPLATE = ROOT / "benchmarks" / "human_eval" / "ratings_template.jsonl"
DEFAULT_RATINGS = ROOT / "benchmarks" / "human_eval" / "ratings.jsonl"
DEFAULT_SUMMARY_JSON = ROOT / "benchmarks" / "human_eval" / "summary.json"
DEFAULT_SUMMARY_MD = ROOT / "benchmarks" / "human_eval" / "summary.md"
DEFAULT_CONDITIONS = ["aft", "naive_cosine"]


def load_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def build_packets(
    *,
    pilot_path: Path = DEFAULT_PILOT,
    results_path: Path = DEFAULT_RESULTS,
    conditions: list[str] | None = None,
) -> dict[str, Any]:
    pilot = load_json(pilot_path)
    results = load_json(results_path)
    selected_conditions = conditions or DEFAULT_CONDITIONS
    scenario_by_system: dict[tuple[str, str], dict[str, Any]] = {}
    for system in results["systems"]:
        for scenario in system["scenarios"]:
            scenario_by_system[(scenario["scenario_id"], system["system"])] = scenario

    packets = []
    for scenario in pilot["scenarios"]:
        condition_payloads = []
        for condition in selected_conditions:
            scenario_payload = scenario_by_system.get((scenario["scenario_id"], condition))
            if scenario_payload is None:
                continue
            condition_payloads.append(
                {
                    "condition": condition,
                    "aggregate_metrics": scenario_payload["metrics"],
                    "query_cards": _build_query_cards(scenario_payload),
                }
            )
        packets.append(
            {
                "scenario_id": scenario["scenario_id"],
                "prompt": scenario["prompt"],
                "rating_dimensions": pilot["rating_dimensions"],
                "conditions": condition_payloads,
            }
        )
    return {
        "name": pilot["name"],
        "description": pilot["description"],
        "rating_dimensions": pilot["rating_dimensions"],
        "conditions": selected_conditions,
        "packet_count": len(packets),
        "packets": packets,
    }


def _build_query_cards(scenario_payload: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for session in scenario_payload["sessions"]:
        for query in session["queries"]:
            top_result = query["results"][0] if query["results"] else None
            cards.append(
                {
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "challenge_type": query["challenge_type"],
                    "top_result_memory_alias": (
                        None if top_result is None else top_result.get("memory_alias")
                    ),
                    "top_result_content": (
                        None if top_result is None else top_result.get("content")
                    ),
                    "retrieved_memory_aliases": query["retrieved_memory_aliases"],
                }
            )
    return cards


def write_packets(
    payload: dict[str, Any],
    *,
    out_path: Path = DEFAULT_PACKETS,
    template_path: Path = DEFAULT_TEMPLATE,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    template_lines = [
        json.dumps(
            {
                "scenario_id": packet["scenario_id"],
                "condition": condition["condition"],
                "rater_id": "",
                "ratings": dict.fromkeys(packet["rating_dimensions"], None),
                "note": "",
            }
        )
        for packet in payload["packets"]
        for condition in packet["conditions"]
    ]
    template_payload = "\n".join(template_lines) + ("\n" if template_lines else "")
    template_path.write_text(template_payload, encoding="utf-8")


def krippendorff_alpha(
    reliability_data: Sequence[Sequence[float | None]],
    *,
    level: Literal["nominal", "ordinal"] = "ordinal",
) -> float | None:
    """Krippendorff's alpha inter-rater reliability coefficient.

    Uses the coincidence-matrix formulation (Krippendorff 2011).

    Parameters
    ----------
    reliability_data:
        Matrix[unit][observer] = rating value, or None for missing.
    level:
        Measurement level: "ordinal" (default, Likert) or "nominal".

    Returns
    -------
    float | None
        Alpha, or None when undetermined (< 2 units with ≥ 2 observers,
        or all observations are identical).
    """
    units: list[list[float]] = [[v for v in unit if v is not None] for unit in reliability_data]
    pairable = [u for u in units if len(u) >= 2]
    if not pairable:
        return None

    all_vals = sorted({v for u in pairable for v in u})
    if len(all_vals) < 2:
        return 1.0  # all observations identical — perfect agreement

    val_to_idx = {v: i for i, v in enumerate(all_vals)}
    n_vals = len(all_vals)

    # Build coincidence matrix C[v][w] via ordered pairs within each unit
    C = [[0.0] * n_vals for _ in range(n_vals)]
    for unit in pairable:
        m = len(unit)
        w = 1.0 / (m - 1)
        for o_idx in range(m):
            for p_idx in range(m):
                if o_idx != p_idx:
                    v = val_to_idx[unit[o_idx]]
                    u_val = val_to_idx[unit[p_idx]]
                    C[v][u_val] += w

    # Marginal counts (equal to raw observation counts per value)
    nc = [sum(C[v][w] for w in range(n_vals)) for v in range(n_vals)]
    N = sum(nc)
    if N < 2:
        return None

    def _delta_sq(v: int, w: int) -> float:
        if level == "ordinal":
            lo, hi = min(v, w), max(v, w)
            inner = sum(nc[g] for g in range(lo, hi + 1))
            gap = inner - (nc[lo] + nc[hi]) / 2.0
            return gap * gap
        return 0.0 if v == w else 1.0  # nominal

    D_o = sum(C[v][w] * _delta_sq(v, w) for v in range(n_vals) for w in range(n_vals) if v != w)
    D_e_numerator = sum(
        nc[v] * nc[w] * _delta_sq(v, w) for v in range(n_vals) for w in range(n_vals) if v != w
    )
    D_e = D_e_numerator / (N - 1)

    if D_e < 1e-10:
        return 1.0
    return 1.0 - D_o / D_e


def load_ratings(path: Path = DEFAULT_RATINGS) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"ratings file not found: {path}. Copy and fill {DEFAULT_TEMPLATE.name} first."
        )
    records = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed line %d in %s: %s", i, path.name, exc)
    return records


def _classify_rating_record(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    scenario_id = str(record.get("scenario_id", "")).strip()
    condition = str(record.get("condition", "")).strip()
    rater_id = str(record.get("rater_id", "")).strip()
    note = str(record.get("note", "")).strip()
    ratings_map = record.get("ratings", {})

    numeric_ratings = {
        dimension: float(value)
        for dimension, value in ratings_map.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    missing_dimensions = [
        dimension
        for dimension, value in ratings_map.items()
        if not isinstance(value, (int, float)) or isinstance(value, bool)
    ]
    is_template = not rater_id and not note and not numeric_ratings

    payload = {
        "scenario_id": scenario_id,
        "condition": condition,
        "rater_id": rater_id,
        "ratings": numeric_ratings,
        "missing_dimensions": missing_dimensions,
        "note": note,
    }
    if is_template:
        return "template", payload
    if not scenario_id or not condition:
        return "invalid", {**payload, "reason": "missing scenario_id or condition"}
    if not rater_id:
        return "invalid", {**payload, "reason": "missing rater_id"}
    if not numeric_ratings:
        return "incomplete", {**payload, "reason": "no completed rating dimensions"}
    if missing_dimensions:
        return "incomplete", {**payload, "reason": "missing rating dimensions"}
    return "complete", payload


def summarize_ratings(ratings: list[dict[str, Any]]) -> dict[str, Any]:
    condition_dimension_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    condition_counts: dict[str, int] = defaultdict(int)
    scenario_counts: dict[tuple[str, str], int] = defaultdict(int)
    notes: list[dict[str, str]] = []
    complete_count = 0
    invalid_records: list[dict[str, Any]] = []
    incomplete_records: list[dict[str, Any]] = []
    template_record_count = 0
    unique_raters: set[str] = set()

    # Tracks per-dimension, per-(scenario, condition), per-rater ratings for Krippendorff alpha
    _dim_item_rater: dict[str, dict[tuple[str, str], dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for record in ratings:
        status, payload = _classify_rating_record(record)
        if status == "template":
            template_record_count += 1
            continue
        if status == "invalid":
            invalid_records.append(payload)
            continue
        if status == "incomplete":
            incomplete_records.append(payload)
            continue

        complete_count += 1
        scenario_id = payload["scenario_id"]
        condition = payload["condition"]
        rater_id = payload["rater_id"]
        unique_raters.add(rater_id)
        condition_counts[condition] += 1
        scenario_counts[(scenario_id, condition)] += 1
        for dimension, value in payload["ratings"].items():
            condition_dimension_scores[condition][dimension].append(value)
            _dim_item_rater[dimension][(scenario_id, condition)][rater_id] = value
        note = payload["note"]
        if note:
            notes.append(
                {
                    "scenario_id": scenario_id,
                    "condition": condition,
                    "note": note,
                }
            )

    conditions = []
    for condition, dimensions in sorted(condition_dimension_scores.items()):
        dimension_means = {
            dimension: round(mean(values), 3)
            for dimension, values in sorted(dimensions.items())
            if values
        }
        conditions.append(
            {
                "condition": condition,
                "ratings_count": condition_counts[condition],
                "dimension_means": dimension_means,
            }
        )

    if complete_count == 0:
        raise ValueError(
            "No completed ratings found. Fill ratings.jsonl from ratings_template.jsonl before "
            "running the summary."
        )

    # Compute Krippendorff's alpha per dimension (ordinal, Likert scale)
    krippendorff_by_dim: dict[str, float | None] = {}
    for dim, item_rater in _dim_item_rater.items():
        all_raters = sorted({r for rv in item_rater.values() for r in rv})
        matrix: list[list[float | None]] = [
            [item_rater[item].get(rater) for rater in all_raters] for item in sorted(item_rater)
        ]
        krippendorff_by_dim[dim] = krippendorff_alpha(matrix)

    return {
        "status": "complete",
        "completed_ratings_count": complete_count,
        "unique_rater_count": len(unique_raters),
        "template_record_count": template_record_count,
        "incomplete_records": incomplete_records,
        "invalid_records": invalid_records,
        "conditions": conditions,
        "scenario_counts": [
            {
                "scenario_id": scenario_id,
                "condition": condition,
                "count": count,
            }
            for (scenario_id, condition), count in sorted(scenario_counts.items())
        ],
        "notes": notes,
        "krippendorff_alpha_by_dimension": krippendorff_by_dim,
    }


def write_summary(
    summary: dict[str, Any],
    *,
    out_json: Path = DEFAULT_SUMMARY_JSON,
    out_md: Path = DEFAULT_SUMMARY_MD,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Human Evaluation Pilot Summary",
        "",
        f"Completed ratings: {summary['completed_ratings_count']}",
        f"Unique raters: {summary['unique_rater_count']}",
        f"Template-only records ignored: {summary['template_record_count']}",
        f"Incomplete records: {len(summary['incomplete_records'])}",
        f"Invalid records: {len(summary['invalid_records'])}",
        "",
        "| Condition | Ratings | Dimension Means |",
        "|---|---:|---|",
    ]
    for condition in summary["conditions"]:
        formatted = ", ".join(
            f"{dimension}={value:.2f}" for dimension, value in condition["dimension_means"].items()
        )
        lines.append(
            f"| `{condition['condition']}` | {condition['ratings_count']} | {formatted} |"
        )

    alpha_by_dim = summary.get("krippendorff_alpha_by_dimension", {})
    if alpha_by_dim:
        lines += [
            "",
            "## Inter-Rater Agreement (Krippendorff's alpha, ordinal)",
            "",
            "| Dimension | alpha |",
            "|---|---:|",
        ]
        for dim, alpha_val in sorted(alpha_by_dim.items()):
            val_str = f"{alpha_val:.3f}" if alpha_val is not None else "—"
            lines.append(f"| `{dim}` | {val_str} |")

    if summary["incomplete_records"]:
        lines.extend(["", "## Incomplete Records", ""])
        lines.extend(
            f"- `{item['scenario_id']}` / `{item['condition']}` / `{item['rater_id']}`: "
            f"{item['reason']}"
            for item in summary["incomplete_records"]
        )
    if summary["invalid_records"]:
        lines.extend(["", "## Invalid Records", ""])
        lines.extend(
            f"- `{item['scenario_id']}` / `{item['condition']}` / `{item['rater_id']}`: "
            f"{item['reason']}"
            for item in summary["invalid_records"]
        )
    if summary["notes"]:
        lines.extend(["", "## Notes", ""])
        lines.extend(
            f"- `{item['scenario_id']}` / `{item['condition']}`: {item['note']}"
            for item in summary["notes"]
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Human evaluation pilot utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    packets_parser = subparsers.add_parser("packets")
    packets_parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    packets_parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    packets_parser.add_argument("--out", type=Path, default=DEFAULT_PACKETS)
    packets_parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    packets_parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    packets_parser.add_argument(
        "--conditions",
        type=lambda value: [item.strip() for item in value.split(",") if item.strip()],
        default=DEFAULT_CONDITIONS,
    )

    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    summary_parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    summary_parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_MD)

    args = parser.parse_args()

    if args.command == "packets":
        payload = build_packets(
            pilot_path=args.pilot,
            results_path=args.results,
            conditions=args.conditions,
        )
        write_packets(
            payload,
            out_path=args.out,
            template_path=args.template,
        )
        print(f"human-eval packets written: {args.out}")
        print(f"ratings template written: {args.template}")
        print(f"fill ratings file manually from template: {args.ratings}")
        return

    ratings = load_ratings(args.ratings)
    summary = summarize_ratings(ratings)
    write_summary(summary, out_json=args.out_json, out_md=args.out_md)
    print(f"human-eval summary written: {args.out_json}")
    print(f"human-eval summary markdown written: {args.out_md}")


if __name__ == "__main__":
    main()
