"""Blinded human relevance packets and strictly paired analysis."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from benchmarks.human_eval.pipeline import krippendorff_alpha
from benchmarks.msc.dataset import SEED, Case, digest
from benchmarks.msc.statistics import paired_report

ARMS = ("aft", "hybrid", "cosine", "mem0")


def packets(cases: list[Case], raters: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(raters) != 3 or len(set(raters)) != 3 or any(not r.strip() for r in raters):
        raise ValueError("Exactly three distinct rater pseudonyms are required")
    private: dict[str, Any] = {
        "cases_digest": digest([c.model_dump() for c in cases]),
        "raters": raters,
        "lookup": {},
    }
    public = {}
    for rater in raters:
        rng = random.Random(f"{SEED}:{rater}")
        ordered = list(cases)
        rng.shuffle(ordered)
        cards = []
        for case in ordered:
            memories = list(case.candidates)
            rng.shuffle(memories)
            for candidate in memories:
                token = digest([rater, case.case_id, candidate.memory_id, "relevance"])[:24]
                private["lookup"][token] = [rater, case.case_id, candidate.memory_id]
                cards.append(
                    {
                        "item_id": token,
                        "context": case.context,
                        "query": case.query,
                        "memory": candidate.text,
                        "memory_session": candidate.session,
                        "memory_order": candidate.order,
                        "history": [
                            {"session": c.session, "order": c.order, "text": c.text}
                            for c in case.candidates
                        ],
                        "relevant": None,
                        "affect_required": None,
                        "note": "",
                    }
                )
        public[rater] = {"rater_id": rater, "source": "human", "items": cards}
    return public, private


def collect_gold(
    cases: list[Case], key: dict[str, Any], submissions: list[dict[str, Any]]
) -> dict[str, Any]:
    if key["cases_digest"] != digest([c.model_dump() for c in cases]):
        raise ValueError("Packet/case mismatch")
    if sorted(s["rater_id"] for s in submissions) != sorted(key["raters"]):
        raise ValueError("Missing or duplicate rater submission")
    seen = set()
    votes: dict[tuple[str, str], list[int]] = defaultdict(list)
    affect: dict[tuple[str, str], set[int]] = defaultdict(set)
    for submission in submissions:
        if submission.get("source") != "human":
            raise ValueError("Only actual human submissions are admissible")
        rater = submission["rater_id"]
        for item in submission["items"]:
            token = item["item_id"]
            if token in seen or token not in key["lookup"]:
                raise ValueError("Duplicate/unknown item")
            owner, case_id, memory_id = key["lookup"][token]
            if owner != rater:
                raise ValueError("Wrong rater for packet")
            for name in ("relevant", "affect_required"):
                if type(item.get(name)) is not int or item[name] not in (0, 1):
                    raise ValueError("Unfilled or non-binary human rating")
            votes[case_id, memory_id].append(item["relevant"])
            affect[case_id, rater].add(item["affect_required"])
            seen.add(token)
    if seen != set(key["lookup"]):
        raise ValueError("Incomplete annotation coverage")
    if any(len(v) != 1 for v in affect.values()):
        raise ValueError("affect_required must be constant for each case/rater")
    rows = []
    for case in cases:
        relevant = [
            c.memory_id for c in case.candidates if sum(votes[case.case_id, c.memory_id]) >= 2
        ]
        affect_count = sum(next(iter(affect[case.case_id, r])) for r in key["raters"])
        rows.append(
            {
                "case_id": case.case_id,
                "relevant_ids": relevant,
                "eligible": bool(relevant),
                "affect_required": affect_count >= 2,
            }
        )
    all_votes = list(votes.values())
    # The legacy helper returns 1 for constant labels; here reliability is undefined.
    alpha = (
        krippendorff_alpha(all_votes, level="nominal")
        if len({v for row in all_votes for v in row}) > 1
        else None
    )
    return {
        "cases_digest": key["cases_digest"],
        "raters": key["raters"],
        "source": "human",
        "alpha_nominal": alpha,
        "votes": [
            {"case_id": c, "memory_id": m, "ratings": v} for (c, m), v in sorted(votes.items())
        ],
        "cases": rows,
    }


def evaluate(cases: list[Case], gold: dict[str, Any], rankings: dict[str, Any]) -> dict[str, Any]:
    expected_digest = digest([c.model_dump() for c in cases])
    if gold["cases_digest"] != expected_digest or rankings["cases_digest"] != expected_digest:
        raise ValueError("Case/gold/ranking binding mismatch")
    if gold.get("source") != "human" or len(gold["cases"]) != len(cases):
        raise ValueError("Complete human gold is required")
    gold_by_id = {r["case_id"]: r for r in gold["cases"]}
    if len(gold_by_id) != len(cases) or set(gold_by_id) != {c.case_id for c in cases}:
        raise ValueError("Gold coverage mismatch")
    ranks = rankings["rankings"]
    if set(ranks) != {c.case_id for c in cases}:
        raise ValueError("Ranking coverage mismatch")
    hits: dict[str, list[bool]] = {a: [] for a in ARMS}
    for case in cases:
        allowed = {c.memory_id for c in case.candidates}
        row = gold_by_id[case.case_id]
        if not set(row["relevant_ids"]) <= allowed or row["eligible"] != bool(row["relevant_ids"]):
            raise ValueError("Invalid relevance gold")
        if set(ranks[case.case_id]) != set(ARMS):
            raise ValueError("Missing system (cannot silently drop Mem0)")
        for arm in ARMS:
            ranking = ranks[case.case_id][arm]
            if len(set(ranking)) != len(ranking) or not set(ranking) <= allowed:
                raise ValueError("Duplicate or foreign source memory ID")
            if row["eligible"]:
                hits[arm].append(bool(ranking) and ranking[0] in row["relevant_ids"])
    primary = paired_report(hits["aft"], hits["hybrid"])
    n = len(hits["aft"])
    return {
        "status": rankings["stage"],
        "config_digest": rankings.get("config_digest"),
        "manifest_sha256": rankings.get("manifest_sha256"),
        "primary": primary,
        "secondary_top1": {a: sum(v) / n for a, v in hits.items()},
        "excluded_no_relevant_memory": len(cases) - n,
        "alpha_nominal": gold["alpha_nominal"],
    }


def response_packets(
    cases: list[Case],
    responses: dict[str, dict[str, str]],
    raters: list[str],
    gold_raters: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(raters) != 3 or len(set(raters)) != 3 or set(raters) & set(gold_raters):
        raise ValueError("Need three raters distinct from the gold annotators")
    if len(cases) != 100 or set(responses) != {c.case_id for c in cases}:
        raise ValueError("Exactly the 100 presampled cases are required")
    public = {}
    key: dict[str, Any] = {"raters": raters, "gold_raters": gold_raters, "lookup": {}}
    for rater in raters:
        cards = []
        for case in cases:
            if set(responses[case.case_id]) != set(ARMS):
                raise ValueError("Incomplete response arms")
            for arm in ARMS:
                answer = responses[case.case_id][arm]
                if not answer.strip():
                    raise ValueError("Empty answer")
                token = digest([rater, case.case_id, arm, "response"])[:24]
                key["lookup"][token] = [rater, case.case_id, arm]
                cards.append(
                    {
                        "item_id": token,
                        "context": case.context,
                        "query": case.query,
                        "history": [c.text for c in case.candidates],
                        "answer": answer,
                        "usefulness": None,
                        "correctness": None,
                        "note": "",
                    }
                )
        random.Random(f"{SEED}:{rater}:responses").shuffle(cards)
        public[rater] = {"rater_id": rater, "source": "human", "items": cards}
    return public, key


def summarize_responses(key: dict[str, Any], submissions: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate by case before bootstrap: never count raters as independent trials."""
    import numpy as np

    if sorted(s["rater_id"] for s in submissions) != sorted(key["raters"]):
        raise ValueError("Incomplete or duplicate raters")
    data: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    seen = set()
    for submission in submissions:
        if submission.get("source") != "human":
            raise ValueError("Human ratings required")
        for item in submission["items"]:
            token = item["item_id"]
            if token in seen or token not in key["lookup"]:
                raise ValueError("Duplicate/unknown response")
            rater, case, arm = key["lookup"][token]
            if rater != submission["rater_id"]:
                raise ValueError("Wrong rater")
            seen.add(token)
            for dim in ("usefulness", "correctness"):
                value = item.get(dim)
                if type(value) is not int or not 1 <= value <= 5:
                    raise ValueError("Incomplete 1-5 rating")
                data[case, arm, dim].append(value)
    if seen != set(key["lookup"]):
        raise ValueError("Missing response ratings")
    case_ids = sorted({k[0] for k in data})
    report: dict[str, Any] = {"status": "SECONDARY_EXPLORATORY", "n_cases": len(case_ids)}
    for dim in ("usefulness", "correctness"):
        values = {a: [float(np.mean(data[c, a, dim])) for c in case_ids] for a in ARMS}
        diff = np.array(values["aft"]) - np.array(values["hybrid"])
        rng = np.random.default_rng(SEED)
        draws = rng.choice(diff, (10000, len(diff)), replace=True).mean(axis=1)
        votes = [v for k, v in data.items() if k[2] == dim]
        report[dim] = {
            "case_means": values,
            "delta_aft_hybrid": float(diff.mean()),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "alpha_ordinal": krippendorff_alpha(votes),
        }
    return report
