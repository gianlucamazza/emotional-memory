"""Scoring utilities for the LoCoMo benchmark.

Metrics (per the original LoCoMo paper + mem0 eval harness):
  F1        token-overlap F1 after normalization (lowercase, strip articles/punct)
  BLEU-1    unigram BLEU (clipped precision)
  LLM judge GPT-4o-mini binary CORRECT / WRONG (temperature 0)
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

_ARTICLES = {"a", "an", "the"}


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w not in _ARTICLES]


def token_f1(prediction: str, gold: str) -> float:
    """Token-overlap F1 between *prediction* and *gold*."""
    pred_tokens = _normalize(prediction)
    gold_tokens = _normalize(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: str, gold: str) -> float:
    """Unigram BLEU (clipped precision)."""
    pred_tokens = _normalize(prediction)
    gold_tokens = _normalize(gold)
    if not pred_tokens:
        return 0.0
    gold_counts = Counter(gold_tokens)
    clipped = sum(min(cnt, gold_counts[tok]) for tok, cnt in Counter(pred_tokens).items())
    return clipped / len(pred_tokens)


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
Your task is to label an answer to a question as 'CORRECT' or \
'WRONG'. You will be given the following data:

(1) a question (posed by one user to another user),
(2) a 'gold' (ground truth) answer,
(3) a generated answer

which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the \
other user based on their prior conversations.

The gold answer will usually be a concise and short answer that includes the referenced topic.

The generated answer might be much longer, but you should be generous with your \
grading — as long as it touches on the same topic as the gold answer, it should \
be counted as CORRECT.

For time related questions the gold answer will be a specific date, month, year, \
etc. The generated answer might be much longer or use relative time references, \
but you should be generous — as long as it refers to the same date or time \
period, count as CORRECT.

Now it's time for the real question:

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then \
finish with CORRECT or WRONG.

Do NOT include both CORRECT and WRONG in your response.

Just return the label CORRECT or WRONG in a json format with the key as "label".\
"""


def build_judge_prompt(question: str, gold_answer: str, generated_answer: str) -> str:
    return _JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )


def parse_judge_response(response: str) -> bool:
    """Return True if the judge said CORRECT."""
    import json

    label = ""
    try:
        data = json.loads(response)
        label = str(data.get("label", "")).upper()
    except Exception:
        label = ""
    if label == "CORRECT":
        return True
    if label == "WRONG":
        return False
    # fallback: plain text
    upper = response.upper()
    if "CORRECT" in upper and "WRONG" not in upper:
        return True
    if "WRONG" in upper:
        return False
    return False


def score_predictions(
    predictions: list[dict[str, Any]],
    *,
    include_adversarial: bool = False,
) -> dict[str, Any]:
    """Aggregate F1, BLEU-1, and optionally LLM judge scores.

    Parameters
    ----------
    predictions:
        List of dicts with keys: ``question``, ``gold``, ``prediction``,
        ``category`` (int), ``judge_correct`` (bool | None).
    include_adversarial:
        Whether to include category-5 in aggregate metrics.

    Returns
    -------
    dict with ``aggregate`` and ``by_category`` sub-dicts.
    """
    by_category: dict[str, list[dict[str, float]]] = {}
    for pred in predictions:
        cat = pred["category"]
        if cat == 5 and not include_adversarial:
            continue
        from benchmarks.locomo.dataset import QA_CATEGORY_NAMES

        cat_name = QA_CATEGORY_NAMES.get(cat, f"category_{cat}")
        f1 = token_f1(pred["prediction"], pred["gold"])
        b1 = bleu1(pred["prediction"], pred["gold"])
        row: dict[str, float] = {"f1": f1, "bleu1": b1}
        jc = pred.get("judge_correct")
        if jc is not None:
            row["judge"] = float(jc)
        by_category.setdefault(cat_name, []).append(row)

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    def _cat_summary(rows: list[dict[str, float]]) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "n": len(rows),
            "f1": round(_mean([r["f1"] for r in rows]), 4),
            "bleu1": round(_mean([r["bleu1"] for r in rows]), 4),
        }
        judges = [r["judge"] for r in rows if "judge" in r]
        if judges:
            summary["judge_accuracy"] = round(_mean(judges), 4)
        return summary

    all_rows = [row for rows in by_category.values() for row in rows]
    result: dict[str, Any] = {
        "aggregate": _cat_summary(all_rows),
        "by_category": {cat: _cat_summary(rows) for cat, rows in sorted(by_category.items())},
    }
    return result


# ---------------------------------------------------------------------------
# Adversarial (category-5) evaluation
# ---------------------------------------------------------------------------

_NOT_MENTIONED_PATTERNS = [
    re.compile(r"not mentioned", re.IGNORECASE),
    re.compile(r"not provided", re.IGNORECASE),
    re.compile(r"not found", re.IGNORECASE),
    re.compile(r"no information", re.IGNORECASE),
    re.compile(r"don't know", re.IGNORECASE),
]


def is_adversarial_correct(prediction: str) -> bool:
    """Return True if the prediction correctly refuses to answer (cat-5)."""
    return any(p.search(prediction) for p in _NOT_MENTIONED_PATTERNS)
