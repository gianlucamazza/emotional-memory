"""Generate ``realistic_recall_v5_gate.json`` via LLM (Addendum Q — affect gating).

One structured-JSON scenario per LLM call, mirroring ``generate_v4_noAF``: the
Python side fixes the structural skeleton — memory ids, the two-session split,
the frozen six-slot emotional arc, and the (challenge_type -> target event)
mapping — so the LLM only fills in textual content. Each scenario is validated
against the v5_gate schema (``runner_hq`` models + ``_validate_dataset``) AND
the mechanical lexical-leakage gate before inclusion.

Frozen arc (pre-reg + this generator):
    e1 hopeful anticipation (mild positive)     [session 1]
    e2 first setback (negative)                 [session 1]
    e3 grinding doubt (negative)                [session 1]
    e4 breakthrough / relief (strong positive)  [session 2]  <- affective target
    e5 sharp late scare (strong negative, SAME setting/vocabulary as e4)
    e6 final resolution, quiet relief (positive end tone)

Affective queries target e4: its tone matches the end-of-arc tone while the
semantically tied distractor e5 is BOTH more recent and opposite in tone, so
recency works against the target and only the affect channel favours it.

Why this design:
    * Leakage-free: themes disjoint from v3 and v4; scenario ids use the new
      ``qNN_`` namespace (never ``sNN_`` / ``pNN_``).
    * Author-blind: the generator never runs any retrieval system.
    * Mechanical leakage gate: for every affect_congruent_tiebreak query,
      content-word overlap(query, target e4) must not exceed
      overlap(query, distractor e5) by more than 2 tokens (pre-reg
      §Dataset acceptance gates, item 2); offending scenarios are retried.

Usage::

    uv run python -m benchmarks.datasets.generate_v5_gate --dry-run --limit 1
    uv run python -m benchmarks.datasets.generate_v5_gate --n-scenarios 50

Env: requires EMOTIONAL_MEMORY_LLM_API_KEY (+ optional EMOTIONAL_MEMORY_LLM_*),
loaded from .env if present (via python-dotenv when available).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmarks.appraisal_confound.runner_hq import _GateScenario

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "benchmarks" / "datasets" / "realistic_recall_v5_gate.json"
DEFAULT_TOP_K = 2

# Themes disjoint from realistic_recall_v3 (launch crises, marathons, breakups,
# promotions, family illness, thesis defence, travel mishaps, ...) and from the
# v4 list (rescue dog, flooded library, pottery exhibition, sailing, feuding
# neighbours, wildfire shelter, grandparent letters, beekeeping, falcon rehab,
# choir, blacksmith, ...). Everyday affect-rich arcs across distinct domains.
THEMES: tuple[str, ...] = (
    "restoring a great-aunt's narrowboat for one final voyage",
    "a community fridge emptied by a heatwave outage, then revived by the street",
    "a unicycle troupe crossing the city for a charity pledge",
    "raising ducklings orphaned by a road accident",
    "a lighthouse-keeping apprenticeship through storm season",
    "decoding a late father's shorthand diaries",
    "a paper-boat regatta organised for a children's ward",
    "a rooftop rainwater garden on a condemned building",
    "an escape-room launch sabotaged by a power cut",
    "subtitling a silent home-movie reel for a golden anniversary",
    "a marble-carving commission that cracked at the base",
    "running a midnight soup kitchen through a blizzard",
    "a homemade rocket club chasing a stratosphere record",
    "rebuilding a burned-down tree house with the kids",
    "a ferry breakdown stranding a touring brass band",
    "teaching a parrot to stop swearing before a family visit",
    "restoring a carousel horse for the town centenary",
    "retaking a failed driving test at fifty",
    "a beach-cleanup crew finding a message in a bottle",
    "a darts league final played with a torn shoulder",
    "a community radio station saved from closure",
    "hosting an exchange student through a homesick spiral",
    "sewing a quilt from a late mother's dresses",
    "an ice-rink reopening with a faulty chiller",
    "a typewriter repair shop's last big order",
    "a synchronized-swimming routine after an ear injury",
    "a model-railway exhibition wrecked in transit",
    "a coral-nursery dive programme after a bleaching event",
    "a clock-tower bell silenced by a cracked yoke",
    "a debate-team underdog run to the nationals",
    "wiring a derelict windmill back onto the grid",
    "a night-market lantern display ruined by rain and rebuilt by dawn",
    "a repair cafe fixing a town's broken heirlooms",
    "learning to freedive after a panic episode",
    "a school robotics kit stolen before the regional bout",
    "recording a village dialect before the last speaker moves away",
    "a mushroom-foraging course after a poisoning scare",
    "a ballroom studio's pipes bursting before the showcase",
    "a puppet theatre rebuilding after a van theft",
    "a sea-glass jewellery commission for a memorial",
    "chasing a hot-air balloon licence through a windy season",
    "authenticating an antique map that split the family",
    "a community pool reopening after a decade shut",
    "a houseboat community fighting an eviction notice",
    "a stained-glass window restoration after a hailstorm",
    "a wheelchair-basketball team's first away tournament",
    "a fire-damaged violin restored for a graduation recital",
    "a hedgehog hospital overwhelmed in autumn",
    "learning braille to keep reading with a losing-sight spouse",
    "a beached whale calf vigil through the night",
)

_ARC_SPEC = (
    "Event #1: hopeful anticipation as the undertaking begins (mildly positive).\n"
    "Event #2: a first concrete setback (negative).\n"
    "Event #3: grinding doubt or tension while pushing on (negative).\n"
    "Event #4: the breakthrough — a moment of strong relief/triumph (strongly positive).\n"
    "Event #5: a sharp late scare or relapse (strongly negative). CRITICAL: #5 must "
    "happen in the SAME setting as #4 and reuse its core vocabulary (same place, "
    "objects, people), differing in what happens and how it feels.\n"
    "Event #6: the final resolution, quiet relief, things settle (positive end tone).\n"
    "ALL events: convey emotion through behaviour, body and imagery — NEVER name "
    "emotions explicitly (no 'joy', 'fear', 'relief', 'dread', etc.)."
)

# (target event slot 1-6, distractor slot or None, instruction)
_CHALLENGE_SPEC: dict[str, tuple[int, int | None, str]] = {
    "semantic_confound": (
        5,
        4,
        "ask for the specific moment of event #5, phrased so the semantically "
        "similar #4 is a tempting distractor; the wording must let only #5 be "
        "correct (semantically determinable).",
    ),
    "recency_confound": (
        2,
        None,
        "ask for the EARLIER moment of event #2, even though a more recent "
        "same-topic event (#5 or #6) is fresher; the answer must be the older #2 "
        "(semantically determinable).",
    ),
    "same_topic_distractor": (
        1,
        None,
        "ask about event #1 where another event shares the same sub-topic; the "
        "phrasing must disambiguate #1 from the same-topic distractor "
        "(semantically determinable).",
    ),
    "affect_congruent_tiebreak": (
        4,
        5,
        "ask for the moment of event #4 phrased as a NEAR-TIE with event #5: use "
        "their shared setting words; the query's content words must overlap #5 at "
        "least as much as #4; do NOT use unique content nouns of #4 and do NOT use "
        "any emotion words; the only discriminating cue is that the asked-about "
        "moment is the one that matched how the whole story ended up (the good "
        "turn), not the bad one.",
    ),
    "affective_arc_blind": (
        4,
        None,
        "ask for the turning point of the emotional trajectory only (e.g. 'the "
        "moment the long slog finally broke'), with MINIMAL content-word overlap "
        "with event #4's text; #4 is the unique turning point.",
    ),
}

_AFFECT_FREE_ORDER: tuple[str, ...] = (
    "semantic_confound",
    "recency_confound",
    "same_topic_distractor",
)
_AFFECTIVE_ORDER: tuple[str, ...] = ("affect_congruent_tiebreak", "affective_arc_blind")

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "session1_description": {"type": "string"},
        "session2_description": {"type": "string"},
        "events": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 6,
            "maxItems": 6,
        },
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 4,
        },
    },
    "required": [
        "description",
        "session1_description",
        "session2_description",
        "events",
        "queries",
    ],
    "additionalProperties": False,
}

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "him",
        "his",
        "how",
        "i",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whose",
        "will",
        "with",
        "you",
        "your",
    ]
)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _leakage_ok(query: str, target: str, distractor: str, *, max_excess: int = 2) -> bool:
    """Pre-reg acceptance gate 2: overlap(q, target) - overlap(q, distractor) <= 2."""
    q = _content_words(query)
    return len(q & _content_words(target)) - len(q & _content_words(distractor)) <= max_excess


def _load_dotenv() -> None:
    """Load .env from repo root if python-dotenv is available (best-effort)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _challenge_types_for(index: int) -> list[str]:
    """2 affect-free (rotating the dropped one) + the 2 affective types."""
    dropped = _AFFECT_FREE_ORDER[index % len(_AFFECT_FREE_ORDER)]
    affect_free = [c for c in _AFFECT_FREE_ORDER if c != dropped]
    return affect_free + list(_AFFECTIVE_ORDER)


def _build_prompt(theme: str, challenges: list[str]) -> str:
    query_lines = []
    for j, ctype in enumerate(challenges, start=1):
        target_idx, _, instruction = _CHALLENGE_SPEC[ctype]
        query_lines.append(f"  Query {j} [{ctype}, target event #{target_idx}]: {instruction}")
    queries_block = "\n".join(query_lines)
    return (
        "You are authoring one scenario for an affect-rich memory-retrieval benchmark.\n"
        f"Theme: {theme}.\n\n"
        "Produce a two-session emotional arc with exactly 6 short event narratives "
        "(events 1-3 happen in session 1, events 4-6 in session 2). Each event is "
        "1-2 vivid sentences, emotionally charged but WITHOUT any explicit "
        "valence/arousal numbers. The arc is fixed:\n"
        f"{_ARC_SPEC}\n\n"
        "Then write exactly 4 retrieval queries, in this order, each realising its "
        "challenge type against the indicated target event:\n"
        f"{queries_block}\n\n"
        "Queries must be answerable from the events alone and must make the target "
        "event the single correct answer. Keep everything in English.\n\n"
        "Return ONLY a JSON object with keys: description (one sentence), "
        "session1_description, session2_description, events (array of 6 strings), "
        "queries (array of 4 strings). No markdown, no commentary."
    )


def _extract_json(raw: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response, tolerating code fences/prose."""
    try:
        return json.loads(raw)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        raise ValueError(f"No JSON object found in LLM response: {raw[:200]!r}")
    return json.loads(match.group(0))  # type: ignore[no-any-return]


def _scenario_from_payload(
    *, index: int, theme: str, challenges: list[str], payload: dict[str, Any]
) -> _GateScenario:
    from benchmarks.appraisal_confound.runner_hq import (
        AFFECTIVE_CHALLENGES,
        _GateScenario,
    )

    sid = f"q{index:02d}_{_slug(theme)}"
    events: list[str] = [str(e).strip() for e in payload["events"]]
    queries: list[str] = [str(q).strip() for q in payload["queries"]]
    if len(events) != 6 or len(queries) != 4:
        raise ValueError(f"{sid}: expected 6 events / 4 queries, got {len(events)}/{len(queries)}")

    # Mechanical lexical-leakage gate (pre-reg acceptance gate 2).
    for j, ctype in enumerate(challenges):
        target_idx, distractor_idx, _ = _CHALLENGE_SPEC[ctype]
        if ctype == "affect_congruent_tiebreak":
            assert distractor_idx is not None
            if not _leakage_ok(queries[j], events[target_idx - 1], events[distractor_idx - 1]):
                raise ValueError(
                    f"{sid}: leakage gate failed for {ctype} "
                    f"(query lexically favours target #{target_idx} over #{distractor_idx})"
                )

    event_ids = [f"{sid}_e{n}" for n in range(1, 7)]
    session1 = {
        "session_id": "session_1",
        "description": str(payload.get("session1_description", "")).strip() or "Opening events.",
        "events": [
            {"memory_id": event_ids[n], "content": events[n], "metadata": {"slot": f"e{n + 1}"}}
            for n in range(3)
        ],
        "queries": [],
    }
    query_objs = []
    for j, ctype in enumerate(challenges):
        target_idx, _, _ = _CHALLENGE_SPEC[ctype]
        query_objs.append(
            {
                "query_id": f"{sid}_q{j + 1}",
                "query": queries[j],
                "expected_memory_ids": [event_ids[target_idx - 1]],
                "challenge_type": ctype,
                "gate_label": "affective" if ctype in AFFECTIVE_CHALLENGES else "affect_free",
            }
        )
    session2 = {
        "session_id": "session_2",
        "description": str(payload.get("session2_description", "")).strip() or "Later events.",
        "events": [
            {"memory_id": event_ids[n], "content": events[n], "metadata": {"slot": f"e{n + 1}"}}
            for n in range(3, 6)
        ],
        "queries": query_objs,
    }
    return _GateScenario.model_validate(
        {
            "scenario_id": sid,
            "description": str(payload.get("description", "")).strip() or theme,
            "sessions": [session1, session2],
        }
    )


def _slug(theme: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", "", theme.lower()).split()
    return "_".join(words[:4])


def _generate(args: argparse.Namespace) -> None:
    n = min(args.n_scenarios, len(THEMES))
    if args.limit is not None:
        n = min(n, args.limit)

    if args.dry_run:
        for i in range(n):
            theme = THEMES[i]
            challenges = _challenge_types_for(i)
            _build_prompt(theme, challenges)
            print(f"[dry-run] scenario {i + 1}/{n}: {theme!r} challenges={challenges}")
        print(f"[dry-run] would generate {n} scenarios; no LLM calls made.")
        return

    _load_dotenv()
    from emotional_memory.llm_http import make_httpx_llm_from_env

    llm = make_httpx_llm_from_env()
    if llm is None:
        raise SystemExit(
            "EMOTIONAL_MEMORY_LLM_API_KEY not set. Configure .env or environment "
            "(see docs/contributing/llm-environment.md)."
        )

    scenarios: list[dict[str, Any]] = []
    for i in range(n):
        theme = THEMES[i]
        challenges = _challenge_types_for(i)
        prompt = _build_prompt(theme, challenges)
        last_err: Exception | None = None
        for attempt in range(args.max_retries):
            try:
                raw = llm(prompt, _RESPONSE_SCHEMA)
                payload = _extract_json(raw)
                scenario = _scenario_from_payload(
                    index=i + 1, theme=theme, challenges=challenges, payload=payload
                )
                scenarios.append(scenario.model_dump())
                print(f"[ok] scenario {i + 1}/{n}: {scenario.scenario_id}")
                break
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                last_err = exc
                print(f"[retry {attempt + 1}/{args.max_retries}] scenario {i + 1}: {exc}")
        else:
            raise SystemExit(f"scenario {i + 1} ({theme!r}) failed after retries: {last_err}")

    dataset = {
        "name": "realistic_recall_v5_gate",
        "version": "1.0.0",
        "description": (
            "Mixed-gate scenarios for Addendum Q (affect-aware gating). Each scenario "
            "carries 2 affect-free + 2 affective queries with ground-truth gate_label. "
            "Scenario ids use the qNN_ namespace, thematically disjoint from v3 and v4. "
            "No preset valence/arousal; the LLM appraisal engine is the only affect "
            "source. Affective targets (e4) are tone-congruent with the end of the arc "
            "while their semantic distractor (e5) is more recent and opposite in tone."
        ),
        "default_top_k": DEFAULT_TOP_K,
        "scenarios": scenarios,
    }
    # Validate the full dataset before writing (schema + structural constraints).
    from benchmarks.appraisal_confound.runner_hq import _GateDataset, _validate_dataset

    parsed = _GateDataset.model_validate(dataset)
    _validate_dataset(parsed, DEFAULT_TOP_K)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    nq = sum(len(s["queries"]) for sc in scenarios for s in sc["sessions"])
    print(f"Wrote {len(scenarios)} scenarios / {nq} queries to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate realistic_recall_v5_gate.json via LLM (Addendum Q)."
    )
    parser.add_argument("--n-scenarios", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="Cap scenarios (smoke test).")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="No LLM calls; print plan only.")
    args = parser.parse_args()
    _generate(args)


if __name__ == "__main__":
    main()
