"""Generate ``realistic_recall_v4_noAF.json`` via LLM (Addendum P — Hg1 re-run).

One structured-JSON scenario per LLM call. The Python side fixes the structural
skeleton — memory ids, the two-session split, and the (challenge_type → target
event) mapping — so the LLM only fills in textual content (event narratives and
query phrasings). Each generated scenario is validated against the noAF schema
(reusing ``runner_hg1`` models + ``_validate_dataset``) before inclusion.

Why this design:
    * Leakage-free: scenario topics are seeded from THEMES, chosen to be
      thematically disjoint from realistic_recall_v3 (the Addendum O calibration
      set). Scenario ids use a new ``pNN_`` namespace, never ``sNN_``.
    * Author-blind: the generator NEVER runs the retrieval systems. It only
      authors content, exactly as a human author would, then freezes the JSON.
    * Robust: ids and target/confound structure are deterministic Python; the
      LLM cannot produce dangling ``expected_memory_ids``.

Usage::

    uv run python -m benchmarks.datasets.generate_v4_noAF --dry-run --limit 1
    uv run python -m benchmarks.datasets.generate_v4_noAF --n-scenarios 40

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
    from benchmarks.appraisal_confound.runner_hg1 import _NoAFScenario

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "benchmarks" / "datasets" / "realistic_recall_v4_noAF.json"
DEFAULT_TOP_K = 2

# Topics deliberately disjoint from realistic_recall_v3 (launch crises, marathon
# training, breakups, promotions, family illness, thesis defence, travel mishaps,
# etc.). Everyday affect-rich arcs across distinct life domains.
THEMES: tuple[str, ...] = (
    "adopting a rescue dog with a difficult past",
    "restoring a flooded community library",
    "a first pottery exhibition that almost didn't open",
    "learning to sail after a capsizing scare",
    "mediating a long feud between two neighbours",
    "a botched home renovation that became a refuge",
    "volunteering at a wildfire evacuation shelter",
    "reconnecting with an estranged grandparent through letters",
    "a community-garden harvest threatened by frost",
    "coaching a kids' chess team to a regional final",
    "a beekeeping season disrupted by a swarm loss",
    "organising a surprise reunion that nearly leaked",
    "recovering a lost wedding ring on a hiking trail",
    "a neighbourhood bakery surviving an inspection scare",
    "fostering twins through a turbulent first month",
    "a choir preparing for a cathedral debut",
    "repairing a vintage motorcycle inherited from a friend",
    "a tide-pool research trip that went off-schedule",
    "rescuing a stalled food-truck launch at a festival",
    "learning sign language to talk with a new colleague",
    "a glassblowing apprenticeship and a shattered first piece",
    "saving a dying orchard with grafting experiments",
    "a podcast episode that exposed a personal secret",
    "training a horse spooked by a thunderstorm",
    "a museum night-tour that lost a child briefly",
    "building an accessible ramp for a stubborn relative",
    "a kite-surfing trip interrupted by a jellyfish bloom",
    "reviving a shuttered village cinema",
    "an amateur astronomy night clouded out at the peak",
    "a cheese-making batch ruined then unexpectedly saved",
    "teaching a parent to swim later in life",
    "a street-mural project stalled by a permit dispute",
    "fostering reconciliation after a inheritance quarrel",
    "a long-distance cycling charity ride in a heatwave",
    "rescuing seedlings after a greenhouse heater failed",
    "a debut stand-up set bombing then recovering",
    "restoring an old rowing club after vandalism",
    "a search for a runaway cat across a snowstorm",
    "an apprenticeship in a blacksmith's forge",
    "a family recipe lost then reconstructed from memory",
    "a tide-timed wedding on a disappearing sandbar",
    "rehabilitating an injured falcon for release",
    "a school play saved after the lead lost their voice",
    "a winter expedition delayed by an avalanche warning",
    "reopening a trail after a landslide",
)

# Per-challenge-type instruction + which structural event slot is the target.
# Events are e1..e6 (session 1 = e1,e2,e3 ; session 2 = e4,e5,e6).
# Targets are chosen so that recency_confound targets an OLD (session-1) event
# while a more-recent same-topic distractor exists in session 2.
_CHALLENGE_SPEC: dict[str, tuple[int, str]] = {
    "semantic_confound": (
        4,
        "ask for the specific moment of event #4, phrased so a semantically "
        "similar but WRONG event (e.g. #5) is a tempting distractor; the wording "
        "must let only #4 be correct.",
    ),
    "recency_confound": (
        2,
        "ask for the EARLIER moment of event #2, even though a more recent "
        "same-topic event (#5 or #6) is fresher; the answer must be the older #2.",
    ),
    "affective_arc": (
        3,
        "ask a question that can only be answered by tracking the EMOTIONAL arc "
        "(e.g. 'the moment things turned from dread to relief'), pointing at #3.",
    ),
    "same_topic_distractor": (
        1,
        "ask about event #1 where another event shares the same sub-topic; the "
        "phrasing must disambiguate #1 from the same-topic distractor.",
    ),
    "momentum_alignment": (
        5,
        "ask a question whose emotional momentum aligns with event #5 "
        "(a build-up toward a charged turning point), pointing at #5.",
    ),
}
_CHALLENGE_ORDER: tuple[str, ...] = (
    "semantic_confound",
    "recency_confound",
    "affective_arc",
    "same_topic_distractor",
    "momentum_alignment",
)

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
    """Pick 4 of the 5 challenge types, rotating the dropped one for balance."""
    dropped = _CHALLENGE_ORDER[index % len(_CHALLENGE_ORDER)]
    return [c for c in _CHALLENGE_ORDER if c != dropped]


def _build_prompt(theme: str, scenario_id: str, challenges: list[str]) -> str:
    query_lines = []
    for j, ctype in enumerate(challenges, start=1):
        target_idx, instruction = _CHALLENGE_SPEC[ctype]
        query_lines.append(f"  Query {j} [{ctype}, target event #{target_idx}]: {instruction}")
    queries_block = "\n".join(query_lines)
    return (
        "You are authoring one scenario for an affect-rich memory-retrieval benchmark.\n"
        f"Theme: {theme}.\n\n"
        "Produce a two-session emotional arc with exactly 6 short event narratives "
        "(events 1-3 happen in session 1, events 4-6 in session 2). Each event is "
        "1-2 vivid sentences, emotionally charged but WITHOUT any explicit "
        "valence/arousal numbers. Events must form a coherent arc (e.g. tension -> "
        "setback -> turning point -> relief).\n\n"
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
) -> _NoAFScenario:
    from benchmarks.appraisal_confound.runner_hg1 import _NoAFScenario

    sid = f"p{index:02d}_{_slug(theme)}"
    events: list[str] = [str(e).strip() for e in payload["events"]]
    queries: list[str] = [str(q).strip() for q in payload["queries"]]
    if len(events) != 6 or len(queries) != 4:
        raise ValueError(f"{sid}: expected 6 events / 4 queries, got {len(events)}/{len(queries)}")

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
        target_idx, _ = _CHALLENGE_SPEC[ctype]
        query_objs.append(
            {
                "query_id": f"{sid}_q{j + 1}",
                "query": queries[j],
                "expected_memory_ids": [event_ids[target_idx - 1]],
                "challenge_type": ctype,
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
    return _NoAFScenario.model_validate(
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
            _build_prompt(theme, f"p{i + 1:02d}", challenges)
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
        prompt = _build_prompt(theme, f"p{i + 1:02d}", challenges)
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
        "name": "realistic_recall_v4_noAF",
        "version": "1.0.0",
        "description": (
            "Affect-free scenarios for Addendum P (Hg1 re-run with the recalibrated "
            "SEC->affect mapping). Scenario ids use the pNN_ namespace, thematically "
            "disjoint from realistic_recall_v3 to avoid train/test leakage with the "
            "Addendum O calibration set. No preset valence/arousal; LLM must infer affect."
        ),
        "default_top_k": DEFAULT_TOP_K,
        "scenarios": scenarios,
    }
    # Validate the full dataset before writing (schema + structural constraints).
    from benchmarks.appraisal_confound.runner_hg1 import _NoAFDataset, _validate_dataset

    parsed = _NoAFDataset.model_validate(dataset)
    _validate_dataset(parsed, DEFAULT_TOP_K)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    nq = sum(len(s["queries"]) for sc in scenarios for s in sc["sessions"])
    print(f"Wrote {len(scenarios)} scenarios / {nq} queries to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate realistic_recall_v4_noAF.json via LLM (Addendum P)."
    )
    parser.add_argument("--n-scenarios", type=int, default=40)
    parser.add_argument("--limit", type=int, default=None, help="Cap scenarios (smoke test).")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="No LLM calls; print plan only.")
    args = parser.parse_args()
    _generate(args)


if __name__ == "__main__":
    main()
