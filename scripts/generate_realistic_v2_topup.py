"""Extend realistic_recall_v2_{it,es}.json from 20→30 scenarios (N=80→120).

Two modes:
  --lang it   : generate 10 new Italian scenarios via LLM, inject programmatic valence/arousal
  --lang es   : translate the 10 new IT scenarios (--translate-from) into Spanish

Precondition: commit benchmarks/preregistration_addendum_hd2_powertopup.md BEFORE running.

Usage:
    # Step 1 — generate Italian top-up
    uv run python scripts/generate_realistic_v2_topup.py \\
        --lang it \\
        --dataset benchmarks/datasets/realistic_recall_v2_it.json \\
        --n-new 10 --seed 42 \\
        --out benchmarks/datasets/realistic_recall_v2_it.json \\
        --provenance benchmarks/datasets/realistic_recall_v2_topup_provenance.jsonl

    # Step 2 — translate to Spanish
    uv run python scripts/generate_realistic_v2_topup.py \\
        --lang es \\
        --dataset benchmarks/datasets/realistic_recall_v2_es.json \\
        --translate-from benchmarks/datasets/realistic_recall_v2_it.json \\
        --n-new 10 --seed 42 \\
        --out benchmarks/datasets/realistic_recall_v2_es.json \\
        --provenance benchmarks/datasets/realistic_recall_v2_topup_provenance.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# ── dotenv + API key bridge ──────────────────────────────────────────────────
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

if "EMOTIONAL_MEMORY_LLM_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["EMOTIONAL_MEMORY_LLM_API_KEY"]

# ── Challenge assignment (10 new scenarios, 8 occurrences per type) ──────────
# Each row = 4 challenge_types assigned to queries of that scenario.
# Every type appears in exactly 8 of 10 scenarios -> 8x1 per scenario = 8 queries per type.
CHALLENGE_ASSIGNMENT: list[list[str]] = [
    ["semantic_confound", "affective_arc", "recency_confound", "same_topic_distractor"],
    ["affective_arc", "recency_confound", "same_topic_distractor", "momentum_alignment"],
    ["recency_confound", "same_topic_distractor", "momentum_alignment", "semantic_confound"],
    ["same_topic_distractor", "momentum_alignment", "semantic_confound", "affective_arc"],
    ["momentum_alignment", "semantic_confound", "affective_arc", "recency_confound"],
    ["semantic_confound", "affective_arc", "recency_confound", "same_topic_distractor"],
    ["affective_arc", "recency_confound", "same_topic_distractor", "momentum_alignment"],
    ["recency_confound", "same_topic_distractor", "momentum_alignment", "semantic_confound"],
    ["same_topic_distractor", "momentum_alignment", "semantic_confound", "affective_arc"],
    ["momentum_alignment", "semantic_confound", "affective_arc", "recency_confound"],
]

# ── Programmatic v/a templates ───────────────────────────────────────────────
# Applied positionally: 4 events in session1 (idx 0-3), 3 in session2 (idx 4-6).
# Derived from mean structure of the 20 existing IT scenarios.
VA_PATTERNS: dict[str, list[tuple[float, float]]] = {
    "affective_arc": [
        (-0.80, 0.90),  # s1e1 crisis
        (-0.90, 0.85),  # s1e2 escalation
        (-0.50, 0.40),  # s1e3 doubt (quiet)
        (+0.70, 0.60),  # s1e4 first turn
        (+0.95, 0.85),  # s2e1 TARGET climax
        (+0.80, 0.65),  # s2e2 celebration
        (+0.50, 0.30),  # s2e3 aftermath (quiet)
    ],
    "recency_confound": [
        (-0.70, 0.75),  # s1e1 negative background
        (-0.95, 0.95),  # s1e2 key negative event
        (+0.30, 0.40),  # s1e3 first comfort
        (-0.80, 0.70),  # s1e4 secondary negative
        (+0.65, 0.50),  # s2e1 TARGET (less recent, stronger)
        (+0.80, 0.50),  # s2e2 new positive habit
        (+0.10, 0.30),  # s2e3 DISTRACTOR recent but neutral
    ],
    "same_topic_distractor": [
        (-0.85, 0.90),  # s1e1 negative same-topic event
        (+0.20, 0.60),  # s1e2 neutral (second opinion etc.)
        (-0.70, 0.75),  # s1e3 negative filler
        (+0.50, 0.50),  # s1e4 moderate positive support
        (-0.60, 0.80),  # s2e1 negative same-topic (semantic distractor)
        (+0.95, 0.80),  # s2e2 TARGET positive climax (same topic)
        (+0.85, 0.55),  # s2e3 follow-up positive walk/activity
    ],
    "semantic_confound": [
        (-0.90, 0.95),  # s1e1 TARGET strong negative with key keywords
        (-0.80, 0.70),  # s1e2 negative follow-up (diagnosis etc.)
        (+0.10, 0.50),  # s1e3 neutral/start of recovery
        (-0.65, 0.60),  # s1e4 secondary negative (watching others)
        (+0.75, 0.65),  # s2e1 DISTRACTOR positive with overlapping keywords
        (+0.95, 0.90),  # s2e2 positive climax
        (+0.60, 0.45),  # s2e3 quiet positive aftermath
    ],
    "momentum_alignment": [
        (-0.60, 0.85),  # s1e1 anxious start
        (-0.80, 0.95),  # s1e2 panic / escalation
        (+0.50, 0.70),  # s1e3 partial success
        (-0.65, 0.75),  # s1e4 setback/stumble
        (+0.95, 0.90),  # s2e1 TARGET climax (high arousal + valence)
        (+0.88, 0.75),  # s2e2 celebration / call
        (+0.10, 0.10),  # s2e3 quiet procedural aftermath
    ],
}

# Target event index per challenge_type (0-based over 7 events)
TARGET_INDICES: dict[str, int] = {
    "affective_arc": 4,  # s2e1
    "recency_confound": 4,  # s2e1 (older but stronger)
    "same_topic_distractor": 5,  # s2e2
    "semantic_confound": 0,  # s1e1
    "momentum_alignment": 4,  # s2e1
}

# Query retrieval state per challenge_type
QUERY_STATES: dict[str, dict[str, float]] = {
    "affective_arc": {"valence": 0.85, "arousal": 0.75},
    "recency_confound": {"valence": 0.60, "arousal": 0.40},
    "same_topic_distractor": {"valence": 0.90, "arousal": 0.75},
    "semantic_confound": {"valence": -0.90, "arousal": 0.95},
    "momentum_alignment": {"valence": 0.90, "arousal": 0.85},
}

# ── Few-shot example fragments (abbreviated, for LLM conditioning) ───────────
FEW_SHOT: dict[str, str] = {
    "affective_arc": (
        'Scenario "lavoro_promosso_01": progetto critico → sabotaggio collega → notizia promozione (TARGET) → email contratto. '
        "Query (state v=+0.9,a=0.8): 'Qual è stato il momento esatto in cui ho saputo della promozione dalla direttrice?' "
        "expected: ['lavoro_promozione_notizia']"
    ),
    "recency_confound": (
        'Scenario "relazione_rottura_02": rottura dolorosa (s1) → primo sorriso dopo settimane (TARGET s2e1) → chat neutra ex (DISTRACTOR s2e3 recente). '
        "Query (state v=+0.6,a=0.4): 'Quando ho iniziato a sentire che stavo guarendo, non solo la chat neutra?' "
        "expected: ['relazione_primo_sorriso'] (meno recente ma più intenso del distractor)"
    ),
    "same_topic_distractor": (
        'Scenario "salute_diagnosi_03": diagnosi negativa (DISTRACTOR topic=salute) → notizia guarigione (TARGET topic=salute). '
        "Query (state v=+0.9,a=0.75): 'Qual è il momento di sollievo profondo, non la semplice passeggiata?' "
        "expected: ['salute_risultato_finale'] (stesso topic del distractor ma valenza opposta)"
    ),
    "semantic_confound": (
        'Scenario "sport_gara_04": infortunio momento (TARGET v=-0.9,a=0.95) vs primo-passo-senza-dolore (DISTRACTOR v=+0.75, keywords simili). '
        "Query (state v=-0.9,a=0.95): 'Quando ho capito di essermi infortunato sul serio, non la diagnosi?' "
        "expected: ['sport_infortunio_momento'] (semanticamente vicino al distractor positivo ma affettivamente opposto)"
    ),
    "momentum_alignment": (
        'Scenario "studio_esame_07": notte bianca → panico → prima domanda ok → sbaglio → TRENTA CON LODE (TARGET v=+0.95,a=0.90) → chiamata genitori. '
        "Query (state v=+0.9,a=0.85): 'Quale ricordo riflette lo slancio trasformativo dall'angoscia all'euforia?' "
        "expected: ['studio_voto_trenta'] (climax di arousal)"
    ),
}

# ── Challenge type descriptions for prompt ───────────────────────────────────
CHALLENGE_DESCRIPTIONS: dict[str, str] = {
    "affective_arc": (
        "L'arco emotivo va da fortemente negativo a positivo. "
        "La query chiede il momento culminante positivo (s2e1, valenza alta), non il momento di celebrazione successivo. "
        "I distrattori sono eventi positivi vicini nel tempo ma meno intensi."
    ),
    "recency_confound": (
        "L'evento TARGET (s2e1) è emotivamente più intenso ma meno recente dell'evento DISTRACTOR (s2e3 neutro/debole). "
        "La query chiede l'evento più significativo emotivamente, non l'ultimo. "
        "Il distractor (s2e3) deve essere più recente ma affettivamente debole."
    ),
    "same_topic_distractor": (
        "Due eventi parlano dello STESSO TOPIC ma con valenze opposte: s1e1 negativo e s2e2 TARGET positivo. "
        "La query chiede l'evento positivo su quel topic, non quello negativo. "
        "I testi degli eventi devono usare parole del topic comune."
    ),
    "semantic_confound": (
        "L'evento TARGET (s1e1) è fortemente negativo. Esiste un DISTRACTOR (s2e1) positivo che usa parole simili. "
        "La query usa il lessico del target negativo, quindi un sistema semantico puro recupererebbe il distractor positivo. "
        "Solo il segnale affettivo negativo identifica il target."
    ),
    "momentum_alignment": (
        "L'arousal cresce da s1e1 a s2e1 (TARGET). Il TARGET è il momento climax con arousal e valenza massimi. "
        "La query usa state con valenza e arousal alti, che corrispondono al momentum del climax. "
        "s2e3 è quieto/procedurale e serve come contrasto di basso arousal."
    ),
}

# ── Existing IT topic stems to avoid collisions ──────────────────────────────
EXISTING_STEMS = frozenset(
    {
        "lavoro",
        "relazione",
        "salute",
        "sport",
        "studio",
        "lutto",
        "trasloco",
        "pensionamento",
        "genitore",
        "amicizia_conflitto",
        "imprenditore",
        "artista",
        "divorzio",
        "promozione",
        "competizione",
        "incidente",
        "maternita",
        "carriera",
        "hobby",
        "viaggio_avventura",
    }
)


# ── LLM call helper ──────────────────────────────────────────────────────────


def _llm_call(
    prompt: str, model: str, api_key: str, base_url: str, *, max_retries: int = 3
) -> str:
    import httpx

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.7,
    }

    for attempt in range(max_retries):
        try:
            resp = httpx.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=90.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            print(f"  LLM attempt {attempt + 1} failed: {exc}. Retrying in 3s…", file=sys.stderr)
            time.sleep(3)
    raise RuntimeError("unreachable")


def _extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = raw.rstrip("`").strip()
    result: dict[str, Any] = json.loads(raw)
    return result


# ── Prompt builders ──────────────────────────────────────────────────────────


def _it_generation_prompt(
    challenge_types: list[str],
    scenario_number: int,
    existing_ids: set[str],
) -> str:
    primary_ct = challenge_types[0]
    return f"""Sei un generatore di dataset per benchmark di recupero della memoria emotiva.

Devi generare UNO scenario narrativo in italiano (scenario numero {scenario_number}/30).
Tema: libero — scegli un contesto di vita reale NON già usato tra questi stem: {sorted(EXISTING_STEMS)}.

Struttura richiesta:
- Sessione 1: ESATTAMENTE 4 eventi (indici 0-3) — costruzione del contesto emotivo
- Sessione 2: ESATTAMENTE 3 eventi (indici 4-6) + 4 query

Challenge type PRINCIPALE (determina il pattern affettivo): {primary_ct}
{CHALLENGE_DESCRIPTIONS[primary_ct]}

Challenge type SUPPLEMENTARI (per le altre 3 query, scegli gli eventi esistenti appropriati):
- {challenge_types[1]}: {CHALLENGE_DESCRIPTIONS[challenge_types[1]][:120]}…
- {challenge_types[2]}: {CHALLENGE_DESCRIPTIONS[challenge_types[2]][:120]}…
- {challenge_types[3]}: {CHALLENGE_DESCRIPTIONS[challenge_types[3]][:120]}…

Esempio di scenario simile per il challenge type principale:
{FEW_SHOT[primary_ct]}

ID già usati nel dataset (evita collisioni): usa uno slug NUOVO non presente tra questi: {sorted(existing_ids)[:20]}…

Output JSON (ESATTAMENTE questo schema, niente altro):
{{
  "slug": "tema_concetto",
  "description": "una frase che descrive lo scenario",
  "session1_description": "descrizione breve della prima sessione",
  "session1_events": [
    {{"id_suffix": "fase_breve_1", "content": "testo evento 1 (1-2 frasi naturali in italiano)", "topic": "tema", "phase": "fase_narrativa"}},
    {{"id_suffix": "fase_breve_2", "content": "testo evento 2", "topic": "tema", "phase": "fase_narrativa"}},
    {{"id_suffix": "fase_breve_3", "content": "testo evento 3", "topic": "tema", "phase": "fase_narrativa"}},
    {{"id_suffix": "fase_breve_4", "content": "testo evento 4", "topic": "tema", "phase": "fase_narrativa"}}
  ],
  "session2_description": "descrizione breve della seconda sessione",
  "session2_events": [
    {{"id_suffix": "fase_breve_5", "content": "testo evento 5", "topic": "tema", "phase": "fase_narrativa"}},
    {{"id_suffix": "fase_breve_6", "content": "testo evento 6", "topic": "tema", "phase": "fase_narrativa"}},
    {{"id_suffix": "fase_breve_7", "content": "testo evento 7 (quieto/procedurale se momentum_alignment)", "topic": "tema", "phase": "fase_narrativa"}}
  ],
  "queries": [
    {{
      "challenge_type": "{primary_ct}",
      "id_suffix": "query_principale",
      "query": "domanda che chiede il momento emotivo giusto (sfida: {primary_ct})",
      "expected_id_suffix": "id_suffix dell'evento target tra quelli generati sopra"
    }},
    {{
      "challenge_type": "{challenge_types[1]}",
      "id_suffix": "query_sup1",
      "query": "domanda per {challenge_types[1]}",
      "expected_id_suffix": "id_suffix dell'evento appropriato"
    }},
    {{
      "challenge_type": "{challenge_types[2]}",
      "id_suffix": "query_sup2",
      "query": "domanda per {challenge_types[2]}",
      "expected_id_suffix": "id_suffix dell'evento appropriato"
    }},
    {{
      "challenge_type": "{challenge_types[3]}",
      "id_suffix": "query_sup3",
      "query": "domanda per {challenge_types[3]}",
      "expected_id_suffix": "id_suffix dell'evento appropriato"
    }}
  ]
}}

Regole:
1. Ogni evento ha un id_suffix UNICO (snake_case, max 4 parole, descrittivo della fase narrativa).
2. Lo slug è diverso da tutti i temi esistenti.
3. Il contenuto è in italiano naturale (non traduzione dall'inglese).
4. NON includere valori numerici (valenza, arousal) — vengono assegnati programmaticamente.
5. Il challenge_type principale determina quale evento (indice {TARGET_INDICES[primary_ct]}) è il target affettivo.
6. Per recency_confound: l'evento s2e3 (indice 6) DEVE essere recente ma emotivamente debole (non il target).
7. Per semantic_confound: s1e1 (indice 0) è negativo (target); s2e1 (indice 4) usa parole simili ma è positivo (distractor).
8. expected_id_suffix DEVE corrispondere esattamente a uno degli id_suffix generati.
"""


def _es_translation_prompt(it_scenario: dict[str, Any]) -> str:
    it_json = json.dumps(it_scenario, ensure_ascii=False, indent=2)
    return f"""Traduci il seguente scenario di benchmark dal italiano allo spagnolo.

Regole di traduzione:
1. Traduci SOLO i campi testuali: description, session*_description, content, topic, phase, query.
2. Gli id_suffix: sostituisci le parole italiane con equivalenti spagnoli (snake_case).
3. Lo slug: traduci in spagnolo (es. "amicizia_tradimento" → "amistad_traicion").
4. I challenge_type NON si traducono (sono costanti tecniche in inglese).
5. NON modificare valori numerici — non sono presenti ma non aggiungerne.
6. La traduzione deve essere naturale in spagnolo castigliano.
7. Adatta nomi propri e contesti culturali (es. "Giulia" → "Marta", "Milano" → "Madrid").

Scenario italiano:
{it_json}

Output: JSON con la stessa struttura, tutti i testi in spagnolo.
"""


# ── Assembly helpers ─────────────────────────────────────────────────────────


def _build_events_with_va(
    events_raw: list[dict[str, Any]],
    va_pattern: list[tuple[float, float]],
    slug: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for i, (ev, (v, a)) in enumerate(zip(events_raw, va_pattern, strict=False)):
        result.append(
            {
                "memory_id": f"{slug}_{ev['id_suffix']}",
                "content": ev["content"],
                "valence": v,
                "arousal": a,
                "metadata": {
                    "topic": ev.get("topic", slug),
                    "phase": ev.get("phase", f"phase_{i + 1}"),
                },
            }
        )
    return result


def _build_queries(
    queries_raw: list[dict[str, Any]],
    slug: str,
    challenge_types: list[str],
    scenario_number: int,
) -> list[dict[str, Any]]:
    nn = str(scenario_number).zfill(2)
    result: list[dict[str, Any]] = []
    for q in queries_raw:
        ct = q["challenge_type"]
        if ct not in QUERY_STATES:
            raise ValueError(f"Unknown challenge_type in LLM output: {ct!r}")
        result.append(
            {
                "query_id": f"q_{slug}_{nn}_{q['id_suffix']}",
                "query": q["query"],
                "expected_memory_ids": [f"{slug}_{q['expected_id_suffix']}"],
                "challenge_type": ct,
                "state": QUERY_STATES[ct],
            }
        )
    return result


def _assemble_scenario(
    llm_output: dict[str, Any],
    scenario_number: int,
    challenge_types: list[str],
    lang: str,
) -> dict[str, Any]:
    slug = llm_output["slug"]
    nn = str(scenario_number).zfill(2)
    scenario_id = f"{slug}_{nn}"
    session_prefix = "sessione" if lang == "it" else "sesion"

    s1_events = llm_output["session1_events"]
    s2_events = llm_output["session2_events"]
    all_events = s1_events + s2_events

    primary_ct = challenge_types[0]
    va_pattern = VA_PATTERNS[primary_ct]
    all_events_with_va = _build_events_with_va(all_events, va_pattern, slug)

    s1_with_va = all_events_with_va[:4]
    s2_with_va = all_events_with_va[4:]

    queries = _build_queries(llm_output["queries"], slug, challenge_types, scenario_number)

    return {
        "scenario_id": scenario_id,
        "description": llm_output["description"],
        "sessions": [
            {
                "session_id": f"{session_prefix}_1",
                "description": llm_output.get("session1_description", "Prima sessione"),
                "events": s1_with_va,
                "queries": [],
            },
            {
                "session_id": f"{session_prefix}_2",
                "description": llm_output.get("session2_description", "Seconda sessione"),
                "events": s2_with_va,
                "queries": queries,
            },
        ],
    }


def _validate_scenario(scenario: dict[str, Any], existing_ids: set[str]) -> list[str]:
    errors: list[str] = []
    all_memory_ids: set[str] = set()
    for sess in scenario["sessions"]:
        for ev in sess["events"]:
            mid = ev["memory_id"]
            if mid in existing_ids:
                errors.append(f"memory_id collision: {mid!r}")
            if mid in all_memory_ids:
                errors.append(f"memory_id duplicate within scenario: {mid!r}")
            all_memory_ids.add(mid)

    errors.extend(
        f"expected_memory_id {eid!r} not found in scenario events"
        for sess in scenario["sessions"]
        for q in sess.get("queries", [])
        for eid in q["expected_memory_ids"]
        if eid not in all_memory_ids
    )

    return errors


# ── Generate mode ────────────────────────────────────────────────────────────


def generate_it(
    dataset: dict[str, Any],
    n_new: int,
    seed: int,
    model: str,
    api_key: str,
    base_url: str,
    provenance_path: Path,
) -> list[dict[str, Any]]:
    import random as rng_mod

    rng_mod.seed(seed)

    existing_ids: set[str] = {
        ev["memory_id"]
        for s in dataset["scenarios"]
        for sess in s["sessions"]
        for ev in sess["events"]
    }
    existing_slugs: set[str] = {s["scenario_id"].rsplit("_", 1)[0] for s in dataset["scenarios"]}

    start_number = len(dataset["scenarios"]) + 1
    new_scenarios: list[dict[str, Any]] = []

    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    prov_file = provenance_path.open("a", encoding="utf-8")

    try:
        for i in range(n_new):
            scenario_number = start_number + i
            challenge_types = CHALLENGE_ASSIGNMENT[i]
            print(
                f"  [{i + 1}/{n_new}] scenario {scenario_number} — challenges: {challenge_types}",
                flush=True,
            )

            prompt = _it_generation_prompt(
                challenge_types, scenario_number, existing_ids | existing_slugs
            )
            raw = _llm_call(prompt, model, api_key, base_url)
            llm_out = _extract_json(raw)

            prov_file.write(
                json.dumps(
                    {
                        "type": "it_generation",
                        "scenario_number": scenario_number,
                        "challenge_types": challenge_types,
                        "seed": seed,
                        "model": model,
                        "slug": llm_out.get("slug", ""),
                        "prompt_length": len(prompt),
                        "response_raw": raw[:2000],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            prov_file.flush()

            try:
                scenario = _assemble_scenario(llm_out, scenario_number, challenge_types, lang="it")
                errs = _validate_scenario(scenario, existing_ids)
                if errs:
                    print(
                        f"    Validation errors for scenario {scenario_number}: {errs}",
                        file=sys.stderr,
                    )
                    print("    Skipping — manual fix required.", file=sys.stderr)
                    continue
            except (KeyError, ValueError) as exc:
                print(f"    Assembly error for scenario {scenario_number}: {exc}", file=sys.stderr)
                print(f"    LLM output: {raw[:500]}", file=sys.stderr)
                continue

            new_ids: set[str] = {
                ev["memory_id"] for sess in scenario["sessions"] for ev in sess["events"]
            }
            existing_ids |= new_ids
            existing_slugs.add(llm_out["slug"])
            new_scenarios.append(scenario)
            print(
                f"    OK — {len(new_ids)} events, {sum(len(sess.get('queries', [])) for sess in scenario['sessions'])} queries"
            )

    finally:
        prov_file.close()

    return new_scenarios


# ── Translate mode ───────────────────────────────────────────────────────────


def translate_es(
    it_dataset: dict[str, Any],
    es_dataset: dict[str, Any],
    n_new: int,
    model: str,
    api_key: str,
    base_url: str,
    provenance_path: Path,
) -> list[dict[str, Any]]:
    it_new_scenarios = it_dataset["scenarios"][-n_new:]

    prov_file = provenance_path.open("a", encoding="utf-8")
    translated: list[dict[str, Any]] = []

    try:
        for i, it_s in enumerate(it_new_scenarios):
            start_number = len(es_dataset["scenarios"]) + 1 + i
            print(
                f"  [{i + 1}/{n_new}] translating {it_s['scenario_id']} → es_{start_number}",
                flush=True,
            )

            it_llm_repr = {
                "slug": it_s["scenario_id"].rsplit("_", 1)[0],
                "description": it_s["description"],
                "session1_description": it_s["sessions"][0]["description"],
                "session1_events": [
                    {
                        "id_suffix": ev["memory_id"].split("_", 1)[-1]
                        if "_" in ev["memory_id"]
                        else ev["memory_id"],
                        "content": ev["content"],
                        "topic": ev["metadata"].get("topic", ""),
                        "phase": ev["metadata"].get("phase", ""),
                    }
                    for ev in it_s["sessions"][0]["events"]
                ],
                "session2_description": it_s["sessions"][1]["description"],
                "session2_events": [
                    {
                        "id_suffix": ev["memory_id"].split("_", 1)[-1]
                        if "_" in ev["memory_id"]
                        else ev["memory_id"],
                        "content": ev["content"],
                        "topic": ev["metadata"].get("topic", ""),
                        "phase": ev["metadata"].get("phase", ""),
                    }
                    for ev in it_s["sessions"][1]["events"]
                ],
                "queries": [
                    {
                        "challenge_type": q["challenge_type"],
                        "id_suffix": q["query_id"].split("_", 2)[-1]
                        if "_" in q["query_id"]
                        else q["query_id"],
                        "query": q["query"],
                        "expected_id_suffix": q["expected_memory_ids"][0].split("_", 1)[-1]
                        if q["expected_memory_ids"]
                        else "",
                    }
                    for q in it_s["sessions"][1].get("queries", [])
                ],
            }

            prompt = _es_translation_prompt(it_llm_repr)
            raw = _llm_call(prompt, model, api_key, base_url)
            es_llm_out = _extract_json(raw)

            prov_file.write(
                json.dumps(
                    {
                        "type": "es_translation",
                        "it_scenario_id": it_s["scenario_id"],
                        "model": model,
                        "es_slug": es_llm_out.get("slug", ""),
                        "response_raw": raw[:2000],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            prov_file.flush()

            # Determine challenge_types from IT scenario query order
            challenge_types = [q["challenge_type"] for q in it_s["sessions"][1].get("queries", [])]

            try:
                es_scenario = _assemble_scenario(
                    es_llm_out, start_number, challenge_types, lang="es"
                )
                # Copy valence/arousal from Italian (identical by design)
                it_events_flat = [ev for sess in it_s["sessions"] for ev in sess["events"]]
                es_events_flat = [ev for sess in es_scenario["sessions"] for ev in sess["events"]]
                for it_ev, es_ev in zip(it_events_flat, es_events_flat, strict=False):
                    es_ev["valence"] = it_ev["valence"]
                    es_ev["arousal"] = it_ev["arousal"]
            except (KeyError, ValueError) as exc:
                print(f"    Assembly error: {exc}", file=sys.stderr)
                print(f"    Skipping ES scenario {start_number}", file=sys.stderr)
                continue

            translated.append(es_scenario)
            print(f"    OK — slug: {es_llm_out.get('slug', '?')}")

    finally:
        prov_file.close()

    return translated


# ── Dataset I/O ──────────────────────────────────────────────────────────────


def _verify_dataset(path: Path, expected_n: int, expected_queries: int) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    n_scenarios = len(d["scenarios"])
    n_queries = sum(
        len(q_list)
        for s in d["scenarios"]
        for sess in s["sessions"]
        for q_list in [sess.get("queries", [])]
    )
    if n_scenarios != expected_n:
        raise ValueError(f"Expected {expected_n} scenarios, got {n_scenarios}")
    if n_queries != expected_queries:
        raise ValueError(f"Expected {expected_queries} queries, got {n_queries}")
    # Verify global memory_id uniqueness
    all_ids = [
        ev["memory_id"] for s in d["scenarios"] for sess in s["sessions"] for ev in sess["events"]
    ]
    n_dupes = len(all_ids) - len(set(all_ids))
    if n_dupes > 0:
        raise ValueError(f"Duplicate memory_ids found: {n_dupes} duplicates")
    print(f"  Sanity OK — {n_scenarios} scenarios, {n_queries} queries, all IDs unique")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lang", choices=["it", "es"], required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="Existing dataset to extend")
    parser.add_argument(
        "--translate-from",
        type=Path,
        default=None,
        help="IT dataset with new scenarios (required for --lang es)",
    )
    parser.add_argument("--n-new", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", type=Path, required=True, help="Output path (may overwrite input)"
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("benchmarks/datasets/realistic_recall_v2_topup_provenance.jsonl"),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate only 1 scenario and print without saving"
    )
    args = parser.parse_args()

    api_key = os.environ.get(
        "EMOTIONAL_MEMORY_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")
    ).strip()
    if not api_key:
        sys.exit("EMOTIONAL_MEMORY_LLM_API_KEY not set. Export it before running.")
    base_url = os.environ.get("EMOTIONAL_MEMORY_LLM_BASE_URL", "https://api.openai.com/v1").rstrip(
        "/"
    )
    model = os.environ.get("EMOTIONAL_MEMORY_LLM_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"

    print(f"Model: {model}  Lang: {args.lang}  n_new: {args.n_new}  seed: {args.seed}")

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    n_base = len(dataset["scenarios"])
    print(f"Loaded dataset: {n_base} existing scenarios")

    if args.dry_run:
        args.n_new = 1

    if args.lang == "it":
        print("Generating Italian scenarios…")
        new_scenarios = generate_it(
            dataset, args.n_new, args.seed, model, api_key, base_url, args.provenance
        )
    else:
        if args.translate_from is None:
            sys.exit("--translate-from is required for --lang es")
        it_dataset = json.loads(args.translate_from.read_text(encoding="utf-8"))
        print("Translating Italian scenarios to Spanish…")
        new_scenarios = translate_es(
            it_dataset, dataset, args.n_new, model, api_key, base_url, args.provenance
        )

    if not new_scenarios:
        sys.exit("No scenarios generated (all failed validation). Check LLM output above.")

    if args.dry_run:
        print("\n--- DRY RUN: first scenario ---")
        print(json.dumps(new_scenarios[0], ensure_ascii=False, indent=2)[:3000])
        return

    dataset["scenarios"].extend(new_scenarios)
    args.out.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out} ({len(dataset['scenarios'])} scenarios total)")

    expected_n = n_base + len(new_scenarios)
    expected_q = sum(
        len(q_list)
        for s in dataset["scenarios"]
        for sess in s["sessions"]
        for q_list in [sess.get("queries", [])]
    )
    _verify_dataset(args.out, expected_n, expected_q)


if __name__ == "__main__":
    main()
