"""Build synthetic-persona dataset from raw DailyDialog corpus.

Downloads the raw corpus from HuggingFace Hub (roskoN/dailydialog) using
``huggingface_hub.hf_hub_download`` — no custom loading scripts required.

Run once to produce ``benchmarks/datasets/dailydialog_personas_v1.json``:

    uv run python -m benchmarks.dailydialog.persona_builder \\
        --n 120 --seed 0 \\
        --out benchmarks/datasets/dailydialog_personas_v1.json

The generated file is committed to the repo; the benchmark runner reads it
directly without needing ``huggingface_hub`` or network access.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from zipfile import ZipFile

from benchmarks.dailydialog.dataset import (
    EKMAN_LABEL_NAMES,
    EKMAN_PAD_MAP,
    DailyDialogPersonaDataset,
    DailyTurn,
    Persona,
    PersonaSession,
    RawDialog,
)
from benchmarks.dailydialog.query_generator import build_queries

EMOTION_DENSITY_THRESHOLD = 0.30
SESSIONS_PER_PERSONA_MIN = 4
SESSIONS_PER_PERSONA_MAX = 5

# roskoN/dailydialog has train/validation/test zips with the same format as
# the original ijcnlp_dailydialog release but accessible without a loading script.
_HF_REPO = "roskoN/dailydialog"
_SPLITS = ("train", "validation")

DEFAULT_OUT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "datasets"
    / "dailydialog_personas_v1.json"
)


# ---------------------------------------------------------------------------
# Raw corpus loader — direct zip parsing, no loading scripts
# ---------------------------------------------------------------------------


def _load_raw_dialogs() -> list[RawDialog]:
    """Load DailyDialog train + validation from roskoN/dailydialog on HF Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "The persona builder requires huggingface_hub.\n"
            "Install with: pip install huggingface_hub"
        ) from exc

    dialogs: list[RawDialog] = []
    for split in _SPLITS:
        zip_path = hf_hub_download(_HF_REPO, f"{split}.zip", repo_type="dataset")
        with Path(zip_path).open("rb") as fh, ZipFile(io.BytesIO(fh.read())) as zf:
            dialog_file = f"{split}/dialogues_{split}.txt"
            act_file = f"{split}/dialogues_act_{split}.txt"
            emotion_file = f"{split}/dialogues_emotion_{split}.txt"

            with (
                zf.open(dialog_file) as df,
                zf.open(act_file) as af,
                zf.open(emotion_file) as ef,
            ):
                for idx, (d_line, a_line, e_line) in enumerate(zip(df, af, ef, strict=False)):
                    d_raw = d_line.decode("utf-8", errors="replace").strip()
                    a_raw = a_line.decode("utf-8", errors="replace").strip()
                    e_raw = e_line.decode("utf-8", errors="replace").strip()
                    if not d_raw:
                        continue

                    texts = [t.strip() for t in d_raw.split("__eou__") if t.strip()]
                    acts = [int(x) for x in a_raw.split() if x.strip()]
                    emotions = [int(x) for x in e_raw.split() if x.strip()]

                    n = min(len(texts), len(acts), len(emotions))
                    if n == 0:
                        continue

                    turns = tuple(
                        DailyTurn(text=texts[i], emotion=emotions[i], act=acts[i])
                        for i in range(n)
                    )
                    dialogs.append(RawDialog(dialog_id=f"{split}_{idx:06d}", turns=turns))

    return dialogs


# ---------------------------------------------------------------------------
# Filtering + persona construction
# ---------------------------------------------------------------------------


def _filter_dialogs(
    dialogs: list[RawDialog],
    *,
    threshold: float = EMOTION_DENSITY_THRESHOLD,
) -> list[RawDialog]:
    """Retain only dialogs with emotion density >= threshold."""
    return [d for d in dialogs if d.emotion_density >= threshold]


def _build_persona_session(persona_id: str, session_idx: int, dialog: RawDialog) -> PersonaSession:
    """Wrap a RawDialog as a PersonaSession with PAD values."""
    dominant = dialog.dominant_emotion
    pad = EKMAN_PAD_MAP[dominant]
    return PersonaSession(
        session_id=f"{persona_id}_s{session_idx}",
        dialog_id=dialog.dialog_id,
        turns=dialog.turns,
        dominant_emotion=dominant,
        dominant_emotion_name=EKMAN_LABEL_NAMES[dominant],
        valence=pad[0],
        arousal=pad[1],
        dominance=pad[2],
        emotion_density=dialog.emotion_density,
    )


def build_personas(
    dialogs: list[RawDialog],
    *,
    n: int = 120,
    seed: int = 0,
    sessions_min: int = SESSIONS_PER_PERSONA_MIN,
    sessions_max: int = SESSIONS_PER_PERSONA_MAX,
    max_attempts_per_persona: int = 50,
) -> list[Persona]:
    """Build *n* synthetic personas from *dialogs*.

    Each persona consists of 4-5 randomly sampled dialogs (sessions).
    Dialogs are sampled without replacement *within* a persona.
    Personas that fail query generation after *max_attempts_per_persona*
    draws are skipped and a warning is printed.
    """
    rng = random.Random(seed)
    personas: list[Persona] = []
    persona_idx = 0

    while len(personas) < n:
        if persona_idx > n * 5:
            raise RuntimeError(
                f"Could not build {n} valid personas after {persona_idx} attempts. "
                "Try lowering the emotion_density_threshold or increasing n_dialogs."
            )

        persona_id = f"persona_{len(personas):03d}"
        n_sessions = rng.randint(sessions_min, sessions_max)

        if len(dialogs) < n_sessions:
            raise ValueError(
                f"Not enough filtered dialogs ({len(dialogs)}) "
                f"to sample {n_sessions} sessions per persona."
            )

        sampled = rng.sample(dialogs, n_sessions)
        sessions = [_build_persona_session(persona_id, i, d) for i, d in enumerate(sampled)]

        persona = Persona(persona_id=persona_id, sessions=sessions)

        queries = build_queries(persona, rng=rng)
        if not queries:
            persona_idx += 1
            continue

        persona.queries = queries
        personas.append(persona)
        persona_idx += 1

    return personas


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DailyDialog synthetic-persona dataset")
    p.add_argument("--n", type=int, default=120, help="Number of personas to generate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=EMOTION_DENSITY_THRESHOLD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dry-run", action="store_true", help="Build 5 personas and print summary")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    n = 5 if args.dry_run else args.n

    print("Loading raw DailyDialog from HuggingFace Hub (roskoN/dailydialog) …")
    raw = _load_raw_dialogs()
    print(f"  Loaded {len(raw)} dialogs")

    filtered = _filter_dialogs(raw, threshold=args.threshold)
    print(f"  After emotion-density filter (≥{args.threshold:.0%}): {len(filtered)} dialogs")

    print(f"Building {n} personas (seed={args.seed}) …")
    personas = build_personas(filtered, n=n, seed=args.seed)
    print(
        f"  Built {len(personas)} personas, "
        f"{sum(p.n_turns for p in personas)} total turns, "
        f"{sum(len(p.queries) for p in personas)} queries"
    )

    dataset = DailyDialogPersonaDataset(
        version="1",
        n_personas=len(personas),
        seed=args.seed,
        emotion_density_threshold=args.threshold,
        personas=personas,
    )

    if args.dry_run:
        for p in personas[:2]:
            print(f"\n  Persona {p.persona_id}: {p.n_sessions} sessions, {len(p.queries)} queries")
            for q in p.queries:
                print(f"    [{q.query_type}] {q.text}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(dataset.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved to {args.out}")
    print(
        f"Total personas: {len(personas)}, queries: {dataset.total_queries}, "
        f"turns: {dataset.total_turns}"
    )


if __name__ == "__main__":
    main()
