"""ES-MemEval/EvoEmo dataset loader (Addendum X2).

Loads ``data/evo_emo.json`` from the ES-MemEval v1.0.0 release (WWW 2026,
CC-BY-4.0; Chen et al., arXiv:2602.01885; Zenodo 10.5281/zenodo.18338564),
pinned by sha256 per the pre-registration
``benchmarks/preregistration_addendum_x2_esmemeval_third_party.md``. The loader
FAILS on any hash mismatch — the study is defined on these exact bytes.

Session documents replicate the upstream session-level evaluation
(``SessionWiseMemoryInplaceStrategy``): one document per support session, text =
the transcript rendered as ``"<speaker>: <utterance>"`` lines (seeker name /
``supporter``). Gold = the set of same-seeker sessions referenced by the
question's ``"<session_id>:<turn>"`` evidence refs (session-level binary
relevance, upstream ``message_indices``-intersection semantics). Event-timeline
refs (``p*_event_*``) never resolve to session documents — identical to
upstream. In-family queries are those with >=1 resolvable session-level gold.

Session ids are NOT globally unique across seekers (401 sessions, 400 distinct
ids); documents are keyed by the composite ``"<seeker_id>/<session_id>"``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "esmemeval"
EVO_EMO_FILE = DATA_DIR / "evo_emo.json"

# Pinned in the pre-registration; ES-MemEval v1.0.0 (GitHub tree 6926242).
EVO_EMO_SHA256 = "f30698e87fddaeff51270a666c654da604f487a3456ec60d2b6ae08a6fecd420"

N_SEEKERS = 18
N_SESSIONS = 401
N_QUESTIONS = 1427
N_IN_FAMILY = 1133

# Upstream qa_retrieval design: 50 provided candidates per query.
POOL_SIZE = 50
POOL_SEED = 0

# Diagnostic D1 label mapping, fixed ex-ante in the pre-registration. Sessions
# with a single positive label form the positive class; ambiguous/compound
# labels are excluded; every other single label is negative.
POSITIVE_EMOTIONS = frozenset({"relief", "hope", "hopeful", "hopefulness", "pride"})
AMBIGUOUS_EMOTIONS = frozenset({"nostalgia", "mixed emotions", "conflicted", "misunderstanding"})


@dataclass(frozen=True)
class EsmemSession:
    """One support session, rendered as a retrieval document."""

    key: str  # "<seeker_id>/<session_id>" (session ids are not globally unique)
    seeker_id: str
    session_id: str
    timestamp: str  # YYYY-MM-DD
    emotion: str  # raw upstream label (compound labels joined with ", ")
    text: str  # full transcript, "<speaker>: <utterance>" per line


@dataclass(frozen=True)
class EsmemQuery:
    """One QA retrieval query (question text, session-level gold)."""

    query_id: int  # global file order over all questions (0-based)
    seeker_id: str
    group_id: str
    capability: str
    text: str
    gold_keys: frozenset[str]  # composite session keys; empty = zero-gold


@dataclass(frozen=True)
class EsmemDataset:
    sessions: tuple[EsmemSession, ...]
    queries: tuple[EsmemQuery, ...]  # in-family only (>=1 resolvable gold)
    n_zero_gold: int  # excluded questions (upstream scores them 0 on all arms)


def emotion_class(emotion: str) -> str:
    """Classify a raw session emotion label for diagnostic D1.

    Returns ``"positive"``, ``"negative"``, or ``"excluded"`` (ambiguous or
    compound labels; mapping fixed ex-ante in the pre-registration).
    """
    label = emotion.strip().lower()
    if "," in label or label in AMBIGUOUS_EMOTIONS:
        return "excluded"
    if label in POSITIVE_EMOTIONS:
        return "positive"
    return "negative"


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"{path.name}: sha256 mismatch — got {digest}, pinned {expected}. "
            "Refusing to run on unpinned data (see Addendum X2 pre-registration)."
        )


def _render_transcript(seeker_name: str, dialogue: list[dict[str, object]]) -> str:
    """Replicate upstream ``SessionWiseMemoryInplaceStrategy._as_text``."""
    lines: list[str] = []
    for turn in dialogue:
        role, content = str(turn["role"]), str(turn["content"])
        if role == "seeker":
            lines.append(f"{seeker_name}: {content}")
        elif role == "supporter":
            lines.append(f"supporter: {content}")
        else:
            raise RuntimeError(f"unexpected dialogue role: {role!r}")
    return "\n".join(lines)


def _normalize_emotion(raw: object) -> str:
    # One session carries a list of labels; joined -> compound -> D1-excluded.
    if isinstance(raw, list):
        return ", ".join(str(x) for x in raw)
    return str(raw)


def load_dataset(
    data_file: Path = EVO_EMO_FILE,
    *,
    verify_hashes: bool = True,
) -> EsmemDataset:
    """Load and validate the pinned ES-MemEval v1.0.0 EvoEmo corpus."""
    if verify_hashes:
        _verify_sha256(data_file, EVO_EMO_SHA256)

    with data_file.open(encoding="utf-8") as fh:
        seekers = json.load(fh)

    sessions: list[EsmemSession] = []
    queries: list[EsmemQuery] = []
    n_zero_gold = 0
    query_id = 0
    for seeker in seekers:
        seeker_id = str(seeker["id"])
        seeker_name = str(seeker["basic_info"]["name"])
        own_session_ids: set[str] = set()
        for sess in seeker["dialog_history"]:
            session_id = str(sess["id"])
            own_session_ids.add(session_id)
            sessions.append(
                EsmemSession(
                    key=f"{seeker_id}/{session_id}",
                    seeker_id=seeker_id,
                    session_id=session_id,
                    timestamp=str(sess["timestamp"]),
                    emotion=_normalize_emotion(sess["emotion"]),
                    text=_render_transcript(seeker_name, sess["dialogue"]),
                )
            )
        for group in seeker["questions"]:
            group_id = str(group["id"])
            for question in group["questions"]:
                evidence = question.get("evidence") or []
                gold = {
                    f"{seeker_id}/{sid}"
                    for sid in (str(ref).split(":", 1)[0] for ref in evidence if ":" in str(ref))
                    if sid in own_session_ids
                }
                if gold:
                    queries.append(
                        EsmemQuery(
                            query_id=query_id,
                            seeker_id=seeker_id,
                            group_id=group_id,
                            capability=str(question["capability"]),
                            text=str(question["question"]),
                            gold_keys=frozenset(gold),
                        )
                    )
                else:
                    n_zero_gold += 1
                query_id += 1

    if len(seekers) != N_SEEKERS or len(sessions) != N_SESSIONS:
        raise RuntimeError(
            f"expected {N_SEEKERS} seekers / {N_SESSIONS} sessions, "
            f"got {len(seekers)} / {len(sessions)}"
        )
    if query_id != N_QUESTIONS or len(queries) != N_IN_FAMILY:
        raise RuntimeError(
            f"expected {N_QUESTIONS} questions / {N_IN_FAMILY} in-family, "
            f"got {query_id} / {len(queries)}"
        )
    return EsmemDataset(sessions=tuple(sessions), queries=tuple(queries), n_zero_gold=n_zero_gold)


def build_pools(
    dataset: EsmemDataset,
    *,
    pool_size: int = POOL_SIZE,
    seed: int = POOL_SEED,
) -> dict[int, tuple[str, ...]]:
    """Build the per-query 50-candidate pools (upstream design, shared by arms).

    Per query: all gold sessions + all remaining same-seeker sessions +
    cross-seeker sessions sampled at random to fill to ``pool_size``, then
    shuffled. RNG: one master seed, one derived seed per query in file order
    (fixed ex-ante in the pre-registration §Protocol).
    """
    by_seeker: dict[str, list[str]] = {}
    for sess in dataset.sessions:
        by_seeker.setdefault(sess.seeker_id, []).append(sess.key)

    master = Random(seed)
    pools: dict[int, tuple[str, ...]] = {}
    for query in dataset.queries:
        rng = Random(master.randrange(2**32))
        own = by_seeker[query.seeker_id]
        cross = [k for sid, keys in by_seeker.items() if sid != query.seeker_id for k in keys]
        pool = list(own)  # gold + same-seeker non-gold (banks are 13-33 < pool_size)
        fill = pool_size - len(pool)
        if fill < 0:
            raise RuntimeError(f"query {query.query_id}: bank larger than pool size")
        pool.extend(rng.sample(cross, fill))
        rng.shuffle(pool)
        pools[query.query_id] = tuple(pool)
    return pools
