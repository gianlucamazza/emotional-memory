"""LoCoMo dataset loader with on-demand local caching.

Dataset: Maharana et al., ACL 2024 — 10 long multi-session conversations, ~1986 QA pairs
         (includes adversarial cat-5; ~1540 if cat-5 excluded, as in the original paper tables).
Repo:    https://github.com/snap-research/locomo  (CC BY-NC 4.0)
File:    locomo10.json (~2.7 MB)

The raw file is NOT redistributed — it is downloaded on first use to
``~/.cache/emotional_memory/locomo/locomo10.json``.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "emotional_memory" / "locomo"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "locomo10.json"

QA_CATEGORY_NAMES: dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


@dataclass(frozen=True)
class Turn:
    dia_id: str
    speaker: str
    text: str


@dataclass(frozen=True)
class Session:
    session_num: int
    date_time: str
    turns: list[Turn]


@dataclass(frozen=True)
class QAPair:
    question: str
    answer: str
    category: int
    evidence_dia_ids: list[str]
    adversarial_answer: str | None = None

    @property
    def category_name(self) -> str:
        return QA_CATEGORY_NAMES.get(self.category, f"category_{self.category}")

    @property
    def is_adversarial(self) -> bool:
        return self.category == 5


@dataclass
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session]
    qa_pairs: list[QAPair]

    @property
    def all_turns(self) -> list[tuple[int, Turn]]:
        """(session_num, turn) for every turn in conversation order."""
        return [(s.session_num, t) for s in self.sessions for t in s.turns]


@dataclass
class LoCoMoDataset:
    conversations: list[Conversation]

    @property
    def total_qa(self) -> int:
        return sum(len(c.qa_pairs) for c in self.conversations)

    @property
    def total_turns(self) -> int:
        return sum(
            len(t) for c in self.conversations for _, t in [(s, s.turns) for s in c.sessions]
        )


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading LoCoMo dataset from {url} …")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted URL)
    print(f"Saved to {dest}")


def _parse_conversation(raw: dict[str, Any]) -> Conversation:
    conv_data = raw.get("conversation", {})
    speaker_a: str = conv_data.get("speaker_a", "Speaker A")
    speaker_b: str = conv_data.get("speaker_b", "Speaker B")

    sessions: list[Session] = []
    session_num = 1
    while True:
        key = f"session_{session_num}"
        if key not in conv_data:
            break
        date_time_raw = conv_data.get(f"session_{session_num}_date_time", "")
        turns_raw: list[dict[str, Any]] = conv_data[key]
        turns = [
            Turn(
                dia_id=t.get("dia_id", ""),
                speaker=t.get("speaker", ""),
                text=(t.get("text") or t.get("blip_caption") or "").strip(),
            )
            for t in turns_raw
            if (t.get("text") or t.get("blip_caption", "")).strip()
        ]
        sessions.append(
            Session(session_num=session_num, date_time=str(date_time_raw), turns=turns)
        )
        session_num += 1

    qa_pairs: list[QAPair] = [
        QAPair(
            question=qa.get("question", ""),
            answer=qa.get("answer", ""),
            category=int(qa.get("category", 4)),
            evidence_dia_ids=list(qa.get("evidence", [])),
            adversarial_answer=qa.get("adversarial_answer"),
        )
        for qa in raw.get("qa", [])
    ]

    return Conversation(
        sample_id=raw.get("sample_id", ""),
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        sessions=sessions,
        qa_pairs=qa_pairs,
    )


def load_dataset(
    path: Path | None = None,
    *,
    url: str = LOCOMO_URL,
    limit_conversations: int | None = None,
    limit_qa_per_conversation: int | None = None,
) -> LoCoMoDataset:
    """Load LoCoMo dataset from *path*, downloading if absent.

    Parameters
    ----------
    path:
        Local path to ``locomo10.json``. Defaults to
        ``~/.cache/emotional_memory/locomo/locomo10.json``.
    url:
        Download URL if the file is absent.
    limit_conversations:
        Only load the first N conversations (for smoke tests).
    limit_qa_per_conversation:
        Cap QA pairs per conversation (for cost estimation).
    """
    dest = path if path is not None else DEFAULT_CACHE_FILE
    if not dest.exists():
        _download(url, dest)

    raw_list: list[dict[str, Any]] = json.loads(dest.read_text(encoding="utf-8"))

    if limit_conversations is not None:
        raw_list = raw_list[:limit_conversations]

    conversations: list[Conversation] = []
    for raw in raw_list:
        conv = _parse_conversation(raw)
        if limit_qa_per_conversation is not None:
            conv = Conversation(
                sample_id=conv.sample_id,
                speaker_a=conv.speaker_a,
                speaker_b=conv.speaker_b,
                sessions=conv.sessions,
                qa_pairs=conv.qa_pairs[:limit_qa_per_conversation],
            )
        conversations.append(conv)

    return LoCoMoDataset(conversations=conversations)
