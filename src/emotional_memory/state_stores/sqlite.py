"""SQLite-backed persistence for the current affective state snapshot."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from emotional_memory.state import AffectiveState

_CREATE_STATE = """
CREATE TABLE IF NOT EXISTS affective_state (
    slot TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
"""


class SQLiteAffectiveStateStore:
    """Persist the current affective state in a small SQLite table.

    Unlike ``SQLiteStore``, this backend does not require ``sqlite-vec`` and is
    safe to use in the base package for local persistence or replay harnesses.
    """

    __slots__ = ("_conn", "_lock", "_path")

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_STATE)
        self._conn.commit()
        self._lock: threading.RLock = threading.RLock()

    def save(self, state: AffectiveState) -> None:
        payload = json.dumps(state.snapshot())
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO affective_state (slot, data) VALUES ('current', ?)",
                (payload,),
            )

    def load(self) -> AffectiveState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM affective_state WHERE slot = 'current'"
            ).fetchone()
        if row is None:
            return None
        return AffectiveState.restore(json.loads(row["data"]))

    def clear(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM affective_state WHERE slot = 'current'")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SQLiteAffectiveStateStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={self._path!r})"
