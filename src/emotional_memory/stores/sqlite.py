"""SQLiteStore — persistent MemoryStore backed by SQLite + sqlite-vec.

Requires the ``sqlite-vec`` optional dependency::

    pip install emotional-memory[sqlite]

Design decisions
----------------
- Full ``Memory`` model is stored as a JSON blob (``Memory.model_dump_json()``)
  to avoid fragile normalisation of the deep Pydantic model tree.
- Embeddings are stored separately as raw float32 bytes in a ``sqlite-vec``
  virtual table for ANN-accelerated ``search_by_embedding``.
- The embedding vector dimension is detected on the first ``save()`` call
  that includes a non-null embedding; the virtual table is created at that
  point.
- All writes use explicit transactions for atomicity across the memories and
  memory_vec tables.
- ``__len__`` is a single ``COUNT(*)`` — never materialises the full dataset.

Usage::

    from emotional_memory.stores.sqlite import SQLiteStore

    store = SQLiteStore("memories.db")
    engine = EmotionalMemory(store, embedder)
"""

from __future__ import annotations

import sqlite3
import struct
import types
from pathlib import Path

from emotional_memory.models import Memory


def _load_sqlite_vec() -> types.ModuleType:
    """Import sqlite-vec at runtime, raising a clear error when missing."""
    try:
        import importlib

        return importlib.import_module("sqlite_vec")
    except ImportError as exc:
        raise ImportError(
            "sqlite-vec is required for SQLiteStore.\n"
            "Install with:  pip install 'emotional-memory[sqlite]'\n"
            "or:            pip install sqlite-vec"
        ) from exc


# ---------------------------------------------------------------------------
# Byte-packing helpers for float32 embeddings
# ---------------------------------------------------------------------------


def _pack_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _unpack_embedding(data: bytes) -> list[float]:
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


# ---------------------------------------------------------------------------
# SQLiteStore
# ---------------------------------------------------------------------------

_CREATE_MEMORIES = """
CREATE TABLE IF NOT EXISTS memories (
    id      TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    data    TEXT NOT NULL
);
"""

_CREATE_VEC = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec "
    "USING vec0(id TEXT PRIMARY KEY, embedding float[{dim}]);"
)


class SQLiteStore:
    """Persistent MemoryStore backed by SQLite with sqlite-vec vector search.

    Parameters
    ----------
    path:
        File path for the SQLite database, or ``":memory:"`` for an ephemeral
        in-memory database (useful in tests without writing to disk).
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        sqlite_vec = _load_sqlite_vec()
        self._conn: sqlite3.Connection = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        sqlite_vec.load(self._conn)
        self._conn.execute(_CREATE_MEMORIES)
        self._conn.commit()
        self._vec_ready = False
        self._dim: int = 0
        self._init_vec_from_db()

    # ------------------------------------------------------------------
    # MemoryStore protocol
    # ------------------------------------------------------------------

    def save(self, memory: Memory) -> None:
        self._ensure_vec(memory)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO memories (id, content, data) VALUES (?, ?, ?)",
                (memory.id, memory.content, memory.model_dump_json()),
            )
            if self._vec_ready and memory.embedding is not None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO memory_vec (id, embedding) VALUES (?, ?)",
                    (memory.id, _pack_embedding(memory.embedding)),
                )

    def get(self, memory_id: str) -> Memory | None:
        row = self._conn.execute("SELECT data FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return Memory.model_validate_json(row["data"])

    def update(self, memory: Memory) -> None:
        self._ensure_vec(memory)
        with self._conn:
            self._conn.execute(
                "UPDATE memories SET content = ?, data = ? WHERE id = ?",
                (memory.content, memory.model_dump_json(), memory.id),
            )
            if self._vec_ready and memory.embedding is not None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO memory_vec (id, embedding) VALUES (?, ?)",
                    (memory.id, _pack_embedding(memory.embedding)),
                )

    def delete(self, memory_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            if self._vec_ready:
                self._conn.execute("DELETE FROM memory_vec WHERE id = ?", (memory_id,))

    def list_all(self) -> list[Memory]:
        rows = self._conn.execute("SELECT data FROM memories").fetchall()
        return [Memory.model_validate_json(row["data"]) for row in rows]

    def search_by_embedding(self, embedding: list[float], top_k: int) -> list[Memory]:
        """Return top_k memories by approximate cosine similarity (sqlite-vec).

        Falls back to brute-force cosine scan when the vector table is not yet
        initialised (e.g. no memories with embeddings have been saved yet).
        """
        if not self._vec_ready:
            return self._brute_force_search(embedding, top_k)

        rows = self._conn.execute(
            """
            SELECT m.data
            FROM memory_vec v
            JOIN memories m ON m.id = v.id
            WHERE v.embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (_pack_embedding(embedding), top_k),
        ).fetchall()
        return [Memory.model_validate_json(row["data"]) for row in rows]

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return int(row[0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_vec(self, memory: Memory) -> None:
        """Create the vec virtual table on the first memory with an embedding."""
        if self._vec_ready or memory.embedding is None:
            return
        dim = len(memory.embedding)
        self._conn.execute(_CREATE_VEC.format(dim=dim))
        self._conn.commit()
        self._vec_ready = True
        self._dim = dim

    def _init_vec_from_db(self) -> None:
        """Re-attach vec table if it already exists from a previous session."""
        try:
            row = self._conn.execute("SELECT embedding FROM memory_vec LIMIT 1").fetchone()
            if row is not None:
                dim = len(_unpack_embedding(bytes(row["embedding"])))
                self._dim = dim
            self._vec_ready = True
        except sqlite3.OperationalError:
            # Table does not exist yet — will be created on first save()
            self._vec_ready = False

    def _brute_force_search(self, embedding: list[float], top_k: int) -> list[Memory]:
        from emotional_memory._math import cosine_similarity

        memories = self.list_all()
        scored = [
            (cosine_similarity(embedding, m.embedding), m)
            for m in memories
            if m.embedding is not None
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
