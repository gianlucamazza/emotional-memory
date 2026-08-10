"""SQLiteStore — persistent MemoryStore backed by SQLite + sqlite-vec.

Requires the ``sqlite-vec`` optional dependency::

    pip install emotional-memory[sqlite]

Design decisions
----------------
- Full ``Memory`` model is stored as a JSON blob (``Memory.model_dump_json()``)
  to avoid fragile normalisation of the deep Pydantic model tree.
- Embeddings are stored separately as raw float32 bytes in a ``sqlite-vec``
  virtual table for ANN-accelerated ``search_by_embedding``. The table is
  created with ``distance_metric=cosine`` so the ANN ordering matches the
  cosine similarity used by the retrieval scorer, the brute-force fallback,
  and ``InMemoryStore``. (``vec0`` defaults to L2, which ranks differently
  from cosine whenever embeddings are not L2-normalised.)
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

import re
import sqlite3
import struct
import threading
import types
import warnings
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

# Cosine is the metric the rest of the pipeline scores with (retrieval._cosine,
# InMemoryStore, _brute_force_search); vec0 would otherwise default to L2.
_CREATE_VEC = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec "
    "USING vec0(id TEXT PRIMARY KEY, embedding float[{dim}] distance_metric=cosine);"
)
# Fallback for sqlite-vec builds that do not accept the distance_metric option.
_CREATE_VEC_LEGACY = (
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

    __slots__ = ("_conn", "_dim", "_lock", "_path", "_vec_cosine", "_vec_ready")

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        sqlite_vec = _load_sqlite_vec()
        # check_same_thread=False is required when the store is used through
        # SyncToAsyncStore (asyncio.to_thread dispatches on arbitrary threads).
        # WAL mode allows concurrent readers without blocking writers.
        self._conn: sqlite3.Connection = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_MEMORIES)
        self._conn.commit()
        self._vec_ready = False
        self._vec_cosine = True
        self._dim: int = 0
        # Serialise all connection access across threads: a single shared
        # sqlite3.Connection is not safe for concurrent use even with
        # check_same_thread=False (Python's sqlite3 module leaves locking to
        # the caller).
        self._lock: threading.RLock = threading.RLock()
        self._init_vec_from_db()

    # ------------------------------------------------------------------
    # MemoryStore protocol
    # ------------------------------------------------------------------

    def save(self, memory: Memory) -> None:
        with self._lock:
            self._ensure_vec(memory)
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO memories (id, content, data) VALUES (?, ?, ?)",
                    (memory.id, memory.content, memory.model_dump_json()),
                )
                if self._vec_ready and memory.embedding is not None:
                    self._conn.execute("DELETE FROM memory_vec WHERE id = ?", (memory.id,))
                    self._conn.execute(
                        "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
                        (memory.id, _pack_embedding(memory.embedding)),
                    )

    def get(self, memory_id: str) -> Memory | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        if row is None:
            return None
        return Memory.model_validate_json(row["data"])

    def update(self, memory: Memory) -> None:
        with self._lock:
            self._ensure_vec(memory)
            with self._conn:
                cursor = self._conn.execute(
                    "UPDATE memories SET content = ?, data = ? WHERE id = ?",
                    (memory.content, memory.model_dump_json(), memory.id),
                )
                if cursor.rowcount == 0:
                    return
                if self._vec_ready and memory.embedding is not None:
                    self._conn.execute("DELETE FROM memory_vec WHERE id = ?", (memory.id,))
                    self._conn.execute(
                        "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
                        (memory.id, _pack_embedding(memory.embedding)),
                    )

    def delete(self, memory_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            if self._vec_ready:
                self._conn.execute("DELETE FROM memory_vec WHERE id = ?", (memory_id,))

    def list_all(self) -> list[Memory]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM memories").fetchall()
        return [Memory.model_validate_json(row["data"]) for row in rows]

    def search_by_embedding(self, embedding: list[float], top_k: int) -> list[Memory]:
        """Return top_k memories by approximate cosine similarity (sqlite-vec).

        Falls back to brute-force cosine scan when the vector table is not yet
        initialised (e.g. no memories with embeddings have been saved yet).

        Zero-length vectors have no defined cosine distance and come back with a
        NULL distance, which SQLite sorts *first*; they are moved to the end so a
        degenerate embedding cannot outrank genuine matches — mirroring
        ``_math.cosine_similarity``, which scores a zero vector 0.0.
        """
        if top_k <= 0:
            return []
        with self._lock:
            if not self._vec_ready:
                return self._brute_force_search(embedding, top_k)

            rows = self._conn.execute(
                """
                SELECT m.data AS data, v.distance AS distance
                FROM memory_vec v
                JOIN memories m ON m.id = v.id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                (_pack_embedding(embedding), top_k),
            ).fetchall()
        ordered = [row for row in rows if row["distance"] is not None]
        ordered += [row for row in rows if row["distance"] is None]
        return [Memory.model_validate_json(row["data"]) for row in ordered]

    def __len__(self) -> int:
        with self._lock:
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
        try:
            self._conn.execute(_CREATE_VEC.format(dim=dim))
        except sqlite3.OperationalError:
            # sqlite-vec build without distance_metric support: keep working on
            # the default (L2) index, but say so — ANN ordering then differs from
            # the cosine scoring used everywhere else.
            self._conn.execute(_CREATE_VEC_LEGACY.format(dim=dim))
            self._vec_cosine = False
            warnings.warn(
                "sqlite-vec does not support 'distance_metric=cosine'; the vector "
                "index will rank by L2 distance, which differs from the cosine "
                "similarity used by retrieval scoring. Upgrade sqlite-vec to get "
                "cosine-ordered candidate prefiltering.",
                stacklevel=2,
            )
        else:
            self._vec_cosine = True
        self._conn.commit()
        self._vec_ready = True
        self._dim = dim

    def _init_vec_from_db(self) -> None:
        """Re-attach vec table if it already exists from a previous session."""
        try:
            row = self._conn.execute("SELECT embedding FROM memory_vec LIMIT 1").fetchone()
            schema_row = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'memory_vec'"
            ).fetchone()
            if row is not None:
                self._dim = len(_unpack_embedding(bytes(row["embedding"])))
            elif schema_row is not None and schema_row[0]:
                # Table exists but is empty — parse dimension from schema SQL.
                m = re.search(r"float\[(\d+)\]", schema_row[0])
                if m:
                    self._dim = int(m.group(1))
            schema_sql = schema_row[0] if schema_row is not None and schema_row[0] else ""
            self._vec_cosine = "distance_metric=cosine" in schema_sql.replace(" ", "")
            if not self._vec_cosine:
                # Written by emotional-memory < 0.18, whose index ranked by L2.
                # The table is derived data (embeddings also live in the JSON
                # blob), so it is safe to rebuild — but never silently.
                warnings.warn(
                    f"{self._path!r} has a legacy L2-ranked vector index; candidate "
                    "prefiltering will not match the cosine similarity used by "
                    "retrieval scoring. Call rebuild_vector_index() once to migrate.",
                    stacklevel=2,
                )
            self._vec_ready = True
        except sqlite3.OperationalError:
            # Table does not exist yet — will be created on first save()
            self._vec_ready = False

    def rebuild_vector_index(self) -> int:
        """Recreate the vector index from the stored memories, using cosine ranking.

        The vec table holds only derived data (every embedding is also inside the
        ``memories`` JSON blob), so it can be dropped and rebuilt safely. Use this
        to migrate a database created before the index switched from L2 to cosine
        ordering.

        Returns:
            Number of embeddings re-indexed.
        """
        with self._lock:
            memories = [m for m in self.list_all() if m.embedding is not None]
            with self._conn:
                self._conn.execute("DROP TABLE IF EXISTS memory_vec")
            self._vec_ready = False
            self._dim = 0
            if not memories:
                return 0
            self._ensure_vec(memories[0])
            with self._conn:
                for memory in memories:
                    if memory.embedding is None:  # pragma: no cover - filtered above
                        continue
                    self._conn.execute(
                        "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
                        (memory.id, _pack_embedding(memory.embedding)),
                    )
        return len(memories)

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

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={self._path!r}, count={len(self)})"

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
