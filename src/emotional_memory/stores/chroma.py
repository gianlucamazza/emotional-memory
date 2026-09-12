"""ChromaStore — remote MemoryStore backed by a Chroma vector database.

Requires the HTTP-only Chroma client from the ``chroma`` optional extra::

    pip install emotional-memory[chroma]

Design contract
---------------
- **Embeddings are mandatory**. ``ChromaStore`` is a vector-first adapter and
  rejects ``Memory`` objects with ``embedding=None`` via ``ValueError``. In
  normal usage through :class:`EmotionalMemory`, the engine always embeds
  content before calling ``store.save()``, so this is transparent.
- Full ``Memory`` model is stored as a JSON string in the Chroma document
  field, mirroring SQLiteStore and QdrantStore's approach of avoiding fragile
  normalisation of the deep Pydantic model tree.
- Memory IDs are used directly as Chroma point IDs (Chroma accepts arbitrary
  strings, unlike Qdrant which requires UUID format).
- The Chroma collection is created lazily on the first ``save()`` call; the
  vector dimension is inferred from the first embedding and stored in the
  collection metadata under ``"dim"`` for persistence-resume support.
- Cosine distance is the default similarity metric, matching ``InMemoryStore``,
  ``SQLiteStore``, and ``QdrantStore``.
- Embeddings are passed as ``np.float32`` arrays, matching chromadb's native
  representation.

Usage::

    from emotional_memory.stores.chroma import ChromaStore

    store = ChromaStore(host="localhost", port=8000)   # remote HTTP server
    engine = EmotionalMemory(store, embedder)

Security
--------
The optional dependency is Chroma's HTTP-only client: the vulnerable embedded
server is deliberately not installed. ``ChromaStore`` always supplies embeddings
on write/query and must connect to a separately managed, trusted, patched server.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from emotional_memory.models import Memory

if TYPE_CHECKING:
    from chromadb import Collection as _ChromaCollection
    from chromadb.api import ClientAPI as _ChromaClientAPI

logger = logging.getLogger(__name__)

_SCROLL_PAGE = 1_000
_INCLUDE_DOCS: Any = ["documents"]
_INCLUDE_EMBEDDINGS: Any = ["embeddings"]


def _require_embedding(memory: Memory) -> list[float]:
    if memory.embedding is None:
        raise ValueError(
            f"ChromaStore requires memories to have an embedding. "
            f"Memory id={memory.id!r} was passed with embedding=None. "
            "Use EmotionalMemory(store, embedder) so the engine embeds "
            "content before storage, or set memory.embedding explicitly."
        )
    return memory.embedding


class ChromaStore:
    """MemoryStore backed by a Chroma vector database.

    Parameters
    ----------
    path:
        Unsupported. Retained to provide a migration error for callers that used
        the former embedded persistent mode.
    host:
        Remote Chroma server hostname, e.g. ``"localhost"``. Required.
    port:
        Remote Chroma server port (default 8000). Only used when ``host``
        is provided.
    collection_name:
        Name of the Chroma collection. Created lazily on first ``save()``.
    """

    __slots__ = (
        "_client",
        "_collection",
        "_collection_name",
        "_dim",
    )

    def __init__(
        self,
        *,
        path: str | None = None,
        host: str | None = None,
        port: int = 8000,
        collection_name: str = "emotional_memory",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "the Chroma HTTP client is required for ChromaStore. "
                "Install with: pip install 'emotional-memory[chroma]'"
            ) from exc

        if path is not None:
            raise ValueError(
                "ChromaStore embedded persistence is unavailable for security reasons; "
                "run a patched Chroma server and pass host= instead"
            )
        if host is None:
            raise ValueError(
                "ChromaStore requires host= because the [chroma] extra installs only "
                "the HTTP client"
            )
        self._client: _ChromaClientAPI = chromadb.HttpClient(host=host, port=port)
        self._collection_name = collection_name
        self._collection: _ChromaCollection | None = None
        self._dim: int = 0
        self._init_collection_state()

    # ------------------------------------------------------------------
    # MemoryStore protocol
    # ------------------------------------------------------------------

    def save(self, memory: Memory) -> None:
        embedding = _require_embedding(memory)
        col = self._ensure_collection(len(embedding))
        col.upsert(
            ids=[memory.id],
            embeddings=np.array([embedding], dtype=np.float32),
            documents=[memory.model_dump_json()],
            metadatas=[{"id": memory.id}],
        )
        logger.debug("ChromaStore.save: upserted point for memory %s", memory.id)

    def get(self, memory_id: str) -> Memory | None:
        col = self._collection
        if col is None:
            return None
        results = col.get(ids=[memory_id], include=_INCLUDE_DOCS)
        docs = results.get("documents") or []
        if not docs or docs[0] is None:
            return None
        return Memory.model_validate_json(docs[0])

    def update(self, memory: Memory) -> None:
        if self.get(memory.id) is None:
            return
        self.save(memory)

    def delete(self, memory_id: str) -> None:
        col = self._collection
        if col is None:
            return
        col.delete(ids=[memory_id])

    def list_all(self) -> list[Memory]:
        col = self._collection
        if col is None:
            return []
        memories: list[Memory] = []
        offset = 0
        while True:
            results = col.get(
                limit=_SCROLL_PAGE,
                offset=offset,
                include=_INCLUDE_DOCS,
            )
            docs = results.get("documents") or []
            if not docs:
                break
            memories.extend(Memory.model_validate_json(doc) for doc in docs if doc is not None)
            if len(docs) < _SCROLL_PAGE:
                break
            offset += len(docs)
        return memories

    def search_by_embedding(self, embedding: list[float], top_k: int) -> list[Memory]:
        col = self._collection
        if col is None:
            return []
        count = len(self)
        if count == 0:
            return []
        n = min(top_k, count)
        results = col.query(
            query_embeddings=np.array([embedding], dtype=np.float32),
            n_results=n,
            include=_INCLUDE_DOCS,
        )
        docs_list = results.get("documents") or [[]]
        return [Memory.model_validate_json(doc) for doc in docs_list[0] if doc is not None]

    def __len__(self) -> int:
        col = self._collection
        if col is None:
            return 0
        return int(col.count())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_collection_state(self) -> None:
        """Attach to an existing collection if present (persistent-mode resume)."""
        try:
            col = self._client.get_collection(self._collection_name)
        except Exception as exc:
            logger.debug(
                "ChromaStore: could not attach collection '%s' (%s: %s)",
                self._collection_name,
                type(exc).__name__,
                exc,
            )
            return
        dim_val = (col.metadata or {}).get("dim")
        if dim_val is not None:
            self._dim = int(dim_val)
            self._collection = col
            logger.debug(
                "ChromaStore: attached to existing collection '%s' (dim=%d)",
                self._collection_name,
                self._dim,
            )
            return
        # No dim metadata — try to infer from existing embeddings
        results = col.get(limit=1, include=_INCLUDE_EMBEDDINGS)
        existing: Any = results.get("embeddings") or []
        if existing and existing[0] is not None:
            self._dim = len(existing[0])
            self._collection = col
            logger.debug(
                "ChromaStore: attached to existing collection '%s' (dim=%d, inferred)",
                self._collection_name,
                self._dim,
            )
        else:
            raise TypeError(
                f"ChromaStore: collection '{self._collection_name}' has no 'dim' metadata "
                "and is empty — cannot determine embedding dimension. "
                "Use a different collection_name or delete this collection first."
            )

    def _ensure_collection(self, dim: int) -> _ChromaCollection:
        """Return the Chroma collection, creating it on the first call."""
        col = self._collection
        if col is not None:
            if dim != self._dim:
                raise ValueError(
                    f"ChromaStore: embedding dimension {dim} does not match "
                    f"collection dimension {self._dim}"
                )
            return col
        col = self._client.create_collection(
            self._collection_name,
            metadata={"hnsw:space": "cosine", "dim": str(dim)},
        )
        self._collection = col
        self._dim = dim
        logger.debug("ChromaStore: created collection '%s' (dim=%d)", self._collection_name, dim)
        return col

    def close(self) -> None:
        """Close the ChromaStore (Chroma clients manage their own lifecycle)."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(collection={self._collection_name!r}, count={len(self)})"

    def __enter__(self) -> ChromaStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
