"""QdrantStore — persistent MemoryStore backed by a Qdrant vector database.

Requires the ``qdrant-client`` optional dependency::

    pip install emotional-memory[qdrant]

Design contract
---------------
- **Embeddings are mandatory**. ``QdrantStore`` is a vector-first adapter and
  rejects ``Memory`` objects with ``embedding=None`` via ``ValueError``. In
  normal usage through :class:`EmotionalMemory`, the engine always embeds
  content before calling ``store.save()``, so this is transparent.
- Full ``Memory`` model is stored as a JSON blob (``Memory.model_dump_json()``)
  in the Qdrant point payload under the ``data`` key, mirroring SQLiteStore's
  approach of avoiding fragile normalisation of the deep Pydantic model tree.
- Point IDs are UUID5 strings deterministically derived from ``memory.id``,
  enabling O(1) lookup and consistent identity across restarts.
- The Qdrant collection is created lazily on the first ``save()`` call;
  the vector dimension is inferred from the first embedding (no preset
  ``vector_size`` parameter required).
- Cosine distance is the default similarity metric, matching ``InMemoryStore``
  and ``SQLiteStore``.

Usage::

    from emotional_memory.stores.qdrant import QdrantStore

    store = QdrantStore()                              # in-memory
    store = QdrantStore(path="./qdrant_data")          # local on-disk
    store = QdrantStore(url="http://localhost:6333")   # remote server
    engine = EmotionalMemory(store, embedder)
"""

from __future__ import annotations

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from emotional_memory.models import Memory

logger = logging.getLogger(__name__)

_NAMESPACE = uuid.NAMESPACE_OID
_SCROLL_PAGE = 1_000


def _uuid_for(memory_id: str) -> str:
    """Return a deterministic UUID5 string for a memory ID."""
    return str(uuid.uuid5(_NAMESPACE, memory_id))


def _require_embedding(memory: Memory) -> list[float]:
    if memory.embedding is None:
        raise ValueError(
            f"QdrantStore requires memories to have an embedding. "
            f"Memory id={memory.id!r} was passed with embedding=None. "
            "Use EmotionalMemory(store, embedder) so the engine embeds "
            "content before storage, or set memory.embedding explicitly."
        )
    return memory.embedding


class QdrantStore:
    """MemoryStore backed by a Qdrant vector database.

    Parameters
    ----------
    url:
        Qdrant server URL, e.g. ``"http://localhost:6333"``. Mutually
        exclusive with ``path``. ``None`` (default) selects the ephemeral
        in-memory Qdrant instance, useful for tests and prototyping.
    path:
        Local filesystem path for a persistent on-disk Qdrant instance.
        Mutually exclusive with ``url``.
    collection_name:
        Name of the Qdrant collection. Created lazily on first ``save()``.
    api_key:
        Optional API key for Qdrant Cloud (ignored in local modes).
    """

    __slots__ = (
        "_client",
        "_collection_name",
        "_collection_ready",
        "_dim",
    )

    def __init__(
        self,
        *,
        url: str | None = None,
        path: str | None = None,
        collection_name: str = "emotional_memory",
        api_key: str | None = None,
    ) -> None:
        if url is not None and path is not None:
            raise ValueError("QdrantStore: pass at most one of `url` or `path`, not both")
        if url is not None:
            self._client: QdrantClient = QdrantClient(url=url, api_key=api_key)
        elif path is not None:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(":memory:")
        self._collection_name = collection_name
        self._dim: int = 0
        self._collection_ready: bool = False
        self._init_collection_state()

    # ------------------------------------------------------------------
    # MemoryStore protocol
    # ------------------------------------------------------------------

    def save(self, memory: Memory) -> None:
        embedding = _require_embedding(memory)
        self._ensure_collection(len(embedding))
        self._client.upsert(
            self._collection_name,
            points=[
                PointStruct(
                    id=_uuid_for(memory.id),
                    vector=embedding,
                    payload={"id": memory.id, "data": memory.model_dump_json()},
                )
            ],
        )
        logger.debug("QdrantStore.save: upserted point for memory %s", memory.id)

    def get(self, memory_id: str) -> Memory | None:
        if not self._collection_ready:
            return None
        records = self._client.retrieve(
            self._collection_name,
            ids=[_uuid_for(memory_id)],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        payload = records[0].payload or {}
        return Memory.model_validate_json(str(payload["data"]))

    def update(self, memory: Memory) -> None:
        if self.get(memory.id) is None:
            return
        self.save(memory)

    def delete(self, memory_id: str) -> None:
        if not self._collection_ready:
            return
        self._client.delete(
            self._collection_name,
            points_selector=PointIdsList(points=[_uuid_for(memory_id)]),
        )

    def list_all(self) -> list[Memory]:
        if not self._collection_ready:
            return []
        results: list[Memory] = []
        offset = None
        while True:
            records, next_offset = self._client.scroll(
                self._collection_name,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                payload = rec.payload or {}
                results.append(Memory.model_validate_json(str(payload["data"])))
            if next_offset is None:
                break
            offset = next_offset
        return results

    def search_by_embedding(self, embedding: list[float], top_k: int) -> list[Memory]:
        if not self._collection_ready:
            return []
        response = self._client.query_points(
            self._collection_name,
            query=embedding,
            limit=top_k,
            with_payload=True,
        )
        return [
            Memory.model_validate_json(str((hit.payload or {})["data"])) for hit in response.points
        ]

    def __len__(self) -> int:
        if not self._collection_ready:
            return 0
        return int(self._client.count(self._collection_name, exact=True).count)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_collection_state(self) -> None:
        """Attach to an existing collection if present (server-mode resume)."""
        if not self._client.collection_exists(self._collection_name):
            return
        info = self._client.get_collection(self._collection_name)
        vectors_config = info.config.params.vectors
        if not isinstance(vectors_config, VectorParams):
            raise TypeError(
                f"Collection '{self._collection_name}' uses named vectors; "
                "QdrantStore only supports single (unnamed) vector spaces"
            )
        self._dim = vectors_config.size
        self._collection_ready = True
        logger.debug(
            "QdrantStore: attached to existing collection '%s' (dim=%d)",
            self._collection_name,
            self._dim,
        )

    def _ensure_collection(self, dim: int) -> None:
        """Create the Qdrant collection on first save()."""
        if self._collection_ready:
            if dim != self._dim:
                raise ValueError(
                    f"QdrantStore: embedding dimension {dim} does not match "
                    f"collection dimension {self._dim}"
                )
            return
        self._client.create_collection(
            self._collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        self._dim = dim
        self._collection_ready = True
        logger.debug("QdrantStore: created collection '%s' (dim=%d)", self._collection_name, dim)

    def close(self) -> None:
        """Close the underlying Qdrant client connection."""
        self._client.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(collection={self._collection_name!r}, count={len(self)})"

    def __enter__(self) -> QdrantStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
