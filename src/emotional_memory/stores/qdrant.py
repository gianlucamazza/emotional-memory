"""QdrantStore — persistent MemoryStore backed by a Qdrant vector database.

Requires the ``qdrant-client`` optional dependency::

    pip install emotional-memory[qdrant]

Design decisions
----------------
- Full ``Memory`` model is stored as a JSON blob (``Memory.model_dump_json()``)
  in the Qdrant point payload under the ``data`` key, mirroring SQLiteStore's
  approach of avoiding fragile normalisation of the deep Pydantic model tree.
- Point IDs are UUID5 strings deterministically derived from ``memory.id``,
  enabling O(1) lookup and consistent identity across restarts.
- Memories without embeddings are held in an in-memory fallback dict; they are
  not persisted to Qdrant and will not survive process restarts in server mode.
  This is a known v0.9 limitation.
- The Qdrant collection is created lazily on the first ``save()`` call with a
  non-null embedding; the vector dimension is inferred at that point.
- ``search_by_embedding`` falls back to brute-force cosine scan for memories in
  the fallback dict (no embedding stored in Qdrant for those entries).

Usage::

    from emotional_memory.stores.qdrant import QdrantStore

    store = QdrantStore()  # in-memory (tests / prototyping)
    store = QdrantStore(url="http://localhost:6333")  # remote server
    engine = EmotionalMemory(store, embedder)
"""

from __future__ import annotations

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from emotional_memory._math import cosine_similarity
from emotional_memory.models import Memory

logger = logging.getLogger(__name__)

_NAMESPACE = uuid.NAMESPACE_OID
_SCROLL_LIMIT = 10_000


def _uuid_for(memory_id: str) -> str:
    """Return a deterministic UUID5 string for a memory ID."""
    return str(uuid.uuid5(_NAMESPACE, memory_id))


class QdrantStore:
    """MemoryStore backed by a Qdrant vector database.

    Parameters
    ----------
    url:
        Qdrant server URL, e.g. ``"http://localhost:6333"``.
        Mutually exclusive with ``path``. Use ``None`` (default) for the
        ephemeral in-memory Qdrant instance (useful in tests).
    path:
        Local filesystem path for a persistent on-disk Qdrant instance.
        Mutually exclusive with ``url``.
    collection_name:
        Name of the Qdrant collection. Created lazily on first ``save()``
        with an embedding.
    api_key:
        Optional API key for Qdrant Cloud (ignored in local modes).
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        path: str | None = None,
        collection_name: str = "emotional_memory",
        api_key: str | None = None,
    ) -> None:
        if url is not None:
            self._client: QdrantClient = QdrantClient(url=url, api_key=api_key)
        elif path is not None:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(":memory:")
        self._collection_name = collection_name
        self._dim: int = 0
        self._collection_ready: bool = False
        self._fallback: dict[str, Memory] = {}
        self._init_collection_state()

    # ------------------------------------------------------------------
    # MemoryStore protocol
    # ------------------------------------------------------------------

    def save(self, memory: Memory) -> None:
        if memory.embedding is not None:
            self._ensure_collection(len(memory.embedding))
            self._client.upsert(
                self._collection_name,
                points=[
                    PointStruct(
                        id=_uuid_for(memory.id),
                        vector=memory.embedding,
                        payload={"id": memory.id, "data": memory.model_dump_json()},
                    )
                ],
            )
            self._fallback.pop(memory.id, None)
            logger.debug("QdrantStore.save: upserted point for memory %s", memory.id)
        else:
            self._fallback[memory.id] = memory
            logger.debug(
                "QdrantStore.save: stored embedding-less memory %s in fallback", memory.id
            )

    def get(self, memory_id: str) -> Memory | None:
        if self._collection_ready:
            records = self._client.retrieve(
                self._collection_name,
                ids=[_uuid_for(memory_id)],
                with_payload=True,
                with_vectors=False,
            )
            if records:
                payload = records[0].payload or {}
                return Memory.model_validate_json(str(payload["data"]))
        return self._fallback.get(memory_id)

    def update(self, memory: Memory) -> None:
        self.save(memory)

    def delete(self, memory_id: str) -> None:
        if self._collection_ready:
            self._client.delete(
                self._collection_name,
                points_selector=PointIdsList(points=[_uuid_for(memory_id)]),
            )
        self._fallback.pop(memory_id, None)

    def list_all(self) -> list[Memory]:
        results: list[Memory] = list(self._fallback.values())
        if not self._collection_ready:
            return results
        offset = None
        while True:
            records, next_offset = self._client.scroll(
                self._collection_name,
                limit=_SCROLL_LIMIT,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                payload = rec.payload or {}
                results.append(Memory.model_validate_json(str(payload["data"])))
            if next_offset is None or len(records) < _SCROLL_LIMIT:
                break
            if len(results) >= _SCROLL_LIMIT:
                logger.debug(
                    "QdrantStore.list_all: reached scroll limit %d — truncating", _SCROLL_LIMIT
                )
                break
            offset = next_offset
        return results

    def search_by_embedding(self, embedding: list[float], top_k: int) -> list[Memory]:
        """Return top-k memories by cosine similarity.

        Memories without embeddings (held in the fallback dict) are searched
        via brute-force cosine scan and merged with Qdrant results.
        """
        qdrant_results: list[Memory] = []
        if self._collection_ready:
            response = self._client.query_points(
                self._collection_name,
                query=embedding,
                limit=top_k,
                with_payload=True,
            )
            for hit in response.points:
                payload = hit.payload or {}
                qdrant_results.append(Memory.model_validate_json(str(payload["data"])))

        fallback_results = self._brute_force_search(embedding, top_k)
        if not fallback_results:
            return qdrant_results[:top_k]

        merged: list[tuple[float, Memory]] = [
            (cosine_similarity(embedding, m.embedding), m)
            for m in qdrant_results
            if m.embedding is not None
        ]
        merged += [
            (cosine_similarity(embedding, m.embedding), m)
            for m in fallback_results
            if m.embedding is not None
        ]
        merged.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in merged[:top_k]]

    def __len__(self) -> int:
        count = len(self._fallback)
        if self._collection_ready:
            result = self._client.count(self._collection_name)
            count += result.count
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_collection_state(self) -> None:
        """Attach to an existing collection if present (server-mode resume)."""
        try:
            if self._client.collection_exists(self._collection_name):
                info = self._client.get_collection(self._collection_name)
                vectors_config = info.config.params.vectors
                if isinstance(vectors_config, VectorParams):
                    self._dim = vectors_config.size
                self._collection_ready = True
                logger.debug(
                    "QdrantStore: attached to existing collection '%s' (dim=%d)",
                    self._collection_name,
                    self._dim,
                )
        except Exception as exc:
            logger.debug("QdrantStore: could not attach to existing collection: %s", exc)

    def _ensure_collection(self, dim: int) -> None:
        """Create the Qdrant collection on first save() with an embedding."""
        if self._collection_ready:
            return
        self._client.create_collection(
            self._collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        self._dim = dim
        self._collection_ready = True
        logger.debug("QdrantStore: created collection '%s' (dim=%d)", self._collection_name, dim)
        self._flush_fallback_with_embeddings()

    def _flush_fallback_with_embeddings(self) -> None:
        """Promote any fallback memories that now have embeddings into Qdrant."""
        to_promote = [m for m in self._fallback.values() if m.embedding is not None]
        if not to_promote:
            return
        points = [
            PointStruct(
                id=_uuid_for(m.id),
                vector=m.embedding,
                payload={"id": m.id, "data": m.model_dump_json()},
            )
            for m in to_promote
        ]
        self._client.upsert(self._collection_name, points=points)
        for m in to_promote:
            del self._fallback[m.id]
        logger.debug("QdrantStore: promoted %d fallback memories to Qdrant", len(to_promote))

    def _brute_force_search(self, embedding: list[float], top_k: int) -> list[Memory]:
        scored = [
            (cosine_similarity(embedding, m.embedding), m)
            for m in self._fallback.values()
            if m.embedding is not None
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def close(self) -> None:
        """Close the underlying Qdrant client connection."""
        self._client.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(collection={self._collection_name!r}, count={len(self)})"

    def __enter__(self) -> QdrantStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
