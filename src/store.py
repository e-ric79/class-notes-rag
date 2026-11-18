"""Vector database layer (Qdrant).

Concept: until now the embeddings lived in a numpy array that we rebuilt from scratch every
time. A vector database gives us four things instead:
  1. Persistence with real records: each point = vector + payload (the chunk and its metadata).
  2. Upserts and deletes by id, so re-indexing only touches what changed.
  3. Metadata filtering done inside the database ("only course = X") rather than in our code.
  4. A path to scale: the same client talks to an embedded store today and a Qdrant server or
     cloud cluster later, by changing one setting.

Qdrant runs here in "local mode": no server, no Docker, it just stores files in a folder.
Set QDRANT_URL in .env to point at a real server instead. Nothing else in the code changes.
"""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from .config import COLLECTION, QDRANT_DIR, QDRANT_URL


def chunk_id(chunk: dict) -> str:
    """Deterministic id from the chunk's content. Same text in the same place -> same id, so
    re-ingesting unchanged notes is a no-op, and edited text gets a new id automatically."""
    key = "\x1f".join([chunk["course"], chunk["source"], chunk["location"], chunk["text"]])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class VectorStore:
    def __init__(self, path=QDRANT_DIR, url=QDRANT_URL, collection=COLLECTION):
        # Local mode allows ONE process at a time to open the folder. If you see an
        # "already accessed by another instance" error, close the app / CLI that has it open.
        self.client = QdrantClient(url=url) if url else QdrantClient(path=str(path))
        self.collection = collection

    # ---- writing -----------------------------------------------------------------

    def ensure_collection(self, dim: int):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                # cosine similarity, matching the normalized embeddings we create
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def reset(self):
        """Drop the whole collection (used by --rebuild)."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

    def upsert(self, chunks: list[dict], vectors, batch: int = 128):
        for i in range(0, len(chunks), batch):
            points = [
                PointStruct(id=chunk_id(c), vector=[float(x) for x in v], payload=c)
                for c, v in zip(chunks[i:i + batch], vectors[i:i + batch])
            ]
            self.client.upsert(self.collection, points=points)

    def delete(self, ids: list[str]):
        if ids:
            self.client.delete(self.collection, points_selector=PointIdsList(points=ids))

    def close(self):
        """Release the local-mode folder lock so another process can open it."""
        self.client.close()

    # ---- reading -----------------------------------------------------------------

    def existing_ids(self) -> set[str]:
        """Ids currently stored (empty set if the collection doesn't exist yet)."""
        if not self.client.collection_exists(self.collection):
            return set()
        return set(self.all_chunks())

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection) and self.count() > 0

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def all_chunks(self) -> dict[str, dict]:
        """Every stored chunk as {id: payload}. Used to build the BM25 index and to diff on re-index.
        Vectors are not fetched, so this is cheap."""
        out, offset = {}, None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=512, offset=offset, with_payload=True, with_vectors=False
            )
            for p in points:
                out[str(p.id)] = p.payload
            if offset is None:
                return out

    def search(self, vector, k: int, course: str | None = None) -> list[tuple[str, float]]:
        """Nearest chunks to the query vector as [(id, score)], best first. The course filter is
        applied inside the database."""
        flt = Filter(must=[FieldCondition(key="course", match=MatchValue(value=course))]) if course else None
        res = self.client.query_points(
            self.collection, query=[float(x) for x in vector], limit=k, query_filter=flt, with_payload=False
        )
        return [(str(p.id), p.score) for p in res.points]