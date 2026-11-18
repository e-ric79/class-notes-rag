"""Build and load the search index.

Two indexes over the same chunks:
  - dense: sentence embeddings stored in the Qdrant vector database (matches by meaning)
  - BM25:  keyword scoring held in memory (matches exact terms, e.g. course codes, acronyms)

BM25 is rebuilt from the chunk texts stored in Qdrant each time the app starts, which takes a
moment even for thousands of chunks. Only the (expensive) embeddings are persisted.

Re-indexing is incremental: chunks are identified by a hash of their content, so running
`python -m src.index` again embeds only new or edited chunks and removes chunks whose source
text is gone. Use `--rebuild` to wipe the collection and embed everything again.
"""
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL
from .ingest import ingest
from .store import VectorStore, chunk_id


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def sync_index(store: VectorStore, chunks: list[dict], embed, rebuild: bool = False) -> dict:
    """Make the vector store match `chunks`, embedding only what's new.

    `embed` is any function: list[str] -> array of vectors. Passing it in keeps this logic
    testable without downloading a model.
    """
    wanted = {chunk_id(c): c for c in chunks}
    if rebuild:
        store.reset()
    have = store.existing_ids()

    new_ids = [i for i in wanted if i not in have]
    stale_ids = [i for i in have if i not in wanted]

    if new_ids:
        new_chunks = [wanted[i] for i in new_ids]
        vectors = embed([c["text"] for c in new_chunks])
        store.ensure_collection(len(vectors[0]))
        store.upsert(new_chunks, vectors)
    store.delete(stale_ids)

    return {"added": len(new_ids), "removed": len(stale_ids), "unchanged": len(wanted) - len(new_ids)}


def build_index(rebuild: bool = False):
    chunks = [c.to_dict() for c in ingest()]
    if not chunks:
        # Refuse to continue: syncing an empty list would delete the whole existing index.
        raise SystemExit("No notes found. Put files in data/notes/<course>/ first.")

    model = SentenceTransformer(EMBED_MODEL)
    # normalize so a dot product equals cosine similarity
    embed = lambda texts: model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    store = VectorStore()
    try:
        stats = sync_index(store, chunks, embed, rebuild=rebuild)
        total = store.count()
    finally:
        store.close()
    print(f"Added {stats['added']}, removed {stats['removed']}, unchanged {stats['unchanged']}. "
          f"{total} chunks in the index.")


class IndexMissing(Exception):
    """Raised when no index has been built yet."""


class Index:
    """Loaded index: chunks from the vector DB, BM25 over them, and the query encoder."""

    def __init__(self):
        self.store = VectorStore()
        if not self.store.exists():
            self.store.close()     # release the folder lock so `python -m src.index` can run
            raise IndexMissing("No index yet. Add notes to data/notes/<course>/ and run: python -m src.index")
        self.by_id = self.store.all_chunks()           # {chunk id: chunk dict}
        self.ids = list(self.by_id)                    # position i <-> self.chunks[i] <-> BM25 row i
        self.chunks = [self.by_id[i] for i in self.ids]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.model = SentenceTransformer(EMBED_MODEL)

    @property
    def courses(self) -> list[str]:
        return sorted({c["course"] for c in self.chunks})


if __name__ == "__main__":
    build_index(rebuild="--rebuild" in sys.argv)