"""Build and load the search index.

Two indexes over the same chunks:
  - dense: sentence embeddings (matches by meaning, e.g. "gradient descent" ~ "optimizing weights")
  - BM25:  keyword scoring (matches exact terms, e.g. course codes, acronyms, formulas)
Embeddings are saved to disk so you only pay the encoding cost once per ingest.
"""
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL, INDEX_DIR
from .ingest import ingest


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_index():
    chunks = ingest()
    if not chunks:
        raise SystemExit("No notes found. Put files in data/notes/<course>/ first.")
    model = SentenceTransformer(EMBED_MODEL)
    # normalize so a dot product equals cosine similarity
    embeddings = model.encode([c.text for c in chunks], normalize_embeddings=True, show_progress_bar=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    (INDEX_DIR / "chunks.json").write_text(json.dumps([c.to_dict() for c in chunks]), encoding="utf-8")
    print(f"Indexed {len(chunks)} chunks")


class Index:
    """Loaded index: chunk metadata, embeddings, BM25 and the query encoder."""

    def __init__(self):
        self.chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        self.embeddings = np.load(INDEX_DIR / "embeddings.npy")
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.model = SentenceTransformer(EMBED_MODEL)

    @property
    def courses(self) -> list[str]:
        return sorted({c["course"] for c in self.chunks})


if __name__ == "__main__":
    build_index()
