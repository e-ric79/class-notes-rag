"""Test the vector database layer against real Qdrant (local mode) with a fake embedder.
No API key and no model download needed:  python tests/test_store.py
"""
import os
import sys
import tempfile
import types
import zlib

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

# Stub only what we don't need to test here (the real embedding model and dotenv).
for name in ["sentence_transformers", "dotenv"]:
    try:
        __import__(name)
    except ImportError:
        m = types.ModuleType(name)
        m.SentenceTransformer = object
        m.load_dotenv = lambda *a, **k: None
        sys.modules[name] = m

import src.index as index_mod
from src.index import Index, IndexMissing, sync_index, tokenize
from src.retrieve import retrieve
from src.store import VectorStore, chunk_id

DIM = 64


def embed_text(text: str) -> np.ndarray:
    """Deterministic bag-of-words vector: texts sharing words get similar vectors."""
    v = np.zeros(DIM)
    for tok in tokenize(text):
        v[zlib.crc32(tok.encode()) % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class Embedder:
    def __init__(self):
        self.seen: list[str] = []     # every text we were asked to embed

    def __call__(self, texts):
        self.seen.extend(texts)
        return np.array([embed_text(t) for t in texts])


class FakeModel:
    """Stands in for SentenceTransformer in Index (query encoding)."""
    def __init__(self, *a, **k):
        pass

    def encode(self, texts, normalize_embeddings=True, **k):
        return np.array([embed_text(t) for t in texts])


def chunk(course, source, loc, text):
    return {"course": course, "source": source, "location": loc, "text": text}


A = chunk("ml", "w4.pdf", "p. 1", "backpropagation applies the chain rule to compute gradients")
B = chunk("ml", "w5.pdf", "p. 2", "stochastic gradient descent updates weights using mini batches")
C = chunk("networks", "tcp.pdf", "p. 3", "tcp uses a three way handshake to open a connection")

tmp = tempfile.mkdtemp()
store = VectorStore(path=tmp, url=None, collection="t")
emb = Embedder()

# 1. first sync embeds everything
stats = sync_index(store, [A, B, C], emb)
assert stats == {"added": 3, "removed": 0, "unchanged": 0}, stats
assert store.count() == 3 and len(emb.seen) == 3
print("1 OK  initial sync")

# 2. unchanged re-sync does no embedding at all
emb.seen.clear()
stats = sync_index(store, [A, B, C], emb)
assert stats == {"added": 0, "removed": 0, "unchanged": 3}, stats
assert emb.seen == [], "unchanged chunks must not be re-embedded"
print("2 OK  re-sync is a no-op")

# 3. edit one chunk: only that one is embedded, the old version is removed
B2 = chunk("ml", "w5.pdf", "p. 2", "sgd updates weights per mini batch and needs a learning rate")
emb.seen.clear()
stats = sync_index(store, [A, B2, C], emb)
assert stats == {"added": 1, "removed": 1, "unchanged": 2}, stats
assert emb.seen == [B2["text"]]
assert store.count() == 3
print("3 OK  edited chunk replaced, others untouched")

# 4. a deleted note removes its chunks
stats = sync_index(store, [A, B2], emb)
assert stats["removed"] == 1 and store.count() == 2
print("4 OK  removed source cleaned up")

# 5. payload round trip + deterministic ids
stored = store.all_chunks()
assert stored[chunk_id(A)] == A
assert chunk_id(A) == chunk_id(dict(A)) and chunk_id(A) != chunk_id(B2)
print("5 OK  payloads and ids")

# 6. nearest neighbour + filter applied inside the DB
sync_index(store, [A, B2, C], emb)
top = store.search(embed_text(A["text"]), 3)
assert top[0][0] == chunk_id(A), "a chunk should be its own nearest neighbour"
only_net = store.search(embed_text(A["text"]), 3, course="networks")
assert [cid for cid, _ in only_net] == [chunk_id(C)]
assert store.search(embed_text("x"), 3, course="chemistry") == []
print("6 OK  search + course filter")

# 7. --rebuild wipes and re-embeds everything
emb.seen.clear()
stats = sync_index(store, [A, B2, C], emb, rebuild=True)
assert stats["added"] == 3 and len(emb.seen) == 3
print("7 OK  rebuild")
store.close()

# 8. Index: missing index raises and RELEASES the folder lock; then works after sync
index_mod.VectorStore = lambda: VectorStore(path=tmp + "/idx", url=None, collection="t")
index_mod.SentenceTransformer = FakeModel
try:
    Index()
    raise AssertionError("expected IndexMissing")
except IndexMissing:
    pass
s2 = VectorStore(path=tmp + "/idx", url=None, collection="t")     # would raise if the lock were still held
sync_index(s2, [A, B, C], emb)
s2.close()
print("8 OK  IndexMissing releases the lock")

# 9. retrieval in all three modes, with and without a course filter
idx = Index()
for mode in ("bm25", "dense", "hybrid"):
    res = retrieve(idx, "chain rule gradients backpropagation", mode=mode, k=2)
    assert res[0] == A, (mode, res[0])
    res = retrieve(idx, "connection handshake", mode=mode, k=3, course="networks")
    assert res == [C], (mode, res)
assert retrieve(idx, "gradient", mode="dense", k=5, course="ml")[0]["course"] == "ml"
assert idx.courses == ["ml", "networks"]
print("9 OK  retrieve: bm25 / dense / hybrid + course filter")

idx.store.close()