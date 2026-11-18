"""Retrieval: BM25, dense (vector database), or hybrid (reciprocal rank fusion).

Concept: BM25 and cosine scores live on different scales, so adding them is
meaningless. Reciprocal rank fusion sidesteps that by combining *ranks*:
    score(chunk) = sum over rankers of 1 / (RRF_K + rank)
A chunk ranked high by both rankers wins; one ranked high by only one still surfaces.

Both rankers return lists of chunk ids, best first. The dense ranker asks the vector database
(which applies the course filter itself); BM25 runs in memory with the filter applied as a mask.
"""
import numpy as np

from .config import RRF_K, TOP_K
from .index import Index, tokenize


def bm25_ranking(index: Index, query: str, course: str | None, n: int) -> list[str]:
    scores = np.array(index.bm25.get_scores(tokenize(query)))
    if course:
        scores = np.where([c["course"] == course for c in index.chunks], scores, -np.inf)
    order = np.argsort(-scores)[:n]
    return [index.ids[i] for i in order if np.isfinite(scores[i])]


def dense_ranking(index: Index, query: str, course: str | None, n: int) -> list[str]:
    q = index.model.encode([query], normalize_embeddings=True)[0]
    return [cid for cid, _score in index.store.search(q, n, course)]


def retrieve(index: Index, query: str, mode: str = "hybrid", k: int = TOP_K, course: str | None = None) -> list[dict]:
    pool = max(k * 4, 20)   # look deeper than k before fusing

    if mode == "bm25":
        ids = bm25_ranking(index, query, course, k)
    elif mode == "dense":
        ids = dense_ranking(index, query, course, k)
    else:
        fused: dict[str, float] = {}
        for ranking in (bm25_ranking(index, query, course, pool), dense_ranking(index, query, course, pool)):
            for rank, cid in enumerate(ranking, start=1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        ids = sorted(fused, key=fused.get, reverse=True)[:k]

    return [index.by_id[i] for i in ids]