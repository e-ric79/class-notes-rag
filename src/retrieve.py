"""Retrieval: BM25, dense, or hybrid (reciprocal rank fusion).

Concept: BM25 and cosine scores live on different scales, so adding them is
meaningless. Reciprocal rank fusion sidesteps that by combining *ranks*:
    score(chunk) = sum over rankers of 1 / (RRF_K + rank)
A chunk ranked high by both rankers wins; one ranked high by only one still surfaces.
"""
import numpy as np

from .config import RRF_K, TOP_K
from .index import Index, tokenize


def _allowed(index: Index, course: str | None) -> np.ndarray:
    """Boolean mask of chunks that pass the course filter."""
    if not course:
        return np.ones(len(index.chunks), dtype=bool)
    return np.array([c["course"] == course for c in index.chunks])


def _rank(scores: np.ndarray, mask: np.ndarray, n: int) -> list[int]:
    scores = np.where(mask, scores, -np.inf)
    order = np.argsort(-scores)[:n]
    return [int(i) for i in order if np.isfinite(scores[i])]


def bm25_ranking(index: Index, query: str, mask: np.ndarray, n: int) -> list[int]:
    return _rank(np.array(index.bm25.get_scores(tokenize(query))), mask, n)


def dense_ranking(index: Index, query: str, mask: np.ndarray, n: int) -> list[int]:
    q = index.model.encode([query], normalize_embeddings=True)[0]
    return _rank(index.embeddings @ q, mask, n)


def retrieve(index: Index, query: str, mode: str = "hybrid", k: int = TOP_K, course: str | None = None) -> list[dict]:
    mask = _allowed(index, course)
    pool = max(k * 4, 20)   # look deeper than k before fusing

    if mode == "bm25":
        ids = bm25_ranking(index, query, mask, k)
    elif mode == "dense":
        ids = dense_ranking(index, query, mask, k)
    else:
        fused: dict[int, float] = {}
        for ranking in (bm25_ranking(index, query, mask, pool), dense_ranking(index, query, mask, pool)):
            for rank, i in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
        ids = sorted(fused, key=fused.get, reverse=True)[:k]

    return [index.chunks[i] for i in ids]
