"""Hybrid retrieval: Reciprocal Rank Fusion (RRF) of the dense and BM25 passage rankings.

RRF ignores the raw scores (cosine similarity and BM25 points are on different scales) and uses
only each passage's rank: score(p) = sum over retrievers of 1 / (k + rank), with k = 60 as in the
original paper (Cormack et al., 2009). A passage found by both retrievers collects two terms, so
agreement is rewarded. Each retriever contributes every passage from its 50 fetched chunks, and
the fused top 20 are returned.
"""

from collections.abc import Sequence
from dataclasses import replace

from regrag.retrieve import FETCH_CHUNKS, N_PASSAGES, Hit

RRF_K = 60
FUSE_DEPTH = FETCH_CHUNKS  # passages taken from each retriever before fusing (at most 50)


def rrf(rankings: Sequence[Sequence[Hit]], n: int = N_PASSAGES, k: int = RRF_K) -> list[Hit]:
    """Fuse ranked passage lists. Ties keep first-seen order (earlier lists win)."""
    scores: dict = {}
    first: dict = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking, start=1):
            scores[h.key] = scores.get(h.key, 0.0) + 1.0 / (k + rank)
            first.setdefault(h.key, h)
    fused = [replace(first[key], score=s) for key, s in scores.items()]
    return sorted(fused, key=lambda h: h.score, reverse=True)[:n]


def hybrid_search(client, bm25_index, question: str, query_vector, n: int = N_PASSAGES):
    from regrag.retrieve import dense_search

    dense = dense_search(client, query_vector, n=FUSE_DEPTH)
    sparse = bm25_index.search(question, n=FUSE_DEPTH)
    return rrf([dense, sparse], n=n)
