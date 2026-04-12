"""Dense retrieval: nearest chunks in Qdrant, collapsed back to passages (the citation unit).

A long passage may be stored as several overlapping windows; a passage's score is the score of
its best window, so it is ranked once. We fetch more chunks than passages we return
(FETCH_CHUNKS > N_PASSAGES) so that collapsing windows cannot leave the list short.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from qdrant_client import QdrantClient

from regrag import config
from regrag.corpus import PassageKey
from regrag.embed import Vector

N_PASSAGES = 20
FETCH_CHUNKS = 50


@dataclass(frozen=True)
class Hit:
    key: PassageKey
    score: float
    text: str
    doc_title: str
    headings: tuple[str, ...]


def collapse(points: Iterable, n: int = N_PASSAGES) -> list[Hit]:
    """Keep each passage's best-scoring window; return the top n passages by score."""
    best: dict[PassageKey, Hit] = {}
    for p in points:
        pl = p.payload
        key = (pl["doc_id"], pl["passage_id"])
        if key not in best or p.score > best[key].score:
            best[key] = Hit(key, p.score, pl["text"], pl["doc_title"], tuple(pl["headings"]))
    return sorted(best.values(), key=lambda h: h.score, reverse=True)[:n]


def dense_search(
    client: QdrantClient,
    query_vector: Vector,
    n: int = N_PASSAGES,
    fetch: int = FETCH_CHUNKS,
    collection: str = config.COLLECTION,
) -> list[Hit]:
    res = client.query_points(collection, query=query_vector, limit=fetch, with_payload=True)
    return collapse(res.points, n)
