"""Qdrant storage for the dense index: one point per chunk, cosine distance.

Point ids are UUID5 of the chunk id, so re-indexing overwrites the same points instead of
duplicating them. The payload carries everything needed to cite and display a hit
(DocumentID, PassageID, window, raw text, title, headings), so retrieval never has to go
back to the JSON files. BM25 and fusion live in Python (decision 11), not in Qdrant.
"""

import sys
import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from regrag import config
from regrag.chunking import Chunk
from regrag.embed import Vector

UPSERT_BATCH = 256


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"regrag:{chunk_id}"))


def payload_for(chunk: Chunk, doc_title: str, headings: Sequence[str]) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.key[0],
        "passage_id": chunk.key[1],
        "window": chunk.window,
        "text": chunk.text,
        "doc_title": doc_title,
        "headings": list(headings),
    }


def connect(url: str = config.QDRANT_URL) -> QdrantClient:
    return QdrantClient(url=url)


def ensure_collection(client: QdrantClient, name: str, dims: int, recreate: bool = False) -> None:
    """Create the collection if missing; refuse to mix vector sizes in an existing one."""
    if client.collection_exists(name):
        if recreate:
            client.delete_collection(name)
        else:
            size = client.get_collection(name).config.params.vectors.size
            if size != dims:
                raise RuntimeError(f"collection {name!r} has dims {size}, expected {dims}")
            return
    client.create_collection(name, vectors_config=VectorParams(size=dims, distance=Distance.COSINE))


def upsert_chunks(
    client: QdrantClient,
    name: str,
    chunks: Sequence[Chunk],
    vectors: Sequence[Vector],
    payloads: Sequence[dict],
    progress: bool = False,
) -> None:
    if not (len(chunks) == len(vectors) == len(payloads)):
        raise ValueError("chunks, vectors and payloads must have the same length")
    for start in range(0, len(chunks), UPSERT_BATCH):
        end = start + UPSERT_BATCH
        points = [
            PointStruct(id=point_id(c.chunk_id), vector=list(v), payload=p)
            for c, v, p in zip(
                chunks[start:end], vectors[start:end], payloads[start:end], strict=True
            )
        ]
        client.upsert(name, points=points, wait=True)
        if progress:
            print(f"  upserted {min(end, len(chunks))}/{len(chunks)}", file=sys.stderr)


def build_index(recreate: bool = False) -> None:
    """Load, chunk, embed (cache first) and upsert the whole corpus. Safe to rerun."""
    from regrag.chunking import chunk_corpus
    from regrag.corpus import load_corpus
    from regrag.embed import Embedder

    passages = {p.key: p for p in load_corpus()}
    chunks = chunk_corpus(list(passages.values()))
    emb = Embedder()
    vectors = emb.embed([c.embed_text for c in chunks], progress=True)
    payloads = [payload_for(c, passages[c.key].doc_title, passages[c.key].headings) for c in chunks]
    client = connect()
    ensure_collection(client, config.COLLECTION, emb.dims, recreate=recreate)
    upsert_chunks(client, config.COLLECTION, chunks, vectors, payloads, progress=True)
    n = client.count(config.COLLECTION, exact=True).count
    print(
        f"collection={config.COLLECTION} points={n} chunks={len(chunks)} "
        f"passages={len(passages)} new_embed_requests={emb.requests_made} "
        f"new_embed_tokens={emb.tokens_used}"
    )
    if n != len(chunks):
        raise RuntimeError(f"expected {len(chunks)} points, found {n}")


if __name__ == "__main__":
    build_index(recreate="--recreate" in sys.argv)
