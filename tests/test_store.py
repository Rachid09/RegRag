import pytest
from qdrant_client import QdrantClient

from regrag.chunking import Chunk
from regrag.store import ensure_collection, payload_for, point_id, upsert_chunks


def chunk(doc, pid, window=0, text="t"):
    return Chunk(f"{doc}:{pid}#{window}", (doc, pid), window, text, f"title: x | text: {text}")


@pytest.fixture
def client():
    return QdrantClient(":memory:")  # qdrant-client's local mode, no server needed


def test_point_id_is_stable_and_distinct_per_window():
    assert point_id("1:2.1#0") == point_id("1:2.1#0")
    assert point_id("1:2.1#0") != point_id("1:2.1#1")


def test_upsert_is_idempotent_and_keeps_citation_payload(client):
    ensure_collection(client, "c", dims=2)
    chunks = [chunk(7, "3.1.(a)"), chunk(7, "3.2")]
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    payloads = [payload_for(c, "Doc", ("H1", "H2")) for c in chunks]
    upsert_chunks(client, "c", chunks, vecs, payloads)
    upsert_chunks(client, "c", chunks, vecs, payloads)
    assert client.count("c", exact=True).count == 2
    hit = client.query_points("c", query=[0.9, 0.1], limit=1).points[0]
    assert (hit.payload["doc_id"], hit.payload["passage_id"]) == (7, "3.1.(a)")
    assert hit.payload["headings"] == ["H1", "H2"]


def test_existing_collection_with_other_dims_is_rejected(client):
    ensure_collection(client, "c", dims=2)
    with pytest.raises(RuntimeError, match="dims 2"):
        ensure_collection(client, "c", dims=3)
    ensure_collection(client, "c", dims=3, recreate=True)


def test_length_mismatch_raises(client):
    ensure_collection(client, "c", dims=2)
    with pytest.raises(ValueError):
        upsert_chunks(client, "c", [chunk(1, "1")], [], [])
