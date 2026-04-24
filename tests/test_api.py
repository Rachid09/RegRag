import json

import pytest
from fastapi.testclient import TestClient

from regrag.api import N_SOURCES, create_app
from regrag.retrieve import Hit


def fake_search(question):
    return [
        Hit((7, f"3.{i}"), 1.0 / i, f"passage {i} about {question}", "AML", ("Reporting",))
        for i in range(1, 9)
    ]


def parse_sse(body: str) -> list[tuple[str, object]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


@pytest.fixture
def client():
    with TestClient(create_app(search=fake_search, retriever="fake")) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "retriever": "fake",
        "answers": "retrieval-only",
    }


def test_ask_streams_sources_then_tokens_then_done(client):
    res = client.post("/ask", json={"question": "Who gets SARs?"})
    assert res.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(res.text)
    names = [e for e, _ in events]
    assert names[0] == "sources" and names[-1] == "done" and set(names[1:-1]) == {"token"}
    sources = events[0][1]
    assert len(sources) == N_SOURCES
    assert sources[0] == {
        "rank": 1,
        "doc_id": 7,
        "passage_id": "3.1",
        "doc_title": "AML",
        "headings": ["Reporting"],
        "text": "passage 1 about Who gets SARs?",
    }


def test_too_short_question_is_rejected(client):
    assert client.post("/ask", json={"question": "a"}).status_code == 422


def test_retrieval_failure_becomes_an_error_event_without_details():
    def broken(q):
        raise RuntimeError("qdrant down at 10.0.0.5")

    with TestClient(create_app(search=broken, retriever="fake")) as c:
        events = parse_sse(c.post("/ask", json={"question": "anything?"}).text)
    assert events == [("error", {"message": "Search failed. Please try again."})]


def test_chat_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "RegRAG" in res.text
