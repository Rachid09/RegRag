import json

import pytest
from fastapi.testclient import TestClient

from regrag.api import FALLBACK_NOTICE, create_app
from regrag.cache import DiskCache
from regrag.graph import REFUSAL
from regrag.llm import LLM
from regrag.retrieve import Hit


def fake_search(question):
    return [
        Hit((7, f"3.{i}"), 50.0 / i, f"passage {i} about reporting", "AML", ("Reporting",))
        for i in range(1, 9)
    ]


def fake_llm(tmp_path, reply="Report to the FIU [1] promptly [2]."):
    def backend(model, messages, max_tokens):
        if isinstance(reply, Exception):
            raise reply
        return reply, 10, 5

    return LLM(model="fake-model", cache=DiskCache(tmp_path / "c.sqlite"), backend=backend)


def parse_sse(body: str) -> list[tuple[str, object]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def client_for(tmp_path, reply="Report to the FIU [1] promptly [2].", search=fake_search, **kw):
    llm = fake_llm(tmp_path, reply)
    return TestClient(create_app(search=search, llms={"cheap": llm, "strong": llm}, **kw))


def answer_text(events):
    return "".join(d for e, d in events if e == "token")


def test_health_reports_retriever_and_model(tmp_path):
    with client_for(tmp_path, retriever="fake") as c:
        assert c.get("/health").json() == {
            "status": "ok",
            "retriever": "fake",
            "model": "fake-model",
        }


def test_ask_streams_sources_verified_answer_and_citations(tmp_path):
    with client_for(tmp_path) as c:
        res = c.post("/ask", json={"question": "Who gets SARs?"})
    assert res.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(res.text)
    assert events[0][0] == "sources" and len(events[0][1]) == 5
    assert events[0][1][0]["passage_id"] == "3.1"
    assert answer_text(events) == "Report to the FIU [1] promptly [2]."
    done = events[-1]
    assert done[0] == "done" and done[1]["refused"] is False
    assert [(x["n"], x["passage_id"]) for x in done[1]["citations"]] == [(1, "3.1"), (2, "3.2")]


def test_refusal_hides_sources(tmp_path):
    with client_for(tmp_path, reply="INSUFFICIENT") as c:
        events = parse_sse(c.post("/ask", json={"question": "US SEC 10-K rules?"}).text)
    assert events[0] == ("sources", [])
    assert answer_text(events) == REFUSAL
    assert events[-1][1]["refused"] is True and events[-1][1]["reason"] == "insufficient"


def test_model_outage_falls_back_to_passages(tmp_path):
    outage = Exception("503 overloaded")
    outage.status_code = 418  # not retried
    with client_for(tmp_path, reply=outage) as c:
        events = parse_sse(c.post("/ask", json={"question": "Who gets SARs?"}).text)
    assert len(events[0][1]) == 5
    assert answer_text(events).strip() == FALLBACK_NOTICE
    assert events[-1][1]["fallback"] is True


def test_retrieval_failure_is_a_generic_error(tmp_path):
    def broken(q):
        raise RuntimeError("qdrant down at 10.0.0.5")

    with client_for(tmp_path, search=broken) as c:
        events = parse_sse(c.post("/ask", json={"question": "anything?"}).text)
    assert events == [("error", {"message": "Search failed. Please try again."})]


def test_too_short_question_and_unknown_model_are_rejected(tmp_path):
    with client_for(tmp_path) as c:
        assert c.post("/ask", json={"question": "a"}).status_code == 422
        assert c.post("/ask", json={"question": "valid?", "model": "gpt"}).status_code == 422


def test_rate_limit_per_client(tmp_path):
    with client_for(tmp_path, rate_limit=2) as c:
        codes = [c.post("/ask", json={"question": "Who gets SARs?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


@pytest.mark.parametrize("path", ["/", "/health"])
def test_security_headers(tmp_path, path):
    with client_for(tmp_path) as c:
        res = c.get(path)
    assert res.status_code == 200
    assert res.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in res.headers["content-security-policy"]
