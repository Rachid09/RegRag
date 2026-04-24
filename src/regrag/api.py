"""HTTP API and chat page.

    uv run uvicorn regrag.api:create_app --factory --port 8000   # open http://localhost:8000

GET  /health  -> which retriever is loaded and whether it is ready
POST /ask     -> Server-Sent Events stream: `sources`, then `token`s, then `done` (or `error`)
GET  /        -> the chat page (static files in regrag/web)

The answer pipeline (phase 4) is not built yet, so /ask streams the retrieved passages and a
short notice through the same events the real answer will use; the page will not change.
"""

import json
import logging
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from regrag import config
from regrag.retrieve import Hit

log = logging.getLogger("regrag.api")
WEB_DIR = Path(__file__).parent / "web"
N_SOURCES = 5  # passages shown under an answer
PENDING_ANSWER = (
    "Answer generation is not connected yet, so here are the passages most relevant to your "
    "question. Each one is cited by document and passage number."
)

Search = Callable[[str], list[Hit]]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def source_json(rank: int, h: Hit) -> dict:
    return {
        "rank": rank,
        "doc_id": h.key[0],
        "passage_id": h.key[1],
        "doc_title": h.doc_title,
        "headings": list(h.headings),
        "text": h.text,
    }


def make_search(name: str) -> Search:
    """Build the retriever once at startup (BM25 takes ~10 s to index the corpus)."""
    from regrag.bm25 import build_bm25

    bm25 = build_bm25()
    if name == "bm25":
        return lambda q: bm25.search(q)
    if name == "hybrid":
        from regrag.embed import collection_for, make_embedder
        from regrag.fusion import hybrid_search
        from regrag.store import connect

        emb = make_embedder(config.API_LOCAL_MODEL)
        client = connect()
        collection = collection_for(config.API_LOCAL_MODEL)
        if not client.collection_exists(collection):
            raise RuntimeError(f"collection {collection!r} not built; run regrag.store first")
        return lambda q: hybrid_search(client, bm25, q, emb.embed_query(q), collection=collection)
    raise ValueError(f"unknown retriever {name!r}")


def create_app(search: Search | None = None, retriever: str = config.API_RETRIEVER) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.search = search or await run_in_threadpool(make_search, retriever)
        yield

    app = FastAPI(title="RegRAG", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "retriever": retriever, "answers": "retrieval-only"}

    @app.post("/ask")
    async def ask(req: AskRequest) -> StreamingResponse:
        question = req.question.strip()
        try:
            hits = await run_in_threadpool(app.state.search, question)
        except Exception:
            log.exception("retrieval failed")
            hits = None

        def events() -> Iterator[str]:
            if hits is None:
                yield sse("error", {"message": "Search failed. Please try again."})
                return
            sources = [source_json(i, h) for i, h in enumerate(hits[:N_SOURCES], start=1)]
            yield sse("sources", sources)
            for word in PENDING_ANSWER.split(" "):
                yield sse("token", word + " ")
            yield sse("done", {"retriever": retriever})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
