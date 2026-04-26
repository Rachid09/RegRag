"""HTTP API and chat page.

    uv run uvicorn regrag.api:create_app --factory --port 8000   # open http://localhost:8000

GET  /health  -> retriever and answer model in use
POST /ask     -> Server-Sent Events: `sources`, then `token`s, then `done` (or `error`)
GET  /        -> the chat page (static files in regrag/web)

/ask runs the LangGraph pipeline (regrag.graph). The answer is verified before it is streamed,
so a citation the model invented never reaches the page. If the model is unavailable (daily
quota, provider outage) the page still gets the retrieved passages with a short notice.
"""

import json
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from regrag import config
from regrag.graph import answer, build_graph
from regrag.llm import LLM, make_llm
from regrag.retrieve import Hit

log = logging.getLogger("regrag.api")
WEB_DIR = Path(__file__).parent / "web"
FALLBACK_NOTICE = (
    "The answer service is unavailable right now, so here are the passages most relevant to "
    "your question, each cited by document and passage number."
)

Search = Callable[[str], list[Hit]]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    model: Literal["cheap", "strong"] | None = None


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


class RateLimiter:
    """At most `limit` requests per client per `window` seconds (in memory, one process)."""

    def __init__(self, limit: int, window: float = 60.0):
        self.limit, self.window = limit, window
        self.seen: dict[str, deque] = defaultdict(deque)

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        q = self.seen[client]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


def create_app(
    search: Search | None = None,
    llms: dict[str, LLM] | None = None,
    retriever: str = config.API_RETRIEVER,
    rate_limit: int = config.API_RATE_LIMIT,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.search = search or await run_in_threadpool(make_search, retriever)
        models = llms or {name: make_llm(name) for name in config.GEN_MODELS}
        app.state.graphs = {name: build_graph(app.state.search, m) for name, m in models.items()}
        app.state.models = {name: m.model for name, m in models.items()}
        yield

    if not log.handlers:  # one JSON line per question, next to uvicorn's own logs
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False

    app = FastAPI(title="RegRAG", lifespan=lifespan)
    limiter = RateLimiter(rate_limit)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; "
            "frame-ancestors 'none'"
        )
        return response

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "retriever": retriever,
            "model": app.state.models[config.GEN_MODEL],
        }

    @app.post("/ask")
    async def ask(req: AskRequest, request: Request) -> StreamingResponse:
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(429, "Too many questions. Please wait a minute and try again.")
        name = req.model or config.GEN_MODEL
        started = time.monotonic()
        state, fallback_hits, failed = None, None, False
        try:
            state = await run_in_threadpool(answer, app.state.graphs[name], req.question)
        except Exception:
            log.exception("answer pipeline failed; falling back to retrieval only")
            try:
                fallback_hits = await run_in_threadpool(app.state.search, req.question)
            except Exception:
                log.exception("retrieval failed")
                failed = True

        def events() -> Iterator[str]:
            if failed:
                yield sse("error", {"message": "Search failed. Please try again."})
                return
            if state is None:  # model unavailable: passages only
                hits = fallback_hits[: config.CONTEXT_PASSAGES]
                yield sse("sources", [source_json(i, h) for i, h in enumerate(hits, start=1)])
                for word in FALLBACK_NOTICE.split(" "):
                    yield sse("token", word + " ")
                yield sse("done", {"refused": False, "fallback": True, "citations": []})
                return
            refused = state.get("refused", False)
            hits = [] if refused else state["hits"]
            yield sse("sources", [source_json(i, h) for i, h in enumerate(hits, start=1)])
            words = state["answer"].split(" ")
            for i, word in enumerate(words):
                yield sse("token", word if i == len(words) - 1 else word + " ")
            yield sse(
                "done",
                {
                    "refused": refused,
                    "reason": state.get("reason"),
                    "citations": state.get("citations", []),
                    "model": state.get("model"),
                },
            )

        log.info(
            json.dumps(
                {
                    "event": "ask",
                    "question_chars": len(req.question),
                    "model": name,
                    "refused": state.get("refused") if state else None,
                    "reason": state.get("reason") if state else "fallback",
                    "flags": state.get("flags") if state else None,
                    "usage": state.get("usage") if state else None,
                    "timings": state.get("timings") if state else None,
                    "total_s": round(time.monotonic() - started, 3),
                }
            )
        )
        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
