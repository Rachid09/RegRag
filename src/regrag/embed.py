"""Hosted embeddings through LiteLLM, with a disk cache, request pacing and 429 backoff.

gemini-embedding-2 has no task_type parameter; queries and documents are distinguished by the
documented text prefixes. Documents arrive already formatted by chunking.embed_text_for;
queries are wrapped here. Each text is cached individually under (model, dims, text), so an
interrupted ingestion resumes where it stopped and reruns cost nothing.
"""

import base64
import sys
import time
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from regrag import config
from regrag.cache import DiskCache, retry_on_429

Vector = list[float]


GEMINI_QUERY_FORMAT = "task: search result | query: {q}"


def query_text(question: str) -> str:
    return GEMINI_QUERY_FORMAT.format(q=question)


def encode_vector(v: Sequence[float]) -> str:
    return base64.b64encode(array("f", v).tobytes()).decode()


def decode_vector(s: str) -> Vector:
    a = array("f")
    a.frombytes(base64.b64decode(s))
    return a.tolist()


def _litellm_embed(
    texts: list[str], model: str, dims: int, api_base: str | None = None
) -> tuple[list[Vector], int]:
    import litellm  # imported lazily: slow import, and tests inject a fake

    # Ollama models have a fixed output size and reject the `dimensions` param, so it is
    # only sent for hosted models that accept it (api_base is unset for those).
    kwargs = {"dimensions": dims} if api_base is None else {"api_base": api_base}
    resp = litellm.embedding(model=model, input=texts, **kwargs)
    vectors = [d["embedding"] for d in sorted(resp.data, key=lambda d: d["index"])]
    tokens = (resp.usage.prompt_tokens or 0) if resp.usage else 0
    return vectors, tokens


@dataclass
class Embedder:
    model: str = config.EMBED_MODEL
    dims: int = config.EMBED_DIMS
    batch_size: int = config.EMBED_BATCH
    rpm: int = config.EMBED_RPM
    api_base: str | None = None  # set for local Ollama models, e.g. http://localhost:11434
    # Models are trained with their own text prefixes (Gemini: query format above; nomic:
    # "search_document: " / "search_query: "), so documents and queries are formatted per model.
    doc_prefix: str = ""
    query_format: str = GEMINI_QUERY_FORMAT
    cache: DiskCache = field(default_factory=DiskCache)
    backend: Callable[..., tuple[list[Vector], int]] = _litellm_embed
    requests_made: int = 0
    tokens_used: int = 0
    _last_request: float = 0.0

    def _key_payload(self, text: str) -> dict:
        return {"model": self.model, "dims": self.dims, "text": text}

    def _pace(self) -> None:
        if self.rpm <= 0:
            return
        wait = 60.0 / self.rpm - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _call(self, texts: list[str]) -> list[Vector]:
        @retry_on_429()
        def go():
            self._pace()
            return self.backend(texts, self.model, self.dims, self.api_base)

        vectors, tokens = go()
        if len(vectors) != len(texts):
            raise RuntimeError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
        self.requests_made += 1
        self.tokens_used += tokens
        return vectors

    def embed(self, texts: Sequence[str], progress: bool = False) -> list[Vector]:
        from regrag.cache import cache_key

        keys = [cache_key("embed", self._key_payload(t)) for t in texts]
        out: list[Vector | None] = []
        missing: list[int] = []
        for i, k in enumerate(keys):
            hit = self.cache.get(k)
            out.append(decode_vector(hit) if hit is not None else None)
            if hit is None:
                missing.append(i)
        for start in range(0, len(missing), self.batch_size):
            idx = missing[start : start + self.batch_size]
            vectors = self._call([texts[i] for i in idx])
            for i, v in zip(idx, vectors, strict=True):
                self.cache.set(keys[i], encode_vector(v))
                out[i] = v
            if progress:
                done = min(start + self.batch_size, len(missing))
                print(
                    f"  embedded {done}/{len(missing)} new texts ({self.tokens_used:,} tokens)",
                    file=sys.stderr,
                )
        return out  # type: ignore[return-value]

    def is_cached(self, text: str) -> bool:
        from regrag.cache import cache_key

        return self.cache.get(cache_key("embed", self._key_payload(text))) is not None

    def doc_text(self, text: str) -> str:
        return self.doc_prefix + text

    def embed_documents(self, texts: Sequence[str], progress: bool = False) -> list[Vector]:
        return self.embed([self.doc_text(t) for t in texts], progress=progress)

    def embed_queries(self, questions: Sequence[str], progress: bool = False) -> list[Vector]:
        return self.embed([self.query_format.format(q=q) for q in questions], progress=progress)

    def embed_query(self, question: str) -> Vector:
        return self.embed_queries([question])[0]


def make_embedder(local_model: str | None = None) -> Embedder:
    """Default Embedder for the hosted Gemini model, or a named entry in config.LOCAL_MODELS."""
    if local_model is None:
        return Embedder()
    spec = config.LOCAL_MODELS[local_model]
    # local models run with no rate limit (own machine, no quota) and no pacing delay
    return Embedder(
        model=spec["model"],
        dims=spec["dims"],
        rpm=0,
        api_base=config.OLLAMA_URL,
        doc_prefix=spec["doc_prefix"],
        query_format=spec["query_prefix"] + "{q}",
    )


def collection_for(local_model: str | None = None) -> str:
    return config.COLLECTION if local_model is None else f"obliqa_{local_model.replace('-', '_')}"


def smoke_test(n: int = 20) -> None:
    """Embed the first n chunks and check we get n distinct vectors of the right size."""
    from regrag.chunking import chunk_corpus
    from regrag.corpus import load_corpus

    chunks = chunk_corpus(load_corpus())[:n]
    emb = Embedder()
    vecs = emb.embed_documents([c.embed_text for c in chunks])
    q = emb.embed_query("What must a firm report about suspicious transactions?")
    distinct = len({tuple(round(x, 6) for x in v[:16]) for v in vecs})
    print(
        f"model={emb.model} dims={len(vecs[0])} vectors={len(vecs)} distinct={distinct} "
        f"query_dims={len(q)} requests={emb.requests_made} tokens={emb.tokens_used}"
    )
    assert len(vecs[0]) == emb.dims and distinct == n, "embedding smoke test failed"


if __name__ == "__main__":
    smoke_test(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
