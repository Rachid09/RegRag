"""BM25 keyword retrieval over the same chunks as the dense index, collapsed to passages.

Runs entirely in memory with rank-bm25 (no API, no Qdrant). Each chunk is indexed as
document title + heading path + text, the same content the embedding sees, so the dense vs
BM25 comparison is fair. Tokens are lowercased; rule numbers such as `3.2.1` and list markers
such as `(a)` stay single tokens, because questions often quote them.
"""

import re
from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from regrag.chunking import Chunk
from regrag.corpus import Passage
from regrag.retrieve import FETCH_CHUNKS, N_PASSAGES, Hit, collapse

# order matters: dotted numbers first, then "(a)"-style markers, then plain words/numbers
TOKEN_RE = re.compile(r"\d+(?:\.\d+)+|\([a-z0-9]{1,4}\)|[a-z0-9]+")

_STOPWORDS_TEXT = """
    a an and are as at be by for from has have in is it its of on or that the this to was were
    will with which what who whom whose when where how do does any all such than then there
    these those under must should may can shall
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def index_text(chunk: Chunk, passage: Passage) -> str:
    return " ".join([passage.doc_title, *passage.headings, chunk.text])


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], passages: dict, k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.passages = passages
        self.bm25 = BM25Okapi(
            [tokenize(index_text(c, passages[c.key])) for c in self.chunks], k1=k1, b=b
        )

    def search(self, question: str, n: int = N_PASSAGES, fetch: int = FETCH_CHUNKS) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(question))
        top = np.argsort(-scores, kind="stable")[:fetch]
        hits = []
        for i in top:
            c = self.chunks[i]
            p = self.passages[c.key]
            hits.append(Hit(c.key, float(scores[i]), c.text, p.doc_title, p.headings))
        return collapse(hits, n)


def build_bm25() -> BM25Index:
    from regrag.chunking import chunk_corpus
    from regrag.corpus import load_corpus

    passages = {p.key: p for p in load_corpus()}
    return BM25Index(chunk_corpus(list(passages.values())), passages)
