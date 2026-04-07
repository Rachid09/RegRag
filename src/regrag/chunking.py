"""Turn passages into embeddable chunks: one passage = one chunk, with context.

The text sent to the embedding model is the documented gemini-embedding-2 document format,
`title: <document title> | text: <heading path>\\n<passage>`, so a bare guidance sentence
still carries its document and section. Passages longer than MAX_WORDS are split into
overlapping windows that keep the same (DocumentID, PassageID); retrieval later collapses the
windows back to one passage by taking the best-scoring window.
"""

from dataclasses import dataclass

from regrag.corpus import Passage, PassageKey

MAX_WORDS = 1500  # ~2,000 tokens; well under the 8,192-token embedding limit
OVERLAP_WORDS = 150


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc>:<passage_id>#<window>"
    key: PassageKey
    window: int
    text: str  # the window's raw passage text (what we display)
    embed_text: str  # title + headings + text (what we embed)


def windows(words: list[str], size: int = MAX_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return out


def embed_text_for(passage: Passage, text: str) -> str:
    context = " > ".join(passage.headings)
    body = f"{context}\n{text}" if context else text
    return f"title: {passage.doc_title} | text: {body}"


def chunk_passage(passage: Passage) -> list[Chunk]:
    words = passage.text.split()
    # short passages keep their original whitespace/newlines (tables, numbered lists)
    parts = [passage.text] if len(words) <= MAX_WORDS else windows(words)
    return [
        Chunk(
            chunk_id=f"{passage.doc_id}:{passage.passage_id}#{i}",
            key=passage.key,
            window=i,
            text=part,
            embed_text=embed_text_for(passage, part),
        )
        for i, part in enumerate(parts)
    ]


def chunk_corpus(passages: list[Passage]) -> list[Chunk]:
    return [c for p in passages for c in chunk_passage(p)]
