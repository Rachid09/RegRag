"""Load the ObliQA regulation corpus and QA splits into typed records.

The documents are flat lists of {ID, DocumentID, PassageID, Passage}. The hierarchy only
exists inside PassageID ("1." -> "1.1" -> "1.1.1" -> "1.1.1.(a)"), and section headings are
stored as their own short passages. We rebuild each passage's heading path from its id so the
heading text can travel with the passage as context.
"""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from regrag.config import DOCS_DIR, RAW_DIR

HEADING_MAX_WORDS = 15

PassageKey = tuple[int, str]  # (DocumentID, PassageID): the citation unit and the gold label


@dataclass(frozen=True)
class Passage:
    doc_id: int
    passage_id: str
    text: str
    doc_title: str
    headings: tuple[str, ...]  # ancestor headings, outermost first

    @property
    def key(self) -> PassageKey:
        return (self.doc_id, self.passage_id)


@dataclass(frozen=True)
class QAItem:
    qid: str
    question: str
    gold: tuple[PassageKey, ...]
    group: int


# ---------- document titles ----------

_RTF_HEX = re.compile(r"\\'([0-9a-fA-F]{2})")


def _decode_rtf_escapes(s: str) -> str:
    # In the raw file the escape is \'96; inside the quoted Python literal it appears as \\'96.
    s = s.replace("\\\\'", "\\'")
    return _RTF_HEX.sub(lambda m: bytes.fromhex(m.group(1)).decode("cp1252"), s)


def parse_document_map(rtf_text: str) -> dict[int, str]:
    """Parse the authors' DocumentMap.rtf (a Python dict literal saved as RTF).

    Each id maps to several aliases. The first alias is always the source file stem
    (e.g. "FEES_VER16.181223"), so we skip it and keep the longest remaining alias.
    """
    names: dict[int, list[str]] = {}
    current: int | None = None
    for line in rtf_text.splitlines():
        m = re.match(r"\s*(\d+):\s*\[", line)
        if m:
            current = int(m.group(1))
            names[current] = []
            continue
        if current is not None:
            names[current].extend(_decode_rtf_escapes(s) for s in re.findall(r'"([^"]*)"', line))
    return {
        doc_id: max(aliases[1:] or aliases, key=len) for doc_id, aliases in names.items() if aliases
    }


def load_titles(raw_dir: Path = RAW_DIR) -> dict[int, str]:
    return parse_document_map((raw_dir / "DocumentMap.rtf").read_text(encoding="latin-1"))


# ---------- hierarchy ----------


def parent_ids(passage_id: str) -> list[str]:
    """Candidate ancestor ids, nearest first. '1.1.1.(a)' -> ['1.1.1', '1.1.1.', '1.1', ...]."""
    parts = passage_id.rstrip(".").split(".")
    out: list[str] = []
    for i in range(len(parts) - 1, 0, -1):
        base = ".".join(parts[:i])
        out += [base, base + "."]
    return out


def _is_heading(text: str) -> bool:
    return 0 < len(text.split()) <= HEADING_MAX_WORDS


def build_passages(rows: Iterable[dict], doc_title: str) -> list[Passage]:
    """Merge duplicate PassageIDs, attach heading paths, drop empty passages."""
    merged: dict[str, list[str]] = {}
    doc_id: int | None = None
    for r in rows:
        doc_id = int(r["DocumentID"])
        text = r["Passage"].strip()
        texts = merged.setdefault(r["PassageID"], [])
        if text and text not in texts:
            texts.append(text)
    if doc_id is None:
        return []
    texts_by_id = {pid: "\n".join(ts) for pid, ts in merged.items()}

    passages = []
    for pid, text in texts_by_id.items():
        if not text:
            continue
        headings = []
        for anc in parent_ids(pid):
            anc_text = texts_by_id.get(anc, "")
            if _is_heading(anc_text) and anc_text not in headings:
                headings.append(anc_text)
        passages.append(Passage(doc_id, pid, text, doc_title, tuple(reversed(headings))))
    return passages


def load_corpus(docs_dir: Path = DOCS_DIR, raw_dir: Path = RAW_DIR) -> list[Passage]:
    titles = load_titles(raw_dir)
    out: list[Passage] = []
    for path in sorted(docs_dir.glob("*.json"), key=lambda p: int(p.stem)):
        doc_id = int(path.stem)
        out += build_passages(
            json.loads(path.read_text()), titles.get(doc_id, f"Document {doc_id}")
        )
    return out


# ---------- QA ----------


def parse_qa(records: Iterable[dict]) -> list[QAItem]:
    return [
        QAItem(
            qid=r["QuestionID"],
            question=r["Question"].strip(),
            gold=tuple(
                dict.fromkeys((int(p["DocumentID"]), p["PassageID"]) for p in r["Passages"])
            ),
            group=int(r["Group"]),
        )
        for r in records
    ]


def load_split(split: str, raw_dir: Path = RAW_DIR) -> list[QAItem]:
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"unknown split {split!r}")
    return parse_qa(json.loads((raw_dir / f"ObliQA_{split}.json").read_text()))
