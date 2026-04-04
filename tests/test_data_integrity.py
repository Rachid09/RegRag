"""Checks against the real downloaded data. Skipped when data/raw is absent."""

import pytest

from regrag.config import DOCS_DIR

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not DOCS_DIR.exists(), reason="run `uv run python -m regrag.download`"),
]


@pytest.fixture(scope="module")
def corpus():
    from regrag.corpus import load_corpus

    return load_corpus()


def test_split_sizes():
    from regrag.corpus import load_split

    sizes = {s: len(load_split(s)) for s in ("train", "dev", "test")}
    assert sizes == {"train": 22295, "dev": 2788, "test": 2786}  # README says dev=2,888
    assert sum(sizes.values()) == 27869


def test_forty_documents_all_titled(corpus):
    titles = {p.doc_id: p.doc_title for p in corpus}
    assert sorted(titles) == list(range(1, 41))
    assert not any(t.startswith("Document ") for t in titles.values())


def test_passage_keys_unique(corpus):
    keys = [p.key for p in corpus]
    assert len(keys) == len(set(keys))


def test_every_gold_passage_is_in_corpus(corpus):
    from regrag.corpus import load_split

    keys = {p.key for p in corpus}
    for split in ("train", "dev", "test"):
        for q in load_split(split):
            assert 1 <= len(q.gold) <= 6
            assert set(q.gold) <= keys, (split, q.qid)
