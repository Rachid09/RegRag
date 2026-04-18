from types import SimpleNamespace

import pytest

from regrag.metrics import mean, recall_at_k, reciprocal_rank
from regrag.retrieve import collapse, hit_from_point

A, B, C, D = (1, "1"), (1, "2"), (2, "1"), (3, "9")


def test_recall_at_k_counts_share_of_gold_found():
    ranked = [A, B, C, D]
    assert recall_at_k(ranked, [B, D], k=2) == 0.5
    assert recall_at_k(ranked, [B, D], k=4) == 1.0
    assert recall_at_k(ranked, [B, B], k=2) == 1.0  # duplicate gold counted once


def test_recall_requires_gold():
    with pytest.raises(ValueError):
        recall_at_k([A], [], k=1)


def test_reciprocal_rank_uses_first_gold_hit():
    assert reciprocal_rank([A, B, C], [C, B]) == 0.5
    assert reciprocal_rank([A, B], [D]) == 0.0
    assert mean([1.0, 0.5, 0.0]) == 0.5


def point(key, score, window=0):
    payload = {
        "doc_id": key[0],
        "passage_id": key[1],
        "window": window,
        "text": "t",
        "doc_title": "T",
        "headings": ["H"],
    }
    return SimpleNamespace(score=score, payload=payload)


def test_collapse_keeps_best_window_per_passage_and_caps_n():
    pts = [point(A, 0.9, 0), point(B, 0.8), point(A, 0.95, 1), point(C, 0.7)]
    hits = collapse([hit_from_point(p) for p in pts], n=2)
    assert [h.key for h in hits] == [A, B]
    assert hits[0].score == 0.95
