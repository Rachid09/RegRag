"""Exact retrieval metrics against the gold (DocumentID, PassageID) labels. No LLM involved.

recall_at_k: share of a question's gold passages found in the top k (a question with 3 gold
passages and 2 of them retrieved scores 2/3). reciprocal_rank: 1 / rank of the first gold
passage, 0 if none is retrieved; its mean over questions is MRR.
"""

from collections.abc import Sequence

from regrag.corpus import PassageKey


def recall_at_k(ranked: Sequence[PassageKey], gold: Sequence[PassageKey], k: int) -> float:
    if not gold:
        raise ValueError("question has no gold passages")
    gold_set = set(gold)
    return len(gold_set & set(ranked[:k])) / len(gold_set)


def reciprocal_rank(ranked: Sequence[PassageKey], gold: Sequence[PassageKey]) -> float:
    gold_set = set(gold)
    for rank, key in enumerate(ranked, start=1):
        if key in gold_set:
            return 1.0 / rank
    return 0.0


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
