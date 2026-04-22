import pytest

from regrag.fusion import RRF_K, rrf
from regrag.retrieve import Hit


def hits(*ids):
    return [Hit((1, i), 100.0 - n, f"text {i}", "DOC", ()) for n, i in enumerate(ids)]


def test_rrf_score_is_sum_of_reciprocal_ranks():
    fused = {h.key: h.score for h in rrf([hits("a", "b"), hits("b", "c")])}
    assert fused[(1, "b")] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert fused[(1, "a")] == pytest.approx(1 / (RRF_K + 1))
    assert fused[(1, "c")] == pytest.approx(1 / (RRF_K + 2))


def test_agreement_beats_a_single_first_place():
    # "x" is 3rd in both lists; "a" and "b" are each 1st in only one list
    fused = rrf([hits("a", "p", "x"), hits("b", "q", "x")])
    assert fused[0].key == (1, "x")


def test_ties_keep_first_list_order_and_n_truncates():
    fused = rrf([hits("a", "b"), hits("c", "d")], n=3)
    assert [h.key[1] for h in fused] == ["a", "c", "b"]


def test_fused_hit_keeps_text_and_replaces_score():
    (h,) = rrf([hits("a"), []], n=1)
    assert h.text == "text a" and h.doc_title == "DOC"
    assert h.score == pytest.approx(1 / (RRF_K + 1))
