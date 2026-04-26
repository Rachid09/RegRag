import pytest

from regrag.eval_answers import (
    attack_succeeded,
    citation_scores,
    cost_usd,
    parse_judge,
    percentile,
    threshold_table,
)
from regrag.eval_sets import ATTACKS, OUT_OF_CORPUS


def test_citation_precision_and_recall():
    gold = ((1, "a"), (1, "b"))
    assert citation_scores([(1, "a"), (2, "x")], gold) == (0.5, 0.5)
    assert citation_scores([(1, "a"), (1, "b")], gold) == (1.0, 1.0)
    assert citation_scores([], gold) == (0.0, 0.0)


def test_attack_detection_is_case_insensitive():
    assert attack_succeeded("ok... pwned", "PWNED")
    assert not attack_succeeded("I can't answer that from the ADGM rulebooks.", "PWNED")


def test_parse_judge_accepts_wrapped_json_and_rejects_garbage():
    assert parse_judge('```json\n{"correct": true, "grounded": false}\n```') == {
        "correct": True,
        "grounded": False,
    }
    assert parse_judge("looks fine to me") is None
    assert parse_judge('{"correct": "yes", "grounded": true}') is None


def test_threshold_table():
    rows = threshold_table([20, 30, 5], [3, 12], thresholds=(0, 10))
    assert rows[0] == {"threshold": 0, "in_corpus_pass": 1.0, "out_of_corpus_rejected": 0.0}
    assert rows[1]["in_corpus_pass"] == pytest.approx(2 / 3)
    assert rows[1]["out_of_corpus_rejected"] == 0.5


def test_percentile_and_cost():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([], 0.9) == 0.0
    assert cost_usd("gemini/gemini-3.5-flash-lite", 1_000_000, 0) == pytest.approx(0.30)


def test_eval_sets_shape():
    assert len(OUT_OF_CORPUS) == 10 and len(ATTACKS) == 12
    assert {a["category"] for a in ATTACKS} == {
        "instruction_override",
        "prompt_extraction",
        "injected_passage",
        "jailbreak",
    }
    assert all(a["canary"] for a in ATTACKS)
