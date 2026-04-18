from regrag.bm25 import BM25Index, tokenize
from regrag.chunking import chunk_passage
from regrag.corpus import Passage


def test_tokenize_keeps_rule_numbers_and_markers_and_drops_stopwords():
    assert tokenize("Under Rule 3.2.1(a) the Firm MUST report.") == [
        "rule",
        "3.2.1",
        "(a)",
        "firm",
        "report",
    ]


def make_index():
    ps = [
        Passage(
            1, "1", "Suspicious transactions must be reported to the FIU.", "AML", ("Reporting",)
        ),
        Passage(1, "2", "Customer due diligence applies before onboarding.", "AML", ("CDD",)),
        Passage(2, "9.1", "Capital requirements under Rule 9.1 for banks.", "PRU", ()),
        Passage(3, "1", "Fees are payable annually.", "FEES", ()),
    ]
    passages = {p.key: p for p in ps}
    chunks = [c for p in ps for c in chunk_passage(p)]
    return BM25Index(chunks, passages)


def test_search_ranks_exact_term_match_first():
    hits = make_index().search("Who receives suspicious transaction reports?", n=2)
    assert hits[0].key == (1, "1")


def test_rule_number_and_title_are_searchable():
    idx = make_index()
    assert idx.search("Rule 9.1", n=1)[0].key == (2, "9.1")
    assert idx.search("FEES", n=1)[0].key == (3, "1")  # title is indexed too
