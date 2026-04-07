from regrag.chunking import MAX_WORDS, OVERLAP_WORDS, chunk_passage, windows
from regrag.corpus import Passage


def passage(text, headings=("SUSPICIOUS ACTIVITY", "Internal reporting")):
    return Passage(1, "14.2.3.Guidance.10.", text, "AML Rules", headings)


def test_short_passage_is_one_chunk_with_context():
    (c,) = chunk_passage(passage("Relevant Persons should comply with EOCN guidance."))
    assert c.chunk_id == "1:14.2.3.Guidance.10.#0"
    assert c.key == (1, "14.2.3.Guidance.10.")
    assert c.text == "Relevant Persons should comply with EOCN guidance."
    assert c.embed_text == (
        "title: AML Rules | text: SUSPICIOUS ACTIVITY > Internal reporting\n"
        "Relevant Persons should comply with EOCN guidance."
    )


def test_no_headings_means_no_context_line():
    (c,) = chunk_passage(passage("Text.", headings=()))
    assert c.embed_text == "title: AML Rules | text: Text."


def test_long_passage_split_into_overlapping_windows_same_key():
    words = [f"w{i}" for i in range(MAX_WORDS * 2 + 10)]
    chunks = chunk_passage(passage(" ".join(words)))
    assert len(chunks) == 3
    assert {c.key for c in chunks} == {(1, "14.2.3.Guidance.10.")}
    assert [c.window for c in chunks] == [0, 1, 2]
    assert all(len(c.text.split()) <= MAX_WORDS for c in chunks)
    # consecutive windows overlap, and together they cover every word
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-OVERLAP_WORDS:] == second[:OVERLAP_WORDS]
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == set(words)
    assert all(c.embed_text.startswith("title: AML Rules | text: SUSPICIOUS") for c in chunks)


def test_windows_exact_fit_is_single():
    assert windows(["a"] * MAX_WORDS) == [" ".join(["a"] * MAX_WORDS)]
