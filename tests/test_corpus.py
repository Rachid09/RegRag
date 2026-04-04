from regrag.corpus import build_passages, parent_ids, parse_document_map, parse_qa


def row(pid, text, doc=7):
    return {"ID": "x", "DocumentID": doc, "PassageID": pid, "Passage": text}


def test_parent_ids_nearest_first():
    assert parent_ids("1.1.1.(a)") == ["1.1.1", "1.1.1.", "1.1", "1.1.", "1", "1."]
    assert parent_ids("14.2.3.Guidance.10.")[:2] == ["14.2.3.Guidance", "14.2.3.Guidance."]
    assert parent_ids("100)") == []


def test_headings_attached_outermost_first():
    rows = [
        row("1.", "INTRODUCTION"),
        row("1.1", "Jurisdiction"),
        row("1.1.1", ""),
        row("1.1.1.(a)", "A firm must keep records for six years."),
    ]
    ps = {p.passage_id: p for p in build_passages(rows, "General Rulebook")}
    assert ps["1.1.1.(a)"].headings == ("INTRODUCTION", "Jurisdiction")
    assert ps["1.1.1.(a)"].doc_title == "General Rulebook"
    assert "1.1.1" not in ps  # empty passages are dropped


def test_long_ancestor_is_not_a_heading():
    long_rule = " ".join(["word"] * 40)
    rows = [row("3.2", long_rule), row("3.2.Guidance.1.", "Firms should consider X.")]
    (_, guidance) = build_passages(rows, "T")
    assert guidance.headings == ()


def test_duplicate_passage_ids_are_merged_in_order():
    rows = [
        row("5.2.13", ""),
        row("5.2.13", "(1) First."),
        row("5.2.13", "(2) Second."),
        row("5.2.13", "(1) First."),
    ]
    (p,) = build_passages(rows, "T")
    assert p.text == "(1) First.\n(2) Second."
    assert p.key == (7, "5.2.13")


def test_parse_document_map_skips_file_stem_decodes_cp1252_picks_longest():
    rtf = (
        "{\\rtf1 DOCUMENT_ID_MAP = \\{\\\n"
        "    12: [\\\n"
        '        "PIN_VER05.181223 with a very long file stem name",\\\n'
        '        "PIN",\\\n'
        '        "Prudential \\\\\'96 Insurance Business (PIN)"\\\n'
        "    ],\\\n"
        "    1: [\\\n"
        '        "AML"\\\n'
        "    ]\\\n"
    )
    titles = parse_document_map(rtf)
    assert titles == {12: "Prudential – Insurance Business (PIN)", 1: "AML"}


def test_parse_qa_dedupes_gold_keeps_order():
    rec = {
        "QuestionID": "q1",
        "Question": "  What must a firm do?  ",
        "Group": 3,
        "Passages": [
            {"DocumentID": 2, "PassageID": "1.1", "Passage": "a"},
            {"DocumentID": 1, "PassageID": "9", "Passage": "b"},
            {"DocumentID": 2, "PassageID": "1.1", "Passage": "a"},
        ],
    }
    (q,) = parse_qa([rec])
    assert q.question == "What must a firm do?"
    assert q.gold == ((2, "1.1"), (1, "9"))
