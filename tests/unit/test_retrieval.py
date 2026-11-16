"""Unit tests (no model, no index): read-only access to the index: sections, blocks, charts, outline, abbreviations.
Each test pins down a bug seen in a real run."""
from dance_rag_agentic.retrieval import books


def test_table_rows_get_their_header():
    # 'Rumba = 4/4' sat in a row without 'Time Signature': the writer cited ranges, the judge said unsupported
    blocks = books.with_table_headers(["Some prose.", "Time Signature | Latin", "4/4 | RUMBA", "Next prose."])
    assert blocks == ["Some prose.", "Time Signature | Latin", "[Time Signature | Latin] 4/4 | RUMBA", "Next prose."]


def test_merged_title_row_starts_a_new_table():
    blocks = books.with_table_headers(["A | B", "1 | 2", "Tempo | Tempo", "Rumba | 25-27"])
    assert blocks[2] == "Tempo | Tempo"
    assert blocks[3] == "[Tempo | Tempo] Rumba | 25-27"


def test_compound_foot_actions_are_spelled_out():
    from dance_rag_agentic.retrieval.books import spell_out_compound_codes
    glossary = {"H": "Heel", "T": "Toe", "B": "Ball", "F": "Foot", "ET": "Extreme Toe"}
    assert spell_out_compound_codes("TH", glossary) == "Toe-Heel"
    assert spell_out_compound_codes("BHB then LF", glossary) == "Ball-Heel-Ball then LF"
