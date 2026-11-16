"""Index tests (no model; need the built index): read-only access to the index: sections, blocks, charts, outline, abbreviations.
Each test pins down a bug seen in a real run."""
from dance_rag_agentic.retrieval import books
from .helpers import section_of
from .shared import MUSIC, CHA_FAN_MAN_SIDE_2, CHA_FAN_LADY_SIDE_1, CHA_FORWARD_WALK


def test_dance_lists_come_from_the_knowledge_base():
    by_style = books.dances_by_style()
    assert set(by_style) == {"Latin", "Standard"}
    assert "Rumba" in by_style["Latin"] and "Quickstep" in by_style["Standard"]
    assert "Rumba" in books.books_sentence()


def test_rumba_time_signature_block_is_self_contained():
    row = next(block for block in books.section_blocks(MUSIC) if "RUMBA (4th beat accented)" in block)
    assert row.startswith("[Time Signature")


def test_charts_of_marks_aspect_columns():
    charts = books.charts_of("Hockey Stick", "Rumba", "Man", "hips")
    assert charts and any(chart["has_aspect"] for chart in charts if chart["kind"] == "chart")


def test_part_2_is_found_although_its_heading_is_lost():
    # Cha Cha's '(PART 2) - CHA CHA SPECIFIC PRINCIPLES' heading is missing; the Part 2 start was labelled
    # '13) Body Muscular Actions', so the agent read one sentence and never reached Guapacha Timing
    from dance_rag_agentic.retrieval import outline
    assert outline.heading_path(CHA_FORWARD_WALK) == "Cha Cha p.35 › Part 2 - Cha Cha specific principles › Forward Walk"
    titles = [chapter.title for part in outline.book_tree("Cha Cha") if part.title.startswith("Part 2")
              for chapter in part.children]
    assert "Guapacha Timing" in titles and "Forward Walk" in titles


def test_repeated_part_2_headings_are_subheadings():
    # 'Foot Placement' / 'Timing' repeat under every walk: they belong to the walk, not next to it
    from dance_rag_agentic.retrieval import outline
    # v4: the walk is ONE unit that already holds its Foot Placement / Hip Design / Timing text (in v1 they were
    # 7 sections that the outline had to fold back into the walk)
    unit = outline.unit_of(CHA_FORWARD_WALK)
    assert unit.title == "Forward Walk" and CHA_FORWARD_WALK in unit.unit_sections()
    walk_text = books.library().pages.loc[CHA_FORWARD_WALK, "text"]
    assert "Foot Placement" in walk_text and "Hip Design" in walk_text and "Timing" in walk_text


def test_chart_values_are_spelled_out_not_doubled():
    # 'Facing DW (Diagonal to Wall)' doubled every value; Squeeze/Rotation/Translation were not expanded at all
    chart = books.section_blocks(CHA_FAN_MAN_SIDE_2)[0]                                   # Cha Cha Fan (Man) Side 2
    assert "Facing Diagonal to Wall" in chart and "(Diagonal to Wall)" not in chart
    assert "Squeeze: Left" in chart and "Translation: Right Diagonal Backward" in chart
    timing_chart = books.section_blocks(CHA_FAN_LADY_SIDE_1)[0]                            # Cha Cha Fan (Lady) Side 1
    assert "Timing: &" in timing_chart                                     # counts stay as printed


def test_timing_abc_is_labelled_per_style():
    from dance_rag_agentic.retrieval.books import spell_out_timing_abc
    assert spell_out_timing_abc("s 5.6 3", "Quickstep") == "Slow/Quick s | in numbers 5.6 | choreographic count 3"
    assert spell_out_timing_abc("1 3 Q", "Samba") == "in numbers 1 | choreographic count 3 | Slow/Quick Q"
    assert spell_out_timing_abc("1 2 2", "Samba") == "1 2 2"           # garbled extraction stays as printed
    assert spell_out_timing_abc("Q", "Jive") == "Q"


def test_tempo_table_keeps_its_title_row():
    # 'Cha Cha Cha | 30-32' lost 'Tempo (Bar per minute)': the merged title row had become a single cell
    blocks = books.section_blocks(section_of("Cha Cha", header="Music in DanceSport"))
    tempo_row = next(block for block in blocks if "Cha Cha Cha | 30-32" in block)
    assert "Tempo" in tempo_row


def test_a_heading_line_and_its_paragraph_are_one_block():
    # '1 Technical Sway' and its description were two blocks: the writer cited only the description, the judge
    # could not see the name, and the repair dropped Technical / Releasing / Cosmetic Sway from the answer
    blocks = books.section_blocks(section_of("Quickstep", header="10) Types of Sway"))
    technical = next(block for block in blocks if "Technical Sway" in block and "entire body" in block)
    assert technical.startswith("1 Technical Sway: ")
    assert not any(block.strip() == "1 Technical Sway" for block in blocks)
