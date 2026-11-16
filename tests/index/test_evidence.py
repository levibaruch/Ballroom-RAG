"""Index tests (no model; need the built index): evidence ids, CheckCites, claim splitting, rendering with sources.
Each test pins down a bug seen in a real run."""
from dance_rag_agentic.retrieval import books
from dance_rag_agentic.agent.evidence import citation_problems, render_with_sources, split_claims
from .shared import (MUSIC, REF_HEADER, REF_24, REF_44, REF_TEMPO, SIG_HEADER, SIG_24, SIG_44, SIG_TEMPO,
                     FAN_LADY_SIDE_1, FAN_LADY_PARTS, HOCKEY_STICK_MAN_SIDE_2, SPOT_TURN_MAN_SIDE_1,
                     SPOT_TURN_MAN_SIDE_2, SLOW_WALTZ_ROTATION, evidence_of)


def test_ranges_are_not_allowed():
    # ranges kept causing trouble (header-to-row ranges, '[1-3]' in the output): ids are listed one by one
    assert citation_problems(f"Rumba is 4/4 [{SIG_HEADER}-{SIG_44}].", evidence_of(MUSIC)) != []
    assert citation_problems(f"Rumba is 4/4 [{SIG_HEADER}, {SIG_44}].", evidence_of(MUSIC)) == []


def test_footnotes_are_listed_not_ranged():
    rendered = render_with_sources(f"Rumba is 4/4 [{SIG_HEADER}, {SIG_24}, {SIG_44}].")
    assert "[1, 2, 3]" in rendered and "[1-3]" not in rendered


def test_citing_unread_section_is_a_problem():
    assert citation_problems(f"Rumba is 4/4 [{SIG_44}].", evidence_of()) != []


def test_not_found_without_ids_is_fine():
    assert citation_problems("Not found in the books. Searched for X.", evidence_of()) == []


def test_answer_without_ids_is_a_problem():
    assert citation_problems("Rumba is 4/4.", evidence_of(MUSIC)) != []


def test_short_fragments_are_not_claims():
    # JudgeClaims flagged 'and' and 'The relevant passages are' as unsupported
    claims = split_claims(f"Rumba is in 4/4 time [{SIG_44}] and [{SIG_24}].\nThe relevant passages are [{SIG_HEADER}].")
    assert [claim["claim"] for claim in claims] == ["Rumba is in 4/4 time"]


def test_render_replaces_ids_with_footnotes_and_book_text():
    rendered = render_with_sources(f"Rumba is 4/4 [{SIG_44}].")
    assert "[1]" in rendered and "SOURCES" in rendered and "Rumba, page 21" in rendered


def test_ids_on_their_own_line_prove_the_text_above():
    # the answer put '[1, 2, 3]' on a separate line: JudgeClaims found 0 claims and the answer went unchecked
    answer = f"Rumba is written in 4/4 with the accent on beat 4.\nIt is not a 3/4 dance.\n[{SIG_44}, {SIG_TEMPO}]"
    claims = split_claims(answer)
    assert claims == [dict(claim="Rumba is written in 4/4 with the accent on beat 4. It is not a 3/4 dance.",
                           ids=[REF_44, REF_TEMPO])]


def test_replace_claim_swaps_only_that_sentence():
    from dance_rag_agentic.agent.evidence import replace_claim
    answer = f"Rumba is 4/4 [{SIG_44}]. Latin dances only use 2/4 or 4/4 [{SIG_24}, {SIG_44}]. Tempo 25-27 [{SIG_TEMPO}]."
    new, found = replace_claim(answer, "Latin dances only use 2/4 or 4/4", f"The book lists Samba in 2/4 [{SIG_24}].")
    assert found and new == f"Rumba is 4/4 [{SIG_44}]. The book lists Samba in 2/4 [{SIG_24}]. Tempo 25-27 [{SIG_TEMPO}]."
    removed, _ = replace_claim(answer, "Latin dances only use 2/4 or 4/4", "")
    assert removed == f"Rumba is 4/4 [{SIG_44}]. Tempo 25-27 [{SIG_TEMPO}]."


def test_uncited_sentence_on_a_cited_line_is_found():
    # 'No, you cannot ...' shared a line with a cited sentence and was never judged
    from dance_rag_agentic.agent.evidence import uncited_sentences
    answer = f"No, you cannot dance Rumba on a 3/4 tempo. The books list Rumba in 4/4 [{SIG_44}]."
    assert uncited_sentences(answer) == ["No, you cannot dance Rumba on a 3/4 tempo"]


def test_line_of_only_ids_is_a_citation_problem():
    # '[ids] [ids]' at the bottom made the judge see the whole answer as one claim; a wrong unit slipped by
    answer = f"Rumba is 4/4 with the 4th beat accented.\nKeep 25-27 per minute.\n\n[{SIG_44}] [{SIG_TEMPO}]"
    assert any("line of only ids" in problem for problem in citation_problems(answer, evidence_of(MUSIC)))
    assert citation_problems(f"Rumba is 4/4 [{SIG_44}]. Tempo 25-27 [{SIG_TEMPO}].", evidence_of(MUSIC)) == []


def test_uncited_fact_sentence_is_a_citation_problem():
    # '(or change the time signature of the music to 4/4)' had no id, was never judged, and was invented
    answer = (f"No.\nRumba is 4/4 [{SIG_44}].\nHow to dance it:\n"
              "Use music in 4/4 or change the time signature of the music.\n"
              "These points show the key differences between the two dances.")      # a summary: passes
    problems = citation_problems(answer, evidence_of(MUSIC))
    assert len(problems) == 1 and "without an id" in problems[0]          # 'No.' and the heading are fine


def test_a_cited_table_row_brings_its_whole_table():
    # the judge got only '4/4 | RUMBA (4th beat accented)', without 'Time Signature | Latin' or the other rows
    from dance_rag_agentic.agent.evidence import table_unit, table_units, unit_text
    unit = table_unit(REF_44)
    assert REF_HEADER in unit                                             # the header row is part of it
    units = table_units([REF_44, REF_24])
    assert len(units) == 1                                                # two rows, one table, shown once
    text = unit_text(*next(iter(units.items())))
    assert f">> [{SIG_44}]" in text and f">> [{SIG_24}]" in text and f"   [{SIG_HEADER}]" in text


def test_sources_show_each_table_once_with_cited_rows_marked():
    rendered = render_with_sources(f"Rumba is 4/4 [{SIG_44}]. Samba is 2/4 [{SIG_24}].")
    assert rendered.count("Rumba, page 21") == 1 and "whole table" in rendered
    assert "SAMBA" in rendered and "Time Signature | Latin" in rendered


def test_a_cited_chart_step_brings_the_whole_figure_chart():
    # FAN (Lady) is split into Side 1, Side 2 and four NOTES sections; one cited step brings all of them
    from dance_rag_agentic.agent.evidence import table_unit
    unit = table_unit((FAN_LADY_SIDE_1, 1))
    assert {section_id for section_id, _ in unit} == FAN_LADY_PARTS and len(FAN_LADY_PARTS) == 6


def test_short_titles_are_not_uncited_claims():
    from dance_rag_agentic.agent.evidence import uncited_sentences
    answer = f"Cha Cha Fan - Side 1\nKey differences (supported by evidence)\nRumba is 4/4 [{SIG_44}]."
    assert uncited_sentences(answer) == []


def test_a_chart_side_is_one_citable_block():
    # step-level ids caused ranges, a Prec rule and step/block mix-ups ('Step 3' was block .4)
    blocks = books.section_blocks(HOCKEY_STICK_MAN_SIDE_2)           # Hockey Stick (Man) Side 2
    assert len(blocks) == 1 and "| Step 1" in blocks[0] and "| Step 3" in blocks[0]
    assert citation_problems(f"On step 3 the man's hip design is Half 8 to Left [{HOCKEY_STICK_MAN_SIDE_2}.1].", evidence_of(HOCKEY_STICK_MAN_SIDE_2)) == []


def test_chart_steps_cited_as_blocks_are_folded_not_rejected():
    # the writer cited Spot Turn steps as [1257.2] .. [1257.4]; a chart side is one block, so CheckCites
    # rejected it twice and the rewrites shrank the answer to its Prec. step
    from dance_rag_agentic.agent.evidence import fold_single_block_ids
    one, two = SPOT_TURN_MAN_SIDE_1, SPOT_TURN_MAN_SIDE_2                     # chart sides: one block each
    answer = (f"Prec.: swivel [{one}.1]. Step 1: walk [{one}.2]. Both sides [{one}.3, {two}.4, {one}.1]. "
              f"Prose stays [{SLOW_WALTZ_ROTATION}.3]. A range stays [{one}.1-{one}.3].")
    folded_answer, folded = fold_single_block_ids(answer)
    assert f"Step 1: walk [{one}.1]." in folded_answer
    assert f"Both sides [{one}.1, {two}.1]." in folded_answer         # duplicates removed, order kept
    assert f"[{SLOW_WALTZ_ROTATION}.3]" in folded_answer and f"[{one}.1-{one}.3]" in folded_answer
    assert folded == [f"{one}.2 -> {one}.1", f"{one}.3 -> {one}.1", f"{two}.4 -> {two}.1"]


def test_debris_lines_are_dropped():
    from dance_rag_agentic.agent.evidence import drop_debris_lines
    answer = f"Answer: [{SIG_44}]\nRumba is 4/4 [{SIG_44}].\n- \n* ;\n[{SIG_24}]\n**What it is**"
    assert drop_debris_lines(answer) == f"Rumba is 4/4 [{SIG_44}].\n**What it is**"
