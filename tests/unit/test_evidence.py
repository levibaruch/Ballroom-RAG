"""Unit tests (no model, no index): evidence ids, CheckCites, claim splitting, rendering with sources.
Each test pins down a bug seen in a real run."""
import pytest
from dance_rag_agentic.agent.evidence import expand_reference, normalize_ranges, parse_evidence, split_claims


@pytest.mark.parametrize("text, ids", [
    ("walk [236.2]", [(236, 2)]),
    ("steps [236.1]–[236.3]", [(236, 1), (236, 2), (236, 3)]),
    ("[236.1-3]", [(236, 1), (236, 2), (236, 3)]),
    ("[236.1, 237.4]", [(236, 1), (237, 4)]),
    ("[237.1‑237.2] non-breaking hyphen", [(237, 1), (237, 2)]),           # gpt-oss writes U+2011
    ("Timing: 4.1 [2]", []),                                               # not an evidence id
])
def test_parse_evidence(text, ids):
    references, _ = parse_evidence(text)
    assert [evidence_id for _, found in references for evidence_id in found] == ids


def test_backwards_range_is_a_problem():
    _, problems = expand_reference("236.4-236.2")
    assert problems


def test_normalize_ranges_joins_two_brackets():
    assert normalize_ranges("[1.1] – [1.3]") == "[1.1-1.3]"


def test_table_rows_become_claims_with_column_names():
    # '| 1 | 2 | [2] |' had no words, so it was never judged; a wrong Prec. timing slipped through
    answer = "| Step | Timing | Source |\n|---|---|---|\n| Prec. | - (no explicit timing given) | [70.1] |\n| 1 | 2 | [70.2] |"
    claims = split_claims(answer)
    assert [claim["claim"] for claim in claims] == ["Step: Prec., Timing: - (no explicit timing given)",
                                                    "Step: 1, Timing: 2"]
    assert claims[1]["ids"] == [(70, 2)]


def test_numbers_count_as_claim_words():
    # 'Step 1 timing: 2 [1]' was skipped (2 words), so 3 of 4 timing claims were never judged
    claims = split_claims("- Step 1 timing: 2 [70.2]\n- Step 2 timing: 3 [70.3]")
    assert [claim["claim"] for claim in claims] == ["Step 1 timing: 2", "Step 2 timing: 3"]


def test_headings_and_summaries_are_not_flagged_but_numbers_are():
    # attempt 1 of the Fan comparison was rejected over a 12-word heading and closing summaries
    from dance_rag_agentic.agent.evidence import uncited_sentences
    answer = ("Cha Cha Fan vs Rumba Fan - timing and step-by-step comparison\n"
              "These tables list every step-by-step similarity and difference in the books.\n"
              "Rumba Fan side 1 has 3 steps with timings 2, 3, 4.1.")
    assert uncited_sentences(answer) == ["Rumba Fan side 1 has 3 steps with timings 2, 3, 4.1"]


def test_lenticular_citations_are_normal_ids():
    # gpt-oss-120b wrote 【183.1】【185.1】 and every sentence failed CheckCites as "without an id"
    from dance_rag_agentic.agent.evidence import cited_ids, normalize_brackets
    answer = normalize_brackets("Level 1 is used【183.1】【185.1】. Hands move【1968.1†L1-L3】.")
    assert answer == "Level 1 is used[183.1][185.1]. Hands move[1968.1]."
    assert cited_ids(answer) == {(183, 1), (185, 1), (1968, 1)}


def test_repeated_fact_lines_are_dropped():
    # 120b answer on Latin vs Standard said "five books for Latin, five for Standard" three times
    from dance_rag_agentic.agent.evidence import drop_repeated_lines
    answer = ("**Scope**\n"
              "- The series includes ten books, five for Standard dances and five for Latin dances [1503.1].\n"
              "- Latin posture is the same in all Latin dances except Paso Doble [1127.2].\n"
              "- The series includes ten books: five for Standard dances and five for Latin dances [1905.1].")
    kept, dropped = drop_repeated_lines(answer)
    assert len(dropped) == 1 and "[1905.1]" in dropped[0]
    assert "**Scope**" in kept and "Paso Doble" in kept


def test_short_ranges_are_listed_by_code():
    # 2 of 3 answer attempts were spent on '[44.30-44.31]: no ranges'
    from dance_rag_agentic.agent.evidence import expand_ranges
    answer, expanded = expand_ranges("Tempo 30-32 [44.30-44.31]. Accent [89.1–89.2].")
    assert "[44.30, 44.31]" in answer and "[89.1, 89.2]" in answer and len(expanded) == 2
    answer, expanded = expand_ranges("Too long [44.1-44.30]. Two sections [44.1-89.2].")
    assert expanded == [] and "[44.1-44.30]" in answer


def test_empty_headings_are_dropped():
    from dance_rag_agentic.agent.evidence import drop_empty_headings
    answer = "Direct answer [1.1].\n\n**What it is**\nFact [1.2].\n\n**How to do it**\n\n**What to watch**\nTip [1.3]."
    assert "How to do it" not in drop_empty_headings(answer)
    assert "What it is" in drop_empty_headings(answer) and "What to watch" in drop_empty_headings(answer)


def test_heading_with_role_subheadings_is_kept():
    # '**How to do it**' followed by '*Lady*' / '*Man*' was dropped as empty
    from dance_rag_agentic.agent.evidence import drop_empty_headings
    answer = "**How to do it**\n\n*Lady*\n- Step [1.1].\n\n*Man*\n\n**What to watch**\n- Tip [1.2]."
    result = drop_empty_headings(answer)
    assert "How to do it" in result and "*Lady*" in result and "*Man*" not in result


def test_a_fact_repeated_as_a_pointer_in_another_part_is_kept():
    from dance_rag_agentic.agent.evidence import drop_repeated_lines
    answer = ("**What it is**\nQuickstep accents the first and third beats strongly [1959.22].\n"
              "**What to watch**\nWatch the first and third beats, they carry strong accents [1959.22].")
    assert drop_repeated_lines(answer)[1] == []


def test_stacked_footnotes_are_merged():
    from dance_rag_agentic.agent.evidence import merge_adjacent_footnotes
    assert merge_adjacent_footnotes("Fact [1, 2, 3, 4] [2] [3] [4].") == "Fact [1, 2, 3, 4]."


def test_a_near_copy_in_another_part_is_dropped_but_a_pointer_is_kept():
    from dance_rag_agentic.agent.evidence import drop_repeated_lines
    answer = ("**What it is**\nQuickstep accents the first and third beats strongly [1959.22].\n"
              "**What to watch**\nQuickstep accents the first and third beats strongly [1959.22].\n"
              "Watch the first and third beats, they carry strong accents [1959.22].")
    kept, dropped = drop_repeated_lines(answer)
    assert len(dropped) == 1 and "Watch the first" in kept


def test_absence_claims_are_citation_problems():
    from dance_rag_agentic.agent.evidence import absence_claims
    assert absence_claims("Cha Cha is fast, and the books contain no description of Viennese Waltz.")
    assert absence_claims("The sources do not describe a lead for this step.")
    assert not absence_claims("Do not lift the heel [1.2]. Not found in the books.")


def test_answer_label_is_stripped():
    from dance_rag_agentic.agent.evidence import strip_answer_label
    assert strip_answer_label("**Answer:** Rumba is 4/4 [1.1]") == "Rumba is 4/4 [1.1]"
