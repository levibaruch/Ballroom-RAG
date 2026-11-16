"""Index tests (no model; need the built index): the agents' tools (search, read_section, outline, read_charts).
Each test pins down a bug seen in a real run."""
from dance_rag_agentic.retrieval import books
from dance_rag_agentic.agent.evidence import EvidenceLog
from .helpers import section_of
from .shared import (MUSIC, FAN_LADY_PARTS, SPOT_TURN_MAN_SIDE_1, LEAD_HOLD_SHAPING, CHA_PART_2, CHA_FORWARD_WALK,
                     RUMBA_FORWARD_WALK)


def test_repeated_tool_call_gets_a_short_answer():
    # the agent called read_section(Music in DanceSport) three times and keyword_search('3/4') twice in one run
    from types import SimpleNamespace
    from dance_rag_agentic.agent.tools import TOOLBOX, AgentDeps
    read_section = next(tool for tool in TOOLBOX if tool.__name__ == "read_section")
    context = SimpleNamespace(deps=AgentDeps(evidence=EvidenceLog(), caller="test"))
    first = read_section(context, section_id=MUSIC)
    second = read_section(context, section_id=MUSIC)
    assert "[Rumba, page 21" in first and MUSIC in context.deps.evidence   # the asked section first, never cut
    assert second.startswith("You already made exactly this call")


def test_read_section_follows_an_unfinished_section():
    # '10) Lead Hold Shaping' ends with '...two different types of lead:'; the lead itself is in the sections after it
    from types import SimpleNamespace
    from dance_rag_agentic.agent.tools import TOOLBOX, AgentDeps
    read_section = next(tool for tool in TOOLBOX if tool.__name__ == "read_section")
    context = SimpleNamespace(deps=AgentDeps(evidence=EvidenceLog(), caller="test"))
    result = read_section(context, section_id=LEAD_HOLD_SHAPING)
    following = books.next_section(LEAD_HOLD_SHAPING)
    assert following in context.deps.evidence and "Lead through connection is based on" in result
    assert books.next_section(following) in context.deps.evidence       # 'following' also ends with ':'


def test_reads_stay_within_the_character_budget():
    # one read of the whole Lead Hold Shaping chapter returned 13,869 characters
    from types import SimpleNamespace
    from dance_rag_agentic.config import READ_UNIT_CHARACTERS
    from dance_rag_agentic.agent.tools import TOOLBOX, AgentDeps
    read_section = next(tool for tool in TOOLBOX if tool.__name__ == "read_section")
    for section_id in (LEAD_HOLD_SHAPING, section_of('Rumba', header='Types of Hold'), section_of('Cha Cha', header='Part 1 - general principles')):
        context = SimpleNamespace(deps=AgentDeps(evidence=EvidenceLog(), caller="test"))
        text_part = read_section(context, section_id=section_id).split("(not read yet")[0].split("(chapter '")[0]
        assert len(text_part) <= READ_UNIT_CHARACTERS * 1.2


def test_reading_a_part_start_lists_its_chapters():
    from types import SimpleNamespace
    from dance_rag_agentic.agent.tools import TOOLBOX, AgentDeps
    read_section = next(tool for tool in TOOLBOX if tool.__name__ == "read_section")
    context = SimpleNamespace(deps=AgentDeps(evidence=EvidenceLog(), caller="test"))
    result = read_section(context, section_id=CHA_PART_2)
    assert f"section_id={section_of('Cha Cha', header='Guapacha Timing')}" in result and "Guapacha Timing" in result


def test_every_book_has_an_outline_and_search_hits_show_their_path():
    from dance_rag_agentic.retrieval import outline
    from dance_rag_agentic.agent.tools import _short_path
    for dance in books.library().dances:
        assert outline.outline_lines(dance)
    assert _short_path(section_of("Cha Cha", header="Guapacha Timing"), False).startswith("Part 2 (Cha Cha only)")


def test_read_section_on_a_chart_part_reads_the_whole_chart():
    from types import SimpleNamespace
    from dance_rag_agentic.agent.tools import TOOLBOX, AgentDeps
    read_section = next(tool for tool in TOOLBOX if tool.__name__ == "read_section")
    context = SimpleNamespace(deps=AgentDeps(evidence=EvidenceLog(), caller="test", role="Lady"))
    a_note = next(part for part in sorted(FAN_LADY_PARTS) if books.library().pages.loc[part, "kind"] == "prose")
    read_section(context, section_id=a_note)                               # one of the Fan (Lady) NOTES
    assert set(context.deps.evidence.section_ids) == FAN_LADY_PARTS


def test_agent_reads_both_roles_when_the_question_names_none():
    from types import SimpleNamespace
    from dance_rag_agentic.retrieval import books
    from dance_rag_agentic.agent.tools import TOOLBOX, AgentDeps
    read_section = next(tool for tool in TOOLBOX if tool.__name__ == "read_section")
    roles_read = {}
    for role in (None, "Man"):
        context = SimpleNamespace(deps=AgentDeps(evidence=EvidenceLog(), caller="test", role=role))
        read_section(context, section_id=SPOT_TURN_MAN_SIDE_1)        # Spot Turn To Left (Man) - Side 1
        roles_read[role] = {books.library().pages.loc[section_id, "role"] for section_id in context.deps.evidence.section_ids}
    assert roles_read[None] == {"Man", "Lady"}
    assert roles_read["Man"] == {"Man"}


def test_same_text_in_another_book_is_not_read_twice():
    # part 1 is the same in every Latin book: 'Music in DanceSport' was read from two books
    from types import SimpleNamespace
    from dance_rag_agentic.agent.tools import TOOLBOX, AgentDeps
    read_section = next(tool for tool in TOOLBOX if tool.__name__ == "read_section")
    context = SimpleNamespace(deps=AgentDeps(evidence=EvidenceLog(), caller="test"))
    read_section(context, section_id=MUSIC)                 # Rumba p.21 Music in DanceSport
    assert "same text as section" in read_section(context, section_id=section_of("Cha Cha", header="Music in DanceSport"))   # Cha Cha p.21, same chapter


def test_part_2_of_another_dance_is_read_even_when_nearly_the_same():
    # Rumba's Forward Walk is ~90% Cha Cha's: refusing it made the answer say 'no Rumba description'
    from types import SimpleNamespace
    from dance_rag_agentic.agent.tools import TOOLBOX, AgentDeps
    read_section = next(tool for tool in TOOLBOX if tool.__name__ == "read_section")
    context = SimpleNamespace(deps=AgentDeps(evidence=EvidenceLog(), caller="test"))
    read_section(context, section_id=CHA_FORWARD_WALK)
    read_section(context, section_id=RUMBA_FORWARD_WALK)
    assert RUMBA_FORWARD_WALK in context.deps.evidence


def test_part_1_of_another_style_is_not_blocked_as_a_duplicate():
    # Viennese Waltz's 'Music in Dancesport' (Standard tempos) was refused as 'same text as Cha Cha's' (Latin)
    from types import SimpleNamespace
    from dance_rag_agentic.agent.tools import TOOLBOX, AgentDeps
    read_section = next(tool for tool in TOOLBOX if tool.__name__ == "read_section")
    context = SimpleNamespace(deps=AgentDeps(evidence=EvidenceLog(), caller="test"))
    latin = section_of("Cha Cha", header="Music in DanceSport")
    standard = section_of("Viennese Waltz", header="Music in Dancesport")
    read_section(context, section_id=latin)
    result = read_section(context, section_id=standard)
    assert standard in context.deps.evidence and "same text as section" not in result
