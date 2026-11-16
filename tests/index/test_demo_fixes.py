"""Fixes from the review of the public demo's traces (VPS, 2026-11-14/15). No model needed: each test pins down
what one real question did wrong."""
import asyncio
import pytest
from dance_rag_agentic.agent import graph
from dance_rag_agentic.agent.entities import extract_entities, mentions_role_words
from .helpers import section_of


@pytest.mark.parametrize("question, role", [
    ("Follow footwork for top spin", "lady"),              # answered as 'the follow (man)'
    ("what is the correct footwork in a threestep in slowfoxtrot for the lead", "man"),
    ("What does the follow do in the Whisk?", "lady"),
    ("what is the position of the follow in respect to the lead", "both"),
    ("Which figures can follow the Fan in Rumba?", None),  # the verb, not the role
    ("How do I lead the Rumba?", None),
    ("how do i lead a flick in quickstep?", None)])
def test_lead_and_follow_nouns_are_roles(question, role):
    assert extract_entities(question)["role"] == role


def test_role_words_note_and_language_are_set_once():
    state = graph.State(question="Follow footwork for top spin")
    asyncio.run(graph.ExtractEntities().run(state))
    asyncio.run(graph.ExtractEntities().run(state))        # a split question passes here twice
    assert state.notes.count(graph.REQUEST_ROLE_WORDS) == 1
    assert state.language == "English" and state.entities["role"] == "lady"
    assert not mentions_role_words("What is the timing of the Whisk?")


def test_character_sections_are_music_principles_and_interpretation():
    sections = graph.character_sections("Rumba")
    headers = [graph.books.library().pages.loc[section_id].header for section_id in sections]
    assert any("Music in DanceSport" in header for header in headers)
    assert any("Interpretation" in header for header in headers)
    assert not any("Paso Doble" in header for header in headers)
    # 'Part 2 - Cha Cha specific principles' is only a heading (111 characters): it gave an empty claim 4 times
    assert section_of("Cha Cha", header="Part 2 - Cha Cha specific principles") not in graph.character_sections("Cha Cha")


def test_shared_music_chapter_is_read_once():
    cha_cha_music = section_of("Cha Cha", header="Music in DanceSport")
    rumba_music = section_of("Rumba", header="Music in DanceSport")
    assert graph.same_text_as(cha_cha_music) == graph.same_text_as(rumba_music)


def test_character_answer_is_not_sent_back_for_missing_the_mood(monkeypatch):
    # the books part only says what shapes the character: 'misses the question' must not start a search loop
    from .shared import REF_44, SIG_44
    monkeypatch.setattr(graph, "judge_claims", lambda answer: [dict(claim="Rumba is 4/4", ids=[REF_44], p_supported=0.95)])
    monkeypatch.setattr(graph, "answers_question", lambda question, answer: 0.1)
    state = graph.State(question="What is the vibe of Rumba?", route="CHARACTER")
    state.answer = f"Rumba is 4/4 [{SIG_44}]."
    assert isinstance(asyncio.run(graph.JudgeClaims().run(state)), graph.Finish)


def test_character_answer_gets_the_frame_and_its_own_heading():
    from .shared import MUSIC, SIG_44
    state = graph.State(question="What is the vibe of Rumba?", route="CHARACTER", language="English")
    state.evidence.add(MUSIC, "test")
    state.general_heading = graph.CHARACTER_HEADING
    state.general_answer = "Slow and sensual."
    state.answer = f"Rumba accents the 4th beat [{SIG_44}]."
    asyncio.run(graph.Finish().run(state))
    assert state.final_answer.startswith(graph.CHARACTER_FRAME["English"])
    assert f"{graph.CHARACTER_HEADING}\nSlow and sensual." in state.final_answer
