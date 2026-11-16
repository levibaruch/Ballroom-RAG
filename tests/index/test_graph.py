"""Index tests (no model; need the built index): the graph's nodes and State (no models: decision model and writers are stubbed).
Each test pins down a bug seen in a real run."""
from dance_rag_agentic.retrieval import books
from dance_rag_agentic.agent.entities import extract_entities
from .shared import (MUSIC, REF_24, REF_44, SIG_24, SIG_44, FAN_LADY_SIDE_2, CHA_FAN_LADY_SIDE_2, classify_stub)


def test_figure_inside_a_longer_name_of_the_dance():
    # 'Samba Whisk naar links' found only 'Whisk'; Samba has 'Samba Whisk To Left/Right', not a plain Whisk
    from dance_rag_agentic.agent.graph import longer_name_in_dance
    assert "Whisk" in longer_name_in_dance("Whisk", ["Samba"])
    assert longer_name_in_dance("Whisk", ["Quickstep"]) is None          # Quickstep has a plain Whisk
    assert longer_name_in_dance("Fan", ["Jive"]) is None                  # no Jive figure contains 'Fan'


def test_finish_keeps_the_last_answer_that_passed_the_citation_check():
    # attempt 2 was fine; the judge's rewrite (attempt 3) brought a range back and became the final answer
    import asyncio
    from dance_rag_agentic.agent.graph import Finish, State
    state = State(question="Can I dance the Rumba in 3/4?")
    state.passed_answer = f"No: the books list Rumba in 4/4 [{SIG_44}]."
    state.answer = f"Rumba is 4/4 [{SIG_24}-{SIG_44}]."
    state.citation_problems = [f"[{SIG_24}-{SIG_44}]: no ranges"]
    asyncio.run(Finish().run(state))
    assert state.final_answer.startswith("No: the books list Rumba in 4/4 [1]")
    assert any("rewrite was dropped" in warning for warning in state.warnings)   # in the trace/viewer, not the text


def test_judge_sends_only_the_failing_claims_to_repair(monkeypatch):
    # targeted repair: the failing claim goes to RepairClaims, the supported one is kept and not judged again
    import asyncio
    from dance_rag_agentic.agent import graph
    monkeypatch.setattr(graph, "judge_claims", lambda answer: [
        dict(claim="Rumba is 4/4", ids=[REF_44], p_supported=0.95),
        dict(claim="Latin dances only use 2/4 or 4/4", ids=[REF_24, REF_44], p_supported=0.3)])
    monkeypatch.setattr(graph, "answers_question", lambda question, answer: 0.9)
    state = graph.State(question="Can I dance the Rumba in 3/4?", route="TECHNIQUE")
    state.answer = f"Rumba is 4/4 [{SIG_44}]. Latin dances only use 2/4 or 4/4 [{SIG_24}, {SIG_44}]."
    next_node = asyncio.run(graph.JudgeClaims().run(state))
    assert isinstance(next_node, graph.RepairClaims)
    assert [claim["claim"] for claim in state.claims_to_repair] == ["Latin dances only use 2/4 or 4/4"]
    assert graph.same_claim("Rumba is 4/4") in state.verified


def test_judge_finishes_when_all_claims_are_supported(monkeypatch):
    import asyncio
    from dance_rag_agentic.agent import graph
    monkeypatch.setattr(graph, "judge_claims", lambda answer: [dict(claim="Rumba is 4/4", ids=[REF_44], p_supported=0.9)])
    monkeypatch.setattr(graph, "answers_question", lambda question, answer: 0.9)
    state = graph.State(question="q", route="TECHNIQUE")
    state.answer = f"Rumba is 4/4 [{SIG_44}]."
    assert isinstance(asyncio.run(graph.JudgeClaims().run(state)), graph.Finish)


def test_answer_that_misses_the_question_is_repaired(monkeypatch):
    # every claim supported, but only half of 'Can I ...? If not, how do I do it?' answered
    import asyncio
    from dance_rag_agentic.agent import graph
    monkeypatch.setattr(graph, "judge_claims", lambda answer: [dict(claim="Rumba is 4/4", ids=[REF_44], p_supported=0.9)])
    monkeypatch.setattr(graph, "answers_question", lambda question, answer: 0.2)
    state = graph.State(question="Can I dance the Rumba in 3/4? If not, how do I do it?", route="TECHNIQUE")
    state.evidence.add(MUSIC, "test")
    state.answer = f"Rumba is 4/4 [{SIG_44}]."
    assert isinstance(asyncio.run(graph.JudgeClaims().run(state)), graph.FillGap)
    assert any("every part" in feedback for feedback in state.feedback)


def test_repaired_claim_replaces_its_old_verdict(monkeypatch):
    # the repaired claim scored 1.00, but Finish still warned with the first judge's verdict of the old version
    import asyncio
    from types import SimpleNamespace
    from dance_rag_agentic.agent import graph
    old = dict(claim="Side 2 is the same in both dances", ids=[(FAN_LADY_SIDE_2, 1), (CHA_FAN_LADY_SIDE_2, 1)], p_supported=0.14)
    kept = dict(claim="Rumba is 4/4", ids=[REF_44], p_supported=0.95)
    new = f"On step 3 the Cha Cha uses a Back Pelvic Tilt and the Rumba Neutral then Direct to Left [{FAN_LADY_SIDE_2}.1] [{CHA_FAN_LADY_SIDE_2}.1]."
    class FakeWriter:
        async def run(self, request, **run_options):
            return SimpleNamespace(output=f"1: {new}")
    monkeypatch.setattr(graph, "repair_writer", lambda: FakeWriter())
    monkeypatch.setattr(graph, "judge_claims", lambda text: [dict(claim=new[:new.index(' [')], ids=[(FAN_LADY_SIDE_2, 1), (CHA_FAN_LADY_SIDE_2, 1)], p_supported=1.0)])
    state = graph.State(question="q", route="FIGURE_COMPARE")
    state.answer = f"Side 2 is the same in both dances [{FAN_LADY_SIDE_2}.1] [{CHA_FAN_LADY_SIDE_2}.1]. Rumba is 4/4 [{SIG_44}]."
    state.judged_claims, state.claims_to_repair = [old, kept], [old]
    asyncio.run(graph.RepairClaims().run(state))
    assert old not in state.judged_claims and kept in state.judged_claims
    assert graph.unsupported(state.judged_claims) == [] and state.claims_repaired_count == 1


def test_term_path_reads_sections_whose_heading_is_named():
    # "hand level" is not in the book index (only "Lead - Hold Shaping", p.27); the answer is in "Hand Levels"
    # (p.29). gpt-oss-120b then answered with pressure levels instead.
    import asyncio
    from dance_rag_agentic.agent import graph
    from dance_rag_agentic.agent.entities import extract_entities
    question = "Wat betekent hand level 3 en level 0 bij de lead hold shaping?"
    state = graph.State(question=question)
    state.entities = extract_entities(question)
    asyncio.run(graph.TermPath().run(state))
    labels = [graph.books.section_label(section_id) for section_id in state.evidence.section_ids]
    assert any("Hand Levels" in label for label in labels)


def test_mixed_question_is_split(monkeypatch):
    # 'how do I do X, and how do I stop being nervous?' needs the books AND general advice
    import asyncio
    from dance_rag_agentic.agent import graph
    monkeypatch.setattr(graph, "classify_question", classify_stub("TECHNIQUE", 0.8))
    state = graph.State(question="q", entities=extract_entities("How do I keep my posture?"))
    assert isinstance(asyncio.run(graph.ClassifyKind().run(state)), graph.SplitQuestion)


def test_split_happens_only_once(monkeypatch):
    import asyncio
    from dance_rag_agentic.agent import graph
    monkeypatch.setattr(graph, "classify_question", classify_stub("TECHNIQUE", 0.8))
    state = graph.State(question="q", entities=extract_entities("How do I keep my posture?"), split_done=True)
    assert isinstance(asyncio.run(graph.ClassifyKind().run(state)), graph.TechniqueAgent)


def test_general_question_naming_a_figure_is_split(monkeypatch):
    # GENERAL wins, but a named figure has a books part
    import asyncio
    from dance_rag_agentic.agent import graph
    monkeypatch.setattr(graph, "classify_question", classify_stub("GENERAL", 0.1))
    state = graph.State(question="q", entities=extract_entities("Any tips to enjoy the Hockey Stick in Rumba?"))
    assert state.entities["figures"]
    assert isinstance(asyncio.run(graph.ClassifyKind().run(state)), graph.SplitQuestion)


def test_general_question_goes_to_general_advice(monkeypatch):
    import asyncio
    from dance_rag_agentic.agent import graph
    monkeypatch.setattr(graph, "classify_question", classify_stub("GENERAL", 0.1))
    state = graph.State(question="q", entities=extract_entities("How often should I practise?"))
    assert isinstance(asyncio.run(graph.ClassifyKind().run(state)), graph.GeneralAdvice)


def test_general_advice_comes_last_under_its_own_heading():
    import asyncio
    from dance_rag_agentic.agent.graph import GENERAL_HEADING, Finish, State
    state = State(question="Can I dance the Rumba in 3/4?")
    state.answer = f"No: the books list Rumba in 4/4 [{SIG_44}]."
    state.general_answer = "Practise a little every day."
    asyncio.run(Finish().run(state))
    books_part, general_part = state.final_answer.split(GENERAL_HEADING)
    assert "[1]" in books_part and "Practise a little" in general_part


def test_chart_terms_are_short_names_without_numbers():
    # Background agent gets the chart's own terms (positions, actions), not directions or lead sentences
    from dance_rag_agentic.agent.graph import chart_terms
    charts = [chart["section_id"] for chart in books.charts_of("Fan", "Rumba")]
    terms = chart_terms(charts)
    assert terms and "Side Walk" in terms
    assert all(not any(character.isdigit() for character in term) and len(term.split()) <= 4 for term in terms)


def test_tips_come_before_the_sources():
    # the reviewer saw no tips: they were after a long SOURCES list
    import asyncio
    from dance_rag_agentic.agent.graph import GENERAL_HEADING, Finish, State
    state = State(question="q")
    state.answer = f"Rumba is 4/4 [{SIG_44}]."
    state.general_answer = "Breathe."
    asyncio.run(Finish().run(state))
    assert state.final_answer.index(GENERAL_HEADING) < state.final_answer.index("SOURCES")


def test_polish_that_drops_a_teacher_part_is_rejected(monkeypatch):
    import asyncio
    from dance_rag_agentic.agent import graph
    class FakeResult:
        output = f"**Tempo**\nRumba is 4/4 [{SIG_44}]."
    async def fake_run(agent, request, **options):
        return FakeResult()
    monkeypatch.setattr(graph, "run_counted", fake_run)
    monkeypatch.setattr(graph, "polish_writer", lambda: None)
    state = graph.State(question="q")
    state.evidence.add(MUSIC, "t")
    state.answer = f"Direct [{SIG_44}].\n**What it is**\nRumba is 4/4 [{SIG_44}]."
    asyncio.run(graph.Polish().run(state))
    assert "What it is" in state.answer


def test_a_comparison_of_dances_asks_for_each_dance():
    from dance_rag_agentic.agent.graph import State, _agent_request
    state = State(question="q", entities=extract_entities("What is the difference between cha cha and rumba?"))
    assert "COMPARISON" in _agent_request(state)
