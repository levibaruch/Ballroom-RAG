"""Unit tests (no model, no index): the graph's nodes and State (no models: decision model and writers are stubbed).
Each test pins down a bug seen in a real run."""
import pytest


def test_trace_keeps_the_border_on_multi_line_items():
    from dance_rag_agentic.agent.graph import State
    state = State(question="q")
    state.show("note", ["FOUND: x\n\nMISSING: none"])
    assert all(line.startswith("│") for line in state.transcript)


def test_trace_lines_are_written_to_the_log_while_working(tmp_path):
    # the log was only written at the end: nothing on disk after a crash, no tail -f during a run
    from dance_rag_agentic.agent.graph import State
    log_path = tmp_path / "run.log"
    state = State(question="q", log_path=log_path)
    state.header("ExtractEntities")
    state.show("dances", ["Rumba"])
    assert "ExtractEntities" in log_path.read_text() and "Rumba" in log_path.read_text()


def test_agent_thinking_goes_to_the_trace_files_only(tmp_path, capsys):
    # the agents' reasoning was dropped; now it is kept in .log/.jsonl, but never printed or sent to the sink
    import json
    from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart
    from dance_rag_agentic.agent.agents import record_thinking, take_thinking
    from dance_rag_agentic.agent.graph import State
    take_thinking()
    record_thinking("answer writer", [ModelResponse(parts=[ThinkingPart(content="the chart says HT"),
                                                            TextPart(content="Heel-toe [12.3].")])])
    sent = []
    log_path = tmp_path / "run.log"
    state = State(question="q", live=True, log_path=log_path, event_sink=sent.append)
    state.header("Answer")
    state.log("Answer", "attempt 1")
    thinking = [event for event in map(json.loads, log_path.with_suffix(".jsonl").read_text().splitlines())
                if event["type"] == "thinking"]
    assert thinking == [dict(thinking[0], agent="answer writer", text="the chart says HT")]
    assert "the chart says HT" in log_path.read_text()
    assert "the chart says HT" not in capsys.readouterr().out + "".join(state.transcript)
    assert all(event["type"] != "thinking" for event in sent + state.events)
    assert take_thinking() == []


def test_agent_thinking_is_read_while_the_agent_runs():
    # the reasoning was written at the end of the node, under all the tool calls; now each thought is available
    # when the tool it chose runs, so the trace puts it right above that call
    import asyncio
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel
    from dance_rag_agentic.agent.agents import run_counted, take_thinking

    def model(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ThinkingPart(content="search the walks"), ToolCallPart("search", {})])
        return ModelResponse(parts=[ThinkingPart(content="enough read"), TextPart(content="FOUND: walks")])

    seen_by_the_tool = []
    agent = Agent(FunctionModel(model), name="TECHNIQUE agent")

    @agent.tool_plain
    def search() -> str:
        seen_by_the_tool.extend(take_thinking())
        return "results"

    take_thinking()
    asyncio.run(run_counted(agent, "q"))
    assert seen_by_the_tool == [("TECHNIQUE agent", "search the walks")]
    assert take_thinking() == [("TECHNIQUE agent", "enough read")]


def test_unsure_aspect_falls_back_to_all():
    from dance_rag_agentic.agent.graph import choose_aspect
    coin_toss = dict(choice="timing", probabilities={"timing": 0.36, "all": 0.36, "footwork": 0.09, "lead": 0.07})
    clear = dict(choice="lead", probabilities={"lead": 0.64, "all": 0.12})
    assert choose_aspect(coin_toss) == "all" and choose_aspect(clear) == "lead"


def test_retyped_chart_gets_one_style_rewrite():
    from dance_rag_agentic.agent import graph
    state = graph.State(question="How do I do samba runs?")
    state.entities, state.aspect = {"steps": []}, "all"
    state.answer = "\n".join(f"Step {number} - side step [1864.1]" for number in range(1, 10))
    assert graph.retyped_chart(state)
    state.entities = {"steps": [3]}                                       # asked for steps: a list is right
    assert graph.retyped_chart(state) is None


@pytest.mark.parametrize("reply, parts", [
    ("BOOKS: What is the timing?\nGENERAL: How do I relax?", ("What is the timing?", "How do I relax?")),
    ("**BOOKS:** NONE\n**GENERAL:** How do I relax?", ("", "How do I relax?")),
    ("BOOKS: What is the timing?\nGENERAL: none.", ("What is the timing?", "")),
    ("nonsense", ("", "")),
])
def test_split_reply_is_parsed(reply, parts):
    from dance_rag_agentic.agent.graph import _split_lines
    assert _split_lines(reply) == parts


def test_only_general_advice_has_no_empty_books_part():
    import asyncio
    from dance_rag_agentic.agent.graph import GENERAL_HEADING, Finish, State
    state = State(question="How often should I practise?", route="GENERAL")
    state.general_answer = "Practise a little every day."
    asyncio.run(Finish().run(state))
    assert state.final_answer.startswith(GENERAL_HEADING)


def test_writer_gets_the_agents_ids_not_its_prose():
    # the agent's note had ranges and an invented 'sensual character'; the writer copied both
    from dance_rag_agentic.agent.graph import agent_pointers
    note = "FOUND: Rumba is slower, sensual [44.30-44.31], [89.1]\nMISSING: nothing"
    pointers = agent_pointers(note)
    assert "44.30, 44.31, 89.1" in pointers and "sensual" not in pointers and "could not" not in pointers


def test_missing_teacher_parts_ask_for_one_rewrite():
    from dance_rag_agentic.agent.graph import State, missing_parts
    state = State(question="q", answer="Direct [1.1].\n**Tempo**\nFact [1.2].")
    assert "What it is" in missing_parts(state)
    state.answer = "Direct [1.1].\n**What it is**\nA [1.2].\n**How to do it**\nB [1.3].\n**What to watch**\nC [1.4]."
    assert missing_parts(state) is None
    state.answer = "Not found in the books. Searched: x."
    assert missing_parts(state) is None


def test_narrow_and_dutch_questions_do_not_demand_the_english_parts():
    from dance_rag_agentic.agent.graph import State, missing_parts
    state = State(question="Hoe draait de dame in de Fan?", answer="Antwoord [1.1].\n**Wat het is**\nX [1.2].")
    assert missing_parts(state) is None
    state = State(question="fan step 3 foot action", answer="Ball flat [1.1].", entities={"steps": ["3"]})
    assert missing_parts(state) is None


def test_a_term_question_with_an_aspect_still_needs_the_three_parts():
    # 'How is sway defined?' got aspect 'sway' and was treated as narrow: a two-line answer was never sent back
    from dance_rag_agentic.agent.graph import State, missing_parts
    state = State(question="How is sway defined?", route="TERM", aspect="sway", entities={"steps": []},
                  answer="Sway is an inclination of the body [1.1].")
    assert missing_parts(state) is not None
    state.route = "FIGURE"
    assert missing_parts(state) is None          # one column of a figure chart: a short answer is right
