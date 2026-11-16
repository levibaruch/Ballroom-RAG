"""The public API (api.py) without models: the graph run is replaced by a fake that returns a finished state."""
import json
import pytest
from fastapi.testclient import TestClient
from dance_rag_agentic.apps import api
from dance_rag_agentic.agent import graph
from .helpers import block_of, section_of


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "SPEND_FILE", tmp_path / "spend.json")
    monkeypatch.setattr(api, "TRACE_DIR", tmp_path)
    api.recent_questions.clear()
    return TestClient(api.app)


def fake_run(answer_text):
    async def run(question, log_path=None, event_sink=None, **options):
        state = graph.State(question=question, route="TECHNIQUE")
        state.answer = answer_text
        if event_sink:
            event_sink({"type": "end", "node": "Answer"})
        return state
    return run


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_answer_streams_steps_then_short_sources(client, monkeypatch):
    section_id, block = block_of(section_of("Rumba", contains="Forward Walk is a step forwards"), "Cuban Motion")
    monkeypatch.setattr(api, "answer_question", fake_run(f"The walk uses Cuban Motion [{section_id}.{block}]."))
    received = events(client.post("/v1/ask", json={"question": "What is the Rumba forward walk?"}))
    assert received[0]["type"] == "step" and any(event.get("node") == "Answer" for event in received)
    answer = received[-1]
    assert answer["type"] == "answer" and "[1]" in answer["body"]
    source = answer["sources"][0]
    assert source["book"] == "Rumba" and source["page"] > 0
    assert len(source["quote"]) <= api.QUOTE_CHARACTERS + 2          # short quote only (copyright)


def test_steps_and_details_stream_in_order(client, monkeypatch):
    async def run(question, log_path=None, event_sink=None, **options):
        event_sink({"type": "node", "node": "ClassifyKind"})
        event_sink({"type": "show", "label": "decision model: kind", "lines": [],
                    "data": {"kind": "probabilities", "marked": "TERM", "items": [["TERM", 0.8], ["FIGURE", 0.2]]}})
        event_sink({"type": "show", "label": "judge input, batch 1", "lines": ["[1.1] secret book text"]})
        event_sink({"type": "end", "node": "ClassifyKind", "message": "route TERM", "seconds": 0.4})
        return graph.State(question=question, route="TERM", answer="Not found in the books.")
    monkeypatch.setattr(api, "answer_question", run)
    received = events(client.post("/v1/ask", json={"question": "What is DW?"}))
    kinds = [event["type"] for event in received]
    assert kinds[:4] == ["step", "node", "detail", "step"] and kinds[-1] == "answer"
    assert "secret book text" not in json.dumps(received)
