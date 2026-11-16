"""The public API (api.py) without models: the graph run is replaced by a fake that returns a finished state."""
import pytest
from fastapi.testclient import TestClient
from dance_rag_agentic.apps import api
from dance_rag_agentic.agent import graph


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


def test_rate_limit_per_visitor(client, monkeypatch):
    monkeypatch.setattr(api, "answer_question", fake_run("Not found in the books."))
    monkeypatch.setattr(api, "QUESTIONS_PER_HOUR", 2)
    codes = [client.post("/v1/ask", json={"question": f"q{number}"}).status_code for number in range(3)]
    assert codes == [200, 200, 429]


def test_daily_budget_stops_answering(client, monkeypatch):
    monkeypatch.setattr(api, "spent_today", lambda: 5.0)
    response = client.post("/v1/ask", json={"question": "anything"})
    assert response.status_code == 429 and "budget" in response.json()["error"]


def test_bad_input_is_refused(client):
    assert client.post("/v1/ask", json={"question": ""}).status_code == 400
    assert client.post("/v1/ask", json={"question": "x" * 1000}).status_code == 400


def test_health_and_examples(client):
    health = client.get("/v1/health").json()
    assert health["ok"] and health["index"] == "v4"
    assert client.get("/v1/examples").json()["examples"]


def test_only_allowed_origins_get_cors(client):
    allowed = client.options("/v1/ask", headers={"Origin": "https://levibaruch.dev", "Access-Control-Request-Method": "POST"})
    other = client.options("/v1/ask", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert allowed.headers.get("access-control-allow-origin") == "https://levibaruch.dev"
    assert "access-control-allow-origin" not in other.headers


def test_judge_input_and_tool_results_are_never_public():
    assert api.public_detail({"type": "show", "label": "judge input, batch 1 of 2 (3 claims, 900 characters)",
                              "lines": ["[123456.3] The Forward Walk is a step forwards ..."]}) is None
    assert api.public_detail({"type": "show", "label": "request to the agent (tool call limit 15)", "lines": ["x"]}) is None
    assert api.public_detail({"type": "show", "label": "a detail added to the graph later", "lines": ["x"]}) is None
    tool = api.public_tool_call({"type": "tool", "call": "read_section(section_id=1)", "lines": ["book text"]})
    assert tool == {"type": "tool", "call": "read_section(section_id=1)"}


def test_claims_are_public_without_their_evidence():
    detail = api.public_detail({"type": "show", "label": "claims judged by the decision model (unsupported below 0.7)",
                                "lines": ["ok p=0.9 ..."], "data": {"kind": "claims", "threshold": 0.7, "items": [
                                    {"claim": "Rumba is 4/4", "p": 0.95, "verdict": "SUPPORTED", "ids": ["1.2"],
                                     "evidence": "book text that must not pass"}]}})
    item = detail["data"]["items"][0]
    assert item == {"claim": "Rumba is 4/4", "p": 0.95, "verdict": "SUPPORTED", "ids": ["1.2"]}
