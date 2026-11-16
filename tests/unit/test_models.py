"""Unit tests (no model, no index): model settings, prompts, the token meter.
Each test pins down a bug seen in a real run."""
import pytest


def test_use_models_switches_at_runtime():
    from dance_rag_agentic.agent import agents
    from dance_rag_agentic.config import MODELS, use_models
    before = (MODELS.answer, MODELS.decision)
    try:
        use_models(answer="qwen3.6:35b", decision="pplx-decider-flash:4b")
        assert (MODELS.answer, MODELS.decision) == ("qwen3.6:35b", "pplx-decider-flash:4b")
        assert agents.model().model_name == "qwen3.6:35b"          # the client follows the switch
        use_models(answer=None)                                      # None keeps the current choice
        assert MODELS.answer == "qwen3.6:35b"
    finally:
        use_models(*before)


def test_role_check_reports_missing_capability(monkeypatch):
    from dance_rag_agentic.agent import models
    from dance_rag_agentic.config import MODELS, use_models
    fake = [dict(name="gpt-oss:20b", parameters="20.9B", capabilities={"completion", "tools"}),
            dict(name="autotrust-jev:latest", parameters="9B", capabilities={"decision"})]
    monkeypatch.setattr(models, "installed_models", lambda: fake)
    before = (MODELS.answer, MODELS.decision)
    try:
        use_models("gpt-oss:20b", "autotrust-jev")
        assert models.role_problems() == []                          # ':latest' is the same model
        use_models(decision="gpt-oss:20b")
        assert "lacks ['decision']" in models.role_problems()[0]
        use_models(decision="not-installed")
        assert "not installed" in models.role_problems()[0]
    finally:
        use_models(*before)


def test_no_credits_stops_without_retry(monkeypatch):
    # cloud run 2026-11-12: HTTP 402 was retried after 5 s, then the graph answered "Not found in the books"
    import requests
    from dance_rag_agentic.agent import decision as decision_model
    from dance_rag_agentic.config import CloudAccessError
    calls = []

    def refused(*args, **kwargs):
        calls.append(1)
        response = requests.Response()
        response.status_code, response._content = 402, b'{"message": "Add usage credits"}'
        return response
    monkeypatch.setattr(decision_model.requests, "post", refused)
    with pytest.raises(CloudAccessError):
        decision_model.decide("state", {"x": {"type": "noul", "instructions": "?"}})
    assert len(calls) == 1


def test_decision_model_failure_is_a_clear_error(monkeypatch):
    import requests
    from dance_rag_agentic.agent import decision as decision_model
    monkeypatch.setattr(decision_model.time, "sleep", lambda seconds: None)
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("connection dropped")
    monkeypatch.setattr(decision_model.requests, "post", refuse)
    with pytest.raises(decision_model.DecisionModelError):
        decision_model.decide("question", {})


def test_token_meter_counts_per_node_and_prices():
    from dance_rag_agentic.agent import usage
    meter = usage.TokenMeter()
    meter.add("answer", "local-model", 1000, 200)
    mark = meter.mark()
    meter.add("decision", "priced-model", 3000, 5)
    meter.add("decision", "priced-model", 1000, None)          # missing output count = 0
    old_prices = dict(usage.PRICES)
    usage.PRICES["priced-model"] = (0.24, 0.0)
    try:
        totals = meter.totals()
        assert totals[("answer", "local-model")] == {"calls": 1, "input": 1000, "output": 200, "usd": None}
        assert totals[("decision", "priced-model")]["calls"] == 2
        assert totals[("decision", "priced-model")]["input"] == 4000
        assert abs(totals[("decision", "priced-model")]["usd"] - 4000 * 0.24 / 1e6) < 1e-12
        assert "answer" not in meter.summary(since=mark)        # node footer counts only its own calls
        assert meter.report_lines()[-1].split()[0] == "total"
    finally:
        usage.PRICES.clear()
        usage.PRICES.update(old_prices)


def test_decision_usage_read_from_response():
    from dance_rag_agentic.agent import usage
    usage.METER.reset()
    usage.record_decision_usage({"answers": {}, "usage": {"input_tokens": 174, "output_tokens": 1}}, "m")
    usage.record_decision_usage({"answers": {}}, "m")           # server without usage: counted as 0, no crash
    assert usage.METER.totals()[("decision", "m")] == {"calls": 2, "input": 174, "output": 1, "usd": None}
    usage.METER.reset()
