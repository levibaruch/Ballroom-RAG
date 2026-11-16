"""Index tests (no model; need the built index): JudgeClaims: claim verdicts and batching.
Each test pins down a bug seen in a real run."""
from .shared import SIG_44, FAN_LADY_SIDE_1, CHA_FAN_LADY_SIDE_1


def test_judge_batches_claims_per_cited_table(monkeypatch):
    # one call with 21 claims over two whole figure charts crashed the decision model: batch per table instead
    from dance_rag_agentic.agent import checks
    calls = []
    def fake_choices(state, questions, labels):
        calls.append((len(state), len(questions)))
        return {name: {"choice": "SUPPORTED", "probabilities": {"SUPPORTED": 0.9}} for name in questions}
    monkeypatch.setattr(checks, "choices", fake_choices)
    answer = "\n".join([f"Step {number} of the Cha Cha Fan is like the Rumba Fan [{CHA_FAN_LADY_SIDE_1}.1][{FAN_LADY_SIDE_1}.1]"
                        for number in range(1, 7)] + [f"Rumba is 4/4 [{SIG_44}]."])
    judged = checks.judge_claims(answer)
    assert len(judged) == 7 and [claim["claim"][:6] for claim in judged][-1] == "Rumba "   # answer order kept
    assert len(calls) == 2                                     # chart-pair batch + the Music in DanceSport table batch
    assert max(characters for characters, _ in calls) < 4000   # chart SIDES, not two whole figure charts
