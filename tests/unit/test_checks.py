"""Unit tests (no model, no index): JudgeClaims: claim verdicts and batching.
Each test pins down a bug seen in a real run."""
import pytest


@pytest.mark.parametrize("answer, verdict, is_supported", [
    ({"choice": "SUPPORTED", "probabilities": {"SUPPORTED": 0.8, "CONTRADICTED": 0.1, "NOT_IN_EVIDENCE": 0.1}}, "SUPPORTED", True),
    # a SUPPORTED that barely wins is too unsure: downgraded (the absence claim got 0.62 under yes/no)
    ({"choice": "SUPPORTED", "probabilities": {"SUPPORTED": 0.4, "CONTRADICTED": 0.3, "NOT_IN_EVIDENCE": 0.3}}, "NOT_IN_EVIDENCE", False),
    ({"choice": "NOT_IN_EVIDENCE", "probabilities": {"SUPPORTED": 0.3, "CONTRADICTED": 0.1, "NOT_IN_EVIDENCE": 0.6}}, "NOT_IN_EVIDENCE", False),
    ({"choice": "CONTRADICTED", "probabilities": {"SUPPORTED": 0.2, "CONTRADICTED": 0.7, "NOT_IN_EVIDENCE": 0.1}}, "CONTRADICTED", False),
])
def test_claim_verdicts(answer, verdict, is_supported):
    from dance_rag_agentic.agent.checks import unsupported, verdict_of
    claim = dict(claim="c", ids=[(1, 1)]) | verdict_of(answer)
    assert claim["verdict"] == verdict and (unsupported([claim]) == []) == is_supported
