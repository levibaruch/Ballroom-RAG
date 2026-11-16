"""The ABOUT path: facts about the assistant come from the index, not from hand-written text."""
from dance_rag_agentic.agent.graph import AboutPath
from dance_rag_agentic.agent.prompts import KIND_DESCRIPTIONS


def test_about_is_a_kind_with_its_own_path():
    assert "ABOUT" in KIND_DESCRIPTIONS
    assert AboutPath is not None
