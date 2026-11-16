"""Unit tests (no model, no index): the agents' tools (search, read_section, outline, read_charts).
Each test pins down a bug seen in a real run."""


def test_agents_get_one_search_tool():
    from dance_rag_agentic.agent.tools import toolbox
    names = {tool.name for tool in toolbox()}
    assert "search" in names and not names & {"keyword_search", "semantic_search", "index_lookup"}
