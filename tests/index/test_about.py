"""The ABOUT path: facts about the assistant come from the index, not from hand-written text."""
from dance_rag_agentic.retrieval import books
from dance_rag_agentic.agent.graph import about_facts


def test_about_facts_name_every_dance_in_the_index():
    text = " ".join(about_facts())
    for dance in books.library().dances:
        assert dance in text
    assert f"({len(books.library().dances)} dances)" in text
