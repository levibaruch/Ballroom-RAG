"""Index checks (ingest/build_index.py). They run the package in a separate process and are skipped when the index has
not been built."""


def test_v4_ids_are_stable_content_keys():
    # ids come from the content key, not from a row number: the same key always gives the same id
    from dance_rag_agentic.ingest.build_index import stable_id
    assert stable_id("Rumba|chart|Fan|Lady|side 1", set()) == stable_id("Rumba|chart|Fan|Lady|side 1", set())
    taken = set()
    first = stable_id("a", taken)
    assert stable_id("a", taken) != first                 # a clash moves on, never reuses an id
