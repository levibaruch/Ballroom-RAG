"""Index tests need the built index (cache/agentic/index_v4, `make build`), which needs the books. Without it
this folder is skipped, so `pytest` still runs the unit tests (tests/unit) anywhere, e.g. in CI."""
from dance_rag_agentic.config import INDEX_CACHE_DIR

INDEX_BUILT = (INDEX_CACHE_DIR / "build_info.json").exists()
collect_ignore_glob = [] if INDEX_BUILT else ["*.py"]


def pytest_report_header(config):
    return f"index tests: {'run (index found)' if INDEX_BUILT else 'SKIPPED, no index at ' + str(INDEX_CACHE_DIR)}"
