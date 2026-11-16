"""tests/unit: no model, no index, no books (runs anywhere, e.g. in CI).
tests/index: no model, but needs the built index (cache/agentic/index_v4); skipped when it is missing.
Section ids in the index tests are looked up by content (index/helpers.section_of), never typed in."""
