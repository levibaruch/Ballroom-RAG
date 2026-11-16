"""dance_rag_agentic: answers WDSF dance technique questions with a constrained agentic graph.

Code holds the control flow, a decision model makes the multiple-choice and yes/no calls, and the LLM only
searches (bounded agents) and writes the final answer from cited evidence.

    config.py, settings.toml   paths, models, thresholds and limits (settings.toml: the user-facing choices)
    ingest/                    offline: reading the books (bibr, PyMuPDF, Docling, matrix colours) and building
                               the index (build_index.py -> cache/agentic/index_v4)
    retrieval/                 read-only access to the built index: sections, blocks, charts, searches, outline
    agent/                     the graph (graph.py), its agents and tools, evidence ids, claim checks, prompts
    apps/                      the public API (api.py), the command line (cli.py), the trace viewer (web.py)
    evaluation/                gold-question evaluation (evaluate.py) and the gold-set validator

    make all          extract + Docling + build the index   (see the top-level README / Makefile)
    python -m dance_rag_agentic.apps.cli "question" --trace
    python -m dance_rag_agentic.apps.web
"""
