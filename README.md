# Dance Style RAG

![ci](https://github.com/levibaruch/Ballroom-RAG/actions/workflows/ci.yml/badge.svg)

To develop my RAG skills and possibly add more functionality, this system answers ballroom related technique questions based on WDSF sources. It does custom ingest, routing and agentic RAG using GPT-oss 120:b and decision models.

<!-- TODO: link to the live demo page and a short GIF of the "How the system answered" panel -->


## How a question is answered

[![How a question is answered: question → entities → kind → one path per kind → answer → citation check → claim judge → footnotes](docs/graph.svg)](docs/graph.svg)

Details: [docs/architecture.md](docs/architecture.md).

## How the books are read
 
Since the books are a bit of a mess and hard to read, [bibr](https://github.com/scienceverse/bibr) (developed by a wonderful colleague, [Jakub Werner](https://github.com/thesanogoeffect), in the metascience team at the TU/e) was used as an experiment to properly extract tables (which worked really well!). A combination of parsers was used to use each other's strengths.

[![Ingest: each page is sorted by type (chart, precedes/follows matrix, prose) and read by the extractor that scored best for it, then structured into sections and search entries](docs/ingest.svg)](docs/ingest.svg)

Why this combination of parsers: [docs/parser-comparison.md](docs/parser-comparison.md).

## Stack

Python 3.11 · PydanticAI agents · Ollama / Ollama Cloud and Cerebras (gpt-oss) · a decision model through Ollama's
`/v1/systemone` · BM25 + bge-small embeddings with reciprocal rank fusion · bibr, Docling and PyMuPDF for parsing
· FastAPI (server-sent events) · Docker Compose + Caddy · uv · pytest · ruff · GitHub Actions

## Run it

The WDSF books are copyrighted and are **not** in this repository, and neither is anything built from them. Without
them you can run the lint and the unit tests:

```bash
make setup        # uv sync: the locked environment
make lint
make test         # tests/unit always; tests/index only when the index is built
```

With your own copies of the books in `theory/latin/<Dance>.pdf` and `theory/standard/<Dance>.pdf`, you can process PDF's. The current code is writen for an Apple Silicon Mac.
 (bibr's table OCR runs on MLX), [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com)
and `brew install libmagic`:

```bash
make setup-docling setup-bibr     # two separate parser environments
make extract                      # bibr + PyMuPDF + the books' index pages (cached; ~10-15 min per book)
make docling                      # Docling on the ~525 prose pages (cached; ~30-45 min the first time)
make build                        # the index: cache/agentic/index_v4 (~1 min)

make ask Q="How do I dance the Fan in Rumba?"     # one question with a full trace
make web                          # trace viewer: http://127.0.0.1:8765
make eval                         # gold-question evaluation (cloud models cost money)
make api                          # the public API on :8080 (deploy: docs/deploy.md)
```

Models are set in `dance_rag_agentic/settings.toml`. The default is cloud: `gpt-oss:120b` + `clef`, which needs
`ollama signin`. The free local pair is `make models` (`gpt-oss:20b` + `pplx-decider-flash:4b`).

## Repository

```
dance_rag_agentic/
  ingest/        reading the books and building the index
  retrieval/     read-only access to the index: sections, charts, hybrid search, outline
  agent/         the graph, agents, tools, evidence ids, claim checks, prompts
  apps/          public API, command line, trace viewer
  evaluation/    gold-question evaluation
tests/           unit (no books needed) and index tests
deploy/          Dockerfile, docker-compose, Caddy example
examples/widget/ the demo page widget (vanilla JS/CSS) with a local preview
docs/            architecture, decisions, evaluation, parser comparison, deployment, limitations
```
