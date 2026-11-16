# Reproducible pipeline for dance_rag_agentic. Run `make help`.
# Needs: uv (https://docs.astral.sh/uv/), Ollama (https://ollama.com), the book PDFs in theory/<latin|standard>/.
PY        := uv run python
DOCLING   := .venv-docling/bin/python
SHARDS    := 3

.PHONY: help setup setup-docling setup-bibr models extract docling build all test lint web ask eval routing sync-index api

help:
	@echo "make setup          main environment (.venv from pyproject.toml + uv.lock: uv sync)"
	@echo "make setup-docling  Docling environment (.venv-docling)"
	@echo "make setup-bibr     bibr environment (.venv-bibr, Python 3.12, Apple Silicon; needs brew install libmagic)"
	@echo "make models         pull the local Ollama models"
	@echo "make extract        bibr + PyMuPDF + book index per book (cached; bibr ~10-15 min per book)"
	@echo "make docling        Docling on all prose pages, $(SHARDS) processes (cached; ~30-45 min the first time)"
	@echo "make build          build index v4 (cache/agentic/index_v4, ~1 min)"
	@echo "make all            extract + docling + build"
	@echo "make test           offline tests, no models (tests/unit always; tests/index when the index is built)"
	@echo "make lint           ruff (pyflakes + pycodestyle)"
	@echo "make web            web viewer on http://127.0.0.1:8765"
	@echo "make ask Q='...'    one question with a full trace"
	@echo "make eval           gold-question evaluation (uses the models in settings.toml; cloud models cost money)"
	@echo "make api            the public API locally on :8080 (docs/deploy.md for the VPS)"
	@echo "make sync-index VPS=user@host   copy the built index to the VPS"

setup:
	uv sync

setup-docling:
	uv venv -p 3.11 .venv-docling
	uv pip install -p $(DOCLING) -r requirements-docling.txt

setup-bibr:
	uv venv -p 3.12 .venv-bibr
	uv pip install -p .venv-bibr/bin/python -r requirements-bibr.txt

models:
	ollama pull gpt-oss:20b
	ollama pull pplx-decider-flash:4b

extract:
	$(PY) -m dance_rag_agentic.ingest.extract

docling:
	$(PY) -m dance_rag_agentic.ingest.docling_pages --plan
	@for k in $$(seq 0 $$(($(SHARDS)-1))); do $(DOCLING) dance_rag_agentic/ingest/docling_pages.py $$k/$(SHARDS) & done; wait

build:
	$(PY) -m dance_rag_agentic.ingest.build_index

all: extract docling build

test:
	uv run pytest -q

lint:
	uv run ruff check dance_rag_agentic tests

web:
	$(PY) -m dance_rag_agentic.apps.web

ask:
	$(PY) -m dance_rag_agentic.apps.cli "$(Q)" --trace

eval:
	$(PY) -m dance_rag_agentic.evaluation.evaluate

routing:
	$(PY) -m dance_rag_agentic.evaluation.evaluate --routing-only

# ── deployment (see docs/deploy.md) ───────────────────────────────────────────────────────────────
VPS_DIR := ~/dance-style-RAG

sync-index:          ## copy the built index to the VPS: make sync-index VPS=user@host
	@test -n "$(VPS)" || (echo "usage: make sync-index VPS=user@host" && exit 1)
	ssh $(VPS) "mkdir -p $(VPS_DIR)/cache/agentic"
	rsync -av cache/agentic/index_v4 cache/agentic/book_index.pkl $(VPS):$(VPS_DIR)/cache/agentic/

api:                 ## run the public API locally on :8080 (as on the VPS, but with your local Ollama)
	$(PY) -m dance_rag_agentic.apps.api --port 8080


