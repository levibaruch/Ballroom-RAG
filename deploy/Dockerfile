# Dance RAG API (dance_rag_agentic/apps/api.py) for a Linux x86_64 VPS. Build context = repository root:
#   docker compose -f deploy/docker-compose.yml up -d --build --no-deps rag-api
# Dependencies come from uv.lock (torch = the CPU build on Linux, see pyproject.toml [tool.uv.sources]).
# Layer order matters for rebuild time: everything slow (dependencies, embedding model) comes before the code, so a
# code edit only rebuilds the last layer (seconds). Nothing after the code may touch the big layers (no chown -R).
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.12 /uv /uvx /bin/
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# The user the container runs as (docker-compose: user 1000:1000). It only writes to the mounted cache/agentic;
# the code and the environment stay owned by root and read-only to it.
RUN useradd --create-home --no-log-init --uid 1000 appuser

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv
# Only the dependency files (no README: pyproject has package = false), so editing code or docs never reinstalls
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"

# The embedding model goes into the image, so the container never downloads at runtime
ENV HF_HOME=/app/models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" \
    && chown -R appuser:appuser /app/models
# the model is in the image now: never contact the Hugging Face Hub at runtime (no update check, no warning)
ENV HF_HUB_OFFLINE=1

COPY dance_rag_agentic ./dance_rag_agentic
USER appuser

ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "-m", "dance_rag_agentic.apps.api", "--host", "0.0.0.0", "--port", "8080"]
