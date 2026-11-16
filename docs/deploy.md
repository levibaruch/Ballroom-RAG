# Deploying the API

The API (`dance_rag_agentic/apps/api.py`) runs as its own small Docker Compose project on a Linux VPS, next to
whatever else runs there: separate containers and network, bound to `127.0.0.1`, and one extra block in the host
Caddyfile. Below, `rag.example.com` is the API's domain and `example.com` the site with the demo page.

```
Internet ─▶ Caddy (host, TLS)
             ├─ (other sites)        unchanged
             └─ rag.example.com  ─▶ localhost:8090   rag-api ─▶ rag-ollama ─▶ Ollama Cloud
example.com (demo page) ── fetch ──▶ https://rag.example.com/v1/ask
```

Resources: `rag-api` ~1-1.5 GB RAM (limit 2 GB, `RAG_MEM_LIMIT`), `rag-ollama` ~0.2 GB (limit 512 MB),
~3 GB disk for the image. Check first: `free -h; nproc; df -h /`.

## 1. DNS

Add an A record `rag.example.com` -> the VPS IP.

## 2. Code on the VPS

Either clone the (private) GitHub repo, or copy it from the Mac:

```bash
# on the VPS
git clone git@github.com:<you>/dance-style-RAG.git ~/dance-style-RAG
# or, from the Mac:  rsync -av --exclude-from=.dockerignore ./ <vps>:~/dance-style-RAG/
```

## 3. The index (from the Mac; it contains book text, so never via git)

```bash
# on the Mac, in the repo
make sync-index VPS=<user>@<vps-host>
```

That copies `cache/agentic/index_v4/` and `cache/agentic/book_index.pkl` to `~/dance-style-RAG/cache/agentic/`.
The container runs as uid 1000 and writes traces and the daily spend file there:

```bash
# on the VPS
sudo chown -R 1000:1000 ~/dance-style-RAG/cache/agentic
```

## 4. Settings

Optional: without `deploy/.env` the API runs with its defaults ($1 per day, 5 questions per hour per visitor,
origins: the demo site in `apps/api.py`, models from `settings.toml`). To change them (the example file is hidden: `ls -a deploy`):

```bash
cd ~/dance-style-RAG
cp deploy/.env.example deploy/.env
nano deploy/.env          # allowed origins (your demo site), daily budget, questions per hour, models
```

## 5. Start

```bash
docker compose -f deploy/docker-compose.yml up -d --build
# one time: sign the ollama container in to Ollama Cloud (opens a link to confirm in the browser)
docker compose -f deploy/docker-compose.yml exec ollama ollama signin
docker compose -f deploy/docker-compose.yml exec ollama ollama pull gpt-oss:120b-cloud
docker compose -f deploy/docker-compose.yml exec ollama ollama pull clef:cloud
docker compose -f deploy/docker-compose.yml restart rag-api
```

Test from the VPS:

```bash
curl -s localhost:8090/v1/health
curl -N -s localhost:8090/v1/ask -H 'Content-Type: application/json' -d '{"question":"What is Guapacha timing?"}'
```

## 6. Caddy

Copy the block in `deploy/Caddyfile.example` into the host Caddyfile, with your API domain, then:

```bash
caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
curl -s https://rag.example.com/v1/health
```

## Updating

```bash
cd ~/dance-style-RAG && git pull
docker compose -f deploy/docker-compose.yml up -d --build
# new index? on the Mac: make build && make sync-index VPS=...; then on the VPS: docker compose ... restart rag-api
```

## Watching

```bash
docker compose -f deploy/docker-compose.yml logs -f rag-api
cat ~/dance-style-RAG/cache/agentic/api_spend.json        # USD spent per day
ls ~/dance-style-RAG/cache/agentic/traces/ | tail          # one trace per question (*-api.log)
docker stats rag-api rag-ollama --no-stream                # memory use
```

Stop it any time without touching the other sites: `docker compose -f deploy/docker-compose.yml down`.

## Protection built into the API

- Only the origins in `DANCE_RAG_ALLOWED_ORIGINS` may call it from a browser (CORS).
- `DANCE_RAG_DAILY_BUDGET_USD`: when today's cloud spend reaches it, the API answers "budget used up".
- `DANCE_RAG_QUESTIONS_PER_HOUR` per visitor (IP from Caddy's X-Forwarded-For), questions up to 400 characters,
  one run at a time with a queue of `DANCE_RAG_MAX_QUEUE`.
- Sources are short quotes (max ~200 characters) with book and page; the index stays on the server.
