# Demo widget

The page component of the public demo: a question box that streams the system's steps (routing, searches, claim
verdicts) from the API and then shows the cited answer with short source quotes. Vanilla JavaScript and CSS, no
dependencies; it talks to `dance_rag_agentic/apps/api.py` (`POST /v1/ask` as server-sent events, `GET /v1/examples`).

Preview locally (`index.html` sets the colour and font tokens the CSS expects):

```bash
make api                                            # the API on http://localhost:8080 (needs the built index)
python3 -m http.server 8000 -d examples/widget      # open http://localhost:8000
```

Embed on a site: load `dance-rag.css`, add `<div class="dance-rag" data-api="https://rag.example.com"></div>`,
load `dance-rag.js`, and add the site's origin to `DANCE_RAG_ALLOWED_ORIGINS` on the API.

This is a copy: the live version lives in the website's own repository (`static/js/dance-rag.js`,
`static/css/dance-rag.css`). Copy changes across when one of them changes.
