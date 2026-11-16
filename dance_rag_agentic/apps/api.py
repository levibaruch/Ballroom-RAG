"""Public API (FastAPI) for the demo on levibaruch.dev (the local viewer is web.py; this is the hosted endpoint).

    python -m dance_rag_agentic.apps.api --host 0.0.0.0 --port 8080

    POST /v1/ask      {"question": "..."}  ->  text/event-stream:
                      data: {"type": "node", "node": "ClassifyKind", "message": "Deciding ..."}        (a step starts)
                      data: {"type": "detail", "label": ..., "lines": [...], "data": {...}}         (whitelisted)
                      data: {"type": "tool", "call": "search(query='...')"}                          (no results)
                      data: {"type": "step", "node": ..., "seconds": ..., "summary": "route FIGURE ..."} (step done)
                      data: {"type": "answer", "route": ..., "body": "... [1]", "sources": [...], "tips": "..."}
                      data: {"type": "error", "message": "..."}
    GET  /v1/health   status, index, models, budget left today
    GET  /v1/examples example questions for the page
    GET  /docs        interactive API docs (OpenAPI)

Public-safety rules (user, 2026-11-13):
  - sources are SHORT quotes (QUOTE_CHARACTERS) + book/page: the WDSF books are copyrighted; the index itself stays
    on the server
  - a daily budget cap (cloud models cost money per question) and a rate limit per visitor
  - only the origins in DANCE_RAG_ALLOWED_ORIGINS may call it (browser CORS); one run at a time, a short queue
Settings by environment (see deploy/.env.example).
"""
import argparse
import asyncio
import datetime
import json
import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from dance_rag_agentic.config import CACHE_DIR, MODELS, TRACE_DIR
from dance_rag_agentic.agent.evidence import block_text, with_footnotes
from dance_rag_agentic.agent.graph import CHARACTER_HEADING, GENERAL_HEADING, answer_question
from dance_rag_agentic.config import INDEX_VERSION
from dance_rag_agentic.agent.usage import METER

DAILY_BUDGET_USD = float(os.environ.get("DANCE_RAG_DAILY_BUDGET_USD", "1.00"))
QUESTIONS_PER_HOUR = int(os.environ.get("DANCE_RAG_QUESTIONS_PER_HOUR", "5"))      # per visitor (IP)
MAX_QUEUE = int(os.environ.get("DANCE_RAG_MAX_QUEUE", "3"))                      # waiting runs; more -> 503
MAX_QUESTION_CHARACTERS = 400
QUOTE_CHARACTERS = 200
ALLOWED_ORIGINS = [origin.strip() for origin in os.environ.get(
    "DANCE_RAG_ALLOWED_ORIGINS", "https://levibaruch.dev,https://www.levibaruch.dev,http://localhost:8000").split(",")
    if origin.strip()]
SPEND_FILE = CACHE_DIR / "api_spend.json"
EXAMPLES = [
    "How do I dance the Fan in Rumba?",
    "What is the difference between Cha Cha and Rumba?",
    "Which figures can follow the Fan in Rumba?",
    "What is contrary body movement in the Slow Waltz?",
    "Wat is Guapacha timing?",
    "How should I start learning to dance?",
]

run_lock = asyncio.Lock()                     # one graph run at a time: the token meter is per process
waiting = {"count": 0}
recent_questions = defaultdict(deque)         # visitor -> times of their questions in the last hour


# ------------------------------------------------------------------ budget and rate limit
def spent_today():
    today = datetime.date.today().isoformat()
    try:
        return json.loads(SPEND_FILE.read_text()).get(today, 0.0)
    except (FileNotFoundError, ValueError):
        return 0.0


def add_spend(usd):
    today = datetime.date.today().isoformat()
    try:
        spend = json.loads(SPEND_FILE.read_text())
    except (FileNotFoundError, ValueError):
        spend = {}
    spend[today] = round(spend.get(today, 0.0) + usd, 4)
    SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPEND_FILE.write_text(json.dumps(spend, indent=1))


def visitor(request):
    """The visitor's IP; behind the reverse proxy it is the first X-Forwarded-For address."""
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def rate_limited(who):
    now, times = time.time(), recent_questions[who]
    while times and now - times[0] > 3600:
        times.popleft()
    if len(times) >= QUESTIONS_PER_HOUR:
        return True
    times.append(now)
    return False


# ------------------------------------------------------------------ the public answer
def public_answer(state):
    """Footnote numbers + short quotes with book and page; never whole charts or pages."""
    body, footnotes = with_footnotes(state.answer) if state.answer else (state.final_answer, {})
    from dance_rag_agentic.retrieval.books import library
    pages = library().pages
    sources = []
    for evidence_id, number in sorted(footnotes.items(), key=lambda item: item[1]):
        section_id, _ = evidence_id
        record = pages.loc[section_id]
        quote = " ".join((block_text(evidence_id) or "").split())
        if len(quote) > QUOTE_CHARACTERS:
            quote = quote[:QUOTE_CHARACTERS].rsplit(" ", 1)[0] + " …"
        sources.append({"number": number, "book": record.dance, "page": int(record.page),
                        "section": str(record.header) if isinstance(record.header, str) else "", "quote": quote})
    return {"type": "answer", "route": state.route, "body": body,
            "sources": sources, "tips": state.general_answer or "",
            "tips_heading": TIPS_HEADINGS.get(state.general_heading, TIPS_HEADINGS[GENERAL_HEADING])}


# The widget's heading above the own-knowledge part (the trace's headings are in capitals for the terminal)
TIPS_HEADINGS = {GENERAL_HEADING: "General advice, not from the books",
                 CHARACTER_HEADING: "The feel of the dance: general dance knowledge, not from the books"}


STEP_TEXT = {"ExtractEntities": "Reading the question", "ClassifyKind": "Deciding what kind of question it is",
             "SplitQuestion": "Splitting technique and general advice", "GeneralAdvice": "Writing general advice",
             "FigurePath": "Reading the figure charts", "ComparePath": "Reading the charts to compare",
             "Background": "Looking up the terms the charts use", "TermPath": "Looking the term up in the book index",
             "TechniqueAgent": "Searching the books", "UnsureAgent": "Searching the books", "Answer": "Writing the answer",
             "CheckCites": "Checking the citations", "JudgeClaims": "Checking every claim against the books",
             "RepairClaims": "Correcting unsupported claims", "FillGap": "Searching for missing evidence",
             "Polish": "Tidying the answer", "Refuse": "Outside the books", "AboutPath": "Describing what I know",
             "CharacterPath": "Reading what shapes the dance's character", "SafetyReply": "Pointing to help",
             "Finish": "Done"}


# ------------------------------------------------------------------ the public trace (whitelist)
# The demo shows what the system does (routing, searching, checking, repairing), but only details on this list.
# Everything else is dropped, so a new detail added to the graph later is NOT shown until someone adds it here.
# Never shown: book text (tool results, judge input, evidence blocks), prompts and agent requests, the agent's
# free-form note, file paths, internal errors.
PUBLIC_DETAILS = (                              # label prefixes of graph details that may be shown
    "dances", "figures", "role", "steps",
    "decision model:", "confidence", "aspect used", "override rules",
    "books part", "general part",
    "figure '", "dances asked", "charts in ", "sides", "chart terms handed",
    "book index entries", "section headings named",
    "new evidence", "evidence given",
    "attempt ", "answer (raw", "REWRITE", "problems", "ranges listed by code", "lines that repeat", "style",
    "chart steps cited as blocks",
    "claims judged", "does the answer respond", "the answer says 'Not found",
    "claims to repair", "repair decisions", "new sentences judged", "answer after the repairs",
    "to fill", "polished answer", "polish rejected", "sentences that are new after polishing",
    "agent read only", "the agent read nothing",
    "character sections read by code", "language",
    "tokens used", "warnings",
)
MAX_LINES, MAX_LINE_CHARACTERS = 40, 400


def public_detail(event):
    """A graph 'show' event -> a safe detail, or None. Structured data is passed on only for probabilities and
    judged claims (claim text, verdict, score, cited ids: no evidence text)."""
    label = str(event.get("label") or "")
    if not label.startswith(PUBLIC_DETAILS):
        return None
    lines = [str(line)[:MAX_LINE_CHARACTERS] for line in (event.get("lines") or [])[:MAX_LINES]]
    detail = {"type": "detail", "label": label[:MAX_LINE_CHARACTERS], "lines": lines}
    data = event.get("data")
    if isinstance(data, dict) and data.get("kind") == "probabilities":
        detail["data"] = {"kind": "probabilities", "marked": data.get("marked"),
                          "items": [[str(name), float(value)] for name, value in data.get("items", [])][:12]}
    elif isinstance(data, dict) and data.get("kind") == "claims":
        detail["lines"] = []                    # the structured claims say it better than the text lines
        detail["data"] = {"kind": "claims", "threshold": data.get("threshold"),
                          "items": [{"claim": str(item.get("claim", ""))[:MAX_LINE_CHARACTERS],
                                     "p": item.get("p"), "verdict": item.get("verdict"),
                                     "ids": [str(i) for i in item.get("ids", [])][:8]}
                                    for item in data.get("items", [])[:MAX_LINES]]}
    return detail


def public_tool_call(event):
    """An agent tool call: the call only (its result is book text and is never sent)."""
    return {"type": "tool", "call": str(event.get("call") or "")[:MAX_LINE_CHARACTERS]}


# ------------------------------------------------------------------ the API (FastAPI)
class AskRequest(BaseModel):
    question: str = Field("", description=f"A question about ballroom or Latin technique, at most "
                                          f"{MAX_QUESTION_CHARACTERS} characters", examples=[EXAMPLES[0]])


class Health(BaseModel):
    ok: bool
    index: str
    models: dict[str, str]
    budget_left_today_usd: float
    busy: bool
    waiting: int


class Examples(BaseModel):
    examples: list[str]


app = FastAPI(
    title="Dance Style RAG",
    version="1.0",
    description="Answers questions about ballroom and Latin technique from the WDSF technique books, with every claim "
                "cited and checked. `POST /v1/ask` streams what the system does (routing, searching, checking, "
                "repairing) and then the answer, as server-sent events.",
)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])


def refuse(status_code, message):
    return JSONResponse({"error": message}, status_code=status_code)


@app.post("/v1/ask", summary="Ask a question (streams server-sent events)",
          responses={200: {"content": {"text/event-stream": {}}, "description": "node / detail / tool / step events, "
                                                                                 "then one answer (or error) event"},
                     400: {"description": "empty or too long"}, 429: {"description": "budget or rate limit"},
                     503: {"description": "busy"}})
async def ask(body: AskRequest, request: Request):
    """Event types, one JSON object per `data:` line:
    `node` a step starts · `detail` a whitelisted detail of that step · `tool` an agent tool call (never its result)
    · `step` the step is done (seconds, summary) · `answer` body with footnotes, short sources, tips · `error`."""
    question = " ".join(body.question.split())
    if not question:
        return refuse(400, "empty question")
    if len(question) > MAX_QUESTION_CHARACTERS:
        return refuse(400, f"question longer than {MAX_QUESTION_CHARACTERS} characters")
    if spent_today() >= DAILY_BUDGET_USD:
        return refuse(429, "the demo's budget for today is used up; try again tomorrow")
    if rate_limited(visitor(request)):
        return refuse(429, f"max {QUESTIONS_PER_HOUR} questions per hour; try again later")
    if waiting["count"] >= MAX_QUEUE:
        return refuse(503, "busy; try again in a minute")

    queue = asyncio.Queue()

    def sink(event):                          # graph events -> the visitor: steps + whitelisted details
        kind = event.get("type")
        if kind == "node":
            queue.put_nowait({"type": "node", "node": event["node"],
                              "message": STEP_TEXT.get(event["node"], event["node"])})
        elif kind == "show":
            detail = public_detail(event)
            if detail:
                queue.put_nowait(detail)
        elif kind == "tool":
            queue.put_nowait(public_tool_call(event))
        elif kind == "end":
            queue.put_nowait({"type": "step", "node": event["node"], "seconds": event.get("seconds"),
                              "summary": str(event.get("message") or "")[:MAX_LINE_CHARACTERS],
                              "message": STEP_TEXT.get(event["node"], event["node"])})

    async def run():
        waiting["count"] += 1
        try:
            async with run_lock:
                waiting["count"] -= 1
                queue.put_nowait({"type": "step", "node": "start", "message": "Starting"})
                TRACE_DIR.mkdir(parents=True, exist_ok=True)
                log_path = TRACE_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-api.log"
                log_path.write_text(f"QUESTION: {question}\nMODELS: answer={MODELS.answer} "
                                    f"decision={MODELS.decision}\nAPI\n", encoding="utf-8")
                state = await answer_question(question, log_path=log_path, event_sink=sink)
                add_spend(sum(entry.get("usd") or 0.0 for entry in METER.totals().values()))
                queue.put_nowait(public_answer(state))
        except Exception as error:            # never leak internals to the page
            queue.put_nowait({"type": "error", "message": "something went wrong; please try again later"})
            print(f"api error: {type(error).__name__}: {error}", flush=True)
        finally:
            queue.put_nowait(None)

    asyncio.create_task(run())

    async def stream():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/v1/health", response_model=Health, summary="Status, models and budget left today")
async def health():
    return Health(ok=True, index=INDEX_VERSION, models={"answer": MODELS.answer, "decision": MODELS.decision},
                  budget_left_today_usd=round(DAILY_BUDGET_USD - spent_today(), 2), busy=run_lock.locked(),
                  waiting=waiting["count"])


@app.get("/v1/examples", response_model=Examples, summary="Example questions for the demo page")
async def examples():
    return Examples(examples=EXAMPLES)


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    arguments = parser.parse_args()
    from dance_rag_agentic.retrieval.books import library
    from dance_rag_agentic.retrieval.index import embedding_model
    library()                                  # the index ...
    embedding_model().encode("warm up")        # ... and the embedding model (lazy otherwise: the first visitor waited)
    print(f"dance-rag API on http://{arguments.host}:{arguments.port}  (index {INDEX_VERSION}, "
          f"budget ${DAILY_BUDGET_USD}/day, {QUESTIONS_PER_HOUR} questions/hour per visitor)", flush=True)
    uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="warning", proxy_headers=True)


if __name__ == "__main__":
    main()
