"""Web viewer for the agentic graph: ask a question, watch every node work, read the answer with its sources.

    python -m dance_rag_agentic.apps.web                 # then open http://127.0.0.1:8765
    python -m dance_rag_agentic.apps.web --port 9000

Notes: per run (whole run or one step), saved next to the run as <time>.notes.json; the "All notes" view
lists the notes of every run, so many runs can be reviewed at once.

Every run (from here or from `ask --trace`) is saved in cache/agentic/traces/ as <time>.log (plain text) and
<time>.jsonl (structured events, see State.emit in graph.py). The page lists those runs and reads the .jsonl;
an old run with only a .log is shown as plain text.

One question at a time (two model jobs on one Ollama make both slow): asking while a run is busy is refused.
The run goes on in the background when the page is closed or reloaded; the page polls the .jsonl.
"""
import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from dance_rag_agentic.config import CLOUD_MODELS, MODELS, PRICES, TRACE_DIR, use_models
from dance_rag_agentic.agent.graph import answer_question
from dance_rag_agentic.agent.models import ROLE_NEEDS, installed_models, role_problems

PAGE = Path(__file__).resolve().parent / "static" / "index.html"
DEFAULT_MODELS = {"answer": MODELS.answer, "decision": MODELS.decision}   # settings.toml / env, at start
running = {"name": None}                                                  # the run that is busy now


def read_events(name):
    path = TRACE_DIR / f"{name}.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:        # a line being written right now
            break
    return events


# ------------------------------------------------------------------ notes
NOTE_TAGS = ["bug", "wrong answer", "good", "idea", "question"]


def notes_path(name):
    return TRACE_DIR / f"{name}.notes.json"


def read_notes(name):
    path = notes_path(name)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def write_notes(name, notes):
    notes_path(name).write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")


def run_exists(name):
    return (TRACE_DIR / f"{name}.log").exists() or (TRACE_DIR / f"{name}.jsonl").exists()


def run_summary(name):
    """One row of the run list: question, route, seconds, tokens, finished or not."""
    events = read_events(name)
    if not events:                          # an old run with only a text log
        log_path = TRACE_DIR / f"{name}.log"
        first_line = log_path.read_text(encoding="utf-8").split("\n", 1)[0] if log_path.exists() else ""
        return {"name": name, "question": first_line.removeprefix("QUESTION: "), "structured": False,
                "running": False, "finished": True, "notes": len(read_notes(name))}
    start, final = events[0], next((event for event in events if event["type"] == "final"), None)
    error = next((event for event in events if event["type"] == "error"), None)
    return {"name": name, "question": start.get("question", ""), "structured": True,
            "answer_model": start.get("answer_model"), "decision_model": start.get("decision_model"),
            "route": final["route"] if final else None, "seconds": round(events[-1]["time"] - start["time"]),
            "running": running["name"] == name, "finished": bool(final or error), "error": bool(error),
            "notes": len(read_notes(name))}


async def page(request):
    return FileResponse(PAGE)


async def list_runs(request):
    names = sorted({path.stem for path in TRACE_DIR.glob("*.log")} | {path.stem for path in TRACE_DIR.glob("*.jsonl")},
                   reverse=True)
    return JSONResponse({"runs": [run_summary(name) for name in names], "running": running["name"]})


async def get_run(request):
    name = request.path_params["name"]
    events = read_events(name)
    log_path = TRACE_DIR / f"{name}.log"
    return JSONResponse({"name": name, "events": events, "running": running["name"] == name,
                         "log": log_path.read_text(encoding="utf-8") if log_path.exists() else "",
                         "notes": read_notes(name)})


async def add_note(request):
    """{text, tag, node} -> the note is saved at once. node: the step the note is about ('' = whole run)."""
    name = request.path_params["name"]
    if not run_exists(name):
        return JSONResponse({"error": "no such run"}, status_code=404)
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "empty note"}, status_code=400)
    note = {"id": uuid.uuid4().hex[:8], "time": time.strftime("%Y-%m-%d %H:%M"), "text": text,
            "tag": body.get("tag") or "", "node": body.get("node") or ""}
    write_notes(name, read_notes(name) + [note])
    return JSONResponse(note)


async def change_note(request):
    """PATCH {text?, tag?} edits a note; DELETE removes it (from this run's notes file only)."""
    name, note_id = request.path_params["name"], request.path_params["note_id"]
    notes = read_notes(name)
    note = next((note for note in notes if note["id"] == note_id), None)
    if note is None:
        return JSONResponse({"error": "no such note"}, status_code=404)
    if request.method == "DELETE":
        write_notes(name, [other for other in notes if other["id"] != note_id])
        return JSONResponse({"deleted": note_id})
    body = await request.json()
    note.update({key: body[key] for key in ("text", "tag") if key in body})
    write_notes(name, notes)
    return JSONResponse(note)


async def all_notes(request):
    """Every note of every run, newest run first, with the run's question and route."""
    runs = []
    for path in sorted(TRACE_DIR.glob("*.notes.json"), reverse=True):
        name = path.name.removesuffix(".notes.json")
        notes = read_notes(name)
        if notes:
            runs.append(run_summary(name) | {"note_list": notes})
    return JSONResponse({"runs": runs, "tags": NOTE_TAGS})


async def get_models(request):
    """Installed models that fit each role, the defaults, and the prices known in settings.toml."""
    try:
        models = await run_in_threadpool(installed_models)
    except Exception as error:
        return JSONResponse({"error": f"cannot reach Ollama: {type(error).__name__}"}, status_code=503)
    cloud_names = {name for names in CLOUD_MODELS.values() for name in names}
    fits = {role: sorted(model["name"] for model in models if needs <= model["capabilities"]
                         and model["name"] not in cloud_names and not model["name"].endswith("-cloud"))
            for role, needs in ROLE_NEEDS.items()}
    return JSONResponse({"fits": fits, "cloud": CLOUD_MODELS, "defaults": DEFAULT_MODELS, "prices": PRICES})


async def ask(request):
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "empty question"}, status_code=400)
    if running["name"]:
        return JSONResponse({"error": f"a run is busy ({running['name']}): one model job at a time"}, status_code=409)
    use_models(body.get("answer_model") or DEFAULT_MODELS["answer"],
               body.get("decision_model") or DEFAULT_MODELS["decision"])
    problems = await run_in_threadpool(role_problems)
    if problems:
        return JSONResponse({"error": "cannot run: " + "; ".join(problems)}, status_code=400)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    name = time.strftime("%Y%m%d-%H%M%S")
    log_path = TRACE_DIR / f"{name}.log"
    log_path.write_text(f"QUESTION: {question}\nMODELS: answer={MODELS.answer} decision={MODELS.decision}\n",
                        encoding="utf-8")
    running["name"] = name
    asyncio.create_task(_run(question, bool(body.get("use_judge", True)), log_path))
    return JSONResponse({"name": name})


async def _run(question, use_judge, log_path):
    try:
        state = await answer_question(question, use_judge=use_judge, log_path=log_path)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n" + "═" * 74 + f"\nFINAL ANSWER  (route {state.route}: {state.route_reason})\n"
                           + "═" * 74 + "\n" + state.final_answer + "\n")
    except Exception as error:                  # shown in the page instead of a run that never finishes
        with open(log_path.with_suffix(".jsonl"), "a", encoding="utf-8") as events_file:
            events_file.write(json.dumps({"type": "error", "time": round(time.time(), 2),
                                          "message": f"{type(error).__name__}: {error}"[:2000]}) + "\n")
    finally:
        running["name"] = None


app = Starlette(routes=[
    Route("/", page),
    Route("/api/runs", list_runs),
    Route("/api/runs/{name}", get_run),
    Route("/api/runs/{name}/notes", add_note, methods=["POST"]),
    Route("/api/runs/{name}/notes/{note_id}", change_note, methods=["PATCH", "DELETE"]),
    Route("/api/notes", all_notes),
    Route("/api/models", get_models),
    Route("/api/ask", ask, methods=["POST"]),
])


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    print(f"open http://127.0.0.1:{arguments.port}")
    uvicorn.run(app, host="127.0.0.1", port=arguments.port, log_level="warning")


if __name__ == "__main__":
    main()
