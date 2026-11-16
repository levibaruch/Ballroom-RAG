"""Run the graph over the gold questions and score it with code-only checks (no LLM judge yet).

    python -m dance_rag_agentic.evaluation.evaluate --routing-only            # entities + decision model + rules, ~1 s per question
    python -m dance_rag_agentic.evaluation.evaluate --limit 10                # full graph, minutes per question
    python -m dance_rag_agentic.evaluation.evaluate --status gold             # only items the user marked 'gold'

Results go to cache/agentic/eval/<timestamp>.json (one record per question) and a summary is printed.

Scores per question:
  route_ok      route == expected kind (UNSURE counts separately; a trap is scored on its expected kind)
  section_hit   at least one cited section is one of the gold sources' sections
  page_hit      at least one cited (dance, page) is a gold source page (looser: other sections on the page)
  refusal_ok    OUT_OF_SCOPE items: no citations and a refusal / 'not found'
  general_ok    GENERAL items: own-knowledge tips and no citations
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from dance_rag_agentic.retrieval import books
from dance_rag_agentic.agent.checks import unsupported
from dance_rag_agentic.config import CACHE_DIR, MODELS, TRACE_DIR, use_models
from dance_rag_agentic.config import INDEX_VERSION
from dance_rag_agentic.agent.entities import extract_entities
from dance_rag_agentic.agent.evidence import cited_ids
from dance_rag_agentic.agent.graph import answer_question, choose_route
from dance_rag_agentic.agent.decision import DecisionModelError, classify_question
from dance_rag_agentic.agent.models import add_model_flags, role_problems
from dance_rag_agentic.agent.usage import METER

GOLD_DIR = Path(__file__).resolve().parent / "gold"


def load_gold(status=None):
    items = []
    for path in sorted(GOLD_DIR.glob("*.json")) + sorted((GOLD_DIR / "draft").glob("*.json")):
        items += json.loads(path.read_text())
    if status is None:                                   # everything except what the user rejected in the review app
        return [item for item in items if item.get("status") != "rejected"]
    return [item for item in items if item.get("status") == status]


def routing_record(item):
    entities = extract_entities(item["question"])
    try:
        answers = classify_question(item["question"])
        route, reason = choose_route(answers["kind"]["probabilities"], answers["kind"]["confidence"], entities)
        confidence = round(answers["kind"]["confidence"], 2)
    except DecisionModelError as error:
        route, reason, confidence = "UNSURE", str(error), None
    return dict(id=item["id"], expected=item["expected_kind"], route=route, reason=reason, confidence=confidence,
                route_ok=route == item["expected_kind"], figures_found=entities["figures"],
                figures_expected=item.get("figures", []), dances_found=entities["dances"])


def cited_book_pages(section_ids):
    """{(dance, page)} for cited sections: every page a unit spans and, for Part 1 text that is the same in every
    book of a style, the same pages in the other books ('copies')."""
    pages = books.library().pages
    found = set()
    for section_id in section_ids:
        if section_id not in pages.index:
            continue
        record = pages.loc[section_id]
        last = int(record.page_last) if record.page_last == record.page_last else int(record.page)   # NaN check
        dances = {record.dance}
        copy_of = record.get("copy_of")
        holder = pages.loc[int(copy_of)] if isinstance(copy_of, (int, float)) and copy_of == copy_of else record
        if isinstance(holder.get("copies"), dict):
            dances |= set(holder["copies"])
        found |= {(dance, page) for dance in dances for page in range(int(record.page), last + 1)}
    return found


def answer_record(item, state, seconds):
    gold_sections = {source["section_id"] for source in item.get("sources", []) if source.get("section_id") is not None}
    gold_pages = {(source["dance"], source["page"]) for source in item.get("sources", [])}
    cited_sections = {section_id for section_id, _ in cited_ids(state.answer)}
    cited_pages = cited_book_pages(cited_sections)
    record = dict(id=item["id"], question=item["question"], expected=item["expected_kind"], route=state.route,
                  route_ok=state.route == item["expected_kind"], seconds=round(seconds),
                  answer=state.final_answer, cited_sections=sorted(cited_sections),
                  gold_sections=sorted(gold_sections), nodes=[node for node, _, _ in state.trace],
                  answer_attempts=state.answer_attempts, repair_rounds=state.repair_rounds,
                  citation_problems=state.citation_problems,
                  unsupported_claims=len(unsupported(state.judged_claims)), index=INDEX_VERSION,
                  agent_tool_calls=len(state.searches_done), citation_errors=state.citation_error_counts,
                  teacher_parts=sum(part.lower() in state.final_answer.lower()
                                    for part in ("What it is", "How to do it", "What to watch")))
    if gold_sections:
        record["section_hit"] = bool(cited_sections & gold_sections)
    if gold_pages:
        record["page_hit"] = bool(cited_pages & gold_pages)
    if item["expected_kind"] == "OUT_OF_SCOPE":
        text = state.final_answer.lower()
        record["refusal_ok"] = not cited_sections and ("outside" in text or "not found" in text)
    record["general_advice"] = bool(state.general_answer)     # own-knowledge part (GENERAL route or split question)
    if item["expected_kind"] == "GENERAL":
        record["general_ok"] = bool(state.general_answer) and not cited_sections
    if item.get("reference_answer"):                     # the user's checked answer (gold/review app)
        record["reference_answer"] = item["reference_answer"]
    return record


def token_summary(records):
    """Total and mean tokens (and USD where settings.toml has a price) over all questions of a run."""
    totals = {}
    for record in records:
        for name, entry in record.get("tokens", {}).items():
            total = totals.setdefault(name, {"input": 0, "output": 0, "usd": None})
            total["input"] += entry["input"]
            total["output"] += entry["output"]
            if entry["usd"] is not None:
                total["usd"] = (total["usd"] or 0) + entry["usd"]
    count = max(len(records), 1)
    lines = ["tokens (total | mean per question):"]
    for name, total in totals.items():
        usd = f"  ${total['usd']:.4f} | ${total['usd'] / count:.5f}" if total["usd"] is not None else ""
        lines.append(f"  {name:38} in {total['input']:>10,} | {total['input'] // count:>8,}   "
                     f"out {total['output']:>9,} | {total['output'] // count:>7,}{usd}")
    return "\n".join(lines)


def summarize(records):
    def share(key):
        values = [record[key] for record in records if key in record]
        return f"{sum(values)}/{len(values)}" if values else "-"
    print(f"\n{len(records)} questions | route_ok {share('route_ok')} | UNSURE "
          f"{sum(record['route'] == 'UNSURE' for record in records)} | section_hit {share('section_hit')} | "
          f"page_hit {share('page_hit')} | refusal_ok {share('refusal_ok')} | general_ok {share('general_ok')}")
    print(token_summary(records))
    for record in records:
        if not record["route_ok"]:
            print(f"  route {record['id']}: expected {record['expected']}, got {record['route']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate dance_rag_agentic on the gold questions.")
    parser.add_argument("--routing-only", action="store_true", help="only entities + decision model + rules (fast)")
    parser.add_argument("--limit", type=int, help="first N questions only")
    parser.add_argument("--status", choices=["draft", "gold"], help="only items with this status")
    parser.add_argument("--ids", help="file with question ids (one per line): only these")
    parser.add_argument("--shard", help="k/n: only every n-th question starting at k (run n processes in parallel)")
    parser.add_argument("--tag", default="", help="extra text in the output file name")
    add_model_flags(parser)
    arguments = parser.parse_args()
    use_models(arguments.answer_model, arguments.decision_model)
    problems = role_problems()
    if problems:
        raise SystemExit("cannot run:\n- " + "\n- ".join(problems))
    print(f"models: answer={MODELS.answer}  decision={MODELS.decision}")
    items = load_gold(arguments.status)
    if not items:
        raise SystemExit(f"no gold questions in {GOLD_DIR}/draft: the gold set is private (it quotes the books). "
                         f"The format is in {GOLD_DIR}/example/questions.json")
    if arguments.ids:
        wanted = [line.strip() for line in Path(arguments.ids).read_text().splitlines() if line.strip()]
        items = [item for item in items if item["id"] in wanted]
    if arguments.shard:
        shard, shards = (int(part) for part in arguments.shard.split("/"))
        items = items[shard::shards]
    items = items[:arguments.limit]
    records = []
    for number, item in enumerate(items, start=1):
        METER.reset()                                   # tokens per question (routing-only too)
        if arguments.routing_only:
            records.append(routing_record(item))
        else:
            start = time.time()
            # every eval question gets a trace (.log + .jsonl) like `ask --trace`, so it shows in the web viewer
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            trace_path = TRACE_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-eval-{INDEX_VERSION}-{item['id']}.log"
            trace_path.write_text(f"QUESTION: {item['question']}\nMODELS: answer={MODELS.answer} "
                                  f"decision={MODELS.decision}\nEVAL: {item['id']} index {INDEX_VERSION}\n", encoding="utf-8")
            state = asyncio.run(answer_question(item["question"], log_path=trace_path))
            with open(trace_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"\nFINAL ANSWER (route {state.route})\n{state.final_answer}\n")
            records.append(answer_record(item, state, time.time() - start))
        records[-1].update(answer_model=MODELS.answer, decision_model=MODELS.decision,
                           tokens={f"{role} {model}": entry for (role, model), entry in METER.totals().items()})
        print(f"{number:3}/{len(items)} {'ok ' if records[-1]['route_ok'] else 'X  '} {item['id']:22} "
              f"{records[-1]['route']}", flush=True)
    output_dir = CACHE_DIR / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = f"{MODELS.decision}" if arguments.routing_only else f"{MODELS.answer}+{MODELS.decision}"
    model_tag = model_tag.replace(":", "-").replace("/", "-")
    output_path = output_dir / (f"{time.strftime('%Y%m%d-%H%M')}{'-routing' if arguments.routing_only else ''}"
                                f"-{model_tag}-index_{INDEX_VERSION}{('-' + arguments.tag) if arguments.tag else ''}"
                                f"{('-shard' + arguments.shard.replace('/', 'of')) if arguments.shard else ''}.json")
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False, default=str))
    summarize(records)
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()
