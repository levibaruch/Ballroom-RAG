"""Check the gold question files: valid JSON, required fields, known kinds, and every source checkable.

    python -m dance_rag_agentic.evaluation.gold.validate              # all files in gold/draft/ and gold/
    python -m dance_rag_agentic.evaluation.gold.validate "Rumba"      # one file

A source passes when its evidence id exists and its quote appears in that block (spacing and
punctuation ignored). Problems are listed per question; nothing is changed.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from dance_rag_agentic.retrieval import books

GOLD_DIR = Path(__file__).resolve().parent
REQUIRED = ["id", "question", "language", "variations", "expected_kind", "dances", "figures", "role", "steps",
            "aspect", "answer", "sources", "tests", "notes", "status"]
KINDS = {"FIGURE", "FIGURE_COMPARE", "TECHNIQUE", "TERM", "CHARACTER", "GENERAL", "ABOUT", "OUT_OF_SCOPE"}
NO_BOOK_SOURCES = {"OUT_OF_SCOPE", "GENERAL", "ABOUT"}      # answered without the books' text
ROLES = {"man", "lady", "both", None}
ASPECTS = set(books.ASPECT_COLUMNS) | {"all", None}


def source_problems(source):
    problems = []
    for evidence_id in source.get("evidence_ids", []):
        try:
            section_id, block_number = (int(part) for part in str(evidence_id).split("."))
        except ValueError:
            problems.append(f"unreadable evidence id {evidence_id!r}")
            continue
        if not books.section_exists(section_id):
            problems.append(f"{evidence_id}: no such section")
            continue
        blocks = books.section_blocks(section_id)
        if not 1 <= block_number <= len(blocks):
            problems.append(f"{evidence_id}: section has {len(blocks)} blocks")
            continue
    quote = books.squash(source.get("quote", ""))
    cited_text = "".join(books.squash(books.section_blocks(int(str(e).split(".")[0]))[int(str(e).split(".")[1]) - 1])
                         for e in source.get("evidence_ids", [])
                         if not any(str(e) in problem for problem in problems))
    if quote and cited_text and quote not in cited_text:
        problems.append(f"quote not found in {source.get('evidence_ids')}: {source.get('quote', '')[:60]!r}")
    return problems


def item_problems(item):
    problems = [f"missing field '{name}'" for name in REQUIRED if name not in item]
    if item.get("expected_kind") not in KINDS:
        problems.append(f"unknown expected_kind {item.get('expected_kind')!r}")
    if item.get("role") not in ROLES:
        problems.append(f"unknown role {item.get('role')!r}")
    if item.get("aspect") not in ASPECTS:
        problems.append(f"unknown aspect {item.get('aspect')!r}")
    for dance in item.get("dances", []):
        if dance not in books.library().dances:
            problems.append(f"unknown dance {dance!r}")
    if item.get("expected_kind") not in NO_BOOK_SOURCES and "trap" not in item.get("variations", []) and not item.get("sources"):
        problems.append("no sources")
    for source in item.get("sources", []):
        problems += source_problems(source)
    return problems


def main():
    wanted = sys.argv[1:]
    files = sorted(GOLD_DIR.glob("*.json")) + sorted((GOLD_DIR / "draft").glob("*.json"))
    if wanted:
        files = [path for path in files if path.stem in wanted]
    total, kinds, variations = 0, Counter(), Counter()
    for path in files:
        try:
            items = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            print(f"{path.name}: INVALID JSON: {error}")
            continue
        bad = [(item.get("id", "?"), item_problems(item)) for item in items]
        bad = [(item_id, problems) for item_id, problems in bad if problems]
        print(f"{path.name}: {len(items)} questions, {len(bad)} with problems")
        for item_id, problems in bad:
            for problem in problems:
                print(f"    {item_id}: {problem}")
        total += len(items)
        kinds.update(item.get("expected_kind") for item in items)
        variations.update(variation for item in items for variation in item.get("variations", []))
    print(f"\n{total} questions | kinds {dict(kinds)} | variations {dict(variations.most_common(8))}")


if __name__ == "__main__":
    main()
