"""Docling on the PROSE pages of every book (pages without a chart title), one page at a time, cached:
cache/agentic/docling/<Dance>/p<NNN>.items.json = Docling's reading-order items (headings, text, list items,
tables as Markdown). Used by the v4 index build (prose structure + text; tables are swapped for bibr's).

Run with the Docling environment (.venv-docling, kept apart from the project venv):
    uv run python -m dance_rag_agentic.ingest.docling_pages --plan             # writes the page list
    .venv-docling/bin/python dance_rag_agentic/ingest/docling_pages.py 0/3     # shard 0 of 3 (run 3 in parallel)
Why Docling: in the parser comparison (15 hard pages) it found headings bibr lost (Part 2
'INTRODUCTION') and gave the cleanest text; bibr stays for tables and charts."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCLING_DIR = ROOT / "cache" / "agentic" / "docling"
PLAN_FILE = DOCLING_DIR / "plan.json"


def write_plan():
    """[(pdf path, dance, page)] of all prose pages, from the PyMuPDF cache (chart pages have chart titles)."""
    sys.path.insert(0, str(ROOT))
    import pandas as pd
    from dance_rag_agentic.config import PYMUPDF_CACHE_DIR, all_book_pdfs
    plan = []
    for pdf_path in all_book_pdfs():
        records = pd.read_pickle(PYMUPDF_CACHE_DIR / f"{pdf_path.stem}.pkl")
        chart_pages = set(records[records.kind == "chart"].page)
        plan += [(str(pdf_path), pdf_path.stem, int(page)) for page in sorted(set(records.page) - chart_pages)]
    DOCLING_DIR.mkdir(parents=True, exist_ok=True)
    PLAN_FILE.write_text(json.dumps(plan))
    print(f"{len(plan)} prose pages planned")
    # the shards convert single-page PDFs (whole book + page_range was ~10x slower): make the missing ones here,
    # in the main environment (PyMuPDF). Before 2026-11-14 they were made by hand, so a new book got none.
    import pymupdf
    pages_dir = DOCLING_DIR / "page_pdfs"
    pages_dir.mkdir(parents=True, exist_ok=True)
    made, books = 0, {}
    for pdf_path, dance, page in plan:
        single = pages_dir / f"{dance}-p{page:03d}.pdf"
        if single.exists():
            continue
        book = books.setdefault(pdf_path, pymupdf.open(pdf_path))
        one_page = pymupdf.open()
        one_page.insert_pdf(book, from_page=page - 1, to_page=page - 1)
        one_page.save(single)
        made += 1
    print(f"{made} single-page PDFs made")


def run_shard(shard, shards):
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    plan = json.loads(PLAN_FILE.read_text())[shard::shards]
    started = time.time()
    for number, (pdf_path, dance, page) in enumerate(plan, start=1):
        target = DOCLING_DIR / dance / f"p{page:03d}.items.json"
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # a single-page PDF: converting the whole book with page_range was ~10x slower (39 pages in 15 min)
        single = DOCLING_DIR / "page_pdfs" / f"{dance}-p{page:03d}.pdf"
        result = converter.convert(str(single), raises_on_error=False)
        if result.document is None:             # failed: write nothing, so the next run tries again
            print(f"shard {shard}: {dance} p.{page} failed, not cached", flush=True)
            continue
        items = []
        if result.document is not None:
            for item, _ in result.document.iterate_items():
                label = str(getattr(item, "label", "")).split(".")[-1].lower()
                if "table" in label or "document_index" in label:
                    items.append({"type": "table", "text": item.export_to_markdown(result.document)})
                elif getattr(item, "text", None):
                    items.append({"type": label, "text": item.text})
        target.write_text(json.dumps(items, ensure_ascii=False))
        if number % 20 == 0:
            print(f"shard {shard}: {number}/{len(plan)} pages, {time.time() - started:.0f}s", flush=True)
    print(f"shard {shard}: done {len(plan)} pages in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--plan"]:
        write_plan()
    else:
        shard, shards = (int(part) for part in sys.argv[1].split("/"))
        run_shard(shard, shards)
