"""bibr: run it once per book, and read its JSON.

Native mode is enough: bibr uses the PDF's text layer (ABBYY) for prose and OCRs tables
anyway (PaddleOCR-VL). Output: extractions/bibr/<Dance>.json (~10 min per book on an M3).
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pymupdf

from dance_rag_agentic.config import BIBR_ARGUMENTS, BIBR_DIR, BIBR_EXECUTABLE
from dance_rag_agentic.ingest.text_utils import clean_bibr_text

# bibr silently stops after 200 pages by default (PIPELINE_MAX_PAGES); our books have up to ~230.
BIBR_ENVIRONMENT = {**os.environ, "PIPELINE_MAX_PAGES": "2000"}
LAST_PAGES_ALLOWED_EMPTY = 5      # index / bibliography pages at the end may legitimately give no text


def bibr_json_path(pdf_path):
    return BIBR_DIR / f"{pdf_path.stem}.json"


def bibr_last_page(json_path):
    output = json.loads(Path(json_path).read_text())
    pages = [item["page_number"] for item in output["text"] + output["table"]]
    return max(pages) if pages else 0


def check_bibr_coverage(pdf_path, json_path):
    """Warn when bibr's output stops well before the end of the PDF (e.g. the old 200-page limit)."""
    pdf_pages = len(pymupdf.open(pdf_path))
    last_page = bibr_last_page(json_path)
    if last_page < pdf_pages - LAST_PAGES_ALLOWED_EMPTY:
        print(f"  WARNING bibr {pdf_path.stem}: output stops at page {last_page} of {pdf_pages}. "
              f"Pages after {last_page} fall back to PyMuPDF. Re-run bibr for this book "
              f"(remove or rename {json_path.name} first).")
        return False
    return True


def run_bibr(pdf_path):
    """Run bibr on one PDF unless its JSON already exists. Returns the JSON path."""
    output_path = bibr_json_path(pdf_path)
    if output_path.exists():
        check_bibr_coverage(pdf_path, output_path)
        return output_path
    if shutil.which(BIBR_EXECUTABLE) is None and not Path(BIBR_EXECUTABLE).exists():
        raise FileNotFoundError(
            f"bibr executable not found: {BIBR_EXECUTABLE!r}. Install bibr (with mlx-vlm) and set the "
            f"BIBR_EXECUTABLE environment variable, e.g. BIBR_EXECUTABLE=~/.venvs/bibr/bin/bibr")
    BIBR_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  bibr {pdf_path.stem}: running (this takes ~10 minutes)...", flush=True)
    start = time.time()
    subprocess.run([BIBR_EXECUTABLE, "chew", str(pdf_path), *BIBR_ARGUMENTS, "-o", str(output_path)],
                   check=True, capture_output=True, text=True, env=BIBR_ENVIRONMENT)
    print(f"  bibr {pdf_path.stem}: done in {time.time() - start:.0f}s")
    check_bibr_coverage(pdf_path, output_path)
    return output_path


class BibrBook:
    """One bibr output (= one book), with lookups per page."""

    def __init__(self, json_path):
        output = json.loads(Path(json_path).read_text())
        section_headers = {section["section_id"]: clean_bibr_text(section["header"])
                           for section in output["section"]}
        self.tables = {}      # {page: [table rows, ...]}            in reading order
        self.prose = {}       # {page: [(section header, text), ...]}
        self.notes = {}       # {page: [note text, ...]}             "NOTE - ..." below charts
        for table in output["table"]:
            rows = [[clean_bibr_text(cell) for cell in row] for row in table["contents"]]
            self.tables.setdefault(table["page_number"], []).append(rows)
        for paragraph in output["text"]:
            text = clean_bibr_text(paragraph["text"])
            page = paragraph["page_number"]
            if text.upper().startswith("NOTE"):
                self.notes.setdefault(page, []).append(text)
            else:
                header = section_headers.get(paragraph["section_id"], "")
                self.prose.setdefault(page, []).append((header, text))

    def all_tables(self):
        return [rows for page_tables in self.tables.values() for rows in page_tables]

    def tables_as_prose(self, page):
        """Tables on non-chart pages (e.g. hip design phases) as 'a | b | c' lines."""
        return ["\n".join(" | ".join(row) for row in rows) for rows in self.tables.get(page, [])]
