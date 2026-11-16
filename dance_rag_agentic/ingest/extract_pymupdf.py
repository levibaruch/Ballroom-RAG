"""PyMuPDF extraction: chart titles + metadata, and fallback tables / prose.

bibr does not tell us which figure, role and side a table belongs to; the chart titles
found here do ("OPENING OUT (Man) - Side 2"). Results are cached per book.
"""
import re
import time
from typing import NamedTuple

import pandas as pd
import pymupdf

from dance_rag_agentic.config import PYMUPDF_CACHE_DIR

# The PDF text layer letter-spaces some titles: 'PROMENADE (M a n )-S id e 1' (Paso Doble, Samba, Slow Foxtrot).
# Shared charts: 'CORTA JACA (Man and Lady)', 'FORWARD CHANGE STEP (MAN or LADY)'.
CHART_TITLE = re.compile(r"^(.+?)\s*\(\s*(M\s*a\s*n(?:\s+(?:and|or)\s+L\s*a\s*d\s*y)?|L\s*a\s*d\s*y)\s*\)"
                         r"\s*-\s*S\s*i\s*d\s*e\s*(\d)", re.I)
# groups: 1 = figure name, 2 = role, 3 = side number
CONTINUED_PAGE = re.compile(r"^\s*Continued\s+Step", re.I)     # Jive p.122: the chart above goes on on this page


def clean_role(text):
    """'M a n ' -> 'Man'; 'MAN or LADY' -> 'Man or Lady' (a shared chart: found for both roles)."""
    text = re.sub(r"(?i)m\s*a\s*n", "Man", re.sub(r"(?i)l\s*a\s*d\s*y", "Lady", text.strip()))
    return re.sub(r"\s+(?i:and)\s+", " and ", re.sub(r"\s+(?i:or)\s+", " or ", text))

RECORD_COLUMNS = ["dance", "style", "page", "kind", "figure", "role", "side", "header", "rows", "text"]


class ChartTitle(NamedTuple):
    y_top: float      # vertical position on the page (0 = top)
    text: str         # e.g. "OPENING OUT (Man) - Side 2"
    figure: str       # e.g. "Opening Out"
    role: str         # "Man", "Lady", or a shared chart: "Man and Lady" / "Man or Lady"
    side: int         # 1 or 2


def find_chart_titles(page):
    """All chart titles on a page, sorted top to bottom."""
    titles = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):                 # image blocks have no lines
            line_text = "".join(span["text"] for span in line["spans"]).strip()
            title_match = CHART_TITLE.match(line_text)
            if title_match:
                titles.append(ChartTitle(y_top=line["bbox"][1], text=line_text,
                                         figure=title_match.group(1).strip().title(),
                                         role=clean_role(title_match.group(2)),
                                         side=int(title_match.group(3))))
    return sorted(titles, key=lambda title: title.y_top)


def read_table_cells(page, table):
    """Table as a list of rows of cell texts (row 0 = column names), read via get_textbox."""
    rows = []
    for table_row in table.rows:
        rows.append([page.get_textbox(cell_box).replace("\n", " ").strip() if cell_box else ""
                     for cell_box in table_row.cells])
    return rows


def extract_pdf_pymupdf(pdf_path):
    """One record per chart table, and one record per prose page."""
    document = pymupdf.open(pdf_path)
    dance, style = pdf_path.stem, pdf_path.parent.name
    records, last_title = [], None
    for page_index, page in enumerate(document):
        page_number = page_index + 1
        chart_titles = find_chart_titles(page)
        if not chart_titles and last_title and CONTINUED_PAGE.match(page.get_text()):
            chart_titles = [last_title._replace(y_top=0.0)]    # same figure, role and side as the page before
        if chart_titles:
            last_title = chart_titles[-1]
        if not chart_titles:
            records.append(dict(dance=dance, style=style, page=page_number, kind="prose", figure=None,
                                role=None, side=None, header=None, rows=None, text=page.get_text()))
            continue
        for table in page.find_tables().tables:
            titles_above = [title for title in chart_titles if title.y_top < table.bbox[1]]
            if not titles_above:
                continue
            title = titles_above[-1]                         # closest title above the table
            records.append(dict(dance=dance, style=style, page=page_number, kind="chart",
                                figure=title.figure, role=title.role, side=title.side,
                                header=title.text, rows=read_table_cells(page, table), text=None))
    return pd.DataFrame(records, columns=RECORD_COLUMNS)


def load_pymupdf_pages(pdf_path, rebuild=False):
    """PyMuPDF records for one book, cached in cache/pymupdf/<Dance>.pkl (~1-2 min per book)."""
    cache_file = PYMUPDF_CACHE_DIR / f"{pdf_path.stem}.pkl"
    if cache_file.exists() and not rebuild:
        return pd.read_pickle(cache_file)
    PYMUPDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    pages = extract_pdf_pymupdf(pdf_path)
    pages.to_pickle(cache_file)
    print(f"  PyMuPDF {pdf_path.stem}: {len(pages)} records in {time.time() - start:.0f}s")
    return pages
