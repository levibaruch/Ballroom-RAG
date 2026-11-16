"""Builds the index (v4) straight from the extraction output, with the parser combination that won the parser
comparison on 15 hard pages (1.73 vs bibr 1.44, Docling 1.39 of 2; docs/parser-comparison.md).

    python -m dance_rag_agentic.ingest.build_index            # needs cache/agentic/docling (ingest/docling_pages.py)

Per book page:
  chart page (PyMuPDF finds a chart title) -> bibr tables, with the 2-row 'Timing A/B/C' header fix
                                             (PyMuPDF only when bibr has no usable table); NOTEs from bibr
  Precedes/Follows matrix                  -> read by code from the cell colours (ingest/matrix.py)
  prose page                               -> Docling's reading order and headings; each Docling table replaced
                                             by the bibr table sharing most words; bibr prose when Docling kept
                                             < MIN_WORD_SHARE of the page's words (it dropped Rumba p.27)
Then the book's prose is cut into HEADING UNITS (part > chapter > subheadings, packed up to UNIT_MAX_CHARACTERS),
every section gets a STABLE id derived from its content key ('Rumba|chart|Fan|Lady|1'), and the search index is
built: text chunks with the path, one chunk per chart side (abbreviations spelled out), Part 1 indexed once per
style, figure cards and book-index pointers (made by code; an LLM-written summary per unit was tried and left out:
it needs a cloud or slow local model for a +0.03 retrieval gain).
"""
import hashlib
import json
import pickle
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf

from dance_rag_agentic.ingest.bibr import BibrBook
from dance_rag_agentic.ingest.clean_text import build_vocabulary, clean
from dance_rag_agentic.config import CACHE_DIR, EMBEDDING_MODEL_NAME, all_book_pdfs
from dance_rag_agentic.ingest.extract_pymupdf import find_chart_titles, load_pymupdf_pages
from dance_rag_agentic.ingest.glossary import read_abbreviation_tables
from dance_rag_agentic.ingest.matrix import matrix_text
from dance_rag_agentic.ingest.merge import merge_chart_page
from dance_rag_agentic.ingest.text_utils import word_count

ROOT = Path(__file__).resolve().parents[2]
V4_DIR = CACHE_DIR / "index_v4"
DOCLING_DIR = CACHE_DIR / "docling"
BIBR_DIR = ROOT / "extractions" / "bibr"
MIN_WORD_SHARE = 0.8
JUNK_PAGE_WORDS = 30
UNIT_MAX_CHARACTERS, UNIT_MIN_CHARACTERS = 2_000, 300
TEXT_TYPES = {"text", "list_item", "caption", "footnote", "formula", "code", "paragraph"}
HEADING_TYPES = {"section_header", "title"}
NUMBERED_CHAPTER = re.compile(r"^\s*\d+\)\s*\S")
PART_2_SENTENCE = re.compile(r"characteri[sz]ed by certain specific principles", re.I)
INDEX_PAGE = re.compile(r"P-\s*\d")
FIGURE_LIST = re.compile(r"\b\d{1,2}\.\s+[A-Z][A-Z' /()-]{3,}\s+\d{1,3}\b")
ABBREVIATION_TITLE = re.compile(r"^Abbreviations commonly used for (.+)$", re.I)
LATEX = re.compile(r"\\(?:underline|text|textbf|textit|mathrm)\{([^{}]*)\}")


# ------------------------------------------------------------------ small helpers
def words(text):
    return re.findall(r"[a-z]{3,}", str(text).lower())


LATEX_FRACTION = re.compile(r"\\frac\{+\s*([^{}]+?)\s*\}+\{+\s*([^{}]+?)\s*\}+")


def clean_row(row):
    """'A | A | B' -> 'A | B': bibr writes a merged cell once per spanned column. A row of ONE merged cell
    ('Tempo (Bar per minute) | Tempo (Bar per minute)') stays two cells: it is the table's title row, and as a single
    cell it was no longer a table row, so 'Samba | 50-52' became the header and the unit got lost.
    OCR LaTeX fractions become plain ones: '\\frac{{1}}{{2}} Beat' -> '1/2 Beat'."""
    row = LATEX_FRACTION.sub(r"\1/\2", row)
    cells = [cell.strip() for cell in row.split(" | ")]
    if len(cells) > 1 and len(set(cells)) == 1:
        return f"{cells[0]} | {cells[0]}"
    return " | ".join(cell for position, cell in enumerate(cells) if position == 0 or cell != cells[position - 1])


def clean_heading(text):
    previous = None
    while previous != text:
        previous, text = text, LATEX.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip(" .^'*#")


def markdown_table_rows(markdown):
    """Docling's Markdown table -> 'a | b | c' rows (separator line dropped)."""
    rows = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        rows.append(clean_row(" | ".join(cell.strip() for cell in line.strip("|").split("|"))))
    return rows


def bibr_json(dance):
    """The complete bibr run when there is one (Cha Cha was re-run: the first run stopped at page 200)."""
    full = BIBR_DIR / f"{dance}.full.json"
    return full if full.exists() else BIBR_DIR / f"{dance}.json"


def stable_id(key, taken):
    """A 6-digit id from the content key: the same section gets the same id in every build."""
    number = int(hashlib.sha1(key.encode()).hexdigest()[:10], 16) % 900_000 + 100_000
    while number in taken:
        number = number % 900_000 + 100_001
    taken.add(number)
    return number


# ------------------------------------------------------------------ 1. per page: what each page holds
def prose_items(dance, page_number, page_text, bibr_book):
    """[(type, text)] for one prose page: Docling (tables swapped for bibr's) or, as fallback, bibr."""
    docling_file = DOCLING_DIR / dance / f"p{page_number:03d}.items.json"
    bibr_tables = [table.split("\n") for table in bibr_book.tables_as_prose(page_number)]
    if docling_file.exists():
        items, used = [], set()
        for item in json.loads(docling_file.read_text()):
            if item["type"] == "table":
                docling_words = set(words(item["text"]))
                best, best_share = None, 0.0
                for number, rows in enumerate(bibr_tables):
                    bibr_words = set(words(" ".join(rows)))
                    share = len(docling_words & bibr_words) / max(len(docling_words | bibr_words), 1)
                    if number not in used and share > best_share:
                        best, best_share = number, share
                if best is not None and best_share >= 0.5:
                    used.add(best)
                    items.append(("table", [clean_row(row) for row in bibr_tables[best]]))
                else:
                    items.append(("table", markdown_table_rows(item["text"])))
            elif item["type"] in HEADING_TYPES:
                items.append(("heading", clean_heading(item["text"])))
            elif item["type"] in TEXT_TYPES:
                items.append(("text", item["text"].strip()))
        kept = " ".join(text if isinstance(text, str) else " ".join(text) for _, text in items)
        share = len(set(words(kept)) & set(words(page_text))) / max(len(set(words(page_text))), 1)
        if share >= MIN_WORD_SHARE:
            return items, f"docling ({share:.0%} of the words)"
    items, last_header = [], None                      # fallback: bibr paragraphs under bibr's headings
    for header, text in bibr_book.prose.get(page_number, []):
        if header and header != last_header:
            items.append(("heading", clean_heading(header)))
            last_header = header
        items.append(("text", text))
    items += [("table", [clean_row(row) for row in rows]) for rows in bibr_tables]
    return items, "bibr (Docling missing or incomplete)"


def read_book(pdf_path):
    """Everything of one book: chart / note / matrix records and the prose item stream, in page order."""
    dance, style = pdf_path.stem, pdf_path.parent.name
    bibr_book = BibrBook(bibr_json(dance))
    pymupdf_pages = load_pymupdf_pages(pdf_path)
    document = pymupdf.open(pdf_path)
    figure_records, prose_stream, sources = [], [], Counter()
    records_per_page = dict(list(pymupdf_pages.groupby("page", sort=True)))
    for page_number in range(1, len(document) + 1):
        page = document[page_number - 1]
        page_text = page.get_text()
        page_records = records_per_page.get(page_number)
        if page_records is None:
            # PyMuPDF saw a chart title but no table (Jive p.96, Samba p.152): the page had no records and was
            # skipped, in v1 too. The titles are on the page; bibr may still have the table.
            titles = find_chart_titles(page)
            if not titles:
                if word_count(page_text) < JUNK_PAGE_WORDS:
                    continue
                page_records = pd.DataFrame([dict(dance=dance, style=style, page=page_number, kind="prose")])
            else:
                page_records = pd.DataFrame([dict(dance=dance, style=style, page=page_number, kind="chart",
                                                  figure=title.figure, role=title.role, side=title.side,
                                                  header=title.text, rows=None, text=None) for title in titles])
        charts = page_records[page_records.kind == "chart"]
        matrix = matrix_text(page)
        if matrix:
            title, text = matrix
            figure_records.append(dict(dance=dance, style=style, page=page_number, kind="prose", figure=None, role=None,
                                       side=None, header=title, rows=None, text=text, source="matrix (cell colours)",
                                       unit_role="matrix"))
            sources["matrix"] += 1
        elif len(charts):
            for record in merge_chart_page(page_number, charts, bibr_book):
                if record["kind"] == "chart" and not record.get("rows"):
                    continue                              # no table from bibr nor PyMuPDF
                record["unit_role"] = "chart" if record["kind"] == "chart" else "note"
                figure_records.append(record)
                sources[record["source"].split(" ")[0]] += 1
        elif word_count(page_text) >= JUNK_PAGE_WORDS:
            items, source = prose_items(dance, page_number, page_text, bibr_book)
            prose_stream += [(page_number, kind, text) for kind, text in items]
            sources[source.split(" ")[0]] += 1
    abbreviations = read_abbreviation_tables(bibr_book.all_tables())
    return dict(dance=dance, style=style, figure_records=figure_records, prose_stream=prose_stream,
                abbreviations=abbreviations, sources=sources)


# ------------------------------------------------------------------ 2. structure: parts, chapters, units
def structure(book):
    """Heading units of one book's prose: [{part, title, headings, lines, pages, role}]."""
    stream = book["prose_stream"]
    headings_in_part_2, in_part_2 = Counter(), False
    for _, kind, text in stream:                      # part-2 headings that repeat are subheadings
        if kind == "text" and PART_2_SENTENCE.search(text):
            in_part_2 = True
        if kind == "heading":
            in_part_2 = in_part_2 or "PART 2" in text.upper()
            if in_part_2:
                headings_in_part_2[text.lower()] += 1

    nodes = []                                        # chapters: {part, title, own_lines, pages, children}
    part, part_number, seen_numbered = "Front matter", 0, False
    chapter = subheading = None

    def new_chapter(title, page):
        nonlocal chapter, subheading
        chapter = dict(part=part, title=title, lines=[], pages=[page], children=[])
        subheading = None
        nodes.append(chapter)

    for page, kind, content in stream:
        text = content if isinstance(content, str) else " ".join(content)
        if kind == "text" and len(INDEX_PAGE.findall(text)) >= 5 and part != "Book index":
            part = "Book index"
            new_chapter("Book index", page)
        starts_part_2 = part_number < 2 and part_number >= 1 and (
            (kind == "heading" and ("PART 2" in content.upper() or content.upper() == "INTRODUCTION"))
            or (kind == "text" and PART_2_SENTENCE.search(text)))
        starts_part_1 = part_number < 1 and kind == "heading" and (
            "PART 1" in content.upper() or content.upper() == "INTRODUCTION" or NUMBERED_CHAPTER.match(content))
        if starts_part_2:
            part, part_number = f"Part 2 - {book['dance']} specific principles", 2
            new_chapter(part, page)
            chapter["role"] = "part_intro"
            if kind == "heading":
                continue
        elif starts_part_1:
            part, part_number = "Part 1 - general principles", 1
            if not NUMBERED_CHAPTER.match(content):
                new_chapter(part, page)
                chapter["role"] = "part_intro"
                continue
        if kind == "heading" and part != "Book index":
            same_as = lambda node: node is not None and node["title"].lower() == content.lower()
            if same_as(chapter) or same_as(subheading):    # a heading repeated on the next page: same chapter
                continue
            numbered = bool(NUMBERED_CHAPTER.match(content))
            if part_number == 2:
                is_sub = headings_in_part_2[content.lower()] >= 2 and chapter is not None and chapter.get("role") != "part_intro"
            elif part_number == 1:
                is_sub = not numbered and seen_numbered and chapter is not None
                seen_numbered = seen_numbered or numbered
            else:
                is_sub = False
            if is_sub:
                subheading = dict(title=content, lines=[], pages=[page])
                chapter["children"].append(subheading)
            else:
                new_chapter(content, page)
            continue
        if chapter is None:
            new_chapter(part, page)
        target = subheading or chapter
        target["pages"].append(page)
        target["lines"] += content if kind == "table" else [content]

    units = []
    for node in nodes:
        groups, current, size = [], [node], len(" ".join(node["lines"]))
        for child in node["children"]:                # pack subheadings with their chapter
            child_size = len(" ".join(child["lines"])) + len(child["title"])
            if current and size + child_size > UNIT_MAX_CHARACTERS and size >= UNIT_MIN_CHARACTERS:
                groups.append(current)
                current, size = [], 0
            current.append(child)
            size += child_size
        groups.append(current)
        for group in groups:
            title = node["title"] if group[0] is node else (
                f"{node['title']} › {group[0]['title']}" + (f" … {group[-1]['title']}" if len(group) > 1 else ""))
            lines = []
            for member in group:
                if member is not node:
                    lines.append(member["title"])
                lines += member["lines"]
            pages = [page for member in group for page in member["pages"]]
            units.append(dict(part=node["part"], title=title, headings=[member["title"] for member in group],
                              lines=lines, pages=pages, role=node.get("role", "unit")))
    return units


HEADING_LINE_WORDS = 10


def is_heading_line(line):
    """A short title line: '1 Technical Sway', 'Foot Placement'. Not a sentence, not a table row, not an intro that
    announces a list ('... three different types of Sway:')."""
    text = line.strip()
    return (bool(text) and " | " not in text and len(text.split()) <= HEADING_LINE_WORDS
            and not text.endswith((".", "!", "?", ":", ";", ",")))


def merge_heading_lines(lines):
    """A heading line and the paragraph under it become ONE citable block: '1 Technical Sway: This is the basic form
    of Sway ...'. As two blocks, the writer cited only the description, the judge could not see the name, and the
    repair dropped 'Technical / Releasing / Cosmetic Sway' from the answer. Headings above a table or another
    heading stay on their own line (the table keeps its title row)."""
    merged, index = [], 0
    while index < len(lines):
        line = lines[index]
        following = lines[index + 1] if index + 1 < len(lines) else None
        if (following is not None and is_heading_line(line) and " | " not in following
                and not is_heading_line(following)):
            merged.append(f"{line.strip()}: {following.strip()}")
            index += 2
            continue
        merged.append(line)
        index += 1
    return merged


def split_abbreviations(unit):
    """Abbreviation tables become their own units ('Abbreviations commonly used for Foot Placement')."""
    normal, tables, current = [], [], None
    for line in unit["lines"]:
        title = ABBREVIATION_TITLE.match(line.split(" | ")[0].strip())
        if title:
            current = dict(unit, title=f"Abbreviations commonly used for {title.group(1).strip()}",
                           headings=[], lines=[line], role="abbreviations")
            tables.append(current)
        elif current is not None and " | " in line:
            current["lines"].append(line)
        else:
            current = None
            normal.append(line)
    return ([dict(unit, lines=normal)] if normal else []) + tables


# ------------------------------------------------------------------ 3. sections with stable ids
def clean_extraction_text(pages):
    """Undo letter-spacing and glued words from the PDF extraction in every text field (ingest/clean_text.py):
    'm ethod o f com m unication' -> 'method of communication'. The vocabulary comes from the books themselves.
    The content keys (and so the section ids) are made before this and do not change."""
    texts = list(pages.text.dropna()) + [" ".join(map(str, row)) for rows in pages.rows.dropna() for row in rows]
    vocabulary = build_vocabulary(texts)
    for column in ("text", "header", "path", "headings"):
        if column in pages.columns:
            pages[column] = [clean(value, vocabulary) if isinstance(value, str) else value for value in pages[column]]
    pages["rows"] = [[[clean(str(cell), vocabulary) for cell in row] for row in rows] if isinstance(rows, list) else rows
                     for rows in pages.rows]


def build_pages(books):
    rows, ids, taken = [], [], set()
    for book in books:
        dance, style = book["dance"], book["style"]
        seen = Counter()
        for record in book["figure_records"]:
            if record["unit_role"] == "matrix":
                key = f"{dance}|matrix|{record['header']}"
            elif record["unit_role"] == "chart":
                key = f"{dance}|chart|{record['figure']}|{record['role']}|side {record['side']}"
            else:
                key = f"{dance}|note|{record['figure']}|{record['role']}"
            seen[key] += 1
            key += f"|{seen[key]}" if seen[key] > 1 or record["unit_role"] == "note" else ""
            path = ("figure charts › Precedes and Follows" if record["unit_role"] == "matrix"
                    else f"figure charts › {record['header']}")
            rows.append(dict(record, key=key, path=path, part="figure charts", page_last=record["page"],
                             headings=record["header"], searchable=True))
            ids.append(stable_id(key, taken))
        for unit in [piece for unit in structure(book) for piece in split_abbreviations(unit)]:
            path = " › ".join(dict.fromkeys([unit["part"], unit["title"]]))
            key = f"{dance}|{path}"
            seen[key] += 1
            key += f"|{seen[key]}" if seen[key] > 1 else ""
            text = "\n".join(merge_heading_lines([line for line in unit["lines"] if line.strip()]))
            if not text.strip():
                continue
            flat = re.sub(r"\s*\|\s*", " ", text)              # Docling writes contents lists as a table
            role = "figure list" if len(FIGURE_LIST.findall(flat)) >= 5 else unit["role"]
            rows.append(dict(dance=dance, style=style, page=min(unit["pages"]), page_last=max(unit["pages"]),
                             kind="prose", figure=None, role=None, side=None, header=unit["title"], rows=None,
                             text=text, source="v4 unit", key=key, path=path, part=unit["part"],
                             headings="; ".join(unit["headings"]), unit_role=role,
                             searchable=unit["part"] not in ("Front matter", "Book index") and role != "figure list"))
            ids.append(stable_id(key, taken))
    pages = pd.DataFrame(rows, index=ids)
    clean_extraction_text(pages)
    # the references at the back (and the book's own index printed with them) answer no technique question;
    # they came up as the 2nd search hit for "music tempo paso doble" and were read as evidence (14,759 characters)
    pages.loc[pages.path.fillna("").str.contains("BIBLIOGRAPHY", case=False), "searchable"] = False
    # reading order per book: ids are content hashes, so 'id + 1' no longer means 'next section'
    pages["order"] = list(range(len(pages)))           # rows were added book by book, figures then units ...
    pages = pages.sort_values(["dance", "page", "order"])
    pages["order"] = pages.groupby("dance").cumcount()  # ... so sort by page, then number them per book
    pages["copies"], pages["copy_of"] = None, None
    part_1 = pages[(pages.kind == "prose") & (pages.part == "Part 1 - general principles")]
    groups = {}
    for section_id, record in part_1.sort_values("dance").iterrows():
        groups.setdefault((record["style"], re.sub(r"[^a-z0-9]", "", record.text.lower())[:400]), []).append(section_id)
    for section_ids in groups.values():
        if len(section_ids) > 1:
            pages.at[section_ids[0], "copies"] = {pages.loc[i, "dance"]: int(i) for i in section_ids}
            for section_id in section_ids[1:]:
                pages.at[section_id, "copy_of"] = int(section_ids[0])
                pages.at[section_id, "searchable"] = False
    return pages


# ------------------------------------------------------------------ 4. search chunks (run with the v4 index active)
def build_chunks():
    """Second phase, in a process with DANCE_RAG_AGENTIC_INDEX=v4: books.* now reads the v4 sections."""
    from dance_rag_agentic.retrieval import books
    from dance_rag_agentic.retrieval.index import embedding_model
    pages = books.library().pages
    chunks = []

    def add(section_id, record, text, kind, pointer=None, figure=None, role=None):
        chunks.append(dict(parent=int(section_id), dance=record.dance, style=record["style"], page=int(record.page),
                           kind=kind, figure=figure, role=role, source=record.source, header=text.split("\n")[0],
                           step=None, text=text, copies=record.copies if isinstance(record.copies, dict) else None,
                           pointer=pointer))

    for section_id, record in pages.iterrows():
        if not record.searchable:
            continue
        figure = record.figure if isinstance(record.figure, str) else None
        if record.kind == "chart" or figure:
            body = "\n".join(books.section_blocks(section_id))
        else:
            body = books.spell_out(record.text, books.library().knowledge_base.abbreviations.get(record.dance, {}),
                                   record.dance)
        where = (f"{record['style']} books (same text in {', '.join(sorted(record.copies))})"
                 if isinstance(record.copies, dict) else record.dance)
        prefix = f"{where} › {record.path}"
        tokens = body.split()
        for start in range(0, max(len(tokens) - 40, 1), 160):
            add(section_id, record, f"{prefix}\n{' '.join(tokens[start:start + 200])}", record.kind, figure=figure,
                role=record.role if isinstance(record.role, str) else None)

    charts = pages[pages.kind == "chart"]              # figure cards (search-only pointers)
    for (dance, figure, role), group in charts.groupby(["dance", "figure", "role"], dropna=False):
        group = group.sort_values("side")
        first = group.iloc[0]
        steps = books.section_blocks(int(group.index[0]))[0].split("\n\n")
        values = lambda column: [line.split(": ", 1)[1] for step in steps for line in step.split("\n")
                                 if line.lower().startswith(column) and ": " in line]
        timing = " ".join(values("timing")) or ""
        role_text = f" ({role})" if isinstance(role, str) else ""
        text = (f"{dance} › figure card › {figure}{role_text}\nHow to dance the {figure} in {dance}{role_text}: "
                f"{len(steps)} steps" + (f", timing {timing}" if timing else "")
                + (f", actions {', '.join(dict.fromkeys(values('general action')))}" if values("general action") else ""))
        add(int(group.index[0]), first, text, "chart", pointer="figure card", figure=figure,
            role=role if isinstance(role, str) else None)

    entries = books.book_index()                        # book-index terms -> the unit holding their page
    entries = entries[entries.kind != "figure"]
    units = pages[(pages.kind == "prose") & pages.figure.isna()]
    for entry in entries.itertuples():
        holders = units[(units.dance == entry.dance) & (units.page <= entry.best_page) & (units.page_last >= entry.best_page)]
        if len(holders) > 1:
            holders = holders[holders.text.str.lower().str.contains(entry.name.lower()[:12], regex=False)]
        if not len(holders):
            continue
        target_id = int(holders.index[0])
        target = pages.loc[target_id]
        if not target.searchable and isinstance(target.copy_of, (int, float)) and target.copy_of == target.copy_of:
            target_id, target = int(target.copy_of), pages.loc[int(target.copy_of)]
        add(target_id, target, f"{entry.dance} › book index › {entry.name}\n{entry.name} is explained in {target.path}",
            "prose", pointer="book index")

    chunk_frame = pd.DataFrame(chunks)
    embeddings = embedding_model().encode(list(chunk_frame.text), batch_size=32, show_progress_bar=False)
    chunk_frame.to_pickle(V4_DIR / "chunks.pkl")
    np.save(V4_DIR / "embeddings.npy", embeddings)
    print(f"{len(chunk_frame)} chunks ({chunk_frame.pointer.fillna('text').value_counts().to_dict()})")


def main():
    start = time.time()
    books_read = []
    ready, waiting = [], []
    for pdf_path in all_book_pdfs():                      # a book needs its bibr run before it can be built
        (ready if bibr_json(pdf_path.stem).exists() else waiting).append(pdf_path)
    for pdf_path in waiting:
        print(f"  SKIPPED {pdf_path.stem}: not extracted yet (needs extractions/bibr/{pdf_path.stem}.json and Docling). "
              f"Run: make extract && make docling && make build")
    for pdf_path in ready:
        book = read_book(pdf_path)
        books_read.append(book)
        print(f"{book['dance']:15} sources {dict(book['sources'])}", flush=True)
    pages = build_pages(books_read)
    abbreviations = {book["dance"]: book["abbreviations"] for book in books_read}
    V4_DIR.mkdir(parents=True, exist_ok=True)
    pages.to_pickle(V4_DIR / "pages.pkl")
    (V4_DIR / "abbreviations.pkl").write_bytes(pickle.dumps(abbreviations))
    placeholder = pd.DataFrame([dict(parent=int(pages.index[0]), dance=pages.iloc[0].dance, page=1, kind="prose",
                                     figure=None, role=None, source="", header="placeholder", step=None,
                                     text="placeholder", copies=None, pointer=None, style="")])
    placeholder.to_pickle(V4_DIR / "chunks.pkl")
    np.save(V4_DIR / "embeddings.npy", np.zeros((1, 384), dtype=np.float32))
    (V4_DIR / "build_info.json").write_text(json.dumps(
        dict(embedding_model=EMBEDDING_MODEL_NAME, built_from="bibr (+ .full.json) + PyMuPDF + Docling + matrix code",
             built_at=time.strftime("%Y-%m-%d %H:%M")), indent=2))
    prose = pages[pages.kind == "prose"]
    print(f"{len(pages)} sections: {(pages.kind == 'chart').sum()} chart sides, {len(prose)} prose "
          f"(median {int(prose.text.str.len().median())} chars), {prose.copy_of.notna().sum()} Part 1 copies, "
          f"roles {prose.unit_role.value_counts().to_dict()}")
    subprocess.run([sys.executable, "-c", "from dance_rag_agentic.ingest.build_index import build_chunks; build_chunks()"],
                   check=True, cwd=ROOT)
    print(f"saved {V4_DIR} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
