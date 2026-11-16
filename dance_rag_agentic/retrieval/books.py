"""Read-only access to the books: sections, citable blocks, figures, charts, searches and the book index.

The data comes from the built index in cache/agentic/index_v4 (`make build`, ingest/build_index.py).
This module never writes there. Everything returns plain Python data; tools.py turns it into text
for the agents, and the graph nodes use it directly.
"""
import difflib
import pickle
import re
from dataclasses import dataclass
from functools import cache

import numpy as np
from rank_bm25 import BM25Okapi

from dance_rag_agentic.config import QUERY_PREFIX, select_book_pdfs
from dance_rag_agentic.ingest.glossary import column_key
from dance_rag_agentic.retrieval.index import KnowledgeBase, embedding_model, load_knowledge_base
from dance_rag_agentic.ingest.tables import table_to_steps, step_to_text
from dance_rag_agentic.ingest.text_utils import tokenize

from dance_rag_agentic.config import CACHE_DIR, MAX_HITS, SNIPPET_CHARACTERS

ROLES = {"man": "Man", "lady": "Lady"}

# Which chart columns hold an aspect. Column names are compared squashed (lowercase letters only),
# because the extracted headers lost their spaces: 'Quantityof Turn', 'Riseand Fall'.
# Standard charts have no lead column: leading is in Rotation, marked "(Lead.)".
ASPECT_COLUMNS = {
    "lead": ["lead", "rotation"],
    "timing": ["timing"],
    "hips": ["hip"],
    "footwork": ["footplacement", "footaction", "generalaction"],
    "alignment": ["alignment"],
    "turn": ["quantityofturn", "rotation"],
    "position": ["coupleposition", "position"],
    "rise": ["rise"],
    "sway": ["sway"],
}


def squash(text):
    """Lowercase letters and digits only: 'Quantityof Turn' -> 'quantityofturn'."""
    return re.sub(r"[^a-z0-9]", "", str(text).replace("\\n", " ").lower())


@dataclass
class Library:
    knowledge_base: KnowledgeBase   # the built index (read only)
    full_text_bm25: BM25Okapi     # BM25 over the FULL chunk text (the old index only covers headers)

    @property
    def pages(self):
        return self.knowledge_base.pages     # one row per section (chart table or prose section)

    @property
    def chunks(self):
        return self.knowledge_base.chunks    # retrieval units; chunk.parent = section id

    @property
    def dances(self):
        return self.knowledge_base.dances


@cache
def library():
    """Loaded once (a few seconds)."""
    knowledge_base = load_knowledge_base()
    full_text_bm25 = BM25Okapi([tokenize(text) for text in knowledge_base.chunks.text])
    return Library(knowledge_base, full_text_bm25)


# ------------------------------------------------------------------ sections and blocks
def section_exists(section_id):
    return section_id in library().pages.index


def section_label(section_id):
    """'Rumba, page 70 | FAN (Lady) - Side 1'"""
    record = library().pages.loc[section_id]
    header = record.header if isinstance(record.header, str) else ""
    return f"{record.dance}, page {record.page}" + (f" | {header}" if header else "")


@cache
def section_blocks(section_id):
    """Citable units with abbreviations explained: prose -> one paragraph per block, chart -> ONE block for the
    whole chart side (all steps). Evidence id of block n (1-based) = 'section_id.n', so a chart side is
    'section_id.1'. One id per chart: step-level ids caused ranges, a Prec rule and step/block mix-ups
    ('Step 3' was block .4); the writer names the step in its sentence and the judge reads the whole chart."""
    record = library().pages.loc[section_id]
    glossary = library().knowledge_base.abbreviations.get(record.dance, {})
    if record.kind == "chart":
        column_names, steps = table_to_steps(record.rows)
        blocks = ["\n\n".join(step_to_text(record.dance, record.header, column_names, step_cells)
                               for step_cells in steps)]
    else:
        blocks = with_table_headers([line.strip() for line in str(record.text).split("\n") if line.strip()])
    return tuple(spell_out(block, glossary, record.dance) for block in blocks)


NOT_ABBREVIATED_COLUMNS = {"timing", "timingabc"}   # '&', 'S', 'Q', '4.1' are counts, not abbreviations

# "Timing ABC" holds one step's timing in three notations. The books name them (Quickstep p.46, Samba p.38-39):
# Timing in Slows and Quicks, Timing with numbers, Choreographic Timing. The ORDER differs per style:
# Standard 'S 1.2 1' = S/Q, numbers, choreographic; Latin '1 3 Q' = numbers, choreographic, S/Q (Jive charts
# print it as 'A / 1 : B / 1 : C / Q'). Values with another number of parts stay as printed.
TIMING_ABC_NAMES = {"Standard": ("Slow/Quick", "in numbers", "choreographic count"),
                    "Latin": ("in numbers", "choreographic count", "Slow/Quick")}


@cache
def _fallback_glossary(dance):
    """Every abbreviation of every column of this book, for columns the book has no table for
    (Squeeze, Rotation, Translation). Safe because no abbreviation means two things across columns.
    Lower-case keys ('bwd') also match capitalised ('Bwd')."""
    combined = {}
    for entries in library().knowledge_base.abbreviations.get(dance, {}).values():
        for abbreviation, meaning in entries.items():
            combined.setdefault(abbreviation, meaning)
            combined.setdefault(abbreviation[:1].upper() + abbreviation[1:], meaning)
    return combined


def _replace_tokens(value, column_glossary):
    """Each abbreviation replaced by its meaning ('DW' -> 'Diagonal to Wall'); longest first, whole tokens."""
    if not column_glossary:
        return value
    alternatives = sorted(column_glossary, key=len, reverse=True)
    pattern = re.compile(r"(?<![\w/])(" + "|".join(map(re.escape, alternatives)) + r")(?![\w/])")
    return pattern.sub(lambda match: column_glossary[match.group(1)], value)


def spell_out_timing_abc(value, dance):
    """'s 5.6 3' (Quickstep) -> 'Slow/Quick s | in numbers 5.6 | choreographic count 3'."""
    parts = value.split()
    style = next((style for style, dances in dances_by_style().items() if dance in dances), None)
    names = TIMING_ABC_NAMES.get(style)
    if len(parts) != 3 or not names:
        return value
    slow_quick = parts[names.index("Slow/Quick")]
    if not re.fullmatch(r"[SsQqa&]+", slow_quick):        # garbled extraction ('1 2 2'): leave it as printed
        return value
    return " | ".join(f"{name} {part}" for name, part in zip(names, parts))


def spell_out_compound_codes(value, column_glossary):
    """Foot actions like 'TH', 'BHB' are single-letter codes run together (the book writes 'Toe-Ball (TB)'):
    'TH' -> 'Toe-Heel'. Only when every letter is a one-letter code of this column ('LF', 'IE' stay)."""
    def compound(match):
        letters = match.group(0)
        if all(letter in column_glossary for letter in letters):
            return "-".join(column_glossary[letter] for letter in letters)
        return letters
    return re.sub(r"(?<![\w/])[A-Z]{2,3}(?![\w/])", compound, value)


def spell_out(text, glossary, dance):
    """Chart values in words: abbreviations are REPLACED by their meaning, not added to it
    ('Facing DW (Diagonal to Wall)' doubled every value: 22% of the evidence, and easy to misread).
    Uses the column's own table, else the book's combined glossary; Timing stays as printed."""
    lines = []
    for line in text.splitlines():
        column_name, separator, value = line.partition(": ")
        key = column_key(column_name) if separator else ""
        if separator and key.startswith("timingabc"):          # also 'Timing ABC(i)' (a note marker)
            line = f"{column_name}: {spell_out_timing_abc(value, dance)}"
        elif separator and key not in NOT_ABBREVIATED_COLUMNS and not key.startswith("timing") and "|" not in column_name:
            column_glossary = glossary.get(key) or _fallback_glossary(dance)
            value = _replace_tokens(value, column_glossary)
            if key == "footaction":
                value = spell_out_compound_codes(value, column_glossary)
            line = f"{column_name}: {value}"
        lines.append(line)
    return "\n".join(lines)


def is_table_row(block):
    return " | " in block


def with_table_headers(blocks):
    """Prose sections contain flattened tables: one block per row (31% of all prose blocks). A row alone
    ('4/4 | RUMBA (4th beat accented)') loses its meaning ('Time Signature | Latin'), so the answer cited
    header-to-row ranges and the judge could not confirm the claim. Every row after the first of a run of
    table rows gets that first row (the header) in front. Block count and numbering stay the same."""
    result, header = [], None
    for block in blocks:
        if not is_table_row(block):
            header = None
            result.append(block)
        elif header is None or len({cell.strip() for cell in block.split(" | ")}) == 1:
            header = block               # first row of a run, or a merged title row ('Tempo | Tempo'): new header
            result.append(block)
        else:
            result.append(f"[{header}] {block}")
    return result


def figure_chart(section_id, all_roles=False):
    """All sections of the figure chart a section belongs to: same dance, figure and role, i.e. Side 1, Side 2
    and every NOTES section (the extraction splits one chart into up to 6 sections). A section that is not part
    of a figure chart is a chart of its own: (section_id,).
    all_roles=True: the Man AND Lady charts of that figure (for a question that names no role)."""
    record = library().pages.loc[section_id]
    if not isinstance(record.figure, str) or not record.figure:
        return (int(section_id),)
    roles = [record.role] if isinstance(record.role, str) and not all_roles else ()
    section_ids = library().knowledge_base.chart_parent_ids(record.figure, roles=roles, dances=[record.dance])
    return tuple(sorted(int(found) for found in section_ids)) or (int(section_id),)


def next_section(section_id):
    """The prose section that continues this one (same book, same or next page), or None.
    The extraction splits one text into sections at every subheading: '10) Lead Hold Shaping' ends with
    '...two different types of lead:' and the types follow in the next sections. Better extraction (bibr
    document structure) would keep such text together; until then, readers follow this link."""
    pages = library().pages
    if "order" in pages.columns:                       # index v4: ids are content hashes, use the reading order
        current = pages.loc[section_id]
        same_book = pages[(pages.dance == current.dance) & (pages.order == current.order + 1)]
        candidate = int(same_book.index[0]) if len(same_book) else None
    else:
        candidate = section_id + 1
    if candidate is None or candidate not in pages.index or pages.loc[section_id, "kind"] != "prose":
        return None
    current, following = pages.loc[section_id], pages.loc[candidate]
    current_last = current.page_last if "page_last" in pages.columns else current.page
    if (following.kind == "prose" and following.dance == current.dance
            and following.page in (current_last, current_last + 1)):
        return int(candidate)
    return None


def ends_unfinished(section_id):
    """True when the last block ends with ':' (a list or explanation follows in the next section)."""
    blocks = section_blocks(section_id)
    return bool(blocks) and blocks[-1].rstrip().endswith(":")


def chart_columns(section_id):
    """Column names of a chart (without the step column); [] for prose."""
    record = library().pages.loc[section_id]
    if record.kind != "chart":
        return []
    column_names, _ = table_to_steps(record.rows)
    return [name for name in column_names[1:] if name]


# ------------------------------------------------------------------ names
def resolve_dance(name):
    """'slow waltz' -> 'Slow Waltz'; None/'' -> None; unknown -> ValueError that lists the books."""
    if not name:
        return None
    matches = difflib.get_close_matches(str(name).title(), library().dances, n=1, cutoff=0.8)
    if not matches:
        raise ValueError(f"unknown dance {name!r}; books: {', '.join(library().dances)}")
    return matches[0]


def resolve_figure(name):
    """Fuzzy figure name -> the name used in the books, or None. 'openin out' -> 'Opening Out'."""
    known_figures = library().knowledge_base.known_figures
    matches = difflib.get_close_matches(str(name).title(), known_figures, n=1, cutoff=0.75)
    return matches[0] if matches else None


def similar_figures(name, count=5):
    return difflib.get_close_matches(str(name).title(), library().knowledge_base.known_figures, n=count, cutoff=0.5)


def resolve_role(role):
    """'man' / 'Lady' / None -> 'Man' / 'Lady' / None. Anything else (e.g. 'both') -> None."""
    return ROLES.get(str(role).lower()) if role else None


# ------------------------------------------------------------------ catalog
def dances_by_style():
    """{'Latin': [...], 'Standard': [...]}, read from the knowledge base (never typed in by hand)."""
    pages = library().pages
    return {str(style).title(): sorted(group.dance.unique()) for style, group in pages.groupby("style")}


def books_sentence():
    """'Latin = Cha Cha, Jive, ...; Standard = ...'. Used in every prompt, so a new book shows up everywhere."""
    return "; ".join(f"{style} = {', '.join(dances)}" for style, dances in dances_by_style().items())


def figures_of(dance):
    return library().knowledge_base.figures_per_dance().get(dance, [])


def dances_with_figure(figure):
    return library().knowledge_base.dances_with_figure(figure)


def charts_of(figure, dance=None, role=None, aspect=None):
    """Chart sections (and the NOTES below them) of a figure, as dicts.
    role: 'Man'/'Lady'/None. aspect: a key of ASPECT_COLUMNS or None/'all'.
    Each dict: section_id, dance, page, header, kind, role, columns, has_aspect."""
    knowledge_base = library().knowledge_base
    section_ids = knowledge_base.chart_parent_ids(figure, roles=[role] if role else (),
                                                  dances=[dance] if dance else ())
    wanted_columns = ASPECT_COLUMNS.get(aspect or "", [])
    charts = []
    for section_id in section_ids:
        record = library().pages.loc[section_id]
        columns = chart_columns(section_id)
        has_aspect = (not wanted_columns or record.kind != "chart"
                      or any(wanted in squash(column) for column in columns for wanted in wanted_columns))
        charts.append(dict(section_id=int(section_id), dance=record.dance, page=int(record.page),
                           header=record.header if isinstance(record.header, str) else "",
                           kind=record.kind, role=record.role if isinstance(record.role, str) else None,
                           columns=columns, has_aspect=has_aspect))
    return charts


# ------------------------------------------------------------------ search
def _hits(scores, dance, kind=None, limit=None):
    """Best chunk per section, top MAX_HITS sections. scores: one number per chunk, higher = better.
    kind: 'prose' (explanatory text) or 'chart' (figure charts and their NOTES) keeps only that kind (None = both).
    Figure NOTES used to count as prose: 'contrary body movement' then returned 8 one-line notes ('may start in
    Contrary Body Movement Position') and no explanation."""
    chunks, pages = library().chunks, library().pages
    hits, seen_sections, same_text = [], set(), {}     # same_text: (heading, snippet) -> hit
    max_hits = limit or MAX_HITS
    for position in np.argsort(-scores):
        chunk = chunks.iloc[position]
        # Part 1 text is the same in every book of a style and indexed once: 'copies' = {dance: that book's section}
        copies = chunk.copies if isinstance(chunk.copies, dict) else None
        if dance and chunk.dance != dance and not (copies and dance in copies):
            continue
        if kind and ("chart" if isinstance(chunk.figure, str) and chunk.figure else chunk.kind) != kind:
            continue                                  # figure NOTES belong to the charts: 'prose' = explanations only
        if scores[position] <= 0:                     # BM25: no word in common -> the rest is worse
            break
        parent = copies[dance] if copies and dance in copies else chunk.parent   # the asked book's own copy
        if parent in seen_sections:
            continue
        seen_sections.add(parent)
        pointer = chunk.get("pointer")
        if isinstance(pointer, str):                  # a search-only pointer (figure card, book index) found it; show the book's
            target = pages.loc[parent]                # own text, never the generated summary (not evidence)
            target_text = (" ".join(section_blocks(parent))[:SNIPPET_CHARACTERS] if target.kind == "chart"
                           else str(target.text).replace("\n", " ")[:SNIPPET_CHARACTERS])
            hits.append(dict(section_id=int(parent), dance=target.dance, page=int(target.page), kind=target.kind,
                             header=target.header, snippet=f"(via {pointer}) {target_text}",
                             also_in=[other for other in (copies or {}) if other != target.dance]))
            if len(hits) == max_hits:
                break
            continue
        if copies:
            body = chunk.text.split("\n", 1)[-1].replace("\n", " ")
            hits.append(dict(section_id=int(parent), dance=pages.loc[parent, "dance"], page=int(pages.loc[parent, "page"]),
                             kind=chunk.kind, header=pages.loc[parent, "header"], snippet=body[:SNIPPET_CHARACTERS],
                             also_in=[other for other in copies if other != pages.loc[parent, "dance"]]))
            if len(hits) == max_hits:
                break
            continue
        body = chunk.text.split("\n", 1)[-1].replace("\n", " ").replace("\\n", " ")
        header = pages.loc[chunk.parent, "header"]
        header = header if isinstance(header, str) else "(no heading)"
        # the general chapters are the same in every book of a style ("Posture in Latin" 5 times): one hit,
        # the other books listed, so the other slots go to different texts
        text_key = (header, squash(body[:120]))
        if text_key in same_text:
            same_text[text_key]["also_in"].append(chunk.dance)
            continue
        hits.append(dict(section_id=int(chunk.parent), dance=chunk.dance, page=int(chunk.page), kind=chunk.kind,
                         header=header, snippet=body[:SNIPPET_CHARACTERS], also_in=[]))
        same_text[text_key] = hits[-1]
        if len(hits) == max_hits:
            break
    return hits


def keyword_hits(terms, dance=None):
    return _hits(library().full_text_bm25.get_scores(tokenize(terms)), dance)


RRF_K = 60      # reciprocal rank fusion constant (the usual value): a rank r counts 1 / (RRF_K + r)


def hybrid_hits(query, dance=None, kind=None, limit=None):
    """One ranking from both searches (reciprocal rank fusion): a chunk high in EITHER the word search (BM25) or
    the meaning search (embeddings) comes up. Each hit says which search found it in its top 50."""
    bm25_scores = library().full_text_bm25.get_scores(tokenize(query))
    dense_scores = library().knowledge_base.embeddings @ embedding_model().encode(QUERY_PREFIX + query)
    fused = np.zeros(len(bm25_scores))
    found_by = {}
    for name, scores in (("words", bm25_scores), ("meaning", dense_scores)):
        order = np.argsort(-scores)
        ranks = np.empty(len(scores))
        ranks[order] = np.arange(1, len(scores) + 1)
        usable = scores > 0 if name == "words" else np.ones(len(scores), bool)   # no shared word: no BM25 vote
        fused += np.where(usable, 1.0 / (RRF_K + ranks), 0.0)
        for position in order[:50]:
            if usable[position]:
                chunk = library().chunks.iloc[position]
                # Part 1 text is stored once per style; a hit is shown as the asked book's own copy, so record
                # every copy (else shared text always looked like filler found only "by rank")
                copies = chunk.copies if isinstance(chunk.get("copies"), dict) else {}
                for section_id in {int(chunk.parent), *map(int, copies.values())}:
                    found_by.setdefault(section_id, set()).add(name)
    hits = _hits(fused, dance, kind, limit)
    for hit in hits:
        hit["found_by"] = " + ".join(sorted(found_by.get(hit["section_id"], {"rank"})))
    return hits


def semantic_hits(query, dance=None):
    query_embedding = embedding_model().encode(QUERY_PREFIX + query)
    return _hits(library().knowledge_base.embeddings @ query_embedding, dance)


# ------------------------------------------------------------------ the book's own index
@cache
def book_index():
    """Index entries of all books (topic -> page), from the last page of each PDF.
    Parsed once with dance_rag.book_index, then cached in cache/agentic/book_index.pkl."""
    cache_path = CACHE_DIR / "book_index.pkl"
    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())
    import pandas as pd
    from dance_rag_agentic.ingest.book_index import read_book_index
    entries = pd.concat([read_book_index(pdf_path) for pdf_path in select_book_pdfs()], ignore_index=True)
    entries["best_page"] = entries.found_on.fillna(entries.page).astype(int)   # where the name really is
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps(entries))
    return entries


def sections_on_page(dance, page, kind=None):
    """Section ids of one book page, optionally only 'prose' or 'chart'."""
    pages = library().pages
    selection = pages[(pages.dance == dance) & (pages.page <= page) & (pages.page_last >= page)]   # units span pages
    if kind:
        selection = selection[selection.kind == kind]
    return [int(section_id) for section_id in selection.index]
