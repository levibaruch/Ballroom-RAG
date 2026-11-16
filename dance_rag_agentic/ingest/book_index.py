"""The book's own index (last page): topic / figure -> page, written by the authors.

Layout in the PyMuPDF text, one item per line:
    INDEX
    GENERAL PRINCIPLES        group heading (contains PRINCIPLES or FIGURES)
    Alignment                 entry name (can span several lines)
    P-                        page marker 'P.' / 'P-', sometimes 'P. 137' on one line
    23                        printed page number
    Fan                       figure entries also carry a level:
    (B)                       (B)ronze / (S)ilver / (G)old
    P-
    68
    227                       last line: the printed number of the index page itself

Printed page numbers differ from PDF page numbers (cover pages). The offset is measured per book
from that last line: offset = PDF page of the index - its printed number (+1 in every book so far).
Group headings are not trusted for the dance (the Jive index says 'CHACHACHA FIGURES').
Every entry is verified: does its name appear on the page it points to?
"""
import difflib
import re

import fitz
import pandas as pd

GROUP_HEADING = re.compile(r"PRI?N?CIPLES|FIGURES")        # 'SPECIFIC PRICIPLES' is a typo in the books
PAGE_MARKER = re.compile(r"^[Pp][.\-,]?\s*(\d+)?$")         # 'P-', 'P.', 'P. 137'
LEVEL_MARKER = re.compile(r"^\(([BSG])\)$")
NUMBER = re.compile(r"^\d+$")
NAME_JUNK = re.compile(r"\s+(p[.\-]|nr\*)$|\.$", re.I)      # 'Jive Chasse p-', 'Swing Jive Actions nr*', 'Brush.'
LEVEL_NAMES = {"B": "Bronze", "S": "Silver", "G": "Gold"}
MINIMUM_NAME_MATCH = 0.8          # share of the (normalised) name that must appear on the page


def letters_only(text):
    """Lowercase letters and digits: robust to OCR spacing ('A N ATO M Y') and punctuation."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def group_kind(heading):
    if "FIGURE" in heading:
        return "figure"
    if "SPECIFIC" in heading:
        return "specific principle"
    if "GENERAL" in heading:
        return "general principle"
    return heading.lower()


def find_index_page(document):
    """0-based page number of the last page that starts with 'INDEX', or None."""
    for page_number in range(len(document) - 1, -1, -1):
        if document[page_number].get_text().lstrip()[:5].upper() == "INDEX":
            return page_number
    return None


def column_boundary(words, page_width):
    """x between the two index columns: the first strip without any word to the right of the left
    column's page numbers ('P-  23'). A fixed width/2 fails (in Cha Cha the right column starts left of
    the page centre), and so does 'the widest empty strip' (the gap between 'P-' and '23' is wider
    than the gap between the columns)."""
    left_markers = [word[0] for word in words if PAGE_MARKER.match(word[4]) and word[0] < page_width / 2]
    if not left_markers:
        return page_width / 2
    marker_x = min(left_markers)
    left_page_numbers = [word[2] for word in words                     # right edge of '23' after 'P-'
                         if NUMBER.match(word[4]) and marker_x <= word[0] <= marker_x + 35]
    search_from = int(max(left_page_numbers, default=marker_x + 35)) + 1
    free_x = [x for x in range(search_from, int(page_width * 0.7))
              if not any(word[0] <= x <= word[2] for word in words)]
    if not free_x:
        return page_width / 2
    first_run_end = free_x[0]
    while first_run_end + 1 in free_x:
        first_run_end += 1
    return (free_x[0] + first_run_end) / 2


def index_lines(document, index_page, row_tolerance=4.0):
    """The index text as lines in reading order: left column top to bottom, then the right column,
    page by page. Plain get_text() interleaves the two columns, or lists all names before all numbers
    (Samba, Paso Doble, Viennese Waltz), so we rebuild rows from word positions.
    Each row becomes: name words, then '(B)' if present, then the page number."""
    lines = []
    for page_number in range(index_page, len(document)):
        page = document[page_number]
        # get_text("words"): tuples (x0, y0, x1, y1, word, block_no, line_no, word_no)
        words = page.get_text("words")
        boundary = column_boundary(words, page.rect.width)
        for in_column in (lambda word: word[0] < boundary, lambda word: word[0] >= boundary):
            column_words = sorted((word for word in words if in_column(word)), key=lambda word: word[1])
            rows = []
            for word in column_words:                 # same row = top within row_tolerance of the row's first word
                if rows and word[1] - rows[-1][0][1] < row_tolerance:
                    rows[-1].append(word)
                else:
                    rows.append([word])
            for row in rows:
                row_texts = [word[4] for word in sorted(row, key=lambda word: word[0])]
                name = " ".join(text for text in row_texts
                                if not (NUMBER.match(text) or PAGE_MARKER.match(text) or LEVEL_MARKER.match(text)))
                levels = [text for text in row_texts if LEVEL_MARKER.match(text)]
                numbers = [text for text in row_texts if NUMBER.match(text)]
                lines += [item for item in [name, *levels, *numbers] if item]
    return lines


def parse_index_lines(lines):
    """(entries, printed number of the index page). lines = the index text, one item per line."""
    entries, group, name_parts, level = [], None, [], None
    index_printed_page = None

    def add_entry(printed_page):
        nonlocal name_parts, level
        name = NAME_JUNK.sub("", " ".join(name_parts)).strip()
        entries.append(dict(kind=group_kind(group or ""), name=name,
                            level=LEVEL_NAMES.get(level), printed_page=printed_page))
        name_parts, level = [], None

    for line in lines:
        page_marker = PAGE_MARKER.match(line)
        if NUMBER.match(line):
            if name_parts:
                add_entry(int(line))
            else:
                index_printed_page = int(line)        # a number without a name: the page footer
        elif page_marker:
            if page_marker.group(1) and name_parts:   # 'P. 137' on one line
                add_entry(int(page_marker.group(1)))
        elif LEVEL_MARKER.match(line):
            level = LEVEL_MARKER.match(line).group(1)
        elif line.isupper() and GROUP_HEADING.search(line) and not name_parts:
            group = line
        else:
            name_parts.append(line)
    return entries, index_printed_page


def name_appears_on(name, page_text):
    """True if (almost) the whole name is on the page, ignoring spacing, case and punctuation."""
    name_letters, page_letters = letters_only(name), letters_only(page_text)
    if not name_letters:
        return False
    if name_letters in page_letters:
        return True
    longest = difflib.SequenceMatcher(None, name_letters, page_letters, autojunk=False).find_longest_match()
    return longest.size / len(name_letters) >= MINIMUM_NAME_MATCH


def read_book_index(pdf_path):
    """One row per index entry: dance, kind, name, level, printed_page, page (PDF), verified, found_on.

    page = printed_page + offset (measured from the index page's own number).
    verified = the name appears on that page; found_on = nearest page within +-3 where it does
    appear (None if nowhere near), so a wrong page number in the index is visible."""
    document = fitz.open(pdf_path)
    index_page = find_index_page(document)
    if index_page is None:
        return pd.DataFrame()
    lines = index_lines(document, index_page)
    entries, _ = parse_index_lines(lines[1:])                                    # [0] = 'INDEX'

    page_texts = {}
    def text_of(page):                                                           # 1-based PDF page
        if page not in page_texts:
            page_texts[page] = document[page - 1].get_text() if 1 <= page <= len(document) else ""
        return page_texts[page]

    # offset printed -> PDF page: measured, not assumed. The one under which most names are found
    # literally on their page (the footer number of the index page proved unreliable).
    def exact_hits(offset):
        return sum(letters_only(entry["name"]) in letters_only(text_of(entry["printed_page"] + offset))
                   for entry in entries if entry["name"])
    offset = max(range(-2, 6), key=exact_hits)

    rows = []
    for entry in entries:
        page = entry["printed_page"] + offset
        found_on = next((candidate for distance in range(4) for candidate in (page + distance, page - distance)
                         if name_appears_on(entry["name"], text_of(candidate))), None)
        rows.append(entry | dict(dance=pdf_path.stem, page=page, offset=offset,
                                 verified=found_on == page, found_on=found_on))
    columns = ["dance", "kind", "name", "level", "printed_page", "page", "offset", "verified", "found_on"]
    return pd.DataFrame(rows, columns=columns)
