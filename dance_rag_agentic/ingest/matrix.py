"""Precedes / Follows matrices (colour-coded, scanned image) read by code. Each matrix page is a scanned image of a
grid: one row per preceding figure, one column per following figure, each crossing a cell whose colour says:
orange = may follow with some variation, grey-mauve ('purple' in the book) = without any variation, white = not allowed.

Nothing is hard-coded per book or page; everything is derived from the page itself:
  1. GRID: the ruled lines are found in the rendered pixels (long dark runs), so rows and columns are found even
     when the text layer is missing; irregular spacing (the row-name box, the legend box) is filtered out.
  2. NAMES: row names (horizontal text left of the grid) and column names (rotated text above the grid, either
     direction) come from the PDF text layer by position. Some scans have no text layer for the names (the text is
     baked into the image): then the names come from the sibling pages of the same matrix ('Side 1/2/3' of the same
     PDF): the rows are the figure list, and the columns of the sides, in order, are consecutive slices of that same list.
     That is only accepted if the columns of all sides together add up to exactly the figure list (otherwise None).
  3. COLOURS: calibrated per page: the cell colours are clustered (white / warm / mauve) and the legend swatches,
     when readable, decide which cluster means 'with variation' and which 'without variation'.
Safety rule (matrix_text): None when the grid, the names or the colours look unreliable."""
import re

import numpy as np
import pymupdf
from sklearn.cluster import KMeans

SCALE = 3
TITLE = "PRECEDES AND FOLLOWS"
WITH_VARIATION = "with variation"
WITHOUT_VARIATION = "without variation"
NOT_ALLOWED = "not allowed"


def text_lines(page):
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = " ".join(span["text"] for span in line["spans"]).strip()
            if text:
                lines.append((tuple(round(value) for value in line["dir"]), pymupdf.Rect(line["bbox"]), text))
    return lines


def clean_name(text):
    """'1(B) Underarm ...', '| (S) Alemana ...', 'B) Natural Turn', '(G) Curl' -> '(B) ...' ; no code letter: unchanged."""
    match = re.match(r"^[^A-Za-z(]{0,3}\(?\s*([A-Z])\s*\)\s*(.*)$", text.strip())
    if match:
        return f"({match.group(1)}) {match.group(2)}".strip()
    return text.strip()


# ---------------------------------------------------------------- grid from pixels

def render(page):
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(SCALE, SCALE), colorspace=pymupdf.csRGB)
    return np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.h, pixmap.w, 3).astype(int)


def long_dark_runs(dark, axis):
    """For each column (axis=0) or row (axis=1): the longest run of dark pixels, small gaps (<=3 px) bridged."""
    if axis == 1:
        dark = dark.T
    height, width = dark.shape
    current = np.zeros(width, int)
    gap = np.zeros(width, int)
    best = np.zeros(width, int)
    for y in range(height):
        hit = dark[y]
        current = np.where(hit, current + 1, current)
        gap = np.where(hit, 0, gap + 1)
        best = np.maximum(best, current)
        current = np.where(gap > 3, 0, current)
    return best


def line_centres(flags):
    """Indices of True flags -> centres of each consecutive group (a ruled line is a few pixels thick)."""
    centres, group = [], []
    for index in np.flatnonzero(flags):
        if group and index - group[-1] > 2:
            centres.append(np.mean(group))
            group = []
        group.append(index)
    if group:
        centres.append(np.mean(group))
    return centres


def regular_chain(positions):
    """The largest set of lines with a regular spacing (a single gap 0.6-1.5 spacings), filling in lines that are missing from the scan
    (a gap of 2-3 spacings). Drops the row-name box, legend box and page frame, which are spaced differently."""
    if len(positions) < 3:
        return []
    step = np.percentile(np.diff(positions), 40)
    merged = [positions[0]]
    for position in positions[1:]:                        # one ruled line seen twice (e.g. 3 px apart)
        if position - merged[-1] < 0.4 * step:
            merged[-1] = (merged[-1] + position) / 2
        else:
            merged.append(position)
    positions = merged
    step = np.median([gap for gap in np.diff(positions) if gap < 1.4 * step])
    best = []
    for start in range(len(positions)):
        chain = [positions[start]]
        for position in positions[start + 1:]:
            gap = position - chain[-1]
            parts = 1 if 0.6 * step <= gap <= 1.5 * step else round(gap / step)     # a row may be a bit taller
            if parts >= 1 and abs(gap - parts * step) < 0.3 * step * parts and parts <= 3:
                base = chain[-1]
                chain.extend(base + gap * k / parts for k in range(1, parts + 1))
            elif gap > 3.5 * step:
                break
        if len(chain) > len(best):
            best = chain
    return best


def find_grid(rgb):
    """(column_line_x_list, row_line_y_list) in pixels, or None."""
    dark = rgb.max(axis=2) < 150
    height, width = dark.shape
    vertical = long_dark_runs(dark, 0)
    minimum_height = 0.15 * height
    column_lines = regular_chain(line_centres(vertical > minimum_height))
    if len(column_lines) < 3:
        return None
    left, right = int(column_lines[0]), int(column_lines[-1])
    horizontal = long_dark_runs(dark[:, left:right + 1], 1)      # only inside the grid's own width
    row_lines = regular_chain(line_centres(horizontal > 0.3 * (right - left)))
    if len(row_lines) < 3:
        return None
    return column_lines, row_lines


# ---------------------------------------------------------------- colours

def cell_colours(rgb, column_lines, row_lines):
    """Median colour of each cell's inner area (grid lines and a margin excluded): array rows x columns x 3."""
    result = np.zeros((len(row_lines) - 1, len(column_lines) - 1, 3))
    for row in range(len(row_lines) - 1):
        for column in range(len(column_lines) - 1):
            top, bottom = row_lines[row], row_lines[row + 1]
            left, right = column_lines[column], column_lines[column + 1]
            pad_y, pad_x = 0.28 * (bottom - top), 0.28 * (right - left)
            patch = rgb[int(top + pad_y):int(bottom - pad_y) + 1, int(left + pad_x):int(right - pad_x) + 1]
            result[row, column] = np.median(patch.reshape(-1, 3), axis=0)
    return result


def chroma(colour):
    return float(np.max(colour) - np.min(colour))


def legend_colours(page, rgb):
    """Swatch colour of the two legend entries ('May precede/follow with ...' / '... without ...'). The entry's text may
    wrap over several lines; the coloured square is either behind the text or just left of it. Empty when the legend is
    not in the text layer or not readable."""
    lines = [(rect, text) for direction, rect, text in text_lines(page) if direction == (1, 0)]
    starts = sorted((rect for rect, text in lines if text.lower().startswith("may precede")), key=lambda rect: rect.y0)
    found = {}
    for index, start in enumerate(starts):
        limit = starts[index + 1].y0 if index + 1 < len(starts) else start.y0 + 30
        entry = " ".join(text for rect, text in lines if start.y0 - 1 <= rect.y0 < limit and abs(rect.x0 - start.x0) < 60).lower()
        kind = WITHOUT_VARIATION if "without" in entry else WITH_VARIATION
        for x0, x1 in ((start.x0, start.x1), (start.x0 - 30, start.x0 - 3)):         # behind the text, then left of it
            patch = rgb[int((start.y0 + 1) * SCALE):int((start.y1 - 1) * SCALE) + 1,
                        max(0, int(x0 * SCALE)):max(1, int(x1 * SCALE))].reshape(-1, 3)
            swatch = patch[(patch.max(axis=1) < 235) & (patch.min(axis=1) > 90)]       # not white paper, not a line/letter
            if len(patch) and len(swatch) > 0.5 * len(patch):
                found.setdefault(kind, np.median(swatch, axis=0))
                break
    return found if len(found) == 2 and np.linalg.norm(found[WITH_VARIATION] - found[WITHOUT_VARIATION]) > 30 else {}


def calibrate(colours, legend):
    """Cluster centres for the three meanings, or None. White = the brightest cluster; of the coloured clusters (at
    most the two biggest) the legend swatches decide which is 'with variation'; without a legend the warmer one
    (R - B) is. Note the book's 'purple' is a grey-mauve with hardly any chroma: only the distance to white counts."""
    flat = colours.reshape(-1, 3)
    model = KMeans(n_clusters=min(4, len(flat)), n_init=10, random_state=0).fit(flat)
    sizes = np.bincount(model.labels_, minlength=len(model.cluster_centers_))
    groups = []                                           # (centre, size); merge near-identical centres
    for centre, size in sorted(zip(model.cluster_centers_, sizes), key=lambda pair: -pair[0].min()):
        for index, (other, other_size) in enumerate(groups):
            if np.linalg.norm(other - centre) < 40:
                groups[index] = ((other * other_size + centre * size) / (other_size + size), other_size + size)
                break
        else:
            groups.append((centre, size))
    white = groups[0][0]
    coloured = [(c, size) for c, size in groups[1:] if np.linalg.norm(c - white) > 45 and size >= 0.01 * len(flat)]
    coloured = sorted(coloured, key=lambda pair: -pair[1])[:2]
    if not coloured:
        return None
    coloured = [c for c, _ in sorted(coloured, key=lambda pair: pair[0][0] - pair[0][2], reverse=True)]   # warmest first
    meaning = {NOT_ALLOWED: white}
    if len(coloured) == 1:
        kind = WITH_VARIATION if coloured[0][0] - coloured[0][2] > 60 else WITHOUT_VARIATION
        meaning[kind] = coloured[0]
        return meaning
    warm, other = coloured
    meaning[WITH_VARIATION], meaning[WITHOUT_VARIATION] = warm, other
    if len(legend) == 2:
        straight = np.linalg.norm(legend[WITH_VARIATION] - warm) + np.linalg.norm(legend[WITHOUT_VARIATION] - other)
        crossed = np.linalg.norm(legend[WITH_VARIATION] - other) + np.linalg.norm(legend[WITHOUT_VARIATION] - warm)
        if crossed < straight - 30:                       # the legend says the other way round for this book
            meaning[WITH_VARIATION], meaning[WITHOUT_VARIATION] = other, warm
    return meaning


def classify_cells(colours, meaning):
    """Label per cell: nearest calibrated colour, 'unreadable' when it is far from all of them."""
    labels = np.empty(colours.shape[:2], object)
    for row in range(colours.shape[0]):
        for column in range(colours.shape[1]):
            distances = {kind: np.linalg.norm(colours[row, column] - centre) for kind, centre in meaning.items()}
            kind = min(distances, key=distances.get)
            labels[row, column] = kind if distances[kind] < 45 else "unreadable"
    return labels


# ---------------------------------------------------------------- names

def row_texts(lines, column_lines, row_lines):
    """Per row band: the horizontal text to the left of the grid (cleaned), '' when none."""
    left = column_lines[0] / SCALE
    names = [""] * (len(row_lines) - 1)
    for direction, rect, text in lines:
        if direction[1] != 0 or rect.x1 > left + 2:
            continue
        centre = (rect.y0 + rect.y1) / 2 * SCALE
        for index in range(len(names)):
            if row_lines[index] <= centre < row_lines[index + 1]:
                names[index] = (names[index] + " " + text).strip()
    return [clean_name(name) if name else "" for name in names]


def column_texts(lines, column_lines, row_lines):
    """Per column band: the rotated text (either direction) above the grid, '' when none."""
    top = row_lines[0] / SCALE
    names = [[] for _ in range(len(column_lines) - 1)]
    for direction, rect, text in lines:
        if direction[0] != 0 or rect.y1 > top + 2:
            continue
        centre = (rect.x0 + rect.x1) / 2 * SCALE
        for index in range(len(names)):
            if column_lines[index] <= centre < column_lines[index + 1]:
                names[index].append((rect.x0, text))
    return [clean_name(" ".join(text for _, text in sorted(parts))) if parts else "" for parts in names]


def matrix_pages(page):
    """All pages of the same PDF that hold this matrix, in order 'Side 1, 2, ...'."""
    document = page.parent
    pages = [document[number] for number in range(len(document)) if TITLE in document[number].get_text()]
    side = lambda one: int((re.search(r"Side\s*(\d+)", one.get_text()) or [0, 0])[1])
    return sorted(pages, key=side)


_GRID_CACHE = {}


def analyse_page(page):
    """Grid, cell colours and any names present in the text layer for one page (cached per document and page)."""
    key = (page.parent.name, page.parent.page_count, page.number)
    if key not in _GRID_CACHE:
        rgb = render(page)
        grid = find_grid(rgb)
        if grid is None:
            _GRID_CACHE[key] = None
        else:
            column_lines, row_lines = grid
            lines = text_lines(page)
            _GRID_CACHE[key] = {"rgb": rgb, "column_lines": column_lines, "row_lines": row_lines,
                                "row_names": row_texts(lines, column_lines, row_lines),
                                "column_names": column_texts(lines, column_lines, row_lines)}
    return _GRID_CACHE[key]


def figure_names(page, info):
    """(row_names, column_names) for this page, from its own text layer and, where that has gaps, from the sibling
    pages; None if unreliable. Sanity rule: every side has the same rows, and the columns of all sides together are
    exactly those rows (the matrix is figures x figures)."""
    siblings = [(one, analyse_page(one)) for one in matrix_pages(page)]
    if any(other is None for _, other in siblings):
        return None
    row_names, column_names = list(info["row_names"]), list(info["column_names"])
    if len({len(other["row_names"]) for _, other in siblings}) != 1:
        return None
    if sum(len(other["column_names"]) for _, other in siblings) != len(row_names):
        return None
    if "" in row_names:
        donors = [other["row_names"] for _, other in siblings if "" not in other["row_names"]]
        if not donors:
            return None
        row_names = list(donors[0])
    position = [one.number for one, _ in siblings].index(page.number)
    start = sum(len(other["column_names"]) for _, other in siblings[:position])
    column_names = [name or row_names[start + index] for index, name in enumerate(column_names)]
    return row_names, column_names


def read_matrix_page(page):
    """(column names, {row name: {column name: kind}}) or ([], {}) when no reliable grid/names/colours."""
    info = analyse_page(page)
    if info is None:
        return [], {}
    names = figure_names(page, info)
    if names is None:
        return [], {}
    row_names, column_names = names
    colours = cell_colours(info["rgb"], info["column_lines"], info["row_lines"])
    meaning = calibrate(colours, legend_colours(page, info["rgb"]))
    if meaning is None:
        return [], {}
    labels = classify_cells(colours, meaning)
    cells = {row: {column: labels[r, c] for c, column in enumerate(column_names)} for r, row in enumerate(row_names)}
    return column_names, cells


def matrix_text(page):
    """The matrix of one PDF page as lines 'row -> followed by: column (kind), ...', or None if no matrix."""
    if TITLE not in page.get_text():
        return None
    title = next((text for direction, rect, text in text_lines(page) if TITLE in text), TITLE)
    columns, cells = read_matrix_page(page)
    if len(columns) < 5 or len(cells) < 5:            # no grid found where expected: let the page be read as text
        return None
    values = [value for row in cells.values() for value in row.values()]
    if values.count("unreadable") > 0.05 * len(values) or WITH_VARIATION not in values or WITHOUT_VARIATION not in values:
        return None                                   # colours not calibrated for this scan: do not guess
    lines = [f"{title}: which figure may follow which (row = preceding figure, column = following figure)"]
    for row_name, row in cells.items():
        allowed = [f"{column} ({value})" for column, value in row.items() if value != NOT_ALLOWED]
        lines.append(f"{row_name} is followed by: {', '.join(allowed) or 'none'}")
    return title, "\n".join(lines)
