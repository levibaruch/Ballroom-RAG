# Parser comparison: how the books are read

The WDSF technique books are hard PDFs to read. Every heading uses the same font and size, there are no bookmarks,
the figure charts have merged cells and two-row headers, and the Precedes/Follows matrices are scanned images
that carry meaning only through colour. Before the final index was built, six ways of parsing were compared
on the hardest pages. The index uses the combination that won.

Everything ran locally. No cloud model was used to parse or score.

## Method

- **15 pages from 7 books**, picked so each known problem shows up at least once:
  - a lost "Part 2" heading
  - text that continues under a subheading
  - prose with tables
  - normal charts
  - charts with a two-row "Timing A / B / C" header
  - a colour matrix
  - a page with a diagram
  - a contents list
- **A reference per page**, written from the page image.
- **Six criteria, each scored 0–2**: text complete, headings (found + level), tables (rows, columns, merged cells),
  chart identity (figure / role / side), clean text, and colour/images (matrix and diagram pages only).
- **Reviewers**: LLM subagents compared each candidate's output with the reference and the page image. The
  candidates had neutral names where possible.

## Candidates

| | Candidate | What it is |
|---|---|---|
| A | bibr + PyMuPDF | ABBYY text layer + PaddleOCR-VL tables (MLX), PyMuPDF for chart titles and as fallback (the original pipeline) |
| A+ | A + table fix | the same output, but the table parser accepts the two-row header instead of falling back to PyMuPDF |
| B | pymupdf4llm | PyMuPDF to Markdown |
| C | Docling | layout model + TableFormer |
| D | Marker | layout + OCR (surya) |
| E | local vision model | page image to Markdown (gemma4:31b) |
| F | matrix reader in code | grid from the text positions of the row and column names, colour sampled at each crossing |

## Round 1: single parsers (mean 0–2, 3 reviewers)

| Candidate | Text complete | Headings | Tables | Chart identity | Clean text | Colour / images | **Mean** |
|---|---|---|---|---|---|---|---|
| A current | 1.60 | 0.93 | 1.14 | 2.00 | 0.93 | 0.00 | **1.07** |
| A+ table fix | **1.80** | 0.93 | **1.50** | 2.00 | 1.20 | 0.00 | **1.23** |
| B pymupdf4llm | 1.20 | 0.87 | 0.14 | 1.20 | 0.00 | 0.00 | **0.53** |
| C Docling | 1.73 | **1.27** | 1.21 | 2.00 | **1.53** | 0.11 | **1.32** |
| D Marker | 1.13 | 1.07 | 0.64 | 1.40 | 0.93 | 0.33 | **0.90** |
| F matrix in code (matrix page only) | 2.00 | 2.00 | 2.00 | – | 2.00 | **2.00** | **2.00** |

E (the vision model) produced no page in 25 minutes, so it isn't workable for ~1,600 pages and was dropped.

What the reviewers saw:
- **Docling** had the best headings and the cleanest text. But it gives every heading the same level, and it
  dropped most of one prose page.
- **bibr (A/A+)** had the best tables. A+ was the only candidate that read the two-row "Timing A/B/C" charts right.
- **The table fix (A+) repaired a real bug.** The old table check rejected bibr's correct tables whenever the second
  header row repeated a column name. As a result, 46–52% of the charts in four books came from the PyMuPDF fallback,
  with shifted and garbled cells.
- **Marker** fused table headers and put LaTeX inside tables. **pymupdf4llm** shredded and duplicated tables.
- **No text parser reads colour.** Only the code reader (F) gets the matrices.

## Round 2: the combination (mean 0–2, side by side)

**G** chooses a source for each page:
- **Chart pages:** bibr with the table fix.
- **Colour matrix pages:** the code reader.
- **Prose pages:** Docling's reading order and headings, with heading levels taken from the books' numbering
  pattern. Each Docling table is replaced by the bibr table that shares the most words with it. If Docling keeps
  less than 80% of a page's words, the page falls back to bibr.

| Candidate | Text complete | Headings | Tables | Chart identity | Clean text | Colour / images | **Mean** |
|---|---|---|---|---|---|---|---|
| **G combined** | 1.93 | 1.47 | 1.79 | 2.00 | 1.73 | 1.00 | **1.73** |
| A+ table fix | 1.80 | 1.13 | 1.43 | 2.00 | 1.33 | 0.50 | **1.44** |
| C Docling | 1.73 | 1.13 | 1.21 | 1.80 | 1.47 | 0.50 | **1.39** |

Cost to build all 9 books: 1,607 pages, of which 525 prose pages go through Docling. That takes about 52 minutes
locally, once, and is cached.

## End to end: the new index vs the old ones

29 gold questions, run with the same models. Scoring was done by code, plus a blind review of each answer for
correctness, completeness and learner quality (1–5 each, max 15). The answers were shuffled per question so the
reviewer couldn't tell which index produced them.

| | v1 (original) | v3 (regrouped + enriched) | **v4 (combination G)** |
|---|---|---|---|
| Cites a gold page | 23/27 | 27/27 | **27/27** |
| Unsupported claims left | 1 | 2 | **0** |
| Blind review score (of 15) | 11.28 | 11.10 | **11.45** |
| Clearly best answer | 8× | 6× | **10×** (5 ties) |
| Cost for 29 questions | $0.28 | $0.26 | **$0.25** |

v4 has been the only index since then. The gain is modest, from one run per question. The remaining problems are
in the writing stage and are the same for every index (see `limitations.md`).

## Colour matrix reader

The reader was calibrated on one matrix page, then made generic: it finds the grid from the page's own text
positions and calibrates the colours per page. With that version the index accepts 22 matrices across the 9 books.
It uses a matrix only when the result is clean: at most 5% of cells unreadable, and both colours present.
Other matrix pages stay text, which means headers only. One matrix was checked cell by cell (~80 cells, all
right); the others are not verified by hand.

Code: `dance_rag_agentic/ingest/matrix.py`, `ingest/tables.py` (the two-row header fix), and `ingest/build_index.py`
(choosing a source per page).
