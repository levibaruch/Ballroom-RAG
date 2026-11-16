"""Evidence: the book blocks the answer may cite, and everything about evidence ids.

An evidence id is 'section_id.block', e.g. [2705.2] = block 2 of section 2705.
Prose block = one paragraph, chart block = one step. Code (not the LLM) turns ids back into
the book's text and page, so the answer cannot misquote the book.
"""
import re
from dataclasses import dataclass, field

from dance_rag_agentic.agent.prompts import (FEEDBACK_ABSENCE_CLAIM, FEEDBACK_ID_NOT_READ, FEEDBACK_IDS_ALONE, FEEDBACK_NO_IDS,
                      FEEDBACK_NO_SUCH_BLOCK, FEEDBACK_RANGE, FEEDBACK_UNCITED, FEEDBACK_UNREADABLE_IDS)
from dance_rag_agentic.retrieval.books import figure_chart, is_table_row, library, section_blocks, section_exists, section_label

# ------------------------------------------------------------------ the evidence collected in one run
@dataclass
class EvidenceLog:
    """Sections handed to the answer writer, in the order they were found. Whole sections, all blocks."""
    section_ids: list = field(default_factory=list)
    found_by: dict = field(default_factory=dict)        # {section_id: 'FigurePath' / 'read_section' / ...}

    def add(self, section_id, found_by):
        """True if new."""
        section_id = int(section_id)
        if section_id in self.found_by or not section_exists(section_id):
            return False
        self.section_ids.append(section_id)
        self.found_by[section_id] = found_by
        return True

    def __contains__(self, section_id):
        return section_id in self.found_by

    def __len__(self):
        return len(self.section_ids)

    def characters(self):
        return sum(len(block) for section_id in self.section_ids for block in section_blocks(section_id))

    def as_text(self, section_ids=None):
        """What the answer writer and the decision model see: per section a '[Dance, page N] header' line, then '[id.n] block'."""
        lines = []
        for section_id in section_ids or self.section_ids:
            lines.append(f"=== [{section_label(section_id)}]")
            lines += [f"[{section_id}.{number}] {block}"
                      for number, block in enumerate(section_blocks(section_id), start=1)]
        return "\n".join(lines)


# ------------------------------------------------------------------ reading ids in an answer
BETWEEN_BRACKETS_RANGE = re.compile(r"\[(\d+\.\d+)\]\s*(?:[–—-]|to)\s*\[(\d+\.\d+)\]")   # [a.b]–[c.d]
EVIDENCE_BRACKET = re.compile(r"\[(\d+\.\d+[\d\s.,;–—-]*)\]")                            # [...] starting with a.b
ID_OR_RANGE = re.compile(r"(\d+)\.(\d+)(?:\s*[–—-]\s*(?:(\d+)\.)?(\d+))?")
# groups: start section, start block, end section (optional), end block (optional)
DASH_VARIANTS = str.maketrans({dash: "-" for dash in "‐‑‒–—−"})   # gpt-oss writes U+2011 / U+2212 for '-'


def normalize_ranges(answer):
    """All dash variants -> '-', then '[236.1]–[236.5]' -> '[236.1-236.5]': every reference is one bracket."""
    return BETWEEN_BRACKETS_RANGE.sub(r"[\1-\2]", answer.translate(DASH_VARIANTS))


def expand_reference(bracket_content):
    """'236.1-236.5' / '236.1-5' / '236.1, 236.3' -> ([(236, 1), ...], [problem, ...])."""
    ids, problems = [], []
    for item in re.split(r"[,;]", bracket_content):
        item_match = ID_OR_RANGE.fullmatch(item.strip())
        if not item_match:
            problems.append(f"cannot read evidence id '{item.strip()}'")
            continue
        start_section, start_block, end_section, end_block = item_match.groups()
        start_section, start_block = int(start_section), int(start_block)
        if end_block is None:
            ids.append((start_section, start_block))
        elif end_section is not None and int(end_section) != start_section:
            problems.append(f"range '{item.strip()}' spans two sections; cite each section separately")
        elif int(end_block) < start_block:
            problems.append(f"range '{item.strip()}' goes backwards")
        else:
            ids += [(start_section, block) for block in range(start_block, int(end_block) + 1)]
    return ids, problems


def parse_evidence(answer):
    """[(bracket text, [(section_id, block), ...]), ...] for every reference in the answer, + problems."""
    references, problems = [], []
    for bracket_match in EVIDENCE_BRACKET.finditer(normalize_ranges(answer)):
        ids, item_problems = expand_reference(bracket_match.group(1))
        references.append((bracket_match.group(0), ids))
        problems += item_problems
    return references, problems


def cited_ids(answer):
    references, _ = parse_evidence(answer)
    return {evidence_id for _, ids in references for evidence_id in ids}


def block_text(evidence_id):
    section_id, block_number = evidence_id
    blocks = section_blocks(section_id) if section_exists(section_id) else ()
    return blocks[block_number - 1] if 1 <= block_number <= len(blocks) else None


# ------------------------------------------------------------------ CheckCites
def uncited_sentences(answer):
    """Sentences that state a checkable fact (they contain a number: steps, counts, turns, timings) but carry
    no id. Linking text and summaries ('These tables show the differences') pass: flagging them made every
    rewrite worse. Checked per SENTENCE, not per line: in
    'No, you cannot X. It is Y [1.2].' the first sentence has no id although its line does.
    Skipped: short ones ('No.'), headings ('...:', '#'), table lines, 'Not found in the books'."""
    found = []
    for line in normalize_ranges(answer).split("\n"):
        if line.strip().startswith("|"):
            continue
        text, uncited, position = line.replace("**", ""), [], 0
        for bracket_match in EVIDENCE_BRACKET.finditer(text):
            uncited += SENTENCE_END.split(text[position:bracket_match.start()])[:-1]   # before the cited sentence
            position = bracket_match.end()
        uncited += [part for part in SENTENCE_END.split(text[position:])]             # after the last id
        stripped_line = text.strip().strip("*").strip()
        line_is_title = (not EVIDENCE_BRACKET.search(text) and not stripped_line.endswith((".", "!", "?"))
                         and len(WORD_OR_NUMBER.findall(stripped_line)) < MAX_TITLE_WORDS)
        for sentence in uncited:
            sentence = sentence.strip().strip(" -*•.,;")
            is_heading = line_is_title or sentence.endswith(":") or sentence.startswith("#")
            states_checkable_fact = bool(re.search(r"\d", sentence))   # a number, count, step, turn or timing
            if (len(WORD_OR_NUMBER.findall(sentence)) >= MIN_UNCITED_WORDS and not is_heading
                    and states_checkable_fact and "not found in the books" not in sentence.lower()):
                found.append(sentence)
    return found


def replace_claim(answer, claim_text, replacement):
    """Swap one claim and the id bracket(s) right after it for a replacement ('' removes it).
    Returns (new answer, True) or (answer, False) when the claim text is not found."""
    text = normalize_ranges(answer).replace("**", "")
    start = text.find(claim_text)
    if start < 0:
        return text, False
    after_claim = re.compile(r"[ \t.]*(\[[^\]]*\][ \t.]*)+").match(text, start + len(claim_text))
    end = after_claim.end() if after_claim else start + len(claim_text)
    separator = " " if replacement and text[end:end + 1] not in ("", "\n") else ""
    new_text = text[:start] + replacement + separator + text[end:]
    return re.sub(r"[ \t]{2,}", " ", new_text).strip(), True


LENTICULAR_CITATION = re.compile(r"【([^】]*)】")      # gpt-oss-120b cites its own way: 【183.1】, 【1†L1-L3】


def normalize_brackets(answer):
    """【183.1】 -> [183.1]; a '†...' part (gpt-oss line markers) is dropped. Seen: a gpt-oss-120b answer with only
    【】 ids failed CheckCites three times ("no evidence ids") although every fact had its id."""
    return LENTICULAR_CITATION.sub(lambda match: "[" + match.group(1).split("†")[0].strip() + "]", answer)


REPEAT_OVERLAP = 0.6       # share of words two answer lines have in common to count as the same fact
REPEAT_OVERLAP_OTHER_PART = 0.9   # across parts a fact may come back as a pointer; only a near-copy is dropped


def drop_repeated_lines(answer):
    """Lines that repeat an earlier line are dropped (ids ignored): within one part from REPEAT_OVERLAP of the
    shorter line's words, across parts only near-copies (REPEAT_OVERLAP_OTHER_PART). Seen: one fact three times;
    and the blind review: the same source sentence in all three parts. Headings are never dropped.
    Returns (answer, [dropped lines])."""
    kept, dropped, this_part, earlier_parts = [], [], [], []
    for line in answer.split("\n"):
        if is_heading(line):
            earlier_parts += this_part
            this_part = []
        text = EVIDENCE_BRACKET.sub("", line)
        words = set(re.findall(r"[a-z]{3,}", text.lower()))
        details = line_details(text)
        if EVIDENCE_BRACKET.search(line) and len(words) >= 5:
            def repeats(earlier, threshold):
                earlier_words, earlier_details = earlier
                return (len(words & earlier_words) / min(len(words), len(earlier_words)) >= threshold
                        and details <= earlier_details)     # a new number, count, role or side is a new fact
            if (any(repeats(earlier, REPEAT_OVERLAP) for earlier in this_part)
                    or any(repeats(earlier, REPEAT_OVERLAP_OTHER_PART) for earlier in earlier_parts)):
                dropped.append(line.strip())
                continue
            this_part.append((words, details))
        kept.append(line)
    return "\n".join(kept), dropped


DETAIL_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "half", "quarter", "first", "second",
                "third", "last", "man", "lady", "left", "right", "forward", "backward", "side")


def line_details(text):
    """What makes a line a different fact even when its wording repeats another: numbers ('step 3', '1/4', '4.1'),
    counts ('S, S, S, S', '&'), roles, sides and directions. 'For a one-bar bounce ... (S, S)' and 'For a two-bar
    bounce ... (S, S, S, S)' share almost every word; the bar count and the counts differ."""
    lowered = text.lower()
    details = set(re.findall(r"\d+(?:[./]\d+)?", text))
    details |= {word for word in re.findall(r"[a-z]+", lowered) if word in DETAIL_WORDS}
    details |= {re.sub(r"\s", "", count) for count in re.findall(r"\b[SQ](?:\s*[,&-]\s*[SQ&])+\b", text)}
    return details


MAX_EXPANDED_RANGE = 8     # a range up to this many blocks of ONE section is listed by code instead of rejected


def expand_ranges(answer):
    """[44.30-44.31] -> [44.30, 44.31] by code. Writers keep writing ranges whatever the prompt says, and every
    rejection cost a whole rewrite (seen: 2 of 3 attempts, each one worse). Ranges over two sections, backwards or
    longer than MAX_EXPANDED_RANGE stay as they are, so CheckCites still rejects those.
    Returns (answer, ["44.30-44.31 -> 44.30, 44.31", ...])."""
    expanded = []

    def expand(bracket_match):
        content = bracket_match.group(1)
        if "-" not in content:
            return bracket_match.group(0)
        ids, problems = expand_reference(content)
        if problems or len(ids) > MAX_EXPANDED_RANGE * max(1, content.count(",") + 1):
            return bracket_match.group(0)
        listed = ", ".join(f"{section_id}.{block}" for section_id, block in dict.fromkeys(ids))
        expanded.append(f"{content.strip()} -> {listed}")
        return f"[{listed}]"

    return EVIDENCE_BRACKET.sub(expand, normalize_ranges(answer)), expanded


def is_heading(line):
    """A short line without an id and without . ! ? at the end ('**How to do it**', 'Cha Cha Fan - Side 1')."""
    text = line.strip().strip("*#").strip()
    return (bool(text) and not EVIDENCE_BRACKET.search(line) and not text.endswith((".", "!", "?"))
            and len(text.split()) <= 15)


def drop_debris_lines(answer):
    """Lines left over after claims were removed: an empty bullet ('-', '* ;'), a footnote alone ('[3]'), a label
    with only ids ('Answer: [1]'). Seen in 4 of 7 answers after RepairClaims removed sentences."""
    kept = []
    for line in answer.split("\n"):
        outside = EVIDENCE_BRACKET.sub("", line)
        words = re.findall(r"[A-Za-z]{2,}", outside)
        if line.strip() and (not words or (EVIDENCE_BRACKET.search(line) and len(words) < 2)):
            continue
        kept.append(line)
    return "\n".join(kept)


def drop_empty_headings(answer):
    """Headings with nothing under them are removed. Seen: repairs made two lines say the same, the repeat was
    dropped, and '**How to do it**' was left standing over nothing. Two levels: a bold or '#' heading is top level
    and keeps its role subheadings ('*Lady*', '*Man*') when they have content (the first version dropped
    '**How to do it**' because a subheading followed it)."""
    def top_level(line):
        text = line.strip()
        return text.startswith("#") or (text.startswith("**") and text.rstrip(" :").endswith("**"))
    kept, content_below_top, content_below_sub = [], False, False
    for line in reversed(answer.split("\n")):              # bottom-up: we know what lies below each heading
        if not line.strip():
            kept.append(line)
        elif not is_heading(line):
            content_below_top = content_below_sub = True
            kept.append(line)
        elif top_level(line):
            if content_below_top:
                kept.append(line)
            content_below_top = content_below_sub = False
        else:
            if content_below_sub:
                kept.append(line)
            content_below_sub = False
    return re.sub(r"\n{3,}", "\n\n", "\n".join(reversed(kept))).strip()


def fold_single_block_ids(answer):
    """A chart side is ONE block, but writers number its steps: [1257.3] for "Step 2". The content is right and
    the section is right, so code maps it to the only block ([1257.1]) instead of asking for a rewrite (two
    rewrites once shrank a full answer to its Prec. step). Only sections with exactly one block; ranges are
    left alone (CheckCites rejects them). Returns (answer, ["1257.3 -> 1257.1", ...])."""
    folded = []

    def fold(bracket_match):
        content = bracket_match.group(1)
        if any(dash in content for dash in "-‐‑‒–—−"):
            return bracket_match.group(0)
        ids, problems = expand_reference(content)
        if problems:
            return bracket_match.group(0)
        new_ids = []
        for section_id, block_number in ids:
            if block_number > 1 and section_exists(section_id) and len(section_blocks(section_id)) == 1:
                folded.append(f"{section_id}.{block_number} -> {section_id}.1")
                block_number = 1
            if (section_id, block_number) not in new_ids:
                new_ids.append((section_id, block_number))
        if new_ids == ids:
            return bracket_match.group(0)
        return "[" + ", ".join(f"{section_id}.{block}" for section_id, block in new_ids) + "]"

    return EVIDENCE_BRACKET.sub(fold, answer), folded


ABSENCE_CLAIM = re.compile(r"\b(books?|sources?|evidence|manuals?)\b[^.\n]{0,40}\b(contain|contains|give|gives|provide|"
                           r"provides|describe|describes|mention|mentions|include|includes|cover|covers)\s+no\b|"
                           r"\b(do|does)\s+not\s+(contain|give|provide|describe|mention|include|cover)\b|"
                           r"\bno (description|information|mention)\b|\bcannot be compared\b", re.I)


def absence_claims(answer):
    """Sentences that say what the books do NOT contain. The system reads only a few pages, so such a claim is
    almost never known (seen: 'the books contain no description of Viennese Waltz' after a read was blocked).
    'Not found in the books' (nothing at all was found) is allowed."""
    sentences = re.split(r"(?<=[.!?])\s+|\n", EVIDENCE_BRACKET.sub("", answer))
    return [sentence.strip() for sentence in sentences
            if ABSENCE_CLAIM.search(sentence) and "not found in the books" not in sentence.lower()]


ECHOED_QUESTION = re.compile(r"^\s*(\*\*)?(QUESTION|VRAAG)\s*:(\*\*)?[^\n]*\n+", re.I)
ANSWER_LABEL = re.compile(r"^\s*(\*\*)?(direct(e)?\s+(answer|antwoord)|answer|antwoord)\b\s*:?(\*\*)?\s*:?[ \t]*\n?",
                          re.I)


def strip_answer_label(answer):
    """Labels the writer puts in front of the answer, which the dancer should never see (demo traces 2026-11-14):
    an echoed 'QUESTION: ...' line (the request starts with one) and 'Answer:' / 'Direct answer' /
    'Directe antwoord:' before the first line."""
    answer = ECHOED_QUESTION.sub("", answer, count=1)
    return ANSWER_LABEL.sub("", answer, count=1)


NOT_FOUND_IDS = re.compile(r"\s*\[\s*\d+\.\d+(?:\s*,\s*\d+\.\d+)*\s*\]")


def clean_not_found(answer):
    """A 'Not found in the books' answer cites nothing: ids after 'the search covered ...' listed whole tables
    (one demo answer ended in 44 ids). Drop them; the sentence about what was searched stays."""
    if not answer.lower().lstrip(" *").startswith("not found in the books"):
        return answer
    return NOT_FOUND_IDS.sub("", answer).strip()


def citation_problems(answer, evidence):
    """Code check of the answer's evidence ids. [] = fine."""
    references, problems = parse_evidence(answer)
    for sentence in absence_claims(answer):
        problems.append(FEEDBACK_ABSENCE_CLAIM.format(sentence=sentence[:80]))
    cited = {evidence_id for _, ids in references for evidence_id in ids}
    for bracket_text, ids in references:       # no ranges: they hid which blocks really carry the fact
        if len(ids) > 1 and "-" in bracket_text:
            listed = ", ".join(f"{section_id}.{block}" for section_id, block in ids)
            problems.append(FEEDBACK_RANGE.format(bracket=bracket_text, listed=listed))
    for section_id, block_number in sorted(cited):
        if section_id not in evidence:
            problems.append(FEEDBACK_ID_NOT_READ.format(section_id=section_id, block=block_number))
        elif block_text((section_id, block_number)) is None:
            problems.append(FEEDBACK_NO_SUCH_BLOCK.format(section_id=section_id, block=block_number))
    for sentence in uncited_sentences(answer):            # facts without an id are never judged
        problems.append(FEEDBACK_UNCITED.format(sentence=sentence[:80]))
    for line in normalize_ranges(answer).split("\n"):    # ids alone on a line: the judge can only see one big claim
        if EVIDENCE_BRACKET.search(line) and not HAS_WORDS.search(EVIDENCE_BRACKET.sub("", line)):
            problems.append(FEEDBACK_IDS_ALONE.format(line=line.strip()))
            break
    if not cited and re.search(r"\[\s*\d+\.\d+", answer):
        problems.append(FEEDBACK_UNREADABLE_IDS)
    elif not cited and "not found in the books" not in answer.lower():
        problems.append(FEEDBACK_NO_IDS)
    return problems


# ------------------------------------------------------------------ claims (for JudgeClaims)
SENTENCE_END = re.compile(r"(?<=[.!?])(?<!vs\.)\s+(?=[A-Z])")   # ". Next", not "4.1" or "236.1"
HAS_WORDS = re.compile(r"[A-Za-z]{2,}")
WORD_OR_NUMBER = re.compile(r"\w+")
MAX_TITLE_WORDS = 15         # a line without . ! ? up to this long is a heading ('Cha Cha Fan - Side 1'), not a claim
MIN_UNCITED_WORDS = 5        # shorter uncited lines ('No.', 'Yes, it can.') are not checked for an id
MIN_CLAIM_WORDS = 3          # words or numbers; "and [1.2]" is a leftover, "Step 1 timing: 2" is a claim
# A sentence that ends dangling only points at its sources: "The relevant passages are [1.2]", "See [1.2]".
POINTS_AT_SOURCES = re.compile(r"\b(are|is|was|were|see|in|at|from|by|and|or|of|the|listed|shown|given)$", re.I)


def split_claims(answer):
    """Answer -> [{'claim': text, 'ids': [(section_id, block), ...]}]: the text before a bracket is its claim.
    Table rows and list items are handled line by line; uncited sentences get ids=[]."""
    claims, table_header = [], None
    pending_text = []                      # uncited lines since the last id: a line of only ids proves them
    for raw_line in normalize_ranges(answer).split("\n"):
        if raw_line.strip().startswith("|"):
            cells = [cell.strip().replace("**", "") for cell in raw_line.strip().strip("|").split("|")]
            if all(set(cell) <= set("-: ") for cell in cells):        # the |---|---| line
                continue
            if table_header is None:
                table_header = cells
                continue
            ids = [evidence_id for cell in cells for match in EVIDENCE_BRACKET.finditer(cell)
                   for evidence_id in expand_reference(match.group(1))[0]]
            pairs = [f"{name}: {EVIDENCE_BRACKET.sub('', cell).strip()}" for name, cell in zip(table_header, cells)
                     if EVIDENCE_BRACKET.sub("", cell).strip()]
            if ids and pairs:       # '| 1 | 2 | [2] |' -> 'Step: 1, Timing: 2' (numbers alone were never judged)
                claims.append(dict(claim=", ".join(pairs), ids=ids))
            continue
        table_header = None
        line = raw_line.strip().replace("**", "")
        if not line:
            continue
        if not HAS_WORDS.search(EVIDENCE_BRACKET.sub("", line)):          # only ids (and punctuation) on this line
            ids = [evidence_id for match in EVIDENCE_BRACKET.finditer(line)
                   for evidence_id in expand_reference(match.group(1))[0]]
            if ids and pending_text:
                claims.append(dict(claim=" ".join(pending_text), ids=ids))
            pending_text = []
            continue
        if not EVIDENCE_BRACKET.search(line):
            pending_text.append(line.strip(" -*•"))
            continue
        pending_text = []
        text_start = 0
        for bracket_match in EVIDENCE_BRACKET.finditer(line):
            ids, _ = expand_reference(bracket_match.group(1))
            sentences = SENTENCE_END.split(line[text_start:bracket_match.start()])
            claim_text = sentences[-1].strip(" -*•|,;:")
            if len(WORD_OR_NUMBER.findall(claim_text)) >= MIN_CLAIM_WORDS and not POINTS_AT_SOURCES.search(claim_text):
                claims.append(dict(claim=claim_text, ids=ids))
            elif claims and claims[-1]["ids"]:              # "[236.1] [236.2]": second bracket joins the claim
                claims[-1]["ids"] += ids
            text_start = bracket_match.end()
    return claims


# ------------------------------------------------------------------ showing the answer
# ------------------------------------------------------------------ whole tables as context
def table_unit(evidence_id, scope="figure"):
    """All evidence ids of the table a cited block belongs to, as a tuple:
    - a chart step -> the whole figure chart (Side 1, Side 2 and NOTES of that role);
    - a prose table row -> the run of consecutive table rows around it;
    - any other block -> itself.
    Citing stays per row/step, but a row alone ('4/4 | RUMBA') means little without its table, so the judge,
    the agents and the SOURCES list always get the whole table.
    scope='figure': a chart step brings the whole figure chart (both sides + NOTES): for reading and SOURCES.
    scope='section': a chart step brings its own chart side (a complete table): for the judge, whose memory
    use grows with its input (two whole figure charts x 21 claims crashed the decision model)."""
    section_id, block_number = evidence_id
    blocks = section_blocks(section_id)
    chart_sections = figure_chart(section_id) if scope == "figure" else (section_id,)
    if library().pages.loc[section_id, "kind"] == "chart" or len(chart_sections) > 1:
        return tuple((part, number) for part in chart_sections
                     for number in range(1, len(section_blocks(part)) + 1))
    if not is_table_row(blocks[block_number - 1]):
        return (evidence_id,)
    first = last = block_number
    while first > 1 and is_table_row(blocks[first - 2]):
        first -= 1
    while last < len(blocks) and is_table_row(blocks[last]):
        last += 1
    return tuple((section_id, number) for number in range(first, last + 1))


def table_units(evidence_ids, scope="figure"):
    """{unit: set of cited ids in it}, in order of first citation; each table appears once."""
    units = {}
    for evidence_id in evidence_ids:
        if block_text(evidence_id) is not None:
            units.setdefault(table_unit(evidence_id, scope), set()).add(evidence_id)
    return units


def unit_label(unit):
    """'Rumba, page 70 | FAN (LADY) - Side 1  +  Rumba, page 70 | FAN (LADY) - Side 2  + ...' (each section once)."""
    return "  +  ".join(dict.fromkeys(section_label(section_id) for section_id, _ in unit))


def unit_text(unit, cited, markers=None):
    """A whole table, one row/step per line, cited ones marked: '>> [1119.27] ...' (or footnote markers);
    a '--- section label' line wherever the table crosses into another section (Side 2, NOTES)."""
    lines, current_section = [], None
    for evidence_id in unit:
        if len({section_id for section_id, _ in unit}) > 1 and evidence_id[0] != current_section:
            current_section = evidence_id[0]
            lines.append(f"--- {section_label(current_section)}")
        if markers is not None:
            label = f"{markers.get(evidence_id, ''):>10}"
        else:
            label = (">> " if evidence_id in cited else "   ") + f"[{evidence_id[0]}.{evidence_id[1]}]"
        lines.append(f"{label} {block_text(evidence_id)}")
    return "\n".join(lines)


def with_footnotes(answer):
    """(answer with ids replaced by footnote numbers '[1, 2]', {evidence_id: number}) in order of first citation."""
    footnote_numbers = {}

    def to_footnotes(bracket_match):
        ids, _ = expand_reference(bracket_match.group(1))
        valid_ids = [evidence_id for evidence_id in ids if block_text(evidence_id) is not None]
        for evidence_id in valid_ids:
            footnote_numbers.setdefault(evidence_id, len(footnote_numbers) + 1)
        numbers = sorted({footnote_numbers[evidence_id] for evidence_id in valid_ids})
        return "[" + ", ".join(map(str, numbers)) + "]" if numbers else ""     # listed, never '1-3'

    rendered = EVIDENCE_BRACKET.sub(to_footnotes, normalize_ranges(answer))
    return merge_adjacent_footnotes(rendered), footnote_numbers


def merge_adjacent_footnotes(text):
    """'[1, 2, 3, 4] [2] [3] [4]' -> '[1, 2, 3, 4]': stacked brackets after one sentence (blind review)."""
    def merge(match):
        numbers = sorted({int(number) for number in re.findall(r"\d+", match.group(0))})
        return "[" + ", ".join(map(str, numbers)) + "]"
    return re.sub(r"\[\d+(?:, \d+)*\](?:\s*\[\d+(?:, \d+)*\])+", merge, text)


def answer_parts(answer):
    """The final answer as data for the web viewer:
    {"body": text with footnote numbers,
     "sources": [{"numbers": [1, 2], "label": "Rumba, page 59 | ...",
                  "sections": [{"label": "...", "kind": "chart"|"prose",
                                "blocks": [{"id": "1257.1", "footnote": 1 or None, "text": "..."}]}]}]}
    Each table appears once, whole, with its cited blocks marked by their footnote number."""
    body, footnote_numbers = with_footnotes(answer)
    sources = []
    for unit, cited in table_units(footnote_numbers).items():
        sections = []
        for evidence_id in unit:
            section_id = evidence_id[0]
            if not sections or sections[-1]["section_id"] != section_id:
                kind = library().pages.loc[section_id, "kind"] if section_id in library().pages.index else "prose"
                sections.append({"section_id": section_id, "label": section_label(section_id), "kind": str(kind),
                                 "blocks": []})
            sections[-1]["blocks"].append({"id": f"{evidence_id[0]}.{evidence_id[1]}",
                                           "footnote": footnote_numbers.get(evidence_id),
                                           "text": block_text(evidence_id)})
        sources.append({"numbers": sorted(footnote_numbers[evidence_id] for evidence_id in cited),
                        "label": unit_label(unit), "sections": sections})
    return {"body": body, "sources": sources}


def render_with_sources(answer):
    """Ids -> footnote numbers, plus a SOURCES list with the book's own words and page."""
    body, footnote_numbers = with_footnotes(answer)
    if not footnote_numbers:
        return body
    sources = []
    for unit, cited in table_units(footnote_numbers).items():   # each table once, cited rows marked [n]
        numbers = sorted(footnote_numbers[evidence_id] for evidence_id in cited)
        markers = {evidence_id: f"[{footnote_numbers[evidence_id]}]" for evidence_id in cited}
        heading = f"[{', '.join(map(str, numbers))}] {unit_label(unit)}"
        sources.append(heading + ("" if len(unit) == 1 else "   (whole table, cited rows marked)")
                       + "\n" + unit_text(unit, cited, markers))
    return body + "\n\nSOURCES\n" + "\n\n".join(sources)
