"""The toolbox. One toolbox, two callers:
  - graph nodes call the functions in books.py directly (fixed route, same every run);
  - agents call the tools below when they see a reason to.

Rules for every tool:
  - the docstring is the prompt: it says WHEN to use the tool, not only what it does;
  - errors come back as text, so the agent can read them and try again;
  - reading tools put whole sections into the run's EvidenceLog. Only evidence there can be cited.
"""
import functools
import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass

from pydantic_ai import RunContext, Tool

from dance_rag_agentic.retrieval import books
from dance_rag_agentic.retrieval import outline as book_outline  # the module; 'outline' is also the tool's name
from dance_rag_agentic.config import READ_UNIT_CHARACTERS
from dance_rag_agentic.agent.evidence import EvidenceLog


@dataclass
class AgentDeps:
    """What every tool can reach during one agent run (PydanticAI passes it as ctx.deps)."""
    evidence: EvidenceLog
    caller: str                    # 'TECHNIQUE agent', 'UNSURE agent', 'FillGap'
    report: Callable | None = None # live printing of each tool call (None = quiet)
    calls_made: set = None         # exact calls in this run: a repeat gets a short answer, not the same result again
    role: str | None = None        # 'Man' / 'Lady' if the question names one; None = both roles are read

    def __post_init__(self):
        self.calls_made = set()


def reported(tool):
    """Wrap a tool so every call is reported while the agent works. functools.wraps keeps the name,
    docstring and type hints, which is what PydanticAI reads to build the tool for the LLM."""
    signature = inspect.signature(tool)

    @functools.wraps(tool)
    def wrapper(ctx, *args, **kwargs):
        arguments = ", ".join([repr(value) for value in args] + [f"{key}={value!r}" for key, value in sorted(kwargs.items())])
        call = f"{tool.__name__}({arguments})"
        # A repeat is the same tool with the same arguments, however written: positional or by name, any case,
        # extra spaces ('overturn' 3x and 'Overturn waltz' after 'overturn waltz' in one UNSURE run).
        bound = signature.bind(ctx, *args, **kwargs)
        bound.apply_defaults()
        key = (tool.__name__, tuple(sorted((name, " ".join(value.lower().split()) if isinstance(value, str) else value)
                                           for name, value in bound.arguments.items() if name != "ctx")))
        if key in ctx.deps.calls_made:            # the agent repeated a call (seen: 3x read_section(1119))
            result = (f"You already made exactly this call ({call}); its result is above. "
                      f"Use other words, another tool, or stop if you have what you need.")
        else:
            ctx.deps.calls_made.add(key)
            result = tool(ctx, *args, **kwargs)
        if ctx.deps.report:
            ctx.deps.report(call, result)
        return result
    return wrapper


def _dance_or_error(dance):
    """(resolved dance, None) or (None, error text)."""
    try:
        return books.resolve_dance(dance), None
    except ValueError as error:
        return None, str(error)


def _section_text(section_id, max_characters=None):
    """The section with numbered blocks. max_characters: blocks after that are left out of the text (they stay in
    the evidence; the answer writer gets them) and the cut is said."""
    lines, shown = [f"[{books.section_label(section_id)}]"], 0
    blocks = books.section_blocks(section_id)
    for number, block in enumerate(blocks, 1):
        if max_characters is not None and shown and shown + len(block) > max_characters:
            lines.append(f"(... {len(blocks) - number + 1} more blocks of section {section_id} not shown, "
                         f"[{section_id}.{number}] to [{section_id}.{len(blocks)}]: mostly table rows)")
            break
        lines.append(f"[{section_id}.{number}] {block}")
        shown += len(block)
    return "\n".join(lines)


# ------------------------------------------------------------------ finding sections
def search(ctx: RunContext[AgentDeps], query: str, dance: str | None = None, kind: str | None = None) -> str:
    """Hybrid search (words + meaning, reciprocal rank fusion). The agent sees SEARCH_DESCRIPTION, not this text;
    the Args below are its parameter descriptions.

    Args:
        query: the topic, e.g. "head position closed hold"
        dance: limit to one book, e.g. "Slow Waltz"; leave empty to search all books
        kind: "prose" (explanations) or "chart" (figure step tables and their notes); leave empty for both
    """
    resolved_dance, error = _dance_or_error(dance)
    if error:
        return error
    if kind not in (None, "", "prose", "chart"):
        return "kind must be 'prose', 'chart' or empty"
    return _search_results(ctx, query, resolved_dance, kind or None)


_STOP_WORDS = {"the", "and", "with", "what", "how", "does", "dance", "figure", "figures", "technique", "step", "steps",
               "turn", "man", "lady", "for", "from", "into", "this", "that", "when", "which", "about", "explain"}


def _content_words(query):
    """Query words that say something (3+ letters: 'Fan'; not generic, not a dance name), as 5-letter stems."""
    dances = {word for name in books.library().dances for word in name.lower().split()}
    words = [word for word in re.findall(r"[a-z]+", query.lower())
             if len(word) >= 3 and word not in _STOP_WORDS and word not in dances]
    return [word[:5] for word in words]


def _mentions(text, stems):
    return sum(stem in text for stem in stems)


def _best_sentence(section_id, stems, length=170):
    """The sentence of a section that holds most query words (not the section's first 200 characters)."""
    text = " ".join(books.section_blocks(section_id)).replace("\n", " ")
    sentences = [sentence for sentence in re.split(r"(?<=[.;])\s+", text) if len(sentence) >= 30] or [text]
    best = max(sentences, key=lambda sentence: _mentions(sentence.lower(), stems)) if stems else sentences[0]
    return best[:length] + ("…" if len(best) > length else "")


def figure_names_matching(query, dance=None, limit=5):
    """Figure names whose words start with the query's content words ('overturn' -> Overturned Running Spin Turn)."""
    stems = _content_words(query)
    if not stems:
        return []
    found = []
    for name_dance in ([dance] if dance else books.library().dances):
        for name in books.figures_of(name_dance):
            name_words = [word for word in name.lower().split() if len(word) > 2]
            score = sum(any(word.startswith(stem) for word in name_words) for stem in stems)
            if score:
                found.append((score, name, name_dance))
    found.sort(key=lambda item: (-item[0], len(item[1])))
    return found[:limit]


def _clean_snippet(text):
    return " ".join(str(text).replace("\\n", " ").split())



SEARCH_DESCRIPTION = """Search the books for a topic. Words and meaning are searched together.
Returns FIGURES (matching figure charts, one line per figure, read them with read_charts) and TEXT (matching
explanations: where in the book, how long, and the sentence that matches; read them with read_section). TEXT is
ordered best first; "weaker matches" share fewer words with the query. Snippets are not evidence: read what you rely
on. Describe the topic in the book's terms; no negations, no bare numbers.

Args:
    query: the topic, e.g. "head position closed hold"
    dance: limit to one book, e.g. "Slow Waltz"; leave empty to search all books
    kind: "prose" (explanations only) or "chart" (figure charts only); leave empty for both
"""
SEARCH_FIGURES, SEARCH_STRONG, SEARCH_WEAK = 4, 5, 3


def _size_word(characters):
    return "1-2 sentences" if characters < 250 else "short" if characters < 900 else "a page" if characters < 2500 else "long"


def _short_path(section_id, with_dance):
    """'Part 2 (Paso Doble) › Timing › Musical Structure' or 'Part 1 (all Latin books) › Lead Hold Shaping'."""
    record = books.library().pages.loc[section_id]
    parts = [part.strip() for part in str(record.get("path") or record.header).split("›")]
    head = parts[0]
    if head.startswith("Part 1"):
        head = f"Part 1 (all {record.style} books)" if record.get("copies") or record.get("copy_of") == record.get("copy_of") else "Part 1"
    elif head.startswith("Part 2"):
        head = f"Part 2 ({record.dance} only)"
    tail = [part for part in parts[1:] if part][-2:]
    return " › ".join([head] + tail) + (f" [{record.dance}]" if with_dance and not head.startswith("Part 2") else "")


def _search_results(ctx, query, dance, kind):
    """What the search tool returns (format "mid", chosen by A/B test: docs/decisions.md): one line per matching
    figure, text hits with where they live, their size and the sentence that matches, weaker matches apart."""
    stems = _content_words(query)
    pages = books.library().pages
    hits = books.hybrid_hits(query, dance, kind, limit=30)
    figures, notes, strong, weak = {}, {}, [], []
    for hit in hits:
        record = pages.loc[hit["section_id"]]
        if isinstance(record.figure, str) and record.figure:
            figures.setdefault((record.figure, record.dance), True)
            if record.kind != "chart" and stems and (record.figure, record.dance) not in notes and \
                    _mentions(" ".join(books.section_blocks(hit["section_id"])).lower(), stems):
                notes[(record.figure, record.dance)] = _clean_snippet(_best_sentence(hit["section_id"], stems, 150))
            continue
        text = " ".join(books.section_blocks(hit["section_id"])).lower()
        good = hit["found_by"] != "rank" and (not stems or _mentions(text, stems))
        target = strong if good else weak
        if len(target) < (SEARCH_STRONG if good else SEARCH_WEAK):
            target.append(hit["section_id"])
    for _, name, name_dance in figure_names_matching(query, dance):
        if kind != "prose" or not figures:
            figures.setdefault((name, name_dance), True)
    lines = []
    if stems and not strong and not notes:
        words = [word for word in re.findall(r"[a-z]+", query.lower()) if word[:5] in stems]
        lines.append(f"No text contains {', '.join(repr(word) for word in dict.fromkeys(words))}"
                     + (" (try without kind, or list_figures)" if kind else " (try the book's own term, or list_figures)"))
    if figures:
        lines.append("FIGURES:")
        for name, name_dance in list(figures)[:SEARCH_FIGURES]:
            roles = sorted(set(pages[(pages.figure == name) & (pages.dance == name_dance) & (pages.kind == "chart")].role.dropna()))
            already = " (already read)" if any(section_id in ctx.deps.evidence for section_id in
                                               pages[(pages.figure == name) & (pages.dance == name_dance)].index) else ""
            note = f' — NOTE: "{notes[(name, name_dance)]}"' if (name, name_dance) in notes else ""
            lines.append(f"- {name} ({name_dance}): chart {' + '.join(roles)}{already}{note}  -> read_charts(figure={name!r}, dance={name_dance!r})")
        if len(figures) > SEARCH_FIGURES:
            lines.append("  also: " + ", ".join(f"{name} ({name_dance})" for name, name_dance in list(figures)[SEARCH_FIGURES:8]))

    def text_line(section_id):
        size = sum(len(block) for block in books.section_blocks(section_id))
        already = ", already read" if section_id in ctx.deps.evidence else ""
        sentence = _clean_snippet(_best_sentence(section_id, stems, 200))
        return f"- {_short_path(section_id, not dance)} ({_size_word(size)}{already}): \"{sentence}\"  -> read_section({section_id})"
    if strong:
        lines.append("TEXT:")
        lines += [text_line(section_id) for section_id in strong]
    if weak:
        lines.append("Weaker matches:")
        lines += [text_line(section_id) for section_id in weak]
    return "\n".join(lines) or "no matches"


# ------------------------------------------------------------------ catalog
def outline(ctx: RunContext[AgentDeps], dance: str, section_id: int | None = None) -> str:
    """The table of contents of one book: its parts (front matter, part 1 general principles shared by every book
    of the style, part 2 principles specific to this dance) and their chapters, with section_ids and sizes.
    Use to find WHERE a topic is explained (the character, walks, timing or actions of a dance, a chapter on a
    principle) and to see what a chapter contains, instead of guessing search words. Then read_section a chapter.
    Figure charts are not listed: use list_figures / select_charts for those.

    Args:
        dance: the book, e.g. "Samba"
        section_id: optional: a chapter's section_id to list its subheadings
    """
    resolved_dance, error = _dance_or_error(dance)
    if error or not resolved_dance:
        return error or "outline needs a dance"
    return "\n".join(book_outline.outline_lines(resolved_dance, section_id))


def list_figures(ctx: RunContext[AgentDeps], dance: str) -> str:
    """All figures that have charts in one book. Use for 'which figures does X have' questions,
    or to find the exact figure name before select_charts.

    Args:
        dance: e.g. "Quickstep"
    """
    resolved_dance, error = _dance_or_error(dance)
    if error:
        return error
    return f"{resolved_dance} figures: " + ", ".join(books.figures_of(resolved_dance))


def dances_with_figure(ctx: RunContext[AgentDeps], figure: str) -> str:
    """In which books a figure has charts. Use for 'which dances have X' questions, or when a figure
    is not in the dance the user named.

    Args:
        figure: figure name as the book writes it; typos are fine
    """
    figure_name = books.resolve_figure(figure)
    if not figure_name:
        return f"no figure named {figure!r}. Similar names: {', '.join(books.similar_figures(figure)) or 'none'}"
    return f"'{figure_name}' has charts in: {', '.join(books.dances_with_figure(figure_name))}"


# ------------------------------------------------------------------ charts
def select_charts(ctx: RunContext[AgentDeps], figure: str, dance: str | None = None, role: str | None = None,
                  aspect: str | None = None) -> str:
    """List the charts of a figure (step tables per role and side, plus their NOTES) with their COLUMNS,
    without reading them. Use to see which chart holds the aspect you need, e.g. for a concrete example
    of a principle in one figure. Then read_charts or read_section.

    Args:
        figure: figure name, typos are fine
        dance: e.g. "Rumba"; leave empty for every dance that has the figure
        role: "man", "lady", or empty for both
        aspect: one of lead, timing, hips, footwork, alignment, turn, position, rise, sway; empty = all columns
    """
    figure_name = books.resolve_figure(figure)
    if not figure_name:
        return f"no figure named {figure!r}. Similar names: {', '.join(books.similar_figures(figure)) or 'none'}"
    resolved_dance, error = _dance_or_error(dance)
    if error:
        return error
    available_in = books.dances_with_figure(figure_name)
    if resolved_dance and resolved_dance not in available_in:
        return f"'{figure_name}' has no charts in the {resolved_dance} book. It has charts in: {', '.join(available_in)}."
    charts = books.charts_of(figure_name, resolved_dance, books.resolve_role(role), aspect)
    lines = [f"'{figure_name}' has charts in: {', '.join(available_in)}"]
    for chart in charts:
        content = ("columns: " + ", ".join(chart["columns"])) if chart["kind"] == "chart" else "notes printed below the charts"
        marker = "" if chart["has_aspect"] else "  (no column for this aspect)"
        lines.append(f"section_id={chart['section_id']} | {chart['dance']} p.{chart['page']} | {chart['header']} | "
                     f"{content}{marker}")
    return "\n".join(lines)


def read_charts(ctx: RunContext[AgentDeps], figure: str, dance: str, role: str | None = None,
                aspect: str | None = None) -> str:
    """Read the charts of ONE figure in ONE dance: every step as a block with its evidence id.
    Use when you need what a figure actually does (steps, timing, lead, footwork...).
    Always the whole chart per role: Side 1, Side 2 and the NOTES. aspect only tells you which columns matter.

    Args:
        figure: figure name, typos are fine
        dance: e.g. "Cha Cha" (required: charts differ per dance)
        role: not used: the role named in the question is read (both roles if it names none)
        aspect: one of lead, timing, hips, footwork, alignment, turn, position, rise, sway; empty = all
    """
    figure_name = books.resolve_figure(figure)
    if not figure_name:
        return f"no figure named {figure!r}. Similar names: {', '.join(books.similar_figures(figure)) or 'none'}"
    resolved_dance, error = _dance_or_error(dance)
    if error or not resolved_dance:
        return error or "read_charts needs a dance; use select_charts or dances_with_figure first"
    # the question decides the role, not the agent: a question without a role gets both (a dancer asking
    # "how do I do X" may be either partner)
    charts = books.charts_of(figure_name, resolved_dance, ctx.deps.role, aspect)   # whole chart
    if not charts:
        return (f"no charts of '{figure_name}' in {resolved_dance} for that role/aspect. "
                f"It has charts in: {', '.join(books.dances_with_figure(figure_name))}.")
    for chart in charts:
        ctx.deps.evidence.add(chart["section_id"], f"{ctx.deps.caller}: read_charts")
    return "\n\n".join(_section_text(chart["section_id"]) for chart in charts)


def _same_text_already_read(section_id, evidence):
    """The id of an already-read section with (nearly) the same text, or None (only sections of 200+ characters).
    Nearly: the copies differ by OCR noise ('34 1/2' vs '12 1/ / 2' in the same Music in DanceSport table).
    Only PART 1 (general principles, the same in every book of a style). Part 2 texts of two dances can be 90% the
    same and still be two dances' facts: blocking Rumba's Forward Walk after Cha Cha's made the answer claim the
    books had no Rumba description."""
    import difflib

    def in_part_1(any_id):
        unit = book_outline.unit_of(any_id)
        while unit is not None and unit.parent is not None:
            unit = unit.parent
        return unit is not None and unit.title.startswith("Part 1")
    text = books.squash(" ".join(books.section_blocks(section_id)))
    if len(text) < 200 or not in_part_1(section_id):
        return None
    style = books.library().pages.loc[section_id, "style"]
    for read_id in evidence.section_ids:
        # same style only: the Latin and Standard Part 1 are ~90% the same text but their tables differ (tempos);
        # blocking Viennese Waltz's 'Music in Dancesport' after Cha Cha's made the answer say 'no Viennese Waltz'
        if not in_part_1(read_id) or books.library().pages.loc[read_id, "style"] != style:
            continue
        other = books.squash(" ".join(books.section_blocks(read_id)))
        matcher = difflib.SequenceMatcher(None, text, other, autojunk=False)
        if abs(len(other) - len(text)) < 0.1 * len(text) and matcher.quick_ratio() > 0.9 and matcher.ratio() > 0.9:
            return read_id
    return None


def read_section(ctx: RunContext[AgentDeps], section_id: int) -> str:
    """Read the book text around a section, split into numbered blocks (paragraphs; a chart side is one block)
    with abbreviations explained. Each block starts with its evidence id, e.g. [2705.2].
    Prose is read as a whole heading unit: the chapter the section belongs to, with its subheadings (a very large
    chapter: the part you asked for, plus a list of its other subheadings). A part of a figure chart reads the
    whole chart. Use with a section_id from a search, outline, index_lookup or select_charts.

    Args:
        section_id: a section_id from another tool's result
    """
    if not books.section_exists(section_id):
        return f"no section with section_id={section_id}"
    if section_id in ctx.deps.evidence:
        return (f"section {section_id} ({books.section_label(section_id)}) is already in the evidence; "
                f"the answer writer will get it. Do not read it again: read something else or stop.")
    same_text = _same_text_already_read(section_id, ctx.deps.evidence)
    if same_text is not None:                # part 1 is the same text in every book of a style
        return (f"section {section_id} has the same text as section {same_text} ({books.section_label(same_text)}), "
                f"which is already in the evidence. Read something else or stop.")
    # part of a figure chart: read Side 1, Side 2 and NOTES; for both roles when the question names no role
    whole_chart = books.figure_chart(section_id, all_roles=ctx.deps.role is None)
    if len(whole_chart) > 1:
        for part in whole_chart:
            ctx.deps.evidence.add(part, f"{ctx.deps.caller}: read_section (whole chart of {section_id})")
        return (f"(section {section_id} is part of a figure chart: the whole chart is given)\n\n"
                + "\n\n".join(_section_text(part) for part in whole_chart))
    unit = book_outline.unit_of(section_id)
    if unit is not None and unit.level == 0:                 # a part: its intro + its chapters to choose from
        for part_section in unit.section_ids:
            ctx.deps.evidence.add(part_section, f"{ctx.deps.caller}: read_section")
        dance = books.library().pages.loc[section_id, "dance"]
        chapters = [line for line in book_outline.outline_lines(dance) if line.startswith("  ")
                    and any(f"section_id={child.node_id} " in line for child in unit.children)]
        return ("\n\n".join(_section_text(part_section) for part_section in unit.section_ids) +
                f"\n\n(this is the start of '{unit.title}'; its chapters, read_section the ones you need:)\n" + "\n".join(chapters))
    if unit is not None and unit.characters() <= READ_UNIT_CHARACTERS:
        to_read, rest = unit.unit_sections(), []
    elif unit is not None:                                   # too big: the asked subheading (or the chapter's own text)
        asked = book_outline.node_of(section_id)
        to_read = asked.section_ids if asked is not unit else unit.section_ids
        rest = [child for child in unit.children if child is not asked]
    else:
        to_read, rest = [section_id], []
    asked_node = book_outline.node_of(section_id)
    if asked_node is not None and asked_node is not unit:     # what was asked comes first, then the rest of the unit
        to_read = asked_node.section_ids + [part for part in to_read if part not in asked_node.section_ids]
    to_read = [part for part in to_read if part not in ctx.deps.evidence]
    # one read shows at most READ_UNIT_CHARACTERS: a unit, then ':' continuations, until the budget is used up;
    # what does not fit is listed, not read (seen: one read of 13,869 characters)
    budget = READ_UNIT_CHARACTERS
    parts = ([f"(section {section_id}, then the rest of its chapter '{unit.title}' "
               f"({book_outline.heading_path(unit.node_id)}))"] if unit is not None and to_read else [])
    not_read = []

    def read(part, why):
        nonlocal budget
        if budget <= 0:
            not_read.append(part)
            return False
        ctx.deps.evidence.add(part, f"{ctx.deps.caller}: read_section{why}")
        text = _section_text(part, max_characters=budget)
        parts.append(text)
        budget -= len(text)
        return True

    for part in to_read:
        read(part, "" if part == section_id else f" (unit of {section_id})")
    current = ([part for part in to_read if part not in not_read] or [section_id])[-1]
    for _ in range(2):                          # a section ending in ':' is unfinished: read on (max 2)
        following = books.next_section(current)
        if following is None or not books.ends_unfinished(current) or following in ctx.deps.evidence:
            break
        if not read(following, f" (continuation of {current})"):
            break
        parts[-1] = f"(continued automatically: section {current} ended with ':')\n" + parts[-1]
        current = following
    if not_read:
        parts.append("(not read yet, to keep this result short; read_section them if you need them:)\n" +
                     "\n".join(f"  section_id={part} | {book_outline.heading_path(part)}" for part in not_read))
    if rest:
        parts.append(f"(chapter '{unit.title}' is large; its other subheadings, read_section the ones you need:)\n" +
                     "\n".join(f"  section_id={child.node_id} | p.{child.page} | {child.title} | {child.characters()} chars"
                                for child in rest))
    following = books.next_section(current)
    if following is not None and following not in ctx.deps.evidence and following not in not_read:
        parts.append(f"(the text continues in section {following}: {book_outline.heading_path(following)}; "
                     f"read_section it if you need the rest)")
    return "\n\n".join(parts)

TOOLBOX = [reported(tool) for tool in (search, outline, list_figures, dances_with_figure, select_charts, read_charts,
                                       read_section)]


def toolbox():
    """The tools an agent gets."""
    # strict=False for every tool: PydanticAI marks some tools strict and others not, and the Cerebras API
    # rejects that mix ("Tools with mixed values for 'strict' are not allowed"); Ollama ignores the flag.
    return [Tool(tool, strict=False, description=SEARCH_DESCRIPTION if tool.__name__ == "search" else None)
            for tool in TOOLBOX]
