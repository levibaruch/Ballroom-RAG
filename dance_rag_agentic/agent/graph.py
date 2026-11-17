"""The graph, as a plain Python state machine: every node reads/writes one State and returns the next node.

    question
      -> self-harm wording -> SafetyReply (fixed text with crisis contacts) -> Finish
      -> ExtractEntities (code) -> ClassifyKind (decision model + override rules)
      -> mixed question (technique + general advice) -> SplitQuestion (LLM) -> GeneralAdvice (general part)
                                                     -> ExtractEntities again with the books part
      -> FigurePath -> Background (agent: prose explaining the chart's terms)
         | ComparePath | TechniqueAgent | UnsureAgent | TermPath | Refuse
         | CharacterPath (code reads music + principles, short agent, own-knowledge 'feel' part)
         | GeneralAdvice (own knowledge, marked as not from the books) -> Finish
      -> Answer (LLM, no tools) -> CheckCites (code, rewrite) -> JudgeClaims (decision model, per claim)
      -> unsupported claims / 'Not found' -> FillGap (searches for those claims) -> Answer   (max REPAIR_ROUNDS)
      -> Finish

Run:  state = await answer_question("...")   (async code)
      state = ask("...")                     (in a script; apps/cli.py is the command line)

The file follows the flow, one section per stage: State and trace helpers / understand (ExtractEntities,
ClassifyKind) / fixed paths (code) / searching agents / answer, checks, repair / run. It stays one module on
purpose: the nodes hand over to each other in a cycle (SplitQuestion goes back to ExtractEntities, agents fall
back to GeneralAdvice, repairs go back to Answer), so separate modules would need imports inside functions.
"""
import asyncio
import json
import re
import time
from dataclasses import dataclass, field

from dance_rag_agentic.retrieval import books
from dance_rag_agentic.agent.prompts import (REQUEST_CHARACTER_NOTE, REQUEST_LANGUAGE, REQUEST_ROLE_WORDS, REQUEST_FIGURE_LIST, FEEDBACK_MISSES_QUESTION, FEEDBACK_MISSING_PARTS, FEEDBACK_NEW_EVIDENCE, FEEDBACK_NO_NEW_EVIDENCE, FEEDBACK_RETYPED_CHART, FILL_MISSING_PART, FILL_NOT_FOUND, REQUEST_ALREADY_READ, REQUEST_ANSWER_NOTES, REQUEST_BACKGROUND, REQUEST_COMPARISON, REQUEST_ENTITIES, REQUEST_FIGURE_CANDIDATES, REQUEST_FILLGAP, REQUEST_GENERAL_BOOKS_PART, REQUEST_GENERAL_NOTHING_IN_BOOKS, REQUEST_NOTES, REQUEST_REPAIR_EVIDENCE, REQUEST_REPAIR_NEW_EVIDENCE, REQUEST_REPAIR_VERDICT, REQUEST_REWRITE, REQUEST_SEARCHES_DONE)  # noqa: E501
from dance_rag_agentic.agent.agents import (about_writer, answer_writer, character_writer, general_writer, polish_writer,
                                            prompt, repair_writer, run_counted, run_search, split_writer,
                                            take_thinking)
from dance_rag_agentic.agent.prompts import (BACKGROUND_PROMPT, CHARACTER_PROMPT, FILLGAP_PROMPT, TECHNIQUE_PROMPT,
                                             UNSURE_PROMPT)
from dance_rag_agentic.agent.checks import answers_question, decision_parallel, judge_claims, unsupported
from dance_rag_agentic.config import (CloudAccessError, TERM_HEADING_SECTIONS, AGENT_TOOL_CALLS, ANSWER_ATTEMPTS, ANSWERS_QUESTION_FROM, ASPECT_MIN_PROBABILITY, FILLGAP_TOOL_CALLS, JUDGE_SUPPORTED_FROM,
                     REPAIR_ROUNDS, MIXED_FROM, BACKGROUND_TOOL_CALLS, BACKGROUND_TERMS, CHARACTER_TOOL_CALLS,
                     MODELS, TRACE_RESULT_LINES, UNSURE_BELOW)
from dance_rag_agentic.agent.entities import (clean_name, extract_entities, figures_ending_in_question_word, find_names, mentions_role_words,
                       question_language)
from dance_rag_agentic.agent.evidence import (EvidenceLog, answer_parts, clean_not_found, strip_answer_label, drop_debris_lines, drop_empty_headings, drop_repeated_lines, expand_ranges, normalize_brackets, block_text, cited_ids, citation_problems, fold_single_block_ids, render_with_sources, replace_claim,
                       split_claims, table_units, unit_label, unit_text)
from dance_rag_agentic.agent.decision import DecisionModelError, classify_question
from dance_rag_agentic.agent.safety import SAFETY_REPLY, is_self_harm
from dance_rag_agentic.agent.tools import AgentDeps
from dance_rag_agentic.agent.usage import METER


# ------------------------------------------------------------------ state and trace helpers
def out_of_scope_answer():
    return (f"Thanks for asking! I'm a helper for the WDSF technique books ({books.books_sentence()}), so this "
            f"question is outside what I can answer well. Ask me about a figure, its timing or footwork, leading "
            f"and following, or the character and music of one of these dances, and I'll gladly help.")


GENERAL_HEADING = "TIPS (general advice, not from the books)"
CHARACTER_HEADING = "THE FEEL OF THE DANCE (general dance knowledge, not from the books)"
# Put above the books part of a CHARACTER answer by code: the writer may not say what the books lack (CheckCites
# rejects absence claims), but here it is the honest frame: the books give technique, the feel comes after.
CHARACTER_FRAME = {
    "English": "The books describe technique, not the feel of a dance. What they say that shapes its character:",
    "Dutch": "De boeken beschrijven techniek, niet het gevoel van een dans. Wat ze zeggen dat het karakter vormt:"}
CHARACTER_NOTHING = {
    "English": "The books describe technique, not the feel of a dance, and they had nothing on this question.",
    "Dutch": "De boeken beschrijven techniek, niet het gevoel van een dans, en ze hadden niets over deze vraag."}


def in_language(texts, language):
    """A fixed text in the question's language when we have it, else English."""
    return texts.get(language) or texts["English"]


def language_lines(state):
    """The request line that fixes the reply language, as a list ([] when unclear: the prompt rule decides)."""
    return [REQUEST_LANGUAGE.format(language=state.language)] if state.language else []


@dataclass
class State:
    question: str
    use_judge: bool = True
    entities: dict = field(default_factory=dict)       # dances, figures, role, steps (code)
    kind_probabilities: dict = field(default_factory=dict)
    kind_confidence: float = 0.0
    aspect: str | None = None
    route: str = ""                                     # FIGURE / FIGURE_COMPARE / TECHNIQUE / UNSURE / TERM / GENERAL / OUT_OF_SCOPE
    route_reason: str = ""
    original_question: str = ""                         # the dancer's words; question becomes the books part after a split
    mixed_probability: float | None = None              # decision model P(technical part + general-advice part)
    split_done: bool = False                            # SplitQuestion ran (it never runs twice)
    general_question: str = ""                          # the part answered by GeneralAdvice
    general_answer: str = ""                            # own-knowledge advice, shown under its own heading
    general_heading: str = GENERAL_HEADING              # CharacterPath: CHARACTER_HEADING
    language: str | None = None                         # the question's language (code), for every writer
    books_part_pending: bool = False                    # after GeneralAdvice: route the books part
    sides: list = field(default_factory=list)           # [{'dance', 'figure', 'available'}] for figure routes
    evidence: EvidenceLog = field(default_factory=EvidenceLog)
    notes: list = field(default_factory=list)           # handed to the answer writer (missing sides, agent notes)
    to_fill: list = field(default_factory=list)         # what FillGap must search for
    fill_returns_to: str = "Answer"                     # FillGap hands back to 'Answer' or 'RepairClaims'
    repair_rounds: int = 0                              # whole-answer repairs (misses the question / Not found)
    claims_to_repair: list = field(default_factory=list)
    claims_repaired: bool = False                       # RepairClaims ran: the answer gets one Polish at the end
    claims_repaired_count: int = 0
    searched_for_claims: bool = False                   # claim repair searches at most once
    sections_found_for_claims: list = field(default_factory=list)
    removed_claims: list = field(default_factory=list)  # claims dropped because they stayed unsupported
    verified: set = field(default_factory=set)          # claims the judge accepted (squashed text)
    searches_done: list = field(default_factory=list)   # every agent tool call so far, shown to the next agent
    answer: str = ""
    answer_attempts: int = 0
    style_rewrites: int = 0                             # rewrites asked for a retyped chart (max 1)
    feedback: list = field(default_factory=list)        # for the next Answer attempt
    citation_problems: list = field(default_factory=list)
    citation_error_counts: dict = field(default_factory=dict)   # per kind, over all attempts (copy errors of ids)
    judged_claims: list = field(default_factory=list)
    answers_question: float | None = None               # decision model P(answer responds to the whole question)
    judge_failed: str = ""                              # set when the decision model did not answer
    passed_answer: str = ""                             # last answer that passed CheckCites (a rewrite can be worse)
    passed_judged: list = field(default_factory=list)   # JudgeClaims result for passed_answer
    final_answer: str = ""
    warnings: list = field(default_factory=list)        # shown under the final answer
    trace: list = field(default_factory=list)           # [(node, what happened, seconds)]
    tokens: dict = field(default_factory=dict)          # per run: {"role model": {calls, input, output, usd}}

    live: bool = False                                  # print every step while working
    node_started: float = 0.0
    node_token_mark: int = 0                            # METER position when the node started
    transcript: list = field(default_factory=list)      # the readable trace, line by line (saved by ask --trace)

    # ---- the readable trace: one block per node
    #   ━━ NodeName ━━━━━━
    #   │ label
    #   │     value lines
    #   └─ summary (seconds)
    log_path: object = None                             # a Path: every trace line is appended here at once
    # the same trace as structured events (for the web viewer): written to log_path with suffix .jsonl and
    # passed to event_sink while working. Event types: start, node, show, tool, end, final.
    events: list = field(default_factory=list)
    event_sink: object = None                           # callable(event dict) or None

    def emit(self, event_type, **details):
        event = {"type": event_type, "time": round(time.time(), 2), **details}
        self.events.append(event)
        if self.log_path:
            with open(self.log_path.with_suffix(".jsonl"), "a", encoding="utf-8") as events_file:
                events_file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        if self.event_sink:
            self.event_sink(event)

    def say(self, text):
        self.transcript.append(text)
        if self.live:
            print(text, flush=True)
        if self.log_path:                               # written while working: readable during a run (tail -f)
            with open(self.log_path, "a", encoding="utf-8") as log_file:   # and kept if the run crashes
                log_file.write(text + "\n")

    def header(self, node):
        self.say(f"\n━━ {node} " + "━" * max(3, 70 - len(node)))
        self.emit("node", node=node)

    def show(self, label, value=None, data=None):
        """A labelled detail inside the current node; value may be a string (multi-line) or a list of lines.
        data: the same detail as structured data for the web viewer (probabilities, judged claims)."""
        self.write_thinking()
        self.say(f"│ {label}")
        items = [] if value is None else value if isinstance(value, list) else [value]
        lines = [line for item in items for line in str(item).split("\n")]      # multi-line items keep the │
        self.emit("show", label=label, lines=lines if value is not None else None, data=data)
        if value is None:
            return
        for line in lines or ["(none)"]:
            self.say(f"│     {line}")

    def write_thinking(self):
        """The agents' reasoning since the last trace line, into the trace FILES only (.log and .jsonl). Called
        before every detail, tool call and node end, so it sits where it happened. Not printed, not in the
        transcript, not sent to event_sink: the API's visitors and the live viewer never see it."""
        for agent_name, text in take_thinking():
            if not self.log_path:
                continue
            with open(self.log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"│ thinking ({agent_name}, trace only)\n"
                               + "".join(f"│ ┊   {line}\n" for line in text.split("\n")))
            event = {"type": "thinking", "time": round(time.time(), 2), "agent": agent_name, "text": text}
            with open(self.log_path.with_suffix(".jsonl"), "a", encoding="utf-8") as events_file:
                events_file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def log(self, node, message):
        self.write_thinking()
        seconds = round(time.time() - self.node_started, 1)
        self.trace.append([node, message, seconds])
        tokens = METER.summary(since=self.node_token_mark)      # only nodes that called a model
        self.say(f"└─ {message}   ({seconds}s{', ' + tokens if tokens else ''})")
        self.emit("end", node=node, message=message, seconds=seconds,
                  tokens=[{"role": role, "model": model, **entry}
                          for (role, model), entry in METER.totals(since=self.node_token_mark).items()])


def bar(probability, width=30):
    return "█" * round(probability * width) + "·" * (width - round(probability * width))


def probability_lines(probabilities, marked=None):
    return [f"{name:15} {value:4.2f} {bar(value)}{'   <- ' + marked if name == marked else ''}"
            for name, value in sorted(probabilities.items(), key=lambda item: -item[1])]


def evidence_lines(state):
    return [f"{section_id:6} {books.section_label(section_id)[:70]:70} "
            f"{sum(len(block) for block in books.section_blocks(section_id)):5} chars  via {state.evidence.found_by[section_id]}"
            for section_id in state.evidence.section_ids]


# ------------------------------------------------------------------ understand
class ExtractEntities:
    async def run(self, state):
        if is_self_harm(state.question):
            state.log("ExtractEntities", "self-harm wording -> SafetyReply")
            return SafetyReply()
        if not state.trace:                              # first pass: the dancer's own words, not a split part
            state.language = question_language(state.question)
            if mentions_role_words(state.question):
                state.notes.append(REQUEST_ROLE_WORDS)
        state.entities = extract_entities(state.question)
        state.show("question", state.question)
        for name, value in state.entities.items():
            state.show(f"{name:8} {value or '-'}")
        state.show(f"language {state.language or 'unclear'}")
        state.log("ExtractEntities", "code: fuzzy match against the books' own dance and figure lists")
        return ClassifyKind()


def choose_route(probabilities, confidence, entities):
    """The decision model proposes, code decides. Returns (route, reason)."""
    allowed_kinds = dict(probabilities)
    figures, dances = entities["figures"], entities["dances"]
    if not figures:                                              # rule 1: no figure named -> no figure kinds
        allowed_kinds.pop("FIGURE", None)
        allowed_kinds.pop("FIGURE_COMPARE", None)
    elif len(figures) >= 2 or len(dances) >= 2 or entities["role"] == "both":
        return "FIGURE_COMPARE", "rule 2: two figures, two dances, or man and lady named"
    best_kind = max(allowed_kinds, key=allowed_kinds.get)
    if allowed_kinds[best_kind] < UNSURE_BELOW:                  # rule 3: no kind clearly ahead
        return "UNSURE", f"top kind {best_kind} p {allowed_kinds[best_kind]:.2f} < {UNSURE_BELOW}"
    model_choice = max(probabilities, key=probabilities.get)
    reason = "decision model" if best_kind == model_choice else f"rule 1: no figure named (decision model said {model_choice})"
    return best_kind, reason


def choose_aspect(aspect_answer):
    """The decision model's aspect, or 'all' when it is not clearly ahead (top probability below
    ASPECT_MIN_PROBABILITY): a narrow aspect on a coin toss made the answer skip whole chart sides."""
    probabilities = aspect_answer.get("probabilities", {})
    top_probability = max(probabilities.values(), default=1.0)
    return aspect_answer["choice"] if top_probability >= ASPECT_MIN_PROBABILITY else "all"


class ClassifyKind:
    async def run(self, state):
        try:
            answers = classify_question(state.question)
        except DecisionModelError as error:                       # no decision model: the broadest route
            state.route, state.route_reason = "UNSURE", f"decision model failed: {error}"
            state.notes.append("The question could not be classified; decide yourself what it asks.")
            state.log("ClassifyKind", f"route UNSURE ({state.route_reason})")
            return UnsureAgent()
        state.kind_probabilities = {kind: round(p, 2) for kind, p in answers["kind"]["probabilities"].items()}
        state.kind_confidence = round(answers["kind"]["confidence"], 2)
        state.aspect = choose_aspect(answers["aspect"])
        state.route, state.route_reason = choose_route(state.kind_probabilities, state.kind_confidence, state.entities)
        state.mixed_probability = round(answers.get("mixed", {}).get("noul", 0.0), 2)
        model_choice = answers["kind"]["choice"]
        state.show("decision model: kind", probability_lines(state.kind_probabilities, model_choice),
                   data=probability_data(state.kind_probabilities, model_choice))
        state.show(f"confidence {state.kind_confidence:.2f}  (UNSURE when the top allowed kind is below p {UNSURE_BELOW})")
        aspect_probabilities = answers["aspect"].get("probabilities", {})
        top_aspects = dict(sorted(aspect_probabilities.items(), key=lambda item: -item[1])[:4])
        state.show("decision model: aspect", probability_lines(top_aspects, answers["aspect"]["choice"]),
                   data=probability_data(top_aspects, answers["aspect"]["choice"]))
        state.show(f"aspect used: {state.aspect}" + ("" if state.aspect == answers["aspect"]["choice"] else
                   f"  (top aspect below {ASPECT_MIN_PROBABILITY}: too unsure, so all aspects)"))
        figures, dances = state.entities["figures"], state.entities["dances"]
        state.show("override rules", [
            f"1 no figure named -> no figure kinds          {'APPLIES' if not figures else 'no (figures: ' + ', '.join(figures) + ')'}",
            f"2 two figures/dances or man+lady -> COMPARE   "
            f"{'APPLIES' if figures and (len(figures) >= 2 or len(dances) >= 2 or state.entities['role'] == 'both') else 'no'}",
            f"3 top kind p < {UNSURE_BELOW} -> UNSURE               "
            f"{'APPLIES' if state.route == 'UNSURE' else 'no'}"])
        general_with_figure = state.route == "GENERAL" and bool(figures)
        state.show(f"decision model: technique AND general advice in one question?  P(yes) {state.mixed_probability:.2f} "
                   f"{bar(state.mixed_probability)} (split from {MIXED_FROM})")
        if general_with_figure:
            state.show("GENERAL, but a figure is named: the figure part goes to the books -> split")
        if not state.split_done and (state.mixed_probability >= MIXED_FROM or general_with_figure):
            state.log("ClassifyKind", f"mixed question (P {state.mixed_probability:.2f}"
                                      f"{', GENERAL with a figure' if general_with_figure else ''}) -> SplitQuestion")
            return SplitQuestion()
        state.log("ClassifyKind", f"route {state.route}  ({state.route_reason})")
        return {"FIGURE": FigurePath, "FIGURE_COMPARE": ComparePath, "TECHNIQUE": TechniqueAgent,
                "UNSURE": UnsureAgent, "TERM": TermPath, "GENERAL": GeneralAdvice, "OUT_OF_SCOPE": Refuse,
                "ABOUT": AboutPath, "CHARACTER": CharacterPath}[state.route]()


# ------------------------------------------------------------------ fixed paths (code)
def read_figure_charts(state, figure, dance, role, found_by):
    """The WHOLE chart of one figure in one dance per role: Side 1, Side 2 and all NOTES (the aspect no longer
    drops a side: a step without its other columns, or a chart without its notes, was too thin as evidence)."""
    charts = books.charts_of(figure, dance, role, state.aspect)
    return [chart["section_id"] for chart in charts if state.evidence.add(chart["section_id"], found_by)]


def longer_name_in_dance(figure, dances):
    """A figure found in the question may be part of a longer name in the dance asked: the Samba book has no
    'Whisk', but 'Samba Whisk To Left' and '... To Right'. Returns the first such name, or None.
    Left/right variants are not told apart yet: the judge + FillGap can read the other one."""
    if len(dances) != 1 or dances[0] in books.dances_with_figure(figure):
        return None
    candidates = [name for name in books.figures_of(dances[0]) if figure.lower() in name.lower()]
    return candidates[0] if candidates else None


def figure_mentions(figure, dance, limit=3):
    """Sections of one book whose text names the figure: its words in order, at most one extra word between them
    ('Basic in Place' finds 'Basic Movement in Place'). Only the book's own text: Part 1 is shared by the style."""
    words = [re.escape(word) for word in figure.split()]
    # not followed by 'Position': 'Fan' in Jive is only 'Fan Position', a couple position, not the figure
    pattern = re.compile(r"\b" + r"\W+(?:\w+\W+)?".join(words) + r"\b(?!\s+Pos)", re.I)
    pages = books.library().pages
    found = []
    own_text = pages[(pages.dance == dance) & ~pages.part.fillna("").str.startswith("Part 1")]   # not the shared text
    for section_id, record in own_text.iterrows():
        text = record.text if isinstance(record.text, str) else ""
        if isinstance(record.rows, list):
            text += " " + " ".join(" ".join(map(str, row)) for row in record.rows)
        if pattern.search(text):
            found.append(int(section_id))
    return list(dict.fromkeys(found))[:limit]


FIGURE_MODIFIERS = ("in place", "on the spot", "to left", "to right", "to the left", "to the right", "with turn",
                    "turning left", "turning right", "forward", "backward", "backwards")


def split_modifier(figure):
    """('Basic', 'in place') for 'Basic In Place'; (figure, '') when the name does not end in a modifier."""
    lowered = figure.lower()
    for modifier in sorted(FIGURE_MODIFIERS, key=len, reverse=True):
        if lowered.endswith(" " + modifier) and len(lowered) > len(modifier) + 2:
            return figure[:-len(modifier)].strip(), modifier
    return figure, ""


def base_figures_of(base, dance):
    """Figures of one book that contain the base name ('Basic' -> 'Close Basic Movement', 'Open Basic Movement')."""
    words = base.lower().split()
    names = [name for name in books.figures_of(dance) if all(word in name.lower().split() for word in words)]
    # the plain figure first: '<base> Movement' names, then the shortest ('Cross Basic With Turn' last)
    names.sort(key=lambda name: (f"{base.lower()} movement" not in name.lower(), len(name)))
    return names[:3]


def where_figure_is(figure):
    """"Cha Cha p.94, Rumba p.70" """
    return ", ".join(f"{dance} p.{books.library().knowledge_base.chart_pages_of(figure, dance)[0]}"
                     for dance in books.dances_with_figure(figure))


class FigurePath:
    async def run(self, state):
        figure = state.entities["figures"][0]
        asked = state.entities["dances"]
        longer_name = longer_name_in_dance(figure, asked)
        if longer_name:
            state.show(f"'{figure}' is not in {asked[0]}, but '{longer_name}' is: using that")
        figure = longer_name or figure                              # 'Whisk' in Samba -> 'Samba Whisk To Left'
        available = books.dances_with_figure(figure)
        state.show(f"figure '{figure}' has charts in: {', '.join(available) or 'no book'}")
        state.show(f"dances asked: {', '.join(asked) or 'none (use every dance that has it)'}")
        targets = [dance for dance in asked if dance in available] if asked else available
        if not targets:                                          # no chart in the dance asked
            # The book may still describe it without a chart of its own: Cha Cha p.52, a NOTE of the Time Step:
            # 'danced with all steps in place ... known as "Basic Movement in Place"'. Look before saying no.
            mentions = [(dance, section_id) for dance in asked for section_id in figure_mentions(figure, dance)]
            base, modifier = split_modifier(figure)          # 'Basic In Place' -> 'Basic' danced 'in place'
            base_figures = {dance: base_figures_of(base, dance) for dance in asked} if modifier else {}
            if any(base_figures.values()):
                state.show(f"'{figure}' read as '{base}' + '{modifier}'",
                           [f"{dance}: {', '.join(names)}" for dance, names in base_figures.items() if names])
                state.notes.append(f"'{figure}' reads as the {base} danced {modifier}. Read the chart of the basic "
                                   f"{base} figure in " + "; ".join(f"{dance} ({', '.join(names)})"
                                                                    for dance, names in base_figures.items() if names)
                                   + f" and explain how it is danced {modifier}, with what the text says about it.")
            if mentions or any(base_figures.values()):
                for dance, section_id in mentions:
                    state.evidence.add(section_id, f"FigurePath: '{figure}' named in the {dance} text")
                state.show(f"no chart in {', '.join(asked)}, but the text names '{figure}'",
                           [books.section_label(section_id) for _, section_id in mentions])
                state.notes.append(f"The {', '.join(asked)} book has no chart called '{figure}', but its text names it "
                                   f"(evidence given). Explain from that text what it is in {', '.join(asked)}"
                                   + (f"; it has its own chart in: {where_figure_is(figure)}." if available else "."))
                state.route, state.route_reason = "TECHNIQUE", "figure named in the text, no chart of its own"
                state.log("FigurePath", f"no chart in {asked}, named in {len(mentions)} section(s) -> TECHNIQUE agent")
                return TechniqueAgent()
            state.final_answer = (f"The {', '.join(asked)} book has no figure called '{figure}'. "
                                  f"It has charts in: {where_figure_is(figure)}.")
            state.log("FigurePath", "figure not in the dance asked -> end")
            return Finish()
        role = books.resolve_role(state.entities["role"])
        state.sides = [dict(dance=dance, figure=figure, available=True) for dance in targets]
        state.show(f"role {role or 'both'}, aspect {state.aspect} -> columns {books.ASPECT_COLUMNS.get(state.aspect, 'all')}")
        for dance in targets:
            state.show(f"charts in {dance}", [
                f"{chart['section_id']:6} {chart['header'][:45]:45} p.{chart['page']:<4} "
                f"{'has the aspect' if chart['has_aspect'] else 'no column for the aspect'}"
                for chart in books.charts_of(figure, dance, role, state.aspect)])
        added = [section_id for dance in targets
                 for section_id in read_figure_charts(state, figure, dance, role, "FigurePath")]
        state.show("evidence now", evidence_lines(state))
        if not asked and len(targets) > 1:
            state.notes.append(f"No dance was named; '{figure}' exists in {', '.join(targets)}: answer for each.")
        state.log("FigurePath", f"{figure} in {targets}, role {role}, aspect {state.aspect}: sections {added}")
        return Background()


class ComparePath:
    async def run(self, state):
        figures, dances = state.entities["figures"], state.entities["dances"]
        state.sides = [dict(dance=dance, figure=figure, available=dance in books.dances_with_figure(figure))
                       for figure in figures for dance in (dances or books.dances_with_figure(figure))]
        role = books.resolve_role(state.entities["role"])           # 'both' -> None = read both roles
        for side in state.sides:
            if side["available"]:
                read_figure_charts(state, side["figure"], side["dance"], role, "ComparePath")
        missing = [side for side in state.sides if not side["available"]]
        sides_text = "; ".join(f"{side['dance']} / {side['figure']}" for side in state.sides)
        state.show("sides", [f"{side['dance']:15} {side['figure']:30} {'in the book' if side['available'] else 'NOT in the book'}"
                             for side in state.sides])
        state.show(f"role {role or 'both'}, aspect {state.aspect}")
        state.show("evidence now", evidence_lines(state))
        if missing:                                                  # one side missing -> technique question
            for side in missing:
                state.notes.append(f"The {side['dance']} book has no figure '{side['figure']}' "
                                   f"(it has charts in: {where_figure_is(side['figure'])}). Something with a similar "
                                   f"name may exist in {side['dance']}; if so it is a DIFFERENT thing: say so.")
            state.log("ComparePath", f"sides {sides_text}; missing {len(missing)} -> TECHNIQUE agent")
            state.route, state.route_reason = "TECHNIQUE", "comparison with a missing side"
            return TechniqueAgent()
        roles_read = sorted({record.role for record in (books.library().pages.loc[section_id]
                                                         for section_id in state.evidence.section_ids)
                             if isinstance(record.role, str)})
        focus = (f"The question is mostly about: {state.aspect}. " if state.aspect not in (None, "all") else "")
        state.notes.append(f"Compare these sides: {sides_text}. {focus}Charts were read for: {', '.join(roles_read)}. "
                           f"Name the differences that matter most, step by step, for each role; say in one line "
                           f"per chart side what is the same. Skip identical details and wording-only differences. "
                           f"Short and to the point beats complete.")
        state.log("ComparePath", f"sides {sides_text}, aspect {state.aspect}: {len(state.evidence)} sections")
        return Answer()


class TermPath:
    async def run(self, state):
        from dance_rag_agentic.ingest.text_utils import tokenize
        entries = books.book_index()
        entries = entries[entries.kind != "figure"]
        if state.entities["dances"]:
            entries = entries[entries.dance.isin(state.entities["dances"])]
        names = {clean_name(name): name for name in entries.name.unique() if clean_name(name)}
        matched = find_names(tokenize(state.question), names)
        state.show("book index entries matching the question", matched or ["none"])
        for entry in entries[entries.name.isin(matched)].itertuples():
            state.show(f"'{entry.name}' -> {entry.dance} p.{entry.best_page}")
            for section_id in books.sections_on_page(entry.dance, entry.best_page, kind="prose"):
                state.evidence.add(section_id, f"TermPath: index '{entry.name}'")
        # the index misses many headings: "hand level" is not in it, but section "Hand Levels" (2 pages after
        # the index's "Lead - Hold Shaping") holds the answer. Headings of 2+ words named in the question are read too.
        pages = books.library().pages
        prose = pages[pages.kind == "prose"]
        if state.entities["dances"]:
            prose = prose[prose.dance.isin(state.entities["dances"])]
        # a unit holds several headings ('headings': 'Shaping; Hand Levels'): each one points to its unit
        heading_names = prose.headings.str.split("; ").explode()
        headings = {clean_name(header): header for header in heading_names.dropna().unique()
                    if isinstance(header, str) and len(header.split()) >= 2 and clean_name(header)}
        matched_headings = find_names(tokenize(state.question), headings)
        state.show("section headings named in the question", matched_headings or ["none"])
        added_by_heading = 0
        for heading in matched_headings:
            holder = (prose.headings.str.split("; ").apply(lambda names: heading in names)
                      if "headings" in prose.columns else prose.header == heading)
            section_id = int(prose.index[holder][0])                          # same text in every book: one copy
            if section_id not in state.evidence and added_by_heading < TERM_HEADING_SECTIONS:
                state.evidence.add(section_id, f"TermPath: heading '{heading}'")
                added_by_heading += 1
        state.show("evidence now", evidence_lines(state))
        if not len(state.evidence):
            state.log("TermPath", f"no index entry in the question (matched {matched}) -> TECHNIQUE agent")
            return TechniqueAgent()
        state.log("TermPath", f"index entries {matched}: sections {state.evidence.section_ids}")
        return Answer()


# terms worth explaining come from these chart columns (positions, actions, hips, lead); directions,
# foot placement, amounts of turn and timing are what the chart itself already says plainly
BACKGROUND_COLUMN_WORDS = ("position", "action", "hip", "lead", "shaping", "design", "sway", "body")


def chart_terms(section_ids, limit=BACKGROUND_TERMS):
    """Distinct short values of the concept columns of the charts read: 'Fan Left Angle', 'Cuban Motion'.
    'A then B' gives A and B; long values (a whole lead description) and numbers are skipped."""
    terms = []
    for section_id in section_ids:
        if books.library().pages.loc[section_id, "kind"] != "chart":
            continue
        for block in books.section_blocks(section_id):
            for line in block.split("\n")[1:]:
                column, _, value = line.partition(":")
                if not value or not any(word in column.lower() for word in BACKGROUND_COLUMN_WORDS):
                    continue
                for part in re.split(r"\s+then\s+|/|;|,", value):
                    part = part.split("(")[0].strip(" .)")          # 'To Left (Technical Sway)' -> 'To Left'
                    if part.lower().startswith(("to left", "to right", "left", "right")) and len(part.split()) <= 2:
                        continue                                      # a direction, nothing to explain
                    if part and 1 <= len(part.split()) <= 4 and not re.search(r"\d", part) and part not in terms:
                        terms.append(part)
    return terms[:limit]


class Background:
    """FIGURE route: the charts say WHAT to do step by step; a learner also needs to know what the positions and
    actions in them mean. A small agent reads the prose that explains the chart's own terms."""
    async def run(self, state):
        # A narrow question (given steps, or one aspect such as timing) gets a short answer without the teacher
        # parts (missing_parts): background prose would not be used, and the agent cost ~10 s (6 tool rounds).
        if state.entities.get("steps") or state.aspect not in (None, "all"):
            state.log("Background", f"narrow question (steps {state.entities.get('steps') or '-'}, aspect "
                                    f"{state.aspect}): no background needed -> Answer")
            return Answer()
        terms = chart_terms(state.evidence.section_ids)
        state.show("chart terms handed to the agent", terms or ["none"])
        if not terms:
            state.log("Background", "no chart terms to explain -> Answer")
            return Answer()
        extra = REQUEST_BACKGROUND.format(terms="; ".join(terms))
        await _run_agent(state, "Background agent", prompt(BACKGROUND_PROMPT), _agent_request(state, extra),
                         BACKGROUND_TOOL_CALLS)
        return Answer()


def _split_lines(reply):
    """'BOOKS: ...' / 'GENERAL: ...' -> (books part, general part); 'NONE' or a missing line -> ''."""
    parts = {}
    for line in reply.splitlines():
        label, _, text = line.strip().strip("*").partition(":")
        label = label.strip(" *").upper()
        if label in ("BOOKS", "GENERAL"):
            text = text.strip(" *")
            parts[label] = "" if text.upper().rstrip(".") == "NONE" else text
    return parts.get("BOOKS", ""), parts.get("GENERAL", "")


class SplitQuestion:
    """A question with a technique part and a general-advice part: the general part goes to GeneralAdvice,
    the books part takes the normal route (ExtractEntities + ClassifyKind again, without a second split)."""
    async def run(self, state):
        state.split_done = True
        state.original_question = state.original_question or state.question
        try:
            reply = (await run_counted(split_writer(), f"QUESTION: {state.question}")).output
        except CloudAccessError:
            raise
        except Exception as error:
            reply = ""
            state.show(f"split writer FAILED: {type(error).__name__}: {error}"[:200])
        state.show("split (raw)", reply or "(none)")
        books_part, general_part = _split_lines(reply)
        state.show("books part", books_part or "none")
        state.show("general part", general_part or "none")
        if not general_part:                             # nothing general after all: route the whole question
            state.log("SplitQuestion", "no general part -> route the question as it is")
            return ClassifyKind()
        state.general_question = general_part
        if books_part:
            state.question = books_part
            state.books_part_pending = True
            state.notes.append(f"The dancer's full question was: '{state.original_question}'. A general-advice part "
                               f"('{general_part}') is answered separately; answer only this question.")
        state.log("SplitQuestion", f"general part -> GeneralAdvice{', then the books part' if books_part else ''}")
        return GeneralAdvice()


class GeneralAdvice:
    """Own-knowledge advice for a learner (no evidence, no ids, not judged). Shown under GENERAL_HEADING so it is
    never mistaken for the books. Also the landing place when the UNSURE agent found nothing in the books."""
    async def run(self, state):
        question = state.general_question or state.question
        request = [f"QUESTION: {question}"]
        if state.books_part_pending:
            request.append(REQUEST_GENERAL_BOOKS_PART.format(question=state.question))
        if state.searches_done and not len(state.evidence):
            request.append(REQUEST_GENERAL_NOTHING_IN_BOOKS)
        request += language_lines(state)
        state.show("request to the general-advice writer", request)
        try:
            state.general_answer = (await run_counted(general_writer(), "\n".join(request))).output.strip()
        except CloudAccessError:
            raise
        except Exception as error:
            state.general_answer = ""
            state.show(f"general writer FAILED: {type(error).__name__}: {error}"[:200])
        state.show("general advice", state.general_answer or "(none)")
        if state.books_part_pending:
            state.books_part_pending = False
            state.log("GeneralAdvice", "done -> route the books part")
            return ExtractEntities()
        if not state.general_answer and not state.answer:
            state.final_answer = out_of_scope_answer()
        state.log("GeneralAdvice", "done -> Finish")
        return Finish()


def about_facts():
    """What the assistant is and knows, from the index itself (a new book shows up without code changes)."""
    library = books.library()
    figures = {dance: len(books.figures_of(dance)) for dance in library.dances}
    styles = books.dances_by_style()
    lines = ["I answer questions about ballroom and Latin dance technique from the WDSF technique books "
             "(World DanceSport Federation), one book per dance.",
             f"Books ({len(library.dances)} dances): " + "; ".join(
                 f"{style}: " + ", ".join(f"{dance} ({figures[dance]} figures)" for dance in dances)
                 for style, dances in styles.items()) + ".",
             "Each book has general principles (posture, foot actions, timing and music, lead and follow, sway, "
             "rise and fall and more), the specific principles of that dance, and step-by-step charts for each "
             "figure, for the man and the lady (feet, alignment, turn, timing, positions, hips, rise, sway).",
             "Questions I can answer: how to dance a figure or a step of it; the difference between two figures, "
             "two dances, or the man's and the lady's part; technique and principles; the music and tempo of a "
             "dance; what a term or abbreviation means; which figures a dance has or which may follow another.",
             "The character or feel of a dance: what the books say shapes it (music, typical actions), plus the "
             "feel itself from general dance knowledge, marked as not from the books.",
             "Every answer is checked against the books and shows its sources (book and page). I also give some "
             "general tips for learners (practising, getting started), marked as not from the books.",
             "I cannot help with other dance styles (such as Salsa, Bachata, Argentine Tango or ballet), dance "
             "history, competition results or judging, or topics outside dancing."]
    return lines


class AboutPath:
    """Questions about the assistant itself ('What can you answer?', 'Which dances do you know?'): code gathers the
    facts from the index, the writer phrases them in the question's language. No book evidence, nothing to judge."""
    async def run(self, state):
        facts = about_facts()
        state.show("facts about the assistant (from the index)", facts)
        request = "\n\n".join([f"QUESTION: {state.question}", "FACTS:\n- " + "\n- ".join(facts)]
                              + language_lines(state))
        try:
            state.final_answer = (await run_counted(about_writer(), request)).output.strip()
        except CloudAccessError:
            raise
        except Exception as error:
            state.final_answer = "\n".join(facts)               # the facts themselves are a usable answer
            state.show(f"about writer FAILED: {type(error).__name__}: {error}"[:200])
        state.log("AboutPath", f"{len(facts)} facts from the index -> answer")
        return Finish()


class Refuse:
    async def run(self, state):
        state.final_answer = out_of_scope_answer()
        state.log("Refuse", "fixed text, no LLM")
        return Finish()


class SafetyReply:
    """Self-harm or suicide wording: a fixed, caring reply with crisis contacts. No model, no books."""
    async def run(self, state):
        state.route, state.route_reason = "SAFETY", "self-harm wording (code)"
        state.final_answer = SAFETY_REPLY
        state.log("SafetyReply", "fixed text with crisis contacts, no LLM")
        return Finish()


CHARACTER_SECTION_HEADERS = r"Music in Dance ?sport|Musical Structure$|specific principles|Interpretation"
CHARACTER_SECTION_MIN_CHARACTERS = 150          # 'Part 2 - Cha Cha specific principles' alone is a 111-char heading


def character_sections(dance):
    """The sections that state what shapes a dance's character: its music facts (tempo, accents), its own
    principles (part 2 opening) and its interpretation chapter. Code picks them: the agent searched 'vibe' and
    'character', found nothing, and the answer turned delayed-walk rules into a mood (demo trace 2026-11-15)."""
    pages = books.library().pages
    rows = pages[(pages.dance == dance) & (pages.kind == "prose")
                 & pages.header.str.contains(CHARACTER_SECTION_HEADERS, case=False, na=False)]
    if dance != "Paso Doble":                    # the Paso Doble music chapter is printed in every Latin book
        rows = rows[~rows.header.str.contains("Paso Doble", na=False)]
    return [section_id for section_id in rows.sort_values("page").index
            if sum(len(block) for block in books.section_blocks(section_id)) >= CHARACTER_SECTION_MIN_CHARACTERS]


def same_text_as(section_id):
    """A section printed in several books points to its first copy (copy_of); the copy is the same text."""
    copy_of = books.library().pages.loc[section_id].copy_of
    if copy_of is None or copy_of != copy_of:                        # None or NaN: the section is the original
        return section_id
    return int(copy_of)


class CharacterPath:
    """The feel of a dance. Books part: code reads the music, principles and interpretation sections of each named
    dance, a short agent adds the dance's typical actions, the writer states what SHAPES the character (with ids).
    Feel part: the character writer, from own knowledge, under CHARACTER_HEADING (never as the books)."""
    async def run(self, state):
        state.general_heading = CHARACTER_HEADING
        read_texts = {same_text_as(section_id) for section_id in state.evidence.section_ids}
        added = []
        for dance in state.entities["dances"][:3]:
            for section_id in character_sections(dance):
                if same_text_as(section_id) in read_texts:   # same music chapter in Cha Cha and Rumba: once
                    continue
                read_texts.add(same_text_as(section_id))
                state.evidence.add(section_id, f"CharacterPath: {dance} music / principles / interpretation")
                added.append(section_id)
        state.show("character sections read by code", [books.section_label(section_id) for section_id in added]
                   or ["none (no dance named: the agent searches)"])
        request = [f"QUESTION: {state.question}"] + language_lines(state)
        feel_task = run_counted(character_writer(), "\n".join(request))
        agent_task = _run_agent(state, "CHARACTER agent", prompt(CHARACTER_PROMPT), _agent_request(state),
                                CHARACTER_TOOL_CALLS)
        feel_result, _ = await asyncio.gather(feel_task, agent_task, return_exceptions=True)
        if isinstance(feel_result, CloudAccessError):
            raise feel_result
        if isinstance(feel_result, BaseException):
            state.show(f"character writer FAILED: {type(feel_result).__name__}: {feel_result}"[:200])
        else:
            state.general_answer = feel_result.output.strip()
        state.show("the feel (own knowledge)", state.general_answer or "(none)")
        if not len(state.evidence):
            state.final_answer = in_language(CHARACTER_NOTHING, state.language)
            state.log("CharacterPath", "nothing read from the books -> only the feel part")
            return Finish()
        state.notes.append(REQUEST_CHARACTER_NOTE)
        state.log("CharacterPath", f"{len(added)} sections by code, {len(state.evidence)} in all -> Answer")
        return Answer()


# ------------------------------------------------------------------ searching agents
def _agent_request(state, extra=""):
    lines = [f"QUESTION: {state.question}",
             REQUEST_ENTITIES.format(entities=state.entities)]
    if not state.entities["figures"]:
        candidates = figures_ending_in_question_word(state.question, state.entities["dances"])
        if candidates:
            lines.append(REQUEST_FIGURE_CANDIDATES + "; ".join(candidates))
    if len(state.entities["dances"]) >= 2 and not state.entities["figures"]:
        # one sub-question per named dance (comparison research, DER-RAG 2026): runs read one dance well and the
        # other only by chance ('the books contain no description of Rumba')
        lines.append(REQUEST_COMPARISON + "; ".join(state.entities["dances"]))
    if state.notes:
        lines.append(REQUEST_NOTES + " ".join(state.notes))
    if state.searches_done:
        lines.append(REQUEST_SEARCHES_DONE + "\n  ".join(state.searches_done))
    if len(state.evidence):
        lines.append(REQUEST_ALREADY_READ +
                     "; ".join(f"{section_id} = {books.section_label(section_id)}" for section_id in state.evidence.section_ids))
    return "\n".join(lines + ([extra] if extra else []))


def _report_tool_call(state, call, result):
    lines = str(result).split("\n")
    shown = lines[:TRACE_RESULT_LINES] + ([f"... ({len(lines) - TRACE_RESULT_LINES} more lines)"]
                                          if len(lines) > TRACE_RESULT_LINES else [])
    state.write_thinking()                   # the reasoning that chose this call goes right above it
    state.say(f"│ ▸ {call}")
    for line in shown:
        state.say(f"│       {line[:150]}")
    state.emit("tool", call=call, lines=lines)


def agent_pointers(note):
    """'FOUND: ... [44.30-44.31] ... MISSING: x' -> 'key blocks: [44.30, 44.31]; missing: x'."""
    found_ids = cited_ids(expand_ranges(note)[0])
    missing = re.search(r"MISSING:\s*(.+)", note, re.S)
    missing_text = missing.group(1).strip().split("\n")[0] if missing else ""
    parts = []
    if found_ids:
        parts.append("key blocks it found: [" + ", ".join(f"{s}.{b}" for s, b in sorted(found_ids)) + "]")
    if missing_text and missing_text.lower().strip(" .") != "nothing":
        parts.append(f"it could not find: {missing_text}")
    return "; ".join(parts) or "no pointers"


async def _run_agent(state, node_name, prompt, request, tool_calls_limit):
    deps = AgentDeps(evidence=state.evidence, caller=node_name, role=books.resolve_role(state.entities.get("role")),
                     report=lambda call, result: _report_tool_call(state, call, result))
    before = list(state.evidence.section_ids)
    state.show(f"request to the agent (tool call limit {tool_calls_limit})", request)
    state.show("tool calls, with the start of each result:")
    note, calls, stop_reason = await run_search(prompt, request, deps, tool_calls_limit)
    state.searches_done += calls
    if note:                                 # the writer gets the agent's pointers, not its prose: that prose
        state.notes.append(f"{node_name}: {agent_pointers(note)}")   # leaked ranges and invented descriptions
    state.show("agent's note to the graph", note.strip() or "(none)")
    state.show("new evidence", [line for line, section_id in zip(evidence_lines(state), state.evidence.section_ids)
                                if section_id not in before] or ["none"])
    state.log(node_name, f"{len(calls)} tool calls, {len(state.evidence) - len(before)} new sections, {stop_reason}")


class TechniqueAgent:
    async def run(self, state):
        await _run_agent(state, "TECHNIQUE agent", prompt(TECHNIQUE_PROMPT), _agent_request(state), AGENT_TOOL_CALLS)
        return Answer()


class UnsureAgent:
    async def run(self, state):
        extra = (f"The classifier's guesses (probabilities): {state.kind_probabilities}; "
                 f"aspect guess: {state.aspect}.")
        if state.entities["dances"] and not state.entities["figures"]:
            # Often the question names a figure the code did not recognise ('overturn' -> Overturned Running Spin
            # Turn): give the dance's figure list up front instead of letting the agent search for it word by word.
            extra += "".join(REQUEST_FIGURE_LIST.format(dance=dance, figures=", ".join(books.figures_of(dance)))
                             for dance in state.entities["dances"][:2])
        await _run_agent(state, "UNSURE agent", prompt(UNSURE_PROMPT), _agent_request(state, extra), AGENT_TOOL_CALLS)
        if not len(state.evidence) and not state.general_answer:   # the books have nothing: a kind reply, not 'Not found'
            state.show("the agent read nothing from the books -> general-advice writer")
            return GeneralAdvice()
        return Answer()


# ------------------------------------------------------------------ answer, checks, repair
class Answer:
    async def run(self, state):
        state.answer_attempts += 1
        parts = [f"QUESTION: {state.question}"]
        if state.notes:
            parts.append(REQUEST_ANSWER_NOTES + "\n- ".join(state.notes))
        if state.feedback:
            parts.append(REQUEST_REWRITE.format(answer=state.answer, problems="\n- ".join(state.feedback)))
        parts.append("EVIDENCE:\n" + (state.evidence.as_text() if len(state.evidence) else "(none)"))
        parts += language_lines(state)
        state.show(f"attempt {state.answer_attempts} of max {ANSWER_ATTEMPTS}")
        if state.notes:
            state.show("notes given to the writer", state.notes)
        if state.feedback:
            state.show("REWRITE - problems the writer must fix", state.feedback)
        state.show(f"evidence given: {len(state.evidence)} sections", evidence_lines(state))
        try:
            state.answer = (await run_counted(answer_writer(), "\n\n".join(parts))).output
        except CloudAccessError:
            raise
        except Exception as error:
            state.answer = state.answer or f"Not found in the books. (answer step failed: {type(error).__name__})"
            state.show(f"answer writer FAILED: {type(error).__name__}: {error}"[:200])
        state.feedback = []
        state.show("answer (raw, with evidence ids)", state.answer)
        state.log("Answer", f"attempt {state.answer_attempts}, {len(state.answer)} characters")
        return CheckCites()


STEP_LINE = re.compile(r"^\s*(?:[-*•]\s*)?(?:\*\*)?(?:step\s*)?(?:\d+|prec\.?)\b(?!\s*/)", re.I)
MAX_STEP_LINES = 8


def retyped_chart(state):
    """A 'how do I dance X' answer that lists the chart line by line (samba runs: 36 step lines, 8,500
    characters) although the tables are shown below it. Not when the question names steps or one aspect."""
    if state.entities.get("steps") or state.aspect not in (None, "all"):
        return None
    step_lines = sum(bool(STEP_LINE.match(line)) for line in state.answer.splitlines())
    if step_lines <= MAX_STEP_LINES:
        return None
    return FEEDBACK_RETYPED_CHART.format(step_lines=step_lines)


TEACHER_PARTS = ("What it is", "How to do it", "What to watch")


def missing_parts(state):
    """The three teacher parts the answer prompt asks for (user: "a more human way to answer"). The review of 13
    book answers found them missing or renamed in about 11. Not for 'Not found' answers, nor for narrow questions
    (given steps or one aspect): a part with nothing to say became "the books do not give ..." (blind review)."""
    # narrow = about given steps, or one column of a figure's chart. The aspect only means something for figure
    # routes: "What is sway?" got aspect 'sway', counted as narrow, and a two-line answer passed unchallenged.
    figure_route = state.route in ("FIGURE", "FIGURE_COMPARE")
    narrow = bool(state.entities.get("steps")) or (figure_route and state.aspect not in (None, "all"))
    if "not found in the books" in state.answer.lower() or narrow or _non_english_question(state.question):
        return None                          # a non-English answer translates the headings ('Wat het is')
    missing = [part for part in TEACHER_PARTS if part.lower() not in state.answer.lower()]
    if not missing:
        return None
    return FEEDBACK_MISSING_PARTS.format(missing=", ".join(missing))


def _non_english_question(question):
    """Dutch/other-language questions get translated headings ('Wat het is'): don't demand the English ones."""
    return question_language(question) not in (None, "English")


class CheckCites:
    async def run(self, state):
        state.answer = clean_not_found(strip_answer_label(normalize_brackets(state.answer)))
        state.answer, expanded = expand_ranges(state.answer)
        if expanded:
            state.show("ranges listed by code (no rewrite needed)", expanded)
        state.answer, folded = fold_single_block_ids(state.answer)
        state.answer, repeated = drop_repeated_lines(state.answer)
        if repeated:
            state.show("lines that repeat an earlier fact, dropped by code", repeated)
        if folded:
            state.show("chart steps cited as blocks, mapped to the chart's one block (no rewrite needed)", folded)
        state.citation_problems = citation_problems(state.answer, state.evidence)
        for problem in state.citation_problems:            # measured, to see whether 6-digit ids get miscopied
            kind = ("id not read" if "is not in the evidence" in problem else "no such block" if "has no block" in problem
                    else "range" if "no ranges" in problem else "uncited fact" if "without an id" in problem else "other")
            state.citation_error_counts[kind] = state.citation_error_counts.get(kind, 0) + 1
        state.show("cited ids", sorted(f"{section_id}.{block}" for section_id, block in cited_ids(state.answer)) or ["none"])
        state.show("problems", state.citation_problems or ["none"])
        if not state.citation_problems:
            state.passed_answer, state.passed_judged = state.answer, []
            state.show("this answer is saved as the last one that passed the citation check")
        style_problems = [problem for problem in (retyped_chart(state), missing_parts(state)) if problem]
        if style_problems and not state.citation_problems and state.style_rewrites < 1:
            state.style_rewrites += 1
            state.feedback = style_problems
            state.show("style", style_problems)
            state.log("CheckCites", "style (chart retyped / teacher parts missing) -> one rewrite")
            return Answer()
        if state.citation_problems and state.answer_attempts < ANSWER_ATTEMPTS:
            state.feedback = state.citation_problems
            state.log("CheckCites", f"REJECTED -> rewrite (attempt {state.answer_attempts + 1} of max {ANSWER_ATTEMPTS})")
            return Answer()
        if state.citation_problems:
            state.log("CheckCites", f"problems remain, no attempts left (max {ANSWER_ATTEMPTS}) -> "
                                    f"{'JudgeClaims' if state.use_judge else 'Finish'}")
        else:
            state.log("CheckCites", f"passed -> {'JudgeClaims' if state.use_judge else 'Finish'}")
        return JudgeClaims() if state.use_judge else Finish()


def probability_data(probabilities, marked=None):
    return {"kind": "probabilities", "marked": marked,
            "items": sorted(([name, value] for name, value in probabilities.items()), key=lambda item: -item[1])}


def claims_data(judged_claims):
    return {"kind": "claims", "threshold": JUDGE_SUPPORTED_FROM,
            "items": [{"claim": claim["claim"], "p": claim["p_supported"], "verdict": claim.get("verdict"),
                       "ids": [f"{section_id}.{block}" for section_id, block in claim["ids"]]}
                      for claim in judged_claims]}


def judged_lines(judged_claims):
    return [f"{'ok ' if claim not in unsupported([claim]) else 'NO '} {VERDICT_SHORT.get(claim.get('verdict'), '')}"
            f" p={claim['p_supported']:.2f} "
            f"{bar(claim['p_supported'], 15)}  {claim['claim'][:90]}   "
            f"[{', '.join(f'{section_id}.{block}' for section_id, block in claim['ids'])}]"
            for claim in judged_claims] or ["no claims found"]


VERDICT_SHORT = {"SUPPORTED": "supported   ", "CONTRADICTED": "CONTRADICTED", "NOT_IN_EVIDENCE": "NOT IN EVID."}


def same_claim(text):
    """Claims are compared without spaces, case and punctuation (the polish step may re-space them)."""
    return books.squash(text)


class JudgeClaims:
    """Every cited claim against its blocks + does the answer respond to the whole question.
    Failing claims are repaired ONE BY ONE (RepairClaims); the rest of the answer stays as it is.
    Only a whole-answer problem (misses part of the question, 'Not found') rewrites the whole answer."""
    async def run(self, state):
        try:
            if decision_parallel() > 1:                 # cloud: the claim batches and this check at the same time
                state.judged_claims, state.answers_question = await asyncio.gather(
                    asyncio.to_thread(judge_claims, state.answer),
                    asyncio.to_thread(answers_question, state.question, state.answer))
            else:                                       # local decision model: one job at a time
                state.judged_claims = judge_claims(state.answer)
                state.answers_question = answers_question(state.question, state.answer)
        except DecisionModelError as error:
            state.judge_failed = str(error)
            state.log("JudgeClaims", f"skipped: {error}")
            return Finish()
        if state.answer == state.passed_answer:
            state.passed_judged = state.judged_claims
        bad_claims = unsupported(state.judged_claims)
        state.verified |= {same_claim(claim["claim"]) for claim in state.judged_claims if claim not in bad_claims}
        says_not_found = "not found in the books" in state.answer.lower()
        misses_question = state.answers_question < ANSWERS_QUESTION_FROM
        if state.route == "CHARACTER":
            # the books part only states what shapes the character; the feel is answered under its own heading.
            # A judge reading the books part alone says it misses the question: searching again cannot fix that.
            if says_not_found:
                state.answer, state.final_answer = "", in_language(CHARACTER_NOTHING, state.language)
                state.log("JudgeClaims", "CHARACTER: books part says Not found -> only the feel part")
                return Finish()
            misses_question = False
        state.show(f"claims judged by the decision model (unsupported below {JUDGE_SUPPORTED_FROM})",
                   judged_lines(state.judged_claims), data=claims_data(state.judged_claims))
        state.show(f"does the answer respond to everything the question asks?  P(yes) {state.answers_question:.2f} "
                   f"{bar(state.answers_question)} (repair below {ANSWERS_QUESTION_FROM})")
        if state.judged_claims and "judge_input" in state.judged_claims[0]:   # exactly what the decision model read
            from dance_rag_agentic.agent.prompts import CLAIM_CRITERIA, CLAIM_INSTRUCTIONS
            judge_inputs = list(dict.fromkeys(claim["judge_input"] for claim in state.judged_claims))
            for number, judge_input in enumerate(judge_inputs, start=1):
                claims_here = sum(claim["judge_input"] == judge_input for claim in state.judged_claims)
                state.show(f"judge input, batch {number} of {len(judge_inputs)} ({claims_here} claims, "
                           f"{len(judge_input)} characters)", judge_input)
            state.show("judge input: question per claim", CLAIM_INSTRUCTIONS.format(ids="[ids]", claim="<claim>"))
            state.show("judge input: verdicts", [f"{answer}: {text}" for answer, text in CLAIM_CRITERIA.items()])
        if says_not_found:
            state.show("the answer says 'Not found in the books'")

        # whole-answer problems: search for what is missing, then a new answer (max REPAIR_ROUNDS)
        if (misses_question or says_not_found) and state.repair_rounds < REPAIR_ROUNDS:
            searched_and_found_nothing = not len(state.evidence) and state.route in ("TECHNIQUE", "UNSURE")
            if searched_and_found_nothing and says_not_found and not misses_question:
                state.log("JudgeClaims", "'Not found' after an agent found nothing at all: searching again won't help -> Finish")
                return Finish()
            state.repair_rounds += 1
            state.to_fill = ([FILL_MISSING_PART.format(question=state.question)] if misses_question else
                             [FILL_NOT_FOUND])
            state.feedback = [FEEDBACK_MISSES_QUESTION] if misses_question else []
            state.fill_returns_to = "Answer"
            state.log("JudgeClaims", f"{'misses (part of) the question' if misses_question else 'says Not found'} -> "
                                     f"FillGap, then a new answer (round {state.repair_rounds} of max {REPAIR_ROUNDS})")
            return FillGap()
        if bad_claims:                                     # repair only these claims, keep the rest
            state.claims_to_repair = bad_claims
            state.log("JudgeClaims", f"{len(bad_claims)} of {len(state.judged_claims)} claims unsupported -> "
                                     f"RepairClaims (the other claims stay as they are)")
            return RepairClaims()
        state.log("JudgeClaims", f"{len(state.judged_claims)} claims, all supported -> "
                                 f"{'Polish' if state.claims_repaired else 'Finish'}")
        return Polish() if state.claims_repaired else Finish()


def _blocks_text(evidence_ids):
    return "\n".join(f"[{section_id}.{block}] ({books.section_label(section_id)}) {block_text((section_id, block))}"
                     for section_id, block in evidence_ids if block_text((section_id, block)))


class RepairClaims:
    """Each failing claim is rewritten to what its blocks say, removed, or marked NEED_EVIDENCE. Only the new
    sentences are judged. NEED_EVIDENCE (or a rewrite that fails again) -> one targeted FillGap -> repair once more;
    whatever still fails after that is removed and listed as a warning."""
    async def run(self, state):
        new_evidence_ids = [(section_id, number) for section_id in state.sections_found_for_claims
                            for number in range(1, len(books.section_blocks(section_id)) + 1)]
        # every table the claims cite, each ONCE and whole, cited rows marked '>>' (two claims on one table: one copy)
        cited_by_claims = [evidence_id for claim in state.claims_to_repair for evidence_id in claim["ids"]]
        request = [f"QUESTION: {state.question}"] + language_lines(state) + [REQUEST_REPAIR_EVIDENCE]
        request += [f"({unit_label(unit)})\n{unit_text(unit, cited)}"
                    for unit, cited in table_units(cited_by_claims, scope="section").items()]
        for number, claim in enumerate(state.claims_to_repair, start=1):
            judged_as = REQUEST_REPAIR_VERDICT.get(claim.get("verdict"), "")
            request.append(f"\nCLAIM {number}: {claim['claim']}   (it cites "
                           f"{', '.join(f'[{section_id}.{block}]' for section_id, block in claim['ids'])}"
                           f"{'; ' + judged_as if judged_as else ''})")
        if new_evidence_ids:
            request.append(REQUEST_REPAIR_NEW_EVIDENCE + _blocks_text(new_evidence_ids))
        state.show("claims to repair", [f"{number}: {claim['claim']}" for number, claim in
                                        enumerate(state.claims_to_repair, start=1)])
        try:
            reply = (await run_counted(repair_writer(), "\n".join(request))).output
        except CloudAccessError:
            raise
        except Exception as error:
            reply = ""
            state.show(f"repair writer FAILED: {type(error).__name__}: {error}"[:200])
        state.show("repair decisions (raw)", reply or "(none)")
        decisions = {int(match.group(1)): match.group(2).strip()
                     for match in re.finditer(r"^\s*(\d+)\s*[:.)]\s*(.+)$", reply, re.MULTILINE)}
        replacements, still_missing = [], []
        for number, claim in enumerate(state.claims_to_repair, start=1):
            decision = decisions.get(number, "NEED_EVIDENCE")
            if decision.upper().startswith("NEED_EVIDENCE"):
                still_missing.append(claim)
            elif decision.upper().startswith("REMOVE"):
                state.answer, _ = replace_claim(state.answer, claim["claim"], "")
                state.removed_claims.append(claim["claim"])
            else:
                state.answer, found = replace_claim(state.answer, claim["claim"], decision)
                if found:
                    replacements.append(decision)
                else:
                    state.show(f"could not find the claim in the answer to replace it: '{claim['claim'][:60]}'")
        new_judged = []
        if replacements:                                   # judge ONLY the new sentences
            try:
                new_judged = judge_claims("\n".join(replacements))
            except DecisionModelError as error:
                state.judge_failed = str(error)
            state.show("new sentences judged", judged_lines(new_judged), data=claims_data(new_judged))
        state.verified |= {same_claim(claim["claim"]) for claim in new_judged if claim not in unsupported(new_judged)}
        repaired_texts = {same_claim(claim["claim"]) for claim in state.claims_to_repair}
        state.judged_claims = [claim for claim in state.judged_claims          # old verdicts of repaired claims go,
                               if same_claim(claim["claim"]) not in repaired_texts] + new_judged   # new ones count
        state.claims_repaired_count += len(replacements)
        still_missing += unsupported(new_judged)
        state.claims_repaired = True
        state.show("answer after the repairs", state.answer)
        if still_missing and not state.searched_for_claims:
            state.searched_for_claims = True
            state.claims_to_repair = still_missing
            state.to_fill = [f"evidence for this claim: '{claim['claim']}'" for claim in still_missing]
            state.fill_returns_to = "RepairClaims"
            state.log("RepairClaims", f"{len(replacements)} rewritten, {len(state.removed_claims)} removed, "
                                      f"{len(still_missing)} need evidence -> FillGap for these claims only")
            return FillGap()
        for claim in still_missing:                        # searched once already: drop what still fails
            state.answer, _ = replace_claim(state.answer, claim["claim"], "")
            state.removed_claims.append(claim["claim"])
        state.log("RepairClaims", f"{len(replacements)} rewritten, {len(state.removed_claims)} removed -> Polish")
        return Polish()


class FillGap:
    """Searches evidence for exactly what failed (claims, or the missing part of the question), then returns to
    the step that asked: RepairClaims (claim repair) or Answer (whole-answer problem)."""
    async def run(self, state):
        sections_before = list(state.evidence.section_ids)
        state.show("to fill", state.to_fill)
        extra = REQUEST_FILLGAP + "\n- ".join(state.to_fill)
        await _run_agent(state, "FillGap", prompt(FILLGAP_PROMPT), _agent_request(state, extra), FILLGAP_TOOL_CALLS)
        new_sections = [section_id for section_id in state.evidence.section_ids if section_id not in sections_before]
        if state.fill_returns_to == "RepairClaims":
            state.sections_found_for_claims = new_sections
            return RepairClaims()
        state.feedback.append(FEEDBACK_NEW_EVIDENCE.format(count=len(new_sections)) if new_sections
                              else FEEDBACK_NO_NEW_EVIDENCE)
        state.answer_attempts = 0                       # a new answer gets its own CheckCites rewrites
        return Answer()


class Polish:
    """One tidy-up of the whole answer after claim repairs. Code checks the ids; only sentences that are new
    after polishing are judged. If anything gets worse, the unpolished answer is kept."""
    async def run(self, state):
        before = state.answer
        try:
            request = "\n\n".join([f"QUESTION: {state.question}", f"ANSWER:\n{before}"] + language_lines(state))
            polished = strip_answer_label((await run_counted(polish_writer(), request)).output.strip())
        except CloudAccessError:
            raise
        except Exception as error:
            state.log("Polish", f"polish writer failed ({type(error).__name__}): unpolished answer kept")
            return Finish()
        state.show("polished answer", polished)
        problems = citation_problems(polished, state.evidence)
        if cited_ids(polished) != cited_ids(before):
            problems.append("the polish changed which ids are cited")
        lost_parts = [part for part in TEACHER_PARTS if part.lower() in before.lower() and part.lower() not in polished.lower()]
        if lost_parts:                       # seen in 3 of 9 runs: polish renamed the parts ('Tempo', 'Key points')
            problems.append(f"the polish dropped the part(s) {', '.join(lost_parts)}")
        new_claims = [claim for claim in split_claims(polished) if claim["ids"] and same_claim(claim["claim"]) not in state.verified]
        judged = []
        if new_claims and not problems:
            try:
                judged = judge_claims("\n".join(f"{claim['claim']} [{', '.join(f'{s}.{b}' for s, b in claim['ids'])}]"
                                                for claim in new_claims))
            except DecisionModelError as error:
                problems.append(f"could not judge the new sentences: {error}")
            state.show("sentences that are new after polishing, judged", judged_lines(judged), data=claims_data(judged))
        problems += [f"unsupported after polishing: '{claim['claim'][:60]}'" for claim in unsupported(judged)]
        if problems:
            state.show("polish rejected", problems)
            state.log("Polish", "worse than before -> unpolished answer kept")
            return Finish()
        state.answer = polished
        state.log("Polish", f"accepted ({len(new_claims)} new sentences, all supported)")
        return Finish()


class Finish:
    async def run(self, state):
        sources_text = ""
        if not state.final_answer and state.answer:
            warnings = []
            if state.citation_problems and state.passed_answer:     # a rewrite broke the citations: go back
                state.answer, state.judged_claims = state.passed_answer, state.passed_judged
                state.citation_problems = []
                warnings.append("a later rewrite was dropped because its citations failed the check")
                state.show("a rewrite broke the citations: the last answer that passed CheckCites is used instead")
            if state.judge_failed:
                warnings.append("claims were NOT checked (decision model failed)")
            if state.removed_claims:
                warnings.append("removed because the books do not support them: " +
                                "; ".join(claim[:80] for claim in state.removed_claims))
            if state.citation_problems:
                warnings.append("citation problems left: " + "; ".join(state.citation_problems))
            bad_claims = unsupported(state.judged_claims)
            if bad_claims:
                warnings.append("possibly unsupported: " + "; ".join(claim["claim"][:80] for claim in bad_claims))
            state.answer, repeated = drop_repeated_lines(state.answer)     # repairs can repeat a kept fact
            if repeated:
                state.show("lines that repeat an earlier fact, dropped by code", repeated)
            state.answer = drop_empty_headings(drop_debris_lines(state.answer))
            if state.route == "CHARACTER":                  # the honest frame above the books part
                state.answer = f"{in_language(CHARACTER_FRAME, state.language)}\n\n{state.answer}"
            state.warnings = warnings
            body, _, sources_text = render_with_sources(state.answer).partition("\n\nSOURCES\n")
            state.final_answer = body               # warnings stay in the trace and the viewer, not in the
                                                    # dancer's text (review: "removed because..." read as a leak)
            state.show("warnings", warnings or ["none"])
        # web body: the cited answer, or a fixed text (refusal, 'figure not in this dance') when no answer was written
        parts = answer_parts(state.answer) if state.answer else {"body": state.final_answer, "sources": []}
        if state.general_answer:                            # own knowledge: always under its own heading, last
            general_part = f"{state.general_heading}\n{state.general_answer}"
            state.final_answer = f"{state.final_answer}\n\n{general_part}" if state.final_answer else general_part
        if sources_text:                                    # the books' text last: tips must not hide below it
            state.final_answer += "\n\nSOURCES\n" + sources_text
        state.tokens = {f"{role} {model}": entry for (role, model), entry in METER.totals().items()}
        state.emit("final", route=state.route, route_reason=state.route_reason, **parts,
                   warnings=state.warnings, general=state.general_answer, general_heading=state.general_heading,
                   split=dict(question=state.original_question, books=state.question if state.split_done else "",
                              general=state.general_question) if state.split_done else None,
                   tokens=[{"role": role, "model": model, **entry} for (role, model), entry in METER.totals().items()])
        state.show("tokens used by this run (reported by the model servers)", METER.report_lines())
        state.log("Finish", f"route {state.route}, {state.claims_repaired_count} claims repaired, "
                            f"{len(state.removed_claims)} removed, {state.repair_rounds} whole-answer repairs, "
                            f"{len(state.evidence)} evidence sections")
        return None


# ------------------------------------------------------------------ run
async def answer_question(question, use_judge=True, live=False, log_path=None, event_sink=None):
    """live=True prints every node and every agent tool call as it happens; log_path writes the same lines
    to a file as they happen (and structured events to the .jsonl next to it); event_sink gets every event."""
    state = State(question=question, use_judge=use_judge, live=live, log_path=log_path, event_sink=event_sink)
    METER.reset()                                       # one question per run: count from zero
    take_thinking()                                     # ... and no reasoning left over from an earlier run
    state.emit("start", question=question, answer_model=MODELS.answer, decision_model=MODELS.decision,
               use_judge=use_judge)
    node = ExtractEntities()
    while node is not None:
        state.node_started = time.time()
        state.node_token_mark = METER.mark()
        state.header(type(node).__name__)
        try:
            node = await node.run(state)
        except CloudAccessError as error:                # no credits / not signed in: stop, say why
            state.write_thinking()
            message = (f"STOPPED: the model server refused this account. {error}\n"
                       "Check `ollama signin` and the credits at https://ollama.com/settings (cloud models), "
                       "or pick a local model.")
            state.say(f"└─ {message}")
            state.final_answer = message
            state.emit("error", message=message)
            return state
    return state


def ask(question, use_judge=True, live=False, log_path=None):
    """For scripts. In Jupyter use: state = await answer_question(question, live=True)."""
    return asyncio.run(answer_question(question, use_judge, live, log_path))
