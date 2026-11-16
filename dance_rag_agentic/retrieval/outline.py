"""The books' heading tree, from the index's heading units (no LLM).

Why: the search tools found and read single small sections. An agent could not see that 'Forward Walk' sits in
'Cha Cha specific principles', nor ask what a chapter contains, so it spent its calls on one-sentence intros.
With this tree:
  - outline(dance)       shows a book's table of contents with section ids (tools.outline),
  - heading_path(id)     tells a search hit where it lives: 'Cha Cha p.35 › Part 2 - Cha Cha specific principles › Forward Walk',
  - unit_of(id)          is the heading unit a section belongs to, so read_section reads one meaningful piece
                         instead of one fragment (parent-document reading).

The index (ingest/build_index.py) already cuts each book into parts (front matter, Part 1 general principles, Part 2
the dance's own principles) and heading units (a chapter with its subheadings); this module only arranges them.
"""
from dataclasses import dataclass, field
from functools import cache

from dance_rag_agentic.retrieval import books

PATH_SEPARATOR = " › "


@dataclass
class Node:
    node_id: int                    # first section of the node
    title: str
    level: int                      # 0 part, 1 chapter/topic, 2 subheading
    page: int
    section_ids: list = field(default_factory=list)
    parent: "Node | None" = None
    children: list = field(default_factory=list)

    def characters(self):
        return sum(len(block) for section_id in self.section_ids for block in books.section_blocks(section_id))

    def unit_sections(self):
        """This node and every node under it, in book order."""
        return self.section_ids + [section_id for child in self.children for section_id in child.unit_sections()]


@cache
def book_tree(dance):
    """[part nodes] of one book; every prose unit that is not a figure NOTE belongs to exactly one node."""
    pages = books.library().pages
    units = pages[(pages.dance == dance) & (pages.kind == "prose") & pages.part.notna()
                  & (pages.part != "figure charts")].sort_values("order")
    parts = {}
    for section_id, record in units.iterrows():
        part = parts.get(record.part)
        if part is None:
            part = parts[record.part] = Node(int(section_id), record.part, 0, int(record.page))
        if record.unit_role == "part_intro":
            part.section_ids.append(int(section_id))
        else:
            part.children.append(Node(int(section_id), record.header, 1, int(record.page), [int(section_id)], part))
    return list(parts.values())


@cache
def _node_of_section():
    """section_id -> the node that holds it (all books)."""
    found = {}

    def visit(node):
        for section_id in node.section_ids:
            found[section_id] = node
        for child in node.children:
            visit(child)
    for dance in books.library().dances:
        for part in book_tree(dance):
            visit(part)
    return found


def node_of(section_id):
    """The outline node of a prose section, or None (charts, figure NOTES)."""
    return _node_of_section().get(int(section_id))


def heading_path(section_id):
    """'Cha Cha p.35 › Part 2 - Cha Cha specific principles › Forward Walk' (charts: '... › figure charts › ...')."""
    record = books.library().pages.loc[section_id]
    return f"{record.dance} p.{int(record.page)}{PATH_SEPARATOR}{record.path}"


def unit_of(section_id):
    """The heading unit to read for a section: a chapter/topic with its subheadings. A subheading alone is too
    small (one 'Timing' line), so it is read with its chapter; a part is too big, so a part gives only its own
    intro (the outline lists its chapters)."""
    node = node_of(section_id)
    if node is None:
        return None
    while node.level == 2 and node.parent is not None:
        node = node.parent
    return node


def outline_lines(dance, section_id=None):
    """Lines of the table of contents. No section_id: every part and its chapters. With a section_id: the
    subheadings of that chapter."""
    if section_id is not None:
        node = unit_of(section_id)
        if node is None:
            return [f"section {section_id} is not in the outline (a chart or figure NOTES): use select_charts"]
        lines = [f"{heading_path(node.node_id)}  ({node.characters()} chars in all)"]
        lines += [f"  section_id={child.node_id} | p.{child.page} | {child.title} | {child.characters()} chars"
                  for child in node.children] or ["  (no subheadings)"]
        return lines
    lines = []
    for part in book_tree(dance):
        shared = "  [same text in every book of this style]" if part.title.startswith("Part 1") else ""
        lines.append(f"{part.title}  (section_id={part.node_id}, p.{part.page}){shared}")
        for chapter in part.children:
            extra = f", {len(chapter.children)} subheadings" if chapter.children else ""
            lines.append(f"  section_id={chapter.node_id} | p.{chapter.page} | {chapter.title} | "
                         f"{len(chapter.unit_sections())} sections, {sum(len(b) for s in chapter.unit_sections() for b in books.section_blocks(s))} chars{extra}")
    return lines
