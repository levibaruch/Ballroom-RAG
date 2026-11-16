"""Look sections and blocks up by their content instead of typing index ids: v4 ids are hashes of the content key."""
from dance_rag_agentic.retrieval import books


def section_of(dance, header=None, contains=None, figure=None, role=None, side=None, kind=None, notes=False):
    """The ONE v4 section id of a dance that matches every given field. header: exact (case-insensitive) or, when
    it ends the path, the last heading ('Music in DanceSport' also matches 'Timing › Music in DanceSport').
    contains: a text snippet of the section. side: 1, 2 ... for chart sides; notes=True picks the NOTES sections of a
    figure instead (kind prose). Raises when nothing or several match, so a test never silently reads another section."""
    pages = books.library().pages
    found = pages[pages.dance == dance]
    if figure is not None:
        found = found[found.figure == figure]
        found = found[found.kind == ("prose" if notes else "chart")] if figure and (notes or side is not None) else found
    if role is not None:
        found = found[found.role == role]
    if side is not None:
        found = found[found.side == side]
    if kind is not None:
        found = found[found.kind == kind]
    if header is not None:
        wanted = header.lower()
        heading = found.header.astype(str).str.lower()
        found = found[(heading == wanted) | found.path.astype(str).str.lower().str.endswith("› " + wanted)]
    if contains is not None:
        found = found[found.text.astype(str).str.contains(contains, regex=False)]
    if len(found) != 1:
        raise LookupError(f"section_of({dance!r}, header={header!r}, contains={contains!r}, figure={figure!r}, "
                          f"role={role!r}, side={side!r}) matched {len(found)} sections")
    return int(found.index[0])


def block_of(section_id, contains):
    """(section_id, block number, 1-based) of the first block of the section that holds the text."""
    for number, block in enumerate(books.section_blocks(section_id), 1):
        if contains in block:
            return section_id, number
    raise LookupError(f"no block of section {section_id} contains {contains!r}")
