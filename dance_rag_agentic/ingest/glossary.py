"""The books' own abbreviation tables ("Abbreviations commonly used for Alignment & Direction").

There is one table per chart column, so abbreviations are read per column: 'C' in the
Alignment column is Centre, 'B' in Foot Action is Ball. books.spell_out uses them.
"""
import re

ABBREVIATION_TABLE_TITLE = re.compile(r"^Abbreviations? commonly used for (.+)$", re.I)


def column_key(column_name):
    """Same key for the chart column and the glossary title:
    'Alignment & Direction' / 'Alignment Direction' -> 'alignmentdirection'."""
    return re.sub(r"[^a-z]", "", column_name.lower().replace(" and ", " "))


def read_abbreviation_tables(tables):
    """{column key: {abbreviation: meaning}}. Each table row holds (abbreviation, meaning) pairs side by side."""
    glossary = {}
    for rows in tables:
        title_match = ABBREVIATION_TABLE_TITLE.match(rows[0][0]) if rows and rows[0] else None
        if not title_match:
            continue
        column_glossary = glossary.setdefault(column_key(title_match.group(1)), {})
        for row in rows[1:]:
            for position in range(0, len(row) - 1, 2):
                abbreviation, meaning = row[position].strip(), row[position + 1].strip()
                if abbreviation and meaning:
                    column_glossary[abbreviation] = meaning
    return glossary


def fill_from_same_style(abbreviations, style_per_dance):
    """{dance: glossary} where every book also gets the columns its sister books (same style)
    define and it lacks, e.g. Slow Foxtrot has no 'Quantity of Turn' table but the other
    Standard books do. A book's own entries always win."""
    style_glossaries = {}
    for dance, glossary in abbreviations.items():
        style_glossary = style_glossaries.setdefault(style_per_dance.get(dance), {})
        for column, entries in glossary.items():
            for abbreviation, meaning in entries.items():
                style_glossary.setdefault(column, {}).setdefault(abbreviation, meaning)

    filled = {}
    for dance in style_per_dance:
        own_glossary = abbreviations.get(dance, {})
        combined = {column: dict(entries) for column, entries in
                    style_glossaries.get(style_per_dance[dance], {}).items()}
        for column, entries in own_glossary.items():
            combined.setdefault(column, {}).update(entries)
        filled[dance] = combined
    return filled

