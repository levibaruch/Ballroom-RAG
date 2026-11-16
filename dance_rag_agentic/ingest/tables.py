"""Chart tables: raw rows -> one entry per step -> readable text."""
import re
from itertools import zip_longest

from dance_rag_agentic.ingest.text_utils import readable_column_name

VALID_STEP_LABEL = re.compile(r"^(\d+(-\d+)?|Prec\.|Prec\. Step \d)$")
# valid: '3', '1-5' (several steps described together), 'Prec.' (preceding step),
#        'Prec. Step 2' (figures with several preceding steps)


def normalized_step_label(raw_label):
    """One spelling per meaning. 'Prec.', 'Prec Step', 'Step' (a split 'Prec. Step' cell) -> 'Prec.';
    'Prec. Step.1', 'Prec. Step 1', 'Prec. Step.l' (OCR l for 1) -> 'Prec. Step 1'."""
    label = " ".join(raw_label.split())
    squashed = re.sub(r"[^a-z0-9]", "", label.lower())
    if squashed in {"prec", "step", "precstep"}:
        return "Prec."
    numbered_preceding = re.fullmatch(r"precstep([0-9l])", squashed)
    if numbered_preceding:
        return f"Prec. Step {numbered_preceding.group(1).replace('l', '1')}"
    return label


def merge_two_row_header(table_rows):
    """Charts with a 2-row header: 'Timing | Timing | Timing' over 'A | B | C', the first cell repeated
    ('Step Action'). bibr reads them perfectly, but the repeated first cell looked like an invalid step label, so
    the table was rejected and the chart came from the PyMuPDF fallback (46-52% of the charts in Jive, Samba,
    Quickstep, Slow Foxtrot). The sub-columns under one name become ONE column 'Timing ABC' with the values
    joined ('S 1.2 1'), the form the rest of the code already reads."""
    if len(table_rows) < 3 or table_rows[1][0] != table_rows[0][0]:
        return table_rows
    header, sub_header = table_rows[0], table_rows[1]
    groups = []                                   # [(name, [column positions])]
    for position, name in enumerate(header):
        if groups and name == groups[-1][0] and sub_header[position] != name:
            groups[-1][1].append(position)
        else:
            groups.append((name, [position]))
    def merged_name(name, positions):
        if len(positions) == 1:
            return name
        return f"{name} {''.join(sub_header[position] for position in positions)}"
    new_header = [merged_name(name, positions) for name, positions in groups]
    new_rows = [new_header]
    for row in table_rows[2:]:
        new_rows.append([" ".join(row[position] for position in positions if position < len(row) and row[position]).strip()
                         for _, positions in groups])
    return new_rows


def table_to_steps(table_rows):
    """One entry per step.

    Rows without a label are sub-lines of the next labelled row; consecutive rows with the
    SAME label are one step (bibr repeats rowspan cells per sub-row, PyMuPDF splits
    'Prec.' / 'Step'). Identical values are kept once ('3' + '3' -> '3').
    Returns (column_names, steps); each step is a list of cell texts, [0] = step label.
    """
    table_rows = merge_two_row_header(table_rows)
    column_names = [readable_column_name(name) for name in table_rows[0]]
    steps, rows_waiting_for_label = [], []
    for row in table_rows[1:]:
        label = normalized_step_label(row[0])
        if not label:
            rows_waiting_for_label.append(row)
            continue
        rows_of_this_step = rows_waiting_for_label + [row]
        rows_waiting_for_label = []
        if steps and steps[-1][0] == label:
            steps[-1][1].extend(rows_of_this_step)
        else:
            steps.append((label, rows_of_this_step))
    if rows_waiting_for_label and steps:
        steps[-1][1].extend(rows_waiting_for_label)

    merged_steps = []
    for label, rows in steps:
        merged_cells = []
        for column_values in zip_longest(*rows, fillvalue=""):
            unique_values = []
            for value in column_values:
                value = (value or "").strip()
                if value and value not in unique_values:
                    unique_values.append(value)
            merged_cells.append(" / ".join(unique_values))
        merged_cells[0] = label
        merged_steps.append(merged_cells)
    return column_names, merged_steps


def step_to_text(dance, chart_title, column_names, step_cells):
    """One step as readable text: 'Dance | title | Step X' + 'Column: value' lines.
    The dance is part of the text because figure names repeat across books (Fan, Hockey Stick...)."""
    lines = [f"{dance} | {chart_title} | Step {step_cells[0]}"]
    for column_name, value in zip(column_names[1:], step_cells[1:]):
        if value:
            lines.append(f"{column_name}: {value}")
    return "\n".join(lines)


def table_is_usable(table_rows):
    """Checks before we trust a table: it has steps, and every step label looks valid."""
    if not table_rows or len(table_rows) < 2:
        return False
    _, steps = table_to_steps(table_rows)
    labels = [step_cells[0] for step_cells in steps]
    return bool(labels) and all(VALID_STEP_LABEL.match(label) for label in labels)
