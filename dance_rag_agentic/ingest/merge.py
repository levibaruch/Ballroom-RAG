"""Chart pages: bibr tables first, PyMuPDF as fallback. Every record gets a 'source'.

The n-th PyMuPDF chart title on a page gets the n-th bibr table on that page, if the counts match and the
bibr table passes the checks. NOTEs below the tables get their figure as header.
"""
from dance_rag_agentic.ingest.tables import table_is_usable


def merge_chart_page(page_number, chart_records, bibr_book):
    """Chart records of one page (dicts), with the bibr table as 'rows' where it is usable."""
    records = []
    tables_from_bibr = bibr_book.tables.get(page_number, [])
    counts_match = len(tables_from_bibr) == len(chart_records)
    for position, (_, chart_record) in enumerate(chart_records.iterrows()):
        record = chart_record.to_dict()
        if not counts_match:
            record["source"] = (f"pymupdf (bibr found {len(tables_from_bibr)} tables, "
                                f"expected {len(chart_records)})")
        elif not table_is_usable(tables_from_bibr[position]):
            record["source"] = "pymupdf (bibr table failed checks)"
        else:
            record["rows"] = tables_from_bibr[position]
            record["source"] = "bibr"
        records.append(record)

    first_chart = chart_records.iloc[0]                       # NOTEs belong to this figure
    for note_text in bibr_book.notes.get(page_number, []):
        records.append(dict(dance=first_chart.dance, style=first_chart["style"], page=page_number,
                            kind="prose", figure=first_chart.figure, role=first_chart.role, side=None,
                            header=f"{first_chart.figure.upper()} ({first_chart.role}) - NOTES",
                            rows=None, text=note_text, source="bibr (note)"))
    return records

