"""Step 1 of the pipeline: extract every book once (cached), without building an index.

    python -m dance_rag_agentic.ingest.extract [--books Rumba "Cha Cha"]

  - bibr JSON per book in extractions/bibr/ (text layer + table OCR; ~10-15 min per book, skipped when present)
  - PyMuPDF records per book in cache/agentic/pymupdf/ (chart titles, fallback tables)
  - the books' own index pages in cache/agentic/book_index.pkl
Then: Docling on the prose pages (ingest/docling_pages.py) and `python -m dance_rag_agentic.ingest.build_index`.
"""
import argparse

from dance_rag_agentic.ingest.bibr import run_bibr
from dance_rag_agentic.config import select_book_pdfs
from dance_rag_agentic.ingest.extract_pymupdf import load_pymupdf_pages


def main():
    parser = argparse.ArgumentParser(description="Extract the books (bibr + PyMuPDF + book index), cached.")
    parser.add_argument("--books", nargs="*", help="dance names (PDF file names); default: all books")
    arguments = parser.parse_args()
    pdf_paths = select_book_pdfs(arguments.books)
    if not pdf_paths:
        raise SystemExit("no PDFs found: put the books in theory/<latin|standard>/<Dance>.pdf")
    for pdf_path in pdf_paths:
        print(f"{pdf_path.stem}:", flush=True)
        run_bibr(pdf_path)
        load_pymupdf_pages(pdf_path)
    from dance_rag_agentic.retrieval.books import book_index
    print(f"book index: {len(book_index())} entries")


if __name__ == "__main__":
    main()
