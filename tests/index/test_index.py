"""Index checks (ingest/build_index.py). They run the package in a separate process and are skipped when the index has
not been built."""
import subprocess
import sys
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[2]


V4_DIR = ROOT / "cache" / "agentic" / "index_v4"


v4_built = pytest.mark.skipif(not (V4_DIR / "build_info.json").exists(), reason="index v4 not built")


def run_package_code(code):
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


@v4_built
def test_v4_has_part_2_in_every_latin_book_and_units_with_their_subheadings():
    output = run_package_code(
        "from dance_rag_agentic.retrieval import books\n"
        "p = books.library().pages\n"
        "print(sorted(p[p.part.astype(str).str.startswith('Part 2')].dance.unique()))\n"
        "unit = p[(p.dance == 'Cha Cha') & (p.header == 'Forward Walk')].iloc[0]\n"
        "print('Foot Placement' in unit.text, unit.key)")
    assert all(dance in output for dance in ("Cha Cha", "Rumba", "Samba", "Jive", "Paso Doble"))
    assert "True Cha Cha|Part 2 - Cha Cha specific principles › Forward Walk" in output


@v4_built
def test_v4_reads_the_rumba_precedes_follows_matrix():
    output = run_package_code(
        "from dance_rag_agentic.retrieval import books\n"
        "p = books.library().pages\n"
        "m = p[(p.dance == 'Rumba') & (p.unit_role == 'matrix')]\n"
        "print(len(m), 'Close Basic is followed by' in ' '.join(m.text))")
    assert output.split()[1] == "True"


@v4_built
def test_v4_quickstep_chart_comes_from_bibr_with_the_two_row_header():
    output = run_package_code(
        "from dance_rag_agentic.retrieval import books\n"
        "p = books.library().pages\n"
        "c = p[(p.dance == 'Quickstep') & (p.figure == 'Natural Turn') & (p.role == 'Man') & (p.side == 1)].iloc[0]\n"
        "print(c.source, '|', books.section_blocks(int(c.name))[0].split('Step 3')[1][:200].replace(chr(10), ' '))")
    assert output.startswith("bibr") and "1/8 to R" in output
