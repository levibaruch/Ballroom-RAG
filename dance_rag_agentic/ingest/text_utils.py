"""Small text helpers shared by several stages."""
import re


def clean_bibr_text(text):
    """Remove LaTeX from bibr's OCR output and normalise spaces: '\\(^{(1)}\\)' -> '(1)'."""
    text = (text or "").replace("\\(", "").replace("\\)", "")
    text = re.sub(r"\^\{(.*?)\}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def readable_column_name(raw_name):
    """'HipDesign' or 'Hip Design' -> 'Hip Design' (OCR often drops spaces)."""
    without_spaces = raw_name.replace(" ", "")
    return re.sub(r"(?<=[a-z.])(?=[A-Z])", " ", without_spaces)


def tokenize(text):
    """Lowercase words, for BM25: 'OPENING OUT (Man)' -> ['opening', 'out', 'man']."""
    return re.findall(r"\w+", (text or "").lower())


def word_count(text):
    return len(str(text or "").split())
