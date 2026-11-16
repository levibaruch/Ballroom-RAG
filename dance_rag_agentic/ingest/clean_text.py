"""Text cleanup for the index: undo what PDF extraction did to the books' words.

    'th a t', 'w ith o u t', 'm ethod o f com m unication'  ->  'that', 'without', 'method of communication'
    'Quantityof Turn', 'Typeof Sway'                          ->  'Quantity of Turn', 'Type of Sway'
    a literal backslash-n inside a cell                       ->  a space

Both fixes use a vocabulary taken from the books themselves (words that also occur unbroken elsewhere), so no
dictionary of dance terms has to be typed in. A merge only happens when the joined word is common in the books
and at least one piece is NOT a word on its own ('in a hold' stays; 'th a t' becomes 'that').
"""
import re
from collections import Counter

WORD = re.compile(r"[A-Za-z]+")
MIN_JOINED_COUNT = 3        # a piece of 5+ letters counts as a word from this many occurrences
MIN_JOINED_WORD_COUNT = 10  # a joined word must occur unbroken this often: 'Land' (from 'L and R'), 'Rbetween' and
                            # 'TimingS' exist only as extraction glue a few times
MIN_SPLIT_PART_COUNT = 20   # 'Quantityof' -> 'Quantity' + 'of': both parts must be this common
MAX_PIECES = 6              # 'w ith o u t' = 5 pieces
MIN_SHORT_WORD_COUNT = 40   # a piece of 1-4 letters only counts as a word when this common: the broken text itself
                            # makes 'tim' + 'ing' and 'norm' + 'ally' occur a few times


def build_vocabulary(texts):
    """Lower-case word counts over all book texts."""
    counts = Counter()
    for text in texts:
        counts.update(word.lower() for word in WORD.findall(text))
    return counts


def _is_word(piece, vocabulary):
    if len(piece) == 1:
        return piece.lower() in ("a", "i")
    needed = MIN_SHORT_WORD_COUNT if len(piece) <= 3 else MIN_JOINED_WORD_COUNT if len(piece) == 4 else MIN_JOINED_COUNT
    return vocabulary.get(piece.lower(), 0) >= needed


PIECE = re.compile(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$")   # '"W' -> ('"', 'W', ''), 'eight"' -> ('', 'eight', '"')


def join_broken_words(text, vocabulary):
    """Join runs of word pieces separated by single spaces when together they form a common word."""
    tokens = re.split(r"( )", text)                 # keep the spaces so everything else stays as it was
    words = tokens[0::2]
    output, index = [], 0
    while index < len(words):
        best = None
        for end in range(min(len(words), index + MAX_PIECES), index + 1, -1):     # longest join first
            parts = [PIECE.match(piece) for piece in words[index:end]]
            if not all(parts):
                continue
            # punctuation only before the first piece and after the last ('"W eight"' -> '"Weight"')
            if any(part.group(1) for part in parts[1:]) or any(part.group(3) for part in parts[:-1]):
                continue
            cores = [part.group(2) for part in parts]
            joined = "".join(cores)
            if any(core[0].isupper() for core in cores[1:]) and not joined.isupper():
                continue                            # 'of F' (Ball of Foot) is not 'ofF'; 'W eight' -> 'Weight' is fine
            if all(core.isupper() and len(core) <= 2 for core in cores):
                continue                            # notation and abbreviations: 'Q Q S', 'T B', 'L R'
            if vocabulary.get(joined.lower(), 0) < MIN_JOINED_WORD_COUNT:
                continue
            if all(_is_word(core, vocabulary) for core in cores):
                continue                            # 'in a hold' stays: every piece is a word
            best = (end, parts[0].group(1) + joined + parts[-1].group(3))
            break
        if best:
            output.append(best[1])
            index = best[0]
        else:
            output.append(words[index])
            index += 1
    return " ".join(output)


GLUED_FUNCTION_WORDS = ("of", "and", "to", "the", "in")
ALWAYS_GLUED = ("of", "and")    # 'Quantityof Turn', 'Typeof Sway', 'Riseand Fall' are glued in every chart


def split_glued_words(text, vocabulary):
    """Two words glued by extraction: 'Quantityof' -> 'Quantity of', 'PositionUsed' -> 'Position Used',
    'LEFTTO' -> 'LEFT TO'. Only at an inner capital or before a short function word, and only when the glued word
    never occurs as such: 'projected', 'checking' and 'Upon' are rare words, not glue."""
    def common(word):
        return vocabulary.get(word.lower(), 0) >= MIN_SPLIT_PART_COUNT

    def split(match):
        word = match.group(0)
        for function_word in ALWAYS_GLUED:                                   # chart headers: glued in every
            left = word[:-len(function_word)]                                # chart, so frequent, still glue
            if word.lower().endswith(function_word) and len(left) >= 4 and common(left) and left[0].isupper():
                return f"{left} {word[-len(function_word):]}"
        if vocabulary.get(word.lower(), 0) >= MIN_JOINED_COUNT:
            return word
        inner_capital = re.match(r"^([A-Z]?[a-z]+)([A-Z][a-z]+)$", word)       # PositionUsed
        if inner_capital and common(inner_capital.group(1)) and common(inner_capital.group(2)):
            return f"{inner_capital.group(1)} {inner_capital.group(2)}"
        for function_word in GLUED_FUNCTION_WORDS:                          # Quantityof, LEFTTO
            left = word[:-len(function_word)]
            if word.lower().endswith(function_word) and len(left) >= 4 and common(left):
                return f"{left} {word[-len(function_word):]}"
        return word
    return WORD.sub(split, text)


def clean(text, vocabulary):
    if not isinstance(text, str) or not text:
        return text
    text = text.replace("\\n", " ")
    return split_glued_words(join_broken_words(text, vocabulary), vocabulary)
