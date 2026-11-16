"""ExtractEntities: dances, figures, role and step numbers in a question. Code, not an LLM.

Figure names are matched against the books' own figure lists, so 'Fan' is only a figure because
a book has a chart called FAN. The decision model proved bad at this (it missed 7 of 22 figure names).
"""
import difflib
import re
from functools import cache
from typing import TypedDict

from dance_rag_agentic.ingest.text_utils import tokenize

from dance_rag_agentic.retrieval.books import library

MAN_WORDS = {"man", "men", "gentleman", "leader", "his"}
LADY_WORDS = {"lady", "ladies", "woman", "follower", "her"}
# 'lead' and 'follow' as ROLE NOUNS ('the follow', 'footwork for the lead', "follow's"), not as verbs ('which figures
# can follow the Fan', 'how do I lead the Rumba'). Seen on the demo: "Follow footwork for top spin" answered for the
# man as "the follow (man)".
ROLE_NOUN = re.compile(r"(?:\b(?:the|a|as|for|of|to)\s+|^\s*)(lead|follow)(?:'s)?\b"
                       r"(?!\s+(?:the|a|an|this|that|it|him|her|me|you|my|your|through|with|up)\b)", re.I)
# Words that mark a question's language; the reply must be written in it ('rumba' made one English question come back
# in Spanish). Only clear function words, so a dance or figure name never decides it.
LANGUAGE_WORDS = {
    "English": {"the", "how", "what", "is", "do", "does", "i", "my", "in", "a", "of", "to", "and", "which", "why",
                "when", "where", "can", "should", "with", "between", "difference", "best", "way", "for", "on", "at",
                "from", "you", "your", "are", "it", "this", "that", "after", "before", "during"},
    "Dutch": {"hoe", "wat", "welke", "waarom", "de", "het", "een", "ik", "mijn", "is", "en", "van", "bij", "doe",
              "kan", "moet", "tussen", "verschil", "wanneer", "waar", "wie", "heeft", "deze", "goed", "dansen"},
    "German": {"wie", "was", "welche", "warum", "der", "die", "das", "ein", "eine", "ich", "und", "ist", "mit", "beim",
               "tanze", "tanzen", "unterschied", "zwischen"},
    "French": {"comment", "quel", "quelle", "pourquoi", "le", "la", "les", "un", "une", "je", "et", "est", "danser",
               "entre", "différence", "du", "des"},
    "Spanish": {"cómo", "como", "qué", "que", "cuál", "por", "el", "los", "las", "un", "una", "yo", "y", "es",
                "bailar", "entre", "diferencia", "del"},
}
# How people write dance names ('foxtrot'); only aliases of dances that are in the knowledge base are used.
DANCE_ALIASES = {"cha cha": "Cha Cha", "chacha": "Cha Cha", "foxtrot": "Slow Foxtrot", "english waltz": "Slow Waltz",
                 "paso": "Paso Doble", "quick step": "Quickstep", "viennese": "Viennese Waltz"}
# 'Fan Position' is a position, not the figure Fan. A prefix, so typos and other languages count too:
# postion, positie, Position.
POSITION_PREFIXES = ("pos",)
MINIMUM_SIMILARITY = 0.85
SHORT_NAME_LETTERS = 5                          # 'fan', 'whip', 'wing': stricter matching, see similarity()
MINIMUM_WORD_SIMILARITY = 0.8                   # every word of a multi-word name must match this well ('the' ~ 'three' = 0.75)


def clean_name(raw_name):
    """'3. Syncopated Chasse' -> 'syncopated chasse'; OCR junk with single letters ('Back W Hisk') -> None."""
    name_words = tokenize(re.sub(r"^\d+\.\w?\s+", "", raw_name))
    if not name_words or any(len(word) == 1 and word.isalpha() for word in name_words):
        return None
    return " ".join(name_words)


@cache
def figure_names():
    """{cleaned name: name as in the books}"""
    names = {}
    for raw_name in library().knowledge_base.known_figures:
        cleaned = clean_name(raw_name)
        if cleaned:
            names.setdefault(cleaned, raw_name)
    return names


@cache
def dance_names():
    dances = library().dances
    return ({dance.lower(): dance for dance in dances}
            | {alias: dance for alias, dance in DANCE_ALIASES.items() if dance in dances})


def similarity(window, name):
    """How well a window of question words matches a cleaned name (0-1).
    Short names (<= 5 letters) only match exactly or as a typo that keeps the first letter
    and is not longer than the name: 'wisk' -> whisk, but not 'hip' -> whip, 'swing' -> wing, 'links' -> link.
    Multi-word names need every word to match reasonably: 'three types' is not 'three threes'."""
    if len(name) <= SHORT_NAME_LETTERS:          # no plural rule: Dutch 'links' (= left) is not the figure Link
        if window == name:
            return 1.0
        is_short_typo = window[:1] == name[:1] and len(window) <= len(name)
        return difflib.SequenceMatcher(None, window, name).ratio() if is_short_typo else 0.0
    window_words, name_words = window.split(), name.split()
    if len(name_words) > 1 and any(difflib.SequenceMatcher(None, window_word, name_word).ratio() < MINIMUM_WORD_SIMILARITY
                                   for window_word, name_word in zip(window_words, name_words)):
        return 0.0
    return difflib.SequenceMatcher(None, window, name).ratio()


def find_names(question_words, names, minimum_similarity=MINIMUM_SIMILARITY, skip_before=()):
    """All names that fuzzy-match a window of words in the question.
    Best match first (a longer name wins a tie); windows may not overlap, so 'Throwaway Whip' and
    'Whip' can both be found, but 'Close Basic Movement' hides 'Basic Movement'.
    names: {cleaned name: name to return}. skip_before: a match followed by a word starting with one of these
    prefixes is ignored."""
    candidates = []                     # (similarity, number of words, returned name, start position)
    for cleaned_name, returned_name in names.items():
        name_length = len(cleaned_name.split())
        for start in range(len(question_words) - name_length + 1):
            window = " ".join(question_words[start:start + name_length])
            match_quality = similarity(window, cleaned_name)
            end = start + name_length
            next_word = question_words[end] if end < len(question_words) else ""
            followed_by_skip_word = bool(skip_before) and next_word.startswith(tuple(skip_before))
            if match_quality >= minimum_similarity and not followed_by_skip_word:
                candidates.append((round(match_quality, 2), name_length, returned_name, start))
    found_names, used_positions = [], set()
    for _, name_length, returned_name, start in sorted(candidates, key=lambda candidate: (-candidate[0], -candidate[1])):
        window_positions = set(range(start, start + name_length))
        if window_positions & used_positions or returned_name in found_names:
            continue
        used_positions |= window_positions
        found_names.append(returned_name)
    return found_names


class Entities(TypedDict):
    dances: list[str]            # book names, e.g. ['Cha Cha', 'Rumba']
    figures: list[str]           # figure names as in the books, e.g. ['Hockey Stick']
    role: str | None             # 'man' / 'lady' / 'both' / None
    steps: list[int]             # step numbers asked, e.g. [3]


def extract_entities(question) -> Entities:
    """Dances, figures, role and step numbers in the question."""
    question_words = tokenize(question)
    word_set = set(question_words)
    mentions_man, mentions_lady = bool(word_set & MAN_WORDS), bool(word_set & LADY_WORDS)
    for role_noun in ROLE_NOUN.findall(question):
        mentions_man |= role_noun.lower() == "lead"
        mentions_lady |= role_noun.lower() == "follow"
    if mentions_man and mentions_lady:
        role = "both"
    elif mentions_man:
        role = "man"
    elif mentions_lady:
        role = "lady"
    else:
        role = None
    return Entities(dances=list(dict.fromkeys(find_names(question_words, dance_names()))),
                figures=find_names(question_words, figure_names(), skip_before=POSITION_PREFIXES),
                role=role,
                steps=[int(number) for number in re.findall(r"\bsteps?\s+(\d+)", question, re.I)])


def figures_ending_in_question_word(question, dances, limit=6):
    """Figures of the named dances whose LAST word is a word of the question ('samba runs' -> Carioca Runs,
    Promenade To Counter Promenade Runs). Only a hint for the searching agent: a partial name may fit several
    figures, so it never becomes an entity (two figures would trigger the compare route)."""
    from dance_rag_agentic.retrieval import books
    question_words = set(tokenize(question))
    found = []
    for dance in dances:
        for figure in books.figures_of(dance):
            last_word = tokenize(figure)[-1:] if tokenize(figure) else []
            if last_word and len(last_word[0]) >= 4 and last_word[0] in question_words:
                found.append(f"{figure} ({dance})")
    return found[:limit]


def question_language(question):
    """'English', 'Dutch', ... by counting function words; None when no language clearly leads."""
    words = re.findall(r"[a-zà-ÿ']+", question.lower())
    counts = {language: sum(word in marker_words for word in words) for language, marker_words in LANGUAGE_WORDS.items()}
    ranked = sorted(counts.items(), key=lambda item: -item[1])
    (best, best_count), (_, second_count) = ranked[0], ranked[1]
    return best if best_count >= 1 and best_count > second_count else None


def mentions_role_words(question):
    """lead/follow/leader/follower in any form: the writer is told these mean man/lady in the books."""
    return bool(re.search(r"\b(lead|leads|leader|leaders|leading|follow|follows|follower|followers|following)\b",
                          question, re.I))
