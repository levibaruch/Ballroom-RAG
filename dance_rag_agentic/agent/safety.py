"""Questions about self-harm or suicide get a fixed, caring reply with crisis contacts, before any model runs.

Code, not a model: the public demo must never answer such a question with "outside what I can answer" (seen in
the VPS traces, 2026-11-14). The patterns are deliberately broad; a false alarm costs one kind message, a miss
costs far more. Dance phrases that look alike ('kill the turn', 'dying swan') do not match: every pattern needs
a first-person or explicit self-harm wording.

'Hurt / harm / cut myself' is also how dancers ask about injuries ("How do I make sure not to hurt myself in the
rumba?", reported 2026-11-16). Those words count only when they are NOT framed as avoiding an injury (not to,
without, avoid, prevent ...) and NOT followed by a dance or body context (in the rumba, while dancing, my knee ...).
The explicit patterns (suicide, kill myself, want to die ...) always count.
"""
import re

SELF_HARM_PATTERNS = [
    # English
    r"\bkill(ing)?\s+(my|our)sel(f|ves)\b", r"\bsuicid", r"\bend(ing)?\s+(my\s+(own\s+)?life|it\s+all)\b",
    r"\btake\s+my\s+(own\s+)?life\b",
    r"\bself[\s-]?harm", r"\bwant\s+to\s+die\b", r"\bdon'?t\s+want\s+to\s+(live|be alive)\b",
    r"\bbetter\s+off\s+dead\b",
    # Dutch
    r"\bzelfmoord", r"\bmezelf\s+(van\s+het\s+leven\s+beroven|doden|iets\s+aandoen|pijn\s+doen)\b",
    r"\bmijn\s+leven\s+(be[eë]indigen|beeindigen)\b", r"\bwil\s+(niet\s+meer\s+leven|dood)\b", r"\bdood\s+willen\b",
    r"\bzelfbeschadiging",
    # German, French, Spanish (the demo gets questions in other languages too)
    r"\bselbstmord", r"\bmich\s+umbringen\b", r"\bme\s+tuer\b", r"\bmatarme\b",     # 'suicid' covers fr/es/it
]
SELF_HARM = re.compile("|".join(SELF_HARM_PATTERNS), re.I)

# Wording that is self-harm OR a question about injuries: decided by its context (see the module docstring)
HURT_MYSELF = re.compile(r"\b(hurt|harm|cut|injure)(s|ing)?\s+myself\b|\bmezelf\s+(pijn\s+doen|iets\s+aandoen)\b", re.I)
AVOIDING = re.compile(r"\b(not|n't|never|without|avoid\w*|prevent\w*|niet|zonder|voorkom\w*|"
                      r"vermijd\w*)\b(\W+\w+){0,3}\W*$", re.I)          # up to 3 words before the phrase
DANCE_OR_BODY = re.compile(
    r"^\W*(\w+\W+){0,4}?(danc\w*|rumba|cha\s*cha|jive|samba|paso|waltz|wals|tango|foxtrot|quickstep|practi[cs]\w*|"
    r"training|class|lesson|competition|step|spin|turn|figure|lift|warm\w*|stretch\w*|knee|ankle|foot|feet|hip|"
    r"back|shoulder|neck|muscle|toe|heel|wrist|dansen|les|knie|enkel|rug)\b", re.I)   # up to 4 words after it

SAFETY_REPLY = (
    "I'm really sorry you're feeling this way. I'm only a helper for dance technique, but you deserve support "
    "from a real person right now, and you don't have to go through this alone.\n\n"
    "- Netherlands: 113 Zelfmoordpreventie, call 0800-0113 (free, 24/7) or chat at 113.nl\n"
    "- Belgium: Zelfmoordlijn 1813 (call 1813 or chat at zelfmoord1813.be)\n"
    "- Anywhere else: findahelpline.com lists free helplines in your country\n"
    "- In immediate danger: call 112 (or your local emergency number)\n\n"
    "Talking to someone you trust, a friend, family member or your doctor, can help too.\n\n"
    "Het spijt me echt dat je je zo voelt. Je hoeft dit niet alleen te dragen: bel gratis 0800-0113 of chat via "
    "113.nl (dag en nacht). Bij direct gevaar: bel 112.")


def is_self_harm(question):
    if SELF_HARM.search(question):
        return True
    for match in HURT_MYSELF.finditer(question):
        before, after = question[:match.start()], question[match.end():]
        if not (AVOIDING.search(before) or DANCE_OR_BODY.search(after)):
            return True
    return False
