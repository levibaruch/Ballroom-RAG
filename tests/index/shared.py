"""Section ids and helpers shared by the offline tests. The ids are looked up by content (helpers.section_of),
never typed in: they are hashes of the content key."""
from dance_rag_agentic.retrieval import books
from dance_rag_agentic.agent.evidence import EvidenceLog

from ..stubs import KINDS, classify_stub, probabilities  # noqa: F401  (index tests import them from here)
from .helpers import block_of, section_of

# v4 ids are hashes: look the sections up by content. MUSIC = Rumba p.21 'Music in DanceSport'; its time-signature
# table is blocks HEADER (the 'Time Signature | Latin' row), 24 (Samba / Paso Doble row), 44 (Rumba / Cha Cha row).
MUSIC = section_of("Rumba", header="Music in DanceSport")
REF_HEADER = block_of(MUSIC, "Time Signature | Latin")
REF_24 = block_of(MUSIC, "SAMBA (2nd beat accented)")
REF_44 = block_of(MUSIC, "RUMBA (4th beat accented)")
REF_TEMPO = block_of(MUSIC, "Rumba | 25-27")
SIG_HEADER, SIG_24, SIG_44, SIG_TEMPO = (f"{section}.{block}" for section, block in (REF_HEADER, REF_24, REF_44, REF_TEMPO))

FAN_LADY_SIDE_1 = section_of("Rumba", figure="Fan", role="Lady", side=1)
FAN_LADY_SIDE_2 = section_of("Rumba", figure="Fan", role="Lady", side=2)
_pages = books.library().pages
# Rumba Fan (Lady) = Side 1, Side 2 and four NOTES sections: the whole figure chart
FAN_LADY_PARTS = {int(section_id) for section_id in _pages[(_pages.dance == "Rumba") & (_pages.figure == "Fan")
                                                           & (_pages.role == "Lady")].index}

CHA_FAN_MAN_SIDE_2 = section_of("Cha Cha", figure="Fan", role="Man", side=2)
CHA_FAN_LADY_SIDE_1 = section_of("Cha Cha", figure="Fan", role="Lady", side=1)
CHA_FAN_LADY_SIDE_2 = section_of("Cha Cha", figure="Fan", role="Lady", side=2)
HOCKEY_STICK_MAN_SIDE_2 = section_of("Rumba", figure="Hockey Stick", role="Man", side=2)
SPOT_TURN_MAN_SIDE_1 = section_of("Rumba", figure="Spot Turn To Left", role="Man", side=1)
SPOT_TURN_MAN_SIDE_2 = section_of("Rumba", figure="Spot Turn To Left", role="Man", side=2)
LEAD_HOLD_SHAPING = section_of("Rumba", header="10) Lead Hold Shaping")       # ends with '...two different types of lead:'
CHA_PART_2 = section_of("Cha Cha", header="Part 2 - Cha Cha specific principles")
CHA_FORWARD_WALK = section_of("Cha Cha", header="Forward Walk")
RUMBA_FORWARD_WALK = section_of("Rumba", header="Forward Walk")
SLOW_WALTZ_ROTATION = section_of("Slow Waltz", header="8) Rotation")


def evidence_of(*section_ids):
    log = EvidenceLog()
    for section_id in section_ids:
        log.add(section_id, "test")
    return log
