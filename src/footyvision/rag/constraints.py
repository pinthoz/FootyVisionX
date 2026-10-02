"""Read the hard constraints out of a scouting question.

Embedding search ranks by overall resemblance, which is the wrong tool for an absolute
requirement: "left-footed" is one token in a profile of dozens and never dominates the
similarity, so a semantic search for "a young left-footed winger" happily returns
right-footed ones. These constraints are pulled out first and used to narrow the pool
*before* ranking it.

Deliberately deterministic rather than a second LLM call: it costs nothing, it cannot
hallucinate a filter the user did not ask for, and it is testable. The cost is that only
the phrasings listed here are understood — anything else yields no constraint and
retrieval behaves as it did before, which is the safe direction to fail in.

Portuguese counts as a listed phrasing. The assistant answers questions asked in any
language and the retriever was fine-tuned bilingually, but for a while these patterns were
English-only — so "a left-footed winger" was filtered and "um extremo canhoto" was not,
and the Portuguese question fell back to pure similarity ranking, which is precisely what
this module exists to stop. Accents are stripped before matching so "médio" and "medio"
are the same word to it.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

# "young" has no agreed definition; this is the usual scouting shorthand, and the answer
# states which threshold was applied so the reader is not left guessing.
YOUNG_MAX_AGE = 23.0
VETERAN_MIN_AGE = 32.0


@dataclass(frozen=True)
class Constraints:
    foot: str | None = None
    position_group: str | None = None
    max_age: float | None = None
    min_age: float | None = None
    # Stored as the country name exactly as the index spells it, so it can be compared
    # without normalising on every query.
    nationality: str | None = None
    # The side-agnostic role (`ml.features.position_role`), when the question names one
    # precisely: "wing-backs", not "defenders". The four groups are too coarse for a
    # leaders question — the full-backs who make the most carries are not the centre-backs
    # who do, and both are DEF.
    position_role: str | None = None

    def __bool__(self) -> bool:
        return any(
            v is not None
            for v in (
                self.foot,
                self.position_group,
                self.max_age,
                self.min_age,
                self.nationality,
                self.position_role,
            )
        )

    def describe(self) -> str:
        """Human-readable summary, so the answer can state what it filtered on."""
        parts = []
        if self.foot:
            parts.append("two-footed" if self.foot == "both" else f"{self.foot}-footed")
        if self.position_role:
            parts.append(self.position_role.lower() + "s")
        elif self.position_group:
            parts.append(self.position_group)
        if self.nationality:
            parts.append(f"from {self.nationality}")
        if self.max_age is not None:
            parts.append(f"aged {self.max_age:.0f} or under")
        if self.min_age is not None:
            parts.append(f"aged {self.min_age:.0f} or over")
        return ", ".join(parts)


# Two details worth keeping: "ambidestro" ends in "destro", so the word boundary is what
# stops every two-footed player being read as right-footed; and the trailing `s?` is there
# because Portuguese inflects these adjectives ("canhotos", "ambidestros") where the
# English ones this started with never did.
_FOOT_PATTERNS = (
    (
        re.compile(r"\b(left[\s-]?footed|left foot|canhot[oa]|esquerdin[oa]|pe esquerdo)s?\b"),
        "left",
    ),
    (re.compile(r"\b(right[\s-]?footed|right foot|destr[oa]|pe direito)s?\b"), "right"),
    (
        re.compile(r"\b(two[\s-]?footed|both feet|ambidextrous|ambidestr[oa]|ambidextr[oa])s?\b"),
        "both",
    ),
)

# Matched by earliest mention, not by list order. A question names its subject before its
# object — "a winger who takes defenders on" is about the winger — so the first position
# word in the sentence is the one being asked for. Within each pattern the compound forms
# come first so "centre-back" is not read as a forward on the word "centre".
_POSITION_PATTERNS = (
    (re.compile(r"\b(goalkeeper|keeper|goalie|guarda[\s-]?redes|guardiao|goleiro)s?\b"), "GK"),
    (
        re.compile(
            # "lateral" and "central" pluralise to "laterais" and "centrais", which the
            # trailing `s?` cannot build, so both forms are spelled out.
            r"\b(wing[\s-]?back|full[\s-]?back|centre[\s-]?back|center[\s-]?back|"
            r"defender|defence|defense|defesa|defensor(?:es)?|"
            r"centra(?:l|is)|latera(?:l|is)|ala)s?\b"
        ),
        "DEF",
    ),
    # "medio ala" is a midfielder, not a wing-back: the earliest-mention rule settles it,
    # because "medio" is read before "ala" and the first position word in the sentence wins.
    (
        re.compile(
            r"\b(midfield(er)?|playmaker|holding|regista|medio|meio[\s-]?campo|"
            r"meia|trinco)s?\b"
        ),
        "MID",
    ),
    (
        re.compile(
            r"\b(winger|striker|forward|attacker|centre[\s-]?forward|extremo|"
            r"ponta de lanca|avancad[oa]|atacante)s?\b"
        ),
        "FWD",
    ),
)

# Roles, for questions that name one precisely. Each carries its group so the two filters
# can never disagree. Checked against the group patterns by position in the sentence: the
# role only applies when a role phrase is the first position word, so "a midfielder who
# plays as a winger" stays a midfielder, and "medio ala" stays a midfielder too. Compound
# phrases ("defesas centrais") start where their generic word does, which is what lets
# them win against it.
_ROLE_PATTERNS = (
    (
        re.compile(r"\b(goalkeeper|keeper|goalie|guarda[\s-]?redes|guardiao|goleiro)s?\b"),
        "Goalkeeper",
        "GK",
    ),
    (re.compile(r"\b(wing[\s-]?backs?|alas?)\b"), "Wing Back", "DEF"),
    (re.compile(r"\b(full[\s-]?backs?|latera(?:l|is))\b"), "Full Back", "DEF"),
    (
        re.compile(r"\b(cent(?:re|er)[\s-]?backs?|defesas? centra(?:l|is)|centra(?:l|is))\b"),
        "Centre Back",
        "DEF",
    ),
    (
        re.compile(
            r"\b(defensive midfield(?:er)?s?|holding midfield(?:er)?s?|trincos?|"
            r"medios? defensivos?|volantes?)\b"
        ),
        "Defensive Midfield",
        "MID",
    ),
    (
        re.compile(r"\b(attacking midfield(?:er)?s?|playmakers?|medios? ofensivos?|meias?)\b"),
        "Attacking Midfield",
        "MID",
    ),
    (
        re.compile(r"\b(cent(?:re|er)[\s-]?forwards?|strikers?|pontas? de lanca)\b"),
        "Centre Forward",
        "FWD",
    ),
    (re.compile(r"\b(wingers?|extremos?)\b"), "Winger", "FWD"),
)

_UNDER = re.compile(
    r"\b(?:under|younger than|below|u|sub|menos de|abaixo de|ate aos?)[\s-]?(\d{2})\b"
)
_OVER = re.compile(r"\b(?:over|older than|above|mais de|acima de|a partir dos?)[\s-]?(\d{2})\b")
_YOUNG = re.compile(r"\b(young|youngster|prospect|teenager|jovem|jovens|promessa)s?\b")
_VETERAN = re.compile(r"\b(veteran|experienced|older|veteran[oa]|experiente)s?\b")

# Deliberately absent: "media"/"medias" for a female midfielder and "extrema" for a female
# winger. Stripped of accents they collide with "média" (an average) and "extrema"
# (extreme), so "qual e a media de remates" would be silently filtered to midfielders —
# inventing a requirement the question never made. Missing a filter degrades to plain
# ranking; adding a wrong one returns a confidently incorrect answer.


# Demonyms in both languages, keyed by the folded country name they resolve to. Only the
# nationalities actually present in the index are ever applied, so an entry for a country
# nobody comes from costs nothing but is never a filter either.
#
# Portuguese inflects for gender and number, which matters here rather than being pedantry:
# half the pool is women's football, so "brasileira" and "brasileiras" have to land as
# surely as "brasileiro". Two endings cover almost all of them — `[oa]s?` for the
# "brasileiro" shape and `(?:a|es|as)?` for the "frances" shape.
_DEMONYMS: dict[str, str] = {
    # "espanhol" pluralises to "espanhóis", not "espanholes" — the one demonym here whose
    # plural neither of the two common endings builds.
    r"spanish|espanho(?:l|is|la|las)|espanha": "spain",
    r"italians?|italian[oa]s?|italia": "italy",
    r"french|frances(?:a|es|as)?|franca": "france",
    r"english|ingles(?:a|es|as)?|inglaterra": "england",
    r"americans?|american[oa]s?|estados unidos|eua|usa": "united states of america",
    r"germans?|alemao|alema|alemaes|alemas|alemanha": "germany",
    r"brazilians?|brasileir[oa]s?|brasil": "brazil",
    r"argentin(?:e|ian)s?|argentin[oa]s?": "argentina",
    r"senegalese|senegales(?:a|es|as)?": "senegal",
    r"swedish|suec[oa]s?|suecia": "sweden",
    r"dutch|holandes(?:a|es|as)?|neerlandes(?:a|es|as)?|holanda": "netherlands",
    r"portuguese|portugues(?:a|es|as)?": "portugal",
    r"ivorian|marfinenses?|ivory coast|costa do marfim": "cote d'ivoire",
    r"serbians?|servi[oa]s?|servia": "serbia",
    r"scottish|escoces(?:a|es|as)?|escocia": "scotland",
    r"belgians?|belgas?|belgica": "belgium",
    r"danish|dinamarques(?:a|es|as)?|dinamarca": "denmark",
    r"swiss|suic[oa]s?|suica": "switzerland",
    r"welsh|gales(?:a|es|as)?|pais de gales": "wales",
    r"norwegians?|noruegues(?:a|es|as)?|noruega": "norway",
    r"austrians?|austriac[oa]s?": "austria",
    r"irish|irlandes(?:a|es|as)?|irlanda": "ireland",
    r"polish|polac[oa]s?|polones(?:a|es|as)?|polonia": "poland",
    r"uruguayans?|uruguai[oa]s?|uruguai": "uruguay",
    r"japanese|japones(?:a|es|as)?|japao": "japan",
    r"moroccans?|marroquin[oa]s?|marrocos": "morocco",
    r"colombians?|colombian[oa]s?": "colombia",
    r"croatians?|croatas?|croacia": "croatia",
    r"algerians?|argelin[oa]s?|argelia": "algeria",
    r"slovenians?|esloven[oa]s?|eslovenia": "slovenia",
    r"canadians?|canadian[oa]s?|canadenses?": "canada",
    r"nigerians?|nigerian[oa]s?": "nigeria",
    r"chileans?|chilen[oa]s?": "chile",
    r"cameroonians?|camaroneses?|camaroes": "cameroon",
    r"australians?|australian[oa]s?": "australia",
    r"ghanaians?|ganes(?:a|es|as)?|gana": "ghana",
    r"malians?|malian[oa]s?": "mali",
    r"mexicans?|mexican[oa]s?": "mexico",
    r"venezuelans?|venezuelan[oa]s?": "venezuela",
    r"icelandic|islandes(?:a|es|as)?|islandia": "iceland",
}

_DEMONYM_PATTERNS = tuple(
    (re.compile(rf"\b(?:{alternatives})\b"), country) for alternatives, country in _DEMONYMS.items()
)


def _fold_country(name: str) -> str:
    """The comparison key for a country name.

    The stored values are not tidy — "Côte d'Ivoire" carries an accent and "Venezuela
    (Bolivarian Republic)" a parenthetical and a non-breaking space — so the parenthetical
    is dropped and whitespace collapsed before comparing. Matching on the raw string would
    mean a question saying "Venezuela" never found the Venezuelans.
    """
    folded = _fold(name.split("(")[0])
    return " ".join(folded.split())


def _fold(text: str) -> str:
    """Lowercase and strip accents, so "médio" and "medio" are one word to the patterns.

    Length is preserved for the accented vowels Portuguese uses — decomposing "é" and
    dropping the combining mark leaves one character — which matters because the position
    rule compares match offsets to find which position was named first.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _nationality(text: str, countries: Iterable[str] | None) -> str | None:
    """Resolve a nationality to the exact spelling the index uses, or to nothing.

    `countries` is the vocabulary from the index. Without it there is no nationality
    filter at all, which is the safe default: narrowing on a country the pool has never
    heard of returns an empty shortlist and an answer that no such player exists, when the
    truth is that none was ever loaded.
    """
    if not countries:
        return None
    known = {_fold_country(c): c for c in countries}

    for pattern, folded in _DEMONYM_PATTERNS:
        if folded in known and pattern.search(text):
            return known[folded]
    # Then the country's own name, which needs no table and covers the other ninety-odd.
    for folded, original in known.items():
        if folded and re.search(rf"\b{re.escape(folded)}\b", text):
            return original
    return None


def parse_constraints(question: str, countries: Iterable[str] | None = None) -> Constraints:
    """Pull the absolute requirements out of a question; everything else is left to rank."""
    text = _fold(question)

    foot = next((value for pattern, value in _FOOT_PATTERNS if pattern.search(text)), None)

    found = [
        (match.start(), value)
        for pattern, value in _POSITION_PATTERNS
        if (match := pattern.search(text)) is not None
    ]
    position = min(found)[1] if found else None

    roles = [
        (match.start(), role, group)
        for pattern, role, group in _ROLE_PATTERNS
        if (match := pattern.search(text)) is not None
    ]
    role = None
    if roles:
        start, named_role, group = min(roles)
        # Only when no generic position word comes earlier: that word is the subject.
        if not found or start <= min(found)[0]:
            role, position = named_role, group

    max_age: float | None = None
    min_age: float | None = None
    if (explicit := _UNDER.search(text)) is not None:
        max_age = float(explicit.group(1))
    elif _YOUNG.search(text):
        max_age = YOUNG_MAX_AGE
    if (explicit := _OVER.search(text)) is not None:
        min_age = float(explicit.group(1))
    elif _VETERAN.search(text):
        min_age = VETERAN_MIN_AGE

    return Constraints(
        foot=foot,
        position_group=position,
        max_age=max_age,
        min_age=min_age,
        nationality=_nationality(text, countries),
        position_role=role,
    )
