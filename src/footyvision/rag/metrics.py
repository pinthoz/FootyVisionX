"""Read the metric a scouting question asks about, out of the question itself.

A profile names a player's four strongest metrics and their single weakest, because a
document listing all seventeen reads like a spreadsheet and embeds like one too. The cost
turned up in the RAGAS evaluation: asked "quais os medios experientes com mais desarmes?",
the assistant answered that it had no data on tackles. It was telling the truth about its
context and not about the database, which holds `tackles_per90` for every player — the
number just never reached the prompt, because tackles were nobody's top-four strength.

So the metric is pulled out of the question the same way the hard constraints are, by
matching against a fixed vocabulary rather than asking a model. Deterministic, free, and
it cannot invent a metric that was never mentioned. Anything unrecognised yields nothing
and the assistant behaves as it did before, which is the safe direction to fail in.
"""

from __future__ import annotations

import re
import unicodedata

# Phrase to per-90 column, in both languages. Order does not matter here because the
# lookup is built longest-phrase-first: "passes progressivos" has to win against
# "passes", and "golos esperados" against "golos", or every compound metric collapses
# into the simple one it contains.
_VOCABULARY: dict[str, tuple[str, ...]] = {
    "goals_per90": ("goals", "goalscoring", "golos", "gols", "scores", "score", "marcam"),
    "xg_per90": ("xg", "expected goals", "golos esperados", "gols esperados"),
    # The verbs too: questions ask who "shoots" or "remata" at least as often as they
    # ask about "shots", and a missed metric silently turns a ranking question into a
    # similarity one.
    "shots_per90": (
        "shots",
        "shooting",
        "shoot",
        "shoots",
        "remates",
        "rematam",
        "remata",
        "chutes",
    ),
    "assists_per90": ("assists", "assistencias"),
    "passes_completed_per90": (
        "completed passes",
        "complete the most passes",
        "complete passes",
        "pass completion",
        "passes completos",
        "passes certos",
    ),
    "progressive_passes_per90": ("progressive passes", "passes progressivos"),
    "passes_per90": ("passes", "passing", "passe"),
    "dribbles_completed_per90": (
        "completed dribbles",
        "successful dribbles",
        "dribles completos",
        "dribles ganhos",
    ),
    "dribbles_per90": ("dribbles", "dribbling", "dribles", "driblar"),
    "progressive_carries_per90": ("progressive carries", "conducoes progressivas"),
    "carries_per90": ("carries", "conducoes", "conducao", "conduzem", "conduz a bola"),
    "tackles_per90": ("tackles", "tackling", "desarmes", "desarme"),
    "interceptions_per90": ("interceptions", "intercecoes", "intercetacoes"),
    "blocks_per90": ("blocks", "bloqueios"),
    "clearances_per90": ("clearances", "alivios", "cortes"),
    "ball_recoveries_per90": (
        "ball recoveries",
        "recoveries",
        "recuperacoes",
        "bolas recuperadas",
        # Portuguese asks for these as a verb far more often than as a noun: "os
        # guarda-redes que mais recuperam bolas", not "com mais recuperacoes".
        "recuperam bolas",
        "recupera bolas",
        "recuperar bolas",
    ),
    "pressures_per90": (
        "pressures",
        "pressing",
        "press",
        "presses",
        "pressoes",
        "pressionam",
        "pressiona",
        "pressionar",
    ),
}

# Longest first, so a compound phrase is matched before the simple one inside it.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rf"\b{re.escape(phrase)}\b"), column)
    for phrase, column in sorted(
        ((p, c) for c, phrases in _VOCABULARY.items() for p in phrases),
        key=lambda pair: -len(pair[0]),
    )
)

# Enough to answer "who has more tackles and interceptions", short enough that the added
# context cannot crowd out the profiles it is meant to supplement.
MAX_METRICS = 3


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def metrics_in(question: str) -> list[str]:
    """The per-90 columns a question names, in the order they are first mentioned.

    Ordered by mention rather than by the vocabulary, so "tackles and interceptions" and
    "interceptions and tackles" put the subject of the question first, and the cap trims
    the tail rather than an arbitrary member.
    """
    text = _fold(question)
    found: dict[str, int] = {}
    claimed: list[tuple[int, int]] = []

    # Longest phrase first, and a match inside ground already claimed by a longer one is
    # not a second metric: "passes progressivos" contains "passes", which belongs to a
    # different column, so deduplicating by column alone would report both.
    for pattern, column in _PATTERNS:
        for match in pattern.finditer(text):
            if any(start < match.end() and match.start() < end for start, end in claimed):
                continue
            claimed.append((match.start(), match.end()))
            found.setdefault(column, match.start())
            break
    return [c for c, _ in sorted(found.items(), key=lambda kv: kv[1])][:MAX_METRICS]


# Words that turn a question about a metric into a question about who leads it. English
# and Portuguese, folded like the question is. "mais" alone is enough: "quem faz mais
# cortes" asks for an ordering exactly as "who makes the most clearances" does.
_LEADER_WORDS = re.compile(
    r"\b(most|highest|lead|leads|leading|leader|leaders|top|more|mais|maior|maiores|lidera|lideram)\b"
)


def asks_for_leaders(question: str) -> bool:
    """Whether the question wants players ordered by a metric rather than described by one.

    "Which full-backs make the most tackles?" does; "a full-back who tackles well" does
    not, and is left to similarity, which is the right tool for a description.
    """
    return bool(_LEADER_WORDS.search(_fold(question)))


def label(column: str) -> str:
    """ "ball_recoveries_per90" -> "ball recoveries"; "xg_per90" -> "xG"."""
    name = column.removesuffix("_per90")
    return "xG" if name == "xg" else name.replace("_", " ")
