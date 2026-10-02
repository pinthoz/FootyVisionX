"""Build a short English text profile per player — the documents the RAG store embeds.

The text is natural-language *playing-style* prose (not a raw metric dump): a retrieval
model discriminates far better on descriptive language that resembles how scouts phrase
queries than on a table of numbers. Written in English because the embedding models are
English-first; the assistant's final answer can still be Portuguese.

No pronouns: the pool is roughly two fifths women, and a template that said "he excels
at" was simply wrong for a thousand players — and the assistant repeated it, because the
profile is the only thing it knows about them.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from footyvision.ml.features import PER90_FEATURES, peer_column
from footyvision.ml.scoring import score_frame

# Map each per-90 metric to a scout-style phrase, so a high percentile reads naturally.
STYLE_PHRASES: dict[str, str] = {
    "goals_per90": "scoring goals",
    "xg_per90": "getting into dangerous scoring positions",
    "shots_per90": "shooting frequently",
    "assists_per90": "creating assists for teammates",
    "passes_per90": "high passing volume",
    "passes_completed_per90": "retaining possession with accurate passing",
    "progressive_passes_per90": "progressing the ball forward with passes",
    "dribbles_per90": "taking opponents on",
    "dribbles_completed_per90": "beating defenders with dribbles",
    "carries_per90": "carrying the ball",
    "progressive_carries_per90": "driving forward with the ball",
    "tackles_per90": "winning tackles",
    "interceptions_per90": "reading the game and intercepting passes",
    "blocks_per90": "blocking shots and passes",
    "clearances_per90": "clearing danger from defence",
    "ball_recoveries_per90": "recovering loose balls",
    "pressures_per90": "pressing opponents high",
}


def build_profiles(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """One natural-language profile document per player."""
    scored = score_frame(frame)
    docs: list[dict[str, Any]] = []

    for _, sub in scored.groupby(peer_column(scored)):
        percentiles = sub[list(PER90_FEATURES)].rank(pct=True) * 100.0
        for idx, r in sub.iterrows():
            ranked = percentiles.loc[idx].sort_values(ascending=False)
            strengths = ", ".join(_phrase_with_value(r, f, ranked[f]) for f in ranked.index[:4])
            weakest = ranked.index[-1]
            text = (
                f"{r['name']} is a {r['primary_position']} ({r['position_group']}) in "
                f"{r['competition']}{_biography(r)}. "
                f"Playing style: excels at {strengths}. "
                f"Weaker at {_phrase_with_value(r, weakest, ranked[weakest])}. "
                f"Performance score {r['performance_score']:.0f} out of 100 "
                f"over {int(r['minutes'])} minutes played."
            )
            docs.append(
                {
                    "player_id": int(r["player_id"]),
                    "name": r["name"],
                    "text": text,
                    # Carried beside the prose so retrieval can filter on them before
                    # ranking: an embedding is poor at hard constraints, because
                    # "left-footed" is one token among dozens and never dominates.
                    "foot": r.get("foot") if isinstance(r.get("foot"), str) else None,
                    "age": float(r["age"]) if pd.notna(r.get("age")) else float("nan"),
                    "position_group": r["position_group"],
                    # The one hard attribute recorded for every player in the pool: it
                    # comes from the lineup feed, which covers the women's competitions
                    # too, unlike the men's-only source behind age, foot and height.
                    "nationality": (
                        r.get("nationality") if isinstance(r.get("nationality"), str) else None
                    ),
                    # Every per-90 value, not only the five the prose names. The text
                    # stays a readable style summary; these ride alongside so a question
                    # about a metric outside a player's best four can still be answered.
                    "metrics": {f: float(r[f]) for f in PER90_FEATURES},
                }
            )
    return docs


def _biography(row: pd.Series) -> str:
    """ ", from Brazil, 24 years old, left-footed, 1.85m tall" — whichever is known.

    Scouts ask for these constantly ("a young left-footed winger"), and without them in
    the text the retriever has nothing to match such a question against. Each is optional,
    and unevenly so: nationality is recorded for every player, while date of birth, foot
    and height come from a men's football export and are missing for the women entirely.
    """
    parts: list[str] = []
    nationality = row.get("nationality")
    if isinstance(nationality, str) and nationality:
        parts.append(f"from {nationality}")
    age = row.get("age")
    if pd.notna(age):
        parts.append(f"{int(round(float(age)))} years old")
    foot = row.get("foot")
    if isinstance(foot, str) and foot:
        parts.append("two-footed" if foot == "both" else f"{foot}-footed")
    height = row.get("height_cm")
    if pd.notna(height):
        parts.append(f"{float(height) / 100:.2f}m tall")
    return ", " + ", ".join(parts) if parts else ""


def _phrase_with_value(row: pd.Series, feature: str, percentile: float) -> str:
    """ "scoring goals (0.61 per 90, 97th percentile for a FWD)".

    The numbers matter: without them the assistant cannot answer comparative questions
    ("who has the most xG?") because every profile reduces to the same stock phrases.
    """
    return (
        f"{STYLE_PHRASES[feature]} "
        f"({row[feature]:.2f} {_metric_label(feature)} per 90, "
        f"{_ordinal(round(percentile))} percentile for a {row['position_group']})"
    )


def _metric_label(feature: str) -> str:
    """ "ball_recoveries_per90" -> "ball recoveries"; "xg_per90" -> "xG"."""
    name = feature.removesuffix("_per90")
    return "xG" if name == "xg" else name.replace("_", " ")


def _ordinal(n: int) -> str:
    """1 -> 1st, 2 -> 2nd, 83 -> 83rd, 11 -> 11th."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
