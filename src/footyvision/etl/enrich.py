"""Fill in the player details the event stream does not carry.

`sb.events()` names a player and nothing else, so the players table ends up with an id
and a name. The lineup feed for the same match carries two more fields worth having:

  * `player_nickname` — the short form the player is actually known by ("Joselu" for
    "José Luis Sanmartín Mato"). Other providers use this form, so it is a far better
    key for cross-source matching than the full legal name.
  * `country` — nationality, a basic scouting filter that was sitting unused.

Date of birth, preferred foot and height are absent from the open data entirely and are
matched from the Transfermarkt export instead. That export is men's football, so those
three fields exist for the men's leagues and are empty for the women's — which is why the
age and foot filters return nothing from the women's competitions rather than returning
them wrongly. Filling that gap needs a women's biographical source, not more matching.
"""

from __future__ import annotations

import unicodedata
import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from statsbombpy import sb

from footyvision.db.models import Competition, Match, Player, PlayerSeasonStats

warnings.filterwarnings("ignore", message="credentials were not supplied")

# How many matches to read between commits. Small enough that a dropped connection costs
# seconds of re-reading, large enough not to make a round trip per match.
CHECKPOINT_MATCHES = 50


def enrich_from_lineups(
    session: Session,
    on_match: Callable[[int, int, int], None] | None = None,
) -> dict[str, int]:
    """Walk loaded matches filling nickname and country, stopping once nothing is missing.

    A player only appears in the lineups of matches they were in the squad for, so this
    has to walk matches rather than players. It does not have to walk *all* of them: the
    squads repeat heavily, so the set of players still missing detail empties long before
    the fixture list does, and the loop exits there.

    Matches are walked in order of how many pending players their competition holds. Left
    in database order, a second run after a new league is loaded spends a thousand network
    calls re-reading squads it has already filled before it reaches the league that needs
    it — and the early exit never triggers, because the players it is looking for are all
    at the end.

    Progress is committed as it goes. The walk is thousands of network reads long and a
    single commit at the end means a connection dropped in the last minute discards all of
    it — which is exactly what happened. Because `pending` is derived from what is already
    stored, a re-run resumes from the last checkpoint rather than starting over.
    """
    players = {p.id: p for p in session.scalars(select(Player))}
    # Missing *either* field, and cleared on being seen rather than on being filled. The
    # lineup feed has one entry per player, so a second sighting cannot add anything: a
    # player it lists without a nickname has none to give. Keying on "filled" instead left
    # everyone with a country but no nickname permanently pending, and the loop re-read the
    # whole fixture list every run looking for them.
    pending = {pid for pid, p in players.items() if p.nickname is None or p.country is None}

    need_by_competition: dict[int, int] = {}
    for competition_id, player_id in session.execute(
        select(PlayerSeasonStats.competition_id, PlayerSeasonStats.player_id).distinct()
    ):
        if player_id in pending:
            need_by_competition[competition_id] = need_by_competition.get(competition_id, 0) + 1

    match_ids = [
        match_id
        for match_id, _ in sorted(
            session.execute(select(Match.id, Match.competition_id)),
            key=lambda row: -need_by_competition.get(row[1], 0),
        )
    ]

    filled_nickname = 0
    filled_country = 0
    seen = 0
    unavailable = 0

    for match_id in match_ids:
        if not pending:
            break
        seen += 1
        try:
            lineups = sb.lineups(match_id=match_id)
        except Exception:
            # A single unavailable lineup should not abandon the rest of the walk, but a
            # whole competition failing looks exactly like a whole competition already
            # being complete, so the failures are counted and reported.
            unavailable += 1
            continue

        for frame in lineups.values():
            for row in frame.itertuples(index=False):
                pid = int(row.player_id)
                player = players.get(pid)
                if player is None or pid not in pending:
                    continue
                nickname = getattr(row, "player_nickname", None)
                if player.nickname is None and isinstance(nickname, str) and nickname.strip():
                    player.nickname = nickname.strip()[:120]
                    filled_nickname += 1
                country = getattr(row, "country", None)
                if player.country is None and isinstance(country, str) and country.strip():
                    player.country = country.strip()[:120]
                    filled_country += 1
                pending.discard(pid)

        if seen % CHECKPOINT_MATCHES == 0:
            session.commit()
        if on_match:
            on_match(match_id, seen, len(pending))

    session.commit()
    return {
        "matches_read": seen,
        "unavailable": unavailable,
        "nicknames": filled_nickname,
        "countries": filled_country,
        "still_missing": len(pending),
    }


def _normalize(name: str) -> str:
    ascii_ = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(ascii_.lower().split())


def _transfermarkt_lookup(data_dir: str, columns: list[str]) -> dict[str, dict]:
    """Name-keyed view of the Transfermarkt export, with ambiguous names left out.

    A name appearing more than once cannot be resolved by name alone, so those rows are
    dropped rather than guessed at — a wrong date of birth is worse than none.
    """
    frame = pd.read_csv(Path(data_dir) / "players.csv", usecols=["name", *columns])
    frame = frame.dropna(subset=["name"])
    frame["key"] = frame["name"].map(_normalize)
    frame = frame.drop_duplicates("key", keep=False)
    return {r["key"]: {c: r[c] for c in columns} for r in frame.to_dict("records")}


def _covered_by_export(session: Session) -> set[int]:
    """Players the Transfermarkt export could plausibly describe: the men.

    The Kaggle export is men's football, and of the 989 women in the pool exactly 2 have a
    name that appears in it — those are collisions, not matches, and taking them would file
    a man's date of birth and preferred foot against a woman. The export cannot be matched
    on gender because it has no such column, so the filter belongs on our side.
    """
    rows = session.execute(
        select(PlayerSeasonStats.player_id)
        .join(Competition, Competition.id == PlayerSeasonStats.competition_id)
        .where(Competition.gender == "male")
        .distinct()
    )
    return {player_id for (player_id,) in rows}


def _match(player: Player, lookup: dict[str, dict]) -> dict | None:
    """Nickname first, full name second.

    StatsBomb records full legal names ("José Luis Sanmartín Mato") while Transfermarkt
    uses the short form ("Joselu"), so the nickname is usually the one that lands.
    """
    for candidate in (player.nickname, player.name):
        if candidate:
            hit = lookup.get(_normalize(candidate))
            if hit is not None:
                return hit
    return None


def backfill_birthdates(session: Session, data_dir: str = "data") -> dict[str, int]:
    """Attach dates of birth from the Transfermarkt export, matched by name."""
    lookup = _transfermarkt_lookup(data_dir, ["date_of_birth"])
    eligible = _covered_by_export(session)
    matched = 0
    players = list(session.scalars(select(Player)))

    for player in players:
        if player.date_of_birth is not None or player.id not in eligible:
            continue
        hit = _match(player, lookup)
        if hit is None:
            continue
        dob = pd.to_datetime(hit["date_of_birth"], errors="coerce")
        if pd.notna(dob):
            player.date_of_birth = dob.date()
            matched += 1

    session.commit()
    return {
        "players": len(players),
        "matched": matched,
        # Counted within the men's pool only: a woman with no date of birth is not an
        # unmatched row, she is outside what this source covers at all.
        "unmatched": sum(1 for p in players if p.date_of_birth is None and p.id in eligible),
        "outside_export": sum(1 for p in players if p.id not in eligible),
    }


_FEET = {"right", "left", "both"}


def backfill_physical(session: Session, data_dir: str = "data") -> dict[str, int]:
    """Attach preferred foot and height from the Transfermarkt export.

    Foot is the one that matters to the model: a direct laterality signal worth roughly
    ten accuracy points on exact-position prediction, which the per-90 counts cannot
    supply because they carry no side. Height is stored as a scouting attribute only —
    the same ablation showed it adds nothing to position prediction.
    """
    lookup = _transfermarkt_lookup(data_dir, ["foot", "height_in_cm"])
    eligible = _covered_by_export(session)
    feet = 0
    heights = 0
    players = list(session.scalars(select(Player)))

    for player in players:
        if player.id not in eligible:
            continue
        hit = _match(player, lookup)
        if hit is None:
            continue
        foot = hit.get("foot")
        if player.foot is None and isinstance(foot, str) and foot.lower() in _FEET:
            player.foot = foot.lower()
            feet += 1
        height = hit.get("height_in_cm")
        # The export uses 0 for "unknown"; a 0 cm player is worse than a null one.
        if player.height_cm is None and pd.notna(height) and float(height) > 100:
            player.height_cm = float(height)
            heights += 1

    session.commit()
    return {
        "players": len(players),
        "feet": feet,
        "heights": heights,
        "without_foot": sum(1 for p in players if p.foot is None and p.id in eligible),
        "outside_export": sum(1 for p in players if p.id not in eligible),
    }
