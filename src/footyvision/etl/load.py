"""Orchestrate extract -> aggregate -> load for a competition/season.

Idempotent: dimension rows are upserted by primary key; a match's player stats are
deleted and re-inserted on reload so re-running never duplicates data.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from footyvision.db.models import (
    METRIC_COLUMNS,
    Competition,
    Match,
    Player,
    PlayerMatchStats,
    PlayerSeasonStats,
    Season,
    Team,
)
from footyvision.etl import statsbomb
from footyvision.etl.aggregate import aggregate_match


def _parse_date(value) -> date | None:
    if not value or pd.isna(value):
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _get_or_create_season(
    session: Session, competition_id: int, sb_season_id: int, name: str
) -> Season:
    season = session.scalar(
        select(Season).where(
            Season.competition_id == competition_id, Season.sb_season_id == sb_season_id
        )
    )
    if season is None:
        season = Season(competition_id=competition_id, sb_season_id=sb_season_id, name=name)
        session.add(season)
    else:
        season.name = name
    return season


def load_match(session: Session, match_row, competition_id: int, sb_season_id: int) -> int:
    """Load a single match's events. Returns number of player rows written."""
    match_id = int(match_row["match_id"])
    events = statsbomb.events(match_id)
    agg = aggregate_match(events)
    if agg.empty:
        return 0

    # Upsert dimension rows first and flush so their PKs exist before the match /
    # stats rows reference them. The ORM's flush ordering is driven by relationships,
    # and these FKs are bare columns, so we enforce parent-before-child explicitly.
    # Deduplicate by PK: with autoflush off, merging the same PK twice before a flush
    # would create duplicate pending INSERTs (a team repeats once per player).
    teams = {int(r.team_id): r.team for r in agg.itertuples(index=False)}
    players = {int(r.player_id): r.player for r in agg.itertuples(index=False)}
    name_to_team_id = {name: tid for tid, name in teams.items()}
    for tid, tname in teams.items():
        session.merge(Team(id=tid, name=tname))
    for pid, pname in players.items():
        session.merge(Player(id=pid, name=pname))
    session.flush()

    session.merge(
        Match(
            id=match_id,
            competition_id=competition_id,
            sb_season_id=sb_season_id,
            match_date=_parse_date(match_row.get("match_date")),
            home_team_id=name_to_team_id.get(match_row.get("home_team")),
            away_team_id=name_to_team_id.get(match_row.get("away_team")),
        )
    )
    session.flush()

    session.execute(delete(PlayerMatchStats).where(PlayerMatchStats.match_id == match_id))
    for r in agg.itertuples(index=False):
        row = r._asdict()
        session.add(
            PlayerMatchStats(
                match_id=match_id,
                player_id=int(row["player_id"]),
                team_id=int(row["team_id"]),
                position=row.get("position"),
                minutes=float(row.get("minutes", 0.0)),
                **{m: float(row.get(m, 0.0)) for m in METRIC_COLUMNS},
            )
        )
    return len(agg)


def load_competition_season(
    session: Session,
    competition_id: int,
    season_id: int,
    limit: int | None = None,
    on_match=None,
) -> int:
    """Load all (or `limit`) matches of a competition season. Returns matches loaded."""
    comps = statsbomb.competitions()
    meta = comps[(comps.competition_id == competition_id) & (comps.season_id == season_id)]
    if meta.empty:
        raise ValueError(f"No open data for competition={competition_id} season={season_id}")
    meta = meta.iloc[0]

    session.merge(
        Competition(
            id=competition_id,
            name=meta["competition_name"],
            country=meta.get("country_name"),
            gender=meta.get("competition_gender"),
        )
    )
    _get_or_create_season(session, competition_id, season_id, str(meta["season_name"]))
    session.flush()  # persist competition + season before any match references them

    matches_df = statsbomb.matches(competition_id, season_id)
    if limit:
        matches_df = matches_df.head(limit)

    loaded = 0
    for _, match_row in matches_df.iterrows():
        load_match(session, match_row, competition_id, season_id)
        loaded += 1
        if on_match:
            on_match(match_row, loaded, len(matches_df))
    session.commit()
    return loaded


def rebuild_season_aggregates(
    session: Session, competition_id: int, season_id: int, min_minutes: int
) -> int:
    """Recompute player_season_stats (totals + per-90) from player_match_stats.

    Players below `min_minutes` for the season are excluded to avoid small-sample noise.
    Returns the number of player-season rows written.
    """
    session.execute(
        delete(PlayerSeasonStats).where(
            PlayerSeasonStats.competition_id == competition_id,
            PlayerSeasonStats.sb_season_id == season_id,
        )
    )

    sums = [func.sum(getattr(PlayerMatchStats, m)).label(m) for m in METRIC_COLUMNS]
    stmt = (
        select(
            PlayerMatchStats.player_id,
            func.count(func.distinct(PlayerMatchStats.match_id)).label("matches_played"),
            func.sum(PlayerMatchStats.minutes).label("minutes"),
            *sums,
        )
        .join(Match, Match.id == PlayerMatchStats.match_id)
        .where(Match.competition_id == competition_id, Match.sb_season_id == season_id)
        .group_by(PlayerMatchStats.player_id)
    )

    # Primary position = position with the most minutes for that player in the season.
    pos_rows = session.execute(
        select(
            PlayerMatchStats.player_id,
            PlayerMatchStats.position,
            func.sum(PlayerMatchStats.minutes).label("mins"),
        )
        .join(Match, Match.id == PlayerMatchStats.match_id)
        .where(Match.competition_id == competition_id, Match.sb_season_id == season_id)
        .group_by(PlayerMatchStats.player_id, PlayerMatchStats.position)
    ).all()
    primary_pos: dict[int, tuple[float, str | None]] = {}
    for pid, position, mins in pos_rows:
        best = primary_pos.get(pid, (-1.0, None))
        if (mins or 0.0) > best[0]:
            primary_pos[pid] = (mins or 0.0, position)

    written = 0
    for row in session.execute(stmt).all():
        data = row._asdict()
        minutes = float(data["minutes"] or 0.0)
        if minutes < min_minutes:
            continue
        factor = 90.0 / minutes if minutes else 0.0
        totals = {m: float(data[m] or 0.0) for m in METRIC_COLUMNS}
        per90 = {f"{m}_per90": v * factor for m, v in totals.items()}
        session.add(
            PlayerSeasonStats(
                player_id=int(data["player_id"]),
                competition_id=competition_id,
                sb_season_id=season_id,
                primary_position=primary_pos.get(data["player_id"], (0.0, None))[1],
                matches_played=int(data["matches_played"] or 0),
                minutes=minutes,
                **totals,
                **per90,
            )
        )
        written += 1
    session.commit()
    return written
