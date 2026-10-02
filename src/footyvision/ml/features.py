"""Feature matrix and position grouping for the similarity engine.

The comparable features are the per-90 rates (not raw totals), because players differ
in minutes played. Standardisation and similarity are always done *within a position
group* — comparing a goalkeeper's progressive passes to a winger's is meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from footyvision.config import get_settings
from footyvision.db.models import (
    METRIC_COLUMNS,
    Competition,
    Match,
    Player,
    PlayerSeasonStats,
)
from footyvision.db.quality import fragment_seasons

# The per-90 columns that describe playing style — the similarity feature space.
# Deliberately closed: similarity and the percentile radars both read this, and a
# non-style attribute like foot or height has no business shaping "plays like".
PER90_FEATURES: tuple[str, ...] = tuple(f"{m}_per90" for m in METRIC_COLUMNS)

# Preferred foot, one-hot. Not a style metric, so it stays out of PER90_FEATURES — but
# it is the one attribute carrying a *side*, which is why the exact-position classifier
# takes it. See ml/talent.classifier_features for the ablation.
FOOT_FEATURES: tuple[str, ...] = ("foot_left", "foot_right", "foot_both")


def position_group(position: str | None) -> str:
    """Map a granular StatsBomb position to a broad group for fair comparison.

    Order matters: 'Wing Back' contains both 'back' and 'wing' and must resolve to DEF.
    """
    if not position:
        return "Unknown"
    p = position.lower()
    if "goalkeeper" in p:
        return "GK"
    if "back" in p:
        return "DEF"
    if "midfield" in p:
        return "MID"
    if "wing" in p or "forward" in p or "striker" in p:
        return "FWD"
    return "MID"


def position_role(position: str | None) -> str:
    """Map a StatsBomb position to a side-agnostic playing role.

    Deliberately drops left/right. The 17 features are all counts — goals, passes,
    tackles — and carry no lateral information whatsoever, so a left back and a right
    back are the same point in feature space. Measured: a 23-class model gets the exact
    label right 39% of the time but the role right 73%, and 56% of its errors are pure
    left/right swaps.

    Order matters: 'Wing Back' contains 'back' and 'wing', and 'Center Attacking
    Midfield' contains 'midfield', so the specific tests have to come first.
    """
    if not position:
        return "Unknown"
    p = position.lower()
    if "goalkeeper" in p:
        return "Goalkeeper"
    if "wing back" in p:
        return "Wing Back"
    if "center back" in p:
        return "Centre Back"
    if "back" in p:
        return "Full Back"
    if "defensive midfield" in p:
        return "Defensive Midfield"
    if "attacking midfield" in p:
        return "Attacking Midfield"
    if "center midfield" in p:
        return "Central Midfield"
    if "midfield" in p:
        return "Wide Midfield"
    if "wing" in p:
        return "Winger"
    if "forward" in p or "striker" in p:
        return "Centre Forward"
    return "Unknown"


# Feature frames, keyed by the arguments that produced them. Every request that scores,
# ranks, compares or plots a player rebuilt this from the database, and each copy is about
# 16MB: twenty-four concurrent requests peaked at 504MB against Render's 512MB ceiling,
# which is the out-of-memory restart. The pool only changes when a season is imported, so
# the rebuild bought nothing — and, like the model caches, it means the API must be
# restarted after an import.
_FRAME_CACHE: dict[tuple, pd.DataFrame] = {}
# Callers pass a min_minutes query parameter, so the key space is caller-controlled and
# has to be bounded; a handful of distinct pools is far more than anything real uses.
_FRAME_CACHE_MAX = 8


def cached_feature_frame(session: Session, **kwargs) -> pd.DataFrame:
    """`load_feature_frame`, built once per distinct set of arguments.

    Returns a shallow copy: the columns can be added to or reassigned by a caller without
    reaching the cached original, while the underlying arrays stay shared, so the copy
    costs almost nothing. Callers that mutate values in place still need their own deep
    copy, which is what the model training code already takes.
    """
    key = tuple(sorted(kwargs.items()))
    frame = _FRAME_CACHE.get(key)
    if frame is None:
        if len(_FRAME_CACHE) >= _FRAME_CACHE_MAX:
            _FRAME_CACHE.clear()
        frame = load_feature_frame(session, **kwargs)
        _FRAME_CACHE[key] = frame
    return frame.copy(deep=False)


def load_feature_frame(
    session: Session,
    min_minutes: float | None = None,
    competition_id: int | None = None,
    season_id: int | None = None,
    include_fragments: bool = False,
) -> pd.DataFrame:
    """Load player-season rows (with names) into a DataFrame, one row per player-season.

    A `position_group` column is added. Filters are optional so the caller can scope
    the comparison pool (e.g. to one competition/season).
    """
    stmt = (
        select(
            PlayerSeasonStats.player_id,
            Player.name.label("name"),
            PlayerSeasonStats.competition_id,
            Competition.name.label("competition"),
            Competition.gender.label("gender"),
            PlayerSeasonStats.sb_season_id,
            Player.date_of_birth.label("date_of_birth"),
            Player.foot.label("foot"),
            Player.height_cm.label("height_cm"),
            # Labelled "nationality" rather than "country" because the competition brings
            # a country of its own — a Brazilian in La Liga has both, and they differ.
            Player.country.label("nationality"),
            PlayerSeasonStats.primary_position,
            PlayerSeasonStats.matches_played,
            PlayerSeasonStats.minutes,
            *[getattr(PlayerSeasonStats, f) for f in PER90_FEATURES],
        )
        .join(Player, Player.id == PlayerSeasonStats.player_id)
        .join(Competition, Competition.id == PlayerSeasonStats.competition_id)
    )

    if min_minutes is not None:
        stmt = stmt.where(PlayerSeasonStats.minutes >= min_minutes)
    if competition_id is not None:
        stmt = stmt.where(PlayerSeasonStats.competition_id == competition_id)
    if season_id is not None:
        stmt = stmt.where(PlayerSeasonStats.sb_season_id == season_id)

    engine: Engine = session.get_bind()
    frame = pd.read_sql(stmt, engine)
    if not include_fragments:
        frame = _drop_fragment_seasons(session, frame)
    frame["position_group"] = frame["primary_position"].map(position_group)
    frame["position_role"] = frame["primary_position"].map(position_role)
    frame["peer_group"] = _peer_groups(frame)
    frame["age"] = _age_at_season(session, frame)
    # NaN rather than 0 where the foot is unknown: the tree models treat NaN as missing
    # and route it, whereas a 0 would assert "not left-footed".
    for column, value in zip(FOOT_FEATURES, ("left", "right", "both"), strict=True):
        frame[column] = (frame["foot"] == value).astype(float)
        frame.loc[frame["foot"].isna(), column] = float("nan")
    return frame


def peer_column(frame: pd.DataFrame) -> str:
    """The column to rank and standardise within.

    `peer_group` when the frame carries it, `position_group` otherwise — frames built by
    hand in tests have no competition behind them, and falling back keeps them working.
    """
    return "peer_group" if "peer_group" in frame.columns else "position_group"


def _peer_groups(frame: pd.DataFrame) -> pd.Series:
    """Position group, split by the gender of the competition it was played in.

    A single-population database gets exactly the old behaviour: with one gender present
    the labels differ only by a prefix, and every group holds the same players it did.
    """
    gender = frame.get("gender")
    if gender is None:
        return frame["position_group"]
    known = gender.fillna("unknown").astype(str)
    return known.str.slice(0, 1).str.upper() + ":" + frame["position_group"].astype(str)


def _drop_fragment_seasons(session: Session, frame: pd.DataFrame) -> pd.DataFrame:
    """Remove players whose season is too incomplete to compare anyone against.

    Percentiles and z-scores are computed over whichever rows are in this frame, so a
    fragment carries more than its own unreliable numbers.

    Nothing in this database currently qualifies, and the near miss is worth recording.
    The Bundesliga 2015/16 here is 34 matches, which against a league of eighteen clubs
    reads as 11% of a season — but it is Bayer Leverkusen's export, their full campaign,
    and its nineteen players are better sampled than the pool median at up to
    thirty-three appearances. `db.quality` measures a single-club season against that
    club's own fixture list for exactly this reason. Dropping it would have deleted
    complete seasons, and measurement said the ranks it supposedly distorted moved by at
    most 0.8 of a percentile point anyway.
    """
    threshold = get_settings().min_season_coverage
    if threshold <= 0 or frame.empty:
        return frame

    fragments = fragment_seasons(session, threshold)
    if not fragments:
        return frame

    keys = list(zip(frame["competition_id"], frame["sb_season_id"], strict=True))
    keep = [key not in fragments for key in keys]
    return frame[pd.Series(keep, index=frame.index)].copy()


def _age_at_season(session: Session, frame: pd.DataFrame) -> pd.Series:
    """Age in years at the midpoint of the season the row belongs to.

    Not age today: a 2015/16 row describes a player as he was then. The reference comes
    from the fixtures themselves rather than being parsed out of the season name, so it
    works for "2015/2016" and "2023" alike.
    """
    if frame.empty:
        return pd.Series(dtype=float)

    spans = pd.read_sql(
        select(
            Match.competition_id,
            Match.sb_season_id,
            func.min(Match.match_date).label("first"),
            func.max(Match.match_date).label("last"),
        ).group_by(Match.competition_id, Match.sb_season_id),
        session.get_bind(),
    )
    if spans.empty:
        return pd.Series(np.nan, index=frame.index)

    spans["first"] = pd.to_datetime(spans["first"])
    spans["last"] = pd.to_datetime(spans["last"])
    spans["midpoint"] = spans["first"] + (spans["last"] - spans["first"]) / 2

    merged = frame[["competition_id", "sb_season_id"]].merge(
        spans[["competition_id", "sb_season_id", "midpoint"]],
        on=["competition_id", "sb_season_id"],
        how="left",
    )
    dob = pd.to_datetime(frame["date_of_birth"], errors="coerce")
    years = (merged["midpoint"].to_numpy() - dob.to_numpy()) / np.timedelta64(365, "D")
    return pd.Series(np.round(years.astype(float), 1), index=frame.index)
