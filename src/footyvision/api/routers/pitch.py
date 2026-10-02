"""Where a player acts on the pitch, and where they shoot from.

Served from models/spatial/spatial.json, written by `footyvision spatial` from every loaded
match's raw events. The events themselves are a gigabyte of JSON the API has no business
downloading; the published grids and shot lists are a few megabytes, read once.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from footyvision.api.schemas import PitchMap, PitchShot
from footyvision.config import get_settings
from footyvision.db.base import get_session
from footyvision.etl import spatial
from footyvision.ml.features import cached_feature_frame, peer_column

router = APIRouter(tags=["players"])
logger = logging.getLogger(__name__)

_CACHE: dict[str, dict] = {}


def _maps() -> dict:
    if "payload" not in _CACHE:
        payload = spatial.load()
        if payload is None:
            raise HTTPException(
                status_code=503,
                detail="No pitch maps published. Run `footyvision spatial` and deploy "
                "models/spatial/spatial.json.",
            )
        # Indexed by player once, so a request does not scan four thousand seasons.
        by_player: dict[int, list[dict]] = {}
        for season in payload["players"].values():
            by_player.setdefault(int(season["player_id"]), []).append(season)
        _CACHE["payload"] = {"grid": payload["grid"], "by_player": by_player}
    return _CACHE["payload"]


@router.get("/players/{player_id}/pitch", response_model=PitchMap)
def pitch_map(
    player_id: int,
    competition_id: int | None = Query(None),
    season_id: int | None = Query(None),
    session: Session = Depends(get_session),
) -> PitchMap:
    """A player's action grid and shots for one season — by default the busiest one.

    Coordinates are StatsBomb's: x from 0 to 120 towards the opponent's goal, y from 0 to
    80 with the player's own attacking left at y = 0. A client drawing on a pitch with the
    other convention has to flip y, and the dashboard does.
    """
    maps = _maps()
    seasons = maps["by_player"].get(player_id, [])
    if competition_id is not None:
        seasons = [s for s in seasons if s["competition_id"] == competition_id]
    if season_id is not None:
        seasons = [s for s in seasons if s["season_id"] == season_id]
    if not seasons:
        raise HTTPException(status_code=404, detail="No pitch map for this player.")
    # The season with the most recorded actions, which is the one best sampled — and for
    # all but eleven players the only one.
    season = max(seasons, key=lambda s: s["actions"])
    grid = maps["grid"]
    # The map itself needs no database; only the per-90 xT does, for the minutes. A database
    # that cannot answer costs that one figure, not the whole response.
    try:
        xt = _threat(session, season, maps["by_player"])
    except SQLAlchemyError:
        logger.exception("pitch map served without xT per 90: the feature frame failed")
        xt = {}
    return PitchMap(
        player_id=player_id,
        competition_id=season["competition_id"],
        season_id=season["season_id"],
        cols=grid["cols"],
        rows=grid["rows"],
        actions=season["actions"],
        cells=season["grid"],
        shots=[
            PitchShot(x=s[0], y=s[1], xg=s[2], outcome=s[3], penalty=bool(s[4]), body_part=s[5])
            for s in season["shots"]
        ],
        zones={k: round(v, 3) for k, v in spatial.features(season["grid"]).items()},
        xt_added=season.get("xt_added"),
        moves_completed=season.get("moves_completed"),
        **xt,
    )


def _threat(session: Session, season: dict, by_player: dict) -> dict:
    """xT per 90 and its percentile among the player's peers, when they are in the pool.

    Minutes come from the same feature frame the radar ranks against, so a player below the
    minutes floor gets their raw total and no per-90 figure rather than one built on a few
    appearances.
    """
    if season.get("xt_added") is None:
        return {}
    frame = cached_feature_frame(session, min_minutes=get_settings().min_minutes)
    peers = peer_column(frame)
    mine = frame[
        (frame["player_id"] == season["player_id"])
        & (frame["competition_id"] == season["competition_id"])
        & (frame["sb_season_id"] == season["season_id"])
    ]
    if mine.empty or not float(mine["minutes"].iloc[0]):
        return {}
    group = frame[frame[peers] == mine[peers].iloc[0]]
    values = []
    for _, row in group.iterrows():
        for other in by_player.get(int(row["player_id"]), ()):
            if (
                (other["competition_id"], other["season_id"])
                == (
                    int(row["competition_id"]),
                    int(row["sb_season_id"]),
                )
                and other.get("xt_added") is not None
                and row["minutes"]
            ):
                values.append(other["xt_added"] / float(row["minutes"]) * 90)
    per90 = season["xt_added"] / float(mine["minutes"].iloc[0]) * 90
    below = sum(v < per90 for v in values)
    return {
        "xt_per90": round(per90, 4),
        "xt_percentile": round(100 * below / len(values), 1) if values else None,
    }
