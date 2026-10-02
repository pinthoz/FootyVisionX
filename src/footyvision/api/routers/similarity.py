from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from footyvision.api.schemas import (
    RadarMetric,
    RadarResponse,
    SimilarPlayerOut,
    SimilarResponse,
    TargetOut,
)
from footyvision.config import get_settings
from footyvision.db.base import get_session
from footyvision.ml.features import cached_feature_frame
from footyvision.ml.similarity import find_similar, radar_percentiles

router = APIRouter(prefix="/players", tags=["similarity"])


def _resolve_min_minutes(value: float | None) -> float:
    return get_settings().min_minutes if value is None else value


@router.get("/{player_id}/similar", response_model=SimilarResponse)
def similar_players(
    player_id: int,
    top_n: int = Query(10, ge=1, le=50),
    min_minutes: float | None = Query(None, description="Override the season minutes floor."),
    competition_id: int | None = Query(None),
    season_id: int | None = Query(None, description="StatsBomb season_id to scope the pool."),
    session: Session = Depends(get_session),
) -> SimilarResponse:
    frame = cached_feature_frame(
        session,
        min_minutes=_resolve_min_minutes(min_minutes),
        competition_id=competition_id,
        season_id=season_id,
    )
    result = find_similar(frame, player_id, top_n)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Player not found in the feature pool (no season above the minutes floor).",
        )
    target, results = result
    return SimilarResponse(
        target=TargetOut(
            player_id=int(target["player_id"]),
            name=target["name"],
            primary_position=target["primary_position"],
            position_group=target["position_group"],
            minutes=float(target["minutes"]),
        ),
        count=len(results),
        results=[
            SimilarPlayerOut(
                player_id=int(r["player_id"]),
                name=r["name"],
                primary_position=r["primary_position"],
                position_group=r["position_group"],
                competition_id=int(r["competition_id"]),
                sb_season_id=int(r["sb_season_id"]),
                minutes=float(r["minutes"]),
                similarity=round(float(r["similarity"]), 4),
                xg_per90=round(float(r["xg_per90"]), 3),
                progressive_passes_per90=round(float(r["progressive_passes_per90"]), 2),
                tackles_per90=round(float(r["tackles_per90"]), 2),
                dribbles_per90=round(float(r["dribbles_per90"]), 2),
            )
            for _, r in results.iterrows()
        ],
    )


@router.get("/{player_id}/radar", response_model=RadarResponse)
def player_radar(
    player_id: int,
    min_minutes: float | None = Query(None),
    competition_id: int | None = Query(None),
    season_id: int | None = Query(None),
    session: Session = Depends(get_session),
) -> RadarResponse:
    frame = cached_feature_frame(
        session,
        min_minutes=_resolve_min_minutes(min_minutes),
        competition_id=competition_id,
        season_id=season_id,
    )
    result = radar_percentiles(frame, player_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Player not found in the feature pool.")
    target, group, metrics = result
    return RadarResponse(
        player_id=int(target["player_id"]),
        name=target["name"],
        position_group=group,
        minutes=float(target["minutes"]),
        metrics={k: RadarMetric(**v) for k, v in metrics.items()},
    )
