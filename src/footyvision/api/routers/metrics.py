from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from footyvision.api.schemas import DistributionPoint, DistributionResponse
from footyvision.config import get_settings
from footyvision.db.base import get_session
from footyvision.ml.features import PER90_FEATURES, cached_feature_frame

router = APIRouter(tags=["metrics"])


@router.get("/metrics/{metric}/distribution", response_model=DistributionResponse)
def metric_distribution(
    metric: str,
    position_group: str | None = Query(None, description="GK / DEF / MID / FWD"),
    min_minutes: float | None = Query(None),
    session: Session = Depends(get_session),
) -> DistributionResponse:
    """Every player's value for one metric, so a client can draw the distribution.

    The percentile endpoints say where a player ranks; this says what the field he is
    ranked against actually looks like — whether the 96th percentile is out on its own
    or packed in with everyone else.
    """
    if metric not in PER90_FEATURES:
        raise HTTPException(status_code=422, detail=f"Unknown metric: {metric}")

    floor = get_settings().min_minutes if min_minutes is None else min_minutes
    frame = cached_feature_frame(session, min_minutes=floor)
    if position_group:
        frame = frame[frame["position_group"] == position_group]

    return DistributionResponse(
        metric=metric,
        position_group=position_group,
        count=len(frame),
        values=[
            DistributionPoint(
                id=f"{int(r['player_id'])}-{int(r['competition_id'])}-{int(r['sb_season_id'])}",
                player_id=int(r["player_id"]),
                name=r["name"],
                competition=r.get("competition"),
                value=round(float(r[metric]), 3),
            )
            for _, r in frame.iterrows()
        ],
    )
