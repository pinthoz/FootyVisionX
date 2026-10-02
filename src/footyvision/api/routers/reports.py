from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from footyvision.api.limits import rate_limit
from footyvision.api.schemas import ReportContextResponse, ReportResponse
from footyvision.config import get_settings
from footyvision.db.base import get_session
from footyvision.llm.client import LLMClient, LLMError
from footyvision.llm.scouting import build_report_context, generate_report

router = APIRouter(tags=["reports"])


def _min_minutes(value: float | None) -> float:
    return get_settings().min_minutes if value is None else value


@router.get("/llm/health")
def llm_health() -> dict:
    client = LLMClient()
    return {"reachable": client.health(), "base_url": client.base_url, "model": client.model}


@router.get("/players/{player_id}/report/context", response_model=ReportContextResponse)
def report_context(
    player_id: int,
    min_minutes: float | None = Query(None),
    competition_id: int | None = Query(None),
    season_id: int | None = Query(None),
    session: Session = Depends(get_session),
) -> ReportContextResponse:
    """The factual context that grounds the report — no LLM call. Useful for inspection."""
    context = build_report_context(
        session, player_id, _min_minutes(min_minutes), competition_id, season_id
    )
    if context is None:
        raise HTTPException(status_code=404, detail="Player not found in the feature pool.")
    return ReportContextResponse(player_id=player_id, context=context)


@router.post(
    "/players/{player_id}/report", response_model=ReportResponse, dependencies=[Depends(rate_limit)]
)
def player_report(
    player_id: int,
    min_minutes: float | None = Query(None),
    competition_id: int | None = Query(None),
    season_id: int | None = Query(None),
    session: Session = Depends(get_session),
) -> ReportResponse:
    """Generate an LLM scouting report grounded in the player's computed stats."""
    try:
        result = generate_report(
            session,
            player_id,
            min_minutes=_min_minutes(min_minutes),
            competition_id=competition_id,
            season_id=season_id,
        )
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if result is None:
        raise HTTPException(status_code=404, detail="Player not found in the feature pool.")
    return ReportResponse(
        player_id=player_id,
        name=result["context"]["name"],
        report=result["report"],
        context=result["context"],
    )
