from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from footyvision.api.limits import rate_limit
from footyvision.api.schemas import NLSearchRequest, SearchResponse
from footyvision.db.base import get_session
from footyvision.llm.client import LLMError
from footyvision.search.nl import NLParseError, parse_nl
from footyvision.search.query import PlayerQuery, execute_query

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse, dependencies=[Depends(rate_limit)])
def natural_language_search(
    body: NLSearchRequest, session: Session = Depends(get_session)
) -> SearchResponse:
    """Free-text player search: the LLM structures it into a validated PlayerQuery."""
    try:
        query = parse_nl(body.query)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except NLParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    rows = execute_query(session, query)
    return SearchResponse(interpreted=query.model_dump(), count=len(rows), results=rows)


@router.post(
    "/search/structured", response_model=SearchResponse, dependencies=[Depends(rate_limit)]
)
def structured_search(
    query: PlayerQuery, session: Session = Depends(get_session)
) -> SearchResponse:
    """Run a validated PlayerQuery directly — no LLM. Powers testing and power users."""
    rows = execute_query(session, query)
    return SearchResponse(interpreted=query.model_dump(), count=len(rows), results=rows)
