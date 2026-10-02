from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from footyvision.api.limits import rate_limit
from footyvision.api.schemas import AssistantRequest, AssistantResponse
from footyvision.db.base import get_session
from footyvision.llm.client import LLMError
from footyvision.rag.assistant import ScoutAssistant
from footyvision.rag.service import get_store

router = APIRouter(tags=["assistant"])


@router.post("/assistant", response_model=AssistantResponse, dependencies=[Depends(rate_limit)])
def ask_assistant(
    body: AssistantRequest, session: Session = Depends(get_session)
) -> AssistantResponse:
    """Conversational scouting: retrieve relevant players, then answer grounded in them."""
    try:
        store = get_store(session)
        result = ScoutAssistant(store).answer(body.question, k=body.k)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AssistantResponse(**result)
