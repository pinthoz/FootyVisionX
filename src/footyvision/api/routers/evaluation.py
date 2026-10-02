"""What the assistant scored, and what those scores are worth.

Serving this at all is a decision worth stating: a retrieval-augmented answer looks
equally confident whether or not it is grounded, so a product that shows one owes the
reader some evidence about the second. These are the numbers, dated, with the model that
produced them and the two places a bare average misleads.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

from footyvision.api.schemas import RagasEvaluation

router = APIRouter(tags=["evaluation"])

SNAPSHOT = Path(__file__).resolve().parents[2] / "eval" / "ragas.json"


@lru_cache(maxsize=1)
def _snapshot() -> dict | None:
    if not SNAPSHOT.exists():
        return None
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


@router.get("/eval/assistant", response_model=RagasEvaluation)
def assistant_evaluation() -> RagasEvaluation:
    """The most recent RAGAS run over the scouting assistant."""
    payload = _snapshot()
    if payload is None:
        # A checkout that has never run the evaluation has nothing to claim, and saying so
        # is better than serving zeros that read as a failing score.
        raise HTTPException(status_code=404, detail="No evaluation has been published.")
    return RagasEvaluation(**payload)
