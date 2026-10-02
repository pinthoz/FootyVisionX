from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from footyvision.api.schemas import (
    ModelInfoResponse,
    RankingsResponse,
    RoleModelInfo,
    ScoreResponse,
)
from footyvision.config import get_settings
from footyvision.db.base import get_session
from footyvision.ml import precompute
from footyvision.ml.features import cached_feature_frame
from footyvision.ml.scoring import performance_score, rank_players

# Deliberately no import of footyvision.ml.talent here, and none of sklearn or xgboost
# through it. Everything the models are asked at request time depends only on a player's
# own feature row, so it is computed once by `footyvision precompute` and read back from
# JSON: importing those libraries costs 76MB and loading the three fitted models another
# 175MB, against a 512MB container that was being restarted under load. The training code
# is imported lazily below, and only where the published file is missing or stale.
router = APIRouter(tags=["talent"])
logger = logging.getLogger(__name__)

_PREDICTIONS: dict = {}


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _predictions(frame) -> dict:
    """The published model outputs; refitted only where the training stack is installed.

    A working copy with the `train` extra refits when the file is missing or describes a
    different pool. A deployment has no such extra, and that absence is the safeguard: the
    refit costs 250MB, and on a 512MB instance attempting it is an out-of-memory restart.
    So there a stale file is served and marked stale, and a missing one is a 503 naming
    the command that fixes it.
    """
    if "payload" not in _PREDICTIONS:
        payload = precompute.load()
        stale = precompute.stale_for(payload, frame)
        if stale:
            try:
                payload, stale = precompute.build(frame), False
            except ImportError:
                if payload is None:
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "No published model predictions, and the training libraries "
                            "are not installed here. Run `footyvision precompute` where "
                            "they are and deploy models/talent/predictions.json."
                        ),
                    ) from None
                logger.warning(
                    "predictions.json was built on %s for a pool of %s rows; this one has "
                    "%s. Serving it anyway — run `footyvision precompute` and redeploy.",
                    payload.get("built_on"),
                    payload.get("pool"),
                    len(frame),
                )
        _PREDICTIONS["payload"] = {**payload, "stale": stale}
    return _PREDICTIONS["payload"]


def _frame(session: Session, min_minutes: float | None):
    mm = get_settings().min_minutes if min_minutes is None else min_minutes
    return cached_feature_frame(session, min_minutes=mm)


@router.get("/players/{player_id}/score", response_model=ScoreResponse)
def player_score(
    player_id: int,
    min_minutes: float | None = Query(None),
    session: Session = Depends(get_session),
) -> ScoreResponse:
    """Position-aware Performance Score (0-100) + the model's style profile."""
    frame = _frame(session, min_minutes)
    score = performance_score(frame, player_id)
    if score is None:
        raise HTTPException(status_code=404, detail="Player not found in the feature pool.")
    player = _predictions(frame)["players"].get(str(player_id), {})
    profile = player.get("group", {})
    roles = player.get("role", {})
    best_role, confidence = (None, None)
    if roles:
        best_role, confidence = max(roles.items(), key=lambda kv: kv[1])
    shortlist = player.get("exact_shortlist", [])
    return ScoreResponse(
        **score,
        style_profile=profile,
        predicted_role=best_role,
        role_confidence=confidence,
        role_profile=roles,
        position_shortlist=shortlist,
    )


@router.get("/rankings", response_model=RankingsResponse)
def rankings(
    position_group: str | None = Query(None, description="GK / DEF / MID / FWD"),
    top_n: int = Query(20, ge=1, le=100),
    gender: str | None = Query(None, description="all / female / male"),
    min_minutes: float | None = Query(None),
    session: Session = Depends(get_session),
) -> RankingsResponse:
    """Leaderboard by Performance Score, optionally within one position group and/or gender."""
    frame = _frame(session, min_minutes)
    results = rank_players(frame, position_group, top_n, gender=gender)
    return RankingsResponse(count=len(results), results=results)


@router.get("/talent/model-info", response_model=ModelInfoResponse)
def model_info(
    min_minutes: float | None = Query(None), session: Session = Depends(get_session)
) -> ModelInfoResponse:
    """Evaluation of the position classifiers (honest held-out accuracy, both grains)."""
    payload = _predictions(_frame(session, min_minutes))
    models = payload["models"]
    group = models["position_group"]
    return ModelInfoResponse(
        task="position-group classification",
        classes=group["classes"],
        test_accuracy=group["test_accuracy"],
        balanced_accuracy=group["balanced_accuracy"],
        per_class_recall=group["per_class_recall"],
        n_train=group["n_train"],
        n_test=group["n_test"],
        features=group["features"],
        top_features=payload["top_features"],
        predictions_built_on=payload.get("built_on"),
        predictions_stale=payload.get("stale", False),
        role_model=_role_info(models["position_role"]),
        exact_model=_role_info(models["primary_position"]),
    )


def _role_info(meta: dict) -> RoleModelInfo:
    return RoleModelInfo(
        classes=meta["classes"],
        test_accuracy=meta["test_accuracy"],
        balanced_accuracy=meta["balanced_accuracy"],
        per_class_recall=meta["per_class_recall"],
        n_train=meta["n_train"],
        n_test=meta["n_test"],
        top3_accuracy=meta.get("top3_accuracy"),
    )
