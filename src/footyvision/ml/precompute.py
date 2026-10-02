"""Model outputs, computed once and read back without the model.

The API used to load three XGBoost classifiers to answer questions whose answers never
change between imports: the four-group style profile, the ten-role profile, and the
three-name exact-position shortlist all depend only on a player's own feature row.

Doing that in-process is what pushed the deployment over its memory limit. Measured:

    fastapi + pandas + sqlalchemy + numpy + the feature frame     155 MB
    ... plus importing sklearn, xgboost and lightgbm              231 MB
    ... plus loading the three fitted models                      405 MB

Against a 512MB container that left 7MB of headroom at peak, and the instance was being
restarted. Reading a JSON file instead costs nothing and removes the whole 250MB, so the
service fits on the plan it is on.

The trade is the usual one for a cache of a static thing: this file is stale the moment a
new season is imported, and it is `footyvision precompute` that makes it current again.
`stale_for` reports the mismatch rather than letting a wrong answer be served quietly.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ARTIFACT = Path(__file__).resolve().parents[3] / "models" / "talent" / "predictions.json"

# The three targets, and the key each is published under.
TARGETS = ("position_group", "position_role", "primary_position")


def build(frame: pd.DataFrame, top_n_shortlist: int = 3) -> dict[str, Any]:
    """Fit the three classifiers and record what they say about every player.

    Imports the training code locally: this runs from the CLI on a machine that has the
    data, never inside the API, which is the entire point of the file it writes.
    """
    from footyvision.etl.spatial import attach_features
    from footyvision.ml.talent import (
        get_cached_exact_model,
        get_cached_importance,
        get_cached_model,
        get_cached_role_model,
    )

    # Where each player acts, from the published pitch maps. The largest single gain the
    # classifiers have had: the exact position goes from 47% to 70%, because the per-90
    # counts carry neither how high a player plays nor on which side.
    frame = attach_features(frame)

    group = get_cached_model(frame)
    role = get_cached_role_model(frame)
    exact = get_cached_exact_model(frame)

    # One row per player: the season they played most of, which is the row `style_profile`
    # picks. Scored in a single batch per model rather than a call per player — the answer
    # is identical and it is three predictions instead of eight thousand.
    minutes = frame["minutes"].astype(float)
    rows = frame.loc[minutes.groupby(frame["player_id"]).idxmax()]
    ids = [str(int(pid)) for pid in rows["player_id"]]

    def probabilities(model) -> list[dict[str, float]]:
        x = rows[model.features].to_numpy(dtype=float)
        return [
            {cls: round(float(p), 3) for cls, p in zip(model.classes, row, strict=True)}
            for row in model.predict_proba(x)
        ]

    group_probs = probabilities(group)
    role_probs = probabilities(role)
    exact_probs = probabilities(exact)

    players: dict[str, Any] = {}
    for i, pid in enumerate(ids):
        # Only the shortlist is kept from the exact model. Its single best guess is not
        # published — right 71% of the time, against 88% for the three together — so the
        # other eighteen probabilities have no reader and would triple this file.
        shortlist = [name for name, _ in sorted(exact_probs[i].items(), key=lambda kv: -kv[1])][
            :top_n_shortlist
        ]
        players[pid] = {
            "group": group_probs[i],
            "role": role_probs[i],
            "exact_shortlist": shortlist,
        }

    return {
        "built_on": date.today().isoformat(),
        # What the models were fitted against. A pool of a different size means a season
        # was loaded and these answers describe a database that no longer exists.
        "pool": int(len(frame)),
        "top_features": get_cached_importance(group, frame),
        "models": {
            name: _model_meta(model)
            for name, model in zip(TARGETS, (group, role, exact), strict=True)
        },
        "players": players,
    }


def build_team_strengths(session) -> dict[str, Any]:
    """Poisson attack/defence per team, for the whole database and for each competition.

    Fitted here for the same reason the classifiers are: `sklearn.linear_model` is the last
    thing dragging scikit-learn into the API, and the ratings only move when a season is
    imported. Each competition is fitted separately because that is what the endpoint does
    when it is asked for one — a rating is relative to the field it was measured in, so
    filtering a global fit afterwards would answer a different question.
    """
    from footyvision.ml.matches import load_matches, team_strengths

    matches = load_matches(session)
    out: dict[str, Any] = {}
    scopes: list[int | None] = [None, *sorted({int(c) for c in matches["competition_id"]})]
    for scope in scopes:
        subset = matches if scope is None else matches[matches["competition_id"] == scope]
        if subset.empty:
            continue
        info: dict[str, float] = {}
        ratings = team_strengths(subset, info)
        out[str(scope)] = {
            "home_advantage": round(info.get("home_advantage", 0.0), 4),
            "matches": int(len(subset)),
            "teams": [
                {
                    "team_id": r.team_id,
                    "attack": round(r.attack, 4),
                    "defence": round(r.defence, 4),
                    "matches": r.matches,
                }
                for r in ratings
            ],
        }
    # Which competition each side actually played in, read off the fixtures: the teams
    # table carries no competition of its own.
    played_in = {
        int(row.home_team_id): int(row.competition_id) for row in matches.itertuples(index=False)
    } | {int(row.away_team_id): int(row.competition_id) for row in matches.itertuples(index=False)}
    out["played_in"] = {str(k): v for k, v in played_in.items()}
    # A plain row count, taken the same way on both sides, so the check does not depend on
    # what `load_matches` chooses to include. Any difference means these ratings were
    # fitted against a different database from the one being asked.
    out["db_matches"] = count_matches(session)
    return out


def count_matches(session) -> int:
    """How many fixtures the database holds, for the staleness check above."""
    from sqlalchemy import func, select

    from footyvision.db.models import Match

    return int(session.execute(select(func.count()).select_from(Match)).scalar_one())


def _model_meta(model) -> dict[str, Any]:
    """Everything /talent/model-info reports, so it too can answer without the model."""
    return {
        "classes": model.classes,
        "features": model.features,
        "test_accuracy": round(model.test_accuracy, 3),
        "balanced_accuracy": (
            round(model.balanced_accuracy, 3) if model.balanced_accuracy is not None else None
        ),
        "per_class_recall": model.per_class_recall,
        "n_train": model.n_train,
        "n_test": model.n_test,
        "top3_accuracy": (
            round(model.top3_accuracy, 3) if model.top3_accuracy is not None else None
        ),
    }


def save(payload: dict[str, Any], path: Path = ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def load(path: Path = ARTIFACT) -> dict[str, Any] | None:
    """The published predictions, or None where there are none to read."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def stale_for(payload: dict[str, Any] | None, frame: pd.DataFrame) -> bool:
    """Whether these predictions describe a different pool from the one in front of them.

    The same check the model cache makes, for the same reason: loading a competition leaves
    the columns identical while changing every player the models were fitted against, and
    a stale answer has no visible symptom.
    """
    if not payload:
        return True
    pool = payload.get("pool")
    return not isinstance(pool, int) or abs(pool - len(frame)) > max(5, 0.02 * len(frame))
