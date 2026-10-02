"""Performance Score: a transparent, position-aware rating (0-100).

Not a predictive "potential" model (we have no future/age/value labels). It is an
explainable composite of a player's percentiles *within their position group*, weighted
by what matters for the role. Every point is traceable to a metric contribution.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from footyvision.ml.similarity import _target_index

# Per-role metric weights (each role's weights sum to 1.0). Weights encode what the role
# is judged on — a striker on finishing/creation, a defender on stopping play + build-up.
ROLE_WEIGHTS: dict[str, dict[str, float]] = {
    "FWD": {
        "xg_per90": 0.22,
        "goals_per90": 0.18,
        "shots_per90": 0.10,
        "assists_per90": 0.12,
        "dribbles_completed_per90": 0.10,
        "progressive_carries_per90": 0.10,
        "progressive_passes_per90": 0.08,
        "passes_completed_per90": 0.05,
        "ball_recoveries_per90": 0.05,
    },
    "MID": {
        "progressive_passes_per90": 0.18,
        "passes_completed_per90": 0.12,
        "assists_per90": 0.10,
        "xg_per90": 0.08,
        "progressive_carries_per90": 0.10,
        "dribbles_completed_per90": 0.08,
        "ball_recoveries_per90": 0.12,
        "tackles_per90": 0.10,
        "interceptions_per90": 0.12,
    },
    "DEF": {
        "tackles_per90": 0.16,
        "interceptions_per90": 0.16,
        "clearances_per90": 0.14,
        "blocks_per90": 0.10,
        "ball_recoveries_per90": 0.14,
        "progressive_passes_per90": 0.14,
        "passes_completed_per90": 0.10,
        "progressive_carries_per90": 0.06,
    },
    # Outfield-oriented metrics only, so GK scores are weak/indicative — flagged in docs.
    "GK": {
        "passes_completed_per90": 0.35,
        "progressive_passes_per90": 0.30,
        "clearances_per90": 0.20,
        "ball_recoveries_per90": 0.15,
    },
}


def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a `performance_score` column (0-100), computed within each position group."""
    out = frame.copy()
    out["performance_score"] = 0.0
    for group, weights in ROLE_WEIGHTS.items():
        idx = frame.index[frame["position_group"] == group]
        if len(idx) == 0:
            continue
        sub = frame.loc[idx]
        score = pd.Series(0.0, index=idx)
        for feat, weight in weights.items():
            score += weight * (sub[feat].rank(pct=True) * 100.0)
        out.loc[idx, "performance_score"] = score.round(1)
    return out


def performance_score(frame: pd.DataFrame, player_id: int) -> dict[str, Any] | None:
    """Score for one player plus the per-metric contribution breakdown."""
    idx = _target_index(frame, player_id)
    if idx is None:
        return None
    group = frame.loc[idx, "position_group"]
    weights = ROLE_WEIGHTS.get(group, {})
    group_frame = frame[frame["position_group"] == group]

    breakdown: list[dict[str, Any]] = []
    total = 0.0
    for feat, weight in weights.items():
        pct = float(group_frame[feat].rank(pct=True).loc[idx] * 100.0)
        contribution = weight * pct
        total += contribution
        breakdown.append(
            {
                "metric": feat,
                "weight": weight,
                "percentile": round(pct, 1),
                "contribution": round(contribution, 1),
            }
        )
    breakdown.sort(key=lambda b: b["contribution"], reverse=True)
    return {
        "player_id": int(player_id),
        "name": frame.loc[idx, "name"],
        "position_group": group,
        "performance_score": round(total, 1),
        "breakdown": breakdown,
    }


def rank_players(
    frame: pd.DataFrame,
    position_group: str | None = None,
    top_n: int = 20,
    gender: str | None = None,
) -> list[dict[str, Any]]:
    """Top players by performance score, optionally within one position group and/or gender."""
    scored = score_frame(frame)
    if position_group:
        scored = scored[scored["position_group"] == position_group]
    if gender and gender.lower() != "all" and "gender" in scored.columns:
        scored = scored[scored["gender"].astype(str).str.lower() == gender.lower()]
    scored = scored.sort_values("performance_score", ascending=False).head(top_n)
    return [
        {
            "player_id": int(r["player_id"]),
            "name": r["name"],
            "competition": r.get("competition"),
            "gender": r.get("gender"),
            "position_group": r["position_group"],
            "primary_position": r["primary_position"],
            "performance_score": float(r["performance_score"]),
        }
        for _, r in scored.iterrows()
    ]
