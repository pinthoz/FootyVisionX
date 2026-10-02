"""Unit tests for event aggregation — no DB or network required."""

from __future__ import annotations

import pandas as pd

from footyvision.etl.aggregate import _progress, aggregate_match


def test_progress_toward_goal_is_positive():
    # Ball moves from midfield toward the goal at (120, 40).
    assert _progress([60, 40], [100, 40]) == 40.0
    # Backward pass yields negative progress.
    assert _progress([100, 40], [60, 40]) == -40.0


def _events() -> pd.DataFrame:
    rows = [
        # Starting XI declaration with one starter (player 10).
        {
            "type": "Starting XI",
            "minute": 0,
            "tactics_lineup": [{"player": {"id": 10, "name": "Starter"}}],
        },
        # A completed progressive pass by the starter at minute 5.
        {
            "type": "Pass",
            "minute": 5,
            "player_id": 10,
            "player": "Starter",
            "team_id": 1,
            "team": "Alpha FC",
            "position": "Center Midfield",
            "location": [50, 40],
            "pass_end_location": [85, 40],
        },
        # A goal (shot with xG) by the starter.
        {
            "type": "Shot",
            "minute": 30,
            "player_id": 10,
            "player": "Starter",
            "team_id": 1,
            "team": "Alpha FC",
            "position": "Center Midfield",
            "shot_outcome": "Goal",
            "shot_statsbomb_xg": 0.45,
        },
        # A substitute (player 20) comes on; first action at minute 70.
        {
            "type": "Tackle-ish duel",
            "minute": 70,
            "player_id": 20,
            "player": "Sub",
            "team_id": 1,
            "team": "Alpha FC",
            "position": "Right Back",
        },
        # End-of-match marker (no player) — real StatsBomb streams always carry one,
        # which is how full-time minute is inferred.
        {"type": "Half End", "minute": 90},
    ]
    return pd.DataFrame(rows)


def test_aggregate_match_basic_counts():
    out = aggregate_match(_events()).set_index("player_id")

    starter = out.loc[10]
    assert starter["goals"] == 1
    assert starter["shots"] == 1
    assert round(starter["xg"], 2) == 0.45
    assert starter["passes"] == 1
    assert starter["passes_completed"] == 1
    assert starter["progressive_passes"] == 1  # 50->85 is >10 toward goal
    assert starter["minutes"] == 90.0  # starter, never subbed off

    sub = out.loc[20]
    assert sub["minutes"] == 20.0  # first action at 70, full time at 90
    assert sub["position"] == "Right Back"
