"""Aggregate a StatsBomb event stream into per-player-per-match statistics.

The hard parts here are (1) estimating minutes played and (2) the "progressive"
heuristics. Both are documented inline; they are deliberately simple and explainable
rather than perfectly matching any vendor's proprietary definition.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# StatsBomb pitch is 120x80 with the attacking goal centred at (120, 40).
_GOAL_X, _GOAL_Y = 120.0, 40.0

# A pass/carry is "progressive" if it reduces distance-to-goal by at least this many
# pitch units. Passes travel further than carries, hence different thresholds.
_PROG_PASS_THRESHOLD = 10.0
_PROG_CARRY_THRESHOLD = 5.0


def _col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    """Return a column if present, otherwise a same-indexed series of `default`."""
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def _dist_to_goal(point) -> float:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return math.nan
    return math.hypot(_GOAL_X - point[0], _GOAL_Y - point[1])


def _progress(start, end) -> float:
    """Reduction in distance-to-goal (positive = ball moved forward toward goal)."""
    d0, d1 = _dist_to_goal(start), _dist_to_goal(end)
    if math.isnan(d0) or math.isnan(d1):
        return math.nan
    return d0 - d1


def _minutes_and_positions(events: pd.DataFrame) -> dict[int, dict]:
    """Estimate minutes played and the match position for every player in the match.

    Starters begin at minute 0; substitutes begin at their first recorded action.
    A player's end minute is when they were subbed off / sent off, else full time.
    """
    match_end = int(np.nanmax(_col(events, "minute").to_numpy())) if len(events) else 90

    starters: set[int] = set()
    for _, row in events[_col(events, "type") == "Starting XI"].iterrows():
        for entry in row.get("tactics_lineup") or []:
            pid = entry.get("player", {}).get("id")
            if pid is not None:
                starters.add(int(pid))

    # Players going off: substitutions, red cards, second yellows.
    off_minute: dict[int, int] = {}
    subs = events[_col(events, "type") == "Substitution"]
    for _, row in subs.iterrows():
        pid = row.get("player_id")
        if pid is not None:
            off_minute[int(pid)] = int(row.get("minute", match_end))

    bad = events[_col(events, "type") == "Bad Behaviour"]
    if "bad_behaviour_card" in bad.columns:
        for _, row in bad[bad["bad_behaviour_card"] == "Red Card"].iterrows():
            pid = row.get("player_id")
            if pid is not None:
                off_minute[int(pid)] = int(row.get("minute", match_end))

    out: dict[int, dict] = {}
    player_ids = _col(events, "player_id").dropna().unique()
    for raw_pid in player_ids:
        pid = int(raw_pid)
        pe = events[events["player_id"] == pid]
        start = 0 if pid in starters else int(np.nanmin(_col(pe, "minute").to_numpy()))
        end = off_minute.get(pid, match_end)
        minutes = max(0.0, float(end - start))

        position = None
        pos_series = _col(pe, "position").dropna()
        if len(pos_series):
            position = pos_series.mode().iloc[0]

        out[pid] = {"minutes": minutes, "position": position}
    return out


def aggregate_match(events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per player with counting stats, minutes and position."""
    if events.empty:
        return pd.DataFrame()

    ev = events.copy()
    etype = _col(ev, "type")

    ev["is_shot"] = etype == "Shot"
    ev["is_goal"] = ev["is_shot"] & (_col(ev, "shot_outcome") == "Goal")
    ev["is_assist"] = _col(ev, "pass_goal_assist").fillna(False).astype(bool)
    ev["xg"] = _col(ev, "shot_statsbomb_xg").fillna(0.0).where(ev["is_shot"], 0.0)

    ev["is_pass"] = etype == "Pass"
    ev["is_pass_complete"] = ev["is_pass"] & _col(ev, "pass_outcome").isna()

    ev["is_dribble"] = etype == "Dribble"
    ev["is_dribble_complete"] = ev["is_dribble"] & (_col(ev, "dribble_outcome") == "Complete")

    ev["is_carry"] = etype == "Carry"
    ev["is_tackle"] = (etype == "Duel") & (_col(ev, "duel_type") == "Tackle")
    ev["is_interception"] = etype == "Interception"
    ev["is_block"] = etype == "Block"
    ev["is_clearance"] = etype == "Clearance"
    ev["is_recovery"] = etype == "Ball Recovery"
    ev["is_pressure"] = etype == "Pressure"

    # Progressive passes / carries from the location vectors.
    loc = _col(ev, "location")
    pass_end = _col(ev, "pass_end_location")
    carry_end = _col(ev, "carry_end_location")
    pass_prog = [_progress(a, b) for a, b in zip(loc, pass_end, strict=False)]
    carry_prog = [_progress(a, b) for a, b in zip(loc, carry_end, strict=False)]
    ev["is_prog_pass"] = ev["is_pass_complete"] & (
        pd.Series(pass_prog, index=ev.index) >= _PROG_PASS_THRESHOLD
    )
    ev["is_prog_carry"] = ev["is_carry"] & (
        pd.Series(carry_prog, index=ev.index) >= _PROG_CARRY_THRESHOLD
    )

    ev = ev[_col(ev, "player_id").notna()]
    grouped = ev.groupby("player_id")

    stats = pd.DataFrame(
        {
            "team_id": grouped["team_id"].first(),
            "team": grouped["team"].first() if "team" in ev.columns else None,
            "player": grouped["player"].first(),
            "goals": grouped["is_goal"].sum(),
            "assists": grouped["is_assist"].sum(),
            "shots": grouped["is_shot"].sum(),
            "xg": grouped["xg"].sum(),
            "passes": grouped["is_pass"].sum(),
            "passes_completed": grouped["is_pass_complete"].sum(),
            "progressive_passes": grouped["is_prog_pass"].sum(),
            "dribbles": grouped["is_dribble"].sum(),
            "dribbles_completed": grouped["is_dribble_complete"].sum(),
            "carries": grouped["is_carry"].sum(),
            "progressive_carries": grouped["is_prog_carry"].sum(),
            "tackles": grouped["is_tackle"].sum(),
            "interceptions": grouped["is_interception"].sum(),
            "blocks": grouped["is_block"].sum(),
            "clearances": grouped["is_clearance"].sum(),
            "ball_recoveries": grouped["is_recovery"].sum(),
            "pressures": grouped["is_pressure"].sum(),
        }
    )

    mins = _minutes_and_positions(events)
    stats["minutes"] = [mins.get(int(pid), {}).get("minutes", 0.0) for pid in stats.index]
    stats["position"] = [mins.get(int(pid), {}).get("position") for pid in stats.index]

    return stats.reset_index().rename(columns={"player_id": "player_id"})
