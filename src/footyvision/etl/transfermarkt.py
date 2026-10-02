"""Load Transfermarkt market values (Kaggle: davidcariboo/player-scores) for 2015/16.

Two CSVs live in ./data (downloaded once via the Kaggle API):
  - player_valuations.csv : date-stamped historical values (pick the 2015/16 one)
  - players.csv           : names + dates of birth (for the age feature)
This is the real training target for the value predictor.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Transfermarkt competition codes for the four leagues we hold complete 2015/16
# seasons of. Widening the labels from Spain alone took the value model from
# 245/411 matched rows to 1182/1570.
LEAGUE_CODES: tuple[str, ...] = ("ES1", "GB1", "IT1", "FR1")
LA_LIGA_TM = "ES1"  # kept for callers that only want Spain
# A wide window (values are only updated a few times a year, so a tight season window
# misses players); we then keep each player's valuation closest to the season reference.
SEASON_START = "2014-07-01"
SEASON_END = "2017-06-30"
SEASON_REF = pd.Timestamp("2016-01-01")  # mid-season reference for "age" and value pick


def read_market_values_2016(
    data_dir: str | Path = "data", competitions: tuple[str, ...] = LEAGUE_CODES
) -> pd.DataFrame:
    """Return [name, value_eur, age] for players of `competitions` in 2015/16."""
    data_dir = Path(data_dir)
    vals = pd.read_csv(data_dir / "player_valuations.csv", parse_dates=["date"])
    players = pd.read_csv(data_dir / "players.csv")

    window = vals[
        (vals["date"] >= SEASON_START)
        & (vals["date"] <= SEASON_END)
        & (vals["player_club_domestic_competition_id"].isin(competitions))
    ].copy()

    # For each player keep the valuation closest to mid-season.
    window["gap"] = (window["date"] - SEASON_REF).abs()
    picked = window.sort_values("gap").drop_duplicates("player_id")

    meta = players[["player_id", "name", "date_of_birth"]].copy()
    merged = picked.merge(meta, on="player_id", how="left")

    dob = pd.to_datetime(merged["date_of_birth"], errors="coerce")
    merged["age"] = ((SEASON_REF - dob).dt.days / 365.25).round(1)

    out = merged[["name", "market_value_in_eur", "age"]].rename(
        columns={"market_value_in_eur": "value_eur"}
    )
    out = out.dropna(subset=["name"])
    return out.sort_values("value_eur", ascending=False).drop_duplicates("name")


def read_laliga_values_2016(data_dir: str | Path = "data") -> pd.DataFrame:
    """Spain only — the original single-league loader, kept for callers that want it."""
    return read_market_values_2016(data_dir, competitions=(LA_LIGA_TM,))
