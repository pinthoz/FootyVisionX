"""Fetch FIFA player values from SoFIFA (via soccerdata) as a market-value proxy.

Used as the training target for the value predictor. FIFA 16 (version 160001) aligns with
our StatsBomb La Liga 2015/16 features. SoFIFA scrapes per player, so the first run is slow
but soccerdata caches every page (re-runs are instant and resumable).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import pandas as pd

FIFA16_VERSION = 160001
LA_LIGA = "ESP-La Liga"

warnings.filterwarnings("ignore", message="credentials were not supplied")


def _find_col(columns: list, *keywords: str) -> str | None:
    for col in columns:
        name = str(col).lower()
        if any(k in name for k in keywords):
            return col
    return None


def read_laliga_values(version: int = FIFA16_VERSION) -> pd.DataFrame:
    """Return a DataFrame [name, value_eur, overall, potential, age] for La Liga players."""
    import soccerdata as sd

    sofifa = sd.SoFIFA(leagues=LA_LIGA, versions=version)
    ratings = sofifa.read_player_ratings().reset_index()

    cols = list(ratings.columns)
    name_col = _find_col(cols, "player") or "player"
    value_col = _find_col(cols, "value")
    overall_col = _find_col(cols, "overall")
    potential_col = _find_col(cols, "potential")
    age_col = _find_col(cols, "age")

    out = pd.DataFrame({"name": ratings[name_col].astype(str)})
    out["value_eur"] = ratings[value_col].map(parse_value_eur) if value_col else 0.0
    out["overall"] = pd.to_numeric(ratings[overall_col], errors="coerce") if overall_col else None
    out["potential"] = (
        pd.to_numeric(ratings[potential_col], errors="coerce") if potential_col else None
    )
    out["age"] = pd.to_numeric(ratings[age_col], errors="coerce") if age_col else None
    # One row per player (dedupe if a player appears in multiple teams/updates).
    return out.sort_values("value_eur", ascending=False).drop_duplicates("name")


# Lives here rather than in ml/value.py, where it used to: it is a parser for this
# source's text format, and importing it from the model module made loading SoFIFA data
# require lightgbm, which is a training dependency and absent from the runtime install.
def parse_value_eur(raw: Any) -> float:
    """Parse a SoFIFA value ('€5M', '€900K', 5000000, '') into euros; 0 if unknown."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().replace("€", "").replace(",", "")
    if not s:
        return 0.0
    mult = 1.0
    if s[-1].upper() == "M":
        mult, s = 1_000_000.0, s[:-1]
    elif s[-1].upper() == "K":
        mult, s = 1_000.0, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return 0.0
