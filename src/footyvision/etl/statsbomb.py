"""Thin extraction layer over statsbombpy (StatsBomb Open Data).

No credentials are configured, so statsbombpy reads the free open-data repository.
We silence its noisy "no credentials" warning here.
"""

from __future__ import annotations

import warnings

import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore", message="credentials were not supplied")


def competitions() -> pd.DataFrame:
    """All available competition+season rows in the open dataset."""
    return sb.competitions()


def matches(competition_id: int, season_id: int) -> pd.DataFrame:
    return sb.matches(competition_id=competition_id, season_id=season_id)


def events(match_id: int) -> pd.DataFrame:
    """Flattened event stream for a single match (nested fields use underscores)."""
    return sb.events(match_id=match_id)
