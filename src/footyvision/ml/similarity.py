"""Player similarity: z-score within position group, then cosine similarity.

Pure functions over a feature DataFrame (see features.load_feature_frame) so the maths
is unit-testable without a database. UMAP-style 2D projection is intentionally NOT used
for the similarity computation — only (later) for visualisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footyvision.ml.features import PER90_FEATURES, peer_column


def standardize_within_groups(
    frame: pd.DataFrame, features: list[str], group_col: str = "position_group"
) -> pd.DataFrame:
    """Return a z-scored copy of `features`, standardised within each position group.

    Zero-variance features (a stat identical across the group) get std=1 to avoid
    division by zero; their z-scores become 0 and they simply don't discriminate.
    """
    z = frame[features].astype(float).copy()
    for _, idx in frame.groupby(group_col).groups.items():
        sub = frame.loc[idx, features].astype(float)
        std = sub.std(ddof=0).replace(0.0, 1.0)
        z.loc[idx, features] = (sub - sub.mean()) / std
    return z


def _cosine(target: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of `target` (d,) against each row of `matrix` (n, d)."""
    tnorm = np.linalg.norm(target)
    mnorm = np.linalg.norm(matrix, axis=1)
    denom = tnorm * mnorm
    with np.errstate(invalid="ignore", divide="ignore"):
        sims = (matrix @ target) / denom
    return np.nan_to_num(sims, nan=0.0)


def _target_index(frame: pd.DataFrame, player_id: int) -> int | None:
    """Index of the player's most-played season (a player may have several)."""
    rows = frame[frame["player_id"] == player_id]
    if rows.empty:
        return None
    return int(rows["minutes"].astype(float).idxmax())


def find_similar(
    frame: pd.DataFrame,
    player_id: int,
    top_n: int = 10,
    features: tuple[str, ...] = PER90_FEATURES,
) -> tuple[pd.Series, pd.DataFrame] | None:
    """Most similar players to `player_id`, restricted to the same position group.

    Returns (target_row, results_df) where results_df has one row per player with a
    `similarity` column in [-1, 1], sorted descending. Returns None if the player is
    absent from the frame.
    """
    feats = list(features)
    idx = _target_index(frame, player_id)
    if idx is None:
        return None

    peer = peer_column(frame)
    z = standardize_within_groups(frame, feats, group_col=peer)
    target = frame.loc[idx]
    group = target[peer]

    pool_mask = (frame[peer] == group) & (frame["player_id"] != player_id)
    pool_idx = frame.index[pool_mask]
    if len(pool_idx) == 0:
        return target, frame.loc[[]].assign(similarity=[])

    tvec = z.loc[idx, feats].to_numpy(dtype=float)
    matrix = z.loc[pool_idx, feats].to_numpy(dtype=float)
    sims = _cosine(tvec, matrix)

    results = frame.loc[pool_idx].copy()
    results["similarity"] = sims
    results = results.sort_values("similarity", ascending=False).drop_duplicates("player_id")
    return target, results.head(top_n)


def radar_percentiles(
    frame: pd.DataFrame,
    player_id: int,
    features: tuple[str, ...] = PER90_FEATURES,
) -> tuple[pd.Series, str, dict[str, dict[str, float]]] | None:
    """Percentile rank (0-100) of each feature for `player_id` within its position group.

    This is the standard scouting-radar representation: a value near 100 means the
    player is elite in that metric *relative to peers in the same role*.
    """
    idx = _target_index(frame, player_id)
    if idx is None:
        return None

    peer = peer_column(frame)
    target = frame.loc[idx]
    # Ranked inside the peer group, reported as the position group: the peer key is an
    # internal partition ("F:DEF"), and letting it reach the API would change a documented
    # field into something no caller can interpret.
    group_frame = frame[frame[peer] == target[peer]]
    group = target["position_group"]

    out: dict[str, dict[str, float]] = {}
    for f in features:
        pct = group_frame[f].rank(pct=True).loc[idx] * 100.0
        out[f] = {"value": float(target[f]), "percentile": round(float(pct), 1)}
    return target, group, out
