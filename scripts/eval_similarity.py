"""Score the similarity engine against the data itself — no manual labels, no LLM judge.

Every other model in this project reports an honest number: the classifier its held-out
accuracy, the value model its R², the retriever its recall. The similarity engine — the
thing the product is actually for — reported nothing, so "players like X" could neither
be defended nor challenged.

There is no ground truth for "similar", so the checks here are proxies that follow from
what the engine claims to do. It says it matches *playing style within a position group*,
so:

  role@k        share of the neighbours in the same side-agnostic role. The engine already
                restricts the pool to the position *group*, so this is the finer cut it
                does not enforce — a low score would mean the vector is capturing
                something other than how a player plays.
  profile r     mean Spearman correlation between the target's percentile profile and each
                neighbour's. This is the direct test: similar players should rank alike
                across the 17 metrics.
  cross-league  share of neighbours from a different competition. Not a quality score —
                a diagnostic. Style should not stop at a border, and a low number would
                say the vectors encode league effects rather than the player.

Two things deliberately not reported: agreement on the position *group*, which is 1.0 by
construction because the pool is filtered to it, and whether a player is his own top match,
which cannot happen because he is excluded. Printing either as a result would be theatre.

Usage:
    python scripts/eval_similarity.py --k 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footyvision.config import get_settings  # noqa: E402
from footyvision.db.base import SessionLocal  # noqa: E402
from footyvision.ml.features import PER90_FEATURES, load_feature_frame  # noqa: E402
from footyvision.ml.similarity import find_similar  # noqa: E402


def percentile_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Each player's percentile on every metric, within his own position group."""
    out = pd.DataFrame(index=frame.index, columns=list(PER90_FEATURES), dtype=float)
    for _, sub in frame.groupby("position_group"):
        out.loc[sub.index] = (sub[list(PER90_FEATURES)].rank(pct=True) * 100.0).to_numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sample", type=int, default=300, help="Players to score (0 = all).")
    parser.add_argument("--min-minutes", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    minutes = settings.min_minutes if args.min_minutes is None else args.min_minutes
    with SessionLocal() as session:
        frame = load_feature_frame(session, minutes)

    profiles = percentile_profiles(frame)
    by_player = frame.set_index("player_id")
    profile_by_player = profiles.set_index(frame["player_id"])

    players = list(frame["player_id"].unique())
    if args.sample and len(players) > args.sample:
        players = list(np.random.default_rng(42).choice(players, args.sample, replace=False))

    role_hits: list[float] = []
    correlations: list[float] = []
    cross_league: list[float] = []
    scored = 0

    for player_id in players:
        result = find_similar(frame, int(player_id), args.k)
        if result is None:
            continue
        _, neighbours = result
        if neighbours.empty:
            continue

        target = by_player.loc[int(player_id)]
        if isinstance(target, pd.DataFrame):
            target = target.iloc[0]
        target_profile = profile_by_player.loc[int(player_id)]
        if isinstance(target_profile, pd.DataFrame):
            target_profile = target_profile.iloc[0]

        ids = [int(pid) for pid in neighbours["player_id"]]

        roles, leagues, rhos = [], [], []
        for neighbour_id in ids:
            if neighbour_id == int(player_id):
                continue
            row = by_player.loc[neighbour_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            other = profile_by_player.loc[neighbour_id]
            if isinstance(other, pd.DataFrame):
                other = other.iloc[0]

            roles.append(row["position_role"] == target["position_role"])
            leagues.append(row["competition"] != target["competition"])
            rho = spearmanr(target_profile.to_numpy(float), other.to_numpy(float)).statistic
            if np.isfinite(rho):
                rhos.append(rho)

        if not roles:
            continue
        role_hits.append(float(np.mean(roles)))
        cross_league.append(float(np.mean(leagues)))
        if rhos:
            correlations.append(float(np.mean(rhos)))
        scored += 1

    print(f"pool {len(frame)} player-seasons, scored {scored} of them, k={args.k}\n")
    print(f"  role@{args.k}       {np.mean(role_hits):.3f}   same side-agnostic role")
    print(f"  profile r      {np.mean(correlations):+.3f}   percentile-profile correlation")
    print(f"  cross-league   {np.mean(cross_league):.3f}   neighbours from another competition")
    print(
        "\nProfile r is the direct measure. role@k is a sanity check on a distinction the\n"
        "engine never enforces, and cross-league is a diagnostic rather than a score."
    )


if __name__ == "__main__":
    main()
