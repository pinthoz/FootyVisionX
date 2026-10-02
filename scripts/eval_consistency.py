"""Is a player's consistency measurable from a season of matches? Measurably, no.

The idea is sound and the data is there: 66,449 per-match rows sit unused while every
model in the project reads a season mean. Two forwards on 0.6 xG per 90 can be a metronome
and an intermittent one, and nothing in FootyVision can tell them apart. This script was
written to fix that and instead establishes that it cannot be fixed with twenty matches.

RESULT: do not ship a consistency feature from this data.

    split-half reliability, Spearman-Brown corrected to full-season length

      passes      0.47      assists    0.18
      goals       0.34      tackles    0.14
      xg          0.29      dribbles   0.11
      pressures   0.28      shots      0.07

      composite over all eight metrics          0.35
      composite of bad-day floors instead       0.31

A measure that disagrees with itself across odd and even matches of the same season is
describing the sample, not the player. At 0.11 a ranking would substantially reshuffle if
you used the other half of the fixtures — and the ranking is what makes this seductive,
because it reads perfectly: Giroud and Ibrahimović steady, squad forwards erratic. It
reads that way on noise too.

Three things were tried and only the last is worth reusing:

  coefficient of variation   Spearman -0.94 against goals per 90. For counts the standard
                             deviation grows with the square root of the mean, so CV falls
                             as volume rises by arithmetic alone. It is the per-90 rate
                             restated, and adding it to a model adds a copy of a column.

  Poisson dispersion         Fixes that for counts, but asserts what the variance should
                             be. xG is continuous, so its median lands at 0.18 with no
                             player above 1; passes are driven by game state, so 99% are
                             above it. The null is wrong, not the players.

  log-variance residual      Fit log variance on log mean across all player-seasons and
                             keep the residual: more variable than others doing the same
                             volume of the same thing. Assumes no distribution, and comes
                             out decorrelated from both the mean (-0.06 to 0.03) and the
                             appearance count. This is the right construction. It is also
                             the one that fails the reliability test, which is the point:
                             the construction was never the problem, the sample size is.

What this implies for the next model is worth carrying forward. Means are well estimated
from twenty matches and variances are not, so a sequence model hoping to learn form
dynamics within a season is reaching for the same signal this found to be mostly absent.
Its baseline -- assume the player continues at his season mean -- is strong for exactly
the reason documented here.

Usage:
    python scripts/eval_consistency.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footyvision.db.base import engine  # noqa: E402

METRICS = ("goals", "xg", "shots", "assists", "passes", "tackles", "dribbles", "pressures")

# A cameo is not a performance: a per-90 rate from eleven minutes is mostly division by a
# small number, and including them measures substitution patterns rather than consistency.
MIN_MATCH_MINUTES = 30
# Sixteen so each half of the split still holds eight matches.
MIN_MATCHES = 16
KEY = ["player_id", "competition_id", "sb_season_id"]


def load_matches() -> pd.DataFrame:
    columns = ", ".join(f"s.{m}" for m in METRICS)
    frame = pd.read_sql(
        text(f"""
            SELECT s.player_id, s.minutes, m.match_date,
                   m.competition_id, m.sb_season_id, {columns}
            FROM player_match_stats s
            JOIN matches m ON m.id = s.match_id
            WHERE s.minutes >= :mm
        """),
        engine,
        params={"mm": MIN_MATCH_MINUTES},
    ).sort_values("match_date")

    counts = frame.groupby(KEY).size()
    keep = set(counts[counts >= MIN_MATCHES].index)
    frame = frame[frame.set_index(KEY).index.isin(keep)].copy()
    frame["nth"] = frame.groupby(KEY).cumcount()
    return frame


def residuals(rows: pd.DataFrame, statistic: str) -> pd.DataFrame:
    """Each metric's log statistic minus the trend it follows against log mean.

    Variance rises with volume — the fitted slope is near 1.1 for most metrics, with an R²
    around 0.9 — so the level of a player's variance says mostly how much he does. The
    residual is what is left once that is removed.
    """
    out: dict[str, pd.Series] = {}
    for metric in METRICS:
        means, values, keys = [], [], []
        for key, group in rows.groupby(KEY):
            rate = group[metric] / group["minutes"] * 90.0
            mean = rate.mean()
            value = rate.var(ddof=1) if statistic == "var" else rate.quantile(0.25)
            if mean > 1e-4 and value > 0:
                means.append(mean)
                values.append(value)
                keys.append(key)
        if len(means) < 50:
            continue
        x, y = np.log(means), np.log(values)
        fit = stats.linregress(x, y)
        out[metric] = pd.Series(
            y - (fit.slope * x + fit.intercept), index=pd.MultiIndex.from_tuples(keys)
        )
    return pd.DataFrame(out)


def corrected(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    """Split-half Spearman, and what it would be at full season length.

    Halving the data deflates the correlation on its own, so the Spearman-Brown step is
    what makes the number comparable to the measure actually being proposed.
    """
    ok = a.notna() & b.notna()
    r = float(stats.spearmanr(a[ok], b[ok]).statistic)
    return r, 2 * r / (1 + r), int(ok.sum())


def main() -> None:
    frame = load_matches()
    print(
        f"{frame.groupby(KEY).ngroups:,} player-seasons with {MIN_MATCHES}+ matches "
        f"of {MIN_MATCH_MINUTES}+ minutes\n"
    )

    # Odd and even appearances rather than first and second half, so both sides span the
    # whole season and neither one is "early form".
    early = frame[frame["nth"] % 2 == 0]
    late = frame[frame["nth"] % 2 == 1]

    for statistic, label in (
        ("var", "erraticness (log variance residual)"),
        ("p25", "bad-day floor (log 25th-percentile residual)"),
    ):
        a, b = residuals(early, statistic), residuals(late, statistic)
        shared = a.index.intersection(b.index)
        print(f"{label}\n")
        print(f"  {'metric':10s} {'r':>7s} {'corrected':>11s} {'n':>7s}")
        for metric in a.columns.intersection(b.columns):
            r, adj, n = corrected(a.loc[shared, metric], b.loc[shared, metric])
            print(f"  {metric:10s} {r:7.2f} {adj:11.2f} {n:7d}")

        # Averaging independent noisy measures buys reliability; these are not independent,
        # which is why it does not.
        za = ((a - a.mean()) / a.std()).loc[shared].mean(axis=1, skipna=True)
        zb = ((b - b.mean()) / b.std()).loc[shared].mean(axis=1, skipna=True)
        r, adj, n = corrected(za, zb)
        print(f"  {'composite':10s} {r:7.2f} {adj:11.2f} {n:7d}\n")

    print("A measure this unreliable describes the fixtures it was computed from.")


if __name__ == "__main__":
    main()
