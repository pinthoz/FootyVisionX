"""Does the expected table say more about a team than the real one?

Expected points (xPts) come from the xG of every shot a team took and allowed; actual
points from the results. If xPts is only a noisier restatement of the table, the real
points so far should predict a team's future at least as well. If it strips out finishing
and luck, which do not persist, it should predict better.

For every team-season with at least ten matches, the first half of its matches (by date)
is described four ways — points, expected points, goal difference, and xG difference, all
per match — and each is correlated with the points per match it went on to earn in the
second half. Nothing from the second half is used to build a predictor.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footyvision.db.base import SessionLocal  # noqa: E402
from footyvision.ml import expected_table as et  # noqa: E402
from footyvision.ml.matches import load_matches  # noqa: E402


def main() -> None:
    with SessionLocal() as session:
        matches = load_matches(session)
    per_team: dict[tuple, list[dict]] = {}
    for m in matches.itertuples(index=False):
        shots = et.team_shots(int(m.match_id))
        if shots is None:
            continue
        home, away = int(m.home_team_id), int(m.away_team_id)
        h_xg, a_xg = shots.get(home, []), shots.get(away, [])
        p_home, p_draw, p_away = et.match_probabilities(h_xg, a_xg)
        for team, xpts, gf, ga, xgf, xga in (
            (home, 3 * p_home + p_draw, m.home_goals, m.away_goals, sum(h_xg), sum(a_xg)),
            (away, 3 * p_away + p_draw, m.away_goals, m.home_goals, sum(a_xg), sum(h_xg)),
        ):
            points = 3 if gf > ga else 1 if gf == ga else 0
            per_team.setdefault((m.competition_id, m.sb_season_id, team), []).append(
                {
                    "date": m.match_date,
                    "points": points,
                    "xpts": xpts,
                    "gd": gf - ga,
                    "xgd": xgf - xga,
                }
            )

    predictors = {"points": [], "xpts": [], "gd": [], "xgd": []}
    future: list[float] = []
    for games in per_team.values():
        if len(games) < 10:
            continue  # single-club exports give each opponent two matches
        games.sort(key=lambda g: g["date"])
        half = len(games) // 2
        early, late = games[:half], games[half:]
        for key in predictors:
            predictors[key].append(sum(g[key] for g in early) / len(early))
        future.append(sum(g["points"] for g in late) / len(late))

    print(f"{len(future)} team-seasons: first half of the season against points per match")
    print("in the second half\n")
    labels = {
        "points": "points per match",
        "xpts": "expected points",
        "gd": "goal difference",
        "xgd": "xG difference",
    }
    for key, label in labels.items():
        r, _ = pearsonr(predictors[key], future)
        rho, _ = spearmanr(predictors[key], future)
        print(f"  {label:18} pearson {r:+.3f}   spearman {rho:+.3f}")


if __name__ == "__main__":
    main()
