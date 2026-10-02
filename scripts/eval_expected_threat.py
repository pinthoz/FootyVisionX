"""Is Expected Threat worth anything here, or is it a prettier count of passes?

xT values each completed pass and carry by how much it raises the chance that possession
ends in a goal. That is a claim about goals, so it is tested against goals, and against the
simpler things it has to beat:

  completed moves   the volume a pass count already measures
  non-penalty xG    the gold standard for chance quality, which xT should not beat on
                    the same match (xG is measured at the shot, one step from the goal)

Two questions, both at team level where goals are frequent enough to measure:

  1. Same match. Does the threat a team generates track the goals it scores? The threat
     grid is fitted on even-numbered matches and applied to odd ones, so no match is
     scored by a model that learned from it.
  2. Next matches. Does a team's average over the first half of a season predict its goals
     per match over the second half? This is the question that matters for a metric
     meant to describe a style: whether it measures something that persists or only
     records what happened.

Reads the per-match reductions `footyvision spatial` caches under data/spatial_cache/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footyvision.db.base import SessionLocal  # noqa: E402
from footyvision.db.models import Match  # noqa: E402
from footyvision.etl import spatial  # noqa: E402


def load_matches() -> dict[int, dict]:
    with SessionLocal() as session:
        rows = session.execute(
            select(Match.id, Match.competition_id, Match.sb_season_id, Match.match_date)
        ).all()
    out = {}
    for mid, cid, sid, when in rows:
        cached = spatial.CACHE_DIR / f"{mid}.json"
        if not cached.is_file():
            continue
        stored = json.loads(cached.read_text(encoding="utf-8"))
        if stored.get("v") != spatial.CACHE_VERSION:
            continue
        out[int(mid)] = {"season": (cid, sid), "date": when, "players": stored["players"]}
    return out


def team_rows(match: dict, threat: list[float]) -> dict[int, dict[str, float]]:
    teams: dict[int, dict[str, float]] = {}
    for record in match["players"].values():
        team = teams.setdefault(
            record["team"], {"xt": 0.0, "moves": 0.0, "shots": 0.0, "xg": 0.0, "goals": 0.0}
        )
        team["xt"] += spatial.threat_added(record["moves"], threat)
        team["moves"] += sum(record["moves"].values())
        for _x, _y, xg, outcome, penalty, _body in record["shots"]:
            if penalty:
                continue
            team["shots"] += 1
            team["xg"] += xg
            team["goals"] += outcome == "goal"
    return teams


def correlate(label: str, x: list[float], y: list[float]) -> None:
    r, _ = pearsonr(x, y)
    rho, _ = spearmanr(x, y)
    print(f"  {label:22} pearson {r:+.3f}   spearman {rho:+.3f}")


def main() -> None:
    matches = load_matches()
    if not matches:
        sys.exit(f"No v{spatial.CACHE_VERSION} reductions cached: run `footyvision spatial`.")
    fit_on = [m for mid, m in matches.items() if mid % 2 == 0]
    held_out = {mid: m for mid, m in matches.items() if mid % 2 == 1}
    threat = spatial.expected_threat(r for m in fit_on for r in m["players"].values())
    print(
        f"{len(matches)} matches; threat grid fitted on {len(fit_on)}, tested on {len(held_out)}\n"
    )

    rows = [row for m in held_out.values() for row in team_rows(m, threat).values()]
    goals = [r["goals"] for r in rows]
    print(f"1. Same match: {len(rows)} team-matches, against non-penalty goals scored")
    for key, label in (
        ("xt", "xT added"),
        ("moves", "completed moves"),
        ("shots", "shots"),
        ("xg", "non-penalty xG"),
    ):
        correlate(label, [r[key] for r in rows], goals)
    # Does xT carry anything xG does not, once a shot has happened?
    xg, xt = np.array([r["xg"] for r in rows]), np.array([r["xt"] for r in rows])
    g = np.array(goals)
    base = np.linalg.lstsq(np.c_[np.ones_like(xg), xg], g, rcond=None)
    both = np.linalg.lstsq(np.c_[np.ones_like(xg), xg, xt], g, rcond=None)
    tss = float(((g - g.mean()) ** 2).sum())
    r2 = lambda fit: 1 - float(fit[1][0]) / tss  # noqa: E731
    print(f"  goals ~ xG: R² {r2(base):.3f}   goals ~ xG + xT: R² {r2(both):.3f}\n")

    # 2. First half of each team-season predicts the second half. The grid is fitted on
    # every match here: the target is goals in *other* matches, so fitting on the
    # predictors' matches does not leak the outcome.
    threat_all = spatial.expected_threat(r for m in matches.values() for r in m["players"].values())
    by_team_season: dict[tuple, list[tuple]] = {}
    for m in matches.values():
        for team, row in team_rows(m, threat_all).items():
            by_team_season.setdefault((m["season"], team), []).append((m["date"], row))
    first, second = {k: [] for k in ("xt", "moves", "shots", "xg", "goals")}, []
    for games in by_team_season.values():
        if len(games) < 10:
            continue  # the single-club exports give their opponents two games each
        games.sort(key=lambda g: (g[0] is None, g[0]))
        half = len(games) // 2
        early, late = [g[1] for g in games[:half]], [g[1] for g in games[half:]]
        for key in first:
            first[key].append(sum(r[key] for r in early) / len(early))
        second.append(sum(r["goals"] for r in late) / len(late))
    print(f"2. Next matches: {len(second)} team-seasons, first-half average against")
    print("   second-half non-penalty goals per match")
    for key, label in (
        ("xt", "xT added"),
        ("moves", "completed moves"),
        ("shots", "shots"),
        ("xg", "non-penalty xG"),
        ("goals", "goals (so far)"),
    ):
        correlate(label, first[key], second)


if __name__ == "__main__":
    main()
