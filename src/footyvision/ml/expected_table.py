"""The league table the chances deserved: expected points from every shot's xG.

Each shot is treated as an independent chance of scoring with probability equal to its
StatsBomb xG, so a team's goals in a match follow a Poisson-binomial distribution computed
exactly from its shots. Combining both sides gives the probability of a home win, a draw and
an away win, and so the points each side could expect from the chances it made and allowed
(xPts). The gap to the points actually won is the part of the table the chances do not
explain: finishing, goalkeeping, and luck.

The method is the one Understat popularised, and needs no model fitting: xG is supplied per
shot by StatsBomb, so nothing here is estimated from the results it is compared with.

Known simplifications, both of which bias xPts slightly towards the middle:

  * Shots in the same move are not independent — a rebound follows a save — so a flurry of
    chances counts as more separate chances than it really was.
  * Actual points come from the scoreline summed over players' goals, as everywhere else in
    the project, and own goals belong to no player of the side they count for.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from footyvision.etl import spatial

ARTIFACT = spatial.ARTIFACT.parent / "expected_table.json"


def goal_distribution(xgs: list[float], cap: int = 10) -> list[float]:
    """P(goals = k) for k = 0..cap, for independent shots with the given xG."""
    dist = [1.0] + [0.0] * cap
    for p in xgs:
        p = min(max(float(p), 0.0), 1.0)
        for k in range(cap, 0, -1):
            dist[k] = dist[k] * (1 - p) + dist[k - 1] * p
        dist[0] *= 1 - p
    return dist


def match_probabilities(home_xgs: list[float], away_xgs: list[float]) -> tuple[float, float, float]:
    """(home win, draw, away win) from both sides' shots."""
    home, away = goal_distribution(home_xgs), goal_distribution(away_xgs)
    win = sum(home[i] * away[j] for i in range(len(home)) for j in range(len(away)) if i > j)
    draw = sum(home[i] * away[i] for i in range(min(len(home), len(away))))
    return win, draw, max(0.0, 1.0 - win - draw)


def team_shots(match_id: int) -> dict[int, list[float]] | None:
    """Every shot's xG per team in one match, read from the cached reduction."""
    cached = spatial.CACHE_DIR / f"{match_id}.json"
    if not cached.is_file():
        return None
    stored = json.loads(cached.read_text(encoding="utf-8"))
    if stored.get("v") != spatial.CACHE_VERSION:
        return None
    out: dict[int, list[float]] = {}
    for record in stored["players"].values():
        if record.get("team") is None:
            continue
        out.setdefault(int(record["team"]), []).extend(s[2] for s in record["shots"])
    return out


def build(session: Session) -> dict[str, Any]:
    """Actual and expected points for every team in every loaded competition-season."""
    from footyvision.db.models import Competition, Team
    from footyvision.db.quality import COMPLETE_THRESHOLD, season_facts
    from footyvision.ml.matches import load_matches

    matches = load_matches(session)
    # A table only means something for a whole league. The open data's single-club exports
    # (Bundesliga 2015/16 is Leverkusen's 34 matches and two apiece for everyone else) and
    # any season below the completeness threshold are left out rather than ranked.
    facts = season_facts(session)
    whole = {
        key
        for key, f in facts.items()
        if f.focus_team_id is None and f.coverage >= COMPLETE_THRESHOLD
    }
    names = {t.id: t.name for t in session.query(Team).all()}
    competitions = {c.id: c.name for c in session.query(Competition).all()}
    tables: dict[str, dict[int, dict[str, float]]] = {}
    used = 0
    for m in matches.itertuples(index=False):
        if (int(m.competition_id), int(m.sb_season_id)) not in whole:
            continue
        shots = team_shots(int(m.match_id))
        if shots is None:
            continue
        used += 1
        home, away = int(m.home_team_id), int(m.away_team_id)
        p_home, p_draw, p_away = match_probabilities(shots.get(home, []), shots.get(away, []))
        table = tables.setdefault(f"{int(m.competition_id)}-{int(m.sb_season_id)}", {})
        for team, xpts, scored, conceded, xg_for, xg_against in (
            (
                home,
                3 * p_home + p_draw,
                m.home_goals,
                m.away_goals,
                shots.get(home, []),
                shots.get(away, []),
            ),
            (
                away,
                3 * p_away + p_draw,
                m.away_goals,
                m.home_goals,
                shots.get(away, []),
                shots.get(home, []),
            ),
        ):
            row = table.setdefault(
                team,
                {"played": 0, "points": 0, "xpts": 0.0, "gf": 0, "ga": 0, "xgf": 0.0, "xga": 0.0},
            )
            row["played"] += 1
            row["points"] += 3 if scored > conceded else 1 if scored == conceded else 0
            row["xpts"] += xpts
            row["gf"] += int(scored)
            row["ga"] += int(conceded)
            row["xgf"] += sum(xg_for)
            row["xga"] += sum(xg_against)

    out: dict[str, Any] = {}
    for key, table in tables.items():
        cid, sid = (int(v) for v in key.split("-"))
        rows = [
            {
                "team_id": team,
                "name": names.get(team, str(team)),
                **{k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()},
                "luck": round(row["points"] - row["xpts"], 2),
            }
            for team, row in table.items()
        ]
        rows.sort(key=lambda r: (-r["points"], -(r["gf"] - r["ga"])))
        out[key] = {
            "competition_id": cid,
            "season_id": sid,
            "competition": competitions.get(cid),
            "teams": rows,
        }
    return {"built_on": date.today().isoformat(), "matches": used, "seasons": out}


def save(payload: dict[str, Any], path: Path = ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    return path


def load(path: Path = ARTIFACT) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
