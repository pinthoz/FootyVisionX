"""Match outcomes and team strength, from the per-match rows nothing else reads.

FootyVision is a player tool built on season aggregates, and 66,449 per-match rows sit
under it carrying a dimension the rest of the project has none of: teams. A player's per-90
numbers are shaped by the side he plays for — a striker in a weak attack sees fewer chances
than one in a strong one — and until now nothing here could say which was which.

Two models, and the second is arguably the more useful:

  outcome     Home win, draw or away win, from each side's form in *earlier* matches. It
              beats predicting a home win every time by 8.6 points of accuracy, which is
              the only comparison that means anything: home advantage alone gets 46.7%.

  strength    A Poisson attack/defence coefficient per team, the classical football model.
              Goals are Poisson, every team gets a rate for scoring and one for conceding,
              and a scoreline distribution follows from two numbers. It predicts slightly
              worse than the form model but yields something the form model cannot: a
              rating per team, which is what puts a player's numbers in context.

Leakage is the whole difficulty here and worth being explicit about. A match's own
statistics are its result — using them to predict it recovers the scoreline exactly. Every
feature is therefore built from matches played strictly earlier in the same season, and a
side needs `MIN_PRIOR_MATCHES` of them before a fixture is usable at all. Evaluation splits
on time rather than at random, because a random fold trains on matches played after the
ones it predicts, which nobody can do on a Saturday morning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from sqlalchemy.orm import Session

# Home win, draw, away win. Ordered so index 0 is the baseline everyone has to beat.
OUTCOMES = ("home", "draw", "away")

# Before this many earlier matches a side's averages are noise, and a league table after
# two fixtures says more about the fixture list than about the teams.
MIN_PRIOR_MATCHES = 4

# Levels *and* differences. A contest is about the gap, and making a linear model rebuild
# the subtraction from two noisy columns wastes the only thing it is good at.
FEATURES = (
    "d_pts",
    "d_gf",
    "d_ga",
    "d_xgf",
    "d_xga",
    "d_sf",
    "d_sa",
    "d_xg_net",
    "h_pts",
    "a_pts",
    "h_xgf",
    "a_xgf",
    "h_xga",
    "a_xga",
)

_RUNNING = ("pts", "gf", "ga", "xgf", "xga", "sf", "sa")

_MATCH_SQL = """
    SELECT m.id AS match_id, m.competition_id, m.sb_season_id, m.match_date,
           m.home_team_id, m.away_team_id,
           SUM(CASE WHEN s.team_id = m.home_team_id THEN s.goals ELSE 0 END) AS home_goals,
           SUM(CASE WHEN s.team_id = m.away_team_id THEN s.goals ELSE 0 END) AS away_goals,
           SUM(CASE WHEN s.team_id = m.home_team_id THEN s.xg ELSE 0 END) AS home_xg,
           SUM(CASE WHEN s.team_id = m.away_team_id THEN s.xg ELSE 0 END) AS away_xg,
           SUM(CASE WHEN s.team_id = m.home_team_id THEN s.shots ELSE 0 END) AS home_shots,
           SUM(CASE WHEN s.team_id = m.away_team_id THEN s.shots ELSE 0 END) AS away_shots
    FROM matches m
    JOIN player_match_stats s ON s.match_id = m.id
    WHERE m.home_team_id IS NOT NULL AND m.away_team_id IS NOT NULL
    GROUP BY 1, 2, 3, 4, 5, 6
"""


def load_matches(session: Session) -> pd.DataFrame:
    """One row per match with a scoreline derived from its players' goals.

    `matches` records who played whom and when but not the score, so it is summed from the
    per-match rows. Matches with a side missing are dropped rather than guessed at — twenty
    Ligue 1 fixtures have no away team recorded.
    """
    frame = pd.read_sql(text(_MATCH_SQL), session.get_bind())
    frame["match_date"] = pd.to_datetime(frame["match_date"])
    return frame.sort_values(["competition_id", "sb_season_id", "match_date", "match_id"])


def build_features(matches: pd.DataFrame, min_prior: int = MIN_PRIOR_MATCHES) -> pd.DataFrame:
    """Each match described only by what both sides had already done that season.

    The running totals are updated *after* a row is emitted, which is the entire guard
    against leakage: a match never contributes to its own features.
    """
    rows: list[dict] = []
    for _, season in matches.groupby(["competition_id", "sb_season_id"], sort=False):
        history: dict[int, dict[str, float]] = {}
        for match in season.itertuples(index=False):
            home = history.setdefault(match.home_team_id, _blank())
            away = history.setdefault(match.away_team_id, _blank())

            if home["played"] >= min_prior and away["played"] >= min_prior:
                rows.append(_row(match, home, away))
            _update(home, away, match)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in _RUNNING:
        frame[f"d_{column}"] = frame[f"h_{column}"] - frame[f"a_{column}"]
    frame["d_xg_net"] = (frame["h_xgf"] - frame["h_xga"]) - (frame["a_xgf"] - frame["a_xga"])
    return frame


def _blank() -> dict[str, float]:
    return {"played": 0.0, **{c: 0.0 for c in _RUNNING}}


def _row(match, home: dict[str, float], away: dict[str, float]) -> dict:
    row = {
        "match_id": match.match_id,
        "competition_id": match.competition_id,
        "sb_season_id": match.sb_season_id,
        "match_date": match.match_date,
        "home_team_id": match.home_team_id,
        "away_team_id": match.away_team_id,
        "home_goals": match.home_goals,
        "away_goals": match.away_goals,
    }
    for prefix, side in (("h", home), ("a", away)):
        played = side["played"]
        row[f"{prefix}_played"] = played
        for column in _RUNNING:
            row[f"{prefix}_{column}"] = side[column] / played
    row["y"] = (
        0
        if match.home_goals > match.away_goals
        else (1 if match.home_goals == match.away_goals else 2)
    )
    return row


def _update(home: dict[str, float], away: dict[str, float], match) -> None:
    home_points = (
        3.0
        if match.home_goals > match.away_goals
        else (1.0 if match.home_goals == match.away_goals else 0.0)
    )
    away_points = 3.0 - home_points if home_points != 1.0 else 1.0
    for side, values in (
        (
            home,
            (
                home_points,
                match.home_goals,
                match.away_goals,
                match.home_xg,
                match.away_xg,
                match.home_shots,
                match.away_shots,
            ),
        ),
        (
            away,
            (
                away_points,
                match.away_goals,
                match.home_goals,
                match.away_xg,
                match.home_xg,
                match.away_shots,
                match.home_shots,
            ),
        ),
    ):
        side["played"] += 1
        for column, value in zip(_RUNNING, values, strict=True):
            side[column] += float(value)


def time_ordered_split(frame: pd.DataFrame, train_share: float = 0.7) -> pd.Series:
    """True for the last fixtures of each season — the ones a model must not have seen.

    Split per competition-season rather than on one global date, because the database holds
    2015/16 men's leagues alongside 2023/24 women's ones: a single cut-off would train on
    one era and test on another.
    """
    flags = []
    for _, season in frame.groupby(["competition_id", "sb_season_id"], sort=False):
        ordered = season.sort_values("match_date")
        cut = int(len(ordered) * train_share)
        flags.append(pd.Series(np.arange(len(ordered)) >= cut, index=ordered.index))
    return pd.concat(flags).reindex(frame.index)


def train_outcome_model(frame: pd.DataFrame):
    """Multinomial logistic regression over the form differences.

    Logistic regression rather than a gradient-boosted tree, which is measured rather than
    assumed: on the same time-ordered split XGBoost scores 0.510 to this model's 0.553.
    With fourteen features and roughly fourteen hundred training matches there is not
    enough data for a tree ensemble to earn its variance, and the relationship — better
    team wins, by roughly the size of the gap — is close to linear anyway.
    """
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return model.fit(frame[list(FEATURES)].to_numpy(), frame["y"].to_numpy())


@dataclass(frozen=True)
class TeamStrength:
    team_id: int
    attack: float
    defence: float
    matches: int


def team_strengths(
    matches: pd.DataFrame, home_advantage_out: dict | None = None
) -> list[TeamStrength]:
    """Poisson attack and defence coefficients, the classical football rating.

    Goals are modelled as Poisson with a rate set by the scoring side's attack, the
    conceding side's defence and a home term. The coefficients are on a log scale, so 0 is
    an average team, positive attack means more goals than average and *negative* defence
    means fewer conceded — the sign convention that catches people out.

    Unlike the outcome model this uses whole seasons, because it describes teams rather
    than forecasting a fixture: there is nothing to leak into when the output is a rating.
    """
    rows = []
    for match in matches.itertuples(index=False):
        rows.append((match.home_team_id, match.away_team_id, 1.0, match.home_goals))
        rows.append((match.away_team_id, match.home_team_id, 0.0, match.away_goals))
    table = pd.DataFrame(rows, columns=["attack", "defence", "home", "goals"])

    teams = sorted(set(table["attack"]) | set(table["defence"]))
    index = {team: i for i, team in enumerate(teams)}
    design = np.zeros((len(table), 2 * len(teams) + 1))
    for i, row in enumerate(table.itertuples(index=False)):
        design[i, index[row.attack]] = 1.0
        design[i, len(teams) + index[row.defence]] = 1.0
        design[i, -1] = row.home

    # A little regularisation, because a team that played eight matches would otherwise be
    # rated as confidently as one that played thirty-eight.
    fitted = PoissonRegressor(alpha=1e-3, max_iter=5000).fit(design, table["goals"].to_numpy())
    if home_advantage_out is not None:
        home_advantage_out["home_advantage"] = float(fitted.coef_[-1])

    played = table["attack"].value_counts()
    return [
        TeamStrength(
            team_id=int(team),
            attack=float(fitted.coef_[index[team]]),
            defence=float(fitted.coef_[len(teams) + index[team]]),
            matches=int(played.get(team, 0)),
        )
        for team in teams
    ]


def outcome_probabilities(
    home_rate: float, away_rate: float, max_goals: int = 10
) -> tuple[float, float, float]:
    """Home, draw and away probabilities from two Poisson scoring rates.

    Summed over a grid of scorelines rather than approximated: the draw is the diagonal,
    and it is the outcome a classifier trained on labels prices worst, because a draw is
    never the most likely single result even when it is the most likely *kind* of result.
    """
    home = poisson.pmf(np.arange(max_goals + 1), home_rate)
    away = poisson.pmf(np.arange(max_goals + 1), away_rate)
    grid = np.outer(home, away)
    probabilities = np.array(
        [np.tril(grid, -1).sum(), float(np.trace(grid)), np.triu(grid, 1).sum()]
    )
    return tuple(probabilities / probabilities.sum())
