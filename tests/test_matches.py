"""Match outcome and team strength — synthetic fixtures, no database beyond the session."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footyvision.ml.matches import (
    MIN_PRIOR_MATCHES,
    build_features,
    outcome_probabilities,
    team_strengths,
    time_ordered_split,
)


def _matches(results: list[tuple[int, int, int, int]]) -> pd.DataFrame:
    """(home_team, away_team, home_goals, away_goals) in fixture order."""
    rows = []
    for i, (home, away, hg, ag) in enumerate(results):
        rows.append(
            {
                "match_id": 1000 + i,
                "competition_id": 1,
                "sb_season_id": 1,
                "match_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "home_team_id": home,
                "away_team_id": away,
                "home_goals": hg,
                "away_goals": ag,
                "home_xg": hg * 0.9,
                "away_xg": ag * 0.9,
                "home_shots": hg * 4,
                "away_shots": ag * 4,
            }
        )
    return pd.DataFrame(rows)


def _round_robin(strong: int, weak: int, rounds: int) -> list[tuple[int, int, int, int]]:
    """The strong side wins every time, alternating home and away."""
    out = []
    for i in range(rounds):
        out.append((strong, weak, 3, 0) if i % 2 == 0 else (weak, strong, 0, 3))
    return out


def test_a_match_never_contributes_to_its_own_features():
    """The whole guard against leakage, and the one worth a test of its own.

    A match's statistics are its result: if the running totals were updated before the row
    was emitted, `d_gf` would carry the scoreline and the model would 'predict' it
    perfectly. Two identical fixtures ending in opposite scorelines must produce identical
    features.
    """
    history = _round_robin(1, 2, MIN_PRIOR_MATCHES)
    home_win = build_features(_matches([*history, (1, 2, 5, 0)]))
    away_win = build_features(_matches([*history, (1, 2, 0, 5)]))

    feature_columns = [c for c in home_win.columns if c.startswith(("h_", "a_", "d_"))]
    last_home = home_win.iloc[-1][feature_columns]
    last_away = away_win.iloc[-1][feature_columns]

    pd.testing.assert_series_equal(last_home, last_away, check_names=False)
    # ...and the labels do differ, so the test is not passing on two empty frames.
    assert home_win.iloc[-1]["y"] == 0
    assert away_win.iloc[-1]["y"] == 2


def test_a_side_needs_a_history_before_its_fixture_is_usable():
    """A league table after two matches describes the fixture list, not the teams."""
    frame = build_features(_matches(_round_robin(1, 2, MIN_PRIOR_MATCHES + 2)))

    # The first MIN_PRIOR_MATCHES fixtures have no usable history behind them.
    assert len(frame) == 2
    assert (frame["h_played"] >= MIN_PRIOR_MATCHES).all()
    assert (frame["a_played"] >= MIN_PRIOR_MATCHES).all()


def test_form_features_describe_the_stronger_side():
    frame = build_features(_matches([*_round_robin(1, 2, 6), (1, 2, 1, 0)]))
    last = frame.iloc[-1]

    # Team 1 has won everything, so its points and goals per match lead by a distance.
    assert last["d_pts"] > 0
    assert last["d_gf"] > 0
    assert last["d_ga"] < 0


def test_the_split_holds_out_the_later_fixtures():
    """A random fold would train on matches played after the ones it predicts."""
    frame = build_features(_matches(_round_robin(1, 2, 20)))

    is_test = time_ordered_split(frame, train_share=0.7)

    assert is_test.sum() > 0
    assert frame.loc[~is_test, "match_date"].max() < frame.loc[is_test, "match_date"].min()


def test_team_strengths_rank_the_side_that_scores_more():
    """Attack is positive for scoring above average; defence is *negative* for conceding
    below it, which is the sign convention that catches people out."""
    strengths = {t.team_id: t for t in team_strengths(_matches(_round_robin(1, 2, 12)))}

    assert strengths[1].attack > strengths[2].attack
    assert strengths[1].defence < strengths[2].defence
    assert strengths[1].matches == 12


def test_outcome_probabilities_are_a_distribution_over_a_scoreline_grid():
    home, draw, away = outcome_probabilities(1.8, 1.0)

    assert home + draw + away == pytest.approx(1.0)
    assert home > away
    # Evenly matched sides make the draw more likely than when one is far stronger.
    assert outcome_probabilities(1.2, 1.2)[1] > draw


def test_a_home_advantage_shows_up_as_more_goals_at_home():
    """The classical model puts home advantage in the goal rate, not in the label."""
    info: dict[str, float] = {}
    # The same two sides, drawing every time, but the home side always scores more.
    fixtures = [(1, 2, 2, 1) if i % 2 == 0 else (2, 1, 2, 1) for i in range(16)]
    team_strengths(_matches(fixtures), info)

    assert info["home_advantage"] > 0
    assert np.exp(info["home_advantage"]) > 1.0
