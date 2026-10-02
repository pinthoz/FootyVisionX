"""Expected points from shot xG: the arithmetic, and the published table.

Every shot is an independent chance of scoring with probability equal to its xG, so a
side's goals in a match follow a Poisson-binomial distribution; two sides' distributions
give the chance of each result, and 3 x P(win) + P(draw) is the points the chances were
worth.
"""

from __future__ import annotations

import pytest

from footyvision.ml import expected_table as et


def test_goal_distribution_is_exact_and_sums_to_one():
    dist = et.goal_distribution([0.5, 0.5])
    assert dist[:3] == pytest.approx([0.25, 0.5, 0.25])
    assert sum(dist) == pytest.approx(1.0)
    assert et.goal_distribution([])[0] == 1.0  # no shots, no goals


def test_one_even_chance_against_none_is_half_a_win_and_half_a_draw():
    win, draw, loss = et.match_probabilities([0.5], [])
    assert (win, draw, loss) == pytest.approx((0.5, 0.5, 0.0))


def test_identical_chances_are_symmetric():
    win, draw, loss = et.match_probabilities([0.3, 0.1], [0.3, 0.1])
    assert win == pytest.approx(loss)
    assert win + draw + loss == pytest.approx(1.0)


def test_xg_is_clamped_to_a_probability():
    """A malformed xG must not produce a distribution that is not one."""
    assert sum(et.goal_distribution([1.7, -0.2])) == pytest.approx(1.0)


def test_the_published_table_is_served_and_leaves_out_single_club_seasons():
    """Bundesliga 2015/16 in the open data is Leverkusen's matches and two apiece for the
    rest; a table of it would rank seventeen clubs on two games."""
    from fastapi.testclient import TestClient

    from footyvision.api.main import app

    payload = et.load()
    assert payload is not None, f"missing {et.ARTIFACT}: run `footyvision spatial`"
    body = TestClient(app).get("/teams/expected-table").json()
    assert body["seasons"]
    for season in body["seasons"]:
        assert min(t["played"] for t in season["teams"]) >= 10
        for team in season["teams"]:
            assert team["luck"] == pytest.approx(team["points"] - team["xpts"], abs=0.02)
