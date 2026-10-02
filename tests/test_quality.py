"""Season completeness: what decides whether a whole competition-season is in the pool.

`fragment_seasons` is read by `load_feature_frame`, so a mistake here does not show up as a
wrong number on one page — it silently removes every player of a season from every
percentile, similarity score and model in the project. Two of its cases have already been
wrong once: a single-club export read as a tenth of a league, and a handful of matches with
no away side recorded added a phantom club to Ligue 1 and made the whole season a fragment.
"""

from __future__ import annotations

from itertools import count, permutations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from footyvision.db.base import Base
from footyvision.db.models import Match
from footyvision.db.quality import fragment_seasons, season_facts

_ids = count(1)


@pytest.fixture
def session() -> Session:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, future=True)()


def _add(session: Session, key: tuple[int, int], fixtures) -> None:
    comp, season = key
    session.add_all(
        Match(
            id=next(_ids), competition_id=comp, sb_season_id=season, home_team_id=h, away_team_id=a
        )
        for h, a in fixtures
    )
    session.commit()


def _double_round_robin(teams: range):
    return list(permutations(teams, 2))


def test_a_full_league_season_is_complete(session):
    _add(session, (11, 27), _double_round_robin(range(1, 7)))  # 6 teams, 30 fixtures

    facts = season_facts(session)[(11, 27)]
    assert (facts.matches, facts.teams, facts.focus_team_id) == (30, 6, None)
    assert facts.coverage == 1.0
    assert fragment_seasons(session) == set()


def test_a_truncated_league_season_is_a_fragment(session):
    _add(session, (11, 27), _double_round_robin(range(1, 7))[:12])

    assert season_facts(session)[(11, 27)].coverage == pytest.approx(12 / 30)
    assert fragment_seasons(session) == {(11, 27)}


def test_a_single_club_export_is_measured_against_that_club_not_the_league(session):
    """Bundesliga 2015/16 in the open data is Leverkusen's 34 matches. Against a full
    18-team league that is 11%; against Leverkusen's own fixture list it is complete, and
    treating it as a fragment would have deleted nineteen full-season players."""
    club = 1
    opponents = range(2, 19)
    fixtures = [(club, o) for o in opponents] + [(o, club) for o in opponents]
    _add(session, (9, 27), fixtures)

    facts = season_facts(session)[(9, 27)]
    assert facts.focus_team_id == club
    assert facts.coverage == 1.0
    assert fragment_seasons(session) == set()


def test_matches_with_no_away_side_do_not_add_a_phantom_club(session):
    """Twenty Ligue 1 matches carry a null away team. Counted as a side, the null became a
    twenty-first club, the expected fixture list grew, and the season read 0.90."""
    fixtures = _double_round_robin(range(1, 7))
    fixtures[:3] = [(home, None) for home, _ in fixtures[:3]]
    _add(session, (7, 27), fixtures)

    facts = season_facts(session)[(7, 27)]
    assert facts.teams == 6
    assert facts.coverage == 1.0


def test_a_season_with_no_fixtures_is_not_excluded_on_an_unknown(session):
    """No match rows means coverage cannot be judged, not that the season is a fragment —
    otherwise every instance holding aggregates without matches would have an empty pool."""
    assert fragment_seasons(session) == set()
