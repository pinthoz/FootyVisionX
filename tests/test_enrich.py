"""The lineup walk has to know when it is finished.

Nickname and country come from the lineup feed, which lists a player once per match with
whatever it holds — so a second sighting of the same player can never add anything. The
walk therefore has to treat "seen" as done, not "filled". Treating filled as done leaves
every player the feed has no nickname for pending forever, and the walk re-reads the whole
fixture list on every run chasing them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from footyvision.db.base import Base
from footyvision.db.models import Competition, Match, Player


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)()
    s.add(Competition(id=1, name="Test League", country="Nowhere", gender="female"))
    s.add_all(
        [
            Player(id=10, name="Ana Full Name"),
            # Already has a country but no nickname: the case the old rule never revisited.
            Player(id=11, name="Bea Full Name", country="Portugal"),
        ]
    )
    s.add_all(
        [
            Match(id=100, competition_id=1, sb_season_id=1),
            Match(id=101, competition_id=1, sb_season_id=1),
        ]
    )
    s.commit()
    yield s
    s.close()


def _lineup_frame() -> pd.DataFrame:
    """One squad: Ana has a nickname on file, Bea has none. Both have a country."""
    return pd.DataFrame(
        {
            "player_id": [10, 11],
            "player_name": ["Ana Full Name", "Bea Full Name"],
            "player_nickname": ["Ana", np.nan],
            "country": ["Brazil", "Portugal"],
        }
    )


def test_a_player_the_feed_has_no_nickname_for_is_not_chased_forever(session, monkeypatch):
    from footyvision.etl import enrich

    calls: list[int] = []

    def fake_lineups(match_id: int):
        calls.append(match_id)
        return {"Test Team": _lineup_frame()}

    monkeypatch.setattr(enrich.sb, "lineups", fake_lineups)
    stats = enrich.enrich_from_lineups(session)

    # The second match is never fetched: after one sighting there is nothing left to learn.
    assert calls == [100]
    assert stats["matches_read"] == 1
    assert stats["still_missing"] == 0

    ana = session.get(Player, 10)
    bea = session.get(Player, 11)
    assert (ana.nickname, ana.country) == ("Ana", "Brazil")
    # Bea keeps her country and gains nothing — but she is done, not pending.
    assert bea.nickname is None
    assert bea.country == "Portugal"


def test_an_unavailable_lineup_is_counted_rather_than_swallowed(session, monkeypatch):
    """A whole competition failing looks identical to one already complete otherwise."""
    from footyvision.etl import enrich

    def always_fails(match_id: int):
        raise RuntimeError("404")

    monkeypatch.setattr(enrich.sb, "lineups", always_fails)
    stats = enrich.enrich_from_lineups(session)

    assert stats["unavailable"] == 2
    assert stats["nicknames"] == 0
    assert stats["still_missing"] == 2
