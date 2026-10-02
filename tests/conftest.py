"""Shared fixtures: an in-memory database and an API client, both offline.

The API is exercised against SQLite rather than a fake session so the real SQLAlchemy
statements (joins, filters, `pd.read_sql`) are actually executed — but still with no
Postgres, no network and no LLM. `StaticPool` keeps every connection pointing at the same
in-memory database, which matters because `load_feature_frame` reads through the engine
rather than the session.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from footyvision.api.main import app
from footyvision.db.base import Base, get_session
from footyvision.db.models import METRIC_COLUMNS, Competition, Player, PlayerSeasonStats

# Player-seasons covering two position groups, so grouping and percentiles are meaningful.
_SEED = [
    (1, "Alpha Striker", "Center Forward", 2400.0, {"xg_per90": 0.80, "shots_per90": 4.0}),
    (2, "Bravo Striker", "Center Forward", 2100.0, {"xg_per90": 0.55, "shots_per90": 3.0}),
    (3, "Charlie Striker", "Right Wing", 1800.0, {"xg_per90": 0.20, "shots_per90": 1.5}),
    (4, "Delta Anchor", "Center Defensive Midfield", 2700.0, {"tackles_per90": 3.4}),
    (5, "Echo Anchor", "Center Midfield", 2200.0, {"tackles_per90": 2.1}),
    (6, "Foxtrot Fringe", "Center Midfield", 200.0, {"tackles_per90": 9.9}),  # below the floor
]


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """The trained models are memoised per process, so tests would share one another's.

    The same cache is why the running API has to be restarted after loading a season:
    it keeps serving the model it fitted on the smaller pool.
    """
    from footyvision.ml.talent import _CACHE, _SHAP_CACHE

    _CACHE.clear()
    _SHAP_CACHE.clear()
    yield
    _CACHE.clear()
    _SHAP_CACHE.clear()


@pytest.fixture
def db_session() -> Session:
    """A seeded in-memory database session."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)()

    session.add(Competition(id=11, name="La Liga", country="Spain"))
    for pid, name, position, minutes, feats in _SEED:
        session.add(Player(id=pid, name=name, country="Spain"))
        per90 = {f"{m}_per90": 0.0 for m in METRIC_COLUMNS}
        per90.update(feats)
        session.add(
            PlayerSeasonStats(
                player_id=pid,
                competition_id=11,
                sb_season_id=27,
                primary_position=position,
                matches_played=int(minutes // 90),
                minutes=minutes,
                **per90,
            )
        )
    session.commit()

    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def client(db_session: Session) -> TestClient:
    """A TestClient wired to the seeded in-memory database."""
    app.dependency_overrides[get_session] = lambda: db_session
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
