"""Unit tests for structured/NL search — no DB or live LLM."""

from __future__ import annotations

import pandas as pd
import pytest

from footyvision.ml.features import PER90_FEATURES, load_feature_frame, position_group
from footyvision.search import nl, query
from footyvision.search.nl import NLParseError, parse_nl
from footyvision.search.query import Condition, PlayerQuery, execute_query


def test_condition_rejects_unknown_field():
    # "age" used to be the example here, back when the dataset had no dates of birth.
    # It is a real field now, so the rejection has to be shown with something else.
    with pytest.raises(ValueError):
        Condition(field="market_value", op="lt", value=23)


def test_player_query_rejects_unknown_order_by():
    with pytest.raises(ValueError):
        PlayerQuery(order_by="salary")


def test_player_query_tolerates_llm_nulls():
    # LLMs emit explicit nulls; non-optional fields must fall back to defaults.
    q = PlayerQuery.model_validate({"limit": None, "order_desc": None, "position_group": None})
    assert q.limit == 20
    assert q.order_desc is True
    assert q.position_group is None


def _row(pid, name, comp, position, minutes, gender="male", **feats) -> dict:
    d = {
        "player_id": pid,
        "name": name,
        "competition": comp,
        "gender": gender,
        "competition_id": 1,
        "sb_season_id": 1,
        "primary_position": position,
        "matches_played": 20,
        "minutes": minutes,
        "position_group": position_group(position),
    }
    d.update({f: 0.0 for f in PER90_FEATURES})
    d.update(feats)
    return d


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(1, "Winger A", "La Liga", "Right Wing", 2500, xg_per90=0.30, dribbles_per90=5.0),
            _row(2, "Winger B", "La Liga", "Left Wing", 2500, xg_per90=0.20, dribbles_per90=6.0),
            _row(3, "Striker C", "Bundesliga", "Center Forward", 2500, xg_per90=0.55),
            _row(
                4,
                "Alexia Star",
                "Liga F",
                "Center Forward",
                2000,
                gender="female",
                xg_per90=0.85,
                shots_per90=3.2,
            ),
        ]
    )


def test_execute_query_filters_by_gender(monkeypatch):
    monkeypatch.setattr(query, "load_feature_frame", lambda *a, **k: _frame())
    q = PlayerQuery(
        gender="female", conditions=[Condition(field="shots_per90", op="gt", value=0.8)]
    )
    rows = execute_query(session=None, query=q)
    assert [r["name"] for r in rows] == ["Alexia Star"]
    assert rows[0]["gender"] == "female"


def test_execute_query_filters_group_competition_and_condition(monkeypatch):
    monkeypatch.setattr(query, "load_feature_frame", lambda *a, **k: _frame())
    q = PlayerQuery(
        position_group="FWD",
        competition="La Liga",  # excludes the Bundesliga striker
        conditions=[Condition(field="xg_per90", op="gt", value=0.25)],
        order_by="xg_per90",
    )
    rows = execute_query(session=None, query=q)
    assert [r["name"] for r in rows] == ["Winger A"]
    assert rows[0]["stats"]["xg_per90"] == 0.30


def test_execute_query_orders_and_limits(monkeypatch):
    monkeypatch.setattr(query, "load_feature_frame", lambda *a, **k: _frame())
    q = PlayerQuery(order_by="dribbles_per90", order_desc=True, limit=2)
    rows = execute_query(session=None, query=q)
    assert [r["name"] for r in rows] == ["Winger B", "Winger A"]


class _FakeClient:
    def __init__(self, payload: str):
        self.payload = payload

    def chat(self, system: str, user: str, **_) -> str:
        return self.payload


def test_parse_nl_extracts_fenced_json():
    payload = (
        '```json\n{"position_group": "FWD", "conditions": '
        '[{"field": "xg_per90", "op": "gt", "value": 0.25}], "limit": 10}\n```'
    )
    q = parse_nl("strikers with xg over 0.25", client=_FakeClient(payload))
    assert q.position_group == "FWD"
    assert q.conditions[0].field == "xg_per90"
    assert q.limit == 10


def test_parse_nl_extracts_gender():
    payload = '{"conditions": [{"field": "shots_per90", "op": "gt", "value": 0.8}]}'
    q = parse_nl("Female player that has more than 0.8 pct shots", client=_FakeClient(payload))
    assert q.gender == "female"
    assert q.conditions[0].field == "shots_per90"


def test_parse_nl_rejects_invalid_field():
    payload = '{"conditions": [{"field": "market_value", "op": "lt", "value": 23}]}'
    with pytest.raises(NLParseError):
        parse_nl("cheap players", client=_FakeClient(payload))


def test_parse_nl_accepts_an_age_condition():
    """This used to be rejected: the dataset had no dates of birth, and now it does."""
    payload = '{"conditions": [{"field": "age", "op": "lt", "value": 23}]}'

    parsed = parse_nl("under 23", client=_FakeClient(payload))

    assert parsed.conditions[0].field == "age"
    assert parsed.conditions[0].value == 23


def test_extract_json_raises_without_object():
    with pytest.raises(NLParseError):
        nl._extract_json("no json here")


# --- age, foot and height --------------------------------------------------------------


def _seed_attributes(db_session):
    """Dates of birth, feet and a season to measure the ages against."""
    from datetime import date

    from footyvision.db.models import Match, Player, Team

    db_session.add(Team(id=701, name="Home"))
    db_session.add(Team(id=702, name="Away"))
    db_session.flush()
    # A season running Aug 2015 to May 2016: the midpoint lands around Jan 2016.
    for match_id, when, home, away in (
        (7001, date(2015, 8, 1), 701, 702),
        (7002, date(2016, 5, 1), 702, 701),
    ):
        db_session.add(
            Match(
                id=match_id,
                competition_id=11,
                sb_season_id=27,
                match_date=when,
                home_team_id=home,
                away_team_id=away,
            )
        )
    one, two = db_session.get(Player, 1), db_session.get(Player, 2)
    one.date_of_birth, one.foot, one.height_cm = date(1996, 1, 1), "left", 195.0
    two.date_of_birth, two.foot, two.height_cm = date(1986, 1, 1), "right", 170.0
    db_session.commit()


def test_age_is_measured_at_the_season_midpoint_not_today(db_session):
    """A 2015/16 row describes a 2015/16 player; his age today is irrelevant."""
    _seed_attributes(db_session)

    ages = load_feature_frame(db_session, 600).set_index("player_id")["age"]

    assert ages[1] == pytest.approx(20.0, abs=0.2)
    assert ages[2] == pytest.approx(30.0, abs=0.2)


def test_age_is_nan_without_a_date_of_birth(db_session):
    _seed_attributes(db_session)

    frame = load_feature_frame(db_session, 600)

    assert frame[frame["player_id"] == 3]["age"].isna().all()


def test_age_is_searchable(db_session):
    """The under-23 filter the roadmap called the biggest unlock."""
    _seed_attributes(db_session)

    rows = execute_query(
        db_session, PlayerQuery(conditions=[{"field": "age", "op": "lt", "value": 23}])
    )

    assert [r["player_id"] for r in rows] == [1]


def test_age_condition_excludes_players_without_a_birth_date(db_session):
    """A missing age must not quietly pass an age filter."""
    _seed_attributes(db_session)

    rows = execute_query(
        db_session, PlayerQuery(conditions=[{"field": "age", "op": "gt", "value": 0}])
    )

    assert {r["player_id"] for r in rows} == {1, 2}


def test_foot_filters_as_a_category_not_a_condition(db_session):
    _seed_attributes(db_session)

    rows = execute_query(db_session, PlayerQuery(foot="left"))

    assert [r["player_id"] for r in rows] == [1]


def test_nationality_filters_on_the_player_not_the_league(db_session):
    """A Brazilian in La Liga has two countries, and this filter means the first one.

    Matched as a substring because the source spells some of them out in full —
    "Venezuela (Bolivarian Republic)" — and a caller should not have to know that.
    """
    from footyvision.db.models import Player

    db_session.get(Player, 1).country = "Brazil"
    db_session.get(Player, 2).country = "Venezuela\xa0(Bolivarian Republic)"
    db_session.commit()

    assert [
        r["player_id"] for r in execute_query(db_session, PlayerQuery(nationality="Brazil"))
    ] == [1]
    assert [
        r["player_id"] for r in execute_query(db_session, PlayerQuery(nationality="venezuela"))
    ] == [2]
    assert execute_query(db_session, PlayerQuery(nationality="Portugal")) == []


def test_foot_rejects_a_value_that_is_not_a_foot():
    with pytest.raises(ValueError):
        PlayerQuery(foot="either")


def test_height_is_a_numeric_condition(db_session):
    _seed_attributes(db_session)

    rows = execute_query(
        db_session, PlayerQuery(conditions=[{"field": "height_cm", "op": "gte", "value": 190}])
    )

    assert [r["player_id"] for r in rows] == [1]
