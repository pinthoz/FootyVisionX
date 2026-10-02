"""HTTP-level tests for the API — real routing and SQL, no Postgres and no live LLM.

These cover what the routers add on top of the ML/search layers: request validation,
response shape and the error contract (404 for unknown players, 422 for bad input,
503 when the local LLM is unreachable).
"""

from __future__ import annotations

import pytest

from footyvision.api.routers import assistant as assistant_router
from footyvision.api.routers import reports as reports_router
from footyvision.api.routers import search as search_router
from footyvision.llm.client import LLMError
from footyvision.search.nl import NLParseError
from footyvision.search.query import PlayerQuery

# --- meta -----------------------------------------------------------------------------


def test_root_reports_version(client):
    body = client.get("/").json()
    assert body["name"] == "FootyVision API"
    assert body["version"]


def test_health_reports_database_reachable(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database"] is True


def test_openapi_schema_is_generated(client):
    # Catches response_model/annotation mistakes that only surface at schema build time.
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert "/players/{player_id}/similar" in schema.json()["paths"]


# --- players --------------------------------------------------------------------------


def test_list_players_returns_all_seeded(client):
    body = client.get("/players").json()
    assert len(body) == 6
    assert body[0]["name"] == "Alpha Striker"  # ordered by name


def test_list_players_filters_by_name_case_insensitively(client):
    body = client.get("/players", params={"search": "anchor"}).json()
    assert {p["name"] for p in body} == {"Delta Anchor", "Echo Anchor"}


def test_list_players_can_exclude_players_without_a_season_aggregate(client, db_session):
    # A player who only ever appeared in a match has no radar and no score, so the
    # dashboard must be able to keep them out of the search results.
    from footyvision.db.models import Player

    db_session.add(Player(id=99, name="Zulu Benchwarmer"))
    db_session.commit()

    everyone = {p["name"] for p in client.get("/players").json()}
    selectable = {p["name"] for p in client.get("/players", params={"with_stats": True}).json()}
    assert "Zulu Benchwarmer" in everyone
    assert "Zulu Benchwarmer" not in selectable
    assert len(selectable) == 6


def test_list_players_filters_by_gender(client, db_session):
    from footyvision.db.models import METRIC_COLUMNS, Competition, Player, PlayerSeasonStats

    db_session.add(Competition(id=99, name="Liga F", country="Spain", gender="female"))
    db_session.add(Player(id=99, name="Alexia Star", country="Spain"))
    per90 = {f"{m}_per90": 0.0 for m in METRIC_COLUMNS}
    db_session.add(
        PlayerSeasonStats(
            player_id=99,
            competition_id=99,
            sb_season_id=1,
            primary_position="Center Forward",
            matches_played=20,
            minutes=1800.0,
            **per90,
        )
    )
    db_session.commit()

    female_players = client.get("/players", params={"gender": "female"}).json()
    assert len(female_players) == 1
    assert female_players[0]["name"] == "Alexia Star"
    assert female_players[0]["gender"] == "female"

    all_players = client.get("/players").json()
    assert any(p["name"] == "Alexia Star" for p in all_players)


def test_list_players_rejects_out_of_range_limit(client):
    assert client.get("/players", params={"limit": 500}).status_code == 422


def test_get_player_returns_404_for_unknown_id(client):
    response = client.get("/players/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Player not found"


def test_player_seasons_returns_per90_rates(client):
    body = client.get("/players/1/seasons").json()
    assert len(body) == 1
    assert body[0]["xg_per90"] == pytest.approx(0.80)


# --- similarity and radar -------------------------------------------------------------


def test_similar_players_excludes_the_target_and_ranks_by_similarity(client):
    body = client.get("/players/1/similar", params={"min_minutes": 500}).json()
    assert body["target"]["name"] == "Alpha Striker"
    returned = [r["player_id"] for r in body["results"]]
    assert 1 not in returned
    scores = [r["similarity"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_similar_players_stays_within_the_position_group(client):
    body = client.get("/players/1/similar", params={"min_minutes": 500}).json()
    assert {r["position_group"] for r in body["results"]} == {"FWD"}


def test_similar_players_404s_when_below_the_minutes_floor(client):
    # Player 6 has 200 minutes, so a 500-minute floor removes them from the pool.
    response = client.get("/players/6/similar", params={"min_minutes": 500})
    assert response.status_code == 404


def test_radar_returns_percentiles_for_every_metric(client):
    body = client.get("/players/1/radar", params={"min_minutes": 500}).json()
    assert body["position_group"] == "FWD"
    # The best xG per 90 among forwards must sit at the top of its percentile range.
    assert body["metrics"]["xg_per90"]["percentile"] == pytest.approx(100.0)


# --- search ---------------------------------------------------------------------------


def test_structured_search_needs_no_llm(client):
    response = client.post(
        "/search/structured",
        json={
            "position_group": "FWD",
            "min_minutes": 500,
            "conditions": [{"field": "xg_per90", "op": "gt", "value": 0.5}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert {r["name"] for r in body["results"]} == {"Alpha Striker", "Bravo Striker"}


def test_structured_search_rejects_a_field_outside_the_whitelist(client):
    response = client.post(
        "/search/structured",
        json={"conditions": [{"field": "salary", "op": "gt", "value": 1}]},
    )
    assert response.status_code == 422


def test_nl_search_runs_the_query_the_llm_structured(client, monkeypatch):
    monkeypatch.setattr(
        search_router,
        "parse_nl",
        lambda _q: PlayerQuery(position_group="FWD", min_minutes=500, order_by="xg_per90"),
    )
    body = client.post("/search", json={"query": "best forwards"}).json()
    assert body["interpreted"]["position_group"] == "FWD"
    assert body["results"][0]["name"] == "Alpha Striker"


def test_nl_search_returns_503_when_the_llm_is_unreachable(client, monkeypatch):
    def _unreachable(_q):
        raise LLMError("connection refused")

    monkeypatch.setattr(search_router, "parse_nl", _unreachable)
    response = client.post("/search", json={"query": "best forwards"})
    assert response.status_code == 503


def test_nl_search_returns_422_when_the_llm_output_is_unusable(client, monkeypatch):
    def _unparsable(_q):
        raise NLParseError("model did not return valid JSON")

    monkeypatch.setattr(search_router, "parse_nl", _unparsable)
    response = client.post("/search", json={"query": "???"})
    assert response.status_code == 422


# --- LLM-backed endpoints -------------------------------------------------------------


def test_report_context_is_computed_without_an_llm(client):
    body = client.get("/players/1/report/context", params={"min_minutes": 500}).json()
    assert body["player_id"] == 1
    assert body["context"]


def test_report_context_404s_for_a_player_outside_the_pool(client):
    response = client.get("/players/999/report/context", params={"min_minutes": 500})
    assert response.status_code == 404


def test_report_returns_503_when_the_llm_is_unreachable(client, monkeypatch):
    def _unreachable(*_args, **_kwargs):
        raise LLMError("LM Studio is not running")

    monkeypatch.setattr(reports_router, "generate_report", _unreachable)
    response = client.post("/players/1/report")
    assert response.status_code == 503
    assert "LM Studio" in response.json()["detail"]


def test_assistant_returns_503_when_the_llm_is_unreachable(client, monkeypatch):
    def _unreachable(_session):
        raise LLMError("embedding model not loaded")

    monkeypatch.setattr(assistant_router, "get_store", _unreachable)
    response = client.post("/assistant", json={"question": "who wins the ball back?"})
    assert response.status_code == 503


# --- metric distribution --------------------------------------------------------------


def test_metric_distribution_returns_every_player_in_the_pool(client):
    body = client.get("/metrics/xg_per90/distribution", params={"min_minutes": 500}).json()
    assert body["metric"] == "xg_per90"
    # Player 6 sits below the minutes floor, so five of the six seeded players remain.
    assert body["count"] == 5
    assert max(v["value"] for v in body["values"]) == pytest.approx(0.80)


def test_metric_distribution_can_scope_to_a_position_group(client):
    body = client.get(
        "/metrics/xg_per90/distribution",
        params={"min_minutes": 500, "position_group": "FWD"},
    ).json()
    assert body["position_group"] == "FWD"
    assert {v["name"] for v in body["values"]} == {
        "Alpha Striker",
        "Bravo Striker",
        "Charlie Striker",
    }


def test_metric_distribution_rejects_a_metric_outside_the_feature_set(client):
    response = client.get("/metrics/salary/distribution")
    assert response.status_code == 422


# --- rate limiting and CORS -------------------------------------------------------------


def test_llm_endpoints_are_throttled_per_client(client, monkeypatch):
    """An unthrottled public deployment lets anyone drain the API key attached to it."""
    from footyvision.api import limits
    from footyvision.config import Settings, get_settings

    limits.reset()
    get_settings.cache_clear()
    monkeypatch.setattr(
        get_settings, "__wrapped__", lambda: Settings(rate_limit_per_minute=2), raising=False
    )
    monkeypatch.setattr(limits, "get_settings", lambda: Settings(rate_limit_per_minute=2))

    def _answer(self, question, k=6):
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(assistant_router.ScoutAssistant, "answer", _answer)
    monkeypatch.setattr(assistant_router, "get_store", lambda *_a, **_k: object())

    codes = [client.post("/assistant", json={"question": "q"}).status_code for _ in range(3)]

    assert codes[:2] == [200, 200]
    assert codes[2] == 429
    limits.reset()


def test_a_throttled_response_says_when_to_retry(client, monkeypatch):
    from footyvision.api import limits
    from footyvision.config import Settings

    limits.reset()
    monkeypatch.setattr(limits, "get_settings", lambda: Settings(rate_limit_per_minute=1))

    def _answer(self, question, k=6):
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(assistant_router.ScoutAssistant, "answer", _answer)
    monkeypatch.setattr(assistant_router, "get_store", lambda *_a, **_k: object())

    client.post("/assistant", json={"question": "q"})
    blocked = client.post("/assistant", json={"question": "q"})

    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers
    limits.reset()


def test_rate_limit_of_zero_disables_throttling(client, monkeypatch):
    """Local development and the rest of this suite run with the limiter off."""
    from footyvision.api import limits
    from footyvision.config import Settings

    limits.reset()
    monkeypatch.setattr(limits, "get_settings", lambda: Settings(rate_limit_per_minute=0))

    def _answer(self, question, k=6):
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(assistant_router.ScoutAssistant, "answer", _answer)
    monkeypatch.setattr(assistant_router, "get_store", lambda *_a, **_k: object())

    codes = [client.post("/assistant", json={"question": "q"}).status_code for _ in range(5)]

    assert codes == [200] * 5


def test_clients_are_told_apart_by_the_forwarded_header():
    """Render and Vercel terminate TLS in front of the app, so every caller would
    otherwise share one bucket behind the proxy's own address."""
    from starlette.datastructures import Headers

    from footyvision.api.limits import client_key

    class _Request:
        def __init__(self, forwarded):
            self.headers = Headers({"x-forwarded-for": forwarded} if forwarded else {})
            self.client = None

    # The rightmost hop is the one the proxy in front of the app wrote.
    assert client_key(_Request("203.0.113.7, 10.0.0.1")) == "10.0.0.1"
    assert client_key(_Request("198.51.100.4")) == "198.51.100.4"
    assert client_key(_Request(None)) == "unknown"


def test_a_forged_forwarded_header_does_not_buy_a_fresh_limit(monkeypatch):
    """Proxies append to X-Forwarded-For, so whatever sits on the left is the client's own
    invention. Keying on it let a caller send a new value per request and never be
    throttled, which on these endpoints means spending the LLM budget without limit."""
    from fastapi import HTTPException
    from starlette.datastructures import Headers

    from footyvision.api import limits

    monkeypatch.setattr(
        limits, "get_settings", lambda: type("S", (), {"rate_limit_per_minute": 3})()
    )
    limits.reset()

    class _Request:
        def __init__(self, forged):
            # The client's forgery, then the address Render actually saw.
            self.headers = Headers({"x-forwarded-for": f"{forged}, 203.0.113.9"})
            self.client = None

    for i in range(3):
        limits.rate_limit(_Request(f"10.0.0.{i}"))
    with pytest.raises(HTTPException) as refused:
        limits.rate_limit(_Request("10.0.0.99"))
    assert refused.value.status_code == 429
    limits.reset()


def test_idle_clients_are_forgotten_once_the_table_is_full(monkeypatch):
    """Windows used to be emptied but never removed, so every distinct caller stayed in
    memory for the life of the process — a leak with no ceiling on a 512MB instance."""
    from starlette.datastructures import Headers

    from footyvision.api import limits

    monkeypatch.setattr(
        limits, "get_settings", lambda: type("S", (), {"rate_limit_per_minute": 5})()
    )
    monkeypatch.setattr(limits, "MAX_TRACKED_CLIENTS", 3)
    clock = [1000.0]
    monkeypatch.setattr(limits.time, "monotonic", lambda: clock[0])
    limits.reset()

    class _Request:
        def __init__(self, ip):
            self.headers = Headers({"x-forwarded-for": ip})
            self.client = None

    for ip in ("a", "b", "c"):
        limits.rate_limit(_Request(ip))
    clock[0] += limits.WINDOW_SECONDS + 1
    limits.rate_limit(_Request("d"))

    assert set(limits._HITS) == {"d"}
    limits.reset()


def test_cors_defaults_to_local_development_when_unset():
    """ "*" would let any site on the internet spend this instance's LLM budget."""
    from footyvision.config import Settings

    assert Settings(cors_origins="").allowed_origins == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    assert Settings(cors_origins="https://a.app, https://b.app").allowed_origins == [
        "https://a.app",
        "https://b.app",
    ]


def _allowed_origin(origin: str) -> str | None:
    from fastapi.testclient import TestClient

    from footyvision.api.main import app

    response = TestClient(app).get("/", headers={"Origin": origin})
    return response.headers.get("access-control-allow-origin")


def test_cors_admits_the_production_dashboard():
    """Checked on the header the browser acts on. The old version asserted a 200, which
    the API returns to any origin whatever CORS decides."""
    origin = "https://footy-vision-tau.vercel.app"
    assert _allowed_origin(origin) == origin


def test_cors_refuses_other_vercel_apps():
    """*.vercel.app used to be admitted, so any stranger's free deployment could spend this
    API's LLM budget from its visitors' browsers — each with a fresh rate-limit bucket."""
    assert _allowed_origin("https://someone-elses-app.vercel.app") is None
    assert _allowed_origin("https://footy-vision-abc-someone-else.vercel.app") is None


def test_previews_are_admitted_only_for_the_configured_scope():
    import re

    from footyvision.config import Settings

    rx = re.compile(Settings(cors_preview_scope="pinthozs-projects").allowed_origin_regex)
    assert rx.fullmatch("https://footy-vision-3f9a1c-pinthozs-projects.vercel.app")
    assert not rx.fullmatch("https://footy-vision-3f9a1c-other-team.vercel.app")
    assert not re.compile(Settings().allowed_origin_regex).fullmatch(
        "https://footy-vision-3f9a1c-pinthozs-projects.vercel.app"
    )


def test_assistant_evaluation_is_served_from_the_published_snapshot(client):
    """The scores shown in the dashboard are the ones the eval script measured.

    Served from a committed file rather than recomputed: scoring costs dozens of LLM calls
    and the result is a statement about a particular index on a particular day, not
    something to recalculate on a page load.
    """
    response = client.get("/eval/assistant")
    if response.status_code == 404:
        # A checkout where the evaluation has never finished has nothing to publish, and
        # the endpoint says so rather than serving zeros. That is the state right after a
        # run is interrupted by a quota, so it has to be a legal one.
        pytest.skip("no evaluation published yet")

    body = response.json()
    assert body["questions"] > 0
    assert 0.0 <= body["metrics"]["faithfulness"] <= 1.0
    assert len(body["rows"]) == body["questions"]
    # The questions with no answer in the data are counted apart, because the assistant
    # scoring well there means it declined rather than invented.
    assert body["unanswerable"]["count"] >= 1
    # And the headline relevancy excludes them, since RAGAS scores a refusal as irrelevant
    # by design — averaging that in would reward a model that answered anyway.
    assert body["answerable_relevancy"] >= body["metrics"]["answer_relevancy"]


def test_team_strength_rates_the_side_that_keeps_winning(client, db_session):
    """Ratings come from the fixtures, so an empty database must not fabricate any."""
    from footyvision.db.models import Match, PlayerMatchStats, Team

    assert client.get("/teams/strength").json()["teams"] == []

    db_session.add_all([Team(id=910, name="Strong"), Team(id=911, name="Weak")])
    db_session.flush()
    for i in range(12):
        home, away = (910, 911) if i % 2 == 0 else (911, 910)
        db_session.add(
            Match(
                id=9000 + i,
                competition_id=1,
                sb_season_id=1,
                home_team_id=home,
                away_team_id=away,
            )
        )
        # Strong scores three and concedes none, whichever end it plays at. A different
        # player per side: (match, player) is unique, so one cannot appear for both.
        for player_id, team, goals in ((1, 910, 3), (2, 911, 0)):
            db_session.add(
                PlayerMatchStats(
                    match_id=9000 + i,
                    player_id=player_id,
                    team_id=team,
                    minutes=90,
                    goals=goals,
                )
            )
    db_session.commit()

    from footyvision.api.routers import teams as teams_router

    teams_router._CACHE.clear()
    body = client.get("/teams/strength").json()

    rated = {t["name"]: t for t in body["teams"]}
    assert rated["Strong"]["attack"] > rated["Weak"]["attack"]
    # Negative defence means conceding less than average — the sign that catches people out.
    assert rated["Strong"]["defence"] < rated["Weak"]["defence"]
    teams_router._CACHE.clear()


def test_the_docs_page_is_themed_and_still_generated(client):
    """Swagger's default white reads as a different product beside the dashboard.

    Only the presentation is replaced: the page is still built from the OpenAPI schema, so
    an endpoint added tomorrow appears without anyone editing a template. The header is
    counted off the same schema, which is what this pins.
    """
    page = client.get("/docs")

    assert page.status_code == 200
    assert 'id="swagger-ui"' in page.text
    assert "/static/docs.css" in page.text
    assert "FootyVision" in page.text
    # Every tag lands in exactly one section of the map above the reference.
    from footyvision.api.docs import SECTIONS

    spec = client.get("/openapi.json").json()
    tagged = {
        t
        for methods in spec["paths"].values()
        for op in methods.values()
        for t in op.get("tags", [])
    }
    mapped = {tag for _, _, tags in SECTIONS for tag in tags}
    assert tagged <= mapped, f"untagged in the docs map: {sorted(tagged - mapped)}"

    css = client.get("/static/docs.css")
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/assistant", {"question": "x" * 501}),
        ("/assistant", {"question": "left-footed wingers", "k": 5000}),
        ("/assistant", {"question": "left-footed wingers", "k": 0}),
        ("/assistant", {"question": ""}),
        ("/search", {"query": "x" * 501}),
    ],
)
def test_llm_endpoints_refuse_requests_sized_to_run_up_a_bill(path, body):
    """One unbounded request could send every profile in the pool to the model, and the
    rate limiter would count it as one call. Refused before any token is spent."""
    from fastapi.testclient import TestClient

    from footyvision.api.main import app

    assert TestClient(app).post(path, json=body).status_code == 422
