"""What data the database actually holds, read live rather than hard-coded.

The StatsBomb open dataset mixes complete league seasons with fragments that carry the
same competition and season names — La Liga 2018/19, for instance, is only Barcelona's
matches. A count of matches alone does not distinguish them, so every season is reported
against the number of matches a full double round-robin of its teams would have.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from footyvision.api.schemas import CatalogueEntry, CoverageResponse, CoverageSeason
from footyvision.db.base import get_session
from footyvision.db.models import Competition, PlayerSeasonStats, Season, Team
from footyvision.db.quality import COMPLETE_THRESHOLD, season_facts

# What StatsBomb Open Data offers, checked against the live competition list on this date.
# A snapshot rather than a live call: listing every season costs one HTTP request each, and
# the open dataset changes a few times a year, not a few times a day.
CATALOGUE_VERIFIED = "2026-09-02"

# (competition_id, season_id, competition, country, season, matches, teams, gender, kind)
_CATALOGUE: tuple[tuple, ...] = (
    # The only four complete men's domestic seasons, and all in the same campaign.
    (11, 27, "La Liga", "Spain", "2015/2016", 380, 20, "men", "league"),
    (2, 27, "Premier League", "England", "2015/2016", 380, 20, "men", "league"),
    (12, 27, "Serie A", "Italy", "2015/2016", 380, 20, "men", "league"),
    (7, 27, "Ligue 1", "France", "2015/2016", 377, 20, "men", "league"),
    (1238, 108, "Indian Super League", "India", "2021/2022", 115, 11, "men", "league"),
    # The only recent complete league seasons in the open data are the women's game.
    (182, 281, "Liga F", "Spain", "2023/2024", 240, 16, "women", "league"),
    (49, 107, "NWSL", "USA", "2023", 137, 12, "women", "league"),
    (37, 281, "FA Women's Super League", "England", "2023/2024", 132, 12, "women", "league"),
    (135, 281, "Frauen Bundesliga", "Germany", "2023/2024", 132, 12, "women", "league"),
    (131, 281, "Serie A Women", "Italy", "2023/2024", 130, 10, "women", "league"),
    (37, 90, "FA Women's Super League", "England", "2020/2021", 131, 12, "women", "league"),
    (37, 4, "FA Women's Super League", "England", "2018/2019", 107, 11, "women", "league"),
    # Complete, but tournament-shaped: national teams and few matches per player.
    (43, 106, "FIFA World Cup", "International", "2022", 64, 32, "men", "tournament"),
    (72, 107, "Women's World Cup", "International", "2023", 64, 32, "women", "tournament"),
    (1267, 107, "Africa Cup of Nations", "International", "2023", 52, 24, "men", "tournament"),
    (55, 282, "UEFA Euro", "International", "2024", 51, 24, "men", "tournament"),
    (223, 282, "Copa America", "International", "2024", 32, 16, "men", "tournament"),
    # Fragments. They carry a league name and a season but only a slice of the matches —
    # every La Liga season outside 2015/16 is just Barcelona's own games.
    (9, 27, "1. Bundesliga", "Germany", "2015/2016", 34, 18, "men", "league"),
    (9, 281, "1. Bundesliga", "Germany", "2023/2024", 34, 18, "men", "league"),
    (11, 90, "La Liga", "Spain", "2020/2021", 35, 19, "men", "league"),
    (7, 235, "Ligue 1", "France", "2022/2023", 32, 20, "men", "league"),
    (7, 108, "Ligue 1", "France", "2021/2022", 26, 18, "men", "league"),
    (44, 107, "Major League Soccer", "USA", "2023", 6, 7, "men", "league"),
)


router = APIRouter(tags=["coverage"])


@router.get("/coverage", response_model=CoverageResponse)
def coverage(session: Session = Depends(get_session)) -> CoverageResponse:
    # Counted in db.quality rather than here, because `load_feature_frame` decides what
    # enters the comparison pool from the same numbers, and two copies of this arithmetic
    # drifted apart: the one here read a null away side as an extra club.
    facts = season_facts(session)

    player_counts = {
        (r.competition_id, r.sb_season_id): r.n
        for r in session.execute(
            select(
                PlayerSeasonStats.competition_id,
                PlayerSeasonStats.sb_season_id,
                func.count().label("n"),
            ).group_by(PlayerSeasonStats.competition_id, PlayerSeasonStats.sb_season_id)
        )
    }

    names = {
        (r.competition_id, r.sb_season_id): (r.competition, r.country, r.season)
        for r in session.execute(
            select(
                Season.competition_id,
                Season.sb_season_id,
                Competition.name.label("competition"),
                Competition.country.label("country"),
                Season.name.label("season"),
            ).join(Competition, Competition.id == Season.competition_id)
        )
    }

    team_names = {t.id: t.name for t in session.scalars(select(Team))}

    seasons: list[CoverageSeason] = []
    for key, fact in facts.items():
        competition_id, season_id = key
        competition, country, season = names.get(key, ("Unknown", None, str(season_id)))
        seasons.append(
            CoverageSeason(
                competition_id=competition_id,
                competition=competition,
                country=country,
                season_id=season_id,
                season=season,
                matches=fact.matches,
                teams=fact.teams,
                players=player_counts.get(key, 0),
                coverage=round(fact.coverage, 3),
                complete=fact.coverage >= COMPLETE_THRESHOLD,
                # Named when the export covers one club rather than a league, so the
                # reader is told "Bayer Leverkusen, in full" instead of being shown a
                # league that looks nine tenths missing.
                focus_team=team_names.get(fact.focus_team_id) if fact.focus_team_id else None,
            )
        )

    # Most complete first, then biggest — the usable pools lead, fragments sink.
    seasons.sort(key=lambda s: (s.coverage, s.players), reverse=True)

    # A tournament has no round-robin to measure against, so completeness there is a
    # property of the source, not something to recompute from the match count.
    loaded_keys = set(facts)
    catalogue = [
        CatalogueEntry(
            competition_id=cid,
            season_id=sid,
            competition=competition,
            country=country,
            season=season,
            matches=matches,
            teams=teams,
            gender=gender,
            kind=kind,
            complete=(
                True
                if kind == "tournament"
                else matches / (teams * (teams - 1)) >= COMPLETE_THRESHOLD
            ),
            loaded=(cid, sid) in loaded_keys,
        )
        for cid, sid, competition, country, season, matches, teams, gender, kind in _CATALOGUE
    ]

    return CoverageResponse(
        competitions=len({s.competition_id for s in seasons}),
        matches=sum(s.matches for s in seasons),
        players=sum(s.players for s in seasons),
        seasons=seasons,
        catalogue=catalogue,
        catalogue_verified=CATALOGUE_VERIFIED,
    )
