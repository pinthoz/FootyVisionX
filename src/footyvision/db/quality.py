"""How complete each loaded competition-season actually is.

The StatsBomb open dataset mixes whole league seasons with single-club ones carrying the
same competition and season names. The Bundesliga 2015/16 here is Bayer Leverkusen: their
34 matches, and two apiece for the seventeen clubs they played. La Liga 2018/19 is
Barcelona in the same way. A match count alone cannot tell those from a genuinely
truncated season, so each is measured against the fixture list its own shape implies.

Which baseline is used matters more than it sounds. Measured against a full league, that
Bundesliga reads as 11% complete — and it is not incomplete at all: nineteen Leverkusen
players have full seasons in it, better sampled than the pool median, one of them
thirty-three appearances. Treating it as a fragment would have deleted them from every
percentile, similarity and model in the project, and reported it to the user as a tenth
of a league we hold in full.

This lives here rather than in the coverage router because two very different callers need
it: the router reports it, and `load_feature_frame` uses it to keep genuine fragments out
of the pool that percentiles, similarity and every model are computed over.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select, union
from sqlalchemy.orm import Session

from footyvision.db.models import Match

# Below this share of the fixture list its shape implies, a season is a fragment.
COMPLETE_THRESHOLD = 0.9


@dataclass(frozen=True)
class SeasonFacts:
    matches: int
    teams: int
    # The club a single-club dataset is built around: the one side present in every match.
    # None for a league season, where no team is.
    focus_team_id: int | None
    coverage: float


def _facts(session: Session) -> dict[tuple[int, int], SeasonFacts]:
    match_counts = {
        (r.competition_id, r.sb_season_id): r.n
        for r in session.execute(
            select(
                Match.competition_id,
                Match.sb_season_id,
                func.count().label("n"),
            ).group_by(Match.competition_id, Match.sb_season_id)
        )
    }

    # One row per team per match, so a team's appearances can be counted. Null sides are
    # excluded rather than counted: twenty Ligue 1 matches have no away team recorded, and
    # letting the null through adds a phantom twenty-first club, which inflates the
    # expected fixture list and turns a complete season into a fragment.
    sides = union(
        select(
            Match.competition_id.label("cid"),
            Match.sb_season_id.label("sid"),
            Match.home_team_id.label("tid"),
            Match.id.label("mid"),
        ).where(Match.home_team_id.is_not(None)),
        select(Match.competition_id, Match.sb_season_id, Match.away_team_id, Match.id).where(
            Match.away_team_id.is_not(None)
        ),
    ).subquery()

    appearances: dict[tuple[int, int], dict[int, int]] = {}
    for row in session.execute(
        select(sides.c.cid, sides.c.sid, sides.c.tid, func.count(sides.c.mid).label("n")).group_by(
            sides.c.cid, sides.c.sid, sides.c.tid
        )
    ):
        appearances.setdefault((row.cid, row.sid), {})[row.tid] = row.n

    out: dict[tuple[int, int], SeasonFacts] = {}
    for key, matches in match_counts.items():
        per_team = appearances.get(key, {})
        teams = len(per_team)
        # A club dataset is exactly the case where one side played every match loaded.
        # An equality rather than a threshold: it is a statement about the shape of the
        # export, not a judgement about how much of it is here, and a league season can
        # never satisfy it above two teams.
        focus = next((tid for tid, n in per_team.items() if n == matches), None)
        if teams <= 2:
            focus = None

        if focus is not None:
            # That club's own fixture list: home and away against everyone else.
            expected = 2 * (teams - 1)
        else:
            expected = teams * (teams - 1) if teams > 1 else 0

        # Capped at 1.0: play-off and final-stage formats exceed a plain round-robin, and
        # a season being *longer* than the baseline says nothing about it being incomplete.
        coverage = min(matches / expected, 1.0) if expected else 0.0
        out[key] = SeasonFacts(matches, teams, focus, coverage)
    return out


def season_facts(session: Session) -> dict[tuple[int, int], SeasonFacts]:
    """Match count, team count, focus club and coverage, per (competition_id, season_id)."""
    return _facts(session)


def season_coverage(session: Session) -> dict[tuple[int, int], float]:
    """Share of the implied fixture list that is loaded, per (competition_id, season_id)."""
    return {key: facts.coverage for key, facts in _facts(session).items()}


def fragment_seasons(
    session: Session, threshold: float = COMPLETE_THRESHOLD
) -> set[tuple[int, int]]:
    """Competition-seasons too incomplete to compare players across.

    A season with no matches recorded at all is *not* listed. Coverage cannot be judged
    without fixtures, and excluding on an unknown would empty the pool of any instance
    holding aggregates without match rows — including every test fixture.
    """
    return {key for key, ratio in season_coverage(session).items() if ratio < threshold}
