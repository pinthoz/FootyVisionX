"""Team attack and defence ratings, the one thing this project had no notion of.

Every other endpoint describes a player, and a player's per-90 numbers are shaped by the
side around him: a striker in a weak attack sees fewer chances than one in a strong one,
and nothing here could previously say which he was. These are Poisson coefficients on a
log scale, fitted across a whole season — 0 is an average side, a positive attack scores
more than average and a *negative* defence concedes less.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from footyvision.api.schemas import ExpectedTable, TeamStrengthOut, TeamStrengthResponse
from footyvision.db.base import get_session
from footyvision.db.models import Competition, Team
from footyvision.ml import expected_table, precompute

# No import of footyvision.ml.matches, and so none of scikit-learn: it was the last thing
# pulling that 65MB into the API. The ratings only move when a season is loaded, so they
# are fitted by `footyvision precompute` — one fit per competition, because a rating is
# relative to the field it was measured in and slicing a global fit answers something else.
router = APIRouter(tags=["teams"])

_CACHE: dict[str, TeamStrengthResponse] = {}


@router.get("/teams/strength", response_model=TeamStrengthResponse)
def strength(
    session: Session = Depends(get_session),
    competition_id: int | None = Query(None, description="Restrict to one competition."),
) -> TeamStrengthResponse:
    """Attack and defence ratings per team, strongest attack first."""
    key = str(competition_id)
    if key in _CACHE:
        return _CACHE[key]

    payload = (precompute.load() or {}).get("teams") or {}
    scope = payload.get(str(competition_id))
    # Ratings fitted against a different database are worse than none: they name real teams
    # with real-looking numbers and nothing about the response says they are not this
    # database's. Checked on a plain fixture count, taken the same way at both ends.
    if scope is not None and payload.get("db_matches") != precompute.count_matches(session):
        scope = None
    if scope is None:
        # Nothing published for this scope: either the file was never built, or a
        # competition was asked for that did not exist when it was. Fitting here pulls in
        # scikit-learn, which is what the published file exists to avoid, so a deployment
        # taking this path is telling you the file is missing rather than merely slow.
        return _fit(session, competition_id)

    names = {t.id: t.name for t in session.query(Team).all()}
    competitions = {c.id: c.name for c in session.query(Competition).all()}
    played_in = {int(k): v for k, v in payload.get("played_in", {}).items()}

    response = TeamStrengthResponse(
        home_advantage=scope["home_advantage"],
        matches=scope["matches"],
        teams=[
            TeamStrengthOut(
                team_id=rating["team_id"],
                name=names.get(rating["team_id"], str(rating["team_id"])),
                competition=competitions.get(played_in.get(rating["team_id"], -1)),
                attack=rating["attack"],
                defence=rating["defence"],
                matches=rating["matches"],
            )
            for rating in sorted(scope["teams"], key=lambda r: -r["attack"])
        ],
    )
    _CACHE[key] = response
    return response


@router.get("/teams/expected-table", response_model=ExpectedTable)
def expected_table_view(
    competition_id: int | None = Query(None, description="Restrict to one competition."),
) -> ExpectedTable:
    """Actual against expected points for every team, from every shot's xG.

    Read from models/spatial/expected_table.json, which `footyvision spatial` writes from
    the raw events; the API holds neither the events nor the means to rebuild it.
    """
    if "table" not in _CACHE_TABLE:
        payload = expected_table.load()
        if payload is None:
            raise HTTPException(
                status_code=503,
                detail="No expected table published. Run `footyvision spatial` and deploy "
                "models/spatial/expected_table.json.",
            )
        _CACHE_TABLE["table"] = payload
    payload = _CACHE_TABLE["table"]
    seasons = [
        s
        for s in payload["seasons"].values()
        if competition_id is None or s["competition_id"] == competition_id
    ]
    seasons.sort(key=lambda s: s.get("competition") or "")
    return ExpectedTable(built_on=payload["built_on"], matches=payload["matches"], seasons=seasons)


_CACHE_TABLE: dict[str, dict] = {}


def _fit(session: Session, competition_id: int | None) -> TeamStrengthResponse:
    """Fit the Poisson model in-process, for a working copy with no published ratings.

    Only possible where the `train` extra is installed. A deployment does not have it,
    by design, and says so rather than failing inside the import.
    """
    try:
        from footyvision.ml.matches import load_matches, team_strengths
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "No published team ratings for this database, and the training libraries "
                "are not installed here. Run `footyvision precompute` where they are and "
                "deploy models/talent/predictions.json."
            ),
        ) from exc

    matches = load_matches(session)
    if competition_id is not None:
        matches = matches[matches["competition_id"] == competition_id]
    if matches.empty:
        return TeamStrengthResponse(home_advantage=0.0, matches=0, teams=[])

    info: dict[str, float] = {}
    ratings = team_strengths(matches, info)
    names = {t.id: t.name for t in session.query(Team).all()}
    competitions = {c.id: c.name for c in session.query(Competition).all()}
    played_in = {
        int(row.home_team_id): int(row.competition_id) for row in matches.itertuples(index=False)
    } | {int(row.away_team_id): int(row.competition_id) for row in matches.itertuples(index=False)}
    return TeamStrengthResponse(
        home_advantage=round(info.get("home_advantage", 0.0), 4),
        matches=int(len(matches)),
        teams=[
            TeamStrengthOut(
                team_id=r.team_id,
                name=names.get(r.team_id, str(r.team_id)),
                competition=competitions.get(played_in.get(r.team_id, -1)),
                attack=round(r.attack, 4),
                defence=round(r.defence, 4),
                matches=r.matches,
            )
            for r in sorted(ratings, key=lambda r: -r.attack)
        ],
    )
