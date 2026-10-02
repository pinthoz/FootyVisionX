"""Where each player acts: a pitch grid of their actions and a map of their shots.

Everything else in the database is a count — passes, tackles, shots per 90 — and a count
has no location. That turned out to be the ceiling on the position classifiers: ten of the
fourteen held-out wing-backs were predicted as full-backs, because what separates the two
is how high up the pitch they play, and 42% of the exact-position model's errors were
pure left/right swaps, because nothing it saw had a side. StatsBomb records a location for
almost every event, in coordinates oriented to the acting team's attack (x from 0 to 120
towards the opponent's goal, y from 0 to 80 across, the attacker's left at y = 0). So both
the height and the side are in the raw data; the ETL just never kept them.

This reads each match's events straight from the StatsBomb open-data repository as JSON
— far faster than `statsbombpy`'s flattening, and the location is all that is needed —
reduces them to per-player grids and shot lists, and caches that reduction per match under
data/spatial_cache/ so an interrupted run resumes. The per-season result is published to
models/spatial/spatial.json, which the API serves without touching the events again.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "data" / "spatial_cache"
ARTIFACT = ROOT / "models" / "spatial" / "spatial.json"
EVENTS_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{}.json"

PITCH_X, PITCH_Y = 120.0, 80.0
# Ten-by-ten cells: coarse enough that a season of a few hundred actions fills them, fine
# enough to separate a full-back from a wing-back and a left-sided player from a right.
COLS, ROWS = 12, 8


def cell(x: float, y: float) -> int:
    """Index into the row-major COLS x ROWS grid, clamping the touchline itself inside."""
    col = min(int(x / (PITCH_X / COLS)), COLS - 1)
    row = min(int(y / (PITCH_Y / ROWS)), ROWS - 1)
    return max(row, 0) * COLS + max(col, 0)


# Bumped whenever what a match reduces to changes, so a cache written by an older version
# is fetched again instead of being mixed into a newer build.
CACHE_VERSION = 3

# StatsBomb shot outcomes, folded onto the vocabulary the pitch charts use.
_OUTCOMES = {
    "Goal": "goal",
    "Saved": "saved",
    "Saved Off Target": "saved",
    "Saved to Post": "saved",
    "Blocked": "blocked",
    "Off T": "off-target",
    "Wayward": "off-target",
    "Post": "hit-woodwork",
}
_BODY_PARTS = {"Right Foot": "right-foot", "Left Foot": "left-foot", "Head": "head"}
# Restarts are left out of the threat model: a corner or a free kick starts from a place
# the rules chose, not one the team worked the ball into, and counting them would credit
# the taker with threat the foul or the deflection created.
_SET_PIECE_PASSES = {"Corner", "Free Kick", "Kick Off", "Goal Kick"}


def _move(event: dict) -> tuple[list | None, bool] | None:
    """(end location, completed) for an open-play pass or carry; None for anything else.

    A StatsBomb pass with no `outcome` is complete; a carry has no outcome and always ends
    where it ends.
    """
    kind = (event.get("type") or {}).get("name")
    if kind == "Pass":
        detail = event.get("pass") or {}
        if (detail.get("type") or {}).get("name") in _SET_PIECE_PASSES:
            return None
        return detail.get("end_location"), "outcome" not in detail
    if kind == "Carry":
        return (event.get("carry") or {}).get("end_location"), True
    return None


def reduce_events(events: Iterable[dict]) -> dict[int, dict[str, Any]]:
    """Per player in one match: where they acted, where they shot, how they moved the ball.

    Every event with a player and a location counts as an action, pressures included:
    the heatmap is meant to show where someone plays, and a pressing forward's work is
    where they play. Each shot keeps [x, y, xG, outcome, penalty, body part] — the penalty
    flag because a single spot kick is worth 0.76 xG and would otherwise dominate the map
    of a player who took a few.

    For the threat model, `moves_from` counts the open-play passes and carries attempted
    from each cell and `moves` the completed ones as "from-to" cell pairs; `team` lets a
    team's threat in a match be summed from its players.
    """
    out: dict[int, dict[str, Any]] = {}
    for event in events:
        player = (event.get("player") or {}).get("id")
        location = event.get("location")
        if player is None or not location or len(location) < 2:
            continue
        mine = out.setdefault(
            int(player),
            {
                "team": (event.get("team") or {}).get("id"),
                "grid": [0] * (COLS * ROWS),
                "shots": [],
                "moves_from": [0] * (COLS * ROWS),
                "moves": {},
            },
        )
        x, y = float(location[0]), float(location[1])
        here = cell(x, y)
        mine["grid"][here] += 1
        move = _move(event)
        if move is not None:
            end, completed = move
            mine["moves_from"][here] += 1
            if completed and end and len(end) >= 2:
                pair = f"{here}-{cell(float(end[0]), float(end[1]))}"
                mine["moves"][pair] = mine["moves"].get(pair, 0) + 1
        if (event.get("type") or {}).get("name") == "Shot":
            shot = event.get("shot") or {}
            mine["shots"].append(
                [
                    round(x, 1),
                    round(y, 1),
                    round(float(shot.get("statsbomb_xg") or 0.0), 3),
                    _OUTCOMES.get((shot.get("outcome") or {}).get("name"), "other"),
                    int((shot.get("type") or {}).get("name") == "Penalty"),
                    _BODY_PARTS.get((shot.get("body_part") or {}).get("name"), "other"),
                ]
            )
    return out


def _fetch(client: httpx.Client, match_id: int) -> dict[int, dict[str, Any]]:
    cached = CACHE_DIR / f"{match_id}.json"
    if cached.is_file():
        stored = json.loads(cached.read_text(encoding="utf-8"))
        if isinstance(stored, dict) and stored.get("v") == CACHE_VERSION:
            return {int(k): v for k, v in stored["players"].items()}
    last: Exception | None = None
    for _ in range(3):
        try:
            response = client.get(EVENTS_URL.format(match_id))
            response.raise_for_status()
            reduced = reduce_events(response.json())
            cached.write_text(
                json.dumps({"v": CACHE_VERSION, "players": reduced}), encoding="utf-8"
            )
            return reduced
        except Exception as error:  # network hiccups; retried, then reported
            last = error
    raise RuntimeError(f"match {match_id}: {last}")


def build(matches: list[tuple[int, int, int]], workers: int = 8, progress=None) -> dict[str, Any]:
    """Reduce every match and sum the results per player-season.

    `matches` holds (match_id, competition_id, season_id). Keys in the result are
    "player-competition-season", the same identity the distribution endpoint uses, so a
    player who changed league keeps two maps rather than one blended across both.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    seasons: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    with httpx.Client(timeout=60, headers={"Accept-Encoding": "gzip"}) as client:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {
                pool.submit(_fetch, client, mid): (mid, cid, sid) for mid, cid, sid in matches
            }
            for done, future in enumerate(as_completed(pending), 1):
                mid, cid, sid = pending[future]
                try:
                    per_player = future.result()
                except Exception as error:
                    failed.append(str(error))
                    continue
                for pid, data in per_player.items():
                    key = f"{pid}-{cid}-{sid}"
                    season = seasons.setdefault(
                        key,
                        {
                            "player_id": pid,
                            "competition_id": cid,
                            "season_id": sid,
                            "grid": [0] * (COLS * ROWS),
                            "shots": [],
                            "moves_from": [0] * (COLS * ROWS),
                            "moves": {},
                        },
                    )
                    season["grid"] = [
                        a + b for a, b in zip(season["grid"], data["grid"], strict=True)
                    ]
                    season["shots"].extend(data["shots"])
                    season["moves_from"] = [
                        a + b for a, b in zip(season["moves_from"], data["moves_from"], strict=True)
                    ]
                    for pair, count in data["moves"].items():
                        season["moves"][pair] = season["moves"].get(pair, 0) + count
                if progress:
                    progress(done, len(pending))

    threat = expected_threat(seasons.values())
    for season in seasons.values():
        season["actions"] = sum(season["grid"])
        season["moves_completed"] = sum(season["moves"].values())
        season["xt_added"] = round(threat_added(season["moves"], threat), 4)
        # The move pairs were needed to fit and apply the model; published, they would
        # triple the file for a number already computed.
        del season["moves"], season["moves_from"]
    return {
        "built_on": date.today().isoformat(),
        "grid": {"cols": COLS, "rows": ROWS, "pitch": [PITCH_X, PITCH_Y]},
        "xt": [round(v, 5) for v in threat],
        "matches": len(matches) - len(failed),
        "failed": failed,
        "players": seasons,
    }


def expected_threat(records: Iterable[dict], iterations: int = 50) -> list[float]:
    """Karun Singh's Expected Threat: how likely possession in each cell ends in a goal.

    From every record's shots, attempted moves and completed move pairs:

        xT(z) = s(z) g(z) + m(z) * sum over z' of T(z -> z') xT(z')

    s and m are the shares of a cell's actions that are shots and moves, g the share of its
    non-penalty shots that score, and T the chance a move from z ends, completed, in z'. A
    failed move carries no onward value, which is how losing the ball is priced. Solved by
    iteration from zero: each pass extends the horizon by one more move, and the values
    settle well before fifty.
    """
    n = COLS * ROWS
    shots, goals, moves = [0] * n, [0] * n, [0] * n
    completed: dict[tuple[int, int], int] = {}
    for record in records:
        for x, y, _xg, outcome, penalty, _body in record["shots"]:
            if penalty:
                continue
            z = cell(x, y)
            shots[z] += 1
            goals[z] += outcome == "goal"
        for z, count in enumerate(record["moves_from"]):
            moves[z] += count
        for pair, count in record["moves"].items():
            a, b = (int(v) for v in pair.split("-"))
            completed[(a, b)] = completed.get((a, b), 0) + count

    actions = [s + m for s, m in zip(shots, moves, strict=True)]
    score = [g / a if a else 0.0 for g, a in zip(goals, actions, strict=True)]
    move_share = [m / a if a else 0.0 for m, a in zip(moves, actions, strict=True)]
    transitions: dict[int, list[tuple[int, float]]] = {}
    for (a, b), count in completed.items():
        if moves[a]:
            transitions.setdefault(a, []).append((b, count / moves[a]))

    threat = [0.0] * n
    for _ in range(iterations):
        threat = [
            score[z] + move_share[z] * sum(p * threat[b] for b, p in transitions.get(z, ()))
            for z in range(n)
        ]
    return threat


def threat_added(moves: dict[str, int], threat: list[float]) -> float:
    """Total xT a set of completed moves created: the value where each ended, less where
    it began. A backward pass is negative, as it should be."""
    total = 0.0
    for pair, count in moves.items():
        a, b = (int(v) for v in pair.split("-"))
        total += count * (threat[b] - threat[a])
    return total


def save(payload: dict[str, Any], path: Path = ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return path


def load(path: Path = ARTIFACT) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("could not read %s", path)
        return None


def features(grid: list[int]) -> dict[str, float]:
    """Summaries of a grid a classifier can use: how high, how wide, which side.

    Averages over cell centres, weighted by how many actions each cell holds. `mean_y`
    carries the side (below 40 is the left from the player's own attacking point of view),
    which no other feature in the project does.
    """
    total = sum(grid)
    if not total:
        return {}
    cw, ch = PITCH_X / COLS, PITCH_Y / ROWS
    xs = [((i % COLS) + 0.5) * cw for i in range(COLS * ROWS)]
    ys = [((i // COLS) + 0.5) * ch for i in range(COLS * ROWS)]
    share = lambda test: sum(n for n, x, y in zip(grid, xs, ys, strict=True) if test(x, y)) / total  # noqa: E731
    mean_x = sum(n * x for n, x in zip(grid, xs, strict=True)) / total
    mean_y = sum(n * y for n, y in zip(grid, ys, strict=True)) / total
    return {
        "zone_mean_x": mean_x,
        "zone_mean_y": mean_y,
        "zone_spread_x": (sum(n * (x - mean_x) ** 2 for n, x in zip(grid, xs, strict=True)) / total)
        ** 0.5,
        "zone_spread_y": (sum(n * (y - mean_y) ** 2 for n, y in zip(grid, ys, strict=True)) / total)
        ** 0.5,
        "zone_final_third": share(lambda x, y: x >= 80),
        "zone_own_third": share(lambda x, y: x < 40),
        "zone_left": share(lambda x, y: y < PITCH_Y / 3),
        "zone_right": share(lambda x, y: y >= 2 * PITCH_Y / 3),
        "zone_wide": share(lambda x, y: y < 20 or y >= 60),
    }


ZONE_FEATURES = (
    "zone_final_third",
    "zone_left",
    "zone_mean_x",
    "zone_mean_y",
    "zone_own_third",
    "zone_right",
    "zone_spread_x",
    "zone_spread_y",
    "zone_wide",
)
# The ones that say nothing about which side: height, spread, and how much of the play is
# out wide on either flank.
SIDELESS_ZONE_FEATURES = (
    "zone_final_third",
    "zone_mean_x",
    "zone_own_third",
    "zone_spread_x",
    "zone_spread_y",
    "zone_wide",
)


def attach_features(frame, payload: dict[str, Any] | None = None):
    """A copy of a feature frame with the zone features of each player-season added.

    Missing maps become NaN, which the classifiers treat as unknown rather than as zero.
    Without a published artifact the frame comes back unchanged, and the classifiers then
    use the per-90 features alone, exactly as before this existed.
    """
    needed = {"player_id", "competition_id", "sb_season_id"}
    if frame.empty or not needed.issubset(frame.columns):
        return frame
    payload = payload if payload is not None else load()
    if not payload:
        return frame
    keys = (
        frame["player_id"].astype(int).astype(str)
        + "-"
        + frame["competition_id"].astype(int).astype(str)
        + "-"
        + frame["sb_season_id"].astype(int).astype(str)
    )
    summaries = {
        key: features(season["grid"]) for key, season in payload.get("players", {}).items()
    }
    out = frame.copy()
    for column in ZONE_FEATURES:
        out[column] = [summaries.get(key, {}).get(column, float("nan")) for key in keys]
    return out
