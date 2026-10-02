"""Pitch maps: the reduction from raw events, and the features read off it.

The point of these features is that they carry what the per-90 counts cannot — how high a
player plays and on which side — so the orientation is the thing most worth pinning down.
StatsBomb gives every event from the acting team's attacking point of view: x runs from
0 to 120 towards the opponent's goal, y from 0 to 80 with the attacker's left at y = 0.
"""

from __future__ import annotations

import pytest

from footyvision.etl import spatial


def _event(player, x, y, kind="Pass", **extra):
    return {"player": {"id": player}, "location": [x, y], "type": {"name": kind}, **extra}


def test_cells_are_row_major_and_the_touchline_stays_on_the_pitch():
    assert spatial.cell(0, 0) == 0
    assert spatial.cell(119.9, 0) == spatial.COLS - 1
    assert spatial.cell(0, 79.9) == (spatial.ROWS - 1) * spatial.COLS
    # Exactly on the far touchline or goal line is still the last cell, not off the grid.
    assert spatial.cell(120, 80) == spatial.COLS * spatial.ROWS - 1


def test_reduction_counts_located_actions_per_player_and_keeps_shots():
    events = [
        _event(1, 10, 10),
        _event(1, 11, 12, kind="Pressure"),
        _event(2, 100, 40),
        {"type": {"name": "Starting XI"}},  # no player, no location: ignored
        _event(
            2,
            108,
            40,
            kind="Shot",
            shot={
                "statsbomb_xg": 0.78351,
                "outcome": {"name": "Goal"},
                "type": {"name": "Penalty"},
                "body_part": {"name": "Right Foot"},
            },
        ),
    ]
    out = spatial.reduce_events(events)
    assert sum(out[1]["grid"]) == 2 and out[1]["shots"] == []
    assert sum(out[2]["grid"]) == 2
    assert out[2]["shots"] == [[108.0, 40.0, 0.784, "goal", 1, "right-foot"]]


def test_unknown_shot_outcomes_fold_to_other():
    out = spatial.reduce_events(
        [_event(3, 90, 30, kind="Shot", shot={"statsbomb_xg": 0.1, "outcome": {"name": "?"}})]
    )
    assert out[3]["shots"][0][3] == "other"


def test_features_carry_the_side_the_counts_cannot():
    """A player who only ever acts on his own left must read as left, high as high."""
    grid = [0] * (spatial.COLS * spatial.ROWS)
    grid[spatial.cell(100, 5)] = 50  # attacking third, attacker's left touchline
    feats = spatial.features(grid)
    assert feats["zone_mean_y"] < 10
    assert feats["zone_left"] == 1.0 and feats["zone_right"] == 0.0
    assert feats["zone_final_third"] == 1.0 and feats["zone_own_third"] == 0.0
    assert feats["zone_wide"] == 1.0


def test_an_empty_grid_has_no_features_rather_than_divide_by_zero():
    assert spatial.features([0] * (spatial.COLS * spatial.ROWS)) == {}


def test_a_cache_from_an_older_reduction_is_fetched_again(tmp_path, monkeypatch):
    """An older cache lacks shot outcomes; reusing it would mix two formats in one build."""
    import json

    monkeypatch.setattr(spatial, "CACHE_DIR", tmp_path)
    (tmp_path / "7.json").write_text(json.dumps({"1": {"grid": [], "shots": []}}))

    class _Client:
        def get(self, url):
            class _Response:
                def raise_for_status(self):
                    pass

                def json(self):
                    return [_event(5, 60, 40)]

            return _Response()

    reduced = spatial._fetch(_Client(), 7)
    assert 5 in reduced and 1 not in reduced
    assert json.loads((tmp_path / "7.json").read_text())["v"] == spatial.CACHE_VERSION


@pytest.mark.parametrize("sb_y,expected_side", [(5, "left"), (75, "right")])
def test_campos_orientation_mirrors_statsbomb(sb_y, expected_side):
    """The dashboard converts with y_campos = 100 - y_sb / 80 * 100, because Campos puts
    the attacker's left at 100 and StatsBomb at 0. Pinned here so the conversion and its
    reason live next to the data they describe."""
    campos_y = 100 - sb_y / 80 * 100
    assert (campos_y > 50) == (expected_side == "left")


def test_open_play_moves_are_counted_and_set_pieces_are_not():
    events = [
        {**_event(1, 50, 40), "pass": {}},  # no end location: attempted, not placed
        {**_event(1, 50, 40), "pass": {"end_location": [70, 40]}},  # completed
        {
            **_event(1, 50, 40),
            "pass": {"end_location": [70, 40], "outcome": {"name": "Incomplete"}},
        },
        {**_event(1, 50, 40), "pass": {"end_location": [110, 40], "type": {"name": "Corner"}}},
        {**_event(1, 60, 40, kind="Carry"), "carry": {"end_location": [75, 40]}},
    ]
    mine = spatial.reduce_events(events)[1]
    here = spatial.cell(50, 40)
    assert mine["moves_from"][here] == 3  # the corner is not a move at all
    assert mine["moves"][f"{here}-{spatial.cell(70, 40)}"] == 1
    assert mine["moves"][f"{spatial.cell(60, 40)}-{spatial.cell(75, 40)}"] == 1


def _record(shots=(), moves_from=None, moves=None):
    return {
        "shots": list(shots),
        "moves_from": moves_from or [0] * (spatial.COLS * spatial.ROWS),
        "moves": moves or {},
    }


def test_threat_rises_where_possession_leads_to_goals():
    """A cell that scores is worth its conversion; a cell that only feeds it is worth less,
    discounted by how often the ball actually gets there."""
    box, midfield = spatial.cell(110, 40), spatial.cell(60, 40)
    moves_from = [0] * (spatial.COLS * spatial.ROWS)
    moves_from[midfield] = 10
    record = _record(
        shots=[[110, 40, 0.3, "goal", 0, "right-foot"], [110, 40, 0.3, "saved", 0, "head"]],
        moves_from=moves_from,
        moves={f"{midfield}-{box}": 5},  # half the attempted moves reach the box
    )
    threat = spatial.expected_threat([record])
    assert threat[box] == pytest.approx(0.5)  # 1 goal from 2 actions
    assert threat[midfield] == pytest.approx(0.25)  # half the moves arrive, at 0.5 each
    assert threat[spatial.cell(5, 5)] == 0.0  # nothing ever happened there


def test_penalties_are_left_out_of_the_threat_model():
    spot = spatial.cell(108, 40)
    threat = spatial.expected_threat([_record(shots=[[108, 40, 0.78, "goal", 1, "right-foot"]])])
    assert threat[spot] == 0.0


def test_a_backward_move_destroys_threat():
    threat = [0.0] * (spatial.COLS * spatial.ROWS)
    threat[5], threat[2] = 0.1, 0.02
    assert spatial.threat_added({"5-2": 3}, threat) == pytest.approx(-0.24)
    assert spatial.threat_added({"2-5": 1}, threat) == pytest.approx(0.08)
