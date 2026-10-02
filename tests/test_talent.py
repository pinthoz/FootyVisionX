"""Unit tests for the Performance Score and the XGBoost position classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from footyvision.ml.features import PER90_FEATURES, position_group
from footyvision.ml.scoring import performance_score, rank_players, score_frame
from footyvision.ml.talent import style_profile, train_position_classifier

_SPECS = {
    "Center Forward": {"xg_per90": 0.6, "goals_per90": 0.5, "shots_per90": 3.0},
    "Center Back": {"tackles_per90": 3.0, "clearances_per90": 5.0, "interceptions_per90": 2.5},
    "Center Midfield": {"progressive_passes_per90": 9.0, "passes_completed_per90": 45.0},
    "Goalkeeper": {"passes_completed_per90": 22.0, "clearances_per90": 2.0},
}


def _synthetic(n_per_class: int = 14, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    pid = 1
    for position, base in _SPECS.items():
        for _ in range(n_per_class):
            feats = {f: 0.0 for f in PER90_FEATURES}
            for k, v in base.items():
                feats[k] = max(0.0, v + rng.normal(0, 0.12 * v))
            rows.append(
                {
                    "player_id": pid,
                    "name": f"P{pid}",
                    "competition": "Test",
                    "competition_id": 1,
                    "sb_season_id": 1,
                    "primary_position": position,
                    "matches_played": 20,
                    "minutes": 1800,
                    "position_group": position_group(position),
                    **feats,
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def test_score_frame_bounds_and_column():
    scored = score_frame(_synthetic())
    assert "performance_score" in scored.columns
    assert scored["performance_score"].between(0, 100).all()


def test_performance_score_breakdown_sums_to_score():
    frame = _synthetic()
    fwd_id = int(frame[frame["position_group"] == "FWD"].iloc[0]["player_id"])
    res = performance_score(frame, fwd_id)
    assert res["position_group"] == "FWD"
    assert 0 <= res["performance_score"] <= 100
    total = sum(b["contribution"] for b in res["breakdown"])
    assert abs(total - res["performance_score"]) < 0.5  # rounding tolerance


def test_rank_players_within_group_sorted():
    ranked = rank_players(_synthetic(), position_group="FWD", top_n=5)
    scores = [r["performance_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(r["position_group"] == "FWD" for r in ranked)


def test_classifier_learns_and_profiles_style():
    frame = _synthetic(n_per_class=16)
    tm = train_position_classifier(frame, seed=42)
    # Well-separated synthetic classes should be classified well.
    assert tm.test_accuracy >= 0.7
    assert set(tm.classes) == {"GK", "DEF", "MID", "FWD"}

    fwd_id = int(frame[frame["position_group"] == "FWD"].iloc[0]["player_id"])
    profile = style_profile(tm, frame, fwd_id)
    assert abs(sum(profile.values()) - 1.0) < 1e-2  # probs are rounded to 3 decimals
    assert max(profile, key=profile.get) == "FWD"


# --- side-agnostic roles ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ("Goalkeeper", "Goalkeeper"),
        ("Left Center Back", "Centre Back"),
        ("Right Center Back", "Centre Back"),
        ("Left Back", "Full Back"),
        # 'Wing Back' contains both 'back' and 'wing'; it must not fall through to either.
        ("Left Wing Back", "Wing Back"),
        ("Center Defensive Midfield", "Defensive Midfield"),
        # 'Attacking Midfield' contains 'midfield'; the specific test has to win.
        ("Center Attacking Midfield", "Attacking Midfield"),
        ("Left Center Midfield", "Central Midfield"),
        ("Left Midfield", "Wide Midfield"),
        ("Right Wing", "Winger"),
        ("Left Center Forward", "Centre Forward"),
        (None, "Unknown"),
    ],
)
def test_position_role_drops_the_side_but_keeps_the_role(position, expected):
    from footyvision.ml.features import position_role

    assert position_role(position) == expected


def test_position_role_is_side_agnostic_by_construction():
    """Mirror-image positions land on the same role: the metrics carry no side."""
    from footyvision.ml.features import position_role

    for left, right in [
        ("Left Back", "Right Back"),
        ("Left Center Back", "Right Center Back"),
        ("Left Wing", "Right Wing"),
        ("Left Wing Back", "Right Wing Back"),
    ]:
        assert position_role(left) == position_role(right)


def test_foot_is_used_only_where_a_side_is_being_predicted():
    """Foot is a laterality signal: useful for the exact position, noise for the rest.

    Measured on the real pool it takes the exact position from 0.392 to 0.489, while
    costing accuracy on the side-agnostic targets (0.916 -> 0.908, 0.720 -> 0.712).
    """
    from footyvision.ml.features import FOOT_FEATURES
    from footyvision.ml.talent import classifier_features

    frame = _synthetic()
    for column in FOOT_FEATURES:
        frame[column] = 0.0

    assert classifier_features(frame, "position_group") == list(PER90_FEATURES)
    assert classifier_features(frame, "position_role") == list(PER90_FEATURES)
    assert classifier_features(frame, "primary_position") == [*PER90_FEATURES, *FOOT_FEATURES]


def test_classifier_features_tolerate_a_frame_without_foot_columns():
    """Hand-built frames (and any pool loaded before the column existed) have no foot."""
    from footyvision.ml.talent import classifier_features

    assert classifier_features(_synthetic(), "primary_position") == list(PER90_FEATURES)


def test_height_is_never_a_model_feature():
    """Height is a scouting attribute; ablation showed it adds nothing to prediction."""
    from footyvision.ml.talent import classifier_features

    frame = _synthetic()
    frame["height_cm"] = 180.0

    for target in ("position_group", "position_role", "primary_position"):
        assert "height_cm" not in classifier_features(frame, target)


# --- calibrated confidence ---------------------------------------------------------------


def test_the_model_carries_a_calibrator():
    """The dashboard prints the confidence as a headline figure, so it has to be honest.

    Uncalibrated, this classifier said 90% where it was right 75% of the time.
    """
    from footyvision.ml.talent import train_position_classifier

    model = train_position_classifier(_synthetic())

    assert model.calibrator is not None


def test_probabilities_come_from_the_calibrator():
    from footyvision.ml.talent import train_position_classifier

    frame = _synthetic()
    model = train_position_classifier(frame)
    x = frame[model.features].to_numpy(dtype=float)[:5]

    expected = model.calibrator.predict_proba(x)
    assert np.allclose(model.predict_proba(x), expected)


def test_shap_still_reads_the_raw_booster():
    """TreeSHAP cannot see through a calibration wrapper, so the booster is kept."""
    from footyvision.ml.talent import shap_importance, train_position_classifier

    frame = _synthetic()
    model = train_position_classifier(frame)

    assert isinstance(model.model, XGBClassifier)
    assert len(shap_importance(model, frame, top_n=3)) == 3


def test_calibration_is_skipped_when_a_class_is_too_small():
    """Isotonic regression needs at least three of every class for its internal folds.

    Tested on the predicate rather than through a contrived frame: forcing a two-class
    pool would trip XGBoost's own multiclass constraint instead, which is a different
    thing and cannot happen with real data (there are always four position groups).
    """
    import numpy as np

    from footyvision.ml.talent import _calibratable

    assert _calibratable(np.array([0, 0, 0, 1, 1, 1]))
    assert not _calibratable(np.array([0, 0, 1, 1, 1]))
    assert not _calibratable(np.array([0, 0, 0, 0]))


def test_classifier_reports_balanced_accuracy_and_per_class_recall():
    """A class the model never predicts has to be visible somewhere.

    Accuracy cannot show it: `Wing Back` scores 0.00 recall on the real pool and the
    headline number does not move, because there are 441 full backs to be right about and
    54 wing backs to miss. `per_class_recall` is the only field that says so, which is why
    it is part of the model rather than something a caller recomputes.
    """
    model = train_position_classifier(_synthetic(), target="position_group")

    assert model.balanced_accuracy is not None
    assert set(model.per_class_recall) == set(model.classes)
    assert all(0.0 <= r <= 1.0 for r in model.per_class_recall.values())
