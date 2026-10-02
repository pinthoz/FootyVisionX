"""XGBoost position classifier — a genuinely supervised task with REAL labels.

We predict a player's position group from their per-90 style. This showcases the ML +
explainability (SHAP) stack honestly (no fabricated target) and yields two useful signals:
a per-player *style profile* (how FWD/MID/DEF-like they play) and *role mismatches*
(players whose stats look like a different position than they're listed in).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from footyvision.etl.spatial import SIDELESS_ZONE_FEATURES, ZONE_FEATURES
from footyvision.ml.features import FOOT_FEATURES, PER90_FEATURES
from footyvision.ml.similarity import _target_index

# Directory containing pre-trained model artifacts.
MODELS_DIR = Path(__file__).resolve().parents[3] / "models" / "talent"

# Position groups the classifier learns (Unknown excluded; tiny classes handled at fit).
_TRAIN_GROUPS = ("GK", "DEF", "MID", "FWD")


def classifier_features(frame: pd.DataFrame, target: str = "position_group") -> list[str]:
    """Per-90 style metrics, plus preferred foot only where a side is being predicted.

    Foot is the one feature that encodes a side, and the measurements say to use it
    exactly there and nowhere else:

    | target             | per-90 only | + foot |
    |--------------------|-------------|--------|
    | position_group (4) | 0.916       | 0.908  |
    | position_role (10) | 0.720       | 0.712  |
    | primary_position   | 0.392       | 0.489  |

    On the side-agnostic targets it is noise and costs accuracy; on the exact position it
    is worth ten points and cuts pure left/right confusions from 134 errors to 96. Height
    was ablated the same way and added nothing anywhere, so it is never modelled.

    Frames built by hand in tests have no foot columns, so they are added only if present.

    Where the player acts — the zone features from `etl.spatial` — was measured the same
    way (scripts/eval_spatial_features.py, five seeds, mean accuracy / balanced):

    | target             | per-90        | + all zones   | + sideless zones |
    |--------------------|---------------|---------------|------------------|
    | position_group (4) | 0.900 / 0.914 | 0.923 / 0.932 | 0.926 / 0.935    |
    | position_role (10) | 0.748 / 0.597 | 0.808 / 0.662 | 0.806 / 0.657    |
    | primary_position   | 0.467 / 0.321 | 0.703 / 0.521 | —                |

    The same rule as foot: the zones that carry a side go only where a side is predicted.
    On the exact position they are the largest gain in the project — left/right swaps fall
    from 44% of its errors to 24% — and on the side-agnostic targets they add nothing the
    sideless ones do not. Added only when the frame has them (`spatial.attach_features`).
    """
    features = list(PER90_FEATURES)
    zones = ZONE_FEATURES if target == "primary_position" else SIDELESS_ZONE_FEATURES
    features += [c for c in zones if c in frame.columns]
    if target == "primary_position":
        features += [c for c in FOOT_FEATURES if c in frame.columns]
    return features


@dataclass
class TalentModel:
    """A fitted classifier and the calibrator that makes its confidence honest.

    Both are kept because they are needed for different things. TreeSHAP explains the raw
    booster and cannot see through a calibration wrapper, while every probability shown to
    a user comes from the calibrator — the dashboard prints "Plays like FWD (93%)" as a
    headline figure, and uncalibrated the model said 90% where it was right 75% of the time.
    """

    model: XGBClassifier
    classes: list[str]
    features: list[str]
    test_accuracy: float
    n_train: int
    n_test: int
    target: str = "position_group"
    calibrator: CalibratedClassifierCV | None = None
    # Share of held-out players whose true class is among the model's three best guesses.
    # With 21 exact positions a single-label score understates a model that is useful
    # anyway: a scout accepts "one of these three" and works from there.
    top3_accuracy: float | None = None
    # Mean recall across classes, each weighted equally regardless of size. Accuracy is
    # dominated by the big classes and these targets are lopsided — 5:1 across the four
    # groups, 116:1 across the 23 exact positions — so the two numbers come apart badly:
    # the role model is 74% accurate and 59% balanced, and the exact model 46% against 30%.
    balanced_accuracy: float | None = None
    # Recall per class, which is where a dead class shows up and nothing else reveals it.
    # `Wing Back` scores 0.00 here: ten of the fourteen held-out wing backs are predicted
    # as full backs, and the headline accuracy never moves, because there are 441 full
    # backs to be right about and 54 wing backs to miss.
    per_class_recall: dict[str, float] = field(default_factory=dict)

    def predict_proba(self, x):
        """Calibrated probabilities when available, raw ones otherwise."""
        return (self.calibrator or self.model).predict_proba(x)


# Below this many player-seasons a class is dropped rather than modelled. Ten is chosen
# so that a 25% test split leaves at least a couple of examples to score against; the two
# classes this removes from the 23-position target have three and two rows between them.
MIN_CLASS_SIZE = 10


def train_position_classifier(
    frame: pd.DataFrame,
    test_size: float = 0.25,
    seed: int = 42,
    target: str = "position_group",
) -> TalentModel:
    """Fit the position classifier at whatever granularity `target` names.

    `position_group` gives the four broad groups; `position_role` the ten side-agnostic
    roles; `primary_position` the exact StatsBomb positions, which /players/{id}/score
    ships as a three-name shortlist and deliberately not as a single label — its top pick
    is right 71% of the time, the three together 88%. (Before the pitch-map features it was
    47%, with 42% of its misses pure left/right swaps: nothing it saw had a side.)

    Read `balanced_accuracy` beside `test_accuracy` on the finer two. The targets are
    lopsided — 5:1 across the groups, 116:1 across the exact positions, where two classes
    have two examples between them — and accuracy counts players, so the crowded classes
    decide it. The role model is 75% accurate and 60% balanced.

    Class weights were measured and are deliberately not used. Balanced sample weights move
    `Wing Back` from 0.00 recall to 0.07 and cost 1.5 points of accuracy, which is not a
    rescued class, it is a rounding error bought at a price. Ten of the fourteen held-out
    wing backs are predicted as full backs, and that is the model being right about the
    football: the two roles have the same per-90 shape, and what separates them is whether
    the side plays a back three — a property of the team, which none of these features
    carry. Reweighting cannot conjure a distinction the features do not contain.
    """
    data = frame[frame[target] != "Unknown"].copy()
    if target == "position_group":
        data = data[data[target].isin(_TRAIN_GROUPS)]
    # A class needs enough examples to be learned *and* to be measured. Two is the floor
    # for splitting at all, and it is not enough for either: `Right Attacking Midfield`
    # has two player-seasons in the whole database, landed zero of them in the held-out
    # split, and was still reported at 0.00 recall — a score for a class nobody tested.
    counts_all = data[target].value_counts()
    data = data[data[target].isin(counts_all[counts_all >= MIN_CLASS_SIZE].index)]

    classes = sorted(data[target].unique())
    code = {c: i for i, c in enumerate(classes)}
    features = classifier_features(data, target)
    x = data[features].to_numpy(dtype=float)
    y = data[target].map(code).to_numpy()

    # Stratify only if every class has at least two samples.
    counts = data[target].value_counts()
    stratify = y if counts.min() >= 2 else None
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        num_class=len(classes),
        random_state=seed,
    )
    model.fit(x_tr, y_tr)

    # Isotonic, chosen by measurement rather than by default: on this pool it cuts the
    # expected calibration error from 0.052 to 0.006 while costing 0.2 accuracy points,
    # where Platt scaling barely moved it (0.051) and became underconfident instead.
    # Fitted on the training split only, so the held-out score below stays honest.
    calibrator: CalibratedClassifierCV | None = None
    if _calibratable(y_tr):
        calibrator = CalibratedClassifierCV(_fresh_like(model, len(classes), seed), cv=3)
        calibrator.fit(x_tr, y_tr)

    scorer = calibrator or model
    predicted = scorer.predict(x_te)
    acc = float(accuracy_score(y_te, predicted))
    top3 = _top_k_accuracy(scorer.predict_proba(x_te), y_te, k=3)
    # Balanced accuracy and per-class recall are computed over every class the model knows,
    # including any absent from the held-out split, so a class cannot disappear from the
    # report by being too rare to be sampled.
    labels = list(range(len(classes)))
    balanced = float(balanced_accuracy_score(y_te, predicted))
    recalls = recall_score(y_te, predicted, labels=labels, average=None, zero_division=0)
    per_class = {name: round(float(r), 3) for name, r in zip(classes, recalls, strict=True)}
    return TalentModel(
        model,
        classes,
        features,
        acc,
        len(x_tr),
        len(x_te),
        target,
        calibrator,
        top3,
        balanced,
        per_class,
    )


def _top_k_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int = 3) -> float:
    """Whether the true class is among the k highest-scoring ones.

    Returns 1.0 when there are k classes or fewer, because the question is then vacuous.
    """
    if proba.shape[1] <= k:
        return 1.0
    best = np.argsort(-proba, axis=1)[:, :k]
    return float(np.mean([truth in row for truth, row in zip(y_true, best, strict=True)]))


def _calibratable(y_tr) -> bool:
    """Isotonic regression needs enough of every class to fit its 3 internal folds."""
    counts = np.bincount(y_tr)
    return bool(len(counts) > 1 and counts.min() >= 3)


def _fresh_like(model: XGBClassifier, n_classes: int, seed: int) -> XGBClassifier:
    """An unfitted twin. CalibratedClassifierCV refits internally and will not take one
    that has already been fitted."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        num_class=n_classes,
        random_state=seed,
    )


def style_profile(tm: TalentModel, frame: pd.DataFrame, player_id: int) -> dict[str, float] | None:
    """Predicted probability the player belongs to each position group ('style fingerprint')."""
    idx = _target_index(frame, player_id)
    if idx is None:
        return None
    x = frame.loc[[idx], tm.features].to_numpy(dtype=float)
    proba = tm.predict_proba(x)[0]
    return {cls: round(float(p), 3) for cls, p in zip(tm.classes, proba, strict=False)}


def shap_importance(tm: TalentModel, frame: pd.DataFrame, top_n: int = 10) -> list[dict[str, Any]]:
    """Global mean |SHAP| importance per feature (which metrics define position)."""
    import shap

    # Subsample to at most 150 rows: global SHAP rankings converge with ~100 samples,
    # while computing TreeExplainer over thousands of samples in multi-class consumes
    # substantial memory and CPU, risking cold-start OOM on constrained runtimes (e.g. 512MB).
    data = frame
    if len(frame) > 150:
        data = frame.sample(150, random_state=42)

    x = data[tm.features].to_numpy(dtype=float)
    values = np.abs(np.array(shap.TreeExplainer(tm.model).shap_values(x)))
    # Reduce over every axis except the feature axis, wherever it lands.
    feat_axis = next(ax for ax, size in enumerate(values.shape) if size == len(tm.features))
    other = tuple(ax for ax in range(values.ndim) if ax != feat_axis)
    mean_abs = values.mean(axis=other)
    ranked = sorted(zip(tm.features, mean_abs, strict=False), key=lambda kv: kv[1], reverse=True)
    return [{"feature": f, "mean_abs_shap": round(float(v), 4)} for f, v in ranked[:top_n]]


def role_mismatches(
    tm: TalentModel, frame: pd.DataFrame, min_prob: float = 0.6, top_n: int = 15
) -> list[dict[str, Any]]:
    """Players whose predicted position differs from their listed one (confidently)."""
    data = frame[frame["position_group"].isin(tm.classes)].copy()
    proba = tm.predict_proba(data[tm.features].to_numpy(dtype=float))
    pred_idx = proba.argmax(axis=1)
    out: list[dict[str, Any]] = []
    for row_pos, (_, r) in enumerate(data.iterrows()):
        pred = tm.classes[pred_idx[row_pos]]
        conf = float(proba[row_pos, pred_idx[row_pos]])
        if pred != r["position_group"] and conf >= min_prob:
            out.append(
                {
                    "player_id": int(r["player_id"]),
                    "name": r["name"],
                    "listed": r["position_group"],
                    "plays_like": pred,
                    "confidence": round(conf, 3),
                }
            )
    out.sort(key=lambda m: m["confidence"], reverse=True)
    return out[:top_n]


# Trained lazily, persisted to disk, and cached in memory.
_CACHE: dict[str, TalentModel] = {}
_SHAP_CACHE: dict[str, list[dict[str, Any]]] = {}


def data_counts(frame: pd.DataFrame, target: str) -> dict[str, int]:
    """How many player-seasons each class of `target` has in the pool as it stands."""
    if target not in frame.columns:
        return {}
    return frame[frame[target] != "Unknown"][target].value_counts().to_dict()


def _cached(key: str, filename: str, target: str, frame: pd.DataFrame) -> TalentModel:
    """Serve a model from memory, then from disk, and only then pay to fit it.

    The disk copy is discarded when it no longer matches the data in front of it. Checking
    only that the features still exist is not enough: loading a new competition leaves the
    columns identical while changing every percentile the model was fitted against, and a
    stale model would then be served indefinitely with no visible symptom.
    """
    if key in _CACHE:
        return _CACHE[key]

    eligible = int((frame[target] != "Unknown").sum()) if target in frame.columns else len(frame)
    disk_path = MODELS_DIR / filename
    if disk_path.is_file():
        try:
            loaded = joblib.load(disk_path)
            # A pickled dataclass restores the attributes it was saved with, not the ones
            # this class now declares, so an artifact written before a field existed comes
            # back missing it and fails at the point of use rather than here. Checking the
            # full field set retrains instead, and covers every future field for free.
            current_fields = {f.name for f in fields(TalentModel)}
            # A class the current threshold would have dropped means the artifact was fitted
            # under a different policy, which neither the field set nor the pool size can
            # see: both stayed identical when MIN_CLASS_SIZE went from 2 to 10, and a
            # 23-class model was served by a build that only knows about 21.
            # The features it was fitted on must be the ones this code would choose now. A
            # model from before the zone features existed still finds all *its* columns in
            # the frame, so a subset check alone would keep serving it.
            same_features = list(loaded.features) == classifier_features(frame, target)
            counts = data_counts(frame, target)
            same_policy = all(counts.get(cls, 0) >= MIN_CLASS_SIZE for cls in loaded.classes)
            fits_columns = (
                isinstance(loaded, TalentModel)
                and current_fields.issubset(vars(loaded))
                and same_policy
                and same_features
                and set(loaded.features).issubset(frame.columns)
            )
            # Rows are dropped for tiny classes and unknown targets, so an exact match is
            # too strict; a pool that moved by more than a few percent is a different pool.
            fitted_on = loaded.n_train + loaded.n_test if fits_columns else 0
            if fits_columns and abs(fitted_on - eligible) <= max(5, 0.02 * eligible):
                _CACHE[key] = loaded
        except Exception:
            pass

    if key not in _CACHE:
        _CACHE[key] = train_position_classifier(frame, target=target)
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(_CACHE[key], disk_path, compress=3)
        except Exception:
            pass
    return _CACHE[key]


def get_cached_model(frame: pd.DataFrame) -> TalentModel:
    """The four broad position groups — the model the dashboard shows by default."""
    return _cached("model", "position_group.joblib", "position_group", frame)


def get_cached_role_model(frame: pd.DataFrame) -> TalentModel:
    """The finer-grained sibling: ten side-agnostic roles instead of four groups."""
    return _cached("role", "position_role.joblib", "position_role", frame)


def get_cached_exact_model(frame: pd.DataFrame) -> TalentModel:
    """The hardest cut: the exact StatsBomb position, left and right included.

    Around 49% against a 5% coin-flip baseline, and the ceiling is the data rather than
    the model — roughly half the remaining errors are pure left/right swaps, because
    preferred foot is the only feature that carries a side at all. Reported alongside
    top-3, which is the number a scout can act on.
    """
    return _cached("exact", "primary_position.joblib", "primary_position", frame)


def get_cached_importance(
    tm: TalentModel, frame: pd.DataFrame, top_n: int = 5
) -> list[dict[str, Any]]:
    key = str(top_n)
    if key not in _SHAP_CACHE:
        disk_path = MODELS_DIR / "shap_importance.json"
        if disk_path.is_file():
            try:
                with open(disk_path, encoding="utf-8") as f:
                    stored = json.load(f)
                if isinstance(stored, list) and len(stored) >= top_n:
                    _SHAP_CACHE[key] = stored[:top_n]
            except Exception:
                pass
        if key not in _SHAP_CACHE:
            _SHAP_CACHE[key] = shap_importance(tm, frame, top_n=top_n)
            try:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                with open(disk_path, "w", encoding="utf-8") as f:
                    json.dump(_SHAP_CACHE[key], f, indent=2)
            except Exception:
                pass
    return _SHAP_CACHE[key]
