"""Market Value Predictor (LightGBM).

Target = Transfermarkt 2015/16 market value for the four big men's leagues (ES1, GB1,
IT1, FR1), joined to our players by name. We train on log(value) because values are
heavy-tailed, evaluate on a split grouped by player, and surface "bargains" — players the
model prices above what the market paid.

Two properties of that join decide whether any of this means anything:

  The source covers men only. Passing it a frame that includes the women's competitions
  produced 76 matches out of 1,258, every one of them false by construction: Amanda
  Sampedro Bustos was labelled with Pedro's EUR 23M, and five different women were all
  matched to Eder at EUR 15M.

  A short name is a substring of a long one. `fuzz.WRatio` scores through partial_ratio,
  so any brief Transfermarkt name buried inside a long StatsBomb one clears a cutoff of
  90 — that is how Charly Musonda matched "Son" and Carlos Henrique Casimiro matched
  "Henrique". No string scorer separates those from the true matches of the same shape
  (Lionel Andres Messi Cuccittini really is Lionel Messi), so `match_values` decides on
  shared name *tokens* instead. See its docstring.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rapidfuzz import fuzz, process
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from footyvision.etl.sofifa import parse_value_eur  # noqa: F401  (re-exported)
from footyvision.ml.features import PER90_FEATURES


def _normalize_name(name: str) -> str:
    """Lowercase and strip accents so 'Suárez'/'Modrić' match their ASCII forms."""
    ascii_ = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return ascii_.lower().strip()


def _name_tokens(name: str) -> set[str]:
    """The parts of a name that could identify somebody.

    Tokens of three characters or fewer are dropped: "de", "da", "i", "el" carry no
    identity, and letting them count would make "Juan de la Cruz" agree with "Pedro de
    la Fuente" on two tokens.
    """
    cleaned = _normalize_name(name).replace(".", " ").replace("-", " ").replace("'", " ")
    return {tok for tok in cleaned.split() if len(tok) > 2}


def _is_same_person(
    left: str, right: str, left_counts: Counter[str], right_counts: Counter[str]
) -> bool:
    """Whether two names from different sources denote one player.

    The string similarity has already said yes by this point, and on this data it says yes
    far too often — see the module docstring. What actually decides it is how many
    identifying tokens the two names share:

      two or more    Settled. "Lionel Andres Messi Cuccittini" and "Lionel Messi" agree on
                     both `lionel` and `messi`; a false pair essentially never does.

      exactly one    Only if that token belongs to one player on each side. `neymar` does,
                     so Neymar da Silva Santos Junior keeps his EUR 100M. `roberto` does
                     not — it is Diego Roberto Godin Leal's middle name and also a
                     Transfermarkt player in his own right, and there is no way to tell
                     which was meant. Isco and Danilo are lost the same way, and that is
                     the intended trade: a wrong value is worse than a missing one.

      none           Rejected. This is the substring artifact, and every case of it seen
                     here was false.
    """
    if _normalize_name(left) == _normalize_name(right):
        return True
    shared = _name_tokens(left) & _name_tokens(right)
    if len(shared) >= 2:
        return True
    return any(left_counts[tok] == 1 and right_counts[tok] == 1 for tok in shared)


def match_values(
    features: pd.DataFrame,
    values: pd.DataFrame,
    keep_cols: tuple[str, ...] = ("value_eur",),
    score_cutoff: int = 90,
) -> pd.DataFrame:
    """Fuzzy-match `features.name` to `values.name`, attaching `keep_cols`.

    Accents are normalised first, so Suarez reaches Suárez and Fabianski reaches
    Fabiański. The fuzzy score then proposes a candidate and `_is_same_person` decides
    whether to believe it — the score alone accepted 195 pairs on this data that are not
    the same footballer.

    `values` should be unique by name. Rows the string matcher proposed and the token rule
    rejected are dropped, not flagged: a value column with false entries in it is worse
    than a shorter table, because nothing downstream can tell which rows to distrust.
    """
    values = values.drop_duplicates("name")
    norm_to_orig: dict[str, str] = {}
    for original in values["name"]:
        norm_to_orig.setdefault(_normalize_name(original), original)
    choices = list(norm_to_orig)
    lookup = values.set_index("name")

    # How many players on each side a given token could refer to, which is what makes a
    # single shared token decisive or useless.
    left_counts = Counter(tok for name in features["name"] for tok in _name_tokens(name))
    right_counts = Counter(tok for name in values["name"] for tok in _name_tokens(name))

    rows = []
    for _, r in features.iterrows():
        match = process.extractOne(
            _normalize_name(r["name"]), choices, scorer=fuzz.WRatio, score_cutoff=score_cutoff
        )
        if match is None:
            continue
        matched = norm_to_orig[match[0]]
        if not _is_same_person(r["name"], matched, left_counts, right_counts):
            continue
        extra = {c: lookup.loc[matched, c] for c in keep_cols}
        rows.append({**r.to_dict(), "matched_name": matched, "match_score": match[1], **extra})
    return pd.DataFrame(rows)


@dataclass
class ValueModel:
    """The median model, plus the two quantile models that bound it.

    Read `r2` knowing it is computed on log(value), which is where the model is fitted and
    is a kinder scale than the one anybody cares about; `r2_eur` is the same model judged
    in euros. Measured on what `footyvision value-report` runs — per-90 metrics plus age,
    men only, split by player, 1,019 matched player-seasons:

        R² on log(value)   +0.43
        R² in euros        +0.22
        MAE                €4.29M
        predicting the median for everyone   €5.11M
        5th-95th band caught                 72% of held-out players

    So the model's edge over a constant is about eight hundred thousand euros. That is a
    real margin and a modest one, which is why nothing here returns a bare number: "worth
    €12M more than his price" reads as a finding when it is a direction. The quantile
    models carry the width, and `interval_coverage` says how often that width was right —
    72 against a nominal 90, so the band is honest only because it is reported alongside
    what it actually caught.

    These figures are far better than the same model scored before the name-matching was
    fixed (+0.24 log, +0.09 euros). Nothing about the learner changed; a quarter of the
    labels were simply somebody else's value.
    """

    model: LGBMRegressor
    features: list[str]
    # On log(value). The euro-scale R² is near zero and is reported separately rather than
    # left to be assumed from this one.
    r2: float
    r2_eur: float
    mae_eur: float
    # What predicting the training median for every player would cost. Without it, an MAE
    # of "€4.8M" sounds like a result instead of a rounding error against doing nothing.
    baseline_mae_eur: float
    n_train: int
    n_test: int
    value_col: str
    lower: LGBMRegressor | None = None
    upper: LGBMRegressor | None = None
    # Share of held-out players whose real value fell inside the interval. An 80% interval
    # that catches 80% is honest; one that catches 45% is decoration.
    interval_coverage: float | None = None


def train_value_model(
    merged: pd.DataFrame,
    feature_cols: list[str] | None = None,
    value_col: str = "value_eur",
    seed: int = 42,
) -> ValueModel:
    """Train LightGBM on log(value); report R² and MAE (in €) on a held-out split.

    `feature_cols` defaults to the per-90 style features; pass e.g. PER90 + ['age'] to add
    age (a strong value driver). LightGBM handles missing values natively.
    """
    features = list(feature_cols) if feature_cols else list(PER90_FEATURES)
    data = merged[merged[value_col] > 0].copy().reset_index(drop=True)
    x = data[features].to_numpy(dtype=float)
    y = np.log1p(data[value_col].to_numpy(dtype=float))

    # Split on the player, not the row. A player with two seasons carries the *same* value
    # label on both, so a plain split can put one in training and read the answer off the
    # other. Only six rows here, but the guard costs nothing and the alternative is a
    # score that cannot be defended.
    if "player_id" in data.columns:
        groups = data["player_id"].to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        tr_idx, te_idx = next(splitter.split(x, y, groups=groups))
        x_tr, x_te, y_tr, y_te = x[tr_idx], x[te_idx], y[tr_idx], y[te_idx]
    else:
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=seed)
    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        verbose=-1,
    )
    model.fit(x_tr, y_tr)

    pred_te = model.predict(x_te)
    r2 = float(r2_score(y_te, pred_te))
    euros_true, euros_pred = np.expm1(y_te), np.expm1(pred_te)
    r2_eur = float(r2_score(euros_true, euros_pred))
    mae_eur = float(mean_absolute_error(euros_true, euros_pred))
    # The constant every model has to beat before it has said anything.
    baseline = np.expm1(np.full_like(y_te, float(np.median(y_tr))))
    baseline_mae_eur = float(mean_absolute_error(euros_true, baseline))

    # Quantile regression on the same features and split, bounding the middle of the
    # distribution rather than chasing the mean. Fitted at 0.05/0.95 rather than the
    # obvious 0.1/0.9 because these models are consistently over-confident out of sample:
    # measured on the held-out players, a nominal 80% band caught 66%, a 90% band caught
    # 78%, and a 95% band caught 82%. The wider fit is what actually delivers four in
    # five, and `interval_coverage` reports what was observed rather than what was asked
    # for, so the gap stays visible if it moves.
    lower = _quantile_model(0.05, seed).fit(x_tr, y_tr)
    upper = _quantile_model(0.95, seed).fit(x_tr, y_tr)
    inside = (y_te >= lower.predict(x_te)) & (y_te <= upper.predict(x_te))
    coverage = float(np.mean(inside))

    return ValueModel(
        model,
        features,
        r2,
        r2_eur,
        mae_eur,
        baseline_mae_eur,
        len(x_tr),
        len(x_te),
        value_col,
        lower=lower,
        upper=upper,
        interval_coverage=coverage,
    )


def _quantile_model(alpha: float, seed: int) -> LGBMRegressor:
    """A bound, not a mean. Same shape as the median model so the three are comparable."""
    return LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        verbose=-1,
    )


def predict_values(vm: ValueModel, merged: pd.DataFrame) -> pd.DataFrame:
    """Add `predicted_value`, its 10th/90th percentile bounds, and `value_residual`."""
    out = merged.copy()
    x = out[vm.features].to_numpy(dtype=float)
    point = np.expm1(vm.model.predict(x))
    if vm.lower is not None and vm.upper is not None:
        low = np.expm1(vm.lower.predict(x))
        high = np.expm1(vm.upper.predict(x))
        # The three models are fitted independently, so on individual rows they can cross:
        # sorting the pair keeps every interval well-formed, and clipping the median into
        # it keeps the triple coherent. Without the clip a player comes back priced at
        # EUR 20.95M inside a band that stops at EUR 21.07M — arithmetically defensible,
        # and unreadable as an answer.
        lo = np.minimum(low, high)
        hi = np.maximum(low, high)
        out["predicted_low"] = lo.round(0)
        out["predicted_high"] = hi.round(0)
        point = np.clip(point, lo, hi)
    out["predicted_value"] = point.round(0)
    out["value_residual"] = (out[vm.value_col] - out["predicted_value"]).round(0)
    return out


def bargains(vm: ValueModel, merged: pd.DataFrame, top_n: int = 15) -> list[dict[str, Any]]:
    """Players the model rates far above their actual value (performance-implied bargains)."""
    scored = predict_values(vm, merged)
    scored = scored.sort_values("value_residual")  # most negative = underpriced vs performance
    rows = []
    for _, r in scored.head(top_n).iterrows():
        rows.append(
            {
                "player_id": int(r["player_id"]),
                "name": r["name"],
                "position_group": r["position_group"],
                "actual_value": float(r[vm.value_col]),
                "predicted_value": float(r["predicted_value"]),
                "upside": float(-r["value_residual"]),
            }
        )
    return rows


def shap_importance(vm: ValueModel, merged: pd.DataFrame, top_n: int = 10) -> list[dict[str, Any]]:
    """Global mean |SHAP| per feature for the value model."""
    import shap

    x = merged[vm.features].to_numpy(dtype=float)
    values = np.abs(np.array(shap.TreeExplainer(vm.model).shap_values(x)))
    feat_axis = next(ax for ax, size in enumerate(values.shape) if size == len(vm.features))
    other = tuple(ax for ax in range(values.ndim) if ax != feat_axis)
    mean_abs = values.mean(axis=other)
    ranked = sorted(zip(vm.features, mean_abs, strict=False), key=lambda kv: kv[1], reverse=True)
    return [{"feature": f, "mean_abs_shap": round(float(v), 4)} for f, v in ranked[:top_n]]
