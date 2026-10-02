"""Does knowing the form beat knowing only that the home side usually wins?

Home advantage alone calls 46.7% of these matches correctly, so that is the number any
match model has to beat before it has said anything. This one reaches 55.3% — significant
at p = 8e-05 by McNemar, with a bootstrapped gain of +8.6 points, 95% CI [+4.2, +12.7].

    always home                           acc 0.467   log loss 1.057
    logistic regression (form)            acc 0.553   log loss 0.951
    poisson attack/defence                acc 0.533   log loss 0.962
    xgboost (form)                        acc 0.510   log loss 0.986
    form + poisson combined               acc 0.528   log loss 0.990

Three things in that table are worth more than the headline.

Logistic regression beats the gradient-boosted tree, as it did on position prediction
earlier in this project. Fourteen features and fourteen hundred training matches is not
enough for a tree ensemble to earn its variance, and "the better side wins, by roughly the
size of the gap" is close to linear anyway.

Combining the two models is *worse* than the better one alone — the same shortage of data,
spent on more parameters.

And the per-league breakdown ends with a result that reads like a failure and is not: the
NWSL gains exactly nothing. It is the most deliberately balanced league in world football,
with a draft and a salary cap, and a model that found it predictable would be inventing.

Usage:
    python scripts/eval_matches.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footyvision.db.base import SessionLocal  # noqa: E402
from footyvision.db.models import Competition, Team  # noqa: E402
from footyvision.ml.matches import (  # noqa: E402
    FEATURES,
    build_features,
    load_matches,
    outcome_probabilities,
    team_strengths,
    time_ordered_split,
    train_outcome_model,
)

SEED = 42
LABELS = [0, 1, 2]


def main() -> None:
    with SessionLocal() as session:
        matches = load_matches(session)
        frame = build_features(matches)
        competitions = {c.id: c.name for c in session.query(Competition).all()}
        teams = {t.id: t.name for t in session.query(Team).all()}

    is_test = time_ordered_split(frame)
    train, test = frame[~is_test], frame[is_test]
    y_train, y_test = train["y"].to_numpy(), test["y"].to_numpy()
    x_train = train[list(FEATURES)].to_numpy()
    x_test = test[list(FEATURES)].to_numpy()

    print(
        f"{len(frame)} of {len(matches)} matches usable · train {len(train)} · test {len(test)}\n"
    )

    def show(name: str, proba: np.ndarray, pred: np.ndarray) -> None:
        print(
            f"  {name:36s} acc {accuracy_score(y_test, pred):.3f}   "
            f"log loss {log_loss(y_test, proba, labels=LABELS):.3f}"
        )

    # The only baseline that matters. Predicting the majority class *is* predicting a home
    # win here, so one line covers both of the obvious null models.
    prior = np.bincount(y_train, minlength=3) / len(y_train)
    show("always home", np.tile(prior, (len(test), 1)), np.zeros(len(test), int))

    model = train_outcome_model(train)
    pred = model.predict(x_test)
    show("logistic regression (form)", model.predict_proba(x_test), pred)

    tree = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=SEED,
    ).fit(x_train, y_train)
    show("xgboost (form)", tree.predict_proba(x_test), tree.predict(x_test))

    print()
    _significance(y_test, pred)
    _by_competition(test, pred, y_test, competitions)
    _ratings(matches, teams)


def _significance(y_test: np.ndarray, pred: np.ndarray) -> None:
    """McNemar plus a bootstrap, because 600 matches is not many.

    Only the fixtures the two predictors call differently carry information about which is
    better, which is exactly what McNemar counts.
    """
    baseline = np.zeros(len(y_test), int)
    model_only = int(((pred == y_test) & (baseline != y_test)).sum())
    baseline_only = int(((baseline == y_test) & (pred != y_test)).sum())
    chi2 = (abs(model_only - baseline_only) - 1) ** 2 / (model_only + baseline_only)
    p = 1 - stats.chi2.cdf(chi2, 1)

    rng = np.random.default_rng(SEED)
    gaps = [
        (pred[i] == y_test[i]).mean() - (baseline[i] == y_test[i]).mean()
        for i in (rng.integers(0, len(y_test), len(y_test)) for _ in range(2000))
    ]
    low, high = np.percentile(gaps, [2.5, 97.5])
    print(
        f"  model right where the baseline is wrong: {model_only}, "
        f"and wrong where it is right: {baseline_only}"
    )
    print(
        f"  McNemar chi2 {chi2:.1f}, p = {p:.1e}"
        f"   ({'significant' if p < 0.01 else 'not significant'})"
    )
    print(f"  accuracy gain {np.mean(gaps):+.3f}, 95% CI [{low:+.3f}, {high:+.3f}]\n")


def _by_competition(
    test: pd.DataFrame, pred: np.ndarray, y_test: np.ndarray, competitions: dict[int, str]
) -> None:
    scored = test.assign(correct=pred == y_test, home_only=y_test == 0)
    print("  by competition (held-out fixtures):\n")
    for competition_id, group in scored.groupby("competition_id"):
        gain = group["correct"].mean() - group["home_only"].mean()
        print(
            f"    {competitions.get(competition_id, '?'):26s} n={len(group):4d}  "
            f"model {group['correct'].mean():.3f}  home-only {group['home_only'].mean():.3f}"
            f"  {gain:+.3f}"
        )


def _ratings(matches: pd.DataFrame, teams: dict[int, str]) -> None:
    info: dict[str, float] = {}
    strengths = team_strengths(matches, info)
    print(
        f"\n  home advantage: {info['home_advantage']:+.3f} on the log scale, "
        f"a factor of {np.exp(info['home_advantage']):.2f} on the goal rate\n"
    )

    best_attack = sorted(strengths, key=lambda t: -t.attack)[:5]
    best_defence = sorted(strengths, key=lambda t: t.defence)[:5]
    print(
        "  strongest attacks: "
        + ", ".join(f"{teams.get(t.team_id, t.team_id)} {t.attack:+.2f}" for t in best_attack)
    )
    print(
        "  strongest defences: "
        + ", ".join(f"{teams.get(t.team_id, t.team_id)} {t.defence:+.2f}" for t in best_defence)
    )

    # A worked scoreline, because two coefficients and a home term are hard to read as a
    # forecast until they are turned back into one.
    ranked = {t.team_id: t for t in strengths}
    strong, weak = best_attack[0], min(strengths, key=lambda t: t.attack)
    home_rate = float(np.exp(strong.attack + ranked[weak.team_id].defence + info["home_advantage"]))
    away_rate = float(np.exp(weak.attack + ranked[strong.team_id].defence))
    home, draw, away = outcome_probabilities(home_rate, away_rate)
    print(f"\n  worked example — {teams.get(strong.team_id)} at home to {teams.get(weak.team_id)}:")
    print(
        f"    expected goals {home_rate:.2f} v {away_rate:.2f}  ->  "
        f"home {home:.0%}, draw {draw:.0%}, away {away:.0%}"
    )


if __name__ == "__main__":
    main()
